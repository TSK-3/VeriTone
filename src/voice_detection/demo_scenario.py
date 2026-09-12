"""Scripted live-call simulator: real cloned-voice WAVs + a scam transcript.

Drives the exact same pipeline as a real Twilio call (segment scoring, running
risk, trigger-word SMS rules) so the demo works with zero external services. The
first leg uses a genuine clip, the attacker leg uses English cloned voices from
the held-out MLAAD sample — audio is decoded in memory and never persisted.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from .alerts import AlertEngine, detect_trigger_words
from .audio import AudioSegment, decode_wav
from .live import LiveCallRegistry
from .pipeline import analyze_into_session
from .service import DetectionService
from .session_store import SessionStore

ROOT = Path(__file__).parents[2]
STEP_SECONDS = 2.0
STEP_PAUSE = 0.45

SCAM_SCRIPT: list[dict] = [
    {"voice": "genuine", "text": "Hello, this is Arjun from the accounts helpdesk. Am I speaking with the account holder?"},
    {"voice": "spoof", "text": "Sir, I am calling from your bank's fraud department. We see unusual activity on your account."},
    {"voice": "spoof", "text": "A transaction of four lakh rupees is pending on your card. I can stop it if you verify your details now."},
    {"voice": "spoof", "text": "Please send the OTP I just sent you so we can block the payment immediately. It is urgent."},
    {"voice": "spoof", "text": "The money will be deducted in minutes. Approve the transfer on your screen right now."},
    {"voice": "spoof", "text": "Thank you for confirming the wallet payment. Your account is safe now."},
]

_FILE_CACHE: dict[str, AudioSegment] = {}


def _pick_file(voice: str, index: int) -> Path:
    if voice == "genuine":
        folder = ROOT / "data" / "genuine"
        files = sorted(folder.glob("*.wav"))
    else:
        folder = ROOT / "data_unseen" / "spoof"
        files = sorted(folder.glob("*fake_en*.wav")) or sorted(folder.glob("*.wav"))
    if not files:
        raise RuntimeError(f"no demo WAV files found in {folder}")
    return files[index % len(files)]


def _load_audio(voice: str, index: int) -> AudioSegment:
    path = _pick_file(voice, index)
    key = str(path)
    if key not in _FILE_CACHE:
        _FILE_CACHE[key] = decode_wav(path.read_bytes())
    base = _FILE_CACHE[key]
    bytes_needed = int(base.sample_rate * STEP_SECONDS) * 2
    start = max(0, (len(base.samples) - bytes_needed) // 2)
    chunk = base.samples[start:start + bytes_needed]
    if len(chunk) < bytes_needed:
        chunk = chunk.ljust(bytes_needed, b"\0")
    return AudioSegment(samples=chunk, sample_rate=base.sample_rate, duration_s=STEP_SECONDS)


def start_scenario(session_id: str, service: DetectionService, sessions: SessionStore,
                   registry: LiveCallRegistry, alerts: AlertEngine, steps: int | None = None) -> dict:
    """Kick off the simulated scam call in a background thread."""
    count = len(SCAM_SCRIPT) if steps is None else max(1, min(steps, len(SCAM_SCRIPT)))
    try:
        sessions.get(session_id)
    except KeyError:
        sessions.create(session_id, channel_type="simulated_call", scenario="support_call")
    registry.register(session_id, channel="simulated_call", label="Simulated cloned-voice scam call")
    thread = threading.Thread(target=_run, args=(session_id, service, sessions, registry, alerts, count), daemon=True)
    thread.start()
    return {"session_id": session_id, "steps": count}


def _run(session_id: str, service: DetectionService, sessions: SessionStore,
         registry: LiveCallRegistry, alerts: AlertEngine, steps: int) -> None:
    try:
        session = sessions.get(session_id)
        for index, step in enumerate(SCAM_SCRIPT[:steps]):
            audio = _load_audio(step["voice"], index)
            try:
                record = analyze_into_session(session, audio, index * STEP_SECONDS,
                                              service, registry=registry, alerts=alerts)
            except RuntimeError:
                record = None  # strict Tier 2 unconfigured: keep the demo transcript flowing
            risk = int((record or {}).get("risk_score", 0))
            words = detect_trigger_words(step["text"])
            registry.add_transcript(session_id, step["text"], source="simulated",
                                    triggers=words, risk=risk)
            if words:
                registry.add_triggers(session_id, words)
                alert = alerts.check_trigger_words(session_id, words, risk)
                if alert is not None:
                    registry.add_alert(session_id, alert.as_dict())
            time.sleep(STEP_PAUSE)
        registry.complete(session_id)
        registry.add_transcript(session_id, "— call ended —", source="system", triggers=[])
    except Exception as exc:  # never let the demo thread die silently
        registry.add_transcript(session_id, f"simulator error: {exc}", source="system", triggers=[])
        registry.complete(session_id)
