"""Twilio Media Streams adapter: µ-law 8 kHz inbound audio → 16 kHz PCM → live scoring.

Accepts standard Twilio Media Streams events (``start``, ``media``,
``transcription``, ``stop``). Media payloads are base64 µ-law 8 kHz mono; they are
decoded in memory, resampled to 16 kHz PCM16, VAD-gated into 1.2–3.2 s speech
segments and scored through the same pipeline as the REST API. Twilio real-time
transcription events (``<Start><Transcription>``) drive the trigger-word SMS rules.
"""
from __future__ import annotations

import array
import base64
import math
import sys
from uuid import uuid4

from .alerts import AlertEngine, detect_trigger_words
from .audio import AudioSegment
from .live import LiveCallRegistry
from .pipeline import analyze_into_session
from .service import DetectionService
from .session_store import SessionStore

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320  # 20 ms @ 16 kHz


def _build_mulaw_table() -> list[int]:
    """G.711 µ-law decode table scaled to 16-bit PCM (wire MSB: 1 = positive)."""
    table = []
    for byte in range(256):
        sign = 1 if byte & 0x80 else -1
        exponent = (byte >> 4) & 0x07
        mantissa = byte & 0x0F
        magnitude = ((2 * mantissa + 33) << exponent) - 33
        table.append(sign * magnitude * 4)
    return table


ULAW_TO_PCM16 = _build_mulaw_table()


def mulaw_to_pcm16(payload: bytes) -> list[int]:
    return [ULAW_TO_PCM16[byte] for byte in payload]


def pcm16_to_mulaw(sample: int) -> int:
    """Inverse µ-law encoder (tests/simulators): 16-bit PCM sample → wire byte."""
    sample = max(-32768, min(32767, int(sample)))
    sign_bit = 0x80 if sample >= 0 else 0x00
    magnitude = min(abs(sample) >> 2, 8031) + 33  # 14-bit domain + bias
    exponent = 0
    while magnitude > 63:
        magnitude >>= 1
        exponent += 1
    mantissa = magnitude & 0x0F
    return sign_bit | (~((exponent << 4) | mantissa) & 0x7F)


def _pack_pcm16(values: list[int]) -> bytes:
    block = array.array("h", values)
    if sys.byteorder == "big":
        block.byteswap()
    return block.tobytes()


def resample_to_16k(samples: list[int], src_rate: int) -> list[int]:
    """Linear-interpolation resample to 16 kHz (8 kHz telephony in, no deps)."""
    if src_rate == SAMPLE_RATE:
        return samples
    if not samples:
        return []
    ratio = SAMPLE_RATE / src_rate
    output = []
    for i in range(int(len(samples) * ratio)):
        position = i / ratio
        left = int(position)
        fraction = position - left
        a = samples[left]
        b = samples[left + 1] if left + 1 < len(samples) else a
        output.append(int(a + (b - a) * fraction))
    return output


class SpeechSegmenter:
    """VAD-gate a 16 kHz stream into 1.2–3.2 s speech segments (PCM16 bytes)."""

    def __init__(self, min_seconds: float = 1.2, max_seconds: float = 3.2,
                 silence_seconds: float = 0.8, threshold: int = 300) -> None:
        self.min_frames = int(min_seconds * 50)
        self.max_frames = int(max_seconds * 50)
        self.silence_frames = int(silence_seconds * 50)
        self.threshold = threshold
        self._buf: list[int] = []
        self._total = 0
        self._voiced = 0
        self._trailing_silence = 0
        self.segments_emitted = 0

    def feed(self, samples: list[int]) -> list[bytes]:
        self._buf.extend(samples)
        emitted: list[bytes] = []
        index = 0
        limit = len(self._buf)
        while index + FRAME_SAMPLES <= limit:
            frame = self._buf[index:index + FRAME_SAMPLES]
            rms = math.sqrt(sum(value * value for value in frame) / FRAME_SAMPLES)
            if rms >= self.threshold:
                self._voiced += 1
                self._trailing_silence = 0
            else:
                self._trailing_silence += 1
            index += FRAME_SAMPLES
            self._total += 1
            voiced_ratio = self._voiced / self._total
            if (
                (self._total >= self.max_frames and self._voiced > 0)
                or (self._total >= self.min_frames and voiced_ratio >= 0.35)
                or (self._voiced > 0 and self._trailing_silence >= self.silence_frames)
            ):
                emitted.append(_pack_pcm16(self._buf[:self._total * FRAME_SAMPLES]))
                self._buf = self._buf[self._total * FRAME_SAMPLES:]
                limit = len(self._buf)
                index = 0
                self._total = self._voiced = self._trailing_silence = 0
                self.segments_emitted += 1
            elif self._voiced == 0 and self._total >= self.max_frames:
                self._buf = self._buf[index:]  # drop pure-silence buffer, keep tail frame
                limit = len(self._buf)
                index = 0
                self._total = self._voiced = self._trailing_silence = 0
        return emitted


class TwilioCallHandler:
    """Stateful per-connection handler for one Twilio Media Streams socket."""

    def __init__(self, service: DetectionService, sessions: SessionStore,
                 registry: LiveCallRegistry, alerts: AlertEngine) -> None:
        self.service = service
        self.sessions = sessions
        self.registry = registry
        self.alerts = alerts
        self.session_id: str | None = None
        self.stream_sid: str | None = None
        self.segmenter = SpeechSegmenter()
        self.segments_scored = 0
        self._start_s = 0.0

    def handle(self, message: dict) -> dict | None:
        event = message.get("event")
        if event == "start":
            return self._start(message)
        if event == "media":
            return self._media(message)
        if event == "transcription":
            return self._transcription(message)
        if event == "stop":
            return self._stop()
        return None

    def _start(self, message: dict) -> dict:
        start = message.get("start", {})
        params = start.get("customParameters", {}) or {}
        self.stream_sid = start.get("streamSid")
        self.session_id = params.get("session_id") or self.stream_sid or f"twilio-{uuid4()}"
        try:
            self.sessions.get(self.session_id)
        except KeyError:
            self.sessions.create(
                self.session_id, channel_type="twilio",
                scenario=params.get("scenario", "support_call"),
                language_hint=params.get("language_hint"),
                feature_only_logging=str(params.get("feature_only_logging", "")).lower() == "true",
            )
        self.registry.register(self.session_id, channel="twilio",
                               label=params.get("label") or "Twilio live call")
        return {"handled": "start", "session_id": self.session_id}

    def _ensure_session(self):
        try:
            return self.sessions.get(self.session_id)
        except KeyError:
            self.sessions.create(self.session_id, channel_type="twilio")
            self.registry.register(self.session_id, "twilio", "Twilio live call")
            return self.sessions.get(self.session_id)

    def _media(self, message: dict) -> dict | None:
        media = message.get("media", {})
        payload = media.get("payload")
        if not payload:
            return None
        decoded = mulaw_to_pcm16(base64.b64decode(payload))
        speech = self.segmenter.feed(resample_to_16k(decoded, 8000))
        if not speech:
            return None
        if self.session_id is None:
            self.session_id = f"twilio-{uuid4()}"
        scored = 0
        for pcm_bytes in speech:
            audio = AudioSegment(samples=pcm_bytes, sample_rate=SAMPLE_RATE,
                                 duration_s=len(pcm_bytes) / (SAMPLE_RATE * 2))
            try:
                analyze_into_session(self._ensure_session(), audio, self._start_s,
                                     self.service, registry=self.registry, alerts=self.alerts)
            except RuntimeError as exc:  # strict-mode Tier 2 unconfigured
                return {"handled": "media_error", "error": str(exc)}
            self._start_s += audio.duration_s
            self.segments_scored += 1
            scored += 1
        return {"handled": "media", "segments": scored} if scored else None

    def _current_risk(self) -> int:
        call = self.registry.get(self.session_id or "")
        if call and call.latest:
            return int(call.latest.get("risk_score", 0))
        return 0

    def _transcription(self, message: dict) -> dict | None:
        if self.session_id is None:
            return None
        transcription = message.get("transcription", {})
        found: list[str] = []
        for transcript in transcription.get("transcripts", []):
            text = (transcript.get("transcript") or "").strip()
            if not text:
                continue
            words = detect_trigger_words(text)
            risk = self._current_risk()
            self.registry.add_transcript(self.session_id, text, source="twilio",
                                         triggers=words, risk=risk)
            if words:
                found.extend(words)
                self.registry.add_triggers(self.session_id, words)
                action = (self.registry.get(self.session_id).latest or {}).get("action", "warn")
                alert = self.alerts.check_trigger_words(self.session_id, words, risk, action=action)
                if alert is not None:
                    self.registry.add_alert(self.session_id, alert.as_dict())
        return {"handled": "transcription", "triggers": found}

    def _stop(self) -> dict:
        if self.session_id:
            self.registry.complete(self.session_id)
        return {"handled": "stop", "session_id": self.session_id, "segments": self.segments_scored}

    def close(self) -> None:
        if self.session_id:
            self.registry.complete(self.session_id)

