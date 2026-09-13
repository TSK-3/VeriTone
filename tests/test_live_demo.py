"""Tests for the live-call demo pipeline: codec, VAD, triggers, SMS, Twilio handler."""
import base64
import math
import struct

import pytest

from voice_detection.alerts import AlertEngine, TwilioClient, build_sms_body, detect_trigger_words
from voice_detection.audio import AudioSegment, resample_linear
from voice_detection.demo_ensemble import DemoTier2Ensemble
from voice_detection.live import LiveCallRegistry
from voice_detection.service import DetectionService
from voice_detection.session_store import SessionStore
from voice_detection.twilio_stream import (
    ULAW_TO_PCM16,
    SpeechSegmenter,
    TwilioCallHandler,
    mulaw_to_pcm16,
    pcm16_to_mulaw,
)


def tone(seconds: float, rate: int = 16_000, amplitude: int = 6_000) -> AudioSegment:
    n = int(seconds * rate)
    values = [int(amplitude * math.sin(2 * math.pi * 220 * i / rate)) for i in range(n)]
    return AudioSegment(samples=struct.pack(f"<{n}h", *values), sample_rate=rate, duration_s=seconds)


# --- codec -------------------------------------------------------------------

def test_mulaw_roundtrip_preserves_sign_and_scale() -> None:
    assert ULAW_TO_PCM16[0x00] == -32124  # max negative
    assert ULAW_TO_PCM16[0x80] == 32124   # max positive
    assert ULAW_TO_PCM16[0xFF] == 0       # positive zero
    for sample in (0, 1234, -1234, 6000, -6000, 15000, -15000, 30000, -32000):
        decoded = mulaw_to_pcm16(bytes([pcm16_to_mulaw(sample)]))[0]
        assert (decoded > 0) == (sample > 0) or sample == 0
        assert abs(decoded) <= 32768
        assert abs(decoded - sample) <= max(450, abs(sample) * 0.08), (sample, decoded)


def test_resample_8k_to_16k_doubles_length() -> None:
    out = resample_linear([100, 200, 300, 400], 8_000, 16_000)
    assert len(out) == 8 and out[0] == 100 and out[-1] >= 300


# --- VAD ---------------------------------------------------------------------

def test_segmenter_emits_segment_for_continuous_speech() -> None:
    seg = SpeechSegmenter(min_seconds=1.0, max_seconds=2.0, threshold=300)
    speech = struct.unpack(f"<{int(1.5 * 16_000)}h", tone(1.5).samples)
    segments = []
    for i in range(0, len(speech), 320):
        segments.extend(seg.feed(speech[i:i + 320]))
    segments.extend(seg.flush())
    assert len(segments) == 1
    assert len(segments[0]) >= 16_000  # at least 0.5 s of PCM16


def test_segmenter_holds_silence() -> None:
    seg = SpeechSegmenter(min_seconds=1.0, threshold=300)
    silence = [0] * (16_000 * 2)
    assert all(seg.feed(silence[i:i + 1600]) == [] for i in range(0, len(silence), 1600))


# --- trigger words / SMS -----------------------------------------------------

def test_trigger_words_detected() -> None:
    words = detect_trigger_words("Please SEND the money via transaction now.")
    assert {"send", "money", "transaction"} <= set(words)


def test_trigger_words_ignored_without_match() -> None:
    assert detect_trigger_words("the weather is lovely today") == []


def test_sms_body_mentions_prevention() -> None:
    body = build_sms_body(78, ["send", "money"], "escalate")
    assert "78/100" in body and "Do NOT approve payments" in body and "send" in body


def test_alert_engine_console_mode_and_cooldown() -> None:
    # Explicit empty creds: tests must never send real SMS even when the
    # environment has live Twilio variables set for the demo server.
    sender = TwilioClient(account_sid="", auth_token="", from_number="", default_to="")
    engine = AlertEngine(sender=sender, cooldown_s=999, trigger_risk_floor=40, sink=[])
    record = engine.check_trigger_words("call-1", ["money"], risk=80)
    assert record is not None and record.sms_status == "logged_console"
    assert engine.check_trigger_words("call-1", ["otp"], risk=80) is None  # cooldown
    assert engine.check_trigger_words("call-1", ["money"], risk=10) is None  # below floor


def test_threshold_alert_fires_on_prevention_action() -> None:
    engine = AlertEngine(cooldown_s=999, sink=[])
    assert engine.check_after_record("c", {"action": "warn", "risk_score": 55}) is None
    assert engine.check_after_record("c", {"action": "escalate", "risk_score": 85}) is not None


# --- demo ensemble -----------------------------------------------------------

def test_demo_ensemble_labels_members_and_scores_tone() -> None:
    out = DemoTier2Ensemble().score(tone(1.5))
    assert set(out.model_status.values()) == {"demo_signals"}
    assert set(out.contributions) == {"wav2vec2_xlsr", "wavlm_large", "rawnet3", "aasist"}
    assert 0.0 <= out.score <= 1.0 and 0.0 <= out.confidence <= 1.0


def test_demo_ensemble_separates_real_clips() -> None:
    from pathlib import Path
    from voice_detection.audio import decode_wav
    root = Path(__file__).parents[1]
    genuine = sorted((root / "data" / "genuine").glob("*.wav"))
    spoof = sorted((root / "data" / "spoof").glob("*fake_en*.wav"))
    if not genuine or not spoof:
        pytest.skip("demo WAV corpora not present")
    scorer = DemoTier2Ensemble()
    g = scorer.score(decode_wav(genuine[0].read_bytes())).score
    s = scorer.score(decode_wav(spoof[0].read_bytes())).score
    assert s > g, f"expected spoof ({s:.3f}) > genuine ({g:.3f})"


# --- cloned-voice playback ----------------------------------------------------

def test_scam_audio_is_valid_wav() -> None:
    from voice_detection.twilio_play import build_scam_audio
    payload = build_scam_audio()  # default: 16-bit PCM (most compatible with <Play>)
    assert payload[:4] == b"RIFF" and payload[8:12] == b"WAVE" and payload[12:16] == b"fmt "
    fmt = struct.unpack("<IHHIIHH", payload[16:36])
    assert fmt[1] == 1 and fmt[2] == 1 and fmt[3] == 8_000 and fmt[6] == 16  # PCM, mono, 8 kHz, 16-bit
    assert payload[36:40] == b"data"
    data_len = struct.unpack("<I", payload[40:44])[0]
    assert data_len == len(payload) - 44 and data_len > 8_000  # ≥1 s of cloned voice
    mulaw = build_scam_audio("mulaw")
    assert struct.unpack("<IHHIIHH", mulaw[16:36])[1] == 7  # µ-law variant also available


def test_scam_call_twiml_streams_and_plays() -> None:
    from fastapi.testclient import TestClient
    from voice_detection.api import app
    client = TestClient(app)
    response = client.get("/twiml/scam-call")
    assert response.status_code == 200
    body = response.text
    assert "<Connect>" in body and "/v1/streams/twilio" in body
    assert 'session_id' in body and "<Pause" in body  # call stays open for playback


# --- full live pipeline ------------------------------------------------------

def test_twilio_handler_scores_media_and_fires_trigger_sms() -> None:
    service = DetectionService()
    sessions = SessionStore()
    registry = LiveCallRegistry()
    sender = TwilioClient(account_sid="", auth_token="", from_number="", default_to="")
    engine = AlertEngine(sender=sender, cooldown_s=0, trigger_risk_floor=0, sink=[])
    handler = TwilioCallHandler(service, sessions, registry, engine)

    handler.handle({"event": "start", "start": {"streamSid": "CA123", "customParameters": {"session_id": "live-test"}}})
    audio = tone(3.5)  # ≥ max segment length so the VAD emits mid-stream
    samples = struct.unpack(f"<{len(audio.samples) // 2}h", audio.samples)
    mulaw_frames = bytes(pcm16_to_mulaw(v) for v in resample_linear(samples, 16_000, 8_000))
    payload = base64.b64encode(mulaw_frames).decode()
    result = handler.handle({"event": "media", "media": {"payload": payload}})
    assert result is not None and result["handled"] == "media"

    handler.handle({"event": "transcription", "transcription": {"transcripts": [
        {"transcript": "please send the money right now, urgent"}]}})
    call = registry.get("live-test")
    assert call is not None
    assert any("send" in t or "money" in t for t in call.triggers)
    assert call.alerts, "trigger-word SMS alert should have fired at elevated risk"
    assert call.alerts[-1]["sms_status"] == "logged_console"

    stop = handler.handle({"event": "stop"})
    assert stop["session_id"] == "live-test"
    assert registry.get("live-test").status == "completed"


def test_rest_segment_flows_through_registry() -> None:
    from voice_detection.pipeline import analyze_into_session
    sessions = SessionStore()
    session = sessions.create("rest-test")
    registry = LiveCallRegistry()
    engine = AlertEngine(cooldown_s=999, sink=[])
    record = analyze_into_session(session, tone(1.5), 0, DetectionService(), registry=registry, alerts=engine)
    assert registry.get("rest-test").latest is record
    assert 0 <= record["risk_score"] <= 100

