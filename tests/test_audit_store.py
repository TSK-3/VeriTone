"""Tests for the durable encrypted audit store, consented speaker references and erasure."""
import math
import struct

import pytest
from fastapi.testclient import TestClient

from voice_detection.api import app
from voice_detection.audio import AudioSegment, pack_pcm16, unpack_pcm16
from voice_detection.audit_store import AuditStore
from voice_detection.pipeline import analyze_into_session
from voice_detection.service import DetectionService
from voice_detection.session_store import SessionStore
from voice_detection.speaker_refs import (
    check_against_reference,
    compute_speaker_embedding,
    similarity,
)


def tone(seconds: float, rate: int = 16_000, amplitude: int = 6_000, frequency: int = 220) -> AudioSegment:
    n = int(seconds * rate)
    values = [int(amplitude * math.sin(2 * math.pi * frequency * i / rate)) for i in range(n)]
    return AudioSegment(samples=struct.pack(f"<{n}h", *values), sample_rate=rate, duration_s=seconds)


def wav_bytes(samples: list[int], rate: int = 16_000) -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pack_pcm16(samples))
    return buffer.getvalue()


@pytest.fixture()
def store(tmp_path) -> AuditStore:
    return AuditStore(tmp_path / "audit.db")


def store_marker(store: AuditStore, session_id: str, risk: dict) -> None:
    store.put_record(session_id, {"session_id": session_id, "risk": risk})


# --- encrypted record roundtrip ----------------------------------------------

def test_record_roundtrip_and_encryption_at_rest(store: AuditStore) -> None:
    store_marker(store, "sess-1", {"tier1": 90})
    records = store.list_records("sess-1")
    assert len(records) == 1 and records[0]["risk"]["tier1"] == 90
    with open(store.path, "rb") as raw:
        assert b"tier1" not in raw.read()  # plaintext risk markers never at rest


def test_disabled_store_never_persists(store: AuditStore, monkeypatch) -> None:
    monkeypatch.setenv("AUDIT_STORE", "off")
    disabled = AuditStore(store.path)
    assert disabled.enabled is False
    assert disabled.put_record("sess-x", {"risk": {}}) is None
    assert disabled.list_records("sess-x") == []


def test_session_erasure_deletes_every_record(store: AuditStore) -> None:
    for index in range(3):
        store_marker(store, "sess-erase", {"segment": index})
    store_marker(store, "sess-keep", {"segment": "other"})
    assert store.delete_session_records("sess-erase") == 3
    assert store.list_records("sess-erase") == []
    assert len(store.list_records("sess-keep")) == 1


# --- consented speaker references ---------------------------------------------

def test_reference_requires_explicit_consent(store: AuditStore) -> None:
    with pytest.raises(ValueError):
        store.put_speaker_reference("spk-1", compute_speaker_embedding(tone(3.0)), False, "t")


def test_reference_roundtrip_similarity_and_erasure(store: AuditStore) -> None:
    clip = tone(3.0)
    embedding = compute_speaker_embedding(clip)
    store.put_speaker_reference("spk-1", embedding, True, "t")
    reference = store.get_speaker_reference("spk-1")
    assert reference is not None and reference["consent"] is True
    assert similarity(embedding, reference["embedding"]) > 0.95
    # live check on a later crop of the same source â†’ high similarity
    crop = AudioSegment(samples=clip.samples[: int(0.9 * len(clip.samples))], sample_rate=16_000, duration_s=0.9)
    assert check_against_reference(crop, reference["embedding"]) > 0.85
    assert store.delete_speaker_reference("spk-1") == 1
    assert store.get_speaker_reference("spk-1") is None


def test_embedding_separates_sources(store: AuditStore) -> None:
    first = compute_speaker_embedding(tone(3.0, frequency=220))
    second = compute_speaker_embedding(tone(3.0, frequency=440, amplitude=2_000))
    assert similarity(first, second) < 0.9


def test_check_rejects_too_short_segments() -> None:
    with pytest.raises(ValueError):
        check_against_reference(tone(0.2), compute_speaker_embedding(tone(3.0)))


# --- pipeline persistence -----------------------------------------------------

def test_analyze_into_session_persists_record(tmp_path, monkeypatch) -> None:
    store = AuditStore(tmp_path / "audit.db")
    monkeypatch.setattr("voice_detection.pipeline.AUDIT_STORE", store)
    session = SessionStore().create("sess-live", speaker_id="spk-live")
    compact = analyze_into_session(session, tone(2.0), 0.0, DetectionService())
    assert 0 <= compact["risk_score"] <= 100
    stored = store.list_records("sess-live")
    segments = [record for record in stored if record.get("kind") == "segment"]
    alerts = [record for record in stored if record.get("kind") == "alert"]
    assert len(segments) == 1 and segments[0]["risk_score"] == compact["risk_score"]
    # the tone crosses the block threshold, so the SMS alert record is durable too
    assert len(alerts) == 1 and alerts[0]["alert"]["risk"] == compact["risk_score"]


def test_session_consistency_runs_against_consented_reference(tmp_path, monkeypatch) -> None:
    store = AuditStore(tmp_path / "audit.db")
    monkeypatch.setattr("voice_detection.pipeline.AUDIT_STORE", store)
    clip = tone(3.0)
    store.put_speaker_reference("spk-1", compute_speaker_embedding(clip), True, "t")
    session = SessionStore().create("sess-ref", speaker_id="spk-1")
    compact = analyze_into_session(session, AudioSegment(samples=clip.samples, sample_rate=16_000, duration_s=1.2), 0.0, DetectionService())
    consistency = compact["segment"]["consistency_check"]
    assert consistency["ran"] is True and consistency["similarity_score"] > 0.85


# --- REST endpoints -----------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    store = AuditStore(tmp_path / "audit.db")
    monkeypatch.setattr("voice_detection.api.AUDIT_STORE", store)
    monkeypatch.setattr("voice_detection.pipeline.AUDIT_STORE", store)
    return TestClient(app)


def test_reference_endpoint_refuses_without_consent(client: TestClient) -> None:
    body = wav_bytes([6_000] * 48_000)
    response = client.post("/v1/speakers/spk-9/reference", content=body, headers={"content-type": "audio/wav"})
    assert response.status_code == 403


def test_reference_and_live_consistency_over_rest(client: TestClient) -> None:
    body = wav_bytes(unpack_pcm16(tone(3.0).samples))
    response = client.post("/v1/speakers/spk-9/reference?consent=true", content=body, headers={"content-type": "audio/wav"})
    assert response.status_code == 200 and response.json()["consent"] is True
    status = client.get("/v1/speakers/spk-9").json()
    assert status["consent"] is True and status["embedding_dims"] > 0

    session = client.post("/v1/sessions", json={"session_id": "sess-rest", "speaker_id": "spk-9"}).json()
    assert session["session_id"] == "sess-rest"
    record = client.post("/v1/sessions/sess-rest/audio?start_s=0", content=body, headers={"content-type": "audio/wav"}).json()
    consistency = record["segment"]["consistency_check"]
    assert consistency["ran"] is True and consistency["similarity_score"] > 0.85

    audit = client.get("/v1/audit?session_id=sess-rest").json()
    assert audit["encrypted_store"] is True and audit["count"] >= 1
    assert client.delete("/v1/audit/sess-rest").json()["deleted_records"] >= 1
    assert client.delete("/v1/speakers/spk-9").json()["deleted_references"] == 1
    assert client.get("/v1/speakers/spk-9").status_code == 404