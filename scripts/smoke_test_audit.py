"""End-to-end smoke test for the durable audit store + consented reference flow."""
import io
import sys
import time
import wave
from pathlib import Path

import httpx

ROOT = Path(__file__).parents[1]
BASE = "http://127.0.0.1:8901"


def wav_bytes(samples, rate=16_000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(bytearray(b"".join(int(v).to_bytes(2, "little", signed=True) for v in samples))))
    return buf.getvalue()


def main() -> int:
    import math
    import uuid

    run = uuid.uuid4().hex[:8]
    speaker = f"edge-demo-{run}"
    session_id = f"edge-demo-{run}"
    tone = [int(6000 * math.sin(2 * math.pi * 220 * i / 16_000)) for i in range(48_000)]
    body = wav_bytes(tone)

    for _ in range(40):  # wait for uvicorn to finish loading torch
        try:
            if httpx.get(f"{BASE}/health", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(1)
    else:
        print("FAIL: server never became healthy")
        return 1
    print("health:", httpx.get(f"{BASE}/health", timeout=5).json())

    ref = httpx.post(f"{BASE}/v1/speakers/{speaker}/reference?consent=true", content=body,
                     headers={"content-type": "audio/wav"}, timeout=30)
    print("enrol:", ref.status_code, ref.json())
    assert ref.status_code == 200 and ref.json()["consent"] is True

    no_consent = httpx.post(f"{BASE}/v1/speakers/{speaker}-2/reference", content=body,
                            headers={"content-type": "audio/wav"}, timeout=30)
    print("no-consent enrol:", no_consent.status_code, no_consent.json()["detail"][:40])
    assert no_consent.status_code == 403

    session = httpx.post(f"{BASE}/v1/sessions", json={"session_id": session_id, "speaker_id": speaker}, timeout=30)
    print("session:", session.status_code, session.json())
    assert session.status_code == 201

    seg = httpx.post(f"{BASE}/v1/sessions/{session_id}/audio?start_s=0", content=body,
                     headers={"content-type": "audio/wav"}, timeout=60)
    consistency = seg.json()["segment"]["consistency_check"]
    print("segment:", seg.status_code, "consistency:", consistency)
    assert seg.status_code == 200 and consistency["ran"] is True and consistency["similarity_score"] > 0.85

    store = httpx.get(f"{BASE}/v1/audit?session_id={session_id}", timeout=30).json()
    print("durable store: count =", store["count"], "encrypted =", store["encrypted_store"], "path =", store["path"])
    assert store["count"] >= 1 and store["encrypted_store"] is True

    erases = httpx.delete(f"{BASE}/v1/audit/{session_id}", timeout=30).json()
    print("audit erasure:", erases)
    gone = httpx.delete(f"{BASE}/v1/speakers/{speaker}", timeout=30).json()
    print("reference erasure:", gone)
    assert httpx.get(f"{BASE}/v1/speakers/{speaker}", timeout=30).status_code == 404
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
