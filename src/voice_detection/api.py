from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .alerts import AlertEngine
from .audio import decode_wav
from .audit_store import STORE as AUDIT_STORE
from .demo_scenario import start_scenario as run_scenario
from .live import REGISTRY
from .models import now_iso
from .pipeline import analyze_into_session
from .service import DetectionService
from .session_store import Session, SessionStore
from .speaker_refs import compute_speaker_embedding
from .twilio_stream import TwilioCallHandler
from .twilio_play import build_scam_audio, wav_bytes_for_twilio

app = FastAPI(title="Verity Voice Integrity API", version="0.3.0",
              description="Channel-agnostic, privacy-first synthetic-speech risk service with live Twilio analysis and SMS prevention.")

service, sessions = DetectionService(), SessionStore()
alerts = AlertEngine()
twilio_handler = TwilioCallHandler(service, sessions, REGISTRY, alerts)


def require_session(session_id: str) -> Session:
    try: return sessions.get(session_id)
    except KeyError as exc: raise HTTPException(404, "session not found") from exc


@app.get("/health")
def health() -> dict[str, str]: return {"status": "ok", "live_calls": str(len(REGISTRY.list()))}


@app.post("/v1/sessions", status_code=201)
async def start_session(request: Request) -> dict:
    body = await request.json(); session_id = body.get("session_id") or str(uuid4())
    try: session = sessions.create(session_id, channel_type=body.get("channel_type", "web_upload"), scenario=body.get("scenario", "support_call"), language_hint=body.get("language_hint"), feature_only_logging=bool(body.get("feature_only_logging", False)), speaker_id=body.get("speaker_id"))
    except ValueError as exc: raise HTTPException(409, str(exc)) from exc
    REGISTRY.register(session_id, channel=session.channel_type, label=body.get("label") or "Dashboard call")
    return {"session_id": session.session_id, "status": session.status, "scenario": session.scenario, "risk_score": 0}


@app.post("/v1/sessions/{session_id}/context")
async def update_context(session_id: str, request: Request) -> dict:
    session = require_session(session_id); body = await request.json()
    allowed = {"caller_reputation", "transaction_amount", "historical_fraud_indicator", "claimed_identity", "call_origin"}
    session.context.update({key: value for key, value in body.items() if key in allowed})
    return session.current_score()


@app.post("/v1/sessions/{session_id}/audio")
async def submit_audio(session_id: str, request: Request, start_s: float = Query(0, ge=0), speaker_similarity: float | None = Query(None)) -> dict:
    session = require_session(session_id)
    if request.headers.get("content-type", "").split(";", 1)[0] not in {"audio/wav", "audio/x-wav", "application/octet-stream"}: raise HTTPException(415, "submit a 16-bit PCM WAV segment")
    try:
        audio = decode_wav(await request.body())
        return analyze_into_session(session, audio, start_s, service, speaker_similarity, REGISTRY, alerts)
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc: raise HTTPException(503, f"Tier 2 unavailable: {exc}") from exc


@app.get("/v1/sessions/{session_id}/score")
def score(session_id: str) -> dict: return require_session(session_id).current_score()


@app.get("/v1/sessions/{session_id}/verdict")
def verdict(session_id: str) -> dict:
    session = require_session(session_id); session.status = "completed"; current = session.current_score()
    return {**current, "verdict": "synthetic" if current["risk_score"] >= 50 else "genuine", "audit_records": len(session.records)}


@app.get("/v1/sessions/{session_id}/audit")
def audit(session_id: str) -> list[dict]: return require_session(session_id).records

# --- live calls, SMS prevention, demo simulator ------------------------------

@app.get("/v1/live")
def live_calls() -> dict:
    """Dashboard poll: every tracked call with its running risk and SMS config."""
    return {"calls": REGISTRY.list(), "sms_configured": alerts.sender.configured, "alerts_total": len(alerts.sink)}


@app.get("/v1/live/{session_id}")
def live_call_detail(session_id: str) -> dict:
    detail = REGISTRY.detail(session_id)
    if detail is None: raise HTTPException(404, "no live call for this id")
    return detail


@app.get("/v1/alerts")
def alert_log(session_id: str | None = Query(None)) -> list[dict]:
    items = [a for a in alerts.sink if session_id is None or a["session_id"] == session_id]
    return items[-50:]


@app.post("/v1/demo/scenario")
async def demo_scenario(request: Request) -> dict:
    try: body = await request.json()
    except Exception: body = {}
    session_id = body.get("session_id") or f"scam-demo-{uuid4()}"
    started = run_scenario(session_id, service, sessions, REGISTRY, alerts, body.get("steps"))
    return {"status": "started", **started}


@app.post("/v1/demo/sms")
async def demo_sms(request: Request) -> dict:
    """Send a test SMS through the configured pipeline (console mode without creds)."""
    try: body = await request.json()
    except Exception: body = {}
    record = alerts.send_test(body.get("to"), body.get("body"))
    if body.get("session_id"): REGISTRY.add_alert(body["session_id"], record.as_dict())
    return {"sms_status": record.sms_status, "to": record.sms_to, "error": record.sms_error, "body": record.body}


async def _inject_cloned_voice(websocket: WebSocket) -> None:
    """Stream the cloned scam script INTO the live call as outbound media (20 ms µ-law packets)."""
    payload = build_scam_audio("mulaw")[44:]  # strip RIFF header → raw µ-law
    for n, offset in enumerate(range(0, len(payload) - 160, 160)):
        chunk = payload[offset:offset + 160]
        await websocket.send_text(json.dumps({
            "event": "media", "sequenceNumber": str(n),
            "media": {"track": "outbound", "chunk": str(n),
                      "payload": base64.b64encode(chunk).decode()},
        }))
        await asyncio.sleep(0.02)  # real-time pacing
    await asyncio.sleep(2.0)


@app.websocket("/v1/streams/twilio")
async def twilio_media_stream(websocket: WebSocket) -> None:
    """Twilio Media Streams adapter: µ-law → 16 kHz PCM → VAD segments → live scoring.

    Sessions named ``scam-play-*`` also get the cloned-voice scam script injected
    as outbound media, so the detector catches its own attacker mid-call.
    """
    await websocket.accept()
    injector: asyncio.Task | None = None
    try:
        while True:
            text = await websocket.receive_text()
            try: message = json.loads(text)
            except json.JSONDecodeError: continue
            event = message.get("event")
            if event == "start":
                twilio_handler.handle(message)
                params = message.get("start", {}).get("customParameters", {}) or {}
                if str(params.get("session_id", "")).startswith("scam-play") or params.get("inject") == "1":
                    injector = asyncio.create_task(_inject_cloned_voice(websocket))
            elif event == "media":
                await asyncio.to_thread(twilio_handler.handle, message)
            else:
                twilio_handler.handle(message)
    except WebSocketDisconnect: pass
    finally:
        if injector is not None:
            injector.cancel()
        twilio_handler.close()


def _public_base_url(request: Request) -> str:
    """Public base URL as Twilio sees it (behind ngrok / Cloudflare tunnels)."""
    configured = os.getenv("PUBLIC_BASE_URL")
    if configured:
        return configured.rstrip("/")
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or "localhost:8000"
    proto = request.headers.get("x-forwarded-proto") or ("https" if "trycloudflare" in host or "ngrok" in host else request.url.scheme)
    return f"{proto}://{host}"


@app.get("/twiml/voice")
@app.post("/twiml/voice")
def twiml_voice(request: Request, session_id: str = Query("caller-live-1")) -> Response:
    """TwiML for inbound/outbound calls: start transcription and stream media to us.

    The Stream URL is derived from the request host (X-Forwarded-Host behind
    tunnels), so pointing the Twilio number's webhook at
    ``https://<tunnel>/twiml/voice`` is the ONLY console configuration needed.
    """
    base = _public_base_url(request)
    ws_scheme = "wss" if base.startswith("https") else "ws"
    host = base.split("://", 1)[1]
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        '<Start><Transcription track="inbound"/></Start>'
        "<Connect>"
        f'<Stream url="{ws_scheme}://{host}/v1/streams/twilio">'
        f'<Parameter name="session_id" value="{session_id}"/>'
        '<Parameter name="label" value="Live Twilio call"/>'
        "</Stream>"
        "</Connect>"
        '<Pause length="600"/>'
        "</Response>"
    )
    return Response(content=xml, media_type="application/xml")


async def _call_destination(request: Request) -> str:
    """Validate the request body / configured default and return the number to dial."""
    try: body = await request.json()
    except Exception: body = {}
    to = body.get("to") or alerts.sender.default_to
    if not alerts.sender.configured:
        raise HTTPException(400, "Twilio credentials not configured (TWILIO_ACCOUNT_SID/AUTH_TOKEN/FROM_NUMBER)")
    if not to:
        raise HTTPException(400, "no destination: pass {'to': '+91…'} or set ALERT_TO_NUMBER")
    return to


def _place_call(to: str, twiml_url: str) -> dict:
    try: return alerts.sender.place_call(to, twiml_url)
    except RuntimeError as exc: raise HTTPException(502, str(exc)) from exc


@app.post("/v1/twilio/call")
async def twilio_place_call(request: Request) -> dict:
    """Place an outbound demo call: Twilio dials ``to``, bridges media to our stream."""
    to = await _call_destination(request)
    twiml_url = f"{_public_base_url(request)}/twiml/voice"
    return {"status": "placing_call", "to": to, "from": alerts.sender.from_number,
            "twiml_url": twiml_url, **_place_call(to, twiml_url)}


@app.get("/v1/twilio/call/{call_sid}")
def twilio_call_status(call_sid: str) -> dict:
    """Poll a placed call's live status from Twilio."""
    try: return {"call_sid": call_sid, **alerts.sender.call_status(call_sid)}
    except RuntimeError as exc: raise HTTPException(502, str(exc)) from exc


# --- cloned-voice playback into a live call -----------------------------------

@app.get("/v1/demo/scam-audio")
def demo_scam_audio(encoding: str = Query("pcm16")) -> Response:
    """8 kHz WAV of the cloned-voice scam script (Twilio <Play>-compatible)."""
    payload, content_type = wav_bytes_for_twilio(encoding)
    return Response(content=payload, media_type=content_type)


def _scam_twiml(base: str, session_id: str = "scam-play-1", mode: str = "connect") -> str:
    """``connect`` (default): <Connect><Stream> — proven to work; the server then
    injects the cloned voice over the WS. ``play``: <Start><Stream> + <Play>."""
    ws_scheme = "wss" if base.startswith("https") else "ws"
    host = base.split("://", 1)[1]
    stream_url = f"{ws_scheme}://{host}/v1/streams/twilio"
    if mode == "play":
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            "<Start>"
            f'<Stream url="{stream_url}">'
            f'<Parameter name="session_id" value="{session_id}"/>'
            '<Parameter name="label" value="Cloned-voice scam playback"/>'
            "</Stream>"
            "</Start>"
            f'<Play>{base}/v1/demo/scam-audio?encoding=pcm16</Play>'
            '<Pause length="2"/>'
            f'<Redirect method="GET">{base}/twiml/scam-call?session_id={session_id}&amp;mode=play</Redirect>'
            "</Response>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{stream_url}">'
        f'<Parameter name="session_id" value="{session_id}"/>'
        '<Parameter name="label" value="Cloned-voice scam playback"/>'
        '<Parameter name="inject" value="1"/>'
        "</Stream>"
        "</Connect>"
        '<Pause length="600"/>'
        "</Response>"
    )


@app.get("/twiml/scam-call")
@app.post("/twiml/scam-call")
def twiml_scam_call(request: Request, session_id: str = Query("scam-play-1"), mode: str = Query("connect")) -> Response:
    """TwiML: <Connect><Stream> to us; the server injects the cloned voice over the WS."""
    return Response(content=_scam_twiml(_public_base_url(request), session_id, mode), media_type="application/xml")


@app.post("/v1/twilio/scam-call")
async def twilio_scam_call(request: Request) -> dict:
    """Call ``to`` and play the CLONED scam script into the call; we catch it live."""
    to = await _call_destination(request)
    base = os.getenv("PUBLIC_BASE_URL") or _public_base_url(request)
    # Trial accounts reject inline `Twiml`; use the hosted TwiML URL instead.
    twiml_url = f"{base}/twiml/scam-call"
    return {"status": "placing_call", "mode": "cloned_voice_playback", "to": to,
            "from": alerts.sender.from_number, "twiml_url": twiml_url, **_place_call(to, twiml_url)}


@app.post("/v1/calls/{call_id}/segments")
async def legacy_segment(call_id: str, request: Request, start_s: float = Query(0, ge=0), speaker_similarity: float | None = Query(None), feature_only_logging: bool = Query(False)) -> dict:
    try: sessions.get(call_id)
    except KeyError:
        sessions.create(call_id, feature_only_logging=feature_only_logging)
        REGISTRY.register(call_id, channel="upload", label="Dashboard upload")
    return await submit_audio(call_id, request, start_s, speaker_similarity)


# --- durable encrypted audit store + consented speaker references --------------

@app.get("/v1/audit")
def audit_records(session_id: str | None = Query(None), limit: int = Query(100, ge=1, le=500)) -> dict:
    """Decrypted derived records from the durable store (never audio)."""
    records = AUDIT_STORE.list_records(session_id, limit)
    return {"count": len(records), "encrypted_store": AUDIT_STORE.enabled, "path": str(AUDIT_STORE.path), "records": records}


@app.delete("/v1/audit/{session_id}")
def erase_audit_records(session_id: str) -> dict:
    deleted = AUDIT_STORE.delete_session_records(session_id)
    return {"session_id": session_id, "deleted_records": deleted}


@app.post("/v1/speakers/{speaker_id}/reference")
async def enrol_speaker_reference(speaker_id: str, request: Request, consent: bool = Query(False)) -> dict:
    """Enrol a speaker reference (WAV body) — stored encrypted, consent mandatory.

    The waveform is embedded in memory and discarded; only the consented spectral
    reference vector is persisted. Later segments can then be checked live.
    """
    if not consent:
        raise HTTPException(403, "speaker reference requires explicit consent=true — nothing stored")
    if request.headers.get("content-type", "").split(";", 1)[0] not in {"audio/wav", "audio/x-wav", "application/octet-stream"}:
        raise HTTPException(415, "submit a 16-bit PCM WAV enrolment clip")
    try:
        audio = decode_wav(await request.body())
        embedding = compute_speaker_embedding(audio)
        stored = AUDIT_STORE.put_speaker_reference(speaker_id, embedding, consent, now_iso())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"speaker_id": speaker_id, "consent": True, "reference_on_file": stored,
            "embedding_dims": len(embedding), "audio_retained": False}


@app.get("/v1/speakers/{speaker_id}")
def speaker_reference(speaker_id: str) -> dict:
    reference = AUDIT_STORE.get_speaker_reference(speaker_id)
    if reference is None:
        raise HTTPException(404, "no reference on file for this speaker")
    return {"speaker_id": speaker_id, "consent": reference["consent"],
            "created_at": reference["created_at"], "embedding_dims": len(reference.get("embedding", []))}


@app.delete("/v1/speakers/{speaker_id}")
def erase_speaker_reference(speaker_id: str) -> dict:
    deleted = AUDIT_STORE.delete_speaker_reference(speaker_id)
    return {"speaker_id": speaker_id, "deleted_references": deleted}


app.mount("/", StaticFiles(directory=Path(__file__).parents[2] / "web", html=True), name="dashboard")
