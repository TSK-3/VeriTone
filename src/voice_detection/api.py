from __future__ import annotations
import base64, json
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from .audio import AudioSegment, decode_wav
from .service import DetectionService
from .session_store import Session, SessionStore

app = FastAPI(title="Verity Voice Integrity API", version="0.2.0", description="Channel-agnostic, privacy-first synthetic-speech risk service.")
service, sessions = DetectionService(), SessionStore()

def require_session(session_id: str) -> Session:
    try: return sessions.get(session_id)
    except KeyError as exc: raise HTTPException(404, "session not found") from exc

def analyze_into_session(session: Session, audio: AudioSegment, start_s: float, similarity: float | None = None) -> dict:
    segment = service.analyze(audio, start_s, similarity, include_features=not session.feature_only_logging)
    record = segment.audit_record()
    signals = {"tier1": round(segment.tier1.score * 100), "tier2": round(segment.tier2.score * 100), "confidence": round(segment.tier2.confidence * 100), "contributions": segment.tier2.encoder_contributions, "auxiliary": segment.tier2.auxiliary_signals, "disagreement": segment.tier2.disagreement, "capture_quality": segment.tier2.capture_quality, "model_status": segment.tier2.model_status, "features": record["feature_breakdown"], "consistency": record["consistency_check"]}
    return session.add({"segment": record, "signals": signals}, segment)

@app.get("/health")
def health() -> dict[str, str]: return {"status": "ok"}

@app.post("/v1/sessions", status_code=201)
async def start_session(request: Request) -> dict:
    body = await request.json(); session_id = body.get("session_id") or str(uuid4())
    try: session = sessions.create(session_id, channel_type=body.get("channel_type", "web_upload"), scenario=body.get("scenario", "support_call"), language_hint=body.get("language_hint"), feature_only_logging=bool(body.get("feature_only_logging", False)))
    except ValueError as exc: raise HTTPException(409, str(exc)) from exc
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
    try: return analyze_into_session(session, decode_wav(await request.body()), start_s, speaker_similarity)
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

@app.websocket("/v1/streams/twilio")
async def twilio_media_stream(websocket: WebSocket) -> None:
    """Twilio Media Streams adapter. A production worker decodes µ-law to 16kHz PCM before scoring."""
    await websocket.accept(); session: Session | None = None; buffer = bytearray()
    try:
        while True:
            message = json.loads(await websocket.receive_text())
            if message.get("event") == "start":
                start = message.get("start", {}); session_id = start.get("customParameters", {}).get("session_id") or start.get("streamSid", str(uuid4()))
                try: session = sessions.get(session_id)
                except KeyError: session = sessions.create(session_id, channel_type="twilio")
            elif message.get("event") == "media" and session:
                buffer.extend(base64.b64decode(message["media"]["payload"]))
                if len(buffer) >= 8_000:
                    await websocket.send_json({"event": "notice", "message": "Media received. Configure µ-law→16kHz PCM worker to enable scoring."}); buffer.clear()
            elif message.get("event") == "stop": break
    except WebSocketDisconnect: pass

@app.post("/v1/calls/{call_id}/segments")
async def legacy_segment(call_id: str, request: Request, start_s: float = Query(0, ge=0), speaker_similarity: float | None = Query(None), feature_only_logging: bool = Query(False)) -> dict:
    try: sessions.get(call_id)
    except KeyError: sessions.create(call_id, feature_only_logging=feature_only_logging)
    return await submit_audio(call_id, request, start_s, speaker_similarity)

app.mount("/", StaticFiles(directory=Path(__file__).parents[2] / "web", html=True), name="dashboard")
