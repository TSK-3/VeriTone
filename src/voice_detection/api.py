from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles

from .audio import decode_wav
from .aggregation import RunningRiskAggregator
from .service import DetectionService

app = FastAPI(title="VeriTone API", version="0.1.0")
service = DetectionService()
audit: dict[str, list[dict]] = defaultdict(list)
aggregators: dict[str, RunningRiskAggregator] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/calls/{call_id}/segments")
async def score_segment(
    call_id: str,
    request: Request,
    start_s: float = Query(default=0.0, ge=0),
    speaker_similarity: float | None = Query(default=None),
    feature_only_logging: bool = Query(default=False),
) -> dict:
    if request.headers.get("content-type", "").split(";", 1)[0] not in {"audio/wav", "audio/x-wav", "application/octet-stream"}:
        raise HTTPException(415, "submit a WAV audio segment")
    # Reading the body directly avoids multipart upload spooling raw audio to disk.
    payload = await request.body()
    try:
        result = service.analyze(decode_wav(payload), start_s, speaker_similarity, include_features=not feature_only_logging)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    aggregator = aggregators.setdefault(call_id, RunningRiskAggregator(alert_threshold=service.alert_threshold))
    running_score, evidence_segments, alert = aggregator.add(result)
    result = replace(result, running_risk_score=running_score, evidence_segments=evidence_segments, alert=alert,
                     recommended_action="Request secondary verification before proceeding." if alert else None)
    record = result.audit_record()
    audit[call_id].append(record)
    return record


@app.get("/v1/calls/{call_id}/audit")
def call_audit(call_id: str) -> list[dict]:
    """Returns only persisted derived scores and metadata, never waveform bytes."""
    return audit.get(call_id, [])


app.mount("/", StaticFiles(directory=Path(__file__).parents[2] / "web", html=True), name="dashboard")
