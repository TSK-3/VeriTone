"""Shared scoring pipeline used by the REST API, the Twilio stream and the simulator.

One path, one behaviour: score the segment, fold it into the session's running
risk, attach the derived record to the live registry and evaluate SMS alert rules.
"""
from __future__ import annotations

from .alerts import AlertEngine
from .audio import AudioSegment
from .live import LiveCallRegistry, REGISTRY
from .service import DetectionService
from .session_store import Session


def analyze_into_session(session: Session, audio: AudioSegment, start_s: float,
                         service: DetectionService, similarity: float | None = None,
                         registry: LiveCallRegistry = REGISTRY,
                         alerts: AlertEngine | None = None) -> dict:
    """Score one segment and return the compact record the dashboard/API consumes."""
    if alerts is None:
        alerts = AlertEngine()
    segment = service.analyze(audio, start_s, similarity, include_features=not session.feature_only_logging)
    record = segment.audit_record()
    signals = {
        "tier1": round(segment.tier1.score * 100),
        "tier2": round(segment.tier2.score * 100),
        "confidence": round(segment.tier2.confidence * 100),
        "contributions": segment.tier2.encoder_contributions,
        "auxiliary": segment.tier2.auxiliary_signals,
        "disagreement": segment.tier2.disagreement,
        "capture_quality": segment.tier2.capture_quality,
        "model_status": segment.tier2.model_status,
        "features": record["feature_breakdown"],
        "consistency": record["consistency_check"],
        "tier1_latency_ms": segment.tier1.latency_ms,
        "tier2_latency_ms": segment.tier2.latency_ms,
    }
    compact = session.add({"segment": record, "signals": signals}, segment)
    registry.attach_record(session.session_id, compact)
    alert = alerts.check_after_record(session.session_id, compact)
    if alert is not None:
        registry.add_alert(session.session_id, alert.as_dict())
    return compact
