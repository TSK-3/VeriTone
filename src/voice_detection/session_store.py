"""In-memory session state for the hackathon demo; persists only derived records."""
from __future__ import annotations
from dataclasses import dataclass, field
from .aggregation import RunningRiskAggregator
from .models import SegmentResult
from .workflows import choose_action, enrich_risk

@dataclass
class Session:
    session_id: str
    channel_type: str = "upload"
    scenario: str = "support_call"
    language_hint: str | None = None
    feature_only_logging: bool = False
    context: dict = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)
    aggregator: RunningRiskAggregator = field(default_factory=RunningRiskAggregator)
    status: str = "active"

    def current_score(self) -> dict:
        latest = self.records[-1] if self.records else {}
        return {"session_id": self.session_id, "status": self.status, "risk_score": latest.get("risk_score", 0), "evidence_segments": latest.get("evidence_segments", 0), "action": latest.get("action", "pass"), "context": self.context, "latest_signals": latest.get("signals", {})}

    def add(self, record: dict, segment: SegmentResult) -> dict:
        model_risk, evidence_segments, _ = self.aggregator.add(segment)
        risk = enrich_risk(model_risk, self.context)
        action = choose_action(round(risk * 100), self.scenario)
        compact = {**record, "risk_score": round(risk * 100), "model_risk_score": round(model_risk * 100), "evidence_segments": evidence_segments, "action": action.kind, "recommended_action": action.message}
        self.records.append(compact)
        return compact

class SessionStore:
    def __init__(self) -> None: self._sessions: dict[str, Session] = {}
    def create(self, session_id: str, **kwargs) -> Session:
        if session_id in self._sessions: raise ValueError("session already exists")
        session = Session(session_id=session_id, **kwargs); self._sessions[session_id] = session; return session
    def get(self, session_id: str) -> Session:
        if session_id not in self._sessions: raise KeyError("session not found")
        return self._sessions[session_id]
