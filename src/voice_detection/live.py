"""Live call registry shared by the Twilio stream, the demo simulator and the dashboard.

Keeps only derived data (scores, transcript text, alert records) — never audio —
consistent with the project's privacy contract.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LiveCall:
    session_id: str
    channel: str = "twilio"
    label: str = ""
    started_at: str = field(default_factory=now_iso)
    status: str = "live"
    latest: dict = field(default_factory=dict)
    records: deque = field(default_factory=lambda: deque(maxlen=120))
    transcript: deque = field(default_factory=lambda: deque(maxlen=80))
    alerts: list = field(default_factory=list)
    triggers: list = field(default_factory=list)
    peak_risk: int = 0

    def summary(self) -> dict:
        last = self.records[-1] if self.records else {}
        signals = last.get("signals", {})
        return {
            "session_id": self.session_id,
            "channel": self.channel,
            "label": self.label,
            "status": self.status,
            "started_at": self.started_at,
            "risk_score": last.get("risk_score", 0),
            "peak_risk": self.peak_risk,
            "evidence_segments": last.get("evidence_segments", 0),
            "action": last.get("action", "pass"),
            "recommended_action": last.get("recommended_action", ""),
            "signals": signals,
            "alerts": len(self.alerts),
            "trigger_hits": len(self.triggers),
            "transcript_tail": list(self.transcript)[-3:],
            "updated_at": now_iso(),
        }

    def detail(self) -> dict:
        payload = self.summary()
        payload.update({
            "records": list(self.records)[-25:],
            "transcript": list(self.transcript),
            "alerts": self.alerts,
            "triggers": self.triggers[-40:],
        })
        return payload


class LiveCallRegistry:
    """Process-wide singleton registry (one process == one demo console)."""

    def __init__(self) -> None:
        self._calls: dict[str, LiveCall] = {}

    def register(self, session_id: str, channel: str = "twilio", label: str = "") -> LiveCall:
        call = self._calls.get(session_id)
        if call is None:
            call = LiveCall(session_id=session_id, channel=channel, label=label or channel)
            self._calls[session_id] = call
        else:
            call.channel = channel or call.channel
            if label:
                call.label = label
        return call

    def ensure(self, session_id: str) -> LiveCall:
        return self._calls.setdefault(session_id, LiveCall(session_id=session_id))

    def get(self, session_id: str) -> LiveCall | None:
        return self._calls.get(session_id)

    def attach_record(self, session_id: str, record: dict) -> None:
        call = self.ensure(session_id)
        call.records.append(record)
        call.latest = record
        call.peak_risk = max(call.peak_risk, int(record.get("risk_score", 0)))

    def add_transcript(self, session_id: str, text: str, source: str = "twilio",
                       triggers: list[str] | None = None, risk: int | None = None) -> dict:
        call = self.ensure(session_id)
        entry = {"text": text, "source": source, "triggers": triggers or [],
                 "risk": risk, "timestamp": now_iso()}
        call.transcript.append(entry)
        return entry

    def add_alert(self, session_id: str, alert: dict) -> None:
        self.ensure(session_id).alerts.append(alert)

    def add_triggers(self, session_id: str, words: list[str]) -> None:
        if words:
            self.ensure(session_id).triggers.extend(words)

    def complete(self, session_id: str) -> None:
        if session_id in self._calls:
            self._calls[session_id].status = "completed"

    def list(self) -> list[dict]:
        return [call.summary() for call in self._calls.values()]

    def detail(self, session_id: str) -> dict | None:
        call = self._calls.get(session_id)
        return call.detail() if call else None


REGISTRY = LiveCallRegistry()
