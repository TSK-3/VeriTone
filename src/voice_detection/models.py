from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal

Label = Literal["genuine", "synthetic"]
Level = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class Tier1Result:
    score: float
    label: Label
    latency_ms: int


@dataclass(frozen=True)
class Tier2Result:
    score: float
    label: Label
    confidence: float
    encoder_contributions: dict[str, float]
    latency_ms: int


@dataclass(frozen=True)
class ConsistencyResult:
    ran: bool
    similarity_score: float | None
    flag: Literal["consistent", "inconsistent", "no_reference_available"]


@dataclass(frozen=True)
class FeatureBreakdown:
    prosody_irregularity: Level
    spectral_artifacts: Level
    breathing_pattern: Literal["present", "absent", "irregular"]
    background_noise_consistency: Literal["consistent", "inconsistent"]


@dataclass(frozen=True)
class SegmentResult:
    segment_timestamp_range: tuple[float, float]
    tier1: Tier1Result
    tier2: Tier2Result
    combined_risk_score: float
    running_risk_score: float
    evidence_segments: int
    consistency_check: ConsistencyResult
    feature_breakdown: FeatureBreakdown | None
    timestamp: str
    alert: bool
    recommended_action: str | None

    def audit_record(self) -> dict:
        """Return derived data only. This object never has audio or embeddings."""
        return asdict(self)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
