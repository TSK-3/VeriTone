"""Per-call evidence aggregation; deliberately separate from segment inference."""

from __future__ import annotations

from collections import deque

from .models import SegmentResult


class RunningRiskAggregator:
    """Recency-weighted score accumulator for one live call."""

    def __init__(self, window_size: int = 5, alert_threshold: float = 0.5, min_evidence: int = 3) -> None:
        self._scores: deque[float] = deque(maxlen=window_size)
        self.alert_threshold = alert_threshold
        self.min_evidence = min_evidence

    def add(self, result: SegmentResult) -> tuple[float, int, bool]:
        # High-confidence Tier 2 grows from 50% to 90% of each segment's evidence.
        tier2_weight = 0.50 + (0.40 * result.tier2.confidence)
        evidence = (1 - tier2_weight) * result.tier1.score + tier2_weight * result.tier2.score
        if result.consistency_check.flag == "inconsistent":
            evidence = min(1.0, evidence + 0.15)
        self._scores.append(evidence)
        weights = range(1, len(self._scores) + 1)  # freshest evidence weighs most
        running = sum(score * weight for score, weight in zip(self._scores, weights)) / sum(weights)
        count = len(self._scores)
        return round(running, 4), count, count >= self.min_evidence and running >= self.alert_threshold
