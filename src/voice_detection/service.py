from __future__ import annotations

import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .audio import AudioSegment
from .models import ConsistencyResult, FeatureBreakdown, SegmentResult, Tier1Result, Tier2Result, now_iso
from .tier1_adapter import Tier1CheckpointScorer


def clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def label(score: float) -> str:
    return "synthetic" if score >= 0.5 else "genuine"


@dataclass(frozen=True)
class SignalStats:
    rms: int
    zero_crossing_rate: float
    silence_ratio: float
    variation: float


def signal_stats(audio: AudioSegment) -> SignalStats:
    samples = audio.samples
    values = [int.from_bytes(samples[i:i + 2], "little", signed=True) for i in range(0, len(samples), 2)]
    rms = int(math.sqrt(sum(value * value for value in values) / len(values)))
    crossings = sum((a >= 0) != (b >= 0) for a, b in zip(values, values[1:]))
    # 20ms windows: near-silence approximates pauses/breathing opportunity.
    window = max(1, int(audio.sample_rate * 0.02))
    windows = [values[i:i + window] for i in range(0, len(values), window)]
    silence_ratio = sum(1 for chunk in windows if chunk and math.sqrt(sum(x * x for x in chunk) / len(chunk)) < 350) / len(windows)
    variation = statistics.pstdev(abs(v) for v in values) / 32768 if len(values) > 1 else 0.0
    return SignalStats(rms, crossings / max(1, len(values) - 1), silence_ratio, variation)


class HeuristicTier1Scorer:
    """Latency-safe development adapter; replace with an edge model checkpoint."""

    def score(self, stats: SignalStats) -> float:
        # Synthetic clips often present unusually uniform energy and pause patterns.
        uniformity = 1 - min(stats.variation / 0.35, 1)
        no_pause = 1 - min(stats.silence_ratio / 0.12, 1)
        zcr_anomaly = min(abs(stats.zero_crossing_rate - 0.08) / 0.15, 1)
        return clamp(0.45 * uniformity + 0.35 * no_pause + 0.20 * zcr_anomaly)


class HeuristicTier2Scorer:
    """Three independent stand-in heads plus learned-fusion-shaped weighting."""

    def score(self, stats: SignalStats) -> tuple[float, dict[str, float]]:
        spectral = clamp(0.55 * (1 - min(stats.variation / 0.4, 1)) + 0.45 * min(abs(stats.zero_crossing_rate - 0.09) / 0.12, 1))
        prosody = clamp(0.70 * (1 - min(stats.silence_ratio / 0.15, 1)) + 0.30 * (1 - min(stats.variation / 0.3, 1)))
        waveform = clamp(0.60 * min(abs(stats.zero_crossing_rate - 0.07) / 0.13, 1) + 0.40 * (1 - min(stats.rms / 6000, 1)))
        contributions = {"wav2vec2_xlsr": spectral, "wavlm_large": prosody, "rawnet3": waveform}
        # Fixed coefficients define the interface only; production uses trained logistic fusion.
        return clamp(0.36 * spectral + 0.38 * prosody + 0.26 * waveform), contributions


class DetectionService:
    def __init__(self, alert_threshold: float = 0.7, require_tier1_checkpoint: bool = True) -> None:
        self.alert_threshold = alert_threshold
        self._tier1 = HeuristicTier1Scorer()
        self._tier1_checkpoint = Tier1CheckpointScorer() if require_tier1_checkpoint else None
        self._tier2 = HeuristicTier2Scorer()

    def analyze(self, audio: AudioSegment, start_s: float, speaker_similarity: float | None = None, include_features: bool = True) -> SegmentResult:
        if start_s < 0:
            raise ValueError("start_s must be non-negative")
        stats = signal_stats(audio)
        with ThreadPoolExecutor(max_workers=2) as pool:
            tier1_job = pool.submit(self._run_tier1, audio, stats)
            tier2_job = pool.submit(self._run_tier2, stats)
            tier1, tier2 = tier1_job.result(), tier2_job.result()
        consistency = self._consistency(speaker_similarity)
        risk = clamp(0.35 * tier1.score + 0.65 * tier2.score + (0.15 if consistency.flag == "inconsistent" else 0))
        features = self._features(stats) if include_features else None
        # Segment scores are evidence, never alert verdicts in isolation.
        return SegmentResult((start_s, round(start_s + audio.duration_s, 3)), tier1, tier2, risk, 0.0, 0, consistency, features, now_iso(), False, None)

    def _run_tier1(self, audio: AudioSegment, stats: SignalStats) -> Tier1Result:
        started = time.perf_counter()
        # A model checkpoint takes precedence. Heuristics keep the demo operable
        # until training has produced a checkpoint.
        if self._tier1_checkpoint is None:
            score = self._tier1.score(stats)
        else:
            score = self._tier1_checkpoint.score(audio)
        return Tier1Result(score, label(score), round((time.perf_counter() - started) * 1000))

    def _run_tier2(self, stats: SignalStats) -> Tier2Result:
        started = time.perf_counter()
        score, contributions = self._tier2.score(stats)
        confidence = clamp(abs(score - 0.5) * 2)
        return Tier2Result(score, label(score), confidence, contributions, round((time.perf_counter() - started) * 1000))

    @staticmethod
    def _consistency(similarity: float | None) -> ConsistencyResult:
        if similarity is None:
            return ConsistencyResult(False, None, "no_reference_available")
        if not 0 <= similarity <= 1:
            raise ValueError("speaker_similarity must be between 0 and 1")
        return ConsistencyResult(True, similarity, "consistent" if similarity >= 0.72 else "inconsistent")

    @staticmethod
    def _features(stats: SignalStats) -> FeatureBreakdown:
        level = lambda value: "high" if value >= 0.67 else "medium" if value >= 0.34 else "low"
        prosody = level(1 - min(stats.variation / 0.35, 1))
        spectral = level(min(abs(stats.zero_crossing_rate - 0.08) / 0.15, 1))
        breathing = "absent" if stats.silence_ratio < 0.03 else "irregular" if stats.silence_ratio > 0.30 else "present"
        noise = "inconsistent" if stats.variation < 0.06 else "consistent"
        return FeatureBreakdown(prosody, spectral, breathing, noise)
