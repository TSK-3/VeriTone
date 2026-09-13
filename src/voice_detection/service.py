from __future__ import annotations

import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .audio import AudioSegment, frame_rms, unpack_pcm16
from .demo_ensemble import DemoTier2Ensemble
from .models import ConsistencyResult, FeatureBreakdown, SegmentResult, Tier1Result, Tier2Result, now_iso
from .tier1_adapter import Tier1CheckpointScorer
from .tier2_ensemble import Tier2ProductionEnsemble, bounded

# Reused across calls; a per-request pool would dominate segment latency.
_POOL = ThreadPoolExecutor(max_workers=2)


def clamp(value: float) -> float:
    return round(bounded(value), 4)


def label(score: float) -> str:
    return "synthetic" if score >= 0.5 else "genuine"


@dataclass(frozen=True)
class SignalStats:
    rms: int
    zero_crossing_rate: float
    silence_ratio: float
    variation: float


def signal_stats(audio: AudioSegment) -> SignalStats:
    values = unpack_pcm16(audio.samples)
    if not values:
        return SignalStats(0, 0.0, 1.0, 0.0)
    rms = int((sum(v * v for v in values) / len(values)) ** 0.5)
    crossings = sum((a >= 0) != (b >= 0) for a, b in zip(values, values[1:]))
    # 20 ms windows: near-silence approximates pauses/breathing opportunity.
    energies = frame_rms(values, audio.sample_rate)
    silence_ratio = sum(1 for level in energies if level < 350) / len(energies)
    variation = statistics.pstdev(abs(v) for v in values) / 32768
    return SignalStats(rms, crossings / (len(values) - 1), silence_ratio, variation)


class HeuristicTier1Scorer:
    """Latency-safe fallback for deployments without a TIER1_CHECKPOINT."""

    def score(self, stats: SignalStats) -> float:
        # Synthetic clips often present unusually uniform energy and pause patterns.
        uniformity = 1 - min(stats.variation / 0.35, 1)
        no_pause = 1 - min(stats.silence_ratio / 0.12, 1)
        zcr_anomaly = min(abs(stats.zero_crossing_rate - 0.08) / 0.15, 1)
        return clamp(0.45 * uniformity + 0.35 * no_pause + 0.20 * zcr_anomaly)


class DetectionService:
    def __init__(self, alert_threshold: float = 0.7, tier2: Tier2ProductionEnsemble | None = None) -> None:
        self.alert_threshold = alert_threshold
        self._tier1 = HeuristicTier1Scorer()
        self._tier1_checkpoint = Tier1CheckpointScorer()
        # TIER2_MODE: "strict" (default in production) demands the four trained ONNX
        # models via TIER2_MANIFEST; "demo" (default locally) uses the clearly labelled
        # DemoTier2Ensemble so the live-call demo works without model artifacts.
        self._tier2_mode = (os.getenv("TIER2_MODE") or "demo").lower()
        self._tier2 = tier2

    def analyze(self, audio: AudioSegment, start_s: float, speaker_similarity: float | None = None, include_features: bool = True) -> SegmentResult:
        if start_s < 0:
            raise ValueError("start_s must be non-negative")
        stats = signal_stats(audio)
        tier1_job = _POOL.submit(self._run_tier1, audio, stats)
        tier2_job = _POOL.submit(self._run_tier2, audio)
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
        score = self._tier1_checkpoint.score(audio) if self._tier1_checkpoint.available else self._tier1.score(stats)
        return Tier1Result(score, label(score), round((time.perf_counter() - started) * 1000))

    def _run_tier2(self, audio: AudioSegment) -> Tier2Result:
        started = time.perf_counter()
        # Strict mode: no fallback at all — scoring is unavailable until all four
        # trained and exported Tier 2 models have been configured via the manifest.
        # Demo mode: signal-based DemoTier2Ensemble, every member labelled
        # "demo_signals" so it can never be mistaken for a trained-model verdict.
        if self._tier2 is None:
            if self._tier2_mode == "strict" or os.getenv("TIER2_MANIFEST"):
                self._tier2 = Tier2ProductionEnsemble()
            else:
                self._tier2 = DemoTier2Ensemble()
        output = self._tier2.score(audio)
        return Tier2Result(output.score, label(output.score), output.confidence, output.contributions,
                           round((time.perf_counter() - started) * 1000), output.auxiliary,
                           output.disagreement, output.quality.score, output.model_status)

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
