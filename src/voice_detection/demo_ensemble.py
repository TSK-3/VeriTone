"""Demo stand-in for the Tier 2 anti-spoof ensemble.

The production ensemble (``tier2_ensemble.Tier2ProductionEnsemble``) is strict: it
requires four trained ONNX models and refuses to score without them. For hackathon
demos where no manifest is configured, ``DemoTier2Ensemble`` derives plausible
per-model contributions from distinct waveform views (band energies, spectral flux,
zero-crossing cadence) and fuses them with the same calibrated math as production.

Every member is labelled ``demo_signals`` so a demo score can never be mistaken for
a trained-model verdict. Set ``TIER2_MODE=strict`` to disable it entirely.
"""
from __future__ import annotations

import math
import statistics
import struct

from .audio import AudioSegment
from .tier2_ensemble import (
    MODEL_NAMES,
    AudioQuality,
    EnsembleOutput,
    bounded,
    inspect_quality,
    prosody_signal,
)

# Feature calibration (initial values; refined against the repo's own MLAAD clips):
# genuine telephone speech sits near the low end of each feature, cloned/TTS audio
# near the high end. Maps: member_score = 0.5 + gain * (feature - center).
_LOW_CENTER, _LOW_GAIN = 0.34, 2.2      # energy below 0.25 * Nyquist (formant band)
_HIGH_CENTER, _HIGH_GAIN = 0.10, 9.0    # energy above 0.60 * Nyquist (hiss/texture)
_FLUX_CENTER, _FLUX_GAIN = 0.16, 3.4    # frame-to-frame energy flux variability
_ZCR_CENTER, _ZCR_GAIN = 0.09, 4.0      # zero-crossing-rate cadence deviation

_TEMPERATURE = 0.9
_FUSION_WEIGHTS = {"wav2vec2_xlsr": 0.24, "wavlm_large": 0.24, "rawnet3": 0.30, "aasist": 0.22, "prosody": 0.16}


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, value))))


def _one_pole_lowpass(values: list[int], alpha: float) -> list[float]:
    output: list[float] = []
    state = 0.0
    for value in values:
        state += alpha * (value - state)
        output.append(state)
    return output


def _alpha_for_cutoff(cutoff_hz: float, sample_rate: int) -> float:
    return 1 - math.exp(-2 * math.pi * cutoff_hz / sample_rate)


def band_features(audio: AudioSegment) -> dict[str, float]:
    """Cheap O(n) spectral-view features computed at the audio's native rate.

    ``low`` / ``high`` are one-pole band energy ratios (formant band vs. high-band
    texture), ``flux`` is normalised frame-energy change, ``zcr`` the zero-crossing
    rate. All are rate-agnostic because cutoffs are fractions of Nyquist.
    """
    values = struct.unpack(f"<{len(audio.samples) // 2}h", audio.samples)
    if not values:
        return {"low": 0.0, "high": 0.0, "flux": 0.0, "zcr": 0.0}
    nyquist = audio.sample_rate / 2
    low = _one_pole_lowpass(values, _alpha_for_cutoff(0.25 * nyquist, audio.sample_rate))
    high_residual = [value - state for value, state in zip(values, _one_pole_lowpass(values, _alpha_for_cutoff(0.60 * nyquist, audio.sample_rate)))]
    total_energy = sum(value * value for value in values) / len(values)
    low_energy = sum(value * value for value in low) / len(low)
    high_energy = sum(value * value for value in high_residual) / len(high_residual)
    total_energy = max(total_energy, 1e-9)
    # 20 ms frame energies for flux.
    frame = max(1, int(audio.sample_rate * 0.02))
    frame_energies = [
        math.sqrt(sum(value * value for value in values[i:i + frame]) / max(1, len(values[i:i + frame])))
        for i in range(0, len(values), frame)
    ]
    mean_energy = statistics.fmean(frame_energies)
    flux = statistics.fmean(abs(b - a) for a, b in zip(frame_energies, frame_energies[1:])) / (mean_energy + 1e-9) if len(frame_energies) > 1 else 0.0
    crossings = sum((a >= 0) != (b >= 0) for a, b in zip(values, values[1:]))
    return {
        "low": round(low_energy / total_energy, 5),
        "high": round(high_energy / total_energy, 5),
        "flux": round(min(flux, 1.0), 5),
        "zcr": round(crossings / max(1, len(values) - 1), 5),
    }


class DemoTier2Ensemble:
    """Four labelled demo scorers + prosody + quality gate + calibrated fusion."""

    def __init__(self) -> None:
        self.status: dict[str, str] = {name: "demo_signals" for name in MODEL_NAMES}

    def score(self, audio: AudioSegment) -> EnsembleOutput:
        quality = inspect_quality(audio)
        prosody = prosody_signal(audio, quality)
        features_raw = band_features(audio)
        scores = {
            "wav2vec2_xlsr": bounded(0.5 + _LOW_GAIN * (features_raw["low"] - _LOW_CENTER)),
            "wavlm_large": bounded(0.5 + _HIGH_GAIN * (features_raw["high"] - _HIGH_CENTER)),
            "rawnet3": bounded(0.5 + _FLUX_GAIN * (features_raw["flux"] - _FLUX_CENTER)),
            "aasist": bounded(0.5 + _ZCR_GAIN * (features_raw["zcr"] - _ZCR_CENTER)),
        }
        features = {**scores, "prosody": prosody["prosody_score"]}
        raw = sum(features[key] * _FUSION_WEIGHTS.get(key, 0.0) for key in features)
        raw /= max(sum(_FUSION_WEIGHTS.get(key, 0.0) for key in features), 1e-6)
        raw = bounded(raw)
        calibrated = _sigmoid((math.log((raw + 1e-4) / (1 - raw + 1e-4))) / max(_TEMPERATURE, 0.05))
        disagreement = statistics.pstdev(scores.values())
        confidence = bounded(quality.score * (1 - min(disagreement / 0.30, 1)) * 0.95)
        score = 0.5 + (calibrated - 0.5) * (0.55 + 0.45 * confidence)
        return EnsembleOutput(
            score=round(bounded(score), 4),
            confidence=round(confidence, 4),
            contributions={key: round(value, 4) for key, value in scores.items()},
            auxiliary={**prosody, "band_low": features_raw["low"], "band_high": features_raw["high"], "flux": features_raw["flux"]},
            disagreement=round(disagreement, 4),
            quality=quality,
            model_status=self.status.copy(),
        )
