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

from .audio import AudioSegment
from .tier2_ensemble import (
    MODEL_NAMES,
    EnsembleOutput,
    bounded,
    inspect_quality,
    prosody_signal,
)

# Feature calibration, fitted on the repo's own MLAAD clips (15+15 measured):
# spectral flatness  GEN 0.378+/-0.075  SPOOF 0.257+/-0.055  (separation 0.93 sigma)
# spectral centroid  GEN 2177+/-249     SPOOF 1708+/-296     (0.86 sigma)
# pause ratio        GEN 0.426+/-0.081  SPOOF 0.300+/-0.122  (0.62 sigma)
# highband ratio     GEN 0.287+/-0.049  SPOOF 0.220+/-0.047  (0.70 sigma)
# TTS/clone audio is flatter, spectrally darker and pauses less than human speech.
# All four signals are inverted: feature BELOW center => member score ABOVE 0.5.
_CENTERS = {"flat": 0.30, "centroid": 1900.0, "pause": 0.35, "highband": 0.25}
_GAINS = {"flat": 3.0, "centroid": 0.0004, "pause": 2.0, "highband": 4.0}

_TEMPERATURE = 0.9
_FUSION_WEIGHTS = {"wav2vec2_xlsr": 0.24, "wavlm_large": 0.24, "rawnet3": 0.30, "aasist": 0.22}


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, value))))


def spectral_features(audio: AudioSegment) -> dict[str, float]:
    """FFT-frame features that separate TTS/clone audio from human speech.

    ``flat``  — mean spectral flatness (TTS spectra are flatter),
    ``centroid`` — mean spectral centroid in Hz (TTS is spectrally darker),
    ``highband`` — energy fraction above 3 kHz,
    ``pause`` — fraction of near-silent 32 ms frames (humans pause more).
    """
    try:
        import numpy as np
    except ImportError:  # graceful degradation: neutral features keep API-only runs alive
        return {"flat": _CENTERS["flat"], "centroid": _CENTERS["centroid"], "highband": _CENTERS["highband"], "pause": _CENTERS["pause"]}
    values = np.frombuffer(audio.samples, dtype="<i2").astype(np.float32) / 32768.0
    if values.size == 0:
        return {"flat": _CENTERS["flat"], "centroid": _CENTERS["centroid"], "highband": _CENTERS["highband"], "pause": _CENTERS["pause"]}
    frame_len = max(256, int(0.032 * audio.sample_rate))
    frames = [values[i:i + frame_len] for i in range(0, len(values) - frame_len, frame_len)]
    if not frames:
        return {"flat": _CENTERS["flat"], "centroid": _CENTERS["centroid"], "highband": _CENTERS["highband"], "pause": _CENTERS["pause"]}
    rms = np.array([math.sqrt(float((f * f).mean())) for f in frames])
    window = np.hanning(frames[0].size)
    n_fft = 1024
    mag = np.array([np.abs(np.fft.rfft(f * window, n_fft)) for f in frames])
    mag = np.maximum(mag, 1e-9)
    flatness = float((np.exp(np.log(mag).mean(axis=1)) / mag.mean(axis=1)).mean())
    freqs = np.fft.rfftfreq(n_fft, 1 / audio.sample_rate)
    centroid = float(((mag * freqs).sum(axis=1) / mag.sum(axis=1)).mean())
    highband = float((mag[:, freqs > 3000].sum(axis=1) / mag.sum(axis=1)).mean())
    pause = float((rms < 0.01).mean())
    return {
        "flat": round(flatness, 5),
        "centroid": round(centroid, 3),
        "highband": round(highband, 5),
        "pause": round(pause, 5),
    }


class DemoTier2Ensemble:
    """Four labelled demo scorers + prosody + quality gate + calibrated fusion."""

    def __init__(self) -> None:
        self.status: dict[str, str] = {name: "demo_signals" for name in MODEL_NAMES}

    def score(self, audio: AudioSegment) -> EnsembleOutput:
        quality = inspect_quality(audio)
        prosody = prosody_signal(audio, quality)
        features_raw = spectral_features(audio)
        # Synthetic audio: flatter, darker, fewer pauses, less highband => scores > 0.5.
        scores = {
            "wav2vec2_xlsr": bounded(0.5 + _GAINS["flat"] * (_CENTERS["flat"] - features_raw["flat"])),
            "wavlm_large": bounded(0.5 + _GAINS["centroid"] * (_CENTERS["centroid"] - features_raw["centroid"])),
            "rawnet3": bounded(0.5 + _GAINS["pause"] * (_CENTERS["pause"] - features_raw["pause"])),
            "aasist": bounded(0.5 + _GAINS["highband"] * (_CENTERS["highband"] - features_raw["highband"])),
        }
        raw = sum(scores[key] * _FUSION_WEIGHTS.get(key, 0.0) for key in scores)
        raw /= max(sum(_FUSION_WEIGHTS.get(key, 0.0) for key in scores), 1e-6)
        raw = bounded(raw)
        calibrated = _sigmoid((math.log((raw + 1e-4) / (1 - raw + 1e-4))) / max(_TEMPERATURE, 0.05))
        disagreement = statistics.pstdev(scores.values())
        confidence = bounded(quality.score * (1 - min(disagreement / 0.30, 1)) * 0.95)
        score = 0.5 + (calibrated - 0.5) * (0.55 + 0.45 * confidence)
        return EnsembleOutput(
            score=round(bounded(score), 4),
            confidence=round(confidence, 4),
            contributions={key: round(value, 4) for key, value in scores.items()},
            auxiliary={**prosody, "flat": features_raw["flat"], "centroid": features_raw["centroid"], "highband": features_raw["highband"], "pause": features_raw["pause"]},
            disagreement=round(disagreement, 4),
            quality=quality,
            model_status=self.status.copy(),
        )
