"""Robust Tier 2 anti-spoof ensemble with calibration and disagreement handling.

Each deep-model export must accept normalised 16 kHz PCM waveform shaped `[1, T]`
and emit one binary spoof logit or probability. Keeping preprocessing inside each
export makes model servers/versioning deterministic and prevents feature skew.
"""
from __future__ import annotations

import json, math, os, statistics, struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .audio import AudioSegment

MODEL_NAMES = ("wav2vec2_xlsr", "wavlm_large", "rawnet3", "aasist")

def bounded(value: float) -> float: return max(0.0, min(1.0, value))
def sigmoid(value: float) -> float: return 1 / (1 + math.exp(-max(-30.0, min(30.0, value))))

@dataclass(frozen=True)
class AudioQuality:
    score: float
    duration_ok: bool
    low_energy: bool
    clipped_ratio: float
    pause_ratio: float

@dataclass(frozen=True)
class EnsembleOutput:
    score: float
    confidence: float
    contributions: dict[str, float]
    auxiliary: dict[str, float]
    disagreement: float
    quality: AudioQuality
    model_status: dict[str, str]

class BinaryScorer(Protocol):
    def score(self, audio: AudioSegment) -> float: ...

class OnnxBinaryScorer:
    """ONNX adapter for a model export with an embedded waveform preprocessor."""
    def __init__(self, path: str, input_name: str | None = None, output_is_logit: bool = True) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc: raise RuntimeError("install voice-clone-detection[tier2] for ONNX Tier 2 inference") from exc
        self.np = np; self.output_is_logit = output_is_logit
        self.session = ort.InferenceSession(path, providers=ort.get_available_providers())
        self.input_name = input_name or self.session.get_inputs()[0].name

    def score(self, audio: AudioSegment) -> float:
        values = self.np.frombuffer(audio.samples, dtype="<i2").astype(self.np.float32)[None, :] / 32768.0
        value = float(self.session.run(None, {self.input_name: values})[0].reshape(-1)[0])
        return bounded(sigmoid(value) if self.output_is_logit else value)

def inspect_quality(audio: AudioSegment) -> AudioQuality:
    values = struct.unpack(f"<{len(audio.samples)//2}h", audio.samples)
    frame = max(1, int(audio.sample_rate*.02)); energies = [math.sqrt(sum(x*x for x in values[i:i+frame])/max(1,len(values[i:i+frame]))) for i in range(0,len(values),frame)]
    mean = sum(energies)/max(1,len(energies)); pauses = sum(x < 350 for x in energies)/max(1,len(energies))
    clipped = sum(abs(v) > 31_000 for v in values)/max(1,len(values)); duration_ok = .75 <= audio.duration_s <= 3.5
    energy_quality = min(mean / 1500, 1)
    score = bounded((.70*(1-min(clipped/.03,1)) + .30*(1 if duration_ok else .3)) * energy_quality)
    return AudioQuality(round(score,4), duration_ok, mean < 250, round(clipped,4), round(pauses,4))

def prosody_signal(audio: AudioSegment, quality: AudioQuality) -> dict[str, float]:
    """Independent behavioral evidence: pause rhythm, energy microvariation and cadence stability."""
    values = struct.unpack(f"<{len(audio.samples)//2}h", audio.samples); frame = max(1, int(audio.sample_rate*.02))
    energies = [math.sqrt(sum(x*x for x in values[i:i+frame])/max(1,len(values[i:i+frame]))) for i in range(0,len(values),frame)]
    energy_cv = statistics.pstdev(energies)/(statistics.fmean(energies)+1) if len(energies)>1 else 0
    rhythm = bounded(1-min(energy_cv/.75,1)); pause_anomaly = bounded(abs(quality.pause_ratio-.12)/.25)
    return {"prosody_score": round(.58*rhythm+.42*pause_anomaly,4), "pause_ratio": quality.pause_ratio, "rhythm_anomaly": round(rhythm,4)}

class Tier2ProductionEnsemble:
    """Four heterogeneous detectors + prosody, quality gate, calibration and consensus fusion."""
    def __init__(self, manifest_path: str | None = None) -> None:
        manifest_path = manifest_path or os.getenv("TIER2_MANIFEST")
        if not manifest_path:
            raise RuntimeError("TIER2_MANIFEST is required: Tier 2 has no heuristic or development fallback")
        self.manifest = json.loads(Path(manifest_path).read_text())
        self.scorers: dict[str, BinaryScorer] = {}; self.status: dict[str, str] = {}
        for name in MODEL_NAMES:
            config = self.manifest.get("models", {}).get(name)
            if not config or not Path(config.get("path", "")).is_file():
                raise RuntimeError(f"required Tier 2 model unavailable: {name}")
            self.scorers[name] = OnnxBinaryScorer(config["path"], config.get("input_name"), config.get("output_is_logit", True))
            self.status[name] = "onnx"
        self.weights = self.manifest.get("fusion", {}).get("weights", {"wav2vec2_xlsr":.24,"wavlm_large":.24,"rawnet3":.30,"aasist":.22,"prosody":.16})
        self.logit_weights = self.manifest.get("fusion", {}).get("logit_weights")
        self.temperature = float(self.manifest.get("calibration", {}).get("temperature", 1.0)); self.bias = float(self.manifest.get("calibration", {}).get("bias", 0.0))

    def score(self, audio: AudioSegment) -> EnsembleOutput:
        quality = inspect_quality(audio); prosody = prosody_signal(audio, quality)
        with ThreadPoolExecutor(max_workers=4) as pool: scores = dict(zip(MODEL_NAMES, pool.map(lambda key: self.scorers[key].score(audio), MODEL_NAMES)))
        features = {**scores, "prosody": prosody["prosody_score"]}
        if self.logit_weights:
            linear = self.bias + sum(self.logit_weights.get(key, 0.0) * math.log((value+1e-4)/(1-value+1e-4)) for key, value in features.items())
            calibrated = sigmoid(linear / max(self.temperature, .05))
        else:
            raw = sum(features[key]*self.weights.get(key,0) for key in features)
            raw /= max(sum(self.weights.get(key,0) for key in features), 1e-6)
            calibrated = sigmoid((math.log((raw+1e-4)/(1-raw+1e-4))+self.bias)/max(self.temperature,.05))
        disagreement = statistics.pstdev(scores.values()) if len(scores)>1 else 0
        # High disagreement/poor capture decreases confidence and pulls risk toward neutral.
        confidence = bounded(quality.score*(1-min(disagreement/.30,1)))
        score = .5 + (calibrated-.5)*(.55+.45*confidence)
        return EnsembleOutput(round(bounded(score),4), round(confidence,4), {key:round(value,4) for key,value in scores.items()}, prosody, round(disagreement,4), quality, self.status.copy())
