"""Checkpoint-backed Tier 1 inference adapter."""

from __future__ import annotations

import os
import struct
from pathlib import Path

from .audio import AudioSegment
from .tier1_cnn import SAMPLE_RATE, WINDOW_SECONDS, Tier1CausalCNN, iter_sliding_windows, torch


def _default_checkpoint() -> str | None:
    candidates = [
        Path.cwd() / "checkpoints" / "tier1_mlaad.pt",
        Path(__file__).resolve().parents[3].parent / "sih" / "checkpoints" / "tier1_mlaad.pt",
    ]
    return next((str(path) for path in candidates if path.is_file()), None)


def _resample_mono(waveform: "object", src_rate: int) -> "object":
    if src_rate == SAMPLE_RATE:
        return waveform
    target_len = max(1, int(round(waveform.numel() * SAMPLE_RATE / src_rate)))
    return torch.nn.functional.interpolate(
        waveform.view(1, 1, -1).float(), size=target_len, mode="linear", align_corners=False
    ).view(-1)


class Tier1CheckpointScorer:
    """Loads a training checkpoint when ``TIER1_CHECKPOINT`` is configured."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        path = checkpoint_path or os.getenv("TIER1_CHECKPOINT") or _default_checkpoint()
        if not path:
            raise RuntimeError("TIER1_CHECKPOINT must point to a trained Tier 1 checkpoint")
        if torch is None or Tier1CausalCNN is None:
            raise RuntimeError("install the 'ml' extra to use a Tier 1 CNN checkpoint")
        self.model = Tier1CausalCNN()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
        self.model.eval()

    def score(self, audio: AudioSegment) -> float:
        if torch is None:
            raise RuntimeError("install the 'ml' extra to use a Tier 1 CNN checkpoint")
        values = struct.unpack(f"<{len(audio.samples) // 2}h", audio.samples)
        waveform = _resample_mono(torch.tensor(values, dtype=torch.float32) / 32768.0, audio.sample_rate)
        window_bytes = int(WINDOW_SECONDS * SAMPLE_RATE) * 2
        pcm = (waveform.clamp(-1, 1) * 32767.0).to(torch.int16).cpu().numpy().tobytes()
        windows = [pcm.ljust(window_bytes, b"\0")] if len(pcm) <= window_bytes else list(iter_sliding_windows(pcm, SAMPLE_RATE))
        probabilities = []
        with torch.inference_mode():
            for window in windows:
                samples = struct.unpack(f"<{len(window) // 2}h", window)
                tensor = torch.tensor(samples, dtype=torch.float32).unsqueeze(0) / 32768.0
                probabilities.append(float(torch.sigmoid(self.model(tensor)).item()))
        return sum(probabilities) / len(probabilities)
