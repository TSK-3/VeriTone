"""Checkpoint-backed Tier 1 inference adapter."""

from __future__ import annotations

import os
import struct

from .audio import AudioSegment
from .tier1_cnn import SAMPLE_RATE, Tier1CausalCNN, torch


class Tier1CheckpointScorer:
    """Loads a training checkpoint when ``TIER1_CHECKPOINT`` is configured."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        path = checkpoint_path or os.getenv("TIER1_CHECKPOINT")
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
        if audio.sample_rate != SAMPLE_RATE:
            raise ValueError("Tier 1 CNN expects 16 kHz mono audio")
        values = struct.unpack(f"<{len(audio.samples) // 2}h", audio.samples)
        waveform = torch.tensor(values, dtype=torch.float32).unsqueeze(0) / 32768.0
        with torch.inference_mode():
            return float(torch.sigmoid(self.model(waveform)).item())
