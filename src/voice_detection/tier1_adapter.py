"""Optional checkpoint-backed Tier 1 inference adapter.

Loads the ``TIER1_CHECKPOINT`` model and mirrors training-time inference:
16 kHz mono waveform, fixed 1.5 s windows, 50 % overlap, mean window probability.
"""

from __future__ import annotations

import os
import struct

from .audio import AudioSegment
from .tier1_cnn import SAMPLE_RATE, WINDOW_SECONDS, Tier1CausalCNN, torch


class Tier1CheckpointScorer:
    """Loads a training checkpoint when ``TIER1_CHECKPOINT`` is configured."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        self.available = False
        self.model = None
        path = checkpoint_path or os.getenv("TIER1_CHECKPOINT")
        if not path:
            return
        if torch is None or Tier1CausalCNN is None:
            raise RuntimeError("install the 'ml' extra to use a Tier 1 CNN checkpoint")
        self.model = Tier1CausalCNN()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
        self.model.eval()
        self.available = True

    def score(self, audio: AudioSegment) -> float:
        if not self.available or self.model is None or torch is None:
            raise RuntimeError("Tier 1 CNN checkpoint is unavailable")
        values = struct.unpack(f"<{len(audio.samples) // 2}h", audio.samples)
        waveform = torch.tensor(values, dtype=torch.float32) / 32768.0
        if audio.sample_rate != SAMPLE_RATE:
            target_len = max(1, round(waveform.numel() * SAMPLE_RATE / audio.sample_rate))
            waveform = torch.nn.functional.interpolate(
                waveform.view(1, 1, -1), size=target_len, mode="linear", align_corners=False
            ).view(-1)
        windows = self._windows(waveform)
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model(torch.stack(windows)))
        return float(probabilities.mean().item())

    @staticmethod
    def _windows(waveform: "object") -> list:
        """Fixed 1.5 s windows with 50 % overlap; the last partial window is zero-padded."""
        window = int(WINDOW_SECONDS * SAMPLE_RATE)
        total = waveform.numel()
        if total <= window:
            return [torch.nn.functional.pad(waveform, (0, window - total))]
        windows = []
        for start in range(0, total, window // 2):
            chunk = waveform[start:start + window]
            if start + window >= total:
                windows.append(torch.nn.functional.pad(chunk, (0, window - chunk.numel())))
                break
            windows.append(chunk)
        return windows
