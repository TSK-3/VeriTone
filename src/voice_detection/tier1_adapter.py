"""Optional checkpoint-backed Tier 1 inference adapter."""

from __future__ import annotations

import os
import struct

from .audio import AudioSegment
from .tier1_cnn import SAMPLE_RATE, WINDOW_SECONDS, Tier1CausalCNN, iter_sliding_windows, torch


def _resample_mono(samples_16k_norm: "object", src_rate: int) -> "object":
    """Linear-interpolate mono waveform in [-1, 1] to 16 kHz using torch only."""
    assert torch is not None
    waveform = samples_16k_norm  # already a 1-D tensor
    if src_rate == SAMPLE_RATE:
        return waveform
    duration = waveform.numel() / src_rate
    target_len = max(1, int(round(duration * SAMPLE_RATE)))
    return torch.nn.functional.interpolate(
        waveform.view(1, 1, -1).float(), size=target_len, mode="linear", align_corners=False
    ).view(-1)


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
        waveform = _resample_mono(waveform, audio.sample_rate)
        # Match training: model sees fixed 1.5 s windows; average window
        # probabilities with 50% overlap for longer segments.
        window_bytes = int(WINDOW_SECONDS * SAMPLE_RATE) * 2
        pcm = (waveform.clamp(-1, 1) * 32767.0).to(torch.int16).cpu().numpy().tobytes()
        if len(pcm) <= window_bytes:
            windows = [pcm.ljust(window_bytes, b"\0")]
        else:
            windows = list(iter_sliding_windows(pcm, SAMPLE_RATE))
        probs: list[float] = []
        with torch.inference_mode():
            for window in windows:
                vals = struct.unpack(f"<{len(window) // 2}h", window)
                tensor = torch.tensor(vals, dtype=torch.float32).unsqueeze(0) / 32768.0
                probs.append(float(torch.sigmoid(self.model(tensor)).item()))
        return sum(probs) / len(probs)
