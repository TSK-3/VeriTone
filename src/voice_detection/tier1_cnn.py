"""Trainable, edge-oriented Tier 1 causal CNN for synthetic-speech screening.

The model consumes 16 kHz mono waveform windows and produces one synthetic-speech
logit per window. It is intentionally shallow (<2M parameters) and contains no
recurrent state: long-range call confidence is handled by ``RunningRiskAggregator``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError:  # lets API-only deployments run before ML extras are installed
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]


SAMPLE_RATE = 16_000
WINDOW_SECONDS = 1.5
HOP_SECONDS = 0.75  # 50% overlap


def iter_sliding_windows(samples: bytes, sample_rate: int, window_s: float = WINDOW_SECONDS, hop_s: float = HOP_SECONDS) -> Iterator[bytes]:
    """Yield fixed-size PCM16 windows, padding the final partial window with silence."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Tier 1 CNN expects 16 kHz mono audio; resample upstream")
    window_bytes, hop_bytes = int(window_s * sample_rate) * 2, int(hop_s * sample_rate) * 2
    for offset in range(0, len(samples), hop_bytes):
        chunk = samples[offset:offset + window_bytes]
        if not chunk:
            break
        yield chunk.ljust(window_bytes, b"\0")
        if offset + window_bytes >= len(samples):
            break


if torch is not None:
    class CausalConv1d(nn.Conv1d):
        """A convolution that only consumes the present and preceding frames."""

        def forward(self, x: Tensor) -> Tensor:
            left_padding = self.dilation[0] * (self.kernel_size[0] - 1)
            return super().forward(F.pad(x, (left_padding, 0)))


    class CausalMaxPool1d(nn.Module):
        def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
            super().__init__()
            self.kernel_size, self.stride = kernel_size, stride

        def forward(self, x: Tensor) -> Tensor:
            return F.max_pool1d(F.pad(x, (self.kernel_size - 1, 0), value=float("-inf")), self.kernel_size, self.stride)


    def mel_filterbank(sample_rate: int = SAMPLE_RATE, n_fft: int = 512, n_mels: int = 64) -> Tensor:
        """Create a dependency-free triangular mel filterbank."""
        def hz_to_mel(hz: float) -> float: return 2595 * math.log10(1 + hz / 700)
        def mel_to_hz(mel: float) -> float: return 700 * (10 ** (mel / 2595) - 1)
        mel_points = torch.linspace(hz_to_mel(20), hz_to_mel(sample_rate / 2), n_mels + 2)
        bins = torch.floor((n_fft + 1) * torch.tensor([mel_to_hz(m.item()) for m in mel_points]) / sample_rate).long()
        bank = torch.zeros(n_mels, n_fft // 2 + 1)
        for index in range(n_mels):
            left, center, right = bins[index:index + 3]
            if center > left:
                bank[index, left:center] = torch.arange(left, center).sub(left).float() / (center - left)
            if right > center:
                bank[index, center:right] = torch.arange(right - center, 0, -1).float() / (right - center)
        return bank


    class LogMelFrontend(nn.Module):
        def __init__(self, sample_rate: int = SAMPLE_RATE, n_fft: int = 512, hop_length: int = 160, n_mels: int = 64) -> None:
            super().__init__()
            self.n_fft, self.hop_length = n_fft, hop_length
            self.register_buffer("window", torch.hann_window(n_fft))
            self.register_buffer("mel_bank", mel_filterbank(sample_rate, n_fft, n_mels))

        def forward(self, waveform: Tensor) -> Tensor:
            # center=False avoids looking into future audio frames during streaming.
            spectrum = torch.stft(waveform, self.n_fft, self.hop_length, window=self.window, center=False, return_complex=True).abs().pow(2)
            log_mel = torch.log(torch.clamp(torch.matmul(self.mel_bank, spectrum), min=1e-6))
            mean = log_mel.mean(dim=(1, 2), keepdim=True)
            std = log_mel.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
            return (log_mel - mean) / std


    class Tier1CausalCNN(nn.Module):
        """Conv(32,7) → Conv(64,5)+pool → dilated Conv(128,3)x2 → GAP → head."""

        def __init__(self, n_mels: int = 64, dropout: float = 0.3) -> None:
            super().__init__()
            self.frontend = LogMelFrontend(n_mels=n_mels)
            self.features = nn.Sequential(
                CausalConv1d(n_mels, 64, kernel_size=7), nn.BatchNorm1d(64), nn.ReLU(),
                CausalConv1d(64, 128, kernel_size=5), nn.BatchNorm1d(128), nn.ReLU(), CausalMaxPool1d(),
                CausalConv1d(128, 256, kernel_size=3, dilation=2), nn.BatchNorm1d(256), nn.ReLU(),
                CausalConv1d(256, 256, kernel_size=3, dilation=4), nn.BatchNorm1d(256), nn.ReLU(),
                CausalConv1d(256, 256, kernel_size=3, dilation=8), nn.BatchNorm1d(256), nn.ReLU(),
            )
            self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

        def forward(self, waveform: Tensor) -> Tensor:
            """Return logits shaped [batch]; apply BCEWithLogitsLoss during training."""
            if waveform.ndim != 2:
                raise ValueError("expected waveform tensor with shape [batch, samples]")
            return self.head(self.features(self.frontend(waveform))).squeeze(-1)

        @property
        def parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters())

else:
    Tier1CausalCNN = None  # type: ignore[misc,assignment]
