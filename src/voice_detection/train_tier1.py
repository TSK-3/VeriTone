"""Train the Tier 1 causal CNN for synthetic-speech screening.

Usage (smoke test, no dataset needed)::

    pip install -e ".[ml]"
    python -m voice_detection.train_tier1 --synthetic 800 --epochs 8

Usage (real data — directories of 16-bit PCM WAV files)::

    python -m voice_detection.train_tier1 \\
        --genuine-dir data/genuine --spoof-dir data/spoof \\
        --epochs 20 --out checkpoints/tier1_cnn.pt

Each training example is a fixed 1.5 s window (24 000 samples @ 16 kHz) shaped
``[batch, samples]`` and optimized with ``BCEWithLogitsLoss``. Checkpoints are
saved as ``{"model_state_dict": ..., "config": ...}`` so that
``Tier1CheckpointScorer`` (``TIER1_CHECKPOINT``) loads them directly.

Only ``torch`` + stdlib are required — WAV I/O uses the ``wave`` module and
resampling is linear interpolation, so no ``torchaudio``/``librosa`` needed.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import random
import struct
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset, random_split
except ImportError:
    print("install the 'ml' extra first:  pip install -e \".[ml]\"", file=sys.stderr)
    raise SystemExit(2)

from .tier1_cnn import SAMPLE_RATE, WINDOW_SECONDS, Tier1CausalCNN

WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)
TARGET_PARAMS = 2_000_000


# ----------------------------------------------------------------------------
# WAV loading (stdlib only)
# ----------------------------------------------------------------------------

def load_wav_mono_16k(path: Path) -> Tensor:
    """Load a WAV file, mix to mono, resample to 16 kHz, return float tensor in [-1, 1]."""
    with wave.open(str(path), "rb") as wav:
        channels, width, rate, frames = (
            wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes(),
        )
        if width != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV supported")
        raw = wav.readframes(frames)
    n = len(raw) // 2
    values = struct.unpack(f"<{n}h", raw)
    wave_t = torch.tensor(values, dtype=torch.float32) / 32768.0
    if channels == 2:
        wave_t = (wave_t[0::2] + wave_t[1::2]) / 2
    if rate != SAMPLE_RATE:
        duration = wave_t.numel() / rate
        target_len = max(1, int(round(duration * SAMPLE_RATE)))
        wave_t = torch.nn.functional.interpolate(
            wave_t.view(1, 1, -1), size=target_len, mode="linear", align_corners=False
        ).view(-1)
    return wave_t.clamp(-1, 1)


def collect_wavs(directory: Path) -> list[Path]:
    return sorted([p for p in directory.rglob("*") if p.suffix.lower() == ".wav"])


# ----------------------------------------------------------------------------
# Synthetic fallback data (lets you train before ASVspoof is downloaded)
# ----------------------------------------------------------------------------

def _synthetic_waveform(genuine: bool, rng: random.Random) -> Tensor:
    """Generate a 1.5 s clip with a learnable genuine-vs-synthetic gap.

    Genuine: pitch + amplitude vibrato, pause/breath gaps, background noise.
    Synthetic: overly uniform harmonic stack, flat energy, no pauses —
    the same artifact family the heuristic scorers key on.
    """
    n = WINDOW_SAMPLES
    t = torch.arange(n, dtype=torch.float32) / SAMPLE_RATE
    f0 = rng.uniform(100, 220)
    out = torch.zeros(n)
    if genuine:
        vibrato = 1 + 0.02 * torch.sin(2 * math.pi * torch.tensor(5.5) * t + rng.uniform(0, 6.28))
        freq = f0 * vibrato
        phase = torch.cumsum(2 * math.pi * freq / SAMPLE_RATE, dim=0)
        for harmonic, amp in ((1, 1.0), (2, 0.35), (3, 0.15)):
            out = out + amp * torch.sin(phase * harmonic + rng.uniform(0, 6.28))
        envelope = 0.6 + 0.4 * torch.sin(2 * math.pi * torch.tensor(2.2) * t + rng.uniform(0, 6.28))
        out = out * envelope
        # Breath pauses: 1–2 silent gaps.
        for _ in range(rng.randint(1, 2)):
            start = rng.randint(0, n - 2000)
            out[start:start + rng.randint(800, 2400)] *= 0.05
        out = out + torch.randn(n) * 0.03  # room noise
    else:
        phase = 2 * math.pi * f0 * t
        for harmonic, amp in ((1, 1.0), (2, 0.4), (3, 0.2)):
            out = out + amp * torch.sin(phase * harmonic)
        out = out * rng.uniform(0.75, 0.85)  # flat energy, no pauses
        out = out + torch.randn(n) * 0.004  # too-clean background
    peak = out.abs().max().clamp_min(1e-4)
    return (out / peak * rng.uniform(0.5, 0.9)).clamp(-1, 1)


@dataclass
class ClipDataset(Dataset):
    waveforms: list[Tensor]
    labels: list[float]
    augment: bool = False
    crops: int = 1  # windows sampled per clip per epoch (long clips, 1.5 s model)

    def __len__(self) -> int:
        return len(self.waveforms) * self.crops

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        real = index % len(self.waveforms)
        wave_t = fix_length(self.waveforms[real], self.augment)
        if self.augment:
            wave_t = augment_wave(wave_t)
        return wave_t, torch.tensor(self.labels[real], dtype=torch.float32)


@dataclass
class FileClipDataset(Dataset):
    """Lazy path-based dataset: loads WAVs from disk per access (worker-safe)."""

    paths: list[str]
    labels: list[float]
    augment: bool = False
    crops: int = 1

    def __len__(self) -> int:
        return len(self.paths) * self.crops

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        real = index % len(self.paths)
        wave_t = fix_length(load_wav_mono_16k(Path(self.paths[real])), self.augment)
        if self.augment:
            wave_t = augment_wave(wave_t)
        return wave_t, torch.tensor(self.labels[real], dtype=torch.float32)


def fix_length(wave_t: Tensor, training: bool) -> Tensor:
    if wave_t.numel() == WINDOW_SAMPLES:
        return wave_t
    if wave_t.numel() > WINDOW_SAMPLES:
        if training:
            start = random.randint(0, wave_t.numel() - WINDOW_SAMPLES)
        else:  # deterministic center crop for eval
            start = (wave_t.numel() - WINDOW_SAMPLES) // 2
        return wave_t[start:start + WINDOW_SAMPLES]
    pad = torch.zeros(WINDOW_SAMPLES - wave_t.numel())
    return torch.cat([wave_t, pad])


def augment_wave(wave_t: Tensor) -> Tensor:
    # Speed perturbation (0.9x-1.1x): label-preserving tempo change.
    speed = random.uniform(0.9, 1.1)
    wave_t = torch.nn.functional.interpolate(
        wave_t.view(1, 1, -1), size=max(1, int(wave_t.numel() / speed)),
        mode="linear", align_corners=False).view(-1)
    wave_t = fix_length(wave_t, training=True)
    # Gain jitter.
    wave_t = wave_t * random.uniform(0.7, 1.15)
    # Background noise injection (phone/VoIP-realistic, PRD 5.3).
    snr_db = random.uniform(12, 30)
    noise = torch.randn_like(wave_t)
    sig_power = wave_t.pow(2).mean().clamp_min(1e-8)
    noise_power = noise.pow(2).mean().clamp_min(1e-8)
    wave_t = wave_t + noise * torch.sqrt(sig_power / noise_power / (10 ** (snr_db / 10)))
    # Polarity flip + small time shift (label-preserving).
    if random.random() < 0.5:
        wave_t = -wave_t
    shift = random.randint(-800, 800)
    wave_t = torch.roll(wave_t, shifts=shift)
    return wave_t.clamp(-1, 1)


def build_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    if args.synthetic:
        rng = random.Random(args.seed)
        n = args.synthetic
        waveforms: list[Tensor] = []
        labels: list[float] = []
        for i in range(n):
            genuine = i % 2 == 0
            waveforms.append(_synthetic_waveform(genuine, rng))
            labels.append(0.0 if genuine else 1.0)
        print(f"synthetic data: {n} clips ({n // 2} genuine / {n - n // 2} synthetic)")
        full: Dataset = ClipDataset(waveforms, labels, augment=False)
        get = lambda ds, i: (ds.waveforms[i], ds.labels[i])
        wrap = ClipDataset
    else:
        if not args.genuine_dir or not args.spoof_dir:
            raise SystemExit("provide --genuine-dir + --spoof-dir, or use --synthetic N")
        genuine = [str(p) for p in collect_wavs(Path(args.genuine_dir))]
        spoof = [str(p) for p in collect_wavs(Path(args.spoof_dir))]
        if not genuine or not spoof:
            raise SystemExit(f"no WAVs found: genuine={len(genuine)} spoof={len(spoof)}")
        print(f"real data: {len(genuine)} genuine / {len(spoof)} spoof (lazy load)")
        # Lazy: keep paths in memory (~12k clips would be ~6 GB as tensors).
        full = FileClipDataset(genuine + spoof,
                               [0.0] * len(genuine) + [1.0] * len(spoof), augment=False)
        get = lambda ds, i: (ds.paths[i], ds.labels[i])
        wrap = FileClipDataset
    val_count = max(1, int(len(full) * args.val_split))
    train_count = len(full) - val_count
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(full, [train_count, val_count], generator=generator)
    # Enable augmentation on the training subset only (wrap, don't mutate val).
    train_items = [get(full, i) for i in train_set.indices]
    val_items = [get(full, i) for i in val_set.indices]
    train_ds = wrap([t[0] for t in train_items], [t[1] for t in train_items],
                    augment=True, crops=args.crops)
    val_ds = wrap([t[0] for t in val_items], [t[1] for t in val_items])
    return train_ds, val_ds


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(model: Tier1CausalCNN, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    logits_all: list[Tensor] = []
    labels_all: list[Tensor] = []
    for waves, labels in loader:
        logits_all.append(model(waves.to(device)).cpu())
        labels_all.append(labels)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    acc = (preds == labels).float().mean().item()
    # EER estimate via threshold sweep (standard ASVspoof operating metric).
    best_eer, best_gap = 1.0, float("inf")
    for step in range(5, 96, 5):
        threshold = step / 100
        decision = (probs >= threshold).float()
        false_accept = ((decision == 1) & (labels == 0)).sum().item() / max(1, (labels == 0).sum().item())
        false_reject = ((decision == 0) & (labels == 1)).sum().item() / max(1, (labels == 1).sum().item())
        gap, eer = abs(false_accept - false_reject), (false_accept + false_reject) / 2
        if (gap, eer) < (best_gap, best_eer):
            best_gap, best_eer = gap, eer
    return {"loss": loss, "acc": acc, "eer": best_eer}


def _score_to_probs(logits: Tensor, labels: Tensor) -> dict[str, float]:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    probs = torch.sigmoid(logits)
    acc = ((probs >= 0.5).float() == labels).float().mean().item()
    best_eer, best_gap = 1.0, float("inf")
    for step in range(5, 96, 5):
        threshold = step / 100
        decision = (probs >= threshold).float()
        false_accept = ((decision == 1) & (labels == 0)).sum().item() / max(1, (labels == 0).sum().item())
        false_reject = ((decision == 0) & (labels == 1)).sum().item() / max(1, (labels == 1).sum().item())
        gap, eer = abs(false_accept - false_reject), (false_accept + false_reject) / 2
        if (gap, eer) < (best_gap, best_eer):
            best_gap, best_eer = gap, eer
    return {"loss": loss, "acc": acc, "eer": best_eer}


@torch.inference_mode()
def evaluate_sliding(model: Tier1CausalCNN, val_ds, device: torch.device,
                     batch_size: int = 32) -> dict[str, float]:
    """Inference-style eval: average window probabilities (matches the adapter).

    Slower than center-crop eval; use for final reporting, not per-epoch tracking.
    """
    from .tier1_cnn import HOP_SECONDS
    model.eval()
    hop = int(HOP_SECONDS * SAMPLE_RATE)
    all_probs: list[float] = []
    all_labels: list[float] = []
    items = val_ds.paths if isinstance(val_ds, FileClipDataset) else None
    for idx in range(len(val_ds.paths) if items is not None else len(val_ds.waveforms)):
        if items is not None:
            wave_t = load_wav_mono_16k(Path(val_ds.paths[idx]))
            label = val_ds.labels[idx]
        else:
            wave_t = val_ds.waveforms[idx]
            label = val_ds.labels[idx]
        if wave_t.numel() <= WINDOW_SAMPLES:
            windows = [fix_length(wave_t, training=False)]
        else:
            windows = [wave_t[s:s + WINDOW_SAMPLES]
                       if s + WINDOW_SAMPLES <= wave_t.numel()
                       else fix_length(wave_t[s:], training=False)
                       for s in range(0, wave_t.numel() - WINDOW_SAMPLES + 1, hop)]
        probs: list[float] = []
        for start in range(0, len(windows), batch_size):
            batch = torch.stack(windows[start:start + batch_size]).to(device)
            probs.extend(torch.sigmoid(model(batch)).cpu().tolist())
        all_probs.append(sum(probs) / len(probs))
        all_labels.append(label)
    return _score_to_probs(torch.tensor(all_probs), torch.tensor(all_labels))


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Tier 1 causal CNN")
    parser.add_argument("--genuine-dir", default=None, help="directory of genuine (bonafide) WAVs")
    parser.add_argument("--spoof-dir", default=None, help="directory of synthetic/spoof WAVs")
    parser.add_argument("--synthetic", type=int, default=0, help="generate N synthetic clips instead of loading data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--crops", type=int, default=2, help="windows sampled per clip per epoch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="checkpoints/tier1_cnn.pt")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = Tier1CausalCNN()
    params = model.parameter_count
    print(f"parameters: {params:,} (edge budget {TARGET_PARAMS:,})")
    if params > TARGET_PARAMS:
        print("WARNING: model exceeds the 2M-parameter edge budget", file=sys.stderr)

    train_set, val_set = build_datasets(args)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_acc, best_loss, best_state = -1.0, float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for waves, labels in train_loader:
            waves, labels = waves.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(waves), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * waves.size(0)
            count += waves.size(0)
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch:02d}/{args.epochs} "
              f"train_loss={total / max(1, count):.4f} "
              f"val_loss={metrics['loss']:.4f} val_acc={metrics['acc']:.3f} val_eer~{metrics['eer']:.3f}")
        # Select on val accuracy (the API's operating point at threshold 0.5);
        # tie-break on loss. EER is reported for information only.
        if (metrics["acc"], -metrics["loss"]) > (best_acc, -best_loss):
            best_acc, best_loss = metrics["acc"], metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "config": {"sample_rate": SAMPLE_RATE, "window_s": WINDOW_SECONDS}}, str(out))
    print(f"saved checkpoint -> {out}")

    # Latency probe: single 1.5 s window on CPU (edge-budget sanity check).
    model.eval().cpu()
    probe = torch.randn(1, WINDOW_SAMPLES)
    with torch.inference_mode():
        for _ in range(5):
            model(probe)
        started = time.perf_counter()
        runs = 20
        for _ in range(runs):
            model(probe)
        ms = (time.perf_counter() - started) / runs * 1000
    print(f"cpu latency (1.5 s window, mean of {runs}): {ms:.1f} ms - "
          "re-benchmark p95 on target edge hardware before deployment")
    final = evaluate(model, val_loader, torch.device("cpu"))
    print(f"final center-crop: val_acc={final['acc']:.3f} val_eer~{final['eer']:.3f}")
    sliding = evaluate_sliding(model, val_set, torch.device("cpu"), args.batch_size)
    print(f"final sliding-window: val_acc={sliding['acc']:.3f} val_eer~{sliding['eer']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
