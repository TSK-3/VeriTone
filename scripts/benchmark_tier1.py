"""Full-set Tier 1 benchmark: accuracy, EER and latency on the real dataset splits.

Evaluates the serving path exactly (``Tier1CheckpointScorer``: decode → resample →
1.5 s sliding windows → mean window probability), but defaults to a **center crop**
(one 1.5 s window per file) so the complete MLAAD set runs in minutes on CPU.
Use ``--mode sliding`` for the authoritative serving numbers on a sample.

```powershell
$env:PYTHONPATH = "src"
python scripts/benchmark_tier1.py --checkpoint checkpoints/tier1_mlaad.pt
python scripts/benchmark_tier1.py --mode sliding --limit 200   # spot check
```
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from voice_detection.audio import decode_wav  # noqa: E402
from voice_detection.tier1_adapter import Tier1CheckpointScorer  # noqa: E402


def eer(genuine: list[float], spoof: list[float]) -> float:
    """Equal-error-rate from two score distributions (spoof→high scores)."""
    thresholds = sorted({0.0, 1.0, *genuine, *spoof})
    best = 1.0
    for threshold in thresholds:
        fpr = sum(score >= threshold for score in genuine) / max(1, len(genuine))
        fnr = sum(score < threshold for score in spoof) / max(1, len(spoof))
        best = min(best, max(fpr, fnr))
    return best


def collect_scores(scorer: Tier1CheckpointScorer, folder: Path, limit: int, mode: str) -> tuple[list[float], list[float]]:
    """Score every WAV in ``folder``; returns (scores, per-file latencies in ms)."""
    files = sorted(folder.glob("*.wav"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no WAV files under {folder}")
    scores, latencies = [], []
    for index, path in enumerate(files):
        audio = decode_wav(path.read_bytes())
        if mode == "center" and audio.sample_rate != 0:
            audio = _center_crop(audio, 1.5)
        started = time.perf_counter()
        score = scorer.score(audio)
        latencies.append((time.perf_counter() - started) * 1000)
        scores.append(score)
        if (index + 1) % 500 == 0:
            print(f"  {index + 1}/{len(files)} scored…", flush=True)
    return scores, latencies


def _center_crop(audio, window_s: float):
    """One fixed 1.5 s window from the middle of the clip (the serving window)."""
    from voice_detection.audio import pack_pcm16, resample_linear, unpack_pcm16

    if audio.sample_rate != 16_000:
        values = resample_linear(unpack_pcm16(audio.samples), audio.sample_rate, 16_000)
        audio = type(audio)(pack_pcm16(values), 16_000, len(values) / 16_000)
    values = unpack_pcm16(audio.samples)
    window = int(window_s * 16_000)
    if len(values) <= window:
        return audio
    start = (len(values) - window) // 2
    crop = values[start:start + window]
    return type(audio)(pack_pcm16(crop), 16_000, window_s)


def evaluate(scorer: Tier1CheckpointScorer, name: str, genuine_dir: Path, spoof_dir: Path, limit: int, mode: str) -> None:
    print(f"\n== {name} (genuine: {genuine_dir}, spoof: {spoof_dir}, mode={mode}) ==", flush=True)
    genuine, genuine_lat = collect_scores(scorer, genuine_dir, limit, mode)
    spoof, spoof_lat = collect_scores(scorer, spoof_dir, limit, mode)
    correct = sum(score < 0.5 for score in genuine) + sum(score >= 0.5 for score in spoof)
    total = len(genuine) + len(spoof)
    latencies = genuine_lat + spoof_lat
    latencies.sort()
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    print(f"| files | accuracy@0.5 | EER | mean T1 latency | p95 |")
    print(f"|---|---|---|---|---|")
    print(f"| {total} ({len(genuine)} gen / {len(spoof)} spf) | {correct / total:.2f} | {eer(genuine, spoof):.3f} "
          f"| {statistics.fmean(latencies):.1f} ms | {p95:.1f} ms |")
    genuine_mean = statistics.fmean(genuine)
    spoof_mean = statistics.fmean(spoof)
    print(f"score means: genuine {genuine_mean:.3f} · spoof {spoof_mean:.3f} "
          f"(separation {(spoof_mean - genuine_mean):+.3f})")


def main() -> int:
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(root / "checkpoints" / "tier1_mlaad.pt"))
    parser.add_argument("--genuine-dir", type=Path, default=root / "data" / "genuine")
    parser.add_argument("--spoof-dir", type=Path, default=root / "data" / "spoof")
    parser.add_argument("--unseen-genuine-dir", type=Path, default=root / "data_unseen" / "genuine")
    parser.add_argument("--unseen-spoof-dir", type=Path, default=root / "data_unseen" / "spoof")
    parser.add_argument("--mode", choices=["center", "sliding"], default="center")
    parser.add_argument("--limit", type=int, default=0, help="cap per class (0 = full set)")
    args = parser.parse_args()
    scorer = Tier1CheckpointScorer(args.checkpoint)
    if not scorer.available:
        raise SystemExit(f"checkpoint not available: {args.checkpoint}")
    evaluate(scorer, "MLAAD train-sample split", args.genuine_dir, args.spoof_dir, args.limit, args.mode)
    evaluate(scorer, "held-out unseen split", args.unseen_genuine_dir, args.unseen_spoof_dir, args.limit, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
