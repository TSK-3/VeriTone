"""Download a stratified MLAAD-tiny slice for same-day Tier 1 training.

- Train spoof: 20 clips per TTS system x ~64 systems (fake/en) -> data/spoof
- Train genuine: 1280 clips (original/en) -> data/genuine
- Unseen test: 200 German spoof (fake/de) + 200 German genuine (original/de) -> data_unseen/

Run:  $env:HF_HUB_ENABLE_HF_TRANSFER=1; python scripts/download_mlaad_tiny.py
"""

from __future__ import annotations

import collections
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "mueller91/MLAAD-tiny"
ROOT = Path(__file__).parents[1]
SEED = 42
PER_SYSTEM_SPOOF = 20
N_GENUINE = 1280
N_UNSEEN_EACH = 200
WORKERS = 8
FULL = "--full" in sys.argv[1:]


def main() -> int:
    rng = random.Random(SEED)
    api = HfApi()
    print("listing repo files...")
    files = api.list_repo_files(REPO, repo_type="dataset")

    fake_en = [f for f in files if f.startswith("fake/en/") and f.endswith(".wav")]
    orig_en = sorted(f for f in files if f.startswith("original/en/") and f.endswith(".wav"))
    fake_de = [f for f in files if f.startswith("fake/de/") and f.endswith(".wav")]
    orig_de = sorted(f for f in files if f.startswith("original/de/") and f.endswith(".wav"))

    by_system: dict[str, list[str]] = collections.defaultdict(list)
    for f in fake_en:
        by_system[f.split("/")[2]].append(f)
    print(f"tts systems: {len(by_system)}, fake/en files: {len(fake_en)}, orig/en: {len(orig_en)}")

    spoof_pick: list[str] = []
    if FULL:
        spoof_pick = sorted(fake_en)
        genuine_pick = sorted(orig_en)
    else:
        for system, members in sorted(by_system.items()):
            members = sorted(members)
            step = max(1, len(members) // PER_SYSTEM_SPOOF)
            spoof_pick.extend(members[::step][:PER_SYSTEM_SPOOF])
        genuine_pick = rng.sample(orig_en, min(N_GENUINE, len(orig_en)))
    unseen_spoof = rng.sample(sorted(fake_de), min(N_UNSEEN_EACH, len(fake_de)))
    unseen_gen = rng.sample(orig_de, min(N_UNSEEN_EACH, len(orig_de)))
    print(f"train spoof: {len(spoof_pick)}, train genuine: {len(genuine_pick)}, "
          f"unseen: {len(unseen_spoof)} spoof + {len(unseen_gen)} genuine")

    jobs: list[tuple[str, Path]] = (
        [(f, ROOT / "data" / "spoof") for f in spoof_pick]
        + [(f, ROOT / "data" / "genuine") for f in genuine_pick]
        + [(f, ROOT / "data_unseen" / "spoof") for f in unseen_spoof]
        + [(f, ROOT / "data_unseen" / "genuine") for f in unseen_gen]
    )

    cache = ROOT / ".hf_cache"
    os.environ.setdefault("HF_HUB_CACHE", str(cache))

    done = 0

    def fetch(job: tuple[str, Path]) -> str:
        src, dest_dir = job
        dest_dir.mkdir(parents=True, exist_ok=True)
        local = hf_hub_download(REPO, src, repo_type="dataset")
        name = src.replace("/", "_")
        target = dest_dir / name
        if not target.exists():
            target.write_bytes(Path(local).read_bytes())
        return name

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for name in pool.map(fetch, jobs):
            done += 1
            if done % 200 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  ({name})", flush=True)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
