"""Split a training pool into train / val by screenshot, so no screen is on both sides.

    python scripts/split.py --data data/train_pool/dedup.jsonl --val-frac 0.05 --out-dir data/train

Writes train.jsonl and val.jsonl. Use val for learning rates and checkpoint
choice; ScreenSpot is only for final numbers.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

# Setup packaging later.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screenpilot.data import load_jsonl, write_jsonl  # noqa: E402


def split_by_image(examples, val_frac: float, seed: int):
    images = sorted({e.image for e in examples})
    random.Random(seed).shuffle(images)
    val_images = set(images[: max(1, round(len(images) * val_frac))])
    train = [e for e in examples if e.image not in val_images]
    val = [e for e in examples if e.image in val_images]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    train, val = split_by_image(load_jsonl(args.data), args.val_frac, args.seed)
    out = Path(args.out_dir)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(val, out / "val.jsonl")
    for name, part in (("train", train), ("val", val)):
        print(f"{name:<5} {len(part):>6} examples  {dict(Counter(e.platform for e in part))}")


if __name__ == "__main__":
    main()
