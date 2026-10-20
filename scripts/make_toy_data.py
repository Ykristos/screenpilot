"""Generate a tiny synthetic GUI dataset so the whole pipeline runs on a laptop.

Each image is a fake app window with labelled buttons; the instruction names
one of them. Useful for testing the harness today and, in week 4, for
debugging the GRPO loop on a model small enough to iterate in minutes.

    python scripts/make_toy_data.py --n 200 --out data/toy
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screenpilot.data import Example, write_jsonl  # noqa: E402

LABELS = ["Save", "Cancel", "Open", "Settings", "Help", "Delete", "Share", "Export", "Search", "Login"]
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]


def make_one(rng: random.Random, idx: int, img_dir: Path) -> Example:
    W, H = rng.choice([(1280, 800), (1024, 768), (390, 844)])
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 40], fill="#DDDDDD")  # title bar
    labels = rng.sample(LABELS, k=rng.randint(3, 6))
    boxes = []
    for label in labels:
        for _ in range(50):  # rejection-sample a non-overlapping spot
            bw, bh = rng.randint(80, 160), rng.randint(30, 50)
            x1, y1 = rng.randint(10, W - bw - 10), rng.randint(50, H - bh - 10)
            box = (x1, y1, x1 + bw, y1 + bh)
            if all(_disjoint(box, b) for b in boxes):
                boxes.append(box)
                d.rectangle(box, fill=rng.choice(COLORS))
                d.text((x1 + 10, y1 + bh // 2 - 6), label, fill="white")
                break
    target = rng.randrange(len(boxes))
    path = img_dir / f"{idx:05d}.png"
    img.save(path)
    return Example(
        id=f"toy-{idx}",
        image=f"images/{path.name}",
        instruction=f"Click the {labels[target]} button",
        bbox=boxes[target],
        platform="mobile" if W < H else "desktop",
        element_type="text",
    )


def _disjoint(a, b, pad=10):
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="data/toy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    examples = [make_one(rng, i, out / "images") for i in range(args.n)]
    write_jsonl(examples, out / "test.jsonl")
    print(f"wrote {len(examples)} examples to {out}/test.jsonl")


if __name__ == "__main__":
    main()
