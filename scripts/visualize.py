"""Draw target boxes (green) and predicted clicks (red) for a run, so you can see failures.

    python scripts/visualize.py --run runs/Qwen2.5-VL-3B-Instruct-screenspot_v2 \
        --data data/screenspot_v2/test.jsonl --only wrong_element --n 30

Reading 30 to 50 of these by hand is how you build the error taxonomy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screenpilot.data import load_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--only", default=None, help="outcome to keep, e.g. wrong_element")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    data_path = Path(args.data)
    examples = {e.id: e for e in load_jsonl(data_path)}
    out = Path(args.run) / "viz"
    out.mkdir(exist_ok=True)
    count = 0
    for line in open(Path(args.run) / "predictions.jsonl"):
        r = json.loads(line)
        if args.only and r["outcome"] != args.only:
            continue
        ex = examples[r["id"]]
        img = Image.open(ex.image_path(data_path.parent)).convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle(ex.bbox, outline="lime", width=3)
        if r["point"]:
            x, y = r["point"]
            d.ellipse([x - 8, y - 8, x + 8, y + 8], outline="red", width=3)
        d.text((10, 10), f'{r["outcome"]}: {ex.instruction}'[:120], fill="red")
        img.save(out / f'{r["id"]}.png')
        count += 1
        if count >= args.n:
            break
    print(f"saved {count} images to {out}/")


if __name__ == "__main__":
    main()
