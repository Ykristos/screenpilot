"""Convert a downloaded ScreenSpot-v2 release into our JSONL format.

    python scripts/prepare_screenspot.py \
        --annotations raw/ScreenSpot-v2 --images raw/ScreenSpot-v2/screenspotv2_image \
        --out data/screenspot_v2/test.jsonl

Prints counts per platform and element type, and checks every bbox lies inside
its image. Fix any warnings before trusting a number from this set.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screenpilot.data import iter_screenspot_v2, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    examples = list(iter_screenspot_v2(args.annotations, args.images))
    missing = [ex for ex in examples if not Path(ex.image).exists()]
    if missing:
        sys.exit(f"{len(missing)}/{len(examples)} images not found.")
    bad = 0
    for ex in examples:
        w, h = Image.open(ex.image).size
        x1, y1, x2, y2 = ex.bbox
        if not (0 <= x1 < x2 <= w + 1 and 0 <= y1 < y2 <= h + 1):
            bad += 1
            if bad <= 5:
                print(f"WARNING bbox outside image: {ex.id} {ex.bbox} image {w}x{h}")
    write_jsonl(examples, args.out)
    print(f"wrote {len(examples)} examples to {args.out} ({bad} bad boxes)")
    print(Counter((e.platform, e.element_type) for e in examples))


if __name__ == "__main__":
    main()
