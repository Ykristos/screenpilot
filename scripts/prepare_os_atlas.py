"""Sample a training mix from downloaded OS-Atlas files into JSONL format.

    python scripts/prepare_os_atlas.py \
        --source web:raw/os_atlas/seeclick_web.json:raw/os_atlas/seeclick_web_imgs \
        --source mobile:raw/os_atlas/widget_captioning.json:raw/os_atlas/widget_imgs \
        --source desktop:raw/os_atlas/windows.json:raw/os_atlas/windows_imgs \
        --per-source 7000 --out data/train_pool/all.jsonl

Each --source is platform:annotation_json:image_dir. The output still needs
scripts/dedup.py (remove ScreenSpot (test) overlap) and scripts/split.py (train/val).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

# Setup packaging later.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screenpilot.data import iter_os_atlas, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="platform:json:image_dir")
    ap.add_argument("--per-source", type=int, default=7000)
    ap.add_argument("--max-per-image", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    examples = []
    for spec in args.source:
        platform, ann, img_dir = spec.split(":")
        source = Path(ann).stem
        got = list(
            islice(
                iter_os_atlas(ann, img_dir, platform, source, args.seed, args.max_per_image),
                args.per_source,
            )
        )
        if not got:
            sys.exit(f"{source}: 0 examples. Check that {img_dir} directly contains the images.")
        if len(got) < args.per_source:
            print(f"WARNING {source}: only {len(got)} of {args.per_source} requested")
        examples += got
        print(f"{source:<24} {platform:<8} {len(got)} examples")

    write_jsonl(examples, args.out)
    print(f"\nwrote {len(examples)} examples to {args.out}")
    print(Counter((e.platform, e.element_type) for e in examples))


if __name__ == "__main__":
    main()
