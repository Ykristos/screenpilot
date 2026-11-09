"""Drop training examples whose screenshot also appears in an eval set.

    python scripts/dedup.py --train data/train_pool/all.jsonl \
        --eval data/screenspot_v2/test.jsonl --out data/train_pool/dedup.jsonl

Compares perceptual hashes (pHash), which survive resizing and recompression,
so near-copies are caught too, not just identical files. Removed pairs are
written next to --out as removed.jsonl: open a few to confirm they really are
the same screen, and tune --max-distance if not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

# Setup packaging later.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screenpilot.data import load_jsonl, write_jsonl  # noqa: E402


def phash(path: str) -> int:
    with Image.open(path) as im:
        return int(str(imagehash.phash(im)), 16)


def hash_images(paths: set[str]) -> dict[str, int]:
    out = {}
    for i, p in enumerate(sorted(paths)):
        out[p] = phash(p)
        if (i + 1) % 2000 == 0:
            print(f"  hashed {i + 1}/{len(paths)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", action="append", required=True, help="repeat for each eval set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-distance", type=int, default=4, help="Hamming distance out of 64 bits (# of pixel diffs)")
    args = ap.parse_args()

    train = load_jsonl(args.train)
    eval_paths = {str(e.image_path(Path(f).parent)) for f in args.eval for e in load_jsonl(f)}
    print(f"hashing {len(eval_paths)} eval screenshots")
    eval_hashes = hash_images(eval_paths)
    eval_list = list(eval_hashes)
    # List of pHashes
    eval_arr = np.array([eval_hashes[p] for p in eval_list], dtype=np.uint64)

    train_root = Path(args.train).parent
    train_paths = {str(e.image_path(train_root)) for e in train}
    print(f"hashing {len(train_paths)} training screenshots")
    train_hashes = hash_images(train_paths)

    matches = {}  # train image -> (closest eval image, distance)
    for p, h in train_hashes.items():
        # np.unpackbits only works w/ 8-bit ints so rework tensor into appropriate shape.
        xor = (eval_arr ^ np.uint64(h)).view(np.uint8).reshape(-1, 8)
        dist = np.unpackbits(xor, axis=1).sum(axis=1)
        k = int(dist.argmin())
        if dist[k] <= args.max_distance:
            matches[p] = (eval_list[k], int(dist[k]))

    kept = [e for e in train if str(e.image_path(train_root)) not in matches]
    write_jsonl(kept, args.out)
    removed_path = Path(args.out).with_name("removed.jsonl")
    with open(removed_path, "w") as f:
        for p, (q, d) in sorted(matches.items()):
            f.write(json.dumps({"train_image": p, "eval_image": q, "distance": d}) + "\n")

    print(
        f"\n{len(matches)} of {len(train_paths)} training screenshots match an eval screenshot; "
        f"removed {len(train) - len(kept)} of {len(train)} examples"
    )
    print(f"kept {len(kept)} -> {args.out}; matches listed in {removed_path}")


if __name__ == "__main__":
    main()
