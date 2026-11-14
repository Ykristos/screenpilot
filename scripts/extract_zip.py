"""Extract images from a zip without touching file permissions or timestamps.

The RunPod network volume rejects chmod/utime, so `unzip` prints a warning for
every file. Python's zipfile writes plain files and sets neither. It also skips
macOS junk (__MACOSX/, ._*) and can extract a random sample instead of everything.

    python scripts/extract_zip.py /root/dl/mobile_domain/rico_imgs.zip raw/os_atlas/rico
    python scripts/extract_zip.py /root/dl/web_domain/fineweb_3m_s11.zip raw/os_atlas/fineweb --sample 8000
"""

from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import PurePosixPath

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("dest")
    ap.add_argument("--sample", type=int, default=None, help="extract a random N images only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with zipfile.ZipFile(args.zip) as z:
        names = [
            n
            for n in z.namelist()
            if PurePosixPath(n).suffix.lower() in IMAGE_EXT
            and "__MACOSX" not in n
            and not PurePosixPath(n).name.startswith("._")
        ]
        if args.sample is not None and args.sample < len(names):
            names = random.Random(args.seed).sample(sorted(names), args.sample)
        for i, n in enumerate(names):
            z.extract(n, args.dest)
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(names)}", flush=True)
    print(f"extracted {len(names)} images to {args.dest}")


if __name__ == "__main__":
    main()
