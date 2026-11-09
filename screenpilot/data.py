"""Defines utility classes and functions for loading and working with data.

Every dataset is converted once into a simple JSONL format so the eval and
training code never care where the data came from. One line per example:

    {"id": "...", "image": "relative/or/absolute/path.png",
     "instruction": "open settings", "bbox": [x1, y1, x2, y2],
     "platform": "web", "element_type": "icon"}

bbox (bounding box) is in pixels of the original image, top-left / bottom-right corners.
"""

from __future__ import annotations

import json
import random

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Example:
    id: str
    image: str
    instruction: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    platform: str = "unknown"  # web / desktop / mobile
    element_type: str = "unknown"  # text / icon

    def image_path(self, root: Path) -> Path:
        p = Path(self.image)
        return p if p.is_absolute() else root / p


def load_jsonl(path: str | Path) -> list[Example]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                d["bbox"] = tuple(d["bbox"])
                out.append(Example(**d))
    return out


def write_jsonl(examples: list[Example], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")


# Converts ScreenSpot-v2 annotations into Examples.
# Input: screenspot_{web,desktop,mobile}_v2.json, each a list of
#   {"img_filename", "bbox": [x, y, w, h], "instruction", "data_type"}
# Output: bbox converted to [x1, y1, x2, y2]; platform taken from the filename.
# Schema is unverified; a KeyError means a field name differs.
def iter_screenspot_v2(annotation_dir: str | Path, image_dir: str | Path) -> Iterator[Example]:
    annotation_dir, image_dir = Path(annotation_dir), Path(image_dir)
    files = sorted(annotation_dir.glob("screenspot_*_v2.json"))
    if not files:
        raise FileNotFoundError(f"no screenspot_*_v2.json files in {annotation_dir}")
    for f in files:
        platform = f.stem.split("_")[1]  # web / desktop / mobile
        for i, r in enumerate(json.load(open(f))):
            x, y, w, h = r["bbox"]
            yield Example(
                id=f"{platform}-{i}",
                image=str((image_dir / r["img_filename"]).resolve()),
                instruction=r["instruction"],
                bbox=(x, y, x + w, y + h),
                platform=platform,
                element_type=r.get("data_type", "unknown"),
            )


# Converts one OS-Atlas annotation file (e.g. seeclick_web.json) into Examples.
# Input records come in two shapes, both handled:
#   flat:   {"img_filename", "instruction", "bbox", "data_type"?}
#   nested: {"img_filename", "elements": [{"instruction", "bbox", "data_type"?}, ...]}
# bbox is [left, top, right, bottom] as 0-1 fractions of the image size per the
# dataset card; values > 1 are treated as pixels already. Schema is unverified;
# a KeyError means a field name differs.
def iter_os_atlas(
    annotation_file: str | Path,
    image_dir: str | Path,
    platform: str,
    source: str,
    seed: int = 0,
    max_per_image: int = 3,
) -> Iterator[Example]:
    """Yields examples in a seeded random order, so islice(..., n) is a fair sample
    and only the images actually used get opened."""
    from PIL import Image  # header-only read below, so this stays cheap
 
    image_dir = Path(image_dir)
    records = list(enumerate(json.load(open(annotation_file))))
    random.Random(seed).shuffle(records)
    for i, r in records:
        path = (image_dir / r["img_filename"]).resolve()
        if not path.exists():
            continue
        with Image.open(path) as im:
            w, h = im.size
        # cap elements per screenshot so one busy page can't dominate the mix
        for j, el in enumerate(r.get("elements", [r])[:max_per_image]):
            x1, y1, x2, y2 = el["bbox"]
            if max(x1, y1, x2, y2) <= 1.0:
                x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                continue  # degenerate or out-of-image box
            yield Example(
                id=f"{source}-{i}-{j}",
                image=str(path),
                instruction=el["instruction"],
                bbox=(x1, y1, x2, y2),
                platform=platform,
                element_type=el.get("data_type", "unknown"),
            )