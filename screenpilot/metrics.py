"""Scoring: click accuracy, confidence intervals, per-group breakdowns, error taxonomy."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


def point_in_box(point: tuple[float, float] | None, bbox: tuple[float, float, float, float]) -> bool:
    if point is None:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def distance_to_box(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> float:
    """Pixel distance from the point to the nearest edge of the box (0 if inside)."""
    x, y = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval (rather than Wald to better handle small n's) for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# Error taxonomy, coarse on purpose. Refine it after reading 50 failures by hand.
NO_ANSWER = "no_parseable_answer"
OUT_OF_IMAGE = "point_outside_image"
NEAR_MISS = "near_miss"  # within `near_px` of the box: right element, sloppy point
WRONG_ELEMENT = "wrong_element"
CORRECT = "correct"


def classify(point, bbox, image_size, near_px: float = 20.0) -> str:
    if point is None:
        return NO_ANSWER
    w, h = image_size
    if not (0 <= point[0] <= w and 0 <= point[1] <= h):
        return OUT_OF_IMAGE
    if point_in_box(point, bbox):
        return CORRECT
    if distance_to_box(point, bbox) <= near_px:
        return NEAR_MISS
    return WRONG_ELEMENT


def summarize(records: Iterable[dict], group_keys=("platform", "element_type")) -> dict:
    """records have the structure: correct (bool), outcome (str), plus the group keys."""
    records = list(records)
    out = {"overall": _acc(records), "outcomes": _counts(records)}
    for key in group_keys:
        groups = defaultdict(list)
        for r in records:
            groups[r.get(key, "unknown")].append(r)
        out[key] = {g: _acc(rs) for g, rs in sorted(groups.items())}
    # platform x element_type is the table ScreenSpot papers report
    cross = defaultdict(list)
    for r in records:
        cross[f'{r.get("platform")}/{r.get("element_type")}'].append(r)
    out["platform/element_type"] = {g: _acc(rs) for g, rs in sorted(cross.items())}
    return out


def _acc(rs: list[dict]) -> dict:
    """Reports accuracy and relevant metrics."""
    n = len(rs)
    k = sum(bool(r["correct"]) for r in rs)
    lo, hi = wilson_ci(k, n)
    return {"n": n, "acc": k / n if n else 0.0, "ci95": [round(lo, 4), round(hi, 4)]}


def _counts(rs: list[dict]) -> dict:
    c = defaultdict(int)
    for r in rs:
        c[r["outcome"]] += 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))
