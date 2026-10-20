"""Turn free-form model output into a click point.

Models answer in many shapes: '(512, 300)', '{"point_2d": [512, 300]}',
'{"bbox_2d": [500, 290, 530, 310]}', or 'Click at x=512, y=300'. We accept a
box too and click its centre. Parsing failures return None and are counted as
their own error class, never silently scored as wrong-element.
"""

from __future__ import annotations

import json
import re

_NUM = r"-?\d+(?:\.\d+)?"
_BOX = re.compile(rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]")
_PAIR = re.compile(rf"[\(\[]\s*({_NUM})\s*,\s*({_NUM})\s*[\)\]]")
_XY = re.compile(rf"x\s*[=:]\s*({_NUM}).*?y\s*[=:]\s*({_NUM})", re.I | re.S)


def parse_point(text: str) -> tuple[float, float] | None:
    text = text.strip()
    # JSON objects with a known key
    for m in re.finditer(r"\{.*?\}", text, re.S):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            for key in ("point_2d", "point", "click", "coordinate"):
                v = obj.get(key)
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    return float(v[0]), float(v[1])
            v = obj.get("bbox_2d") or obj.get("bbox")
            if isinstance(v, (list, tuple)) and len(v) == 4:
                return _centre(*map(float, v))
    # A 4-number box anywhere
    if m := _BOX.search(text):
        return _centre(*map(float, m.groups()))
    # A coordinate pair
    if m := _PAIR.search(text) or _XY.search(text):
        return float(m.group(1)), float(m.group(2))
    return None


def _centre(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2
