import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screenpilot.metrics import (  # noqa: E402
    CORRECT,
    NEAR_MISS,
    NO_ANSWER,
    OUT_OF_IMAGE,
    WRONG_ELEMENT,
    classify,
    distance_to_box,
    point_in_box,
    summarize,
    wilson_ci,
)
from screenpilot.parse import parse_point  # noqa: E402

BOX = (100, 100, 200, 150)


def test_point_in_box_edges_inclusive():
    assert point_in_box((100, 100), BOX)
    assert point_in_box((200, 150), BOX)
    assert not point_in_box((201, 120), BOX)
    assert not point_in_box(None, BOX)


def test_distance_to_box():
    assert distance_to_box((150, 120), BOX) == 0
    assert distance_to_box((210, 120), BOX) == 10
    assert distance_to_box((203, 154), BOX) == 5


def test_classify():
    size = (1000, 800)
    assert classify(None, BOX, size) == NO_ANSWER
    assert classify((1200, 10), BOX, size) == OUT_OF_IMAGE
    assert classify((150, 120), BOX, size) == CORRECT
    assert classify((210, 120), BOX, size) == NEAR_MISS
    assert classify((600, 600), BOX, size) == WRONG_ELEMENT


def test_wilson_ci_brackets_estimate():
    lo, hi = wilson_ci(30, 100)
    assert lo < 0.30 < hi
    assert wilson_ci(0, 0) == (0.0, 0.0)
    assert wilson_ci(10, 10)[1] == 1.0


def test_parse_formats():
    assert parse_point('{"point_2d": [512, 300]}') == (512, 300)
    assert parse_point('```json\n{"bbox_2d": [500, 290, 530, 310]}\n```') == (515, 300)
    assert parse_point("I would click (12.5, 40)") == (12.5, 40)
    assert parse_point("x=10, y=20") == (10, 20)
    assert parse_point("[1, 2, 3, 4]") == (2, 3)
    assert parse_point("I cannot find it") is None


def test_summarize_groups():
    recs = [
        {"correct": True, "outcome": CORRECT, "platform": "web", "element_type": "icon"},
        {"correct": False, "outcome": WRONG_ELEMENT, "platform": "web", "element_type": "text"},
        {"correct": True, "outcome": CORRECT, "platform": "mobile", "element_type": "icon"},
    ]
    s = summarize(recs)
    assert s["overall"]["n"] == 3
    assert abs(s["overall"]["acc"] - 2 / 3) < 1e-9
    assert s["platform"]["web"]["acc"] == 0.5
    assert s["platform/element_type"]["mobile/icon"]["acc"] == 1.0
    assert s["outcomes"][CORRECT] == 2
