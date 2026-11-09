import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from screenpilot.data import Example, iter_os_atlas  # noqa: E402
from split import split_by_image  # noqa: E402


def _fake_os_atlas(tmp_path, records):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    Image.new("RGB", (1000, 500)).save(img_dir / "a.png")
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(records))
    return ann, img_dir


def test_os_atlas_flat_normalized_box_becomes_pixels(tmp_path):
    ann, img_dir = _fake_os_atlas(
        tmp_path, [{"img_filename": "a.png", "instruction": "save", "bbox": [0.1, 0.2, 0.3, 0.4]}]
    )
    (ex,) = iter_os_atlas(ann, img_dir, "web", "src")
    assert ex.bbox == (100.0, 100.0, 300.0, 200.0)
    assert ex.platform == "web" and Path(ex.image).is_absolute()


def test_os_atlas_nested_pixel_boxes_capped_and_filtered(tmp_path):
    elements = [{"instruction": f"e{i}", "bbox": [10, 10, 50, 50]} for i in range(5)]
    elements[1]["bbox"] = [10, 10, 5000, 50]  # out of image: dropped
    ann, img_dir = _fake_os_atlas(tmp_path, [{"img_filename": "a.png", "elements": elements}])
    got = list(iter_os_atlas(ann, img_dir, "web", "src", max_per_image=3))
    assert [e.instruction for e in got] == ["e0", "e2"]
    assert got[0].bbox == (10, 10, 50, 50)


def test_os_atlas_skips_missing_images(tmp_path):
    ann, img_dir = _fake_os_atlas(
        tmp_path, [{"img_filename": "missing.png", "instruction": "x", "bbox": [0, 0, 0.5, 0.5]}]
    )
    assert list(iter_os_atlas(ann, img_dir, "web", "src")) == []


def test_split_keeps_each_screenshot_on_one_side():
    exs = [Example(f"{i}-{j}", f"img{i}.png", "x", (0, 0, 1, 1)) for i in range(40) for j in range(3)]
    train, val = split_by_image(exs, 0.1, seed=0)
    assert len(train) + len(val) == len(exs)
    assert {e.image for e in train}.isdisjoint({e.image for e in val})
    assert len({e.image for e in val}) == 4
