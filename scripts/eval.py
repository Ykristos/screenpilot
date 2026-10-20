"""Run a model over a JSONL eval set, save every prediction, and print a report.

    python scripts/eval.py --data data/toy/test.jsonl --model center
    python scripts/eval.py --data data/screenspot_v2/test.jsonl --model qwen \
        --model-id Qwen/Qwen2.5-VL-3B-Instruct --limit 200

Outputs go to runs/<model>-<data>/: predictions.jsonl (one line per example,
including the raw model text, which you will read a lot) and summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screenpilot.data import load_jsonl  # noqa: E402
from screenpilot.metrics import CORRECT, classify, summarize  # noqa: E402
from screenpilot.models import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="center", help="center | random | qwen")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--coord-space", default="resized")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=None, help="evaluate the first N examples only")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    data_path = Path(args.data)
    examples = load_jsonl(data_path)[: args.limit]
    model = load_model(
        args.model, model_id=args.model_id, coord_space=args.coord_space, adapter=args.adapter
    )
    run_dir = Path(args.out) / f"{model.name}-{data_path.parent.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    records, t0 = [], time.time()
    with open(run_dir / "predictions.jsonl", "w") as f:
        for i, ex in enumerate(examples):
            image = Image.open(ex.image_path(data_path.parent)).convert("RGB")
            t = time.time()
            raw, point = model.predict(image, ex.instruction)
            outcome = classify(point, ex.bbox, image.size)
            rec = {
                "id": ex.id,
                "platform": ex.platform,
                "element_type": ex.element_type,
                "instruction": ex.instruction,
                "bbox": ex.bbox,
                "point": point,
                "raw": raw,
                "outcome": outcome,
                "correct": outcome == CORRECT,
                "latency_s": round(time.time() - t, 4),
            }
            f.write(json.dumps(rec) + "\n")
            records.append(rec)
            if (i + 1) % 50 == 0:
                acc = sum(r["correct"] for r in records) / len(records)
                print(f"[{i + 1}/{len(examples)}] running acc {acc:.3f}", flush=True)

    summary = summarize(records)
    summary["model"] = model.name
    summary["data"] = str(data_path)
    summary["mean_latency_s"] = round(sum(r["latency_s"] for r in records) / max(len(records), 1), 4)
    summary["wall_s"] = round(time.time() - t0, 1)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print_report(summary)
    print(f"\nSaved to {run_dir}/")


def print_report(s: dict) -> None:
    o = s["overall"]
    print(f"\n{s['model']} on {s['data']}")
    print(f"overall  acc {o['acc']:.3f}  95% CI [{o['ci95'][0]:.3f}, {o['ci95'][1]:.3f}]  n={o['n']}")
    print("\nby platform/element_type:")
    for g, v in s["platform/element_type"].items():
        print(f"  {g:<20} {v['acc']:.3f}  (n={v['n']})")
    print("\noutcomes:", s["outcomes"])


if __name__ == "__main__":
    main()
