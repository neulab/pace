#!/usr/bin/env python3
"""
Standardize VisualWebBench scores from results/raw_results/visualwebbench into CSVs.

- Input layout:
  results/raw_results/visualwebbench/
    {model_name}/
      web_caption.json
      webqa.json
      heading_ocr.json
      element_ocr.json
      element_ground.json
      action_prediction.json
      action_ground.json

- Raw JSON format (list):
  - index 0: aggregate score dict, e.g. {"score": "rouge_1: 35.49, ..."}
  - index 1+: per-instance dicts with "instance_score" key

- Output layout:
  results/standardized_results/visualwebbench/{normalized_model_name}.csv

CSV columns: id, score, metric_name

Tasks and their primary metric (all scores normalized to [0, 1]):
  web_caption      -> rouge_l   (divide by 100)
  webqa            -> f1        (divide by 100)
  heading_ocr      -> rouge_l   (divide by 100)
  element_ocr      -> rouge_l   (divide by 100)
  element_ground   -> correct   (already 0/1)
  action_prediction -> correct  (already 0/1)
  action_ground    -> correct   (already 0/1)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple  # Dict kept for stats type

from utils import write_csv, resolve_paths, print_summary, normalize_model_name

# ── Task configuration ────────────────────────────────────────────────────────
# (task_stem, metric_key, scale_factor)
TASKS: List[Tuple[str, str, float]] = [
    ("web_caption",       "rouge_l", 1 / 100),
    ("webqa",             "f1",      1 / 100),
    ("heading_ocr",       "rouge_l", 1 / 100),
    ("element_ocr",       "rouge_l", 1 / 100),
    ("element_ground",    "correct", 1.0),
    ("action_prediction", "correct", 1.0),
    ("action_ground",     "correct", 1.0),
]


def _process_task(json_path: Path, task: str, metric_key: str, scale: float
                  ) -> List[Tuple[str, float, str]]:
    """Extract per-instance (id, score, metric_name) from one task JSON."""
    with open(json_path) as f:
        data = json.load(f)

    results: List[Tuple[str, float, str]] = []
    # data[0] is the aggregate dict; skip it
    for idx, record in enumerate(data[1:]):
        instance_score = record.get("instance_score", {})
        raw = instance_score.get(metric_key)
        if raw is None:
            continue
        score = max(0.0, min(1.0, float(raw) * scale))
        instance_id = f"{task}_{idx}"
        results.append((instance_id, score, metric_key))
    return results


def main(
    input_dir: Path = Path("results/raw_results/visualwebbench"),
    output_dir: Path = Path("results/standardized_results"),
) -> None:
    input_dir, output_dir = resolve_paths(input_dir, output_dir)
    task_output_dir = output_dir / "visualwebbench"

    print(f"Input  : {input_dir}")
    print(f"Output : {task_output_dir}")
    print("=" * 60)

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return

    model_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    print(f"Found {len(model_dirs)} model directories\n")

    stats: Dict[str, Dict[str, int]] = defaultdict(dict)

    for model_dir in model_dirs:
        raw_name = model_dir.name

        try:
            model_normalized = normalize_model_name(raw_name)
        except ValueError:
            print(f"  [SKIP] {raw_name}: not in MODEL_NAME_LOOKUP")
            continue
        print(f"Processing {raw_name} -> {model_normalized}")

        all_results: List[Tuple[str, float, str]] = []

        for task, metric_key, scale in TASKS:
            json_path = model_dir / f"{task}.json"
            if not json_path.exists():
                print(f"  [WARN] Missing {task}.json")
                continue

            try:
                records = _process_task(json_path, task, metric_key, scale)
                all_results.extend(records)
                print(f"  {task:20s}: {len(records)} instances")
            except Exception as e:
                print(f"  [ERROR] {task}: {e}")

        if not all_results:
            print(f"  [WARN] No results extracted for {raw_name}")
            continue

        output_csv = task_output_dir / f"{model_normalized}.csv"
        write_csv(output_csv, all_results)
        stats["visualwebbench"][model_normalized] = len(all_results)
        print(f"  -> {output_csv.name} ({len(all_results)} total instances)\n")

    print_summary(stats)
    print(f"\nDone! Standardized data written to: {task_output_dir}")


if __name__ == "__main__":
    main()
