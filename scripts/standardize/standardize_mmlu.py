#!/usr/bin/env python3
"""
Standardize MMLU scores from results/raw_results/mmlu_* into CSV.

- Scans all tasks under results/raw_results whose directory name starts with 'mmlu_'
- For each model subdirectory, reads output.jsonl and extracts accuracy scores
- Writes to results/standardized_results/{task}/{normalized_model_name}.csv

The CSV format is:
    score,metric_name
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict

from utils import (
    parse_model_dir_name,
    normalize_model_name,
    write_csv,
    resolve_paths,
    print_summary,
)

from typing import List, Tuple
import json


def _process_mmlu_file(filepath: Path) -> List[Tuple[float, str, str]]:
    """Dataset-specific: parse MMLU output.jsonl into triples.
    Returns (id, score, metric_name) where id is taken from sample_id/idx/id/line.
    """
    results: List[Tuple[float, str, str]] = []
    try:
        with open(filepath, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    accuracy = data.get("accuracy", [])
                    if isinstance(accuracy, list) and accuracy:
                        score = float(accuracy[0])
                    elif isinstance(accuracy, (int, float)):
                        score = float(accuracy)
                    else:
                        continue
                    if "sample_id" in data:
                        rid = str(data["sample_id"])
                    elif "idx" in data:
                        rid = str(data["idx"])
                    elif "id" in data:
                        rid = str(data["id"])
                    else:
                        raise ValueError(f"No valid id retrieved: {data}")
                    results.append((rid, score, "accuracy"))
                except json.JSONDecodeError as e:
                    print(f"  Warning: JSON error at {filepath}:{line_num}: {e}")
                except Exception as e:
                    print(f"  Warning: Error at {filepath}:{line_num}: {e}")
    except OSError as e:
        print(f"  Error reading {filepath}: {e}")
    return results


def standardize_mmlu(input_root: Path, output_root: Path, verbose: bool = True) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Find all mmlu_* task directories
    task_dirs = [d for d in input_root.iterdir() if d.is_dir() and d.name.startswith("mmlu_")]
    if verbose:
        print(f"Found {len(task_dirs)} MMLU task directories")

    for task_dir in sorted(task_dirs):
        task_name = task_dir.name
        output_task_dir = output_root / task_name
        output_task_dir.mkdir(parents=True, exist_ok=True)

        # Model subdirectories
        model_dirs = [d for d in task_dir.iterdir() if d.is_dir()]
        if verbose:
            print(f"\nProcessing task: {task_name} ({len(model_dirs)} models)")

        for model_dir in sorted(model_dirs):
            base_model_raw, reasoning_level = parse_model_dir_name(model_dir.name)
            model_name = normalize_model_name(base_model_raw, reasoning_level)

            output_jsonl = model_dir / "output.jsonl"
            if not output_jsonl.exists():
                if verbose:
                    print(f"  Skipping {model_dir.name}: output.jsonl not found")
                continue

            metrics_data = _process_mmlu_file(output_jsonl)
            if not metrics_data:
                if verbose:
                    print(f"  No metrics found in {model_dir.name}/output.jsonl")
                continue

            output_csv = output_task_dir / f"{model_name}.csv"
            write_csv(output_csv, metrics_data)
            stats[task_name][model_name] = len(metrics_data)
            if verbose:
                print(f"  Wrote {task_name}/{output_csv.name} ({len(metrics_data)} entries)")

    return dict(stats)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Standardize MMLU eval scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/mmlu"),
        help="Root directory containing mmlu_* task directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/standardized_results/mmlu"),
        help="Output directory root for standardized CSV files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print verbose progress information",
    )

    args = parser.parse_args()
    input_dir, output_dir = resolve_paths(args.input_dir, args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    stats = standardize_mmlu(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir}")
    return 0


if __name__ == "__main__":
    main()
