#!/usr/bin/env python3
"""
Standardize HumanEval scores from results/raw_results/humaneval into CSV.

- Reads *.eval_results.json files directly under results/raw_results/humaneval/
- Extracts per-instance base/plus pass indicators as scores
- Writes to results/standardized_results/humaneval/{normalized_model_name}.csv

The CSV format is:
    score,metric_name
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict

from utils import (
    parse_model_name_from_filename,
    normalize_model_name,
    write_csv,
    resolve_paths,
    print_summary,
)

from typing import List, Tuple
import json


def _process_humaneval_file(filepath: Path) -> List[Tuple[float, str, str]]:
    """Dataset-specific: parse HumanEval eval_results.json into triples.
    Returns list of (id, score, metric_name) where id is task_id."""
    results: List[Tuple[float, str, str]] = []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Error reading {filepath}: {e}")
        return results

    eval_data = data.get("eval", {})
    for task_id, task_results in eval_data.items():
        for result in task_results:
            base_status = result.get("base_status", "")
            plus_status = result.get("plus_status", "")
            rid = str(result.get("task_id", task_id))
            results.append((rid, 1.0 if base_status == "pass" else 0.0, "base_pass"))
            results.append((rid, 1.0 if plus_status == "pass" else 0.0, "plus_pass"))
    return results


def standardize_humaneval(input_dir: Path, output_dir: Path, verbose: bool = True) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    output_task_dir = output_dir / "humaneval"
    output_task_dir.mkdir(parents=True, exist_ok=True)

    eval_files = list(input_dir.glob("*.eval_results.json"))
    if verbose:
        print(f"Found {len(eval_files)} eval_results.json files in humaneval/")

    for eval_file in sorted(eval_files):
        model_raw, reasoning_level = parse_model_name_from_filename(eval_file.name)
        model_name = normalize_model_name(model_raw, reasoning_level)

        metrics_data = _process_humaneval_file(eval_file)
        if not metrics_data:
            if verbose:
                print(f"  No metrics found in {eval_file.name}")
            continue

        output_csv = output_task_dir / f"{model_name}.csv"
        write_csv(output_csv, metrics_data)
        stats["humaneval"][model_name] = len(metrics_data)
        if verbose:
            print(f"  Wrote {output_csv.name} ({len(metrics_data)} entries)")

    return dict(stats)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Standardize HumanEval eval_scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/humaneval"),
        help="Directory containing HumanEval *.eval_results.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/standardized_results"),
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

    stats = standardize_humaneval(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir / 'humaneval'}")
    return 0


if __name__ == "__main__":
    main()
