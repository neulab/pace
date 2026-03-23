#!/usr/bin/env python3
"""
Standardize InfoBench scores from results/raw_results/infobench into CSV.

- Reads *_metrics.json files under results/raw_results/infobench/
- Extracts per-instance accuracy as scores (accuracy_percent / 100.0)
- Uses instance 'id' as the standardized id column
- Writes to results/standardized_results/infobench/{normalized_model_name}.csv

The CSV format is:
    id,score,metric_name
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import json

from utils import (
    normalize_model_name,
    write_csv,
    resolve_paths,
    print_summary,
)


def _parse_model_from_filename(filename: str) -> str:
    """Extract a raw model name from an InfoBench filename.

    Expected format: <model>_metrics.json
    Returns the <model> portion if it matches the pattern; otherwise, the basename
    without extension.
    """
    name = filename
    if name.endswith("_metrics.json"):
        return name[:-len("_metrics.json")]
    if name.endswith(".json"):
        return name[:-len(".json")]
    return name


def _process_infobench_file(filepath: Path) -> Tuple[str, List[Tuple[float, str, str]]]:
    """Parse an InfoBench metrics.json file.

    Returns a tuple of (raw_model_name, metrics) where metrics is a list of
    (id, score, metric_name). 'score' is accuracy in [0,1].
    """
    try:
        with open(filepath, "r") as f:
            data: Dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Error reading {filepath}: {e}")
        return _parse_model_from_filename(filepath.name), []

    raw_model = str(_parse_model_from_filename(filepath.name) or data.get("model"))

    instances = data.get("instances", [])
    results: List[Tuple[float, str, str]] = []

    for inst in instances:
        rid = str(inst.get("id", ""))
        if not rid:
            # Fallback: skip instances without id
            continue
        # Prefer accuracy_percent if present; otherwise infer from correct/total_questions
        acc_pct = inst.get("accuracy_percent")
        score: float
        if isinstance(acc_pct, (int, float)):
            score = float(acc_pct) / 100.0
        else:
            correct = inst.get("correct")
            total = inst.get("total_questions")
            if isinstance(correct, (int, float)) and isinstance(total, (int, float)) and total:
                score = float(correct) / float(total)
            else:
                # No metric to extract for this instance
                continue
        results.append((rid, score, "accuracy"))

    return raw_model, results


def standardize_infobench(input_dir: Path, output_dir: Path, verbose: bool = True) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    output_task_dir = output_dir / "infobench"
    output_task_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*_metrics.json"))
    if verbose:
        print(f"Found {len(files)} *_metrics.json files in infobench/")

    for fp in files:
        raw_model, metrics = _process_infobench_file(fp)
        model_name = normalize_model_name(raw_model)

        if not metrics:
            if verbose:
                print(f"  No metrics found in {fp.name}")
            continue

        output_csv = output_task_dir / f"{model_name}.csv"
        write_csv(output_csv, metrics)
        stats["infobench"][model_name] = len(metrics)
        if verbose:
            print(f"  Wrote {output_csv.name} ({len(metrics)} entries)")

    return dict(stats)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Standardize InfoBench eval scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/infobench"),
        help="Directory containing InfoBench *_metrics.json files",
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

    stats = standardize_infobench(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir / 'infobench'}")
    return 0


if __name__ == "__main__":
    main()
