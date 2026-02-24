#!/usr/bin/env python3
"""
Standardize LIFBench scores from results/raw_results/lifbench into CSVs grouped by task.

- Input layout:
  results/raw_results/lifbench/<task>.csv
  where <task> is like: list-blur_offset_query_element, list-offset_query_id, ...

- Output layout (one CSV per model per task):
  results/standardized_results/lifbench/<task>/{normalized_model_name}.csv

CSV columns:
  id,score,metric_name

id is constructed as f"{ins_id}_{param_id}_{length}" and must be unique per
model; if a duplicate is encountered, a numeric suffix is appended to ensure
uniqueness. The metric recorded is total_score.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from utils import (
    normalize_model_name,
    write_csv,
    resolve_paths,
    print_summary,
)


def _standardize_task_csv(task_csv: Path, out_root: Path, verbose: bool) -> Dict[str, int]:
    """Process one lifbench task CSV into per-model standardized CSVs.

    Returns {model_name: num_entries_written} for summary stats.
    """
    task_name = task_csv.stem  # e.g., 'list-blur_offset_query_element'
    out_dir = out_root / "lifbench" / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group rows by model
    by_model: Dict[str, List[Tuple[float, str, str]]] = defaultdict(list)

    # Track IDs per model to ensure uniqueness
    seen_ids: Dict[str, set] = defaultdict(set)

    with open(task_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"model", "ins_id", "param_id", "length", "total_score"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{task_csv.name}: missing required columns: {sorted(missing)}")

        for row in reader:
            raw_model = row["model"].strip()
            model_name = normalize_model_name(raw_model)
            ins_id = str(row["ins_id"]).strip()
            param_id = str(row["param_id"]).strip()
            length = str(row["length"]).strip()
            rid_base = f"{ins_id}_{param_id}_{length}"

            # Ensure unique id per model
            rid = rid_base
            if rid in seen_ids[model_name]:
                suffix = 2
                while f"{rid_base}#{suffix}" in seen_ids[model_name]:
                    suffix += 1
                rid = f"{rid_base}#{suffix}"
            seen_ids[model_name].add(rid)

            try:
                score = float(row["total_score"])  # total_score is numeric
            except Exception:
                # If parsing fails, skip this entry
                if verbose:
                    print(f"  Warning: bad total_score in {task_csv.name}: {row.get('total_score')} (skipped)")
                continue

            by_model[model_name].append((rid, score, "total_score"))

    # Write outputs
    stats: Dict[str, int] = {}
    for model_name, triples in sorted(by_model.items()):
        if not triples:
            continue
        output_csv = out_dir / f"{model_name}.csv"
        write_csv(output_csv, triples)
        stats[model_name] = len(triples)
        if verbose:
            print(f"  Wrote lifbench/{task_name}/{output_csv.name} ({len(triples)} entries)")

    return stats


def standardize_lifbench(input_root: Path, output_root: Path, verbose: bool = True) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(dict)

    task_csvs = sorted([p for p in input_root.glob("*.csv") if p.is_file()])
    if verbose:
        print(f"Found {len(task_csvs)} lifbench task CSV files")

    for task_csv in task_csvs:
        if verbose:
            print(f"\nProcessing task: {task_csv.name}")
        per_model_counts = _standardize_task_csv(task_csv, output_root, verbose)
        stats[f"lifbench/{task_csv.stem}"] = per_model_counts

    return stats


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Standardize LIFBench eval scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/lifbench"),
        help="Directory containing LIFBench <task>.csv files",
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

    stats = standardize_lifbench(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir / 'lifbench'}")
    return 0


if __name__ == "__main__":
    main()
