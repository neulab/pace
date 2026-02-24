#!/usr/bin/env python3
"""
Standardize swebench_bash_only scores into CSV.

- Input layout:
  results/raw_results/swebench_bash_only/{model}.csv

- Output layout (one CSV per model):
  results/standardized_results/swebench_bash_only/{normalized_model}.csv

CSV columns:
  id,score,metric_name

Where:
- id = metadata.instance_id
- score = metadata.scores.resolved (as float 0.0/1.0)
- metric_name = resolved
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from utils import (
    normalize_model_name,
    write_csv,
    resolve_paths,
    print_summary,
)


ID_COL = "metadata.instance_id"
SCORE_COL = "metadata.scores.resolved"


def _standardize_file(csv_path: Path, out_dir: Path, verbose: bool) -> Tuple[str, int]:
    model_raw = csv_path.stem
    model_name = normalize_model_name(model_raw)

    rows: List[Tuple[float, str, str]] = []
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if ID_COL not in fieldnames or SCORE_COL not in fieldnames:
                if verbose:
                    print(f"  Skipping {csv_path.name}: missing required columns {ID_COL}/{SCORE_COL}")
                return model_name, 0
            for row in reader:
                rid = (row.get(ID_COL) or "").strip()
                if not rid:
                    continue
                raw_score = row.get(SCORE_COL)
                try:
                    score = float(raw_score) if raw_score is not None and raw_score != "" else 0.0
                except Exception:
                    # If malformed value, default to 0.0
                    score = 0.0
                rows.append((score, "resolved", rid))
    except Exception as e:
        if verbose:
            print(f"  Error reading {csv_path}: {e}")
        return model_name, 0

    if not rows:
        return model_name, 0

    output_csv = out_dir / f"{model_name}.csv"
    write_csv(output_csv, rows)
    if verbose:
        print(f"  Wrote swebench_bash_only/{output_csv.name} ({len(rows)} entries)")
    return model_name, len(rows)


def standardize_swebench_bash_only(input_root: Path, output_root: Path, verbose: bool = True) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}

    out_dir = output_root / "swebench_bash_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in input_root.glob("*.csv") if p.is_file()])
    if verbose:
        print(f"Found {len(files)} CSV files in swebench_bash_only/")

    task_key = "swebench_bash_only"
    stats[task_key] = {}

    for csv_path in files:
        _, count = _standardize_file(csv_path, out_dir, verbose)
        model_stem = csv_path.stem
        model_name = normalize_model_name(model_stem)
        if count > 0:
            stats[task_key][model_name] = count

    return stats


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Standardize swebench_bash_only eval_scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/swebench_bash_only"),
        help="Directory containing swebench_bash_only {model}.csv files",
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

    stats = standardize_swebench_bash_only(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir / 'swebench_bash_only'}")
    return 0


if __name__ == "__main__":
    main()
