#!/usr/bin/env python3
"""
Standardize Commit0 scores from results/raw_results/commit0 into CSV.

- Input layout (observed):
  results/raw_results/commit0/<model>/{report.json | output.report.json}
  Example:
    commit0/Gemini-3-Pro/report.json
    commit0/claude-opus-4-6/output.report.json

- Output layout (one CSV per top-level model):
  results/standardized_results/commit0/{normalized_model}.csv

CSV columns:
  id,score,metric_name

For each model, we collect resolved_ids (score=1.0) and unresolved_ids (score=0.0)
from any discovered report files. metric_name is "resolved". If the same id appears
in both sets across multiple files, resolved wins (score=1.0).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set

from utils import (
    normalize_model_name,
    write_csv,
    resolve_paths,
    print_summary,
)

# REPORT_BASENAMES = {"report.json", "output.report.json"}
REPORT_BASENAMES = {"output.report.json"}


def _collect_ids_from_report(report_path: Path) -> Tuple[Set[str], Set[str]]:
    """Return (resolved_ids, unresolved_ids) from a single report file.

    Falls back gracefully if keys are missing.
    """
    try:
        data = json.loads(report_path.read_text())
    except Exception:
        return set(), set()

    resolved = set(data.get("resolved_ids", []) or [])
    unresolved = set(data.get("unresolved_ids", []) or [])
    return resolved, unresolved


def standardize_commit0(input_root: Path, output_root: Path, verbose: bool = True) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}

    out_dir = output_root / "commit0"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Top-level model directories under commit0/
    model_dirs = [d for d in input_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if verbose:
        print(f"Found {len(model_dirs)} model directories")

    for model_dir in sorted(model_dirs):
        raw_model = model_dir.name
        model_name = normalize_model_name(raw_model)
        if verbose:
            print(f"\nProcessing model: {raw_model}")

        # Find report files recursively in this model dir
        report_files: List[Path] = []
        for rp in model_dir.rglob("*.json"):
            if rp.name in REPORT_BASENAMES:
                report_files.append(rp)

        if not report_files:
            if verbose:
                print("  No report.json or output.report.json found")
            continue

        resolved_all: Set[str] = set()
        unresolved_all: Set[str] = set()
        for rp in sorted(report_files):
            r, u = _collect_ids_from_report(rp)
            resolved_all.update(r)
            unresolved_all.update(u)

        # If an id is in both, resolved should win
        unresolved_all -= resolved_all

        rows: List[Tuple[float, str, str]] = []
        for rid in sorted(resolved_all):
            rows.append((rid, 1.0, "resolved"))
        for rid in sorted(unresolved_all):
            rows.append((rid, 0.0, "resolved"))

        if not rows:
            if verbose:
                print("  No entries to write (no resolved/unresolved ids)")
            continue

        output_csv = out_dir / f"{model_name}.csv"
        write_csv(output_csv, rows)
        stats.setdefault("commit0", {})[model_name] = len(rows)
        if verbose:
            print(f"  Wrote commit0/{output_csv.name} ({len(rows)} entries)")

    return stats


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Standardize Commit0 eval scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/commit0"),
        help="Root directory containing Commit0 model folders",
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

    stats = standardize_commit0(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir / 'commit0'}")
    return 0


if __name__ == "__main__":
    main()
