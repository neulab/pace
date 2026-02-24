#!/usr/bin/env python3
"""
Standardize ACP-Gen scores from results/raw_results/acp_gen into CSV.

- Input layout:
  results/raw_results/acp_gen/<model>/samples_<task>_<timestamp>.jsonl
  e.g., samples_acp_app_gen_2026-02-03T15-03-30.267970.jsonl

- Output layout (one CSV per model per task):
  results/standardized_results/acp_gen/<task>/{normalized_model_name}.csv

CSV columns:
  id,score,metric_name

We parse each JSONL line and extract per-instance metrics.
Primary metric is "score" (from the "metrics" list), converted to float.
The id is taken from the doc_id field if present; otherwise from line index.
"""
from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from utils import (
    normalize_model_name,
    write_csv,
    resolve_paths,
    print_summary,
)


def _parse_task_from_filename(filename: str) -> str | None:
    """Extract task name from samples_<task>_<timestamp>.jsonl filename."""
    m = re.match(r"^samples_(?P<task>.+?)_\d{4}-\d{2}-\d{2}T", filename)
    if m:
        return m.group("task")
    # Fallback: strip prefix/suffix and take middle
    name = filename
    if name.startswith("samples_"):
        name = name[len("samples_"):]
    if name.endswith(".jsonl"):
        name = name[:-len(".jsonl")]
    # Remove trailing timestamp-ish suffix if present (split on last underscore)
    parts = name.rsplit("_", 1)
    return parts[0] if parts else None


def _extract_metrics_from_line(data: Dict[str, Any], default_metric: str = "score") -> List[Tuple[float, str, str]]:
    """Extract per-instance metrics as (id, score, metric_name).

    - Uses 'metrics' list in the line if available; otherwise falls back to default_metric.
    - id is taken from 'doc_id' if present, else enumerated outside this function.
    """
    rid = str(data.get("doc_id", ""))
    metrics_list = data.get("metrics", [])
    results: List[Tuple[float, str, str]] = []

    def to_float(v: Any) -> float | None:
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        return None

    if isinstance(metrics_list, list) and metrics_list:
        for metric_name in metrics_list:
            if metric_name in data:
                val = to_float(data[metric_name])
                if val is not None:
                    results.append((rid, val, str(metric_name)))
    else:
        if default_metric in data:
            val = to_float(data[default_metric])
            if val is not None:
                results.append((rid, val, default_metric))

    return results


def standardize_acp_gen(input_root: Path, output_root: Path, verbose: bool = True) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    base_out = output_root / "acp_gen"
    base_out.mkdir(parents=True, exist_ok=True)

    # Model directories (e.g., azure__gpt-4o, gemini__gemini-3-pro-preview)
    model_dirs = [d for d in input_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if verbose:
        print(f"Found {len(model_dirs)} model directories")

    for model_dir in sorted(model_dirs):
        raw_model = model_dir.name
        model_name = normalize_model_name(raw_model)
        if verbose:
            print(f"\nProcessing model: {raw_model}")

        # All samples_*.jsonl files for this model
        jsonl_files = sorted(model_dir.glob("samples_*.jsonl"))
        if not jsonl_files and verbose:
            print("  No samples_*.jsonl files found")

        # Process each JSONL (task)
        for jsonl_file in jsonl_files:
            task = _parse_task_from_filename(jsonl_file.name)
            if not task:
                if verbose:
                    print(f"  Could not parse task from filename: {jsonl_file.name}")
                continue

            out_dir = base_out / task
            out_dir.mkdir(parents=True, exist_ok=True)

            # Parse lines
            aggregated: List[Tuple[float, str, str]] = []
            with open(jsonl_file, "r") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        if verbose:
                            print(f"  JSON error in {jsonl_file.name}:{idx+1}: {e}")
                        continue
                    entries = _extract_metrics_from_line(data)
                    # Ensure id present; if missing, use idx
                    fixed: List[Tuple[float, str, str]] = []
                    for score, metric_name, rid in entries:
                        rid2 = rid if rid != "" else str(idx)
                        fixed.append((score, metric_name, rid2))
                    aggregated.extend(fixed)

            if not aggregated:
                if verbose:
                    print(f"  No metrics extracted from {jsonl_file.name}")
                continue

            output_csv = out_dir / f"{model_name}.csv"
            write_csv(output_csv, aggregated)
            stats[f"acp_gen/{task}"][model_name] = len(aggregated)
            if verbose:
                print(f"  Wrote acp_gen/{task}/{output_csv.name} ({len(aggregated)} entries)")

    return dict(stats)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Standardize ACP-Gen eval scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/acp_gen"),
        help="Root directory of acp_gen model folders",
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

    stats = standardize_acp_gen(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir / 'acp_gen'}")
    return 0


if __name__ == "__main__":
    main()
