#!/usr/bin/env python3
"""
Standardize LogiQA scores from results/raw_results/logiqa into CSV.

- Input layout:
  results/raw_results/logiqa/<model>/samples_logiqa_*.jsonl

- Output layout (one CSV per model):
  results/standardized_results/logiqa/{normalized_model_name}.csv

CSV columns:
  id,score,metric_name

We parse each JSONL line and extract per-instance metrics.
Primary metric is "exact_match" (from the "metrics" list or top-level key),
converted to float. The id is taken from doc_id if present; else the line index.
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


def _is_logiqa_file(filename: str) -> bool:
    return bool(re.match(r"^samples_logiqa_.*\.jsonl$", filename))


def _extract_id(data: Dict[str, Any], fallback_idx: int) -> str:
    if "doc_id" in data:
        try:
            return f'{data["doc_id"]}_{data['filter']}'
        except Exception:
            pass
    raise ValueError(f"no valid id")


def _to_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _extract_metrics_from_line(data: Dict[str, Any]) -> List[Tuple[float, str, str]]:
    """Extract per-instance (score, metric_name, id) triples from a LogiQA line."""
    rid = _extract_id(data, fallback_idx=-1)
    metrics_list = data.get("metrics", [])
    results: List[Tuple[float, str, str]] = []

    if isinstance(metrics_list, list) and metrics_list:
        for metric_name in metrics_list:
            if metric_name in data:
                val = _to_float(data[metric_name])
                if val is not None:
                    results.append((rid, val, str(metric_name)))
    else:
        # Fallback to exact_match if present
        if "exact_match" in data:
            val = _to_float(data["exact_match"])
            if val is not None:
                results.append((rid, val, "exact_match"))

    return results


def standardize_logiqa(input_root: Path, output_root: Path, verbose: bool = True) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    out_dir = output_root / "logiqa"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model directories (e.g., azure__gpt-4o, gemini__gemini-3-pro-preview)
    model_dirs = [d for d in input_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if verbose:
        print(f"Found {len(model_dirs)} model directories")

    for model_dir in sorted(model_dirs):
        raw_model = model_dir.name
        model_name = normalize_model_name(raw_model)
        if verbose:
            print(f"\nProcessing model: {raw_model}")

        # Find samples_logiqa_*.jsonl files
        jsonl_files = [p for p in model_dir.glob("samples_*.jsonl") if _is_logiqa_file(p.name)]
        if not jsonl_files:
            if verbose:
                print("  No samples_logiqa_*.jsonl files found")
            continue

        aggregated: List[Tuple[float, str, str]] = []
        for jsonl_file in sorted(jsonl_files):
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
                    # rid = _extract_id(data, fallback_idx=idx)
                    entries = _extract_metrics_from_line(data)
                    for rid, score, metric_name in entries:
                        aggregated.append((rid, score, metric_name))

        if not aggregated:
            if verbose:
                print("  No metrics extracted for LogiQA")
            continue

        output_csv = out_dir / f"{model_name}.csv"
        write_csv(output_csv, aggregated)
        stats["logiqa"][model_name] = len(aggregated)
        if verbose:
            print(f"  Wrote logiqa/{output_csv.name} ({len(aggregated)} entries)")

    return dict(stats)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Standardize LogiQA eval_scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/logiqa"),
        help="Root directory of logiqa model folders",
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

    stats = standardize_logiqa(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir / 'logiqa'}")
    return 0


if __name__ == "__main__":
    main()
