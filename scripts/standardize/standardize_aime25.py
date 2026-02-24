#!/usr/bin/env python3
"""
Standardize AIME25 scores from results/raw_results/aime25 into CSV.

- Input layout:
  results/raw_results/aime25/<model>/samples_aime25_<timestamp>.jsonl

- Output layout (one CSV per model):
  results/standardized_results/aime25/{normalized_model_name}.csv

CSV columns:
  id,score,metric_name

We parse each JSONL line and extract per-instance metrics.
Primary metric is "exact_match" (from the "metrics" list or top-level key), converted to float.
The id is taken from doc_id if present; else from doc.id; else from line index.
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


def _is_aime25_file(filename: str) -> bool:
    return bool(re.match(r"^samples_aime25_\d{4}-\d{2}-\d{2}T", filename))


def _extract_id(line: Dict[str, Any], fallback_idx: int) -> str:
    if "doc_id" in line:
        try:
            return str(line["doc_id"])
        except Exception:
            pass
    doc = line.get("doc") or {}
    if isinstance(doc, dict) and "id" in doc:
        try:
            return str(doc["id"])
        except Exception:
            pass
    return str(fallback_idx)


def _to_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _extract_metrics_from_line(line: Dict[str, Any]) -> List[Tuple[float, str, str]]:
    """Extract per-instance (score, metric_name, id) triples from an AIME25 line."""
    rid = _extract_id(line, fallback_idx=-1)
    metrics_list = line.get("metrics", [])
    results: List[Tuple[float, str, str]] = []

    if isinstance(metrics_list, list) and metrics_list:
        for metric_name in metrics_list:
            if metric_name in line:
                val = _to_float(line[metric_name])
                if val is not None:
                    results.append((rid, val, str(metric_name)))
    else:
        # Fallback to exact_match if present
        if "exact_match" in line:
            val = _to_float(line["exact_match"])
            if val is not None:
                results.append((rid, val, "exact_match"))
    return results


def standardize_aime25(input_root: Path, output_root: Path, verbose: bool = True) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    out_dir = output_root / "aime25"
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

        # Find samples_aime25_*.jsonl files
        jsonl_files = [p for p in model_dir.glob("samples_*.jsonl") if _is_aime25_file(p.name)]
        if not jsonl_files:
            if verbose:
                print("  No samples_aime25_*.jsonl files found")
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
                    # Fill rid fallback with idx if needed
                    # rid = _extract_id(data, fallback_idx=idx)
                    entries = _extract_metrics_from_line(data)
                    fixed: List[Tuple[float, str, str]] = []
                    for rid, score, metric_name in entries:
                        fixed.append((rid, score, metric_name))
                    aggregated.extend(fixed)

        if not aggregated:
            if verbose:
                print("  No metrics extracted for AIME25")
            continue

        output_csv = out_dir / f"{model_name}.csv"
        write_csv(output_csv, aggregated)
        stats["aime25"][model_name] = len(aggregated)
        if verbose:
            print(f"  Wrote aime25/{output_csv.name} ({len(aggregated)} entries)")

    return dict(stats)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Standardize AIME25 eval_scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/aime25"),
        help="Root directory of aime25 model folders",
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

    stats = standardize_aime25(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir / 'aime25'}")
    return 0


if __name__ == "__main__":
    main()
