#!/usr/bin/env python3
"""
Standardize IFEval scores from results/raw_results/ifeval into CSV.

- Input layout:
  results/raw_results/ifeval/<model>/samples_ifeval_<timestamp>.jsonl

- Output layout (one CSV per model per metric):
  results/standardized_results/ifeval/<metric_name>/{normalized_model_name}.csv
  where <metric_name> is one of:
    - prompt_level_strict_acc
    - inst_level_strict_acc
    - prompt_level_loose_acc
    - inst_level_loose_acc

CSV columns:
  id,score,metric_name

We parse each JSONL line and extract per-instance metrics for the four metrics
above. Booleans are converted to 0/1 floats; lists (e.g., per-instruction arrays)
are averaged to a float mean. The id is taken from doc_id if present; else from
doc.key; else the line index.
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

# Only consider these four metrics for standardized outputs
METRICS = (
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
)

def _is_ifeval_file(filename: str) -> bool:
    return bool(re.match(r"^samples_ifeval_\d{4}-\d{2}-\d{2}T", filename))


def _extract_id(data: Dict[str, Any], fallback_idx: int) -> str:
    if "doc_id" in data:
        try:
            return str(data["doc_id"])
        except Exception:
            pass
    doc = data.get("doc") or {}
    if isinstance(doc, dict) and "key" in doc:
        try:
            return str(doc["key"])
        except Exception:
            pass
    return str(fallback_idx)


def _to_mean_score(v: Any) -> float | None:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        # Average booleans/numerics in list
        vals: List[float] = []
        for x in v:
            if isinstance(x, bool):
                vals.append(1.0 if x else 0.0)
            elif isinstance(x, (int, float)):
                vals.append(float(x))
        if vals:
            return sum(vals) / len(vals)
        return None
    return None


def _extract_metrics_from_line(data: Dict[str, Any]) -> List[Tuple[float, str, str]]:
    rid = _extract_id(data, fallback_idx=-1)
    metrics_list = data.get("metrics", [])
    results: List[Tuple[float, str, str]] = []

    if isinstance(metrics_list, list) and metrics_list:
        for metric_name in metrics_list:
            if metric_name in METRICS and metric_name in data:
                score = _to_mean_score(data[metric_name])
                if score is not None:
                    results.append((rid, score, str(metric_name)))
    else:
        # Try common IFEval metrics directly
        for metric_name in METRICS:
            if metric_name in data:
                score = _to_mean_score(data[metric_name])
                if score is not None:
                    results.append((rid, score, metric_name))

    return results


def standardize_ifeval(input_root: Path, output_root: Path, verbose: bool = True) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    base_out = output_root / "ifeval"
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

        # Find samples_ifeval_*.jsonl files
        jsonl_files = [p for p in model_dir.glob("samples_*.jsonl") if _is_ifeval_file(p.name)]
        if not jsonl_files:
            if verbose:
                print("  No samples_ifeval_*.jsonl files found")
            continue

        # Aggregate per metric
        aggregated_by_metric: Dict[str, List[Tuple[float, str, str]]] = defaultdict(list)
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
                        aggregated_by_metric[metric_name].append((rid, score, metric_name))

        if not any(aggregated_by_metric.values()):
            if verbose:
                print("  No metrics extracted for IFEval")
            continue

        # Write one CSV per metric under ifeval/<metric>/
        for metric_name, rows in sorted(aggregated_by_metric.items()):
            if not rows:
                continue
            out_dir = base_out / metric_name
            out_dir.mkdir(parents=True, exist_ok=True)
            output_csv = out_dir / f"{model_name}.csv"
            write_csv(output_csv, rows)
            stats[f"ifeval/{metric_name}"][model_name] = len(rows)
            if verbose:
                print(f"  Wrote ifeval/{metric_name}/{output_csv.name} ({len(rows)} entries)")

    return dict(stats)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Standardize IFEval eval scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/ifeval"),
        help="Root directory of ifeval model folders",
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

    stats = standardize_ifeval(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir / 'ifeval'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
