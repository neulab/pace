#!/usr/bin/env python3
"""
Standardize BEIR NFCorpus reranking scores from results/raw_results/beir_nfcorpus into CSV.

BEIR is a reranking benchmark that evaluates information retrieval models. Unlike
classification or QA benchmarks, reranking benchmarks produce aggregate metrics
over the entire test set (e.g., NDCG@10, MAP@10, Recall@10).

- Input layout:
  results/raw_results/beir_nfcorpus/<model>/metrics.json
  where <model> is like: openai_gpt-5.2, gemini_gemini-2.5-pro, etc.

- Output layout (one CSV per model):
  results/standardized_results/beir_nfcorpus/{normalized_model_name}.csv

CSV columns:
  id,score,metric_name

For reranking benchmarks, each row represents an aggregate metric:
  - id: metric key (e.g., "NDCG@10", "MAP@10", "Recall@10", "P@10")
  - score: the metric value
  - metric_name: same as id (the metric being measured)

This captures all standard BEIR metrics (NDCG, MAP, Recall, Precision) at
various cutoff points (1, 5, 10, 20).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

from utils import (
    normalize_model_name,
    write_csv,
    resolve_paths,
    print_summary,
)

def _extract_metrics_from_json(
    metrics_file: Path, verbose: bool
) -> List[Tuple[str, float, str]]:
    """Extract (id, score, metric_name) triples from a BEIR metrics.json file.

    The metrics.json contains aggregate metrics in categories:
    - ndcg: NDCG@1, NDCG@5, NDCG@10, NDCG@20
    - map: MAP@1, MAP@5, MAP@10, MAP@20
    - recall: Recall@1, Recall@5, Recall@10, Recall@20
    - precision: P@1, P@5, P@10, P@20
    """
    entries: List[Tuple[str, float, str]] = []

    try:
        with open(metrics_file, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        if verbose:
            print(f"  Warning: Failed to read {metrics_file.name}: {e}")
        return entries

    if not isinstance(data, dict):
        if verbose:
            print(f"  Warning: {metrics_file.name} is not a JSON object, skipping")
        return entries

    # Extract metrics from each category
    metric_categories = ["ndcg", "map", "recall", "precision"]

    for category in metric_categories:
        category_data = data.get(category, {})
        if not isinstance(category_data, dict):
            continue

        for metric_key, value in category_data.items():
            if value is None:
                continue
            try:
                score = float(value)
            except (ValueError, TypeError):
                if verbose:
                    print(f"  Warning: Invalid score '{value}' for {metric_key}")
                continue

            # Use the metric key as both id and metric_name
            entries.append((metric_key, score, metric_key))

    return entries


def standardize_beir_nfcorpus(
    input_root: Path, output_root: Path, verbose: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    Process all BEIR NFCorpus results from input_root and write standardized CSVs.

    Input structure: input_root/<model>/metrics.json
    Output structure: output_root/beir_nfcorpus/<model>.csv
    """
    stats: Dict[str, Dict[str, int]] = defaultdict(dict)

    out_dir = output_root / "beir_nfcorpus"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all model directories
    model_dirs = sorted(
        [d for d in input_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

    if verbose:
        print(f"Found {len(model_dirs)} model directories")

    for model_dir in model_dirs:
        raw_model = model_dir.name
        model_name = normalize_model_name(raw_model)

        if verbose:
            print(f"\nProcessing model: {raw_model} -> {model_name}")

        # Look for metrics.json in the model directory
        metrics_file = model_dir / "metrics.json"
        if not metrics_file.exists():
            if verbose:
                print(f"  Warning: No metrics.json found, skipping")
            continue

        entries = _extract_metrics_from_json(metrics_file, verbose)

        if not entries:
            if verbose:
                print(f"  Warning: No metrics extracted, skipping")
            continue

        output_csv = out_dir / f"{model_name}.csv"
        write_csv(output_csv, entries)
        stats["beir_nfcorpus"][model_name] = len(entries)

        if verbose:
            print(f"  Wrote beir_nfcorpus/{output_csv.name} ({len(entries)} metrics)")

    return dict(stats)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Standardize BEIR NFCorpus reranking scores"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/beir_nfcorpus"),
        help="Directory containing BEIR NFCorpus model subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/standardized_results"),
        help="Output directory root for standardized CSV files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
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

    stats = standardize_beir_nfcorpus(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir / 'beir_nfcorpus'}")
    return 0


if __name__ == "__main__":
    main()
