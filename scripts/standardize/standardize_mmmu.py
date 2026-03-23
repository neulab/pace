#!/usr/bin/env python3
"""
Standardize MMMU (Massive Multi-discipline Multimodal Understanding) scores.

MMMU is a multimodal benchmark covering 30 subjects across 6 disciplines.
Each question has a correct answer and the model's parsed prediction.

- Input layout:
  results/raw_results/mmmu/<model>/*_samples_mmmu_val.jsonl
  where <model> is like: azure_ai__gpt-5.2, anthropic__claude-opus-4-6, etc.

- Output layout (one CSV per model):
  results/standardized_results/mmmu/{normalized_model_name}.csv

CSV columns:
  id,score,metric_name

Each row represents one question:
  - id: question id (e.g., "validation_Accounting_1")
  - score: 1.0 if correct, 0.0 if incorrect
  - metric_name: "mmmu_acc"
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


def _get_model_name_for_dir(dirname: str) -> str:
    """Try to normalize a model directory name.

    Falls back to the dirname itself if not found in MODEL_NAME_LOOKUP.
    """
    try:
        return normalize_model_name(dirname)
    except ValueError:
        return dirname


def _extract_entries_from_jsonl(
    jsonl_file: Path, verbose: bool
) -> List[Tuple[str, float, str]]:
    """Extract (id, score, metric_name) triples from an MMMU JSONL file.

    Each line contains:
    - mmmu_acc: dict with 'id', 'answer', 'parsed_pred'
    - Score is 1.0 if answer is in parsed_pred, else 0.0
    """
    entries: List[Tuple[str, float, str]] = []

    try:
        with open(jsonl_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    if verbose:
                        print(f"  Warning: JSON error at line {line_num}: {e}")
                    continue

                mmmu_acc = data.get("mmmu_acc", {})
                if not isinstance(mmmu_acc, dict):
                    continue

                qid = mmmu_acc.get("id")
                if not qid:
                    # Fallback to doc_id or doc.id
                    qid = data.get("doc_id")
                    if qid is None:
                        doc = data.get("doc", {})
                        qid = doc.get("id") if isinstance(doc, dict) else None
                    if qid is None:
                        qid = f"line_{line_num}"

                answer = mmmu_acc.get("answer")
                parsed_pred = mmmu_acc.get("parsed_pred", [])

                # Compute correctness: 1.0 if answer in parsed_pred, else 0.0
                if isinstance(parsed_pred, list) and answer in parsed_pred:
                    score = 1.0
                else:
                    score = 0.0

                entries.append((str(qid), score, "mmmu_acc"))

    except IOError as e:
        if verbose:
            print(f"  Warning: Failed to read {jsonl_file.name}: {e}")

    return entries


def standardize_mmmu(
    input_root: Path, output_root: Path, verbose: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    Process all MMMU results from input_root and write standardized CSVs.

    Input structure: input_root/<model>/*_samples_mmmu_val.jsonl
    Output structure: output_root/mmmu/<model>.csv
    """
    stats: Dict[str, Dict[str, int]] = defaultdict(dict)

    out_dir = output_root / "mmmu"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all model directories
    model_dirs = sorted(
        [d for d in input_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

    if verbose:
        print(f"Found {len(model_dirs)} model directories")

    for model_dir in model_dirs:
        raw_model = model_dir.name
        model_name = _get_model_name_for_dir(raw_model)

        if verbose:
            print(f"\nProcessing model: {raw_model} -> {model_name}")

        # Find MMMU JSONL files
        jsonl_files = sorted(model_dir.glob("*_samples_mmmu_val.jsonl"))
        if not jsonl_files:
            if verbose:
                print("  Warning: No *_samples_mmmu_val.jsonl files found, skipping")
            continue

        all_entries: List[Tuple[str, float, str]] = []
        for jsonl_file in jsonl_files:
            entries = _extract_entries_from_jsonl(jsonl_file, verbose)
            all_entries.extend(entries)
            if verbose:
                correct = sum(1 for _, s, _ in entries if s == 1.0)
                print(f"  {jsonl_file.name}: {len(entries)} entries, {correct} correct ({100*correct/len(entries):.1f}%)")

        if not all_entries:
            if verbose:
                print("  Warning: No entries extracted, skipping")
            continue

        output_csv = out_dir / f"{model_name}.csv"
        write_csv(output_csv, all_entries)
        stats["mmmu"][model_name] = len(all_entries)

        if verbose:
            total_correct = sum(1 for _, s, _ in all_entries if s == 1.0)
            print(f"  Wrote mmmu/{output_csv.name} ({len(all_entries)} entries, accuracy: {100*total_correct/len(all_entries):.1f}%)")

    return dict(stats)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Standardize MMMU eval scores"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/mmmu"),
        help="Directory containing MMMU model subdirectories",
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

    stats = standardize_mmmu(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir / 'mmmu'}")
    return 0


if __name__ == "__main__":
    main()
