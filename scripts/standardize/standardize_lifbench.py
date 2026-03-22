#!/usr/bin/env python3
"""
Standardize LIFBench scores from results/raw_results/lifbench into CSVs grouped by task.

- Input layout:
  results/raw_results/lifbench/<model>/<task>.json
  where <model> is like: gpt-5.2, claude-opus-4-6, etc.
  and <task> is like: onedoc-qa, list-blur_offset_query_element, etc.

- Output layout (one CSV per model per task):
  results/standardized_results/lifbench/<task>/{normalized_model_name}.csv

CSV columns:
  id,score,metric_name

id is constructed as f"{ins_id}_{param_id}_{length}" and must be unique per
model; if a duplicate is encountered, a numeric suffix is appended to ensure
uniqueness. The metric recorded is total_score.
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
        # If not found in lookup, return the dirname as-is (title-cased)
        return dirname


def _extract_entries_from_json(
    json_file: Path, verbose: bool
) -> List[Tuple[str, float, str]]:
    """Extract (id, score, metric_name) triples from a LIFBench JSON file.

    Each entry in the JSON array should have:
    - ins_id: instance id
    - param_id: parameter id
    - length: length parameter
    - score_dict: dict with 'total_score' key
    """
    entries: List[Tuple[str, float, str]] = []
    seen_ids: set = set()

    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        if verbose:
            print(f"  Warning: Failed to read {json_file.name}: {e}")
        return entries

    if not isinstance(data, list):
        if verbose:
            print(f"  Warning: {json_file.name} is not a JSON array, skipping")
        return entries

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        ins_id = str(item.get("ins_id", idx))
        param_id = str(item.get("param_id", 0))
        length = str(item.get("length", 0))
        rid_base = f"{ins_id}_{param_id}_{length}"

        # Ensure unique id
        rid = rid_base
        if rid in seen_ids:
            suffix = 2
            while f"{rid_base}#{suffix}" in seen_ids:
                suffix += 1
            rid = f"{rid_base}#{suffix}"
        seen_ids.add(rid)

        # Extract total_score from score_dict
        score_dict = item.get("score_dict", {})
        if not isinstance(score_dict, dict):
            if verbose:
                print(f"  Warning: Invalid score_dict at index {idx} in {json_file.name}")
            continue

        total_score = score_dict.get("total_score")
        if total_score is None:
            if verbose:
                print(f"  Warning: Missing total_score at index {idx} in {json_file.name}")
            continue

        try:
            score = float(total_score)
        except (ValueError, TypeError):
            if verbose:
                print(f"  Warning: Invalid total_score '{total_score}' at index {idx} in {json_file.name}")
            continue

        entries.append((rid, score, "total_score"))

    return entries


def standardize_lifbench(
    input_root: Path, output_root: Path, verbose: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    Process all LIFBench results from input_root and write standardized CSVs.

    Input structure: input_root/<model>/<task>.json
    Output structure: output_root/lifbench/<task>/<model>.csv
    """
    stats: Dict[str, Dict[str, int]] = defaultdict(dict)

    # Collect all tasks and their model data
    # Structure: {task_name: {model_name: [(id, score, metric_name), ...]}}
    task_data: Dict[str, Dict[str, List[Tuple[str, float, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )

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

        # Find all task JSON files in this model directory
        json_files = sorted([f for f in model_dir.glob("*.json") if f.is_file()])

        for json_file in json_files:
            task_name = json_file.stem  # e.g., 'onedoc-qa', 'list-blur_offset_query_element'
            entries = _extract_entries_from_json(json_file, verbose)

            if entries:
                task_data[task_name][model_name].extend(entries)
                if verbose:
                    print(f"  {task_name}: {len(entries)} entries")

    # Write output CSVs organized by task
    for task_name in sorted(task_data.keys()):
        out_dir = output_root / "lifbench" / task_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for model_name, entries in sorted(task_data[task_name].items()):
            if not entries:
                continue

            output_csv = out_dir / f"{model_name}.csv"
            write_csv(output_csv, entries)
            stats[f"lifbench/{task_name}"][model_name] = len(entries)

            if verbose:
                print(f"  Wrote lifbench/{task_name}/{output_csv.name} ({len(entries)} entries)")

    return dict(stats)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Standardize LIFBench eval scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/lifbench"),
        help="Directory containing LIFBench model subdirectories with task JSON files",
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

    stats = standardize_lifbench(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir / 'lifbench'}")
    return 0


if __name__ == "__main__":
    main()
