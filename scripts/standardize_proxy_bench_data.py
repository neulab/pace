#!/usr/bin/env python3
"""
Standardize proxy_bench_data into a consistent format.

This script:
1. Reads JSONL files from data/proxy_bench_data/<task>/<model>/
2. Extracts per-line metric scores
3. Outputs CSV files in standardized_data/<task-subtask>/<normalized_model_name>.csv

The CSV format is:
    score,metric_name

Model names are normalized to match the naming in results/swebench/.
"""

import os
import json
import csv
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# =============================================================================
# MODEL NAME LOOKUP TABLE
# Maps proxy_bench_data model directory names -> normalized names (swebench style)
# =============================================================================

MODEL_NAME_LOOKUP = {
    # Azure/OpenAI models
    "azure__gpt-4o": "GPT-4o",
    "azure__gpt-5": "GPT-5",
    "azure__gpt-oss-120b": "GPT-OSS-120B",
    "azure__o3": "o3",
    "azure__o4-mini": "o4-mini",
    "azure__Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama-4-Maverick-17B-128E-Instruct-FP8",

    # Gemini models
    "gemini__gemini-2.0-flash": "Gemini-2.0-Flash",
    "gemini__gemini-2.5-flash": "Gemini-2.5-Flash",
    "gemini__gemini-2.5-pro": "Gemini-2.5-Pro",
    "gemini__gemini-3-flash-preview": "Gemini-3-Flash-Preview",
    "gemini__gemini-3-pro-preview": "Gemini-3-Pro-Preview",

    # Anthropic Claude models (neulab)
    "neulab__claude-opus-4-5-20251101": "Claude-4.5-Opus",
    "neulab__claude-sonnet-4-20250514": "Claude-4-Sonnet",
    "neulab__claude-sonnet-4-5-20250929": "Claude-4.5-Sonnet",

    # Other models (neulab)
    "neulab__kimi-k2-0711-preview": "Kimi-K2-Instruct",
    "neulab__qwen3-coder-480b-a35b-instruct": "Qwen3-Coder-480B-A35B-Instruct",
}

# Reverse lookup for debugging
SWEBENCH_MODEL_NAMES = {
    "Claude-4.5-Sonnet.csv": "Claude-4.5-Sonnet",
    "Claude-4-Opus.csv": "Claude-4-Opus",
    "Claude-4-Sonnet.csv": "Claude-4-Sonnet",
    "Gemini-2.5-Pro.csv": "Gemini-2.5-Pro",
    "Gemini-3-Pro-Preview.csv": "Gemini-3-Pro-Preview",
    "GPT-5__high.csv": "GPT-5",
    "GPT-OSS-120B.csv": "GPT-OSS-120B",
    "Kimi-K2-Instruct.csv": "Kimi-K2-Instruct",
    "o3__high.csv": "o3",
    "o4-mini__high.csv": "o4-mini",
    "Qwen2.5-Coder-32B-Instruct.csv": "Qwen2.5-Coder-32B-Instruct",
    "Qwen3-Coder-480B-A35B-Instruct.csv": "Qwen3-Coder-480B-A35B-Instruct",
}


def parse_jsonl_filename(filename: str) -> Tuple[str, str]:
    """
    Parse a JSONL filename to extract task name and date.

    Format: samples_<name>_<date>.jsonl
    Example: samples_acp_areach_gen_2026-02-03T15-03-30.267970.jsonl

    Returns: (subtask_name, date_string)
    """
    # Remove prefix and suffix
    basename = filename.replace("samples_", "").replace(".jsonl", "")

    # Split by date pattern (YYYY-MM-DDTHH-MM-SS)
    # The date starts with 4 digits followed by dash
    date_pattern = r'_(\d{4}-\d{2}-\d{2}T[\d\-\.]+)$'
    match = re.search(date_pattern, basename)

    if match:
        date_str = match.group(1)
        subtask_name = basename[:match.start()]
    else:
        # Fallback: split on last underscore before date-like pattern
        parts = basename.rsplit('_', 1)
        subtask_name = parts[0]
        date_str = parts[1] if len(parts) > 1 else ""

    return subtask_name, date_str


def extract_metrics_from_line(line_data: Dict[str, Any]) -> List[Tuple[float, str]]:
    """
    Extract metric scores from a single JSONL line.

    Returns: List of (score, metric_name) tuples
    """
    metrics = line_data.get("metrics", [])
    results = []

    for metric_name in metrics:
        if metric_name in line_data:
            value = line_data[metric_name]

            # Handle different value types
            if isinstance(value, bool):
                score = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                score = float(value)
            elif isinstance(value, list):
                # For list metrics (like inst_level_strict_acc), compute mean
                bool_vals = [1.0 if v else 0.0 for v in value if isinstance(v, bool)]
                num_vals = [float(v) for v in value if isinstance(v, (int, float))]
                all_vals = bool_vals + num_vals
                if all_vals:
                    score = sum(all_vals) / len(all_vals)
                else:
                    continue
            else:
                continue

            results.append((score, metric_name))

    return results


def process_jsonl_file(filepath: Path) -> List[Tuple[float, str]]:
    """
    Process a single JSONL file and extract all metric scores.

    Returns: List of (score, metric_name) tuples
    """
    results = []

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                metrics = extract_metrics_from_line(data)
                results.extend(metrics)
            except json.JSONDecodeError as e:
                print(f"  Warning: JSON decode error in {filepath}:{line_num}: {e}")
            except Exception as e:
                print(f"  Warning: Error processing {filepath}:{line_num}: {e}")

    return results


def normalize_task_name(task_dir: str) -> str:
    """
    Normalize task directory name for output.
    Handles comma-separated task names by creating a single normalized name.
    """
    # Handle comma-separated task names (e.g., gpqa_diamond_cot_zeroshot,gpqa_main_cot_zeroshot,...)
    if ',' in task_dir:
        # Take the common prefix or create a combined name
        parts = task_dir.split(',')
        # Find common prefix
        common = parts[0]
        for part in parts[1:]:
            while not part.startswith(common) and common:
                common = common[:-1]

        if common.endswith('_'):
            common = common[:-1]

        # If we have a meaningful common prefix, use it; otherwise use first task
        if len(common) > 5:
            return common
        else:
            return parts[0]

    return task_dir


def standardize_data(
    input_dir: Path,
    output_dir: Path,
    verbose: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    Main standardization function.

    Args:
        input_dir: Path to proxy_bench_data directory
        output_dir: Path to output standardized_data directory
        verbose: Print progress information

    Returns:
        Statistics dictionary with processing info
    """
    stats = defaultdict(lambda: defaultdict(int))

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all task directories
    task_dirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if verbose:
        print(f"Found {len(task_dirs)} task directories")
        print("-" * 60)

    for task_dir in sorted(task_dirs):
        task_name = task_dir.name
        normalized_task = normalize_task_name(task_name)

        if verbose:
            print(f"\nProcessing task: {task_name}")
            if normalized_task != task_name:
                print(f"  -> normalized to: {normalized_task}")

        # Get all model directories
        model_dirs = [d for d in task_dir.iterdir() if d.is_dir()]

        for model_dir in sorted(model_dirs):
            model_name_raw = model_dir.name

            # Normalize model name
            if model_name_raw in MODEL_NAME_LOOKUP:
                model_name = MODEL_NAME_LOOKUP[model_name_raw]
            else:
                # Fallback: use raw name with basic cleanup
                model_name = model_name_raw.replace("__", "-")
                if verbose:
                    print(f"  Warning: Unknown model '{model_name_raw}', using '{model_name}'")

            # Find all JSONL files
            jsonl_files = list(model_dir.glob("samples_*.jsonl"))

            if not jsonl_files:
                if verbose:
                    print(f"  No JSONL files found for {model_name_raw}")
                continue

            # Process each JSONL file (subtask)
            for jsonl_file in jsonl_files:
                subtask_name, date_str = parse_jsonl_filename(jsonl_file.name)

                # Create output directory: task-subtask/
                # If subtask is same as task, just use task name
                if subtask_name == normalized_task or subtask_name.startswith(normalized_task):
                    output_subdir = subtask_name
                else:
                    output_subdir = f"{normalized_task}-{subtask_name}"

                output_task_dir = output_dir / output_subdir
                output_task_dir.mkdir(parents=True, exist_ok=True)

                # Process the JSONL file
                metrics_data = process_jsonl_file(jsonl_file)

                if not metrics_data:
                    if verbose:
                        print(f"    No metrics found in {jsonl_file.name}")
                    continue

                # Write CSV output
                output_csv = output_task_dir / f"{model_name}.csv"

                # If file exists from another date, we need to handle this
                # For now, we'll append/overwrite based on whether file exists
                mode = 'w'  # Always overwrite - use latest data

                with open(output_csv, mode, newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['score', 'metric_name'])
                    for score, metric_name in metrics_data:
                        writer.writerow([score, metric_name])

                stats[output_subdir][model_name] = len(metrics_data)

                if verbose:
                    print(f"    {subtask_name} -> {model_name}.csv ({len(metrics_data)} entries)")

    return dict(stats)


def print_summary(stats: Dict[str, Dict[str, int]]):
    """Print a summary of processed data."""
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)

    total_files = 0
    total_entries = 0

    for task_subtask in sorted(stats.keys()):
        models = stats[task_subtask]
        print(f"\n{task_subtask}/")
        for model in sorted(models.keys()):
            count = models[model]
            print(f"  {model}.csv: {count} entries")
            total_files += 1
            total_entries += count

    print("\n" + "-" * 60)
    print(f"Total: {total_files} CSV files, {total_entries} total entries")


def print_lookup_table():
    """Print the model name lookup table for reference."""
    print("\n" + "=" * 60)
    print("MODEL NAME LOOKUP TABLE")
    print("=" * 60)
    print(f"{'proxy_bench_data name':<50} -> {'normalized name'}")
    print("-" * 60)
    for raw, normalized in sorted(MODEL_NAME_LOOKUP.items()):
        print(f"{raw:<50} -> {normalized}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Standardize proxy_bench_data")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/proxy_bench_data"),
        help="Input directory containing task/model/jsonl structure"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/standardized_data"),
        help="Output directory for standardized CSV files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print verbose progress information"
    )
    parser.add_argument(
        "--show-lookup",
        action="store_true",
        help="Show model name lookup table and exit"
    )

    args = parser.parse_args()

    if args.show_lookup:
        print_lookup_table()
        return

    # Convert to absolute paths if needed
    script_dir = Path(__file__).parent.parent  # proxy-bench directory

    if not args.input_dir.is_absolute():
        input_dir = script_dir / args.input_dir
    else:
        input_dir = args.input_dir

    if not args.output_dir.is_absolute():
        output_dir = script_dir / args.output_dir
    else:
        output_dir = args.output_dir

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    stats = standardize_data(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
