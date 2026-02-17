#!/usr/bin/env python3
"""
Standardize eval_scores data into a consistent format.

This script:
1. Reads eval results from data/eval_scores/<task>/<model>/
2. Extracts per-instance metric scores
3. Outputs CSV files in standardized_data/<task>/<normalized_model_name>.csv

The CSV format is:
    score,metric_name

Handles two data formats:
- Code tasks (humaneval, mbpp): .eval_results.json files with pass/fail status
- MMLU tasks: output.jsonl files with accuracy scores
"""

import os
import json
import csv
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# =============================================================================
# MODEL NAME LOOKUP TABLE
# Maps eval_scores model names -> normalized names (swebench style)
# =============================================================================

MODEL_NAME_LOOKUP = {
    # Azure/OpenAI models
    "azure--gpt-4o": "GPT-4o",
    "azure--gpt-5": "GPT-5",
    "azure--gpt-5-mini": "GPT-5-mini",
    "azure--gpt-5-nano": "GPT-5-nano",
    "azure--gpt-oss-120b": "GPT-OSS-120B",
    "azure--o3": "o3",
    "azure--o4-mini": "o4-mini",

    # Gemini models
    "gemini--gemini-2.0-flash-exp": "Gemini-2.0-Flash",
    "gemini--gemini-2.5-flash": "Gemini-2.5-Flash",
    "gemini--gemini-2.5-pro": "Gemini-2.5-Pro",
    "gemini--gemini-3-flash-preview": "Gemini-3-Flash-Preview",
    "gemini--gemini-3-pro-preview": "Gemini-3-Pro-Preview",

    # Anthropic Claude models (neulab)
    "neulab--claude-opus-4-5-20251101": "Claude-4.5-Opus",
    "neulab--claude-sonnet-4-20250514": "Claude-4-Sonnet",
    "neulab--claude-sonnet-4-5-20250929": "Claude-4.5-Sonnet",

    # Other models (neulab)
    "neulab--kimi-k2-0711-preview": "Kimi-K2-Instruct",
    "neulab--qwen3-coder-480b-a35b-instruct": "Qwen3-Coder-480B-A35B-Instruct",
    "neulab--gpt-4.1-mini-2025-04-14": "GPT-4.1-mini",
    "neulab--gpt-4o-2024-08-06": "GPT-4o",
    "neulab--llama4-maverick-instruct": "Llama-4-Maverick-Instruct",
    "neulab--llama4-scout-instruct": "Llama-4-Scout-Instruct",
}


def parse_model_name_from_filename(filename: str) -> Tuple[str, Optional[str]]:
    """
    Parse model name and optional reasoning level from filename.

    Examples:
        azure--gpt-5-mini:reasoning:medium_openai_temp_1.0.eval_results.json
        -> ("azure--gpt-5-mini", "reasoning:medium")

        azure--o3_openai_temp_1.0.eval_results.json
        -> ("azure--o3", None)

    Returns: (model_name, reasoning_level)
    """
    # Remove suffix patterns
    name = filename
    for suffix in [".eval_results.json", ".jsonl", ".raw.jsonl", "_openai_temp_1.0"]:
        name = name.replace(suffix, "")

    # Check for reasoning level (e.g., ":reasoning:medium")
    reasoning_match = re.search(r':reasoning:(low|medium|high)', name)
    if reasoning_match:
        reasoning_level = reasoning_match.group(0)[1:]  # Remove leading ":"
        model_name = name[:reasoning_match.start()]
    else:
        reasoning_level = None
        model_name = name

    return model_name, reasoning_level


def normalize_model_name(raw_name: str, reasoning_level: Optional[str] = None) -> str:
    """
    Normalize model name to standardized format.
    """
    if raw_name in MODEL_NAME_LOOKUP:
        base_name = MODEL_NAME_LOOKUP[raw_name]
    else:
        # Fallback: basic cleanup
        base_name = raw_name.replace("--", "-")

    # Append reasoning level if present
    if reasoning_level:
        return f"{base_name}__{reasoning_level.replace(':', '_')}"

    return base_name


def process_humaneval_mbpp_file(filepath: Path) -> List[Tuple[float, str]]:
    """
    Process HumanEval or MBPP eval_results.json file.

    Extracts per-task pass/fail status as scores.

    Returns: List of (score, metric_name) tuples
    """
    results = []

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Error reading {filepath}: {e}")
        return results

    eval_data = data.get("eval", {})

    for task_id, task_results in eval_data.items():
        for result in task_results:
            # base_status: pass/fail for base tests
            base_status = result.get("base_status", "")
            base_score = 1.0 if base_status == "pass" else 0.0
            results.append((base_score, "base_pass"))

            # plus_status: pass/fail for plus tests (harder)
            plus_status = result.get("plus_status", "")
            plus_score = 1.0 if plus_status == "pass" else 0.0
            results.append((plus_score, "plus_pass"))

    return results


def process_mmlu_file(filepath: Path) -> List[Tuple[float, str]]:
    """
    Process MMLU output.jsonl file.

    Extracts per-sample accuracy scores.

    Returns: List of (score, metric_name) tuples
    """
    results = []

    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Extract accuracy from the line
                    accuracy = data.get("accuracy", [])
                    if isinstance(accuracy, list) and accuracy:
                        score = float(accuracy[0])
                    elif isinstance(accuracy, (int, float)):
                        score = float(accuracy)
                    else:
                        continue

                    results.append((score, "accuracy"))

                except json.JSONDecodeError as e:
                    print(f"  Warning: JSON error at {filepath}:{line_num}: {e}")
                except Exception as e:
                    print(f"  Warning: Error at {filepath}:{line_num}: {e}")
    except IOError as e:
        print(f"  Error reading {filepath}: {e}")

    return results


def standardize_eval_scores(
    input_dir: Path,
    output_dir: Path,
    verbose: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    Main standardization function.

    Args:
        input_dir: Path to eval_scores directory
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

        if verbose:
            print(f"\nProcessing task: {task_name}")

        # Determine task type
        is_code_task = task_name in ["humaneval", "mbpp"]

        # Create output directory for this task
        output_task_dir = output_dir / task_name
        output_task_dir.mkdir(parents=True, exist_ok=True)

        if is_code_task:
            # Code tasks: files are directly in task directory
            process_code_task(task_dir, output_task_dir, task_name, stats, verbose)
        else:
            # MMLU tasks: files are in model subdirectories
            process_mmlu_task(task_dir, output_task_dir, task_name, stats, verbose)

    return dict(stats)


def process_code_task(
    task_dir: Path,
    output_dir: Path,
    task_name: str,
    stats: Dict,
    verbose: bool
):
    """Process HumanEval or MBPP task."""
    # Find all eval_results.json files
    eval_files = list(task_dir.glob("*.eval_results.json"))

    if verbose:
        print(f"  Found {len(eval_files)} eval_results.json files")

    for eval_file in sorted(eval_files):
        # Parse model name from filename
        model_raw, reasoning_level = parse_model_name_from_filename(eval_file.name)
        model_name = normalize_model_name(model_raw, reasoning_level)

        # Process the file
        metrics_data = process_humaneval_mbpp_file(eval_file)

        if not metrics_data:
            if verbose:
                print(f"    No metrics found in {eval_file.name}")
            continue

        # Write CSV output
        output_csv = output_dir / f"{model_name}.csv"

        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['score', 'metric_name'])
            for score, metric_name in metrics_data:
                writer.writerow([score, metric_name])

        stats[task_name][model_name] = len(metrics_data)

        if verbose:
            print(f"    {model_name}.csv ({len(metrics_data)} entries)")


def process_mmlu_task(
    task_dir: Path,
    output_dir: Path,
    task_name: str,
    stats: Dict,
    verbose: bool
):
    """Process MMLU task with model subdirectories."""
    # Get all model directories
    model_dirs = [d for d in task_dir.iterdir() if d.is_dir()]

    if verbose:
        print(f"  Found {len(model_dirs)} model directories")

    for model_dir in sorted(model_dirs):
        model_raw = model_dir.name

        # Check for reasoning level in directory name
        reasoning_match = re.search(r':reasoning:(low|medium|high)', model_raw)
        if reasoning_match:
            reasoning_level = reasoning_match.group(0)[1:]
            base_model = model_raw[:reasoning_match.start()]
        else:
            reasoning_level = None
            base_model = model_raw

        model_name = normalize_model_name(base_model, reasoning_level)

        # Look for output.jsonl
        output_jsonl = model_dir / "output.jsonl"

        if not output_jsonl.exists():
            if verbose:
                print(f"    No output.jsonl found for {model_raw}")
            continue

        # Process the file
        metrics_data = process_mmlu_file(output_jsonl)

        if not metrics_data:
            if verbose:
                print(f"    No metrics found in {model_raw}/output.jsonl")
            continue

        # Write CSV output
        output_csv = output_dir / f"{model_name}.csv"

        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['score', 'metric_name'])
            for score, metric_name in metrics_data:
                writer.writerow([score, metric_name])

        stats[task_name][model_name] = len(metrics_data)

        if verbose:
            print(f"    {model_name}.csv ({len(metrics_data)} entries)")


def print_summary(stats: Dict[str, Dict[str, int]]):
    """Print a summary of processed data."""
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)

    total_files = 0
    total_entries = 0

    for task in sorted(stats.keys()):
        models = stats[task]
        print(f"\n{task}/")
        for model in sorted(models.keys()):
            count = models[model]
            print(f"  {model}.csv: {count} entries")
            total_files += 1
            total_entries += count

    print("\n" + "-" * 60)
    print(f"Total: {total_files} CSV files, {total_entries} total entries")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Standardize eval_scores data")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/eval_scores"),
        help="Input directory containing task/model structure"
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

    args = parser.parse_args()

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

    stats = standardize_eval_scores(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
