#!/usr/bin/env python3
"""
Standardize PlanBench scores from results/raw_results/planbench into CSVs grouped by task.

- Input layout:
  results/raw_results/planbench/<provider>/<model>/task_*.json
  where task names are:
    - task_1_plan_generation
    - task_2_plan_optimality
    - task_3_plan_verification
    - task_4_plan_reuse
    - task_7_plan_execution
    - task_8_1_goal_shuffling
    - task_8_2_full_to_partial
    - task_8_3_partial_to_full

- Output layout (one CSV per model per task):
  results/standardized_results/planbench/<task>/{normalized_model_name}.csv

CSV columns:
  id,score,metric_name

We parse each JSON file and extract per-instance metrics.
The primary metric is "llm_correct" (boolean converted to 0/1).
The id is the instance_id from each instance.
"""
from __future__ import annotations

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

# The 8 tasks in planbench
PLANBENCH_TASKS = (
    "task_1_plan_generation",
    "task_2_plan_optimality",
    "task_3_plan_verification",
    "task_4_plan_reuse",
    "task_7_plan_execution",
    "task_8_1_goal_shuffling",
    "task_8_2_full_to_partial",
    "task_8_3_partial_to_full",
)


def get_model_name(provider: str, model: str) -> str:
    """Get normalized model name from provider and model directory names.
    
    Tries multiple key formats to find a match in MODEL_NAME_LOOKUP via normalize_model_name.
    """
    # Try provider__model format (double underscore)
    key = f"{provider}__{model}"
    try:
        return normalize_model_name(key)
    except ValueError:
        pass
    
    # Try provider--model format (double dash, used by some benchmarks)
    key_dash = f"{provider}--{model}"
    try:
        return normalize_model_name(key_dash)
    except ValueError:
        pass
    
    # Try just the model name
    try:
        return normalize_model_name(model)
    except ValueError:
        pass
    
    raise ValueError(f"Model not found in lookup: provider={provider}, model={model}")


def _extract_id(instance: Dict[str, Any], fallback_idx: int) -> str:
    """Extract instance ID from a planbench instance."""
    if "instance_id" in instance:
        return str(instance["instance_id"])
    return str(fallback_idx)


def _extract_score(instance: Dict[str, Any], task: str) -> float | None:
    """Extract correctness score from a planbench instance.
    
    For task_3_plan_verification, we use llm_correct_binary (if available).
    For other tasks, we use llm_correct.
    """
    # For plan verification, prefer llm_correct_binary
    if task == "task_3_plan_verification":
        if "llm_correct_binary" in instance:
            val = instance["llm_correct_binary"]
            if isinstance(val, bool):
                return 1.0 if val else 0.0
            if isinstance(val, (int, float)):
                return float(val)
    
    # General llm_correct field
    if "llm_correct" in instance:
        val = instance["llm_correct"]
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        if isinstance(val, (int, float)):
            return float(val)
    
    return None


def _get_metric_name(task: str) -> str:
    """Get the metric name for a given task."""
    if task == "task_3_plan_verification":
        return "llm_correct_binary"
    return "llm_correct"


def standardize_planbench(input_root: Path, output_root: Path, verbose: bool = True) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    base_out = output_root / "planbench"
    base_out.mkdir(parents=True, exist_ok=True)

    # Provider directories (e.g., anthropic, azure, gemini, neulab, openai)
    provider_dirs = [d for d in input_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if verbose:
        print(f"Found {len(provider_dirs)} provider directories")

    for provider_dir in sorted(provider_dirs):
        provider = provider_dir.name
        
        # Model directories under each provider
        model_dirs = [d for d in provider_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for model_dir in sorted(model_dirs):
            raw_model = model_dir.name
            model_name = get_model_name(provider, raw_model)
                
            if verbose:
                print(f"\nProcessing: {provider}/{raw_model} -> {model_name}")

            # Find all task_*.json files
            json_files = sorted(model_dir.glob("task_*.json"))
            if not json_files and verbose:
                print("  No task_*.json files found")

            for json_file in json_files:
                # Extract task name from filename (e.g., task_1_plan_generation.json -> task_1_plan_generation)
                task = json_file.stem
                
                if task not in PLANBENCH_TASKS:
                    if verbose:
                        print(f"  Skipping unrecognized task: {task}")
                    continue

                out_dir = base_out / task
                out_dir.mkdir(parents=True, exist_ok=True)

                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                except json.JSONDecodeError as e:
                    if verbose:
                        print(f"  JSON error in {json_file.name}: {e}")
                    continue

                instances = data.get("instances", [])
                if not instances:
                    if verbose:
                        print(f"  No instances in {json_file.name}")
                    continue

                metric_name = _get_metric_name(task)
                aggregated: List[Tuple[str, float, str]] = []
                
                for idx, instance in enumerate(instances):
                    rid = _extract_id(instance, fallback_idx=idx)
                    score = _extract_score(instance, task)
                    if score is not None:
                        aggregated.append((rid, score, metric_name))

                if not aggregated:
                    if verbose:
                        print(f"  No metrics extracted from {json_file.name}")
                    continue

                output_csv = out_dir / f"{model_name}.csv"
                write_csv(output_csv, aggregated)
                stats[f"planbench/{task}"][model_name] = len(aggregated)
                if verbose:
                    print(f"  Wrote planbench/{task}/{output_csv.name} ({len(aggregated)} entries)")

    return dict(stats)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Standardize PlanBench eval scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/planbench"),
        help="Root directory of planbench provider/model folders",
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

    stats = standardize_planbench(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir / 'planbench'}")
    return 0


if __name__ == "__main__":
    main()
