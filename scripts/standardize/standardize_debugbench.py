#!/usr/bin/env python3
"""
Standardize DebugBench scores from results/raw_results/debugbench into CSV.

Directory layout:
    debugbench/{provider}/{model}/debug/*.json          (most providers)
    debugbench/fireworks_ai/accounts/fireworks/models/{model}/debug/*.json

Each *.json file corresponds to one bug type (e.g. "python3_condition error")
and contains a list of instances with fields:
    - slug            : problem identifier
    - test_result_bool: True/False (pass/fail)

Output CSV format:
    id,score,metric_name

where id = "{bug_type}_{slug}", score in {0.0, 1.0}, metric_name = "debug_accuracy"

Model name key is constructed as "{provider_path_joined_with_underscores}__{model_dir}"
e.g.  anthropic__claude-opus-4-6
      azure_ai__gpt-5.2
      fireworks_ai_accounts_fireworks_models_glm-4p7

Usage:
    python standardize_debugbench.py
    python standardize_debugbench.py --output-dir results/standardized_results
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from utils import (
    normalize_model_name,
    write_csv,
    resolve_paths,
    print_summary,
)


def _build_model_key(debug_dir: Path, base_dir: Path) -> str:
    """Construct the MODEL_NAME_LOOKUP key from the debug directory path.

    Given:  .../debugbench/anthropic/claude-opus-4-6/debug
    Returns: anthropic__claude-opus-4-6

    Given:  .../debugbench/fireworks_ai/accounts/fireworks/models/glm-4p7/debug
    Returns: fireworks_ai_accounts_fireworks_models_glm-4p7
    """
    # path relative to base_dir, without the trailing "debug" component
    rel = debug_dir.relative_to(base_dir).parent  # e.g. anthropic/claude-opus-4-6
    parts = list(rel.parts)                        # ['anthropic', 'claude-opus-4-6']

    provider = parts[0]
    model = parts[-1]
    middle = parts[1:-1]  # non-empty only for fireworks_ai

    if middle:
        # e.g. fireworks_ai/accounts/fireworks/models/glm-4p7
        prefix = "_".join([provider] + middle)
        return f"{prefix}_{model}"
    else:
        return f"{provider}__{model}"


def _parse_debug_dir(debug_dir: Path) -> List[Tuple[str, float, str]]:
    """Parse all *.json files under a model's debug/ directory.

    Returns list of (id, score, metric_name) tuples.
    id = "{bug_type}_{slug}"
    score = 1.0 if test_result_bool is True else 0.0
    """
    results: List[Tuple[str, float, str]] = []

    for json_file in sorted(debug_dir.glob("*.json")):
        bug_type = json_file.stem  # e.g. "python3_condition error"

        try:
            with open(json_file) as f:
                instances = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: could not read {json_file}: {e}")
            continue

        for inst in instances:
            slug = inst.get("slug", "")
            if not slug:
                continue

            raw_bool = inst.get("test_result_bool")
            if raw_bool is True or raw_bool == "True":
                score = 1.0
            else:
                score = 0.0

            rid = f"{bug_type}_{slug}"
            results.append((rid, score, "debug_accuracy"))

    return results


def standardize_debugbench(
    input_dir: Path, output_dir: Path, verbose: bool = True
) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    output_task_dir = output_dir / "debugbench"
    output_task_dir.mkdir(parents=True, exist_ok=True)

    debug_dirs = sorted(input_dir.rglob("debug"))
    debug_dirs = [d for d in debug_dirs if d.is_dir()]

    if verbose:
        print(f"Found {len(debug_dirs)} model debug/ directories in debugbench/")

    for debug_dir in debug_dirs:
        model_key = _build_model_key(debug_dir, input_dir)

        try:
            model_name = normalize_model_name(model_key)
        except ValueError as e:
            print(f"  Skipping {model_key}: {e}")
            continue

        metrics = _parse_debug_dir(debug_dir)

        if not metrics:
            if verbose:
                print(f"  No metrics found for {model_key}")
            continue

        output_csv = output_task_dir / f"{model_name}.csv"
        write_csv(output_csv, metrics)
        stats["debugbench"][model_name] = len(metrics)

        if verbose:
            print(f"  Wrote {output_csv.name} ({len(metrics)} entries)")

    return dict(stats)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Standardize DebugBench eval scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/debugbench"),
        help="Root directory of DebugBench raw results",
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
    )

    args = parser.parse_args()
    input_dir, output_dir = resolve_paths(args.input_dir, args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    stats = standardize_debugbench(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir / 'debugbench'}")
    return 0


if __name__ == "__main__":
    main()
