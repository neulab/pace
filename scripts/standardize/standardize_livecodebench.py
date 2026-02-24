#!/usr/bin/env python3
"""
Standardize LiveCodeBench scores from results/raw_results/livecodebench into CSV.

Layout observed:
- results/raw_results/livecodebench/<model>/
    - codegeneration_1_eval.json
    - selfrepair_1_eval.json
    - testoutputprediction_1_eval.json
    - codeexecution_1_cot_eval.json

We extract per-instance pass indicators and write one CSV per subtask:
- results/standardized_results/livecodebench/codegeneration/{model}.csv
- results/standardized_results/livecodebench/selfrepair/{model}.csv
- results/standardized_results/livecodebench/testoutputprediction/{model}.csv
- results/standardized_results/livecodebench/codeexecution/{model}.csv

CSV columns:
    id,score,metric_name

Where score is 1.0 for pass and 0.0 for fail.
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import json

from utils import (
    normalize_model_name,
    write_csv,
    resolve_paths,
    print_summary,
)


# ------------------------- Parsers for subtasks ------------------------- #

def _parse_detail_pass1(file: Path) -> List[Tuple[float, str, str]]:
    """Parse files like *eval.json that contain detail.pass@1 mapping.

    Expected structure:
    [
      {
        "pass@1": <float>,
        "detail": { "pass@1": { "0": 1.0, "1": 0.0, ... } }
      },
      ... other logs ...
    ]
    """
    results: List[Tuple[float, str, str]] = []
    try:
        data = json.loads(file.read_text())
        if not isinstance(data, list) or not data:
            return results
        summary = data[0]
        if not isinstance(summary, dict):
            return results
        detail = summary.get("detail", {}) or {}
        pass1 = detail.get("pass@1", {}) or {}
        if isinstance(pass1, dict):
            for k, v in pass1.items():
                rid = str(k)
                try:
                    score = 1.0 if bool(v) else 0.0
                except Exception:
                    try:
                        score = float(v)
                        score = 1.0 if score >= 0.5 else 0.0
                    except Exception:
                        continue
                results.append((rid, score, "pass@1"))
    except Exception as e:
        print(f"  Error parsing {file}: {e}")
    return results


def _parse_codeexecution(file: Path) -> List[Tuple[float, str, str]]:
    """Parse codeexecution_1_cot_eval.json structure.

    Expected structure:
    [
      {"pass@1": 100.0},
      {"0": [[true]], "1": [[true]], ...}
    ]
    """
    results: List[Tuple[float, str, str]] = []
    try:
        data = json.loads(file.read_text())
        if not isinstance(data, list) or len(data) < 2:
            return results
        per_case = data[1]
        if not isinstance(per_case, dict):
            return results
        for k, v in per_case.items():
            rid = str(k)
            # v can be nested lists of booleans, e.g., [[true]]
            passed = False
            try:
                if isinstance(v, list):
                    # flatten 2 levels defensively
                    flat: List[Any] = []
                    for a in v:
                        if isinstance(a, list):
                            flat.extend(a)
                        else:
                            flat.append(a)
                    passed = any(bool(x) for x in flat)
                else:
                    passed = bool(v)
            except Exception:
                passed = False
            results.append((rid, 1.0 if passed else 0.0, "pass@1"))
    except Exception as e:
        print(f"  Error parsing {file}: {e}")
    return results


# ------------------------- Main standardizer ------------------------- #

def standardize_livecodebench(input_root: Path, output_root: Path, verbose: bool = True) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Subtask -> (filename, parser, output_subdir)
    tasks = {
        "codegeneration": ("codegeneration_1_eval.json", _parse_detail_pass1, "codegeneration"),
        "selfrepair": ("selfrepair_1_eval.json", _parse_detail_pass1, "selfrepair"),
        "testoutputprediction": ("testoutputprediction_1_eval.json", _parse_detail_pass1, "testoutputprediction"),
        "codeexecution": ("codeexecution_1_cot_eval.json", _parse_codeexecution, "codeexecution"),
    }

    # Ensure output subdirs exist under livecodebench/
    base_out = output_root / "livecodebench"
    for _, (_, __, subdir) in tasks.items():
        (base_out / subdir).mkdir(parents=True, exist_ok=True)

    # Model directories
    model_dirs = [d for d in input_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if verbose:
        print(f"Found {len(model_dirs)} model directories")

    for model_dir in sorted(model_dirs):
        raw_model = model_dir.name
        model_name = normalize_model_name(raw_model)

        # Aggregate across runs per subtask (or directly if no run dirs)
        for task_key, (fname, parser, subdir) in tasks.items():
            aggregated: List[Tuple[float, str, str]] = []

            eval_paths: List[Path] = []
            fpath = model_dir / fname
            if fpath.exists():
                eval_paths.append(fpath)
            elif verbose:
                print(f"  Missing {fname} under model dir {model_dir.name}")

            for fpath in eval_paths:
                entries = parser(fpath)
                aggregated.extend(entries)

            if not aggregated:
                if verbose:
                    print(f"  No metrics for subtask '{task_key}'")
                continue

            out_dir = base_out / subdir
            output_csv = out_dir / f"{model_name}.csv"
            write_csv(output_csv, aggregated)
            # Record stats under nested path key for clarity in summary
            stats[f"livecodebench/{subdir}"][model_name] = len(aggregated)
            if verbose:
                print(f"  Wrote livecodebench/{subdir}/{output_csv.name} ({len(aggregated)} entries)")

    return dict(stats)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Standardize LiveCodeBench eval scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/livecodebench"),
        help="Root directory containing LiveCodeBench <model>/<run>/files",
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

    stats = standardize_livecodebench(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)

    print(f"\nDone! Standardized data written to: {output_dir}")
    return 0


if __name__ == "__main__":
    main()
