#!/usr/bin/env python3
"""
Standardize MMLU COT Generative scores from results/raw_results/mmlu_cot into CSV.

Data format:
- Each model has a subdirectory containing:
  - results_*.json: summary results
  - samples_mmlu_{subject}_cot_generative_*.jsonl: per-instance results

- Each jsonl line contains:
  - doc_id: instance identifier within subject
  - doc.subject: subject name (e.g., "abstract_algebra")
  - exact_match: 0.0 or 1.0 score

Output CSV format:
    id,score,metric_name
    {subject}_{doc_id},{0|1},exact_match
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from utils import (
    parse_model_dir_name,
    normalize_model_name,
    write_csv,
    resolve_paths,
    print_summary,
)


def _process_mmlu_cot_samples(model_dir: Path) -> List[Tuple[str, float, str]]:
    """
    Process all samples_mmlu_*_cot_generative_*.jsonl files in a model directory.
    
    Returns list of (instance_id, score, metric_name) tuples.
    Instance ID format: {subject}_{doc_id}
    """
    results: List[Tuple[str, float, str]] = []
    
    # Find all sample jsonl files
    sample_files = list(model_dir.glob("samples_mmlu_*_cot_generative_*.jsonl"))
    
    for sample_file in sorted(sample_files):
        # Extract subject from filename: samples_mmlu_{subject}_cot_generative_*.jsonl
        filename = sample_file.name
        # Remove prefix and suffix to get subject
        # samples_mmlu_abstract_algebra_cot_generative_2026-03-25T23-24-45.218620.jsonl
        prefix = "samples_mmlu_"
        suffix_start = "_cot_generative_"
        
        if not filename.startswith(prefix):
            continue
        
        rest = filename[len(prefix):]
        if suffix_start not in rest:
            continue
        
        subject = rest[:rest.index(suffix_start)]
        
        try:
            with open(sample_file, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Get doc_id
                        doc_id = data.get("doc_id")
                        if doc_id is None:
                            continue
                        
                        # Get exact_match score
                        # Try exact_match field first, then check for flexible-extract
                        score = None
                        if "exact_match" in data:
                            score = float(data["exact_match"])
                        
                        if score is None:
                            continue
                        
                        # Create unique instance ID: subject_docid
                        instance_id = f"{subject}_{doc_id}"
                        results.append((instance_id, score, "exact_match"))
                        
                    except json.JSONDecodeError as e:
                        print(f"  Warning: JSON error at {sample_file.name}:{line_num}: {e}")
                    except Exception as e:
                        print(f"  Warning: Error at {sample_file.name}:{line_num}: {e}")
                        
        except OSError as e:
            print(f"  Error reading {sample_file}: {e}")
    
    return results


def standardize_mmlu_cot(
    input_root: Path, 
    output_root: Path, 
    verbose: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    Standardize MMLU COT Generative results.
    
    Args:
        input_root: Path to results/raw_results/mmlu_cot
        output_root: Path to results/standardized_results/mmlu_cot
    
    Returns:
        Dictionary of {task: {model: count}} statistics
    """
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    task_name = "mmlu_cot"
    
    output_task_dir = output_root
    output_task_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all model subdirectories
    model_dirs = [d for d in input_root.iterdir() if d.is_dir()]
    
    if verbose:
        print(f"Found {len(model_dirs)} model directories")
    
    for model_dir in sorted(model_dirs):
        base_model_raw, reasoning_level = parse_model_dir_name(model_dir.name)
        
        model_name = normalize_model_name(base_model_raw, reasoning_level)
        
        # Process all sample files
        metrics_data = _process_mmlu_cot_samples(model_dir)
        
        if not metrics_data:
            if verbose:
                print(f"  No metrics found in {model_dir.name}")
            continue
        
        output_csv = output_task_dir / f"{model_name}.csv"
        write_csv(output_csv, metrics_data)
        stats[task_name][model_name] = len(metrics_data)
        
        if verbose:
            print(f"  Wrote {output_csv.name} ({len(metrics_data)} instances)")
    
    return dict(stats)


def main() -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description="Standardize MMLU COT Generative eval scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/mmlu_cot"),
        help="Root directory containing model subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/standardized_results/mmlu_cot"),
        help="Output directory for standardized CSV files",
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
    
    stats = standardize_mmlu_cot(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)
    
    print(f"\nDone! Standardized data written to: {output_dir}")
    return 0


if __name__ == "__main__":
    main()
