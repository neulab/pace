#!/usr/bin/env python3
"""
Standardize RepoBench scores from results/raw_results/repobench into CSVs grouped by task and metric.

- Input layout:
  results/raw_results/repobench/
    - repobench_if_python/
        - {model}-python-8k-codebleu.csv
        - {model}-python-8k-edit_similarity.csv
        - {model}-python-8k-exact_match.csv
    - repobench_xff_python/
        - {model}-python-8k-codebleu.csv
        - ...
    - repobench_xfr_python/
        - {model}-python-8k-codebleu.csv
        - ...

- Raw CSV format:
  idx,score,metric_name
  0,76.43161313935813,codebleu
  1,50.0,codebleu
  ...

- Output layout (one CSV per model per task+metric):
  results/standardized_results/repobench/<task>_<metric>/{normalized_model_name}.csv
  
  Example output folders:
    - repobench_if_python_codebleu/
    - repobench_if_python_edit_similarity/
    - repobench_if_python_exact_match/
    - repobench_xff_python_codebleu/
    - repobench_xff_python_edit_similarity/
    - repobench_xff_python_exact_match/
    - repobench_xfr_python_codebleu/
    - repobench_xfr_python_edit_similarity/
    - repobench_xfr_python_exact_match/

CSV columns:
  id,score,metric_name

Score normalization:
  - codebleu: raw values are 0-100, normalized to 0-1
  - edit_similarity: raw values are 0-100, normalized to 0-1  
  - exact_match: already 0 or 1, kept as-is
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from utils import (
    write_csv,
    resolve_paths,
    print_summary,
    normalize_model_name,
)

# Metrics that need normalization from 0-100 to 0-1
METRICS_NORMALIZE_100 = {"codebleu", "edit_similarity"}


def _parse_repobench_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse a RepoBench result filename to extract model name and metric.
    
    Filename format: {model}-python-8k-{metric}.csv
    
    Examples:
        gpt-5-python-8k-codebleu.csv -> ("gpt-5", "codebleu")
        claude-opus-4-5-python-8k-exact_match.csv -> ("claude-opus-4-5", "exact_match")
        Kimi-K2.5-python-8k-edit_similarity.csv -> ("Kimi-K2.5", "edit_similarity")
    
    Returns:
        (model_name, metric_name) or None if parsing fails
    """
    if not filename.endswith(".csv"):
        return None
    
    # Remove .csv extension
    name = filename[:-4]
    
    # Pattern: {model}-python-8k-{metric}
    # The model name can contain hyphens, so we need to find "-python-8k-" as delimiter
    match = re.search(r"-python-8k-(\w+)$", name)
    if not match:
        return None
    
    metric = match.group(1)
    model = name[:match.start()]
    
    return model, metric


def _process_repobench_csv(filepath: Path, metric: str) -> List[Tuple[str, float, str]]:
    """
    Process a RepoBench CSV file and extract per-instance results.
    
    Args:
        filepath: Path to the CSV file
        metric: The metric name (codebleu, edit_similarity, exact_match)
    
    Returns:
        List of (id, score, metric_name) tuples with normalized scores
    """
    results: List[Tuple[str, float, str]] = []
    
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                idx = row.get("idx", "")
                score_raw = float(row.get("score", 0))
                metric_name = row.get("metric_name", metric)
                
                # Normalize score if needed (0-100 -> 0-1)
                if metric in METRICS_NORMALIZE_100:
                    score = score_raw / 100.0
                else:
                    score = score_raw
                
                # Clamp to [0, 1]
                score = max(0.0, min(1.0, score))
                
                # Create instance ID with task context
                instance_id = f"repobench_{idx}"
                
                results.append((instance_id, score, metric_name))
                
            except (ValueError, KeyError) as e:
                continue
    
    return results


def main(
    input_dir: Path = Path("results/raw_results/repobench"),
    output_dir: Path = Path("results/standardized_results"),
) -> None:
    """
    Main function to standardize all RepoBench results.
    """
    input_dir, output_dir = resolve_paths(input_dir, output_dir)
    
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return
    
    # Find all subtask directories
    subtask_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"Found {len(subtask_dirs)} subtask directories: {[d.name for d in subtask_dirs]}")
    
    # Statistics for summary
    stats: Dict[str, Dict[str, int]] = defaultdict(dict)
    
    for subtask_dir in sorted(subtask_dirs):
        subtask_name = subtask_dir.name
        print(f"\nProcessing subtask: {subtask_name}")
        
        # Find all CSV files in this subtask
        csv_files = list(subtask_dir.glob("*.csv"))
        
        # Group files by metric
        files_by_metric: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
        
        for csv_file in csv_files:
            parsed = _parse_repobench_filename(csv_file.name)
            if parsed is None:
                print(f"  [WARN] Could not parse filename: {csv_file.name}")
                continue
            
            model_raw, metric = parsed
            files_by_metric[metric].append((model_raw, csv_file))
        
        # Process each metric
        for metric, model_files in sorted(files_by_metric.items()):
            # Output folder: repobench/{subtask}_{metric}/
            task_metric_name = f"{subtask_name}_{metric}"
            task_output_dir = output_dir / "repobench" / task_metric_name
            
            print(f"  Processing metric: {metric} ({len(model_files)} models)")
            
            for model_raw, csv_file in model_files:
                model_normalized = normalize_model_name(model_raw)
                    
                # Process the CSV file
                results = _process_repobench_csv(csv_file, metric)
                    
                # Write output CSV
                output_csv = task_output_dir / f"{model_normalized}.csv"
                write_csv(output_csv, results)
                    
                # Track statistics
                stats[f"repobench/{task_metric_name}"][model_normalized] = len(results)
    
    # Print summary
    print_summary(stats)
    print(f"\nDone! Standardized data written to: {output_dir / 'repobench'}")


if __name__ == "__main__":
    main()
