#!/usr/bin/env python3
"""
Standardize VisualPuzzles scores from results/raw_results/visualpuzzles into CSVs.

- Input layout:
  results/raw_results/visualpuzzles/
    - {model_name}/
        - *_samples_VisualPuzzles_cot.jsonl
    - anthropic__claude-opus-4-5/
    - azure__gpt-5.2/
    - gemini__gemini-3-flash-preview/
    - ...

- Raw JSONL format (one JSON object per line):
  {
    "doc_id": 0,
    "doc": {"id": 1, "question": "...", "answer": "D", "category": "inductive", "difficulty": "hard"},
    "target": "D",
    "exact_match": 1.0,
    ...
  }

- Output layout:
  results/standardized_results/visualpuzzles/{normalized_model_name}.csv

CSV columns:
  id,score,metric_name

Score: exact_match value (0.0 or 1.0)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from utils import (
    write_csv,
    resolve_paths,
    print_summary,
    normalize_model_name,
)


def _process_visualpuzzles_jsonl(filepath: Path) -> List[Tuple[str, float, str]]:
    """
    Process a VisualPuzzles JSONL file and extract per-instance results.
    
    Args:
        filepath: Path to the JSONL file
    
    Returns:
        List of (id, score, metric_name) tuples
    """
    results: List[Tuple[str, float, str]] = []
    
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                
                # Extract doc_id and exact_match score
                doc_id = record.get("doc_id")
                exact_match = record.get("exact_match", 0.0)
                
                # Create instance ID
                instance_id = f"visualpuzzles_{doc_id}"
                
                # Ensure score is float and in [0, 1]
                score = float(exact_match) if exact_match is not None else 0.0
                score = max(0.0, min(1.0, score))
                
                results.append((instance_id, score, "exact_match"))
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                continue
    
    return results


def main(
    input_dir: Path = Path("results/raw_results/visualpuzzles"),
    output_dir: Path = Path("results/standardized_results"),
) -> None:
    """
    Main function to standardize all VisualPuzzles results.
    """
    input_dir, output_dir = resolve_paths(input_dir, output_dir)
    
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return
    
    # Find all model directories
    model_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"Found {len(model_dirs)} model directories")
    
    # Statistics for summary
    stats: Dict[str, Dict[str, int]] = defaultdict(dict)
    
    # Output directory for visualpuzzles
    task_output_dir = output_dir / "visualpuzzles"
    
    for model_dir in sorted(model_dirs):
        model_raw = model_dir.name
        
        # Find JSONL files in this model directory
        jsonl_files = list(model_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            print(f"  [WARN] No JSONL files found in {model_raw}")
            continue
        
        # Use the first (or only) JSONL file
        jsonl_file = jsonl_files[0]
        
        try:
            # Normalize model name
            model_normalized = normalize_model_name(model_raw)
            
            print(f"Processing {model_raw} -> {model_normalized}")
            
            # Process the JSONL file
            results = _process_visualpuzzles_jsonl(jsonl_file)
            
            if not results:
                print(f"  [WARN] No results extracted from {jsonl_file.name}")
                continue
            
            # Write output CSV
            output_csv = task_output_dir / f"{model_normalized}.csv"
            write_csv(output_csv, results)
            
            # Track statistics
            stats["visualpuzzles"][model_normalized] = len(results)
            
        except Exception as e:
            raise ValueError(f"  [ERROR] Failed to process {model_raw}: {e}")
    
    # Print summary
    print_summary(stats)
    print(f"\nDone! Standardized data written to: {task_output_dir}")


if __name__ == "__main__":
    main()
