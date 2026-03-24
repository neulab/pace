#!/usr/bin/env python3
"""
Standardize BFCL scores from results/raw_results/bfcl into CSVs grouped by task.

- Input layout:
  results/raw_results/bfcl/<model>/
    - agentic/
        - BFCL_v4_web_search_base_score.json
        - BFCL_v4_web_search_no_snippet_score.json
        - memory/
            - kv/BFCL_v4_memory_kv_score.json
            - vector/BFCL_v4_memory_vector_score.json
            - rec_sum/BFCL_v4_memory_rec_sum_score.json
    - live/
        - BFCL_v4_live_irrelevance_score.json
        - BFCL_v4_live_multiple_score.json
        - BFCL_v4_live_parallel_multiple_score.json
        - BFCL_v4_live_parallel_score.json
        - BFCL_v4_live_relevance_score.json
        - BFCL_v4_live_simple_score.json
    - multi_turn/
        - BFCL_v4_multi_turn_base_score.json
        - BFCL_v4_multi_turn_long_context_score.json
        - BFCL_v4_multi_turn_miss_func_score.json
        - BFCL_v4_multi_turn_miss_param_score.json
    - non_live/
        - BFCL_v4_irrelevance_score.json
        - BFCL_v4_multiple_score.json
        - BFCL_v4_parallel_multiple_score.json
        - BFCL_v4_parallel_score.json
        - BFCL_v4_simple_java_score.json
        - BFCL_v4_simple_javascript_score.json
        - BFCL_v4_simple_python_score.json

- Output layout (one CSV per model per task):
  results/standardized_results/bfcl/<task>/{normalized_model_name}.csv
  
  Tasks are named based on the JSON file name, e.g.:
    - agentic_web_search_base
    - agentic_web_search_no_snippet
    - agentic_memory_kv
    - agentic_memory_vector
    - agentic_memory_rec_sum
    - live_irrelevance
    - live_multiple
    - live_parallel_multiple
    - live_parallel
    - live_relevance
    - live_simple
    - multi_turn_base
    - multi_turn_long_context
    - multi_turn_miss_func
    - multi_turn_miss_param
    - non_live_irrelevance
    - non_live_multiple
    - non_live_parallel_multiple
    - non_live_parallel
    - non_live_simple_java
    - non_live_simple_javascript
    - non_live_simple_python

CSV columns:
  id,score,metric_name

IMPORTANT: BFCL raw results only contain instances that the model got WRONG.
The first line contains aggregate stats with total_count and correct_count.
We generate all instance IDs and mark:
- Instances in the file as WRONG (score = 0.0)
- Instances NOT in the file as CORRECT (score = 1.0)

For tasks with simple ID patterns (e.g., non_live tasks), IDs are {test_category}_{i} for i in 0..total_count-1.
For tasks with complex ID patterns (e.g., live, memory tasks), we can only output the wrong instances.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from utils import (
    write_csv,
    resolve_paths,
    print_summary,
)

# Model name mappings for BFCL (model dir name -> normalized name)
BFCL_MODEL_NAME_LOOKUP: Dict[str, str] = {
    "claude-opus-4-5-20251101-FC": "Claude-4.5-Opus",
    "claude-sonnet-4-5-20250929-FC": "Claude-4.5-Sonnet",
    "claude-opus-4-6-FC": "Claude-4.6-Opus",
    "DeepSeek-V3.2-Exp-FC": "DeepSeek-V3.2",
    "gemini-2.5-flash-FC": "Gemini-2.5-Flash",
    "gemini-3-flash-preview-FC": "Gemini-3-Flash-Preview",
    "gemini-3-pro-preview-FC": "Gemini-3-Pro-Preview",
    "gpt-5.2-2025-12-11-FC": "GPT-5.2",
    "gpt-5.2-codex-FC": "GPT-5.2-Codex",
    "kimi-k2-thinking-FC": "Kimi-K2",
    "kimi-k2.5-FC": "Kimi-K2.5",
    "meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-FC": "Llama-4-Maverick-Instruct",
    "meta-llama_Llama-4-Scout-17B-16E-Instruct-FC": "Llama-4-Scout-Instruct",
    "minimax-m2p1-FC": "MiniMax-M2.1",
    "minimax-m2p5-FC": "MiniMax-M2.5",
    "nvidia_nemotron-3-nano-30b-a3b-FC": "Nemotron-3-Nano",
    "o3-2025-04-16-FC": "o3",
    "o4-mini-2025-04-16-FC": "o4-mini",
    "qwen3-235b-a22b-instruct-2507-FC": "Qwen3-235B-A22B",
    "qwen3-30b-a3b-instruct-2507-FC": "Qwen3-30B-A3B",
    "qwen3-coder-480b-a35b-FC": "Qwen3-Coder-480B-A35B-Instruct",
    "GLM-4.7-FC": "GLM-4.7",
}


def normalize_bfcl_model_name(raw_name: str) -> str:
    """Normalize a BFCL model directory name to a standardized format."""
    if raw_name in BFCL_MODEL_NAME_LOOKUP:
        return BFCL_MODEL_NAME_LOOKUP[raw_name]
    # Try removing -FC suffix and looking up again
    stripped = raw_name.replace("-FC", "")
    if stripped in BFCL_MODEL_NAME_LOOKUP:
        return BFCL_MODEL_NAME_LOOKUP[stripped]
    # Return as-is if not found (will be caught during processing)
    raise ValueError(f"BFCL model not in BFCL_MODEL_NAME_LOOKUP: {raw_name}")


def _extract_task_name_from_file(filepath: Path, category: str) -> str:
    """
    Extract task name from a BFCL score file path.
    
    Examples:
        agentic/BFCL_v4_web_search_base_score.json -> agentic_web_search_base
        agentic/memory/kv/BFCL_v4_memory_kv_score.json -> agentic_memory_kv
        live/BFCL_v4_live_simple_score.json -> live_simple
        non_live/BFCL_v4_simple_python_score.json -> non_live_simple_python
    """
    filename = filepath.name
    # Remove BFCL_v4_ prefix and _score.json suffix
    match = re.match(r"BFCL_v\d+_(.+)_score\.json", filename)
    if not match:
        return None
    
    base_task = match.group(1)
    
    # For agentic/memory tasks, we want to prefix with agentic_
    if "memory" in str(filepath.parent):
        return f"agentic_{base_task}"
    
    # For agentic web_search tasks
    if category == "agentic" and "web_search" in base_task:
        return f"agentic_{base_task}"
    
    # For live tasks, remove the 'live_' prefix from base_task since category is 'live'
    if category == "live" and base_task.startswith("live_"):
        return f"live_{base_task[5:]}"
    
    # For multi_turn tasks
    if category == "multi_turn" and base_task.startswith("multi_turn_"):
        return f"multi_turn_{base_task[11:]}"
    
    # For non_live tasks, the files don't have 'non_live_' prefix in filename
    if category == "non_live":
        return f"non_live_{base_task}"
    
    return f"{category}_{base_task}"


def _has_simple_id_pattern(test_category: str) -> bool:
    """
    Check if the test category uses simple ID patterns ({test_category}_{i}).
    
    Non-live tasks generally use simple patterns.
    Live, memory, and multi-turn tasks often use complex patterns with multiple indices.
    """
    # Tasks known to use simple ID patterns: {test_category}_{i}
    simple_pattern_tasks = {
        "irrelevance",  # non_live
        "multiple",
        "parallel",
        "parallel_multiple",
        "simple_python",
        "simple_java",
        "simple_javascript",
    }
    return test_category in simple_pattern_tasks


def _extract_test_category_from_filename(filename: str) -> Optional[str]:
    """
    Extract test_category from a BFCL score filename.
    
    E.g., "BFCL_v4_irrelevance_score.json" -> "irrelevance"
          "BFCL_v4_simple_python_score.json" -> "simple_python"
          "BFCL_v4_live_simple_score.json" -> "live_simple"
          "BFCL_v4_memory_kv_score.json" -> "memory_kv"
    """
    # Remove prefix and suffix
    name = filename
    if name.startswith("BFCL_v4_"):
        name = name[len("BFCL_v4_"):]
    if name.endswith("_score.json"):
        name = name[:-len("_score.json")]
    
    return name if name else None


def _process_score_file(filepath: Path) -> List[Tuple[str, float, str]]:
    """
    Process a BFCL score JSON file and extract per-instance results.
    
    IMPORTANT: The raw BFCL results only contain instances that models got WRONG.
    The first line has aggregate stats (accuracy, correct_count, total_count).
    Subsequent lines are the wrong instances.
    
    For tasks with simple ID patterns, we generate all IDs and mark correct/wrong.
    For tasks with complex ID patterns, we use index-based approach.
    
    Returns list of (id, score, metric_name) tuples.
    """
    lines = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                lines.append(data)
            except json.JSONDecodeError:
                continue
    
    if not lines:
        return []
    
    # First line should have aggregate stats
    first_line = lines[0]
    if "accuracy" not in first_line or "total_count" not in first_line:
        # Unexpected format, skip
        return []
    
    total_count = first_line.get("total_count", 0)
    correct_count = first_line.get("correct_count", 0)
    wrong_count = total_count - correct_count
    
    # Rest of lines are wrong instances
    wrong_instances = lines[1:]
    
    # Collect wrong instance IDs and test_category
    wrong_ids: Set[str] = set()
    test_category: Optional[str] = None
    
    for inst in wrong_instances:
        if "id" in inst:
            wrong_ids.add(str(inst["id"]))
        if test_category is None and "test_category" in inst:
            test_category = inst["test_category"]
    
    # If no wrong instances (100% accuracy), try to get test_category from filename
    if test_category is None:
        test_category = _extract_test_category_from_filename(filepath.name)
    
    if test_category is None:
        # Cannot determine test category
        return []
    
    results: List[Tuple[str, float, str]] = []
    
    # Check if this task uses simple ID patterns
    if _has_simple_id_pattern(test_category):
        # Generate all IDs: {test_category}_{i} for i in 0..total_count-1
        # Mark wrong ones (in file) as 0.0, correct ones (not in file) as 1.0
        for i in range(total_count):
            instance_id = f"{test_category}_{i}"
            if instance_id in wrong_ids:
                score = 0.0
            else:
                score = 1.0
            results.append((instance_id, score, "accuracy"))
    else:
        # Complex ID pattern - use index-based approach
        # Generate IDs as {test_category}_{i} and mark based on extracted indices
        
        # Extract numeric indices from wrong IDs
        wrong_indices: Set[int] = set()
        for wid in wrong_ids:
            # Try to extract the first numeric part after test_category_
            prefix = f"{test_category}_"
            if wid.startswith(prefix):
                rest = wid[len(prefix):]
                # Extract first number (before any hyphen or other separator)
                match = re.match(r"^(\d+)", rest)
                if match:
                    wrong_indices.add(int(match.group(1)))
        
        # Generate all IDs using simple format {test_category}_{i}
        # If we could extract indices from wrong IDs, use them to mark wrong
        # Otherwise (no wrong instances = 100% accuracy), all are correct
        for i in range(total_count):
            instance_id = f"{test_category}_{i}"
            if i in wrong_indices:
                score = 0.0
            else:
                score = 1.0
            results.append((instance_id, score, "accuracy"))
    
    return results


def _find_score_files(model_dir: Path) -> Dict[str, Path]:
    """
    Find all score JSON files in a model directory and map them to task names.
    
    Returns dict mapping task_name -> filepath.
    """
    task_files: Dict[str, Path] = {}
    
    # Process agentic directory
    agentic_dir = model_dir / "agentic"
    if agentic_dir.exists():
        # Direct files in agentic/
        for f in agentic_dir.glob("BFCL_v*_score.json"):
            task = _extract_task_name_from_file(f, "agentic")
            if task:
                task_files[task] = f
        
        # Memory subdirectory files
        memory_dir = agentic_dir / "memory"
        if memory_dir.exists():
            for subdir in memory_dir.iterdir():
                if subdir.is_dir():
                    for f in subdir.glob("BFCL_v*_score.json"):
                        task = _extract_task_name_from_file(f, "agentic")
                        if task:
                            task_files[task] = f
    
    # Process live, multi_turn, non_live directories
    for category in ["live", "multi_turn", "non_live"]:
        cat_dir = model_dir / category
        if cat_dir.exists():
            for f in cat_dir.glob("BFCL_v*_score.json"):
                task = _extract_task_name_from_file(f, category)
                if task:
                    task_files[task] = f
    
    return task_files


def standardize_bfcl(input_root: Path, output_root: Path, verbose: bool = True) -> Dict[str, Dict[str, int]]:
    """
    Standardize BFCL results from raw format to CSV format.
    
    Args:
        input_root: Path to results/raw_results/bfcl
        output_root: Path to results/standardized_results
        verbose: Whether to print progress information
    
    Returns:
        Statistics dict mapping task -> model -> entry count
    """
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    base_out = output_root / "bfcl"
    base_out.mkdir(parents=True, exist_ok=True)
    
    # Find model directories (exclude data_*.csv files)
    model_dirs = [d for d in input_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if verbose:
        print(f"Found {len(model_dirs)} model directories")
    
    for model_dir in sorted(model_dirs):
        raw_model = model_dir.name
        try:
            model_name = normalize_bfcl_model_name(raw_model)
        except ValueError as e:
            if verbose:
                print(f"Skipping unknown model: {raw_model}")
            continue
        
        if verbose:
            print(f"\nProcessing model: {raw_model} -> {model_name}")
        
        # Find all score files and their task names
        task_files = _find_score_files(model_dir)
        if not task_files and verbose:
            print("  No score files found")
            continue
        
        for task, filepath in sorted(task_files.items()):
            results = _process_score_file(filepath)
            if not results:
                if verbose:
                    print(f"  No results extracted from {filepath.name}")
                continue
            
            # Create output directory for this task
            out_dir = base_out / task
            out_dir.mkdir(parents=True, exist_ok=True)
            
            output_csv = out_dir / f"{model_name}.csv"
            write_csv(output_csv, results)
            stats[f"bfcl/{task}"][model_name] = len(results)
            if verbose:
                print(f"  Wrote bfcl/{task}/{output_csv.name} ({len(results)} entries)")
    
    return dict(stats)


def main() -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description="Standardize BFCL eval scores")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw_results/bfcl"),
        help="Root directory of bfcl model folders",
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
    
    stats = standardize_bfcl(input_dir, output_dir, verbose=args.verbose)
    print_summary(stats)
    
    print(f"\nDone! Standardized data written to: {output_dir / 'bfcl'}")
    return 0


if __name__ == "__main__":
    main()
