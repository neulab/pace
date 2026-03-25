#!/usr/bin/env python3
"""
Ability-weighted bootstrap instance selection for multi-benchmark prediction.

This script extends bootstrap_prediction_multibench.py with ability-based weighting:
1. Parse abilities.tex to get ability requirements for each benchmark
2. Compute ability overlap between source and target benchmarks
3. Bootstrap sample instances with probability proportional to ability overlap
4. Evaluate each bootstrap sample with Ridge regression MSE
5. Vote for instances in the best samples
6. Select top-voted instances

The key insight: benchmarks that share more abilities with the target should
contribute more instances, because they test similar underlying capabilities.

Example usage:
    python bootstrap_prediction_abilities.py \
        --sources bfcl livecodebench gpqa humaneval_chat \
        --targets gaia \
        --k_total 200 \
        --n_bootstrap 100
"""

import argparse
import json
import re
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Import utilities from bootstrap.py
from bootstrap import (
    STD_BASE,
    BenchDataset,
    discover_benchmark,
    dicts_to_matrix,
    compute_mean_over_indices,
)

# ==============================================================================
# Ability Parsing
# ==============================================================================

# Ability abbreviations from abilities.tex
ABILITY_ABBREVS = {
    "IF": "Instruction Following",
    "LCA": "Long Context Aggregation", 
    "ER": "Error Recovery",
    "Plan": "Planning",
    "Code": "Programming / Code Gen",
    "IR": "Information Retrieval",
    "CS": "Code Search",
    "TC": "Tool Calling",
    "Reas": "Reasoning",
    "MM": "Multimodal Understanding",
    "Ver": "Verification",
    "Test": "Test Understanding",
}

ABILITY_ORDER = ["IF", "LCA", "ER", "Plan", "Code", "IR", "CS", "TC", "Reas", "MM", "Ver", "Test"]


def parse_abilities_tex(filepath: Path) -> Dict[str, Set[str]]:
    """Parse abilities.tex to extract ability requirements for each benchmark.
    
    Returns:
        Dict mapping benchmark name -> set of ability abbreviations
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    benchmark_abilities: Dict[str, Set[str]] = {}
    
    # Find all table rows with benchmark data
    # Pattern: \textbf{BenchmarkName} & marker & marker & ...
    # where marker is either $\bullet$ (has ability) or empty
    
    lines = content.split('\n')
    for line in lines:
        # Skip header/section lines
        if '\\midrule' in line or '\\toprule' in line or '\\bottomrule' in line:
            continue
        if '\\multicolumn' in line:
            continue
        if '\\textbf{Benchmark}' in line:
            continue
            
        # Extract benchmark name
        match = re.search(r'\\textbf\{([^}]+)\}', line)
        if not match:
            continue
            
        bench_name = match.group(1)
        
        # Remove the benchmark name part and rowcolor if present
        rest = line.split('&')[1:]  # Skip first column (benchmark name)
        
        if len(rest) != 12:
            continue
            
        abilities = set()
        for i, cell in enumerate(rest):
            # Check if cell contains bullet
            if '$\\bullet$' in cell or r'$\bullet$' in cell:
                abilities.add(ABILITY_ORDER[i])
        
        # Normalize benchmark name for matching
        norm_name = normalize_benchmark_name(bench_name)
        benchmark_abilities[norm_name] = abilities
    
    return benchmark_abilities


def normalize_benchmark_name(name: str) -> str:
    """Normalize benchmark name for matching between abilities.tex and data."""
    name = name.lower()
    name = name.replace('-', '').replace('_', '').replace(' ', '')
    name = name.replace('(bashonly)', '')
    # Handle common variations
    name = name.replace('chat', '')  # humaneval_chat -> humaneval
    name = name.replace('multimodal', '')  # swebench_multimodal -> swebench (but keep distinction)
    return name


# Manual mapping for benchmarks with different names
BENCHMARK_NAME_ALIASES = {
    "humaneval_chat": "humaneval",
    "humaneval": "humaneval",
    "swebench_multimodal": "swebenchmultimodal",
    "swtbench": "swtbench",
}


def get_ability_overlap(
    source_abilities: Set[str], 
    target_abilities: Set[str]
) -> Tuple[int, int, float]:
    """Compute ability overlap between source and target.
    
    Returns:
        (overlap_count, target_count, overlap_ratio)
    """
    overlap = source_abilities & target_abilities
    overlap_count = len(overlap)
    target_count = len(target_abilities)
    ratio = overlap_count / target_count if target_count > 0 else 0.0
    return overlap_count, target_count, ratio


# ==============================================================================
# Ridge Regression (from bootstrap_prediction_multibench.py)
# ==============================================================================

def ridge_regression(
    X: np.ndarray, 
    y: np.ndarray, 
    alpha: float = 0.1
) -> Tuple[np.ndarray, float]:
    """Fit Ridge regression: y = X @ coef + intercept."""
    n, p = X.shape
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_c = X - X_mean
    y_c = y - y_mean
    
    # Ridge: (X'X + alpha*I)^{-1} X'y
    XtX = X_c.T @ X_c
    Xty = X_c.T @ y_c
    
    reg_matrix = XtX + alpha * np.eye(p)
    coef = np.linalg.solve(reg_matrix, Xty)
    intercept = y_mean - X_mean @ coef
    
    return coef, float(intercept)


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return float(((y_true - y_pred) ** 2).mean())


# ==============================================================================
# Ability-Weighted Bootstrap Selection
# ==============================================================================

@dataclass
class AbilityWeights:
    """Ability-based weights for each benchmark."""
    benchmark: str
    abilities: Set[str]
    overlap_count: int
    target_count: int
    overlap_ratio: float
    sampling_weight: float  # Normalized weight for sampling


def compute_ability_weights(
    source_bench_names: List[str],
    target_bench_names: List[str],
    abilities_data: Dict[str, Set[str]],
) -> Dict[str, AbilityWeights]:
    """Compute ability-based weights for each source benchmark."""
    
    # Get target abilities (union if multiple targets)
    target_abilities: Set[str] = set()
    for target in target_bench_names:
        norm_target = normalize_benchmark_name(target)
        if norm_target in abilities_data:
            target_abilities |= abilities_data[norm_target]
        else:
            print(f"  WARNING: No abilities found for target '{target}'")
    
    print(f"  Target abilities ({len(target_abilities)}): {sorted(target_abilities)}")
    
    # Compute weights for each source
    weights: Dict[str, AbilityWeights] = {}
    total_weight = 0.0
    
    for source in source_bench_names:
        norm_source = normalize_benchmark_name(source)
        
        if norm_source in abilities_data:
            source_abilities = abilities_data[norm_source]
        else:
            print(f"  WARNING: No abilities found for source '{source}', using empty set")
            source_abilities = set()
        
        overlap_count, target_count, overlap_ratio = get_ability_overlap(
            source_abilities, target_abilities
        )
        
        weights[source] = AbilityWeights(
            benchmark=source,
            abilities=source_abilities,
            overlap_count=overlap_count,
            target_count=target_count,
            overlap_ratio=overlap_ratio,
            sampling_weight=overlap_ratio,  # Will be normalized later
        )
        total_weight += overlap_ratio
        
        print(f"  {source}: {len(source_abilities)} abilities, "
              f"{overlap_count}/{target_count} overlap = {overlap_ratio:.2f}")
    
    # Normalize weights
    if total_weight > 0:
        for w in weights.values():
            w.sampling_weight = w.overlap_ratio / total_weight
    
    return weights


def ability_weighted_bootstrap_sample(
    bench_matrices: Dict[str, np.ndarray],
    ability_weights: Dict[str, AbilityWeights],
    k_total: int,
    rng: np.random.Generator,
) -> Dict[str, List[int]]:
    """Sample instances with ability-weighted probabilities.
    
    For each benchmark, sampling probability is proportional to:
    - Ability overlap ratio (how relevant is this benchmark)
    - Inverse of benchmark size (normalize by size)
    
    Args:
        bench_matrices: Dict[bench_name -> (n_models, n_instances) matrix]
        ability_weights: Ability-based weights for each benchmark
        k_total: Total number of instances to sample across all benchmarks
        rng: Random number generator
    
    Returns:
        Dict[bench_name -> list of selected instance indices]
    """
    selected: Dict[str, List[int]] = {}
    
    # Compute expected instances per benchmark
    # E[k_bench] = k_total * (weight * n_instances) / sum(weight_i * n_i)
    
    weighted_sizes = {}
    total_weighted_size = 0.0
    
    for bench_name, matrix in bench_matrices.items():
        n_instances = matrix.shape[1]
        weight = ability_weights[bench_name].overlap_ratio
        weighted_size = weight * n_instances
        weighted_sizes[bench_name] = weighted_size
        total_weighted_size += weighted_size
    
    if total_weighted_size == 0:
        # Fallback to uniform sampling
        for bench_name, matrix in bench_matrices.items():
            n_instances = matrix.shape[1]
            k_bench = k_total // len(bench_matrices)
            indices = rng.choice(n_instances, size=min(k_bench, n_instances), replace=False)
            selected[bench_name] = indices.tolist()
        return selected
    
    # Sample from each benchmark
    remaining = k_total
    bench_names = list(bench_matrices.keys())
    
    for i, bench_name in enumerate(bench_names):
        matrix = bench_matrices[bench_name]
        n_instances = matrix.shape[1]
        
        if i == len(bench_names) - 1:
            # Last benchmark: take remaining
            k_bench = remaining
        else:
            # Expected count for this benchmark
            expected = k_total * weighted_sizes[bench_name] / total_weighted_size
            # Add some randomness (Poisson-like)
            k_bench = int(rng.poisson(expected))
        
        k_bench = max(1, min(k_bench, n_instances, remaining))
        
        indices = rng.choice(n_instances, size=k_bench, replace=False)
        selected[bench_name] = indices.tolist()
        remaining -= k_bench
        
        if remaining <= 0:
            break
    
    # Fill in empty benchmarks
    for bench_name in bench_matrices:
        if bench_name not in selected:
            selected[bench_name] = []
    
    return selected


def evaluate_selection(
    bench_matrices: Dict[str, np.ndarray],
    selected: Dict[str, List[int]],
    y: np.ndarray,
    alpha: float = 0.1,
) -> Tuple[float, np.ndarray, float]:
    """Evaluate a selection with Ridge regression.
    
    Returns:
        (mse, coefficients, intercept)
    """
    # Build X matrix: (n_models, n_benchmarks)
    bench_names = sorted(bench_matrices.keys())
    n_models = y.shape[0]
    n_benchmarks = len(bench_names)
    
    X = np.zeros((n_models, n_benchmarks))
    
    for j, bench_name in enumerate(bench_names):
        indices = selected[bench_name]
        if len(indices) > 0:
            X[:, j] = bench_matrices[bench_name][:, indices].mean(axis=1)
        else:
            X[:, j] = 0.0
    
    # Fit Ridge regression
    coef, intercept = ridge_regression(X, y, alpha=alpha)
    
    # Compute predictions and MSE
    y_pred = X @ coef + intercept
    mse = compute_mse(y, y_pred)
    
    return mse, coef, intercept


def bootstrap_vote_selection(
    bench_matrices: Dict[str, np.ndarray],
    ability_weights: Dict[str, AbilityWeights],
    y: np.ndarray,
    k_total: int,
    n_bootstrap: int = 100,
    alpha: float = 0.1,
    top_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[Dict[str, List[int]], Dict[str, np.ndarray]]:
    """Bootstrap voting for instance selection.
    
    1. Generate n_bootstrap samples with ability-weighted sampling
    2. Evaluate each sample with Ridge regression MSE
    3. For top-performing samples, add votes to selected instances
    4. Return instances with most votes
    
    Args:
        bench_matrices: Dict[bench_name -> (n_models, n_instances) matrix]
        ability_weights: Ability-based weights for each benchmark
        y: Target scores (n_models,)
        k_total: Total instances to select
        n_bootstrap: Number of bootstrap iterations
        alpha: Ridge regularization
        top_fraction: Fraction of best samples to count votes from
        seed: Random seed
    
    Returns:
        (selected_indices, vote_counts)
    """
    rng = np.random.default_rng(seed)
    
    # Initialize vote counts
    vote_counts: Dict[str, np.ndarray] = {}
    for bench_name, matrix in bench_matrices.items():
        vote_counts[bench_name] = np.zeros(matrix.shape[1])
    
    # Store all samples and their MSEs
    samples = []
    mses = []
    
    print(f"  Running {n_bootstrap} bootstrap iterations...")
    
    for b in range(n_bootstrap):
        # Sample instances
        selected = ability_weighted_bootstrap_sample(
            bench_matrices, ability_weights, k_total, rng
        )
        
        # Evaluate
        mse, coef, intercept = evaluate_selection(
            bench_matrices, selected, y, alpha
        )
        
        samples.append(selected)
        mses.append(mse)
        
        if (b + 1) % 20 == 0:
            print(f"    Iteration {b+1}/{n_bootstrap}, best MSE so far: {min(mses):.6f}")
    
    # Find top samples
    n_top = max(1, int(n_bootstrap * top_fraction))
    top_indices = np.argsort(mses)[:n_top]
    
    print(f"  Top {n_top} samples: MSE range [{mses[top_indices[0]]:.6f}, {mses[top_indices[-1]]:.6f}]")
    
    # Count votes from top samples
    for idx in top_indices:
        selected = samples[idx]
        for bench_name, indices in selected.items():
            for i in indices:
                vote_counts[bench_name][i] += 1
    
    # Select top-voted instances
    final_selected: Dict[str, List[int]] = {}
    
    # Distribute k_total based on ability weights
    bench_names = sorted(bench_matrices.keys())
    total_weight = sum(ability_weights[b].overlap_ratio for b in bench_names)
    
    remaining = k_total
    for i, bench_name in enumerate(bench_names):
        if i == len(bench_names) - 1:
            k_bench = remaining
        else:
            weight = ability_weights[bench_name].overlap_ratio
            k_bench = int(k_total * weight / total_weight) if total_weight > 0 else k_total // len(bench_names)
        
        k_bench = max(1, min(k_bench, bench_matrices[bench_name].shape[1], remaining))
        
        # Select top-voted instances
        votes = vote_counts[bench_name]
        top_k = np.argsort(votes)[-k_bench:][::-1]
        final_selected[bench_name] = top_k.tolist()
        remaining -= k_bench
    
    return final_selected, vote_counts


# ==============================================================================
# Greedy Selection Utilities
# ==============================================================================

def greedy_select_instances_single_bench(
    A_bench: np.ndarray,
    y: np.ndarray,
    k: int,
    seed: int = 0,
) -> Tuple[List[int], float]:
    """Greedily select k instances from a single benchmark to minimize MSE.
    
    Uses simple linear regression: y = a * mean(selected_instances) + b
    """
    rng = np.random.default_rng(seed)
    M, N = A_bench.shape
    
    if k >= N:
        return list(range(N)), 0.0
    
    candidates = list(range(N))
    rng.shuffle(candidates)
    
    selected: List[int] = []
    in_set = np.zeros(N, dtype=bool)
    
    for _ in range(k):
        best_j = None
        best_mse = float('inf')
        
        for j in candidates:
            if in_set[j]:
                continue
            
            trial = selected + [j]
            x = A_bench[:, trial].mean(axis=1)
            
            # Simple linear regression
            x_mean, y_mean = x.mean(), y.mean()
            x_c, y_c = x - x_mean, y - y_mean
            var_x = (x_c ** 2).sum()
            
            if var_x < 1e-12:
                y_pred = np.full_like(y, y_mean)
            else:
                a = (x_c * y_c).sum() / var_x
                b = y_mean - a * x_mean
                y_pred = a * x + b
            
            mse = ((y - y_pred) ** 2).mean()
            
            if mse < best_mse:
                best_mse = mse
                best_j = j
        
        if best_j is None:
            break
        
        selected.append(best_j)
        in_set[best_j] = True
    
    return selected, best_mse


def greedy_select_all_benchmarks(
    bench_matrices: Dict[str, np.ndarray],
    y: np.ndarray,
    k_per_bench: Dict[str, int],
    seed: int = 0,
) -> Dict[str, List[int]]:
    """Greedily select instances from each benchmark independently."""
    selected = {}
    for name in sorted(bench_matrices.keys()):
        A_bench = bench_matrices[name]
        k = k_per_bench[name]
        indices, mse = greedy_select_instances_single_bench(A_bench, y, k, seed)
        selected[name] = indices
    return selected


# ==============================================================================
# TARGET Bootstrap + Greedy Source Selection + Voting
# ==============================================================================

def target_bootstrap_voting(
    bench_matrices: Dict[str, np.ndarray],
    target_matrix: np.ndarray,  # (n_models, n_target_instances)
    k_per_bench: Dict[str, int],
    n_bootstrap: int = 100,
    alpha: float = 0.1,
    seed: int = 0,
) -> Tuple[Dict[str, List[int]], Dict[str, np.ndarray]]:
    """Bootstrap TARGET instances, greedy select sources for each, then vote.
    
    Algorithm:
    1. For each bootstrap iteration:
       a. Sample subset of TARGET instances (with replacement)
       b. Compute y = mean of sampled target instances
       c. Greedily select source instances that best predict this y
       d. Evaluate MSE of this selection
    2. Vote: count how often each source instance appears in top selections
    3. Return top-voted instances
    
    This finds source instances that are ROBUSTLY predictive across
    different subsets of the target benchmark.
    """
    rng = np.random.default_rng(seed)
    bench_names = sorted(bench_matrices.keys())
    n_models = target_matrix.shape[0]
    n_target_instances = target_matrix.shape[1]
    
    # Initialize vote counts
    vote_counts: Dict[str, np.ndarray] = {}
    for name, matrix in bench_matrices.items():
        vote_counts[name] = np.zeros(matrix.shape[1])
    
    # Store results
    all_selections = []
    all_mses = []
    
    print(f"  Running {n_bootstrap} target bootstrap iterations...")
    
    for b in range(n_bootstrap):
        # Bootstrap sample TARGET instances
        sampled_indices = rng.choice(n_target_instances, size=n_target_instances, replace=True)
        y_bootstrap = target_matrix[:, sampled_indices].mean(axis=1)
        
        # Greedy select source instances for this target sample
        selected = greedy_select_all_benchmarks(
            bench_matrices, y_bootstrap, k_per_bench, seed=seed + b
        )
        
        # Evaluate: fit Ridge and compute MSE on the FULL target
        y_full = target_matrix.mean(axis=1)
        mse, _, _ = evaluate_selection(bench_matrices, selected, y_full, alpha)
        
        all_selections.append(selected)
        all_mses.append(mse)
        
        if (b + 1) % 20 == 0:
            print(f"    Iteration {b+1}/{n_bootstrap}, MSE range: [{min(all_mses):.6f}, {max(all_mses):.6f}]")
    
    # Vote: weight by inverse MSE (better fits get more votes)
    # Or simply count appearances in top fraction
    top_fraction = 0.2
    n_top = max(1, int(n_bootstrap * top_fraction))
    top_indices = np.argsort(all_mses)[:n_top]
    
    print(f"  Top {n_top} bootstrap samples: MSE range [{all_mses[top_indices[0]]:.6f}, {all_mses[top_indices[-1]]:.6f}]")
    
    for idx in top_indices:
        selected = all_selections[idx]
        for name, instances in selected.items():
            for i in instances:
                vote_counts[name][i] += 1
    
    # Select top-voted instances
    final_selected: Dict[str, List[int]] = {}
    for name in bench_names:
        k = k_per_bench[name]
        votes = vote_counts[name]
        top_k = np.argsort(votes)[-k:][::-1]
        final_selected[name] = top_k.tolist()
    
    return final_selected, vote_counts


# ==============================================================================
# Hybrid Method: Ability-weighted k allocation + Greedy MSE selection
# ==============================================================================

def ability_weighted_greedy_selection(
    bench_matrices: Dict[str, np.ndarray],
    ability_weights: Dict[str, AbilityWeights],
    y: np.ndarray,
    k_total: int,
    alpha: float = 0.1,
    seed: int = 0,
) -> Dict[str, List[int]]:
    """Hybrid method: ability-weighted k allocation + greedy MSE selection.
    
    1. Use ability weights to determine k_per_bench for each benchmark
    2. Within each benchmark, use greedy MSE optimization to select instances
    
    This combines:
    - Domain knowledge (abilities) for inter-benchmark allocation
    - Data-driven optimization for intra-benchmark selection
    """
    bench_names = sorted(bench_matrices.keys())
    
    # Compute k allocation based on ability weights
    total_weight = sum(ability_weights[b].overlap_ratio for b in bench_names)
    
    # Handle zero total weight
    if total_weight < 1e-12:
        # Fall back to equal allocation
        k_per_bench = {name: k_total // len(bench_names) for name in bench_names}
    else:
        k_per_bench = {}
        remaining = k_total
        
        for i, name in enumerate(bench_names):
            if i == len(bench_names) - 1:
                k_per_bench[name] = remaining
            else:
                weight = ability_weights[name].overlap_ratio
                k = int(k_total * weight / total_weight)
                k = max(1, min(k, bench_matrices[name].shape[1], remaining))
                k_per_bench[name] = k
                remaining -= k
    
    print(f"  Ability-weighted k allocation:")
    for name in bench_names:
        weight = ability_weights[name].overlap_ratio
        print(f"    {name}: k={k_per_bench[name]} (weight={weight:.2f})")
    
    # Greedy selection within each benchmark
    selected: Dict[str, List[int]] = {}
    
    print(f"  Greedy MSE selection within each benchmark:")
    for name in bench_names:
        A_bench = bench_matrices[name]
        k = k_per_bench[name]
        
        indices, mse = greedy_select_instances_single_bench(A_bench, y, k, seed)
        selected[name] = indices
        print(f"    {name}: selected {len(indices)} instances, MSE={mse:.6f}")
    
    return selected


# ==============================================================================
# Data Loading
# ==============================================================================

def load_benchmark(name: str, base_dir: Path) -> Tuple[BenchDataset, np.ndarray, List[str]]:
    """Load a benchmark and return (dataset, matrix, instance_ids)."""
    bench_full, _ = discover_benchmark(name, base_dir=base_dir)
    models = bench_full.list_models()
    kept, dicts = bench_full.load_model_outputs_for_models(models)
    A, ids = dicts_to_matrix(dicts)
    return bench_full, A, ids


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_ability_weighted_pipeline(
    source_bench_names: List[str],
    target_bench_names: List[str],
    abilities_file: Path,
    base_dir: Path,
    output_dir: Path,
    train_models: Optional[List[str]] = None,
    eval_models: Optional[List[str]] = None,
    k_total: int = 200,
    method: str = "hybrid",  # "hybrid", "bootstrap", or "greedy_equal"
    n_bootstrap: int = 100,
    top_fraction: float = 0.1,
    target_train_pct: float = 0.8,
    target_eval_pct: float = 0.2,
    ridge_alpha: float = 0.1,
    split_seed: int = 0,
    select_seed: int = 0,
    annotate: bool = True,
) -> Dict:
    """Run the ability-weighted bootstrap pipeline."""
    
    print("=" * 70)
    print("ABILITY-WEIGHTED PREDICTION PIPELINE")
    print("=" * 70)
    print(f"Sources: {source_bench_names}")
    print(f"Targets: {target_bench_names}")
    print(f"Method: {method}")
    print(f"k_total: {k_total}")
    if method == "bootstrap":
        print(f"n_bootstrap: {n_bootstrap}")
    print(f"Ridge alpha: {ridge_alpha}")
    
    # 1) Parse abilities
    print("\n[1] Parsing abilities...")
    abilities_data = parse_abilities_tex(abilities_file)
    print(f"  Found abilities for {len(abilities_data)} benchmarks")
    
    # 2) Compute ability weights
    print("\n[2] Computing ability weights...")
    ability_weights = compute_ability_weights(
        source_bench_names, target_bench_names, abilities_data
    )
    
    # 3) Load benchmarks
    print("\n[3] Loading benchmarks...")
    source_benchmarks = {}
    for name in source_bench_names:
        bench, A, ids = load_benchmark(name, base_dir)
        source_benchmarks[name] = {"bench": bench, "matrix": A, "ids": ids}
        print(f"  Source {name}: {A.shape[0]} models, {A.shape[1]} instances")
    
    target_benchmarks = {}
    for name in target_bench_names:
        bench, A, ids = load_benchmark(name, base_dir)
        target_benchmarks[name] = {"bench": bench, "matrix": A, "ids": ids}
        print(f"  Target {name}: {A.shape[0]} models, {A.shape[1]} instances")
    
    # 4) Find common models
    all_source_models = [set(source_benchmarks[n]["bench"].list_models()) for n in source_bench_names]
    all_target_models = [set(target_benchmarks[n]["bench"].list_models()) for n in target_bench_names]
    
    common_models = set.intersection(*all_source_models, *all_target_models)
    
    if train_models is not None and eval_models is not None:
        train_model_set = set(train_models) & common_models
        eval_model_set = set(eval_models) & common_models
        train_model_list = sorted(train_model_set)
        eval_model_list = sorted(eval_model_set)
    else:
        all_models = sorted(common_models)
        train_model_list = all_models
        eval_model_list = all_models
    
    print(f"\n[4] Models:")
    print(f"  Train models ({len(train_model_list)}): {train_model_list}")
    print(f"  Eval models ({len(eval_model_list)}): {eval_model_list}")
    
    # 5) Build matrices for train/eval models
    print("\n[5] Building matrices...")
    
    def get_model_indices(bench, model_list):
        all_models = bench.list_models()
        return [all_models.index(m) for m in model_list if m in all_models]
    
    train_bench_matrices: Dict[str, np.ndarray] = {}
    eval_bench_matrices: Dict[str, np.ndarray] = {}
    bench_instance_ids: Dict[str, List[str]] = {}
    
    for name in source_bench_names:
        bench = source_benchmarks[name]["bench"]
        A = source_benchmarks[name]["matrix"]
        ids = source_benchmarks[name]["ids"]
        
        train_idx = get_model_indices(bench, train_model_list)
        eval_idx = get_model_indices(bench, eval_model_list)
        
        train_bench_matrices[name] = A[train_idx, :]
        eval_bench_matrices[name] = A[eval_idx, :]
        bench_instance_ids[name] = ids
        
        print(f"  {name}: train {train_bench_matrices[name].shape}, eval {eval_bench_matrices[name].shape}")
    
    # 6) Build target vector
    print("\n[6] Building target vector...")
    
    # Stack target matrices and average
    target_matrices_train = []
    target_matrices_eval = []
    
    for name in target_bench_names:
        bench = target_benchmarks[name]["bench"]
        A = target_benchmarks[name]["matrix"]
        
        train_idx = get_model_indices(bench, train_model_list)
        eval_idx = get_model_indices(bench, eval_model_list)
        
        target_matrices_train.append(A[train_idx, :])
        target_matrices_eval.append(A[eval_idx, :])
    
    # For target: split instances into train/eval
    A_tgt_train = np.hstack(target_matrices_train)
    A_tgt_eval = np.hstack(target_matrices_eval)
    
    n_tgt_instances = A_tgt_train.shape[1]
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_tgt_instances)
    
    target_train_size = int(n_tgt_instances * target_train_pct)
    target_eval_size = n_tgt_instances - target_train_size
    
    tgt_train_cols = perm[:target_train_size]
    tgt_eval_cols = perm[target_train_size:]
    
    y_train = A_tgt_train[:, tgt_train_cols].mean(axis=1)
    y_eval = A_tgt_eval[:, tgt_eval_cols].mean(axis=1)
    
    print(f"  Target train: {y_train.shape}, eval: {y_eval.shape}")
    
    # 7) Instance selection based on method
    print(f"\n[7] Instance selection (method={method})...")
    
    # Compute k allocation
    bench_names = sorted(train_bench_matrices.keys())
    if method in ["hybrid", "target_bootstrap"]:
        # Ability-weighted k allocation
        total_weight = sum(ability_weights[b].overlap_ratio for b in bench_names)
        if total_weight < 1e-12:
            k_per_bench = {name: k_total // len(bench_names) for name in bench_names}
        else:
            k_per_bench = {}
            remaining = k_total
            for i, name in enumerate(bench_names):
                if i == len(bench_names) - 1:
                    k_per_bench[name] = remaining
                else:
                    weight = ability_weights[name].overlap_ratio
                    k = int(k_total * weight / total_weight)
                    k = max(1, min(k, train_bench_matrices[name].shape[1], remaining))
                    k_per_bench[name] = k
                    remaining -= k
        print(f"  Ability-weighted k allocation:")
        for name in bench_names:
            print(f"    {name}: k={k_per_bench[name]} (weight={ability_weights[name].overlap_ratio:.2f})")
    else:
        # Equal k allocation
        k_per_bench = {name: k_total // len(bench_names) for name in bench_names}
        print(f"  Equal k allocation: {k_total // len(bench_names)} per benchmark")
    
    if method == "target_bootstrap":
        # NEW: Bootstrap TARGET instances, greedy select sources, then vote
        selected, vote_counts = target_bootstrap_voting(
            train_bench_matrices,
            A_tgt_train[:, tgt_train_cols],  # Target train matrix
            k_per_bench=k_per_bench,
            n_bootstrap=n_bootstrap,
            alpha=ridge_alpha,
            seed=select_seed,
        )
        for name, indices in selected.items():
            max_votes = vote_counts[name].max() if len(vote_counts[name]) > 0 else 0
            print(f"  {name}: {len(indices)} instances selected (max votes: {max_votes:.0f})")
    
    elif method == "bootstrap":
        # OLD: Bootstrap source instances randomly (ability-weighted)
        selected, vote_counts = bootstrap_vote_selection(
            train_bench_matrices,
            ability_weights,
            y_train,
            k_total=k_total,
            n_bootstrap=n_bootstrap,
            alpha=ridge_alpha,
            top_fraction=top_fraction,
            seed=select_seed,
        )
        for name, indices in selected.items():
            max_votes = vote_counts[name].max() if len(vote_counts[name]) > 0 else 0
            print(f"  {name}: {len(indices)} instances selected (max votes: {max_votes:.0f})")
    
    elif method == "hybrid":
        # Ability-weighted k + greedy MSE (no bootstrap)
        selected = ability_weighted_greedy_selection(
            train_bench_matrices,
            ability_weights,
            y_train,
            k_total=k_total,
            alpha=ridge_alpha,
            seed=select_seed,
        )
        for name, indices in selected.items():
            print(f"  {name}: {len(indices)} instances selected")
    
    elif method == "greedy_equal":
        # Baseline: equal k + greedy MSE
        selected = greedy_select_all_benchmarks(
            train_bench_matrices, y_train, k_per_bench, seed=select_seed
        )
        for name, indices in selected.items():
            print(f"  {name}: {len(indices)} instances selected")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 8) Final evaluation
    print("\n[8] Final evaluation...")
    
    # Training fit
    train_mse, train_coef, train_intercept = evaluate_selection(
        train_bench_matrices, selected, y_train, ridge_alpha
    )
    
    bench_names = sorted(train_bench_matrices.keys())
    X_train = np.zeros((len(train_model_list), len(bench_names)))
    for j, name in enumerate(bench_names):
        indices = selected[name]
        if len(indices) > 0:
            X_train[:, j] = train_bench_matrices[name][:, indices].mean(axis=1)
    
    y_train_pred = X_train @ train_coef + train_intercept
    
    # Eval
    X_eval = np.zeros((len(eval_model_list), len(bench_names)))
    for j, name in enumerate(bench_names):
        indices = selected[name]
        if len(indices) > 0:
            X_eval[:, j] = eval_bench_matrices[name][:, indices].mean(axis=1)
    
    y_eval_pred = X_eval @ train_coef + train_intercept
    eval_mse = compute_mse(y_eval, y_eval_pred)
    eval_mae = float(np.abs(y_eval - y_eval_pred).mean())
    
    ss_res = ((y_eval - y_eval_pred) ** 2).sum()
    ss_tot = ((y_eval - y_eval.mean()) ** 2).sum()
    eval_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    
    # Correlations
    def pearson_corr(x, y):
        return float(np.corrcoef(x, y)[0, 1])
    
    def spearman_corr(x, y):
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).correlation)
    
    eval_pearson = pearson_corr(y_eval_pred, y_eval)
    eval_spearman = spearman_corr(y_eval_pred, y_eval)
    
    print(f"\n  Per-benchmark coefficients:")
    for i, name in enumerate(bench_names):
        weight = ability_weights[name].overlap_ratio
        print(f"    {name}: coef={train_coef[i]:.4f}, ability_weight={weight:.2f}")
    print(f"    intercept: {train_intercept:.4f}")
    
    print(f"\n  Eval metrics:")
    print(f"    MSE={eval_mse:.6f}, MAE={eval_mae:.6f}, R²={eval_r2:.4f}")
    print(f"    Pearson={eval_pearson:.4f}, Spearman={eval_spearman:.4f}")
    
    # 9) Save outputs
    print("\n[9] Saving outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tag = "+".join(source_bench_names) + "_TO_" + "+".join(target_bench_names)
    
    # Compute error bars
    y_errors = []
    for i in range(len(eval_model_list)):
        model_scores = A_tgt_eval[i, tgt_eval_cols]
        se = model_scores.std() / np.sqrt(len(tgt_eval_cols))
        y_errors.append(se)
    y_errors = np.array(y_errors)
    
    x_errors = np.zeros(len(eval_model_list))  # Simplified
    
    # Scatter plot with CI
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Regression line
    pred_mean = y_eval_pred.mean()
    actual_mean = y_eval.mean()
    slope = np.sum((y_eval_pred - pred_mean) * (y_eval - actual_mean)) / np.sum((y_eval_pred - pred_mean)**2)
    intercept_fit = actual_mean - slope * pred_mean
    
    x_line = np.linspace(y_eval_pred.min() - 0.05, y_eval_pred.max() + 0.05, 100)
    y_line = slope * x_line + intercept_fit
    
    # Confidence interval
    n = len(y_eval_pred)
    residuals = y_eval - (slope * y_eval_pred + intercept_fit)
    se_residuals = np.sqrt(np.sum(residuals**2) / (n - 2)) if n > 2 else 0.1
    x_var = np.sum((y_eval_pred - pred_mean)**2)
    se_line = se_residuals * np.sqrt(1/n + (x_line - pred_mean)**2 / x_var) if x_var > 0 else 0.1
    
    t_val = stats.t.ppf(0.975, max(1, n - 2))
    ci_upper = y_line + t_val * se_line
    ci_lower = y_line - t_val * se_line
    
    ax.fill_between(x_line, ci_lower, ci_upper, alpha=0.2, color='steelblue',
                    label='95% Confidence Interval')
    ax.plot(x_line, y_line, 'b-', linewidth=2, alpha=0.8, label=f'Fitted (slope={slope:.2f})')
    
    ax.errorbar(y_eval_pred, y_eval,
                xerr=x_errors, yerr=y_errors,
                fmt='o', markersize=8, alpha=0.7,
                capsize=3, capthick=1, elinewidth=1,
                ecolor='gray', markerfacecolor='steelblue', markeredgecolor='darkblue',
                label='Models')
    
    if annotate:
        for i, m in enumerate(eval_model_list):
            ax.annotate(m, (y_eval_pred[i], y_eval[i]),
                       fontsize=7, alpha=0.8,
                       xytext=(5, 5), textcoords='offset points')
    
    min_val = min(x_line.min(), y_eval.min() - 0.05)
    max_val = max(x_line.max(), y_eval.max() + 0.05)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=1.5,
            label='Perfect prediction')
    
    ax.legend(loc='upper left')
    ax.set_xlabel("Predicted (from ability-weighted selection)")
    ax.set_ylabel(f"Actual ({', '.join(target_bench_names)})")
    ax.set_title(f"Ability-Weighted Multi-Benchmark Prediction\n"
                 f"MSE={eval_mse:.4f}, R²={eval_r2:.4f}, Pearson={eval_pearson:.4f}")
    
    fig.tight_layout()
    fig.savefig(output_dir / f"{tag}_ability_scatter.png", dpi=150)
    plt.close(fig)
    
    # Coefficient + ability weight comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x_pos = np.arange(len(bench_names))
    
    # Learned coefficients
    ax1 = axes[0]
    bars1 = ax1.bar(x_pos, train_coef)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bench_names, rotation=45, ha='right')
    ax1.set_ylabel("Learned Coefficient")
    ax1.set_title("Ridge Regression Coefficients")
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Ability weights
    ax2 = axes[1]
    ability_ratios = [ability_weights[name].overlap_ratio for name in bench_names]
    bars2 = ax2.bar(x_pos, ability_ratios, color='orange')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bench_names, rotation=45, ha='right')
    ax2.set_ylabel("Ability Overlap Ratio")
    ax2.set_title("Ability-Based Weights (overlap / target_abilities)")
    
    fig.tight_layout()
    fig.savefig(output_dir / f"{tag}_ability_coefficients.png", dpi=150)
    plt.close(fig)
    
    # JSON results
    selected_ids = {}
    for bench_name, indices in selected.items():
        ids = bench_instance_ids[bench_name]
        selected_ids[bench_name] = [ids[i] for i in indices]
    
    # Instance-level predictive power (vote counts) - only for bootstrap methods
    instance_predictive_power = {}
    if method in ["target_bootstrap", "bootstrap"] and 'vote_counts' in dir():
        for bench_name, votes in vote_counts.items():
            ids = bench_instance_ids[bench_name]
            # Get top instances by vote count
            top_indices = np.argsort(votes)[::-1]
            instance_predictive_power[bench_name] = {
                "max_votes": int(votes.max()),
                "mean_votes": float(votes.mean()),
                "top_instances": [
                    {"id": ids[i], "votes": int(votes[i]), "index": int(i)}
                    for i in top_indices[:20]  # Top 20 per benchmark
                    if votes[i] > 0
                ]
            }
    
    results = {
        "title": tag,
        "method": method,
        "source_benchmarks": source_bench_names,
        "target_benchmarks": target_bench_names,
        "ability_weights": {
            name: {
                "abilities": sorted(w.abilities),
                "overlap_count": w.overlap_count,
                "target_count": w.target_count,
                "overlap_ratio": w.overlap_ratio,
            }
            for name, w in ability_weights.items()
        },
        "params": {
            "k_total": k_total,
            "method": method,
            "n_bootstrap": n_bootstrap if method in ["bootstrap", "target_bootstrap"] else None,
            "top_fraction": top_fraction if method == "bootstrap" else None,
            "ridge_alpha": ridge_alpha,
            "target_train_pct": target_train_pct,
            "target_eval_pct": target_eval_pct,
        },
        "model_formula": {
            "benchmark_order": bench_names,
            "coefficients": [float(c) for c in train_coef],
            "intercept": float(train_intercept),
        },
        "metrics_eval": {
            "mse": float(eval_mse),
            "mae": float(eval_mae),
            "r_squared": float(eval_r2),
            "pearson": float(eval_pearson),
            "spearman": float(eval_spearman),
        },
        "selected_instances": {
            "total": sum(len(v) for v in selected_ids.values()),
            "by_benchmark": {k: len(v) for k, v in selected_ids.items()},
            "ids": selected_ids,
        },
        "instance_predictive_power": instance_predictive_power,
    }
    
    out_json = output_dir / f"{tag}_ability_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Saved: {out_json}")
    print(f"  Saved: {output_dir / f'{tag}_ability_scatter.png'}")
    print(f"  Saved: {output_dir / f'{tag}_ability_coefficients.png'}")
    
    return results


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ability-weighted instance selection with multiple methods"
    )
    parser.add_argument("--sources", nargs="+", required=True,
                        help="Source benchmark names")
    parser.add_argument("--targets", nargs="+", required=True,
                        help="Target benchmark names")
    parser.add_argument("--abilities_file", type=str, 
                        default=str(Path(__file__).parent.parent.parent / "abilities.tex"),
                        help="Path to abilities.tex")
    parser.add_argument("--base_dir", type=str, default=str(STD_BASE),
                        help="Base directory for standardized results")
    parser.add_argument("--output_dir", type=str, default="./analysis/ability_weighted",
                        help="Output directory")
    parser.add_argument("--train_models", nargs="+", default=None)
    parser.add_argument("--eval_models", nargs="+", default=None)
    parser.add_argument("--k_total", type=int, default=200,
                        help="Total instances to select across all benchmarks")
    parser.add_argument("--method", type=str, default="target_bootstrap",
                        choices=["target_bootstrap", "hybrid", "bootstrap", "greedy_equal"],
                        help="Selection method: "
                             "target_bootstrap (bootstrap target + greedy source + vote), "
                             "hybrid (ability k + greedy), "
                             "bootstrap (random source sampling + voting), "
                             "greedy_equal (equal k + greedy, baseline)")
    parser.add_argument("--n_bootstrap", type=int, default=100,
                        help="Number of bootstrap iterations (for bootstrap method)")
    parser.add_argument("--top_fraction", type=float, default=0.1,
                        help="Fraction of top samples to count votes from (for bootstrap)")
    parser.add_argument("--train_pct", type=float, default=0.8)
    parser.add_argument("--eval_pct", type=float, default=0.2)
    parser.add_argument("--ridge_alpha", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--select_seed", type=int, default=0)
    parser.add_argument("--annotate", action="store_true")
    
    args = parser.parse_args()
    
    try:
        results = run_ability_weighted_pipeline(
            source_bench_names=args.sources,
            target_bench_names=args.targets,
            abilities_file=Path(args.abilities_file),
            base_dir=Path(args.base_dir),
            output_dir=Path(args.output_dir),
            train_models=args.train_models,
            eval_models=args.eval_models,
            k_total=args.k_total,
            method=args.method,
            n_bootstrap=args.n_bootstrap,
            top_fraction=args.top_fraction,
            target_train_pct=args.train_pct,
            target_eval_pct=args.eval_pct,
            ridge_alpha=args.ridge_alpha,
            split_seed=args.split_seed,
            select_seed=args.select_seed,
            annotate=args.annotate,
        )
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\nAbility Weights:")
        for name, w in results["ability_weights"].items():
            print(f"  {name}: {w['overlap_count']}/{w['target_count']} = {w['overlap_ratio']:.2f}")
        
        print(f"\nPrediction Formula:")
        formula_parts = [f"{results['model_formula']['intercept']:.4f}"]
        for name, coef in zip(results['model_formula']['benchmark_order'],
                              results['model_formula']['coefficients']):
            formula_parts.append(f"{coef:+.4f}*{name}")
        print(f"  {' + '.join(args.targets)} = {' '.join(formula_parts)}")
        
        print(f"\nEval Metrics:")
        print(f"  MSE: {results['metrics_eval']['mse']:.6f}")
        print(f"  R²:  {results['metrics_eval']['r_squared']:.4f}")
        print(f"  Pearson: {results['metrics_eval']['pearson']:.4f}")
        
        print(f"\nSelected {results['selected_instances']['total']} instances:")
        for name, count in results['selected_instances']['by_benchmark'].items():
            print(f"  {name}: {count}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
