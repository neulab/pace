#!/usr/bin/env python3
"""
Run bootstrap pipeline with multi-benchmark linear regression.

This script extends bootstrap_prediction.py to use per-benchmark coefficients:
    y = b + a₀*x₀ + a₁*x₁ + a₂*x₂ + ...
    
where each xᵢ is the average score on benchmark i.

This provides:
1. Instance-level selection: Find predictive instances within each benchmark
2. Benchmark-level weighting: Learn different weights for each benchmark

Uses Ridge regression to prevent overfitting when #benchmarks approaches #models.

Example usage:
    python bootstrap_prediction_multibench.py \\
        --sources bfcl livecodebench gpqa humaneval_chat \\
        --targets swebench \\
        --train_pct 0.8 --eval_pct 0.2 \\
        --k_per_bench 50
"""

import argparse
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    pearson_corr,
    spearman_corr,
)


# ==============================================================================
# Multi-Benchmark Linear Regression
# ==============================================================================

def ridge_regression(X: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> Tuple[np.ndarray, float]:
    """Fit Ridge regression: y = X @ coef + intercept.
    
    Args:
        X: (n_samples, n_features) design matrix
        y: (n_samples,) target values
        alpha: Regularization strength
    
    Returns:
        (coef, intercept) where coef is (n_features,)
    """
    n_samples, n_features = X.shape
    
    # Center the data
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_centered = X - X_mean
    y_centered = y - y_mean
    
    # Ridge solution: coef = (X'X + alpha*I)^(-1) X'y
    XtX = X_centered.T @ X_centered
    Xty = X_centered.T @ y_centered
    
    # Add regularization
    reg_matrix = alpha * np.eye(n_features)
    coef = np.linalg.solve(XtX + reg_matrix, Xty)
    
    # Compute intercept
    intercept = y_mean - X_mean @ coef
    
    return coef, float(intercept)


def mse_with_multibench_fit(
    X: np.ndarray, 
    y: np.ndarray, 
    alpha: float = 0.1,
    diversity_weight: float = 0.0,
) -> Tuple[float, np.ndarray, float, float]:
    """Compute MSE with multi-benchmark Ridge regression.
    
    Args:
        X: (n_models, n_benchmarks) - each column is avg score on one benchmark
        y: (n_models,) - target scores
        alpha: Ridge regularization strength
        diversity_weight: Weight for diversity term (higher = more spread)
    
    Returns:
        (objective, coef, intercept, mse) where objective = mse - diversity_weight * var(source)
    """
    coef, intercept = ridge_regression(X, y, alpha=alpha)
    y_pred = X @ coef + intercept
    mse = float(((y - y_pred) ** 2).mean())
    
    # Diversity term: variance of SOURCE scores (weighted combination)
    # This measures how much models differ on the selected instances
    weighted_source = X @ np.abs(coef)  # Weight each benchmark by its importance
    source_var = float(weighted_source.var())
    
    # Combined objective: minimize MSE, maximize source variance
    objective = mse - diversity_weight * source_var
    
    return objective, coef, intercept, mse


def mae_with_multibench_fit(
    X: np.ndarray, 
    y: np.ndarray, 
    alpha: float = 0.1
) -> float:
    """Compute MAE with multi-benchmark Ridge regression."""
    coef, intercept = ridge_regression(X, y, alpha=alpha)
    y_pred = X @ coef + intercept
    return float(np.abs(y - y_pred).mean())


def r_squared_multibench(
    X: np.ndarray, 
    y: np.ndarray, 
    alpha: float = 0.1
) -> float:
    """Compute R² for multi-benchmark Ridge regression."""
    coef, intercept = ridge_regression(X, y, alpha=alpha)
    y_pred = X @ coef + intercept
    
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    
    return float(1.0 - ss_res / ss_tot)


# ==============================================================================
# Per-Benchmark Instance Selection
# ==============================================================================

def select_instances_per_benchmark_greedy(
    A_bench: np.ndarray,
    y: np.ndarray,
    k: int,
    seed: int = 0,
) -> Tuple[List[int], float]:
    """Greedily select k instances from a single benchmark to minimize MSE.
    
    Uses simple linear regression y = a*x + b where x is mean of selected instances.
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
                mse = ((y - y_mean) ** 2).mean()
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


def select_instances_all_benchmarks(
    bench_matrices: Dict[str, np.ndarray],
    y: np.ndarray,
    k_per_bench: int,
    seed: int = 0,
) -> Dict[str, List[int]]:
    """Select k instances from each benchmark independently."""
    selected = {}
    
    for bench_name, A_bench in bench_matrices.items():
        sel, mse = select_instances_per_benchmark_greedy(
            A_bench, y, k=k_per_bench, seed=seed
        )
        selected[bench_name] = sel
        print(f"    {bench_name}: selected {len(sel)} instances, MSE={mse:.6f}")
    
    return selected


def select_instances_per_benchmark_diverse(
    A_bench: np.ndarray,
    y: np.ndarray,
    k: int,
    diversity_weight: float = 0.0,
    seed: int = 0,
) -> Tuple[List[int], float]:
    """Greedily select k instances with diversity-aware objective.
    
    Objective: MSE - diversity_weight * variance(source_scores)
    
    The key insight: select instances where MODELS DIFFER in performance.
    This spreads out the x-axis (source scores), which helps spread predictions.
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
        best_obj = float('inf')
        
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
                mse = ((y - y_mean) ** 2).mean()
                source_var = 0.0
            else:
                a = (x_c * y_c).sum() / var_x
                b = y_mean - a * x_mean
                y_pred = a * x + b
                mse = ((y - y_pred) ** 2).mean()
                # Variance of SOURCE scores (x), not predictions
                source_var = x.var()
            
            # Combined objective: minimize MSE, maximize source variance
            # Higher source variance = models are more differentiated
            obj = mse - diversity_weight * source_var
            
            if obj < best_obj:
                best_obj = obj
                best_j = j
        
        if best_j is None:
            break
        
        selected.append(best_j)
        in_set[best_j] = True
    
    return selected, best_obj


def select_instances_all_benchmarks_diverse(
    bench_matrices: Dict[str, np.ndarray],
    y: np.ndarray,
    k_per_bench: int,
    diversity_weight: float = 0.0,
    seed: int = 0,
) -> Dict[str, List[int]]:
    """Select k instances from each benchmark with diversity-aware objective."""
    selected = {}
    
    for bench_name, A_bench in bench_matrices.items():
        sel, obj = select_instances_per_benchmark_diverse(
            A_bench, y, k=k_per_bench, diversity_weight=diversity_weight, seed=seed
        )
        selected[bench_name] = sel
        print(f"    {bench_name}: selected {len(sel)} instances, objective={obj:.6f}")
    
    return selected


# ==============================================================================
# Joint Optimization: Select + Weight
# ==============================================================================

def compute_bench_scores_from_selection(
    bench_matrices: Dict[str, np.ndarray],
    selected: Dict[str, List[int]],
) -> np.ndarray:
    """Compute per-benchmark mean scores for selected instances.
    
    Returns:
        X: (n_models, n_benchmarks) matrix
    """
    bench_names = sorted(bench_matrices.keys())
    n_models = list(bench_matrices.values())[0].shape[0]
    n_bench = len(bench_names)
    
    X = np.zeros((n_models, n_bench))
    
    for i, bench_name in enumerate(bench_names):
        A = bench_matrices[bench_name]
        sel = selected[bench_name]
        if len(sel) > 0:
            X[:, i] = A[:, sel].mean(axis=1)
        else:
            X[:, i] = 0.0
    
    return X


def joint_objective_of_selection(
    bench_matrices: Dict[str, np.ndarray],
    selected: Dict[str, List[int]],
    y: np.ndarray,
    alpha: float = 0.1,
    diversity_weight: float = 0.0,
) -> float:
    """Compute objective (MSE - diversity*variance) for given selection."""
    X = compute_bench_scores_from_selection(bench_matrices, selected)
    objective, _, _, _ = mse_with_multibench_fit(X, y, alpha=alpha, diversity_weight=diversity_weight)
    return objective


def iterative_selection_refinement(
    bench_matrices: Dict[str, np.ndarray],
    y: np.ndarray,
    k_per_bench: int,
    n_iters: int = 5,
    alpha: float = 0.1,
    diversity_weight: float = 0.0,
    seed: int = 0,
) -> Dict[str, List[int]]:
    """Iteratively refine instance selection using multi-benchmark objective.
    
    1. Initial selection: greedy per-benchmark (independent)
    2. Refinement: for each benchmark, re-select instances considering
       the multi-benchmark objective (MSE - diversity*variance)
    
    Args:
        diversity_weight: Higher values encourage more spread in predictions
    """
    rng = np.random.default_rng(seed)
    bench_names = sorted(bench_matrices.keys())
    
    # Initial independent selection (uses diversity-aware objective)
    print("  Initial per-benchmark selection:")
    selected = select_instances_all_benchmarks_diverse(
        bench_matrices, y, k_per_bench, diversity_weight, seed
    )
    
    init_obj = joint_objective_of_selection(bench_matrices, selected, y, alpha, diversity_weight)
    print(f"  Initial joint objective: {init_obj:.6f} (diversity_weight={diversity_weight})")
    
    # Iterative refinement
    for it in range(n_iters):
        improved = False
        
        for bench_name in bench_names:
            A_bench = bench_matrices[bench_name]
            N = A_bench.shape[1]
            current_sel = selected[bench_name]
            
            # Try swapping each selected instance with unselected ones
            unselected = [j for j in range(N) if j not in current_sel]
            
            if len(unselected) == 0:
                continue
            
            # Sample subset of unselected to try
            sample_size = min(100, len(unselected))
            try_unsel = rng.choice(unselected, size=sample_size, replace=False).tolist()
            
            for pos in range(len(current_sel)):
                old_j = current_sel[pos]
                best_swap = None
                best_obj = joint_objective_of_selection(
                    bench_matrices, selected, y, alpha, diversity_weight
                )
                
                for new_j in try_unsel:
                    # Try swap
                    trial_sel = current_sel[:pos] + [new_j] + current_sel[pos+1:]
                    trial_selected = {**selected, bench_name: trial_sel}
                    trial_obj = joint_objective_of_selection(
                        bench_matrices, trial_selected, y, alpha, diversity_weight
                    )
                    
                    if trial_obj < best_obj - 1e-9:
                        best_obj = trial_obj
                        best_swap = new_j
                
                if best_swap is not None:
                    current_sel[pos] = best_swap
                    selected[bench_name] = current_sel
                    improved = True
        
        new_obj = joint_objective_of_selection(bench_matrices, selected, y, alpha, diversity_weight)
        print(f"  Iteration {it+1}: joint objective = {new_obj:.6f}")
        
        if not improved:
            print(f"  Converged at iteration {it+1}")
            break
    
    return selected


# ==============================================================================
# Data Loading (similar to bootstrap_prediction.py)
# ==============================================================================

@dataclass
class BenchmarkData:
    """Data for a single benchmark."""
    name: str
    model_dicts: Dict[str, Dict[str, float]]  # model -> {instance_id -> score}
    instance_ids: List[str]
    
    def get_matrix_for_models(self, models: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Get score matrix for specified models."""
        dicts = []
        for m in models:
            if m in self.model_dicts:
                dicts.append(self.model_dicts[m])
            else:
                # Fill with zeros
                dicts.append({iid: 0.0 for iid in self.instance_ids})
        
        return dicts_to_matrix(dicts, fill_value=0.0, id_list=self.instance_ids)


def load_benchmark(bench_name: str, base_dir: Path) -> BenchmarkData:
    """Load a single benchmark's data."""
    bench_full, _ = discover_benchmark(bench_name, base_dir=base_dir)
    models = bench_full.list_models()
    
    model_dicts: Dict[str, Dict[str, float]] = {}
    all_ids: set = set()
    
    for model in models:
        kept, dicts = bench_full.load_model_outputs_for_models([model])
        if kept and dicts:
            model_dicts[model] = dicts[0]
            all_ids.update(dicts[0].keys())
    
    return BenchmarkData(
        name=bench_name,
        model_dicts=model_dicts,
        instance_ids=sorted(all_ids),
    )


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_multibench_pipeline(
    source_bench_names: List[str],
    target_bench_names: List[str],
    base_dir: Path,
    output_dir: Path,
    train_models: Optional[List[str]] = None,
    eval_models: Optional[List[str]] = None,
    k_per_bench: int = 50,
    target_train_pct: float = 0.8,
    target_eval_pct: float = 0.2,
    n_refine_iters: int = 5,
    ridge_alpha: float = 0.1,
    diversity_weight: float = 0.0,
    split_seed: int = 0,
    select_seed: int = 0,
    annotate: bool = True,
) -> Dict:
    """Run the multi-benchmark prediction pipeline."""
    
    print("=" * 70)
    print("MULTI-BENCHMARK PREDICTION PIPELINE")
    print("=" * 70)
    print(f"Sources: {source_bench_names}")
    print(f"Targets: {target_bench_names}")
    print(f"k_per_bench: {k_per_bench}")
    print(f"Ridge alpha: {ridge_alpha}")
    print(f"Diversity weight: {diversity_weight}")
    
    # 1) Load all benchmarks
    print("\n[1] Loading benchmarks...")
    source_benchmarks = {name: load_benchmark(name, base_dir) for name in source_bench_names}
    target_benchmarks = {name: load_benchmark(name, base_dir) for name in target_bench_names}
    
    for name, bench in source_benchmarks.items():
        print(f"  Source {name}: {len(bench.model_dicts)} models, {len(bench.instance_ids)} instances")
    for name, bench in target_benchmarks.items():
        print(f"  Target {name}: {len(bench.model_dicts)} models, {len(bench.instance_ids)} instances")
    
    # 2) Determine models
    if train_models is not None and eval_models is not None:
        all_models = list(set(train_models) | set(eval_models))
        train_model_list = train_models
        eval_model_list = eval_models
    else:
        # Find common models
        all_model_sets = [set(b.model_dicts.keys()) for b in source_benchmarks.values()]
        all_model_sets.extend([set(b.model_dicts.keys()) for b in target_benchmarks.values()])
        common = set.intersection(*all_model_sets) if all_model_sets else set()
        all_models = sorted(common)
        train_model_list = all_models
        eval_model_list = all_models
    
    print(f"\n[2] Models:")
    print(f"  Train models ({len(train_model_list)}): {train_model_list}")
    print(f"  Eval models ({len(eval_model_list)}): {eval_model_list}")
    
    # 3) Build matrices for each source benchmark
    print("\n[3] Building source matrices...")
    train_bench_matrices: Dict[str, np.ndarray] = {}
    eval_bench_matrices: Dict[str, np.ndarray] = {}
    bench_instance_ids: Dict[str, List[str]] = {}
    
    for name, bench in source_benchmarks.items():
        A_train, ids = bench.get_matrix_for_models(train_model_list)
        A_eval, _ = bench.get_matrix_for_models(eval_model_list)
        train_bench_matrices[name] = A_train
        eval_bench_matrices[name] = A_eval
        bench_instance_ids[name] = ids
        print(f"  {name}: train {A_train.shape}, eval {A_eval.shape}")
    
    # 4) Build target matrix and split
    print("\n[4] Building target matrix...")
    # Merge all target benchmarks
    target_dicts_train = []
    target_dicts_eval = []
    for name, bench in target_benchmarks.items():
        for m in train_model_list:
            d = bench.model_dicts.get(m, {iid: 0.0 for iid in bench.instance_ids})
            # Prefix with benchmark name
            target_dicts_train.append({f"{name}::{k}": v for k, v in d.items()})
        for m in eval_model_list:
            d = bench.model_dicts.get(m, {iid: 0.0 for iid in bench.instance_ids})
            target_dicts_eval.append({f"{name}::{k}": v for k, v in d.items()})
    
    # Reshape: we need one dict per model
    n_train = len(train_model_list)
    n_eval = len(eval_model_list)
    n_tgt_bench = len(target_benchmarks)
    
    merged_train_dicts = []
    for i in range(n_train):
        merged = {}
        for j in range(n_tgt_bench):
            merged.update(target_dicts_train[j * n_train + i])
        merged_train_dicts.append(merged)
    
    merged_eval_dicts = []
    for i in range(n_eval):
        merged = {}
        for j in range(n_tgt_bench):
            merged.update(target_dicts_eval[j * n_eval + i])
        merged_eval_dicts.append(merged)
    
    A_tgt_train, tgt_ids = dicts_to_matrix(merged_train_dicts, fill_value=0.0)
    A_tgt_eval, _ = dicts_to_matrix(merged_eval_dicts, fill_value=0.0, id_list=tgt_ids)
    
    print(f"  Target train: {A_tgt_train.shape}")
    print(f"  Target eval: {A_tgt_eval.shape}")
    
    # Split target instances
    N_tgt = A_tgt_train.shape[1]
    target_train_size = int(N_tgt * target_train_pct)
    target_eval_size = int(N_tgt * target_eval_pct)
    
    rng_split = np.random.default_rng(split_seed)
    perm = rng_split.permutation(N_tgt)
    tgt_train_cols = perm[:target_train_size].tolist()
    tgt_eval_cols = perm[target_train_size:target_train_size + target_eval_size].tolist()
    
    y_train = compute_mean_over_indices(A_tgt_train, tgt_train_cols)
    y_eval = compute_mean_over_indices(A_tgt_eval, tgt_eval_cols) if target_eval_size > 0 else compute_mean_over_indices(A_tgt_eval, tgt_train_cols)
    
    print(f"  Target split: train={target_train_size}, eval={target_eval_size}")
    
    # 5) Select instances with iterative refinement
    print("\n[5] Selecting instances (with joint optimization)...")
    selected = iterative_selection_refinement(
        train_bench_matrices,
        y_train,
        k_per_bench=k_per_bench,
        n_iters=n_refine_iters,
        alpha=ridge_alpha,
        diversity_weight=diversity_weight,
        seed=select_seed,
    )
    
    # 6) Compute final metrics
    print("\n[6] Computing final metrics...")
    
    # Training metrics (without diversity term for final evaluation)
    X_train = compute_bench_scores_from_selection(train_bench_matrices, selected)
    _, train_coef, train_intercept, train_mse = mse_with_multibench_fit(X_train, y_train, ridge_alpha, diversity_weight=0.0)
    train_mae = mae_with_multibench_fit(X_train, y_train, ridge_alpha)
    train_r2 = r_squared_multibench(X_train, y_train, ridge_alpha)
    
    y_train_pred = X_train @ train_coef + train_intercept
    train_pearson = pearson_corr(y_train_pred, y_train)
    train_spearman = spearman_corr(y_train_pred, y_train)
    
    # Eval metrics
    X_eval = compute_bench_scores_from_selection(eval_bench_matrices, selected)
    # Use coefficients from training
    y_eval_pred = X_eval @ train_coef + train_intercept
    eval_mse = float(((y_eval - y_eval_pred) ** 2).mean())
    eval_mae = float(np.abs(y_eval - y_eval_pred).mean())
    
    ss_res = ((y_eval - y_eval_pred) ** 2).sum()
    ss_tot = ((y_eval - y_eval.mean()) ** 2).sum()
    eval_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    
    eval_pearson = pearson_corr(y_eval_pred, y_eval)
    eval_spearman = spearman_corr(y_eval_pred, y_eval)
    
    # Diversity metrics - both source and prediction
    train_pred_var = float(y_train_pred.var())
    train_pred_range = float(y_train_pred.max() - y_train_pred.min())
    eval_pred_var = float(y_eval_pred.var())
    eval_pred_range = float(y_eval_pred.max() - y_eval_pred.min())
    
    # Source score variance (how much models differ on selected instances)
    train_source_var = float((X_train @ np.abs(train_coef)).var())
    eval_source_var = float((X_eval @ np.abs(train_coef)).var())
    
    bench_names = sorted(train_bench_matrices.keys())
    
    print(f"\n  Per-benchmark coefficients:")
    for i, name in enumerate(bench_names):
        print(f"    {name}: {train_coef[i]:.4f}")
    print(f"    intercept: {train_intercept:.4f}")
    
    print(f"\n  Train metrics:")
    print(f"    MSE={train_mse:.6f}, MAE={train_mae:.6f}, R²={train_r2:.4f}")
    print(f"    Pearson={train_pearson:.4f}, Spearman={train_spearman:.4f}")
    print(f"    Source variance={train_source_var:.6f}, Pred variance={train_pred_var:.6f}, range={train_pred_range:.4f}")
    
    print(f"\n  Eval metrics:")
    print(f"    MSE={eval_mse:.6f}, MAE={eval_mae:.6f}, R²={eval_r2:.4f}")
    print(f"    Pearson={eval_pearson:.4f}, Spearman={eval_spearman:.4f}")
    print(f"    Source variance={eval_source_var:.6f}, Pred variance={eval_pred_var:.6f}, range={eval_pred_range:.4f}")
    
    # 7) Save outputs
    print("\n[7] Saving outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tag = "+".join(source_bench_names) + "_TO_" + "+".join(target_bench_names)
    
    # Compute error bars (standard error of the mean)
    # For X (predicted): compute SE from selected source instances per benchmark
    # For Y (actual): compute SE from target instances used in eval
    
    # X error: SE of weighted benchmark scores
    x_errors = []
    for i in range(len(eval_model_list)):
        # Variance of per-benchmark means for this model
        bench_means = X_eval[i, :]  # Per-benchmark means
        # Use weighted std as proxy for uncertainty
        weighted_var = np.average((bench_means - bench_means.mean())**2, weights=np.abs(train_coef))
        se = np.sqrt(weighted_var) / np.sqrt(len(bench_names))
        x_errors.append(se)
    x_errors = np.array(x_errors)
    
    # Y error: SE from target instance scores
    # Compute std of target scores for each model on eval instances
    y_errors = []
    for i in range(len(eval_model_list)):
        model_scores = A_tgt_eval[i, tgt_eval_cols]
        se = model_scores.std() / np.sqrt(len(tgt_eval_cols))
        y_errors.append(se)
    y_errors = np.array(y_errors)
    
    # Scatter plot: predicted vs actual with error bars and confidence interval
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute regression line (predicted vs actual)
    # Fit: actual = slope * predicted + intercept
    pred_mean = y_eval_pred.mean()
    actual_mean = y_eval.mean()
    slope = np.sum((y_eval_pred - pred_mean) * (y_eval - actual_mean)) / np.sum((y_eval_pred - pred_mean)**2)
    intercept_fit = actual_mean - slope * pred_mean
    
    # Create x values for the regression line
    x_line = np.linspace(y_eval_pred.min() - 0.05, y_eval_pred.max() + 0.05, 100)
    y_line = slope * x_line + intercept_fit
    
    # Compute confidence interval (95%)
    n = len(y_eval_pred)
    residuals = y_eval - (slope * y_eval_pred + intercept_fit)
    se_residuals = np.sqrt(np.sum(residuals**2) / (n - 2))
    
    # Standard error of prediction at each x
    x_var = np.sum((y_eval_pred - pred_mean)**2)
    se_line = se_residuals * np.sqrt(1/n + (x_line - pred_mean)**2 / x_var)
    
    # 95% CI (t-value for n-2 df)
    t_val = stats.t.ppf(0.975, n - 2)
    ci_upper = y_line + t_val * se_line
    ci_lower = y_line - t_val * se_line
    
    # Plot confidence interval band
    ax.fill_between(x_line, ci_lower, ci_upper, alpha=0.2, color='steelblue', 
                    label='95% Confidence Interval')
    
    # Plot regression line
    ax.plot(x_line, y_line, 'b-', linewidth=2, alpha=0.8, label=f'Fitted (slope={slope:.2f})')
    
    # Plot with error bars
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
    
    # Perfect prediction line
    min_val = min(x_line.min(), y_eval.min() - y_errors.max())
    max_val = max(x_line.max(), y_eval.max() + y_errors.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=1.5, 
            label='Perfect prediction')
    
    ax.legend(loc='upper left')
    ax.set_xlabel("Predicted (from multi-benchmark model)")
    ax.set_ylabel(f"Actual ({', '.join(target_bench_names)})")
    ax.set_title(f"Multi-Benchmark Prediction\nMSE={eval_mse:.4f}, R²={eval_r2:.4f}, Pearson={eval_pearson:.4f}")
    
    fig.tight_layout()
    fig.savefig(output_dir / f"{tag}_multibench_scatter.png", dpi=150)
    plt.close(fig)
    
    # Coefficient bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(bench_names))
    bars = ax.bar(x_pos, train_coef)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bench_names, rotation=45, ha='right')
    ax.set_ylabel("Coefficient")
    ax.set_title(f"Per-Benchmark Coefficients (intercept={train_intercept:.4f})")
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Color positive/negative differently
    for bar, coef in zip(bars, train_coef):
        bar.set_color('green' if coef > 0 else 'red')
    
    fig.tight_layout()
    fig.savefig(output_dir / f"{tag}_coefficients.png", dpi=150)
    plt.close(fig)
    
    # Collect selected instance IDs
    selected_ids: Dict[str, List[str]] = {}
    for bench_name, indices in selected.items():
        ids = bench_instance_ids[bench_name]
        selected_ids[bench_name] = [ids[i] for i in indices]
    
    # JSON results
    results = {
        "title": tag,
        "objective": "multibench_prediction_mse_diverse",
        "source_benchmarks": source_bench_names,
        "target_benchmarks": target_bench_names,
        "train_models": train_model_list,
        "eval_models": eval_model_list,
        "params": {
            "k_per_bench": k_per_bench,
            "target_train_pct": target_train_pct,
            "target_eval_pct": target_eval_pct,
            "ridge_alpha": ridge_alpha,
            "diversity_weight": diversity_weight,
            "n_refine_iters": n_refine_iters,
            "split_seed": split_seed,
            "select_seed": select_seed,
        },
        "model_formula": {
            "description": "y = intercept + sum(coef[i] * x[i])",
            "benchmark_order": bench_names,
            "coefficients": [float(c) for c in train_coef],
            "intercept": float(train_intercept),
        },
        "metrics_train": {
            "mse": float(train_mse),
            "mae": float(train_mae),
            "r_squared": float(train_r2),
            "pearson": float(train_pearson),
            "spearman": float(train_spearman),
            "source_variance": float(train_source_var),
            "prediction_variance": float(train_pred_var),
            "prediction_range": float(train_pred_range),
        },
        "metrics_eval": {
            "mse": float(eval_mse),
            "mae": float(eval_mae),
            "r_squared": float(eval_r2),
            "pearson": float(eval_pearson),
            "spearman": float(eval_spearman),
            "source_variance": float(eval_source_var),
            "prediction_variance": float(eval_pred_var),
            "prediction_range": float(eval_pred_range),
        },
        "selected_instances": {
            "total": sum(len(v) for v in selected_ids.values()),
            "by_benchmark": {k: len(v) for k, v in selected_ids.items()},
            "ids": selected_ids,
        },
    }
    
    out_json = output_dir / f"{tag}_multibench_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Saved: {out_json}")
    print(f"  Saved: {output_dir / f'{tag}_multibench_scatter.png'}")
    print(f"  Saved: {output_dir / f'{tag}_coefficients.png'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-benchmark prediction with per-benchmark coefficients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bootstrap_prediction_multibench.py \\
      --sources bfcl livecodebench gpqa humaneval_chat \\
      --targets swebench \\
      --k_per_bench 50

  # With train/eval model split
  python bootstrap_prediction_multibench.py \\
      --sources bfcl livecodebench \\
      --targets swebench \\
      --train_models GPT-5.2 Claude-4.5-Opus \\
      --eval_models GPT-5.2-Codex MiniMax-M2.1 \\
      --k_per_bench 30
        """
    )
    
    parser.add_argument("--sources", nargs="+", required=True,
                        help="Source benchmark names")
    parser.add_argument("--targets", nargs="+", required=True,
                        help="Target benchmark names")
    parser.add_argument("--base_dir", default=str(STD_BASE),
                        help="Base directory for standardized_results")
    parser.add_argument("--output_dir", default="../../analysis/multibench_prediction",
                        help="Output directory")
    
    parser.add_argument("--train_models", nargs="+", default=None,
                        help="Models for training")
    parser.add_argument("--eval_models", nargs="+", default=None,
                        help="Models for evaluation")
    
    parser.add_argument("--k_per_bench", type=int, default=50,
                        help="Number of instances to select per benchmark")
    parser.add_argument("--train_pct", type=float, default=0.8,
                        help="Fraction of target instances for training")
    parser.add_argument("--eval_pct", type=float, default=0.2,
                        help="Fraction of target instances for evaluation")
    parser.add_argument("--ridge_alpha", type=float, default=0.1,
                        help="Ridge regularization strength")
    parser.add_argument("--diversity_weight", type=float, default=0.0,
                        help="Weight for diversity term (higher = more spread in predictions). "
                             "Recommended: 0.001-0.01 for moderate spread, 0.05+ for aggressive spread")
    parser.add_argument("--n_refine_iters", type=int, default=5,
                        help="Number of refinement iterations")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--select_seed", type=int, default=0)
    parser.add_argument("--annotate", action="store_true",
                        help="Annotate points in scatter plot")
    
    args = parser.parse_args()
    
    if (args.train_models is None) != (args.eval_models is None):
        print("ERROR: Both --train_models and --eval_models must be specified together")
        return 1
    
    try:
        results = run_multibench_pipeline(
            source_bench_names=args.sources,
            target_bench_names=args.targets,
            base_dir=Path(args.base_dir),
            output_dir=Path(args.output_dir),
            train_models=args.train_models,
            eval_models=args.eval_models,
            k_per_bench=args.k_per_bench,
            target_train_pct=args.train_pct,
            target_eval_pct=args.eval_pct,
            n_refine_iters=args.n_refine_iters,
            ridge_alpha=args.ridge_alpha,
            diversity_weight=args.diversity_weight,
            split_seed=args.split_seed,
            select_seed=args.select_seed,
            annotate=args.annotate,
        )
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\nPrediction Formula:")
        print(f"  {' + '.join(args.targets)} = {results['model_formula']['intercept']:.4f}", end="")
        for name, coef in zip(results['model_formula']['benchmark_order'], 
                              results['model_formula']['coefficients']):
            print(f" + {coef:.4f}*{name}", end="")
        print()
        
        print(f"\nEval Metrics:")
        print(f"  MSE: {results['metrics_eval']['mse']:.6f}")
        print(f"  MAE: {results['metrics_eval']['mae']:.6f}")
        print(f"  R²:  {results['metrics_eval']['r_squared']:.4f}")
        
        print(f"\nSelected {results['selected_instances']['total']} instances:")
        for bench, count in results['selected_instances']['by_benchmark'].items():
            print(f"  {bench}: {count}")
        
    except Exception as e:
        traceback.print_exc()
        print(f"\nERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
