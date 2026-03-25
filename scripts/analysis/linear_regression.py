#!/usr/bin/env python3
"""
Run bootstrap pipeline optimizing for prediction accuracy (MSE) instead of correlation.

This script follows the same pipeline as bootstrap_all.py but instead of maximizing
Pearson correlation, it minimizes the Mean Squared Error (MSE) of a linear regression
fit: target_score ≈ a * source_score + b

The key difference from correlation-based selection:
- Correlation: "Do rankings transfer?" (relative ordering)
- Prediction: "Can we accurately predict scores?" (absolute values, after calibration)

Example usage:
    python bootstrap_prediction.py \\
        --sources humaneval mbpp livecodebench \\
        --targets swebench gaia \\
        --train_pct 0.8 --eval_pct 0.2 \\
        --boot_source_k 100 --boot_target_k 100

    # With separate train/eval models
    python bootstrap_prediction.py \\
        --sources humaneval mbpp \\
        --targets swebench \\
        --train_models GPT-5.2 Claude-4.5-Opus \\
        --eval_models GPT-5.2-Codex MiniMax-M2.1
"""

import argparse
import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

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
# Linear Regression and MSE primitives
# ==============================================================================

def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit a simple linear regression y = a*x + b.
    
    Returns:
        (a, b) - slope and intercept
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    
    if n < 2:
        return 0.0, y.mean() if len(y) > 0 else 0.0
    
    x_mean = x.mean()
    y_mean = y.mean()
    
    # Compute slope: a = Cov(x,y) / Var(x)
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    var_x = (x_centered ** 2).sum()
    if var_x < 1e-12:
        # x has no variance - can't fit a meaningful slope
        return 0.0, y_mean
    
    cov_xy = (x_centered * y_centered).sum()
    a = cov_xy / var_x
    b = y_mean - a * x_mean
    
    return float(a), float(b)


def mse_with_linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Compute MSE after fitting linear regression y = a*x + b.
    
    Returns:
        (mse, a, b) - mean squared error, slope, and intercept
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    a, b = linear_regression(x, y)
    y_pred = a * x + b
    mse = float(((y - y_pred) ** 2).mean())
    
    return mse, a, b


def mae_with_linear_fit(x: np.ndarray, y: np.ndarray) -> float:
    """Compute MAE after fitting linear regression y = a*x + b."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    a, b = linear_regression(x, y)
    y_pred = a * x + b
    mae = float(np.abs(y - y_pred).mean())
    
    return mae


def r_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Compute R² (coefficient of determination) for linear fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    a, b = linear_regression(x, y)
    y_pred = a * x + b
    
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    
    return float(1.0 - ss_res / ss_tot)


def mse_of_subset(A: np.ndarray, y: np.ndarray, S: List[int]) -> float:
    """Compute MSE of linear fit for a subset of source instances.
    
    Args:
        A: Source matrix (models x instances)
        y: Target scores (models,)
        S: List of source instance indices
    
    Returns:
        MSE of linear regression fit (lower is better)
    """
    if len(S) == 0:
        return float('inf')
    x = A[:, S].mean(axis=1)
    mse, _, _ = mse_with_linear_fit(x, y)
    return mse


# ==============================================================================
# Greedy subset selection optimizing for MSE
# ==============================================================================

def greedy_min_mse_subset(
    A: np.ndarray,
    y: np.ndarray,
    k: int,
    seed: int = 0,
    candidate_cap: Optional[int] = None,
    return_trace: bool = False,
) -> Tuple[List[int], float] | Tuple[List[int], float, List[float]]:
    """Greedily select k source instances to minimize MSE of linear regression.
    
    Args:
        A: Source matrix (models x instances)
        y: Target scores (models,)
        k: Number of instances to select
        seed: Random seed for shuffling candidates
        candidate_cap: If set, only consider top candidates by individual correlation
        return_trace: If True, return MSE trace per iteration
    
    Returns:
        (S, best_mse) or (S, best_mse, trace)
    """
    rng = np.random.default_rng(seed)
    M, N = A.shape
    
    # Optionally filter to top candidates by individual correlation with y
    if candidate_cap is not None and candidate_cap < N:
        yc = y - y.mean()
        ys = yc.std(ddof=0)
        scores = np.zeros(N, dtype=float)
        for j in range(N):
            col = A[:, j]
            cc = col - col.mean()
            cs = cc.std(ddof=0)
            r = 0.0 if cs == 0 or ys == 0 else float((cc @ yc) / (M * cs * ys))
            scores[j] = abs(r)
        cand = np.argsort(scores)[-candidate_cap:].tolist()
    else:
        cand = list(range(N))
    
    rng.shuffle(cand)
    S: List[int] = []
    in_set = np.zeros(N, dtype=bool)
    best_mse = float('inf')
    trace: List[float] = []
    
    for _ in range(k):
        best_j = None
        best_val = float('inf')
        
        for j in cand:
            if in_set[j]:
                continue
            val = mse_of_subset(A, y, S + [j])
            if val < best_val:
                best_val = val
                best_j = j
        
        if best_j is None:
            break
        
        S.append(best_j)
        in_set[best_j] = True
        best_mse = best_val
        
        if return_trace:
            trace.append(best_mse)
    
    return (S, best_mse, trace) if return_trace else (S, best_mse)


def swap_local_search_mse(
    A: np.ndarray,
    y: np.ndarray,
    S: List[int],
    max_passes: int = 10,
    sample_in: int = 300,
    seed: int = 0,
) -> Tuple[List[int], float]:
    """Local search to improve MSE by swapping instances."""
    rng = np.random.default_rng(seed)
    M, N = A.shape
    S = list(S)
    S_set = set(S)
    outside = [j for j in range(N) if j not in S_set]
    cur = mse_of_subset(A, y, S)
    
    for _ in range(max_passes):
        improved = False
        
        if sample_in is not None and sample_in < len(outside):
            outside_sample = rng.choice(outside, size=sample_in, replace=False).tolist()
        else:
            outside_sample = outside
        
        for out_pos in range(len(S)):
            j_out = S[out_pos]
            base = S[:out_pos] + S[out_pos + 1:]
            
            best_swap = None
            best_val = cur
            
            for j_in in outside_sample:
                val = mse_of_subset(A, y, base + [j_in])
                if val < best_val - 1e-12:  # Lower is better for MSE
                    best_val = val
                    best_swap = j_in
            
            if best_swap is not None:
                S_set.remove(j_out)
                S_set.add(best_swap)
                outside.remove(best_swap)
                outside.append(j_out)
                S[out_pos] = best_swap
                cur = best_val
                improved = True
        
        if not improved:
            break
    
    return S, cur


def multi_restart_select_subset_mse(
    A: np.ndarray,
    y: np.ndarray,
    k: int,
    n_restarts: int = 10,
    greedy_seed0: int = 0,
    swap_passes: int = 10,
    swap_sample_in: int = 300,
    candidate_cap: Optional[int] = None,
    return_trace: bool = False,
) -> Tuple[List[int], float] | Tuple[List[int], float, List[float]]:
    """Multi-restart greedy + local search to minimize MSE."""
    best_S = None
    best_mse = float('inf')
    best_trace: List[float] = []
    
    for r in range(n_restarts):
        seed = greedy_seed0 + r
        
        if return_trace:
            S, _mse, tr = greedy_min_mse_subset(
                A, y, k=k, seed=seed, candidate_cap=candidate_cap, return_trace=True
            )
        else:
            S, _mse = greedy_min_mse_subset(
                A, y, k=k, seed=seed, candidate_cap=candidate_cap, return_trace=False
            )
            tr = []
        
        S, mse_after_swap = swap_local_search_mse(
            A, y, S, max_passes=swap_passes, sample_in=swap_sample_in, seed=seed
        )
        
        if mse_after_swap < best_mse:
            best_mse = mse_after_swap
            best_S = S
            best_trace = tr
    
    return (best_S, best_mse, best_trace) if return_trace else (best_S, best_mse)


def two_sided_bootstrap_min_mse_sample(
    A_src: np.ndarray,
    A_tgt: np.ndarray,
    tgt_pool_cols: List[int],
    k_src: int,
    k_tgt: int,
    n_boot: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Bootstrap sampling to find low-MSE sample pair."""
    rng = np.random.default_rng(seed)
    M, N_src = A_src.shape
    pool = np.asarray(tgt_pool_cols, dtype=int)
    
    if len(pool) == 0:
        raise ValueError("tgt_pool_cols is empty")
    
    best_mse = float('inf')
    best_x = None
    best_y = None
    
    for _ in range(n_boot):
        idx_l = rng.integers(0, N_src, size=k_src)
        idx_s = rng.choice(pool, size=k_tgt, replace=True)
        x = compute_mean_over_indices(A_src, idx_l)
        y = compute_mean_over_indices(A_tgt, idx_s)
        
        mse, _, _ = mse_with_linear_fit(x, y)
        
        if mse < best_mse:
            best_mse = mse
            best_x = x
            best_y = y
    
    return best_x, best_y, float(best_mse)


def vote_source_instances_over_target_bootstraps_mse(
    A_src: np.ndarray,
    A_tgt: np.ndarray,
    tgt_train_cols: List[int],
    y_tgt_eval: np.ndarray,
    k_src: int = 200,
    tgt_boot_k: int = 400,
    n_outer: int = 50,
    n_restarts_inner: int = 10,
    seed: int = 0,
    swap_passes: int = 10,
    swap_sample_in: int = 300,
    candidate_cap: Optional[int] = None,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """Vote for source instances by minimizing MSE over target bootstraps.
    
    Returns:
        (final_S_vote, counts, eval_mses, boot_mses)
    """
    rng = np.random.default_rng(seed)
    M, N_src = A_src.shape
    pool = np.asarray(tgt_train_cols, dtype=int)
    
    counts = np.zeros(N_src, dtype=int)
    eval_mses = np.zeros(n_outer, dtype=float)
    boot_mses = np.zeros(n_outer, dtype=float)
    
    print_every = max(1, n_outer // 10)
    
    for t in range(n_outer):
        # Bootstrap sample target instances
        idx_s = rng.choice(pool, size=tgt_boot_k, replace=True)
        y_boot = compute_mean_over_indices(A_tgt, idx_s)
        
        # Select source instances that minimize MSE with this bootstrap
        S, best_mse = multi_restart_select_subset_mse(
            A_src,
            y_boot,
            k=k_src,
            n_restarts=n_restarts_inner,
            greedy_seed0=seed + 1000 + t * 13,
            swap_passes=swap_passes,
            swap_sample_in=swap_sample_in,
            candidate_cap=candidate_cap,
            return_trace=False,
        )
        
        # Vote for selected instances
        counts[S] += 1
        
        # Evaluate on held-out eval target
        x_S = compute_mean_over_indices(A_src, S)
        eval_mse, _, _ = mse_with_linear_fit(x_S, y_tgt_eval)
        
        eval_mses[t] = eval_mse
        boot_mses[t] = best_mse
        
        if (t + 1) % print_every == 0 or (t + 1) == n_outer:
            print(f"  [vote] iter {t+1}/{n_outer}: boot_mse={best_mse:.6f} | eval_mse={eval_mse:.6f}")
    
    # Select top-k by vote count
    final_S_vote = np.argsort(-counts)[:k_src].tolist()
    return final_S_vote, counts, eval_mses, boot_mses


# ==============================================================================
# MergedBenchDataset (same as bootstrap_all.py)
# ==============================================================================

@dataclass
class MergedBenchDataset:
    """Represents a merged dataset from multiple benchmarks."""
    
    bench_names: List[str]
    label: str
    model_dicts: Dict[str, Dict[str, float]]
    all_instance_ids: List[str]
    
    def list_models(self) -> List[str]:
        return sorted(self.model_dicts.keys())
    
    def get_model_dict(self, model_name: str) -> Dict[str, float]:
        return self.model_dicts.get(model_name, {})
    
    def get_model_dict_with_zeros(self, model_name: str) -> Dict[str, float]:
        """Get model dict, filling missing instances with 0.0."""
        base_dict = self.model_dicts.get(model_name, {})
        return {iid: base_dict.get(iid, 0.0) for iid in self.all_instance_ids}


def load_benchmark_data(
    bench_name: str,
    base_dir: Path,
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Load all model data for a benchmark."""
    bench_full, _ = discover_benchmark(bench_name, base_dir=base_dir)
    models = bench_full.list_models()
    
    model_dicts: Dict[str, Dict[str, float]] = {}
    
    for model in models:
        kept, dicts = bench_full.load_model_outputs_for_models([model])
        if kept and dicts:
            prefixed_dict = {
                f"{bench_name}::{k}": v
                for k, v in dicts[0].items()
            }
            model_dicts[model] = prefixed_dict
    
    return models, model_dicts


def merge_benchmarks(
    bench_names: List[str],
    base_dir: Path,
    label: str,
    target_models: Optional[List[str]] = None,
) -> MergedBenchDataset:
    """Merge multiple benchmarks into a single dataset."""
    print(f"\nMerging benchmarks for {label}: {bench_names}")
    
    all_model_sets: List[set] = []
    bench_data: Dict[str, Dict[str, Dict[str, float]]] = {}
    bench_instance_ids: Dict[str, set] = {}
    
    for bench_name in bench_names:
        models, model_dicts = load_benchmark_data(bench_name, base_dir)
        all_model_sets.append(set(models))
        bench_data[bench_name] = model_dicts
        
        instance_ids = set()
        for d in model_dicts.values():
            instance_ids.update(d.keys())
        bench_instance_ids[bench_name] = instance_ids
        
        print(f"  - {bench_name}: {len(models)} models, {len(instance_ids)} instances")
    
    all_instance_ids = sorted(set().union(*bench_instance_ids.values()) if bench_instance_ids else set())
    
    if target_models is not None:
        models_to_use = target_models
        print(f"  Using specified models: {len(models_to_use)} models")
        
        for model in models_to_use:
            missing_benches = [b for b in bench_names if model not in bench_data[b]]
            if missing_benches:
                print(f"    [INFO] {model} missing from {missing_benches}, will use 0 scores")
    else:
        common_models = set.intersection(*all_model_sets) if all_model_sets else set()
        models_to_use = sorted(common_models)
        print(f"  Common models across all benchmarks: {len(models_to_use)}")
    
    merged_model_dicts: Dict[str, Dict[str, float]] = {}
    
    for model in models_to_use:
        merged_dict: Dict[str, float] = {}
        for bench_name in bench_names:
            if model in bench_data[bench_name]:
                merged_dict.update(bench_data[bench_name][model])
            else:
                for iid in bench_instance_ids[bench_name]:
                    merged_dict[iid] = 0.0
        merged_model_dicts[model] = merged_dict
    
    total_instances = len(all_instance_ids)
    print(f"  Merged dataset: {len(models_to_use)} models, {total_instances} instances per model")
    
    return MergedBenchDataset(
        bench_names=bench_names,
        label=label,
        model_dicts=merged_model_dicts,
        all_instance_ids=all_instance_ids,
    )


# ==============================================================================
# Main pipeline
# ==============================================================================

def run_pipeline_for_merged(
    src_merged: MergedBenchDataset,
    tgt_merged: MergedBenchDataset,
    omit_models: Optional[List[str]] = None,
    train_models: Optional[List[str]] = None,
    eval_models: Optional[List[str]] = None,
    plot_dir: Path = Path("plots"),
    title_prefix: str = "",
    k_source: Optional[int] = None,
    target_train_pct: float = 0.8,
    target_eval_pct: float = 0.2,
    n_outer: int = 10,
    n_restarts_inner: int = 10,
    boot_source_k: int = 200,
    boot_target_k: int = 400,
    n_boot: int = 50,
    boot_seed: int = 0,
    split_seed: int = 0,
    swap_passes: int = 10,
    swap_sample_in: int = 300,
    candidate_cap: Optional[int] = None,
    annotate: bool = True,
) -> Dict:
    """Run the bootstrap pipeline optimizing for prediction (MSE)."""
    
    omit_models = set(omit_models or [])
    tag = title_prefix or f"{'_'.join(src_merged.bench_names)}_to_{'_'.join(tgt_merged.bench_names)}_{k_source}"
    
    # 1) Determine models for training and evaluation
    if train_models is not None and eval_models is not None:
        train_model_set = [m for m in train_models if m not in omit_models]
        eval_model_set = [m for m in eval_models if m not in omit_models]
        print(f"Train models ({len(train_model_set)}): {train_model_set}")
        print(f"Eval models ({len(eval_model_set)}): {eval_model_set}")
        
        if not train_model_set:
            raise RuntimeError("No train models after filtering")
        if not eval_model_set:
            raise RuntimeError("No eval models after filtering")
        
        src_dicts_train = [src_merged.get_model_dict_with_zeros(m) for m in train_model_set]
        tgt_dicts_train = [tgt_merged.get_model_dict_with_zeros(m) for m in train_model_set]
        
        src_dicts_eval = [src_merged.get_model_dict_with_zeros(m) for m in eval_model_set]
        tgt_dicts_eval = [tgt_merged.get_model_dict_with_zeros(m) for m in eval_model_set]
        
        A_src_train, src_ids = dicts_to_matrix(src_dicts_train, fill_value=0.0)
        A_tgt_train, tgt_ids = dicts_to_matrix(tgt_dicts_train, fill_value=0.0)
        
        A_src_eval, src_ids_eval = dicts_to_matrix(src_dicts_eval, fill_value=0.0)
        A_tgt_eval, tgt_ids_eval = dicts_to_matrix(tgt_dicts_eval, fill_value=0.0)
        
        assert src_ids == src_ids_eval
        assert tgt_ids == tgt_ids_eval
        
        print(f"\nTraining: {len(train_model_set)} models")
        print(f"Evaluation: {len(eval_model_set)} models")
        
        models = train_model_set
        use_separate_eval = True
    else:
        src_models = set(src_merged.list_models()) - omit_models
        tgt_models = set(tgt_merged.list_models()) - omit_models
        models = sorted(src_models.intersection(tgt_models))
        
        if not models:
            raise RuntimeError("No overlapping models between merged source and target")
        
        print(f"\nRunning pipeline with {len(models)} models")
        
        src_dicts = [src_merged.get_model_dict(m) for m in models]
        tgt_dicts = [tgt_merged.get_model_dict(m) for m in models]
        
        A_src_train, src_ids = dicts_to_matrix(src_dicts, fill_value=0.0)
        A_tgt_train, tgt_ids = dicts_to_matrix(tgt_dicts, fill_value=0.0)
        
        A_src_eval = A_src_train
        A_tgt_eval = A_tgt_train
        
        train_model_set = models
        eval_model_set = models
        use_separate_eval = False
    
    print(f"Source matrix: {A_src_train.shape} (models x instances)")
    print(f"Target matrix: {A_tgt_train.shape} (models x instances)")
    
    # 2) Split target into train/eval
    N_tgt = A_tgt_train.shape[1]
    
    if target_train_pct + target_eval_pct > 1.0:
        raise ValueError(f"train_pct ({target_train_pct}) + eval_pct ({target_eval_pct}) > 1.0")
    
    target_train_size = int(N_tgt * target_train_pct)
    target_eval_size = int(N_tgt * target_eval_pct)
    
    if target_train_pct > 0 and target_train_size == 0:
        target_train_size = 1
    if target_eval_pct > 0 and target_eval_size == 0:
        target_eval_size = 1
    
    print(f"Target split: train={target_train_size} ({target_train_pct:.0%}), "
          f"eval={target_eval_size} ({target_eval_pct:.0%}) of {N_tgt} total")
    
    rng_split = np.random.default_rng(split_seed)
    perm = rng_split.permutation(N_tgt)
    tgt_train_cols = perm[:target_train_size].tolist()
    tgt_eval_cols = perm[target_train_size:target_train_size + target_eval_size].tolist()
    
    y_tgt_train = compute_mean_over_indices(A_tgt_train, tgt_train_cols)
    if use_separate_eval:
        y_tgt_eval = compute_mean_over_indices(A_tgt_eval, tgt_eval_cols) if target_eval_size > 0 else compute_mean_over_indices(A_tgt_eval, tgt_train_cols)
    else:
        y_tgt_eval = compute_mean_over_indices(A_tgt_train, tgt_eval_cols) if target_eval_size > 0 else compute_mean_over_indices(A_tgt_train, tgt_train_cols)
    
    # 3) Two-sided bootstrap to find low-MSE sample
    x_boot_min, y_boot_min, boot_min_mse = two_sided_bootstrap_min_mse_sample(
        A_src_train, A_tgt_train, tgt_train_cols,
        k_src=boot_source_k, k_tgt=boot_target_k,
        n_boot=n_boot, seed=boot_seed,
    )
    print(f"Bootstrap min MSE: {boot_min_mse:.6f}")
    
    # 4) Voting-based subset selection (minimizing MSE)
    if k_source is None:
        k_source = boot_source_k
    
    S_vote, vote_counts, eval_mses, boot_mses = vote_source_instances_over_target_bootstraps_mse(
        A_src_train,
        A_tgt_train,
        tgt_train_cols,
        y_tgt_eval,
        k_src=k_source,
        tgt_boot_k=boot_target_k,
        n_outer=n_outer,
        n_restarts_inner=n_restarts_inner,
        seed=boot_seed,
        swap_passes=swap_passes,
        swap_sample_in=swap_sample_in,
        candidate_cap=candidate_cap,
    )
    
    # 5) Evaluate final subset
    x_vote_train = compute_mean_over_indices(A_src_train, S_vote)
    
    # Training metrics
    train_mse, train_a, train_b = mse_with_linear_fit(x_vote_train, y_tgt_train)
    train_mae = mae_with_linear_fit(x_vote_train, y_tgt_train)
    train_r2 = r_squared(x_vote_train, y_tgt_train)
    train_corr_p = pearson_corr(x_vote_train, y_tgt_train)
    train_corr_s = spearman_corr(x_vote_train, y_tgt_train)
    
    # Eval metrics
    if use_separate_eval:
        x_vote_eval = compute_mean_over_indices(A_src_eval, S_vote)
    else:
        x_vote_eval = x_vote_train
    
    eval_mse, eval_a, eval_b = mse_with_linear_fit(x_vote_eval, y_tgt_eval)
    eval_mae = mae_with_linear_fit(x_vote_eval, y_tgt_eval)
    eval_r2 = r_squared(x_vote_eval, y_tgt_eval)
    eval_corr_p = pearson_corr(x_vote_eval, y_tgt_eval)
    eval_corr_s = spearman_corr(x_vote_eval, y_tgt_eval)
    
    print(f"\nFinal metrics:")
    print(f"  Train ({len(train_model_set)} models):")
    print(f"    MSE={train_mse:.6f}, MAE={train_mae:.6f}, R²={train_r2:.4f}")
    print(f"    Pearson={train_corr_p:.4f}, Spearman={train_corr_s:.4f}")
    print(f"    Linear fit: y = {train_a:.4f}*x + {train_b:.4f}")
    print(f"  Eval ({len(eval_model_set)} models):")
    print(f"    MSE={eval_mse:.6f}, MAE={eval_mae:.6f}, R²={eval_r2:.4f}")
    print(f"    Pearson={eval_corr_p:.4f}, Spearman={eval_corr_s:.4f}")
    print(f"    Linear fit: y = {eval_a:.4f}*x + {eval_b:.4f}")
    
    # 6) Save plots
    out_dir = plot_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Scatter plot with linear fit
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_vote_eval, y_tgt_eval, alpha=0.7, s=60)
    
    if annotate:
        for i, m in enumerate(eval_model_set):
            ax.annotate(m, (x_vote_eval[i], y_tgt_eval[i]), fontsize=6, alpha=0.7)
    
    # Add regression line
    if len(x_vote_eval) > 1:
        x_line = np.linspace(min(x_vote_eval), max(x_vote_eval), 100)
        y_line = eval_a * x_line + eval_b
        ax.plot(x_line, y_line, "r--", alpha=0.7, label=f"y = {eval_a:.3f}x + {eval_b:.3f}")
        ax.legend()
    
    ax.set_xlabel(f"Source (merged): {', '.join(src_merged.bench_names)}")
    ax.set_ylabel(f"Target (merged): {', '.join(tgt_merged.bench_names)}")
    ax.set_title(f"{tag}\nMSE={eval_mse:.4f}, MAE={eval_mae:.4f}, R²={eval_r2:.4f}")
    
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_prediction_scatter.png", dpi=150)
    plt.close(fig)
    
    # MSE per iteration plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(eval_mses) + 1), eval_mses, 'b-o', label='Eval MSE')
    ax.plot(range(1, len(boot_mses) + 1), boot_mses, 'g--s', label='Boot MSE')
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("MSE")
    ax.set_title(f"{tag}: MSE per iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_mse_per_iter.png", dpi=150)
    plt.close(fig)
    
    # Get selected instance IDs
    top_ids = [src_ids[j] for j in S_vote]
    
    # Group by benchmark
    instances_by_bench: Dict[str, List[str]] = {}
    for inst_id in top_ids:
        bench = inst_id.split("::")[0]
        if bench not in instances_by_bench:
            instances_by_bench[bench] = []
        instances_by_bench[bench].append(inst_id)
    
    # Save JSON results
    out_json = out_dir / f"{tag}_prediction_results.json"
    results_dict = {
        "title": tag,
        "objective": "prediction_mse",
        "source_benchmarks": src_merged.bench_names,
        "target_benchmarks": tgt_merged.bench_names,
        "train_models": train_model_set,
        "eval_models": eval_model_set,
        "params": {
            "k_source": k_source,
            "target_train_pct": target_train_pct,
            "target_eval_pct": target_eval_pct,
            "target_train_size": target_train_size,
            "target_eval_size": target_eval_size,
            "n_outer": n_outer,
            "n_restarts_inner": n_restarts_inner,
            "boot_source_k": boot_source_k,
            "boot_target_k": boot_target_k,
            "n_boot": n_boot,
            "boot_seed": boot_seed,
            "split_seed": split_seed,
        },
        "source_matrix_shape": list(A_src_train.shape),
        "target_matrix_shape": list(A_tgt_train.shape),
        "metrics_train": {
            "mse": float(train_mse),
            "mae": float(train_mae),
            "r_squared": float(train_r2),
            "pearson": float(train_corr_p),
            "spearman": float(train_corr_s),
            "linear_fit": {"a": float(train_a), "b": float(train_b)},
        },
        "metrics_eval": {
            "mse": float(eval_mse),
            "mae": float(eval_mae),
            "r_squared": float(eval_r2),
            "pearson": float(eval_corr_p),
            "spearman": float(eval_corr_s),
            "linear_fit": {"a": float(eval_a), "b": float(eval_b)},
        },
        "boot_min_mse": float(boot_min_mse),
        "eval_mses_per_iter": [float(x) for x in eval_mses],
        "boot_mses_per_iter": [float(x) for x in boot_mses],
        "selected_source_instances": {
            "total": len(top_ids),
            "by_benchmark": {k: len(v) for k, v in instances_by_bench.items()},
            "ids": top_ids,
        },
    }
    
    with open(out_json, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nSaved results to: {out_json}")
    print(f"Selected {len(top_ids)} source instances:")
    for bench, ids in sorted(instances_by_bench.items()):
        print(f"  - {bench}: {len(ids)} instances")
    
    return results_dict


def main():
    parser = argparse.ArgumentParser(
        description="Run bootstrap pipeline optimizing for prediction (MSE) on merged benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge multiple source benchmarks and targets
  python bootstrap_prediction.py \\
      --sources humaneval mbpp livecodebench \\
      --targets swebench gaia \\
      --train_pct 0.8 --eval_pct 0.2 \\
      --boot_source_k 100 --boot_target_k 100

  # With separate train/eval model splits
  python bootstrap_prediction.py \\
      --sources humaneval mbpp \\
      --targets swebench \\
      --train_models GPT-5.2 Claude-4.5-Opus Gemini-3-Pro-Preview \\
      --eval_models GPT-5.2-Codex MiniMax-M2.1
        """
    )
    
    # Benchmark selection
    parser.add_argument("--sources", nargs="+", required=True,
                        help="List of source benchmark names to merge")
    parser.add_argument("--targets", nargs="+", required=True,
                        help="List of target benchmark names to merge")
    parser.add_argument("--base_dir", default=str(STD_BASE),
                        help="Base directory for standardized_results")
    
    # Model selection
    parser.add_argument("--train_models", nargs="+", default=None,
                        help="Models for training (greedy selection). Missing = 0 scores.")
    parser.add_argument("--eval_models", nargs="+", default=None,
                        help="Models for evaluation. Missing = 0 scores.")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="../../analysis/prediction",
                        help="Output directory for results")
    
    # Target split parameters
    parser.add_argument("--train_pct", type=float, default=0.8,
                        help="Fraction of target instances for training (default: 0.8)")
    parser.add_argument("--eval_pct", type=float, default=0.2,
                        help="Fraction of target instances for evaluation (default: 0.2)")
    
    # Bootstrap parameters
    parser.add_argument("--boot_source_k", type=int, default=100,
                        help="Source bootstrap sample size (default: 100)")
    parser.add_argument("--boot_target_k", type=int, default=100,
                        help="Target bootstrap sample size (default: 100)")
    parser.add_argument("--k_source", type=int, default=None,
                        help="Final source subset size (default: boot_source_k)")
    
    # Algorithm parameters
    parser.add_argument("--n_outer", type=int, default=10,
                        help="Outer iterations (default: 10)")
    parser.add_argument("--n_restarts", type=int, default=10,
                        help="Greedy+swap restarts per outer iteration (default: 10)")
    parser.add_argument("--n_bootstraps", type=int, default=50,
                        help="Number of bootstrap samples (default: 50)")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--boot_seed", type=int, default=0)
    parser.add_argument("--swap_passes", type=int, default=10)
    parser.add_argument("--swap_sample_in", type=int, default=300)
    parser.add_argument("--candidate_cap", type=int, default=None)
    
    # Misc
    parser.add_argument("--omit_models", type=str, default="",
                        help="Comma-separated list of model names to omit")
    parser.add_argument("--annotate", action="store_true",
                        help="Annotate points in scatter plots")
    
    args = parser.parse_args()
    
    omit_models = [x.strip() for x in args.omit_models.split(",") if x.strip()]
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    
    # Determine target models for merging
    if args.train_models is not None or args.eval_models is not None:
        if args.train_models is None or args.eval_models is None:
            print("ERROR: Both --train_models and --eval_models must be specified together")
            return 1
        all_models = list(set(args.train_models) | set(args.eval_models))
    else:
        all_models = None
    
    print("=" * 70)
    print("BOOTSTRAP PREDICTION (MSE) ON MERGED BENCHMARKS")
    print("=" * 70)
    print(f"Sources: {args.sources}")
    print(f"Targets: {args.targets}")
    if args.train_models:
        print(f"Train models: {args.train_models}")
        print(f"Eval models: {args.eval_models}")
    
    # Merge benchmarks
    src_merged = merge_benchmarks(
        args.sources, base_dir,
        label="source_" + "_".join(args.sources),
        target_models=all_models,
    )
    
    tgt_merged = merge_benchmarks(
        args.targets, base_dir,
        label="target_" + "_".join(args.targets),
        target_models=all_models,
    )
    
    # Create tag for output
    src_tag = "+".join(args.sources)
    tgt_tag = "+".join(args.targets)
    tag = f"{src_tag}_TO_{tgt_tag}"
    
    # Run pipeline
    try:
        results = run_pipeline_for_merged(
            src_merged,
            tgt_merged,
            omit_models=omit_models,
            train_models=args.train_models,
            eval_models=args.eval_models,
            plot_dir=output_dir,
            title_prefix=tag,
            k_source=args.k_source,
            target_train_pct=args.train_pct,
            target_eval_pct=args.eval_pct,
            n_outer=args.n_outer,
            n_restarts_inner=args.n_restarts,
            boot_source_k=args.boot_source_k,
            boot_target_k=args.boot_target_k,
            n_boot=args.n_bootstraps,
            boot_seed=args.boot_seed,
            split_seed=args.split_seed,
            swap_passes=args.swap_passes,
            swap_sample_in=args.swap_sample_in,
            candidate_cap=args.candidate_cap,
            annotate=args.annotate,
        )
        
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Source benchmarks: {args.sources}")
        print(f"Target benchmarks: {args.targets}")
        print(f"Train models: {len(results['train_models'])}")
        print(f"Eval models: {len(results['eval_models'])}")
        print(f"Source instances: {results['source_matrix_shape'][1]}")
        print(f"Target instances: {results['target_matrix_shape'][1]}")
        print(f"\nPrediction Metrics (Eval):")
        print(f"  MSE: {results['metrics_eval']['mse']:.6f}")
        print(f"  MAE: {results['metrics_eval']['mae']:.6f}")
        print(f"  R²:  {results['metrics_eval']['r_squared']:.4f}")
        print(f"  Linear fit: y = {results['metrics_eval']['linear_fit']['a']:.4f}*x + {results['metrics_eval']['linear_fit']['b']:.4f}")
        print(f"\nCorrelation Metrics (Eval):")
        print(f"  Pearson:  {results['metrics_eval']['pearson']:.4f}")
        print(f"  Spearman: {results['metrics_eval']['spearman']:.4f}")
        print(f"\nSelected {results['selected_source_instances']['total']} proxy instances:")
        for bench, count in results['selected_source_instances']['by_benchmark'].items():
            print(f"  - {bench}: {count}")
        
    except Exception as e:
        traceback.print_exc()
        print(f"\nERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
