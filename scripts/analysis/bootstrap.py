#!/usr/bin/env python3

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Data loading from standardized_results
# ==============================================================================

STD_BASE = Path(__file__).resolve().parents[2] / "results" / "standardized_results"


@dataclass
class BenchDataset:
    """Represents a single dataset (either a whole benchmark or a subtask).

    Attributes:
        bench_name: Name of the benchmark (directory under standardized_results)
        label: Dataset label (subtask name) or "full" for the entire benchmark
        bench_dir: Path to the benchmark root dir under standardized_results
        subtask_dir: Path to the subtask dir (if label != "full"), else None
        has_subtasks: Whether the benchmark has subtask subdirectories
    """

    bench_name: str
    label: str  # "full" or subtask name
    bench_dir: Path
    subtask_dir: Optional[Path]
    has_subtasks: bool

    # ---------- Model enumeration ----------
    def list_models(self) -> List[str]:
        if self.label != "full" and self.subtask_dir is not None:
            return sorted([p.stem for p in self.subtask_dir.glob("*.csv")])
        # full dataset
        if not self.has_subtasks:
            return sorted([p.stem for p in self.bench_dir.glob("*.csv")])
        # aggregate across all subtasks
        names = set()
        for d in sorted([p for p in self.bench_dir.iterdir() if p.is_dir()]):
            for p in d.glob("*.csv"):
                names.add(p.stem)
        return sorted(names)

    # ---------- Model dict loading ----------
    def load_model_outputs_for_models(self, model_names: List[str]) -> Tuple[List[str], List[Dict[str, float]]]:
        kept: List[str] = []
        dicts: List[Dict[str, float]] = []

        if self.label != "full" and self.subtask_dir is not None:
            # Simple: load only from this subtask dir
            for m in model_names:
                f = self.subtask_dir / f"{m}.csv"
                if not f.exists():
                    continue
                d = _load_standardized_csv_as_dict(f)
                dicts.append(d)
                kept.append(m)
            return kept, dicts

        # full dataset
        if not self.has_subtasks:
            for m in model_names:
                f = self.bench_dir / f"{m}.csv"
                if not f.exists():
                    continue
                d = _load_standardized_csv_as_dict(f)
                dicts.append(d)
                kept.append(m)
            return kept, dicts

        # aggregate across all subtasks: prefix ids with subtask label to ensure uniqueness
        for m in model_names:
            d_all: Dict[str, float] = {}
            any_found = False
            for sub in sorted([p for p in self.bench_dir.iterdir() if p.is_dir()]):
                f = sub / f"{m}.csv"
                if not f.exists():
                    continue
                d = _load_standardized_csv_as_dict(f)
                if not d:
                    continue
                any_found = True
                for sid, sc in d.items():
                    d_all[f"{sub.name}::{sid}"] = sc
            if any_found:
                kept.append(m)
                dicts.append(d_all)
        return kept, dicts


# ------------------------------------------------------------------------------
# Standardized CSV loader
# ------------------------------------------------------------------------------

def _load_standardized_csv_as_dict(csv_path: Path) -> Dict[str, float]:
    import pandas as pd  # local import to avoid hard dep if unused

    df = pd.read_csv(csv_path)
    if not {"id", "score", "metric_name"}.issubset(df.columns):
        raise ValueError(
            f"CSV {csv_path} missing required columns: id, score, metric_name. Got: {list(df.columns)[:20]}"
        )
    ids = df["id"].astype(str).tolist()
    # Any missing => 0.0, clip to [0,1]
    scores = pd.to_numeric(df["score"], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float).tolist()
    return dict(zip(ids, scores))


# ------------------------------------------------------------------------------
# Benchmark discovery
# ------------------------------------------------------------------------------

def discover_benchmark(bench_name: str, base_dir: Path = STD_BASE) -> Tuple[BenchDataset, List[BenchDataset]]:
    """Return (full_dataset, list_of_subtask_datasets).

    If no subtasks, the second list is empty.
    """
    bench_dir = (base_dir / bench_name).resolve()
    if not bench_dir.exists() or not bench_dir.is_dir():
        raise FileNotFoundError(f"Benchmark '{bench_name}' not found under {base_dir}")

    # Determine if it has subtasks: folders with .csv inside
    subdirs = [p for p in bench_dir.iterdir() if p.is_dir()]
    has_subtasks = any(list(d.glob("*.csv")) for d in subdirs)

    full = BenchDataset(
        bench_name=bench_name,
        label="full",
        bench_dir=bench_dir,
        subtask_dir=None,
        has_subtasks=has_subtasks,
    )

    subtasks: List[BenchDataset] = []
    if has_subtasks:
        for d in sorted([p for p in subdirs if p.is_dir()]):
            if not any(d.glob("*.csv")):
                continue
            subtasks.append(
                BenchDataset(
                    bench_name=bench_name,
                    label=d.name,
                    bench_dir=bench_dir,
                    subtask_dir=d,
                    has_subtasks=has_subtasks,
                )
            )

    return full, subtasks


# ==============================================================================
# Correlation + optimization primitives (adapted from analysis/bootstrap.py)
# ==============================================================================

def pearson_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    yc = y - y.mean()
    xs = xc.std(ddof=0)
    ys = yc.std(ddof=0)
    if xs == 0 or ys == 0:
        return 0.0
    return float((xc @ yc) / (len(y) * xs * ys))


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    # Equivalent to scipy.stats.rankdata(method='average'), 1-based ranks
    a = np.asarray(a)
    n = a.size
    sorter = np.argsort(a, kind='mergesort')
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(n)
    a_sorted = a[sorter]

    # Find run boundaries
    diffs = np.diff(a_sorted)
    # Indices where a new value starts in sorted order
    run_starts = np.r_[0, np.nonzero(diffs != 0)[0] + 1]
    run_ends = np.r_[run_starts[1:], n]

    ranks_sorted = np.empty(n, dtype=float)
    for s, e in zip(run_starts, run_ends):
        # ordinal ranks are 1..n; average ties
        avg = (s + 1 + e) / 2.0
        ranks_sorted[s:e] = avg
    return ranks_sorted[inv]


def spearman_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = _rankdata_average(x)
    ry = _rankdata_average(y)
    return pearson_corr(rx, ry)


def compute_mean_over_indices(A: np.ndarray, idx: List[int]) -> np.ndarray:
    return A[:, idx].mean(axis=1)


def dicts_to_matrix(model_outputs_dicts: List[Dict], fill_value: float = 0.0, id_list: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    M = len(model_outputs_dicts)
    if id_list is None:
        all_ids = set()
        for d in model_outputs_dicts:
            all_ids.update(d.keys())
        ids = sorted(all_ids)
    else:
        ids = list(id_list)

    id2j = {qid: j for j, qid in enumerate(ids)}
    A = np.full((M, len(ids)), fill_value, dtype=float)

    for i, d in enumerate(model_outputs_dicts):
        for qid, val in d.items():
            j = id2j.get(qid)
            if j is None:
                continue
            A[i, j] = float(val)
    return A, ids


def corr_of_subset(A, y, S):
    if len(S) == 0:
        return 0.0
    x = A[:, S].mean(axis=1)
    return pearson_corr(x, y)


def greedy_max_corr_subset(A, y, k, seed=0, candidate_cap=None, return_trace=False):
    rng = np.random.default_rng(seed)
    M, N = A.shape

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
    S = []
    in_set = np.zeros(N, dtype=bool)
    best_corr = -1.0
    trace: List[float] = []

    for _ in range(k):
        best_j = None
        best_val = -1e18
        for j in cand:
            if in_set[j]:
                continue
            val = corr_of_subset(A, y, S + [j])
            if val > best_val:
                best_val = val
                best_j = j
        if best_j is None:
            break
        S.append(best_j)
        in_set[best_j] = True
        best_corr = best_val
        if return_trace:
            trace.append(best_corr)

    return (S, best_corr, trace) if return_trace else (S, best_corr)


def swap_local_search(A, y, S, max_passes=10, sample_in=300, seed=0):
    rng = np.random.default_rng(seed)
    M, N = A.shape
    S = list(S)
    S_set = set(S)
    outside = [j for j in range(N) if j not in S_set]
    cur = corr_of_subset(A, y, S)

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
                val = corr_of_subset(A, y, base + [j_in])
                if val > best_val + 1e-12:
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


def multi_restart_select_subset(
    A,
    y,
    k,
    n_restarts=10,
    greedy_seed0=0,
    swap_passes=10,
    swap_sample_in=300,
    candidate_cap=None,
    return_trace=False,
):
    best_S, best_corr = None, -1e18
    best_trace: List[float] = []
    for r in range(n_restarts):
        seed = greedy_seed0 + r
        if return_trace:
            S, _c, tr = greedy_max_corr_subset(A, y, k=k, seed=seed, candidate_cap=candidate_cap, return_trace=True)
        else:
            S, _c = greedy_max_corr_subset(A, y, k=k, seed=seed, candidate_cap=candidate_cap, return_trace=False)
            tr = []
        S, c1 = swap_local_search(A, y, S, max_passes=swap_passes, sample_in=swap_sample_in, seed=seed)
        if c1 > best_corr:
            best_corr = c1
            best_S = S
            best_trace = tr
    return (best_S, best_corr, best_trace) if return_trace else (best_S, best_corr)


def two_sided_bootstrap_max_sample(A_src, A_tgt, tgt_pool_cols, k_src, k_tgt, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    M, N_src = A_src.shape
    pool = np.asarray(tgt_pool_cols, dtype=int)
    if len(pool) == 0:
        raise ValueError("tgt_pool_cols is empty")

    best_r = -1e18
    best_x = None
    best_y = None

    for _ in range(n_boot):
        idx_l = rng.integers(0, N_src, size=k_src)  # with replacement
        idx_s = rng.choice(pool, size=k_tgt, replace=True)  # with replacement from train pool
        x = compute_mean_over_indices(A_src, idx_l)
        y = compute_mean_over_indices(A_tgt, idx_s)
        r = pearson_corr(x, y)
        if r > best_r:
            best_r = r
            best_x = x
            best_y = y

    return best_x, best_y, float(best_r)


def vote_source_instances_over_target_bootstraps(
    A_src,
    A_tgt,
    tgt_train_cols,
    y_tgt_eval,
    k_src=200,
    tgt_boot_k=400,
    n_outer=50,
    n_restarts_inner=10,
    seed=0,
    swap_passes=10,
    swap_sample_in=300,
    candidate_cap=None,
):
    rng = np.random.default_rng(seed)
    M, N_src = A_src.shape
    pool = np.asarray(tgt_train_cols, dtype=int)

    counts = np.zeros(N_src, dtype=int)
    eval_corrs = np.zeros(n_outer, dtype=float)
    boot_corrs = np.zeros(n_outer, dtype=float)

    print_every = max(1, n_outer // 10)

    for t in range(n_outer):
        idx_s = rng.choice(pool, size=tgt_boot_k, replace=True)
        y_boot = compute_mean_over_indices(A_tgt, idx_s)

        S, best_corr = multi_restart_select_subset(
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

        counts[S] += 1

        x_S = compute_mean_over_indices(A_src, S)
        eval_corr = pearson_corr(x_S, y_tgt_eval)

        eval_corrs[t] = eval_corr
        boot_corrs[t] = best_corr

        if (t + 1) % print_every == 0 or (t + 1) == n_outer:
            print(f"  [vote] iter {t+1}/{n_outer}: best_corr_on_y_boot={best_corr:.4f} | eval_corr={eval_corr:.4f}")

    final_S_vote = np.argsort(-counts)[:k_src].tolist()
    return final_S_vote, counts, eval_corrs, boot_corrs


# ------------------------------------------------------------------------------
# Plotting helpers (with bootstrap uncertainty band for linear fit)
# ------------------------------------------------------------------------------

def plot_scatter_with_corr(
    x,
    y,
    model_names,
    title,
    out_path,
    xlabel="Source mean",
    ylabel="Target mean",
    annotate=True,
    n_boot=500,
    seed=0,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Base correlations
    r_p = pearson_corr(x, y)
    r_s = spearman_corr(x, y)

    # Linear fit
    a_hat, b_hat = np.polyfit(x, y, deg=1)

    # Bootstrap correlation + fit
    rng = np.random.default_rng(seed)
    n = len(x)

    r_boot_p = np.zeros(n_boot, dtype=float)
    r_boot_s = np.zeros(n_boot, dtype=float)
    a_boot = np.zeros(n_boot, dtype=float)
    b_boot = np.zeros(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        r_boot_p[i] = pearson_corr(xb, yb)
        r_boot_s[i] = spearman_corr(xb, yb)
        if np.std(xb) > 0:
            a_b, b_b = np.polyfit(xb, yb, deg=1)
        else:
            a_b, b_b = 0.0, yb.mean()
        a_boot[i] = a_b
        b_boot[i] = b_b

    r_p_mean = float(np.mean(r_boot_p))
    r_p_std = float(np.std(r_boot_p, ddof=1))
    r_s_mean = float(np.mean(r_boot_s))
    r_s_std = float(np.std(r_boot_s, ddof=1))

    xs = np.linspace(x.min(), x.max(), 200)
    y_hat_mean = a_hat * xs + b_hat
    y_hat_boot = np.outer(a_boot, xs) + b_boot[:, None]
    y_hat_std = y_hat_boot.std(axis=0, ddof=1)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)
    plt.plot(xs, y_hat_mean, linewidth=2, label="Linear fit")
    plt.fill_between(xs, y_hat_mean - y_hat_std, y_hat_mean + y_hat_std, alpha=0.25, label="±1σ fit (bootstrap)")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(
        f"{title}\nPearson r = {r_p_mean:.4f} ± {r_p_std:.4f} | Spearman ρ = {r_s_mean:.4f} ± {r_s_std:.4f}"
    )
    if annotate:
        for xi, yi, name in zip(x, y, model_names):
            plt.annotate(name, (xi, yi), textcoords="offset points", xytext=(3, 3), fontsize=7, alpha=0.9)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    return {
        "pearson_mean": r_p_mean,
        "pearson_std": r_p_std,
        "spearman_mean": r_s_mean,
        "spearman_std": r_s_std,
    }


def plot_curve(values, title, out_path, xlabel="Iteration", ylabel="Value"):
    values = np.asarray(values, dtype=float)
    plt.figure(figsize=(6, 3))
    plt.plot(np.arange(1, len(values) + 1), values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


# ------------------------------------------------------------------------------
# End-to-end pipeline for a given (source dataset, target dataset)
# ------------------------------------------------------------------------------

def run_pipeline_for_pair(
    src_ds: BenchDataset,
    tgt_ds: BenchDataset,
    omit_models: Optional[List[str]] = None,
    plot_dir: Path = Path("plots"),
    title_prefix: str = "",
    # split + bootstrap configs
    k_source: Optional[int] = None,
    target_train_size: int = 400,
    target_eval_size: int = 100,
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
    omit_models = set(omit_models or [])

    # 1) Intersect models
    src_models = [m for m in src_ds.list_models() if m not in omit_models]
    tgt_models = [m for m in tgt_ds.list_models() if m not in omit_models]
    models = sorted(set(src_models).intersection(tgt_models))
    if not models:
        raise RuntimeError(f"No overlapping model CSVs between source={src_ds.bench_name}:{src_ds.label} and target={tgt_ds.bench_name}:{tgt_ds.label}")

    # 2) Load outputs in same order
    kept_src, src_dicts = src_ds.load_model_outputs_for_models(models)
    kept_tgt, tgt_dicts = tgt_ds.load_model_outputs_for_models(models)

    # Ensure both kept the same order
    if kept_src != kept_tgt:
        common = sorted(set(kept_src).intersection(kept_tgt))
        kept_src, src_dicts = src_ds.load_model_outputs_for_models(common)
        kept_tgt, tgt_dicts = tgt_ds.load_model_outputs_for_models(common)
    model_names = kept_src

    # 3) Matrices (missing ids -> 0)
    A_src, src_ids = dicts_to_matrix(src_dicts, fill_value=0.0)
    A_tgt, tgt_ids = dicts_to_matrix(tgt_dicts, fill_value=0.0)

    # 4) Split target into train/eval
    N_tgt = A_tgt.shape[1]
    if target_train_size + target_eval_size > N_tgt:
        # So we don't hard fail if user sets sizes too large for a subtask
        target_train_size = min(target_train_size, N_tgt)
        target_eval_size = max(0, min(target_eval_size, N_tgt - target_train_size))
        print(f"[WARN] Adjusted target split to train={target_train_size}, eval={target_eval_size} (N={N_tgt})")

    rng_split = np.random.default_rng(split_seed)
    perm = rng_split.permutation(N_tgt)
    tgt_train_cols = perm[:target_train_size].tolist()
    tgt_eval_cols = perm[target_train_size:target_train_size + target_eval_size].tolist()

    y_tgt_eval = compute_mean_over_indices(A_tgt, tgt_eval_cols) if target_eval_size > 0 else compute_mean_over_indices(A_tgt, tgt_train_cols)
    y_tgt_train = compute_mean_over_indices(A_tgt, tgt_train_cols)

    # 5) Two-sided bootstrap max corr sample
    x_boot_max, y_boot_max, boot_max_corr = two_sided_bootstrap_max_sample(
        A_src,
        A_tgt,
        tgt_train_cols,
        k_src=boot_source_k,
        k_tgt=boot_target_k,
        n_boot=n_boot,
        seed=boot_seed,
    )

    # 6) Voting-based subset selection
    if k_source is None:
        k_source = boot_source_k

    S_vote, vote_counts, eval_corrs, boot_corrs = vote_source_instances_over_target_bootstraps(
        A_src,
        A_tgt,
        tgt_train_cols,
        y_tgt_eval,
        k_src=k_source,
        tgt_boot_k=target_train_size,
        n_outer=n_outer,
        n_restarts_inner=n_restarts_inner,
        seed=0,
        swap_passes=swap_passes,
        swap_sample_in=swap_sample_in,
        candidate_cap=candidate_cap,
    )

    x_vote = compute_mean_over_indices(A_src, S_vote)
    corr_eval_p = pearson_corr(x_vote, y_tgt_eval)
    corr_train_p = pearson_corr(x_vote, y_tgt_train)
    corr_eval_s = spearman_corr(x_vote, y_tgt_eval)
    corr_train_s = spearman_corr(x_vote, y_tgt_train)

    # 7) Save JSON and plots
    tag = title_prefix or f"{src_ds.bench_name}_{src_ds.label}__vs__{tgt_ds.bench_name}_{tgt_ds.label}"
    out_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plots
    _ = plot_scatter_with_corr(
        x_boot_max,
        y_boot_max,
        model_names,
        title=f"{tag} | Two-sided Bootstrap MAX (SRC k={boot_source_k}, TGT k={boot_target_k}, n_boot={n_boot})",
        out_path=out_dir / f"{tag}_bootstrap2sided_max_scatter.png",
        xlabel="Source mean (selected)",
        ylabel="Target mean (selected)",
        annotate=annotate,
    )

    _ = plot_scatter_with_corr(
        x_vote,
        y_tgt_eval,
        model_names,
        title=f"{tag} | Voted SRC subset (k={k_source}) on TGT eval\nPearson train={corr_train_p:.4f}, eval={corr_eval_p:.4f}",
        out_path=out_dir / f"{tag}_voted_subset_eval_scatter.png",
        xlabel="Source mean (voted subset)",
        ylabel="Target mean (eval)",
        annotate=annotate,
    )

    plot_curve(
        eval_corrs,
        title=f"{tag} | eval corr per outer iter (TGT eval)",
        out_path=out_dir / f"{tag}_eval_corr_per_iter.png",
        xlabel="Outer iteration",
        ylabel="Eval correlation (Pearson)",
    )

    plot_curve(
        boot_corrs,
        title=f"{tag} | best corr on y_boot per outer iter",
        out_path=out_dir / f"{tag}_best_boot_corr_per_iter.png",
        xlabel="Outer iteration",
        ylabel="Best corr on y_boot (Pearson)",
    )

    # JSON summary
    top_ids = [src_ids for src_ids in ["PLACEHOLDER"]]  # overwritten below
    # Map selected column indices back to source instance IDs
    # Need source_ids from matrix build again (cannot reuse unless we cache)
    # Rebuild quickly using same order
    _A_src_tmp, source_ids = dicts_to_matrix(src_dicts, fill_value=0.0)
    top_ids = [source_ids[j] for j in S_vote]

    out_json = out_dir / f"{tag}_voted_subset_top{k_source}.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "title": tag,
                "source": {"bench": src_ds.bench_name, "label": src_ds.label},
                "target": {"bench": tgt_ds.bench_name, "label": tgt_ds.label},
                "models_used": model_names,
                "params": {
                    "k_source": k_source,
                    "target_train_size": target_train_size,
                    "target_eval_size": target_eval_size,
                    "n_outer": n_outer,
                    "n_restarts_inner": n_restarts_inner,
                    "boot_source_k": boot_source_k,
                    "boot_target_k": boot_target_k,
                    "n_boot": n_boot,
                    "boot_seed": boot_seed,
                    "split_seed": split_seed,
                    "swap_passes": swap_passes,
                    "swap_sample_in": swap_sample_in,
                    "candidate_cap": candidate_cap,
                },
                "corr_train": {"pearson": float(corr_train_p), "spearman": float(corr_train_s)},
                "corr_eval": {"pearson": float(corr_eval_p), "spearman": float(corr_eval_s)},
                "boot_max_corr": float(boot_max_corr),
                "eval_corrs_per_outer_iter": eval_corrs.tolist(),
                "boot_corrs_per_outer_iter": boot_corrs.tolist(),
                "selected_source_instance_ids": top_ids,
            },
            f,
            indent=2,
        )

    print(f"Saved results to: {out_json}")

    return {
        "models": model_names,
        "A_src_shape": A_src.shape,
        "A_tgt_shape": A_tgt.shape,
        "corr_train": {"pearson": float(corr_train_p), "spearman": float(corr_train_s)},
        "corr_eval": {"pearson": float(corr_eval_p), "spearman": float(corr_eval_s)},
        "S_vote_size": len(S_vote),
        "S_vote_indices": S_vote,
        "S_vote_ids": top_ids,
        "eval_corrs": eval_corrs,
        "boot_corrs": boot_corrs,
        "boot_max_corr": boot_max_corr,
        "json_path": str(out_json),
        "plots_dir": str(out_dir),
    }


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bootstrap-based subset selection across standardized_results benchmarks")
    parser.add_argument("--source", required=True, help="Source benchmark name (dir under results/standardized_results)")
    parser.add_argument("--target", required=True, help="Target benchmark name (dir under results/standardized_results)")
    parser.add_argument("--base_dir", default=str(STD_BASE), help="Base directory for standardized_results")

    parser.add_argument("--train_size", type=int, required=True, help="Target train size")
    parser.add_argument("--eval_size", type=int, required=True, help="Target eval size")
    parser.add_argument("--omit_models", type=str, default="", help="Comma-separated list of model names to omit")

    parser.add_argument("--plot_dir", type=str, default="plots/analysis_bootstrap", help="Output directory for plots and JSON")

    parser.add_argument("--boot_source_k", type=int, required=True, help="Source bootstrap sample size (k)")
    parser.add_argument("--boot_target_k", type=int, required=True, help="Target bootstrap sample size (k)")
    parser.add_argument("--n_bootstraps", type=int, default=50, help="Number of two-sided bootstrap samples")

    parser.add_argument("--n_outer", type=int, default=10, help="How many times to optimize with different target bootstraps")
    parser.add_argument("--n_restarts", type=int, default=10, help="Number of greedy+swap restarts per outer iteration")

    parser.add_argument("--k_source", type=int, default=None, help="Final source subset size (default=boot_source_k)")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--boot_seed", type=int, default=0)

    parser.add_argument("--swap_passes", type=int, default=10)
    parser.add_argument("--swap_sample_in", type=int, default=300)
    parser.add_argument("--candidate_cap", type=int, default=None)

    parser.add_argument("--annotate", action="store_true", help="Annotate points in scatter plots")

    parser.add_argument("--run_subtasks", choices=["auto", "none", "source", "target", "both"], default="auto",
                        help="If a benchmark has subtasks, run per-subtask pipelines. 'auto': run for any benchmark that has subtasks against the other's FULL dataset; 'both': run for both sides; 'source'/'target': only that side; 'none': only run FULL vs FULL.")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    src_full, src_subs = discover_benchmark(args.source, base_dir=base_dir)
    tgt_full, tgt_subs = discover_benchmark(args.target, base_dir=base_dir)

    omit = [x.strip() for x in args.omit_models.split(",") if x.strip()]

    runs: List[Tuple[BenchDataset, BenchDataset, str]] = []

    # Always include FULL vs FULL
    runs.append((src_full, tgt_full, f"{args.source}_FULL__vs__{args.target}_FULL"))

    def add_source_subtasks():
        for s in src_subs:
            runs.append((s, tgt_full, f"{args.source}_{s.label}__vs__{args.target}_FULL"))

    def add_target_subtasks():
        for t in tgt_subs:
            runs.append((src_full, t, f"{args.source}_FULL__vs__{args.target}_{t.label}"))

    if args.run_subtasks == "both":
        add_source_subtasks()
        add_target_subtasks()
    elif args.run_subtasks == "source":
        add_source_subtasks()
    elif args.run_subtasks == "target":
        add_target_subtasks()
    elif args.run_subtasks == "auto":
        # run per-subtask for any benchmark that has subtasks
        if src_subs:
            add_source_subtasks()
        if tgt_subs:
            add_target_subtasks()
    else:
        # none -> only full vs full
        pass

    out_all: Dict[str, Dict] = {}
    for src_ds, tgt_ds, tag in runs:
        print("\n=== Running:", tag, "===")
        res = run_pipeline_for_pair(
            src_ds,
            tgt_ds,
            omit_models=omit,
            plot_dir=Path(args.plot_dir) / f"{args.source}_to_{args.target}",
            title_prefix=tag,
            k_source=args.k_source,
            target_train_size=args.train_size,
            target_eval_size=args.eval_size,
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
        out_all[tag] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()
        }

    final_json = Path(args.plot_dir) / f"{args.source}_to_{args.target}" / "summary.json"
    final_json.parent.mkdir(parents=True, exist_ok=True)
    with open(final_json, "w") as f:
        json.dump(out_all, f, indent=2)
    print(f"Saved summary to: {final_json}")


if __name__ == "__main__":
    main()
