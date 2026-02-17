import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# -----------------------------
# Config (generic defaults)
# -----------------------------
PLOT_DIR = "plots"

# Train/eval split on TARGET benchmark
SWE_TRAIN_SIZE = 400
SWE_EVAL_SIZE = 100
SWE_SPLIT_SEED = 0

# Two-sided bootstrap
BOOT_LCB_K = 200     # number of source (A) instances sampled per boot
BOOT_SWE_K = 400     # number of target (B) instances sampled per boot
N_BOOT = 50
BOOT_SEED = 0

# Optimization (voting)
K_LCB = 200
N_OUTER_OPT = 10          # how many times to optimize with different TARGET bootstraps
N_RESTARTS_INNER = 10     # greedy+swap restarts per outer iteration
SWAP_PASSES = 10
SWAP_SAMPLE_IN = 300
CANDIDATE_CAP = None      # e.g. 3000 if source has huge number of instances


# -----------------------------
# Common helpers: corr + plotting (generic labels)
# -----------------------------
def compute_mean_over_indices(A: np.ndarray, idx: List[int]) -> np.ndarray:
    """A: (M,N), idx: list/np array of column indices (can contain repeats)"""
    return A[:, idx].mean(axis=1)


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

    # -----------------------------
    # Base correlation
    # -----------------------------
    r = pearson_corr(x, y)

    # -----------------------------
    # Linear fit (closed form)
    # y = a x + b
    # -----------------------------
    a_hat, b_hat = np.polyfit(x, y, deg=1)

    # -----------------------------
    # Bootstrap correlation + fit
    # -----------------------------
    rng = np.random.default_rng(seed)
    n = len(x)

    r_boot = np.zeros(n_boot, dtype=float)
    a_boot = np.zeros(n_boot, dtype=float)
    b_boot = np.zeros(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)  # resample pairs
        xb = x[idx]
        yb = y[idx]

        r_boot[i] = pearson_corr(xb, yb)

        # Guard against degenerate resamples
        if np.std(xb) > 0:
            a_b, b_b = np.polyfit(xb, yb, deg=1)
        else:
            a_b, b_b = 0.0, yb.mean()

        a_boot[i] = a_b
        b_boot[i] = b_b

    r_mean = float(np.mean(r_boot))
    r_std = float(np.std(r_boot, ddof=1))

    # -----------------------------
    # Fit line + uncertainty band
    # -----------------------------
    xs = np.linspace(x.min(), x.max(), 200)

    y_hat_mean = a_hat * xs + b_hat
    y_hat_boot = np.outer(a_boot, xs) + b_boot[:, None]

    y_hat_std = y_hat_boot.std(axis=0, ddof=1)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)

    # Mean fit line
    plt.plot(xs, y_hat_mean, linewidth=2, label="Linear fit")

    # ±1 std band from bootstrap
    plt.fill_between(
        xs,
        y_hat_mean - y_hat_std,
        y_hat_mean + y_hat_std,
        alpha=0.25,
        label="±1σ fit (bootstrap)",
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(
        f"{title}\n"
        f"Pearson r = {r_mean:.4f} ± {r_std:.4f} (bootstrap)"
    )

    if annotate:
        for xi, yi, name in zip(x, y, model_names):
            plt.annotate(
                name,
                (xi, yi),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=7,
                alpha=0.9,
            )

    plt.legend(fontsize=8)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    return r_mean, r_std


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


# -----------------------------
# dicts -> matrix (missing => 0)
# -----------------------------
def dicts_to_matrix(model_outputs_dicts: List[Dict], fill_value: float = 0.0, id_list: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    model_outputs_dicts: list[dict], length=M
      each dict: {instance_id: float in [0,1]}
    Missing id in a model dict => fill_value (default 0).
    """
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
            j = id2j.get(qid, None)
            if j is None:
                continue
            A[i, j] = float(val)
    return A, ids


# -----------------------------
# Optimization: greedy + swap + multi-restart (NO solver)
# -----------------------------
def corr_of_subset(A, y, S):
    if len(S) == 0:
        return 0.0
    x = A[:, S].mean(axis=1)
    return pearson_corr(x, y)


def greedy_max_corr_subset(A, y, k, seed=0, candidate_cap=None):
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

    return S, best_corr


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


def multi_restart_select_subset(A, y, k, n_restarts=10, greedy_seed0=0,
                               swap_passes=10, swap_sample_in=300, candidate_cap=None):
    best_S, best_corr = None, -1e18
    for r in range(n_restarts):
        seed = greedy_seed0 + r
        S, _ = greedy_max_corr_subset(A, y, k=k, seed=seed, candidate_cap=candidate_cap)
        S, c1 = swap_local_search(A, y, S, max_passes=swap_passes, sample_in=swap_sample_in, seed=seed)
        if c1 > best_corr:
            best_corr = c1
            best_S = S
    return best_S, best_corr


# -----------------------------
# Two-sided bootstrap (bootstrap BOTH source(A) and target(B)) -> pick max corr sample
# -----------------------------
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


# -----------------------------
# Target bootstrap + optimize Source repeatedly + voting (generic)
# -----------------------------
def vote_source_instances_over_target_bootstraps(
    A_src, A_tgt, tgt_train_cols, y_tgt_eval,
    k_src=200, tgt_boot_k=400,
    n_outer=50, n_restarts_inner=10,
    seed=0
):
    rng = np.random.default_rng(seed)
    M, N_src = A_src.shape
    pool = np.asarray(tgt_train_cols, dtype=int)

    counts = np.zeros(N_src, dtype=int)
    eval_corrs = np.zeros(n_outer, dtype=float)
    boot_corrs = np.zeros(n_outer, dtype=float)

    print_every = max(1, n_outer // 10)

    for t in range(n_outer):
        # TARGET bootstrap -> y_boot
        idx_s = rng.choice(pool, size=tgt_boot_k, replace=True)
        y_boot = compute_mean_over_indices(A_tgt, idx_s)

        # optimize SOURCE subset against y_boot
        S, best_corr = multi_restart_select_subset(
            A_src, y_boot, k=k_src,
            n_restarts=n_restarts_inner,
            greedy_seed0=seed + 1000 + t * 13,
            swap_passes=SWAP_PASSES,
            swap_sample_in=SWAP_SAMPLE_IN,
            candidate_cap=CANDIDATE_CAP,
        )

        counts[S] += 1

        # eval corr on fixed TARGET eval set
        x_S = compute_mean_over_indices(A_src, S)
        eval_corr = pearson_corr(x_S, y_tgt_eval)

        eval_corrs[t] = eval_corr
        boot_corrs[t] = best_corr

        if (t + 1) % print_every == 0 or (t + 1) == n_outer:
            print(f"  [vote] iter {t+1}/{n_outer}: best_corr_on_y_boot={best_corr:.4f} | eval_corr={eval_corr:.4f}")

    final_S_vote = np.argsort(-counts)[:k_src].tolist()
    return final_S_vote, counts, eval_corrs, boot_corrs


# -----------------------------
# Generic pipeline runner (data provided by utils)
# -----------------------------
def run_bootstrap_pipeline(
    model_names: List[str],
    A_source: np.ndarray,
    A_target: np.ndarray,
    source_ids: List[str],
    target_ids: List[str],
    plot_dir: str = PLOT_DIR,
    title_prefix: str = "",
    annotate: bool = True,
    k_source: int = K_LCB,
    target_train_size: int = SWE_TRAIN_SIZE,
    target_eval_size: int = SWE_EVAL_SIZE,
    n_outer: int = N_OUTER_OPT,
    n_restarts_inner: int = N_RESTARTS_INNER,
    boot_source_k: int = BOOT_LCB_K,
    boot_target_k: int = BOOT_SWE_K,
    boot_seed: int = BOOT_SEED,
    split_seed: int = SWE_SPLIT_SEED,
):
    assert A_source.shape[0] == len(model_names)
    assert A_target.shape[0] == len(model_names)

    print(f"Models used: {len(model_names)}")
    print(f"Source matrix: {A_source.shape}  |  Target matrix: {A_target.shape}")

    # Target train/eval split
    N_tgt = A_target.shape[1]
    if target_train_size + target_eval_size > N_tgt:
        raise ValueError(f"Target split sizes exceed available instances: {target_train_size}+{target_eval_size} > {N_tgt}")

    rng_split = np.random.default_rng(split_seed)
    perm = rng_split.permutation(N_tgt)
    tgt_train_cols = perm[:target_train_size].tolist()
    tgt_eval_cols = perm[target_train_size:target_train_size + target_eval_size].tolist()

    y_tgt_eval = compute_mean_over_indices(A_target, tgt_eval_cols)
    y_tgt_train = compute_mean_over_indices(A_target, tgt_train_cols)

    print(f"Target split: train={len(tgt_train_cols)} eval={len(tgt_eval_cols)} (total {N_tgt})")

    # Two-sided bootstrap max
    x_boot_max, y_boot_max, boot_max_corr = two_sided_bootstrap_max_sample(
        A_source, A_target, tgt_train_cols,
        k_src=boot_source_k, k_tgt=boot_target_k,
        n_boot=N_BOOT,
        seed=boot_seed,
    )
    print(f"[BOOT MAX] (SRC k={boot_source_k}, TGT k={boot_target_k}, n_boot={N_BOOT}) corr={boot_max_corr:.4f}")

    # Voting-based subset selection with per-iter eval corr tracking
    S_vote, vote_counts, eval_corrs, boot_corrs = vote_source_instances_over_target_bootstraps(
        A_source, A_target, tgt_train_cols, y_tgt_eval,
        k_src=k_source,
        tgt_boot_k=target_train_size,
        n_outer=n_outer,
        n_restarts_inner=n_restarts_inner,
        seed=0,
    )

    # Final voted subset evaluation
    x_vote = compute_mean_over_indices(A_source, S_vote)
    corr_eval = pearson_corr(x_vote, y_tgt_eval)
    corr_train = pearson_corr(x_vote, y_tgt_train)
    print(f"[VOTED SUBSET] size={len(S_vote)} corr(trainTGT_mean)={corr_train:.4f} corr(evalTGT_mean)={corr_eval:.4f}")

    # Save voted subset ids + traces
    top_ids = [source_ids[j] for j in S_vote]
    tag = title_prefix or "bootstrap"
    out_json = Path(plot_dir) / f"{tag}_voted_subset_top{k_source}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {
                "title_prefix": title_prefix,
                "models_used": model_names,
                "k_source": k_source,
                "target_train_size": target_train_size,
                "target_eval_size": target_eval_size,
                "n_outer_opt": n_outer,
                "n_restarts_inner": n_restarts_inner,
                "corr_train": corr_train,
                "corr_eval": corr_eval,
                "boot_max_corr": boot_max_corr,
                "eval_corrs_per_outer_iter": eval_corrs.tolist(),
                "boot_corrs_per_outer_iter": boot_corrs.tolist(),
                "selected_source_instance_ids": top_ids,
            },
            f,
            indent=2,
        )
    print(f"Saved voted subset ids + traces to: {out_json}")

    # Plots
    title_boot = f"{title_prefix} | Two-sided Bootstrap MAX\n(SRC k={boot_source_k}, TGT k={boot_target_k}, n_boot={N_BOOT})"
    plot_scatter_with_corr(
        x_boot_max, y_boot_max, model_names,
        title=title_boot,
        out_path=f"{plot_dir}/{tag}_bootstrap2sided_max_scatter.png",
        xlabel="Source mean (selected)", ylabel="Target mean (selected)",
        annotate=annotate
    )

    plt_title = f"{title_prefix} | Voted SRC subset (k={k_source}) on TGT eval\nr_train={corr_train:.4f}, r_eval={corr_eval:.4f}"
    plot_scatter_with_corr(
        x_vote, y_tgt_eval, model_names,
        title=plt_title,
        out_path=f"{plot_dir}/{tag}_voted_subset_eval_scatter.png",
        xlabel="Source mean (voted subset)", ylabel="Target mean (eval)",
        annotate=annotate
    )

    plot_curve(
        eval_corrs,
        title=f"{title_prefix} | eval corr per outer iter (TGT eval)",
        out_path=f"{plot_dir}/{tag}_eval_corr_per_iter.png",
        xlabel="Outer iteration",
        ylabel="Eval correlation",
    )

    plot_curve(
        boot_corrs,
        title=f"{title_prefix} | best corr on y_boot per outer iter",
        out_path=f"{plot_dir}/{tag}_best_boot_corr_per_iter.png",
        xlabel="Outer iteration",
        ylabel="Best corr on y_boot",
    )

    print(f"Saved plots under: {plot_dir}/")

    return {
        "k_source": k_source,
        "corr_train": corr_train,
        "corr_eval": corr_eval,
        "selected_source_indices": S_vote,
        "selected_source_ids": top_ids,
        "eval_corrs": eval_corrs,
        "boot_corrs": boot_corrs,
        "boot_max_corr": boot_max_corr,
    }
