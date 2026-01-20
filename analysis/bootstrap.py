import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import load_model_results_0  # LiveCodeBench per-task results

import pandas as pd


# -----------------------------
# Config
# -----------------------------
OUTPUT_DIR_LCB = "/home/yueqis/LiveCodeBench/output/"  # contains per-model dirs
SWE_CSV_DIR = "/home/yueqis/LiveCodeBench/analysis/swebench"  # folder of SWE CSVs
PLOT_DIR = "plots"

# SWE CSV schema
SWE_ID_COL = "metadata.instance_id"
SWE_SCORE_COL = "metadata.scores.resolved"  # 0/1 or "unknown"

# Split SWE instances: 400 for optimization, 100 for evaluation
SWE_TRAIN_SIZE = 400
SWE_EVAL_SIZE = 100
SWE_SPLIT_SEED = 0

# Two-sided bootstrap (for "bootstrap max" plot)
BOOT_LCB_K = 200
BOOT_SWE_K = 400
N_BOOT = 50
BOOT_SEED = 0

# Optimization (voting)
K_LCB = 200
N_OUTER_OPT = 50          # how many times to optimize with different SWE bootstraps
N_RESTARTS_INNER = 10     # greedy+swap restarts per outer iteration
SWAP_PASSES = 10
SWAP_SAMPLE_IN = 300
CANDIDATE_CAP = None      # e.g. 3000 if LCB has huge number of instances


# -----------------------------
# Common helpers: corr + plotting
# -----------------------------
def compute_mean_over_indices(A, idx):
    """A: (M,N), idx: list/np array of column indices (can contain repeats)"""
    return A[:, idx].mean(axis=1)


def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    yc = y - y.mean()
    xs = xc.std(ddof=0)
    ys = yc.std(ddof=0)
    if xs == 0 or ys == 0:
        return 0.0
    return float((xc @ yc) / (len(y) * xs * ys))


def plot_scatter_with_corr(x, y, model_names, title, out_path, annotate=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = pearson_corr(x, y)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)
    plt.xlabel("LiveCodeBench performance (mean over selected instances)")
    plt.ylabel("SWE-Bench performance (mean over selected instances)")
    plt.title(f"{title}\nPearson r = {r:.4f}")

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

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return r


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
# LiveCodeBench: dicts -> matrix (missing => 0)
# -----------------------------
def dicts_to_matrix(model_outputs_dicts, fill_value=0.0, id_list=None):
    """
    model_outputs_dicts: list[dict], length=M
      each dict: {question_id: 0/1 or {"pass@1": 0/1} or float}
    Missing qid in a model dict => fill_value (default 0).
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
            if isinstance(val, dict):
                A[i, j] = float(val.get("pass@1", fill_value))
            else:
                A[i, j] = float(val)
    return A, ids


# -----------------------------
# SWE: load per-model CSV -> matrix (missing => 0)
# -----------------------------
def load_swe_csv_as_dict(csv_path, id_col=SWE_ID_COL, score_col=SWE_SCORE_COL):
    """
    Returns dict: {instance_id: 0/1}
    Treats non-numeric (e.g., 'unknown') as 0.
    """
    if pd is None:
        raise ImportError("pandas is required to read SWE CSVs. Please `pip install pandas`.")

    df = pd.read_csv(csv_path)
    if id_col not in df.columns or score_col not in df.columns:
        raise ValueError(
            f"CSV {csv_path} missing required columns. "
            f"Need {id_col} and {score_col}. Got columns: {list(df.columns)[:30]}..."
        )

    inst = df[id_col].astype(str)

    sc_num = pd.to_numeric(df[score_col], errors="coerce")
    n_bad = int(sc_num.isna().sum())
    if n_bad > 0:
        print(f"[WARN] {os.path.basename(csv_path)}: {n_bad} non-numeric SWE scores treated as 0")

    sc = sc_num.fillna(0.0).clip(lower=0.0, upper=1.0)
    return dict(zip(inst.tolist(), sc.astype(float).tolist()))


def swe_dicts_to_matrix(model_swe_dicts, fill_value=0.0, id_list=None):
    """
    model_swe_dicts: list[dict], length=M, each dict {swe_instance_id: 0/1}
    Missing => fill_value.
    """
    M = len(model_swe_dicts)
    if id_list is None:
        all_ids = set()
        for d in model_swe_dicts:
            all_ids.update(d.keys())
        ids = sorted(all_ids)
    else:
        ids = list(id_list)

    id2j = {qid: j for j, qid in enumerate(ids)}
    A = np.full((M, len(ids)), fill_value, dtype=float)

    for i, d in enumerate(model_swe_dicts):
        for qid, val in d.items():
            j = id2j.get(qid, None)
            if j is None:
                continue
            A[i, j] = float(val)
    return A, ids


def load_swe_matrix_for_models(model_names, swe_csv_dir=SWE_CSV_DIR):
    """
    Assumes SWE CSV filenames are: <model_name>.csv
    Returns:
      kept_models: list[str]
      A_swe: (M, N_swe)
      swe_ids: list[str]
    """
    swe_dicts = []
    kept_models = []
    missing = []

    for m in model_names:
        csv_path = os.path.join(swe_csv_dir, f"{m}.csv")
        if not os.path.exists(csv_path):
            missing.append(m)
            continue
        swe_dicts.append(load_swe_csv_as_dict(csv_path))
        kept_models.append(m)

    if missing:
        print(f"[WARN] Missing SWE CSV for {len(missing)} models (skipping them): {missing[:10]}{'...' if len(missing)>10 else ''}")

    A_swe, swe_ids = swe_dicts_to_matrix(swe_dicts, fill_value=0.0)
    return kept_models, A_swe, swe_ids


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
# Two-sided bootstrap (bootstrap BOTH LCB and SWE) -> pick max corr sample
# -----------------------------
def two_sided_bootstrap_max_sample(A_lcb, A_swe, swe_pool_cols, k_lcb, k_swe, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    M, N_lcb = A_lcb.shape
    pool = np.asarray(swe_pool_cols, dtype=int)
    if len(pool) == 0:
        raise ValueError("swe_pool_cols is empty")

    best_r = -1e18
    best_x = None
    best_y = None

    for _ in range(n_boot):
        idx_l = rng.integers(0, N_lcb, size=k_lcb)  # with replacement
        idx_s = rng.choice(pool, size=k_swe, replace=True)  # with replacement from train pool
        x = compute_mean_over_indices(A_lcb, idx_l)
        y = compute_mean_over_indices(A_swe, idx_s)
        r = pearson_corr(x, y)
        if r > best_r:
            best_r = r
            best_x = x
            best_y = y

    return best_x, best_y, float(best_r)


# -----------------------------
# SWE bootstrap + optimize LCB repeatedly + voting
# AND: evaluate corr on SWE eval at every outer iteration (no tqdm)
# -----------------------------
def vote_lcb_instances_over_swe_bootstraps(
    A_lcb, A_swe, swe_train_cols, y_swe_eval,
    k_lcb=200, swe_boot_k=400,
    n_outer=50, n_restarts_inner=10,
    seed=0
):
    rng = np.random.default_rng(seed)
    M, N_lcb = A_lcb.shape
    pool = np.asarray(swe_train_cols, dtype=int)

    counts = np.zeros(N_lcb, dtype=int)
    eval_corrs = np.zeros(n_outer, dtype=float)
    boot_corrs = np.zeros(n_outer, dtype=float)

    print_every = max(1, n_outer // 10)

    for t in range(n_outer):
        # SWE bootstrap -> y_boot
        idx_s = rng.choice(pool, size=swe_boot_k, replace=True)
        y_boot = compute_mean_over_indices(A_swe, idx_s)

        # optimize LCB subset against y_boot
        S, best_corr = multi_restart_select_subset(
            A_lcb, y_boot, k=k_lcb,
            n_restarts=n_restarts_inner,
            greedy_seed0=seed + 1000 + t * 13,
            swap_passes=SWAP_PASSES,
            swap_sample_in=SWAP_SAMPLE_IN,
            candidate_cap=CANDIDATE_CAP,
        )

        counts[S] += 1

        # eval corr on fixed SWE eval set
        x_S = compute_mean_over_indices(A_lcb, S)
        eval_corr = pearson_corr(x_S, y_swe_eval)

        eval_corrs[t] = eval_corr
        boot_corrs[t] = best_corr

        if (t + 1) % print_every == 0 or (t + 1) == n_outer:
            print(f"  [vote] iter {t+1}/{n_outer}: best_corr_on_y_boot={best_corr:.4f} | eval_corr={eval_corr:.4f}")

    final_S_vote = np.argsort(-counts)[:k_lcb].tolist()
    return final_S_vote, counts, eval_corrs, boot_corrs


# -----------------------------
# Main per-task pipeline
# -----------------------------
def get_task_corr_subset(task, plot_dir=PLOT_DIR, annotate=True):
    print(f"\n=== TASK: {task} ===")

    all_models = sorted(os.listdir(OUTPUT_DIR_LCB))

    kept_models, A_swe, swe_ids = load_swe_matrix_for_models(all_models, swe_csv_dir=SWE_CSV_DIR)
    model_names = kept_models

    model_outputs = []
    for m in model_names:
        model_output, _ = load_model_results_0(m, task, OUTPUT_DIR_LCB)
        model_outputs.append(model_output)

    A_lcb, lcb_ids = dicts_to_matrix(model_outputs, fill_value=0.0)

    assert A_lcb.shape[0] == len(model_names)
    assert A_swe.shape[0] == len(model_names)

    print(f"Models used: {len(model_names)}")
    print(f"LCB matrix: {A_lcb.shape}  |  SWE matrix: {A_swe.shape}")

    # SWE train/eval split
    N_swe = A_swe.shape[1]
    if SWE_TRAIN_SIZE + SWE_EVAL_SIZE > N_swe:
        raise ValueError(f"SWE split sizes exceed available instances: {SWE_TRAIN_SIZE}+{SWE_EVAL_SIZE} > {N_swe}")

    rng_split = np.random.default_rng(SWE_SPLIT_SEED)
    perm = rng_split.permutation(N_swe)
    swe_train_cols = perm[:SWE_TRAIN_SIZE].tolist()
    swe_eval_cols = perm[SWE_TRAIN_SIZE:SWE_TRAIN_SIZE + SWE_EVAL_SIZE].tolist()

    y_swe_eval = compute_mean_over_indices(A_swe, swe_eval_cols)
    y_swe_train = compute_mean_over_indices(A_swe, swe_train_cols)

    print(f"SWE split: train={len(swe_train_cols)} eval={len(swe_eval_cols)} (total {N_swe})")

    # Two-sided bootstrap max
    x_boot_max, y_boot_max, boot_max_corr = two_sided_bootstrap_max_sample(
        A_lcb, A_swe, swe_train_cols,
        k_lcb=BOOT_LCB_K, k_swe=BOOT_SWE_K,
        n_boot=N_BOOT,
        seed=BOOT_SEED,
    )
    print(f"[BOOT MAX] (LCB k={BOOT_LCB_K}, SWE k={BOOT_SWE_K}, n_boot={N_BOOT}) corr={boot_max_corr:.4f}")

    # Voting-based subset selection with per-iter eval corr tracking
    S_vote, vote_counts, eval_corrs, boot_corrs = vote_lcb_instances_over_swe_bootstraps(
        A_lcb, A_swe, swe_train_cols, y_swe_eval,
        k_lcb=K_LCB,
        swe_boot_k=SWE_TRAIN_SIZE,
        n_outer=N_OUTER_OPT,
        n_restarts_inner=N_RESTARTS_INNER,
        seed=0,
    )

    # Final voted subset evaluation
    x_vote = compute_mean_over_indices(A_lcb, S_vote)
    corr_eval = pearson_corr(x_vote, y_swe_eval)
    corr_train = pearson_corr(x_vote, y_swe_train)
    print(f"[VOTED SUBSET] size={len(S_vote)} corr(trainSWE_mean)={corr_train:.4f} corr(evalSWE_mean)={corr_eval:.4f}")

    # Save voted subset ids + traces
    top_ids = [lcb_ids[j] for j in S_vote]
    out_json = Path(plot_dir) / f"{task}_voted_subset_top{K_LCB}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {
                "task": task,
                "models_used": model_names,
                "k_lcb": K_LCB,
                "swe_train_size": SWE_TRAIN_SIZE,
                "swe_eval_size": SWE_EVAL_SIZE,
                "n_outer_opt": N_OUTER_OPT,
                "n_restarts_inner": N_RESTARTS_INNER,
                "corr_train": corr_train,
                "corr_eval": corr_eval,
                "boot_max_corr": boot_max_corr,
                "eval_corrs_per_outer_iter": eval_corrs.tolist(),
                "boot_corrs_per_outer_iter": boot_corrs.tolist(),
                "selected_lcb_instance_ids": top_ids,
            },
            f,
            indent=2,
        )
    print(f"Saved voted subset ids + traces to: {out_json}")

    # Plots
    plot_scatter_with_corr(
        x_boot_max, y_boot_max, model_names,
        title=f"{task} | Two-sided Bootstrap MAX\n(LCB k={BOOT_LCB_K}, SWE k={BOOT_SWE_K}, n_boot={N_BOOT})",
        out_path=f"{plot_dir}/{task}_bootstrap2sided_max_scatter.png",
        annotate=annotate
    )

    plt_title = f"{task} | Voted LCB subset (k={K_LCB}) on SWE eval\nr_train={corr_train:.4f}, r_eval={corr_eval:.4f}"
    plot_scatter_with_corr(
        x_vote, y_swe_eval, model_names,
        title=plt_title,
        out_path=f"{plot_dir}/{task}_voted_subset_eval_scatter.png",
        annotate=annotate
    )

    plot_curve(
        eval_corrs,
        title=f"{task} | eval corr per outer iter (SWE eval)",
        out_path=f"{plot_dir}/{task}_eval_corr_per_iter.png",
        xlabel="Outer iteration",
        ylabel="Eval correlation",
    )

    plot_curve(
        boot_corrs,
        title=f"{task} | best corr on y_boot per outer iter",
        out_path=f"{plot_dir}/{task}_best_boot_corr_per_iter.png",
        xlabel="Outer iteration",
        ylabel="Best corr on y_boot",
    )

    print(f"Saved plots under: {plot_dir}/")


if __name__ == "__main__":
    get_task_corr_subset("codegeneration", annotate=True)
    get_task_corr_subset("codeexecution", annotate=True)
    get_task_corr_subset("selfrepair", annotate=True)
    get_task_corr_subset("testoutputprediction", annotate=True)
