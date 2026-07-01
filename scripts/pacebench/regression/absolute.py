"""Task A: absolute-score prediction.

Step 4 of the pipeline (regression): Spearman-weighted 1-D projection + OLS on
the pooled bootstrap dataset (U_pool, y_pool). Produces scalar prediction ŷ_{k, t}.

Fold functions `_fold_H_abs` / `_fold_Dpt_abs` run one LOMO fold (held-out model i);
LOOCV wrappers `loocv_H_abs` / `loocv_Dpt_abs` parallelise over the 14 folds.
"""

import numpy as np
from joblib import Parallel, delayed

from pacebench.config import N_JOBS
from pacebench.stats import spearman_X_vs_Y, ols_1d
from pacebench.embedding import h_embed, dpt_embed
from pacebench.bootstrap import pooled_absolute


# ──────────────────────────────────────────────────────────────
# Core regression step (shared by both H and D-pt paths)
# ──────────────────────────────────────────────────────────────

def _fit_predict_abs(U_pool, y_pool, u_te):
    """Spearman-weighted 1-D projection + 1-D OLS."""
    corrs = spearman_X_vs_Y(U_pool, y_pool.reshape(-1, 1))[:, 0]
    w = corrs / (np.abs(corrs).sum() + 1e-12)
    return ols_1d(U_pool @ w, y_pool, float(u_te @ w))


# ──────────────────────────────────────────────────────────────
# Per-fold functions (one for each embedding strategy)
# ──────────────────────────────────────────────────────────────

def _fold_H_abs(i, X_valid, Y, P_target, nc, M, B, seed):
    n, T = X_valid.shape[0], Y.shape[1]
    tr = [j for j in range(n) if j != i]
    X_tr = X_valid[tr].astype(np.float64)
    preds_i = np.zeros(T)
    for t_idx in range(T):
        rng = np.random.RandomState(seed + i * 997 + t_idx * 31)
        y_tr = Y[tr, t_idx]
        P_t  = P_target[t_idx][tr].astype(np.float64)
        U_tr, u_te = h_embed(X_valid, i, X_tr, y_tr, M, nc)
        U_pool, y_pool = pooled_absolute(U_tr, P_t, B, rng)
        preds_i[t_idx] = _fit_predict_abs(U_pool, y_pool, u_te)
    return preds_i


def _fold_Dpt_abs(i, X_valid, Y, Vt_full, P_target, nc, N, B, seed):
    n, T = X_valid.shape[0], Y.shape[1]
    tr = [j for j in range(n) if j != i]
    X_tr = X_valid[tr].astype(np.float64)
    preds_i = np.zeros(T)
    for t_idx in range(T):
        rng = np.random.RandomState(seed + i * 997 + t_idx * 31)
        y_tr = Y[tr, t_idx]
        P_t  = P_target[t_idx][tr].astype(np.float64)
        U_tr, u_te = dpt_embed(X_valid, i, X_tr, y_tr, Vt_full, N, nc)
        U_pool, y_pool = pooled_absolute(U_tr, P_t, B, rng)
        preds_i[t_idx] = _fit_predict_abs(U_pool, y_pool, u_te)
    return preds_i


# ──────────────────────────────────────────────────────────────
# LOOCV parallel wrappers — 14 folds
# ──────────────────────────────────────────────────────────────

def loocv_H_abs(X_v, Y, nc, M, P_target, B, seed):
    """Task A LOMO with H embedding. Returns (n, T) predictions."""
    n = X_v.shape[0]
    res = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(_fold_H_abs)(i, X_v, Y, P_target, nc, M, B, seed)
        for i in range(n))
    return np.array(res)


def loocv_Dpt_abs(X_v, Y, Vt_full, nc, N, P_target, B, seed):
    """Task A LOMO with D-pt embedding. Returns (n, T) predictions."""
    n = X_v.shape[0]
    res = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(_fold_Dpt_abs)(i, X_v, Y, Vt_full, P_target, nc, N, B, seed)
        for i in range(n))
    return np.array(res)


# ──────────────────────────────────────────────────────────────
# Strict-budget joint LOMO: |H ∪ D| = C per (fold, target).
# H and D-pt selections are coupled via joint_top_C; regressions stay
# independent (so the m-mix in cli.py is unchanged).
# ──────────────────────────────────────────────────────────────

def _fold_joint_abs(i, X_valid, Y, Vt_full, per_target_qC, nc_h_per_t,
                     nc_d_per_t, P_target, B, seed):
    """One LOMO fold for strict-budget joint selection.

    per_target_qC : dict {t_idx: (q, C)} — joint (ratio, total budget) per target.
    nc_h_per_t    : dict {t_idx: nc_h}   — H embedding rank per target.
    nc_d_per_t    : dict {t_idx: nc_d}   — D-pt embedding rank per target.

    Returns (preds_H_i, preds_D_i, sel_per_t) where sel_per_t is a list of
    (h_idx, d_idx) one entry per target.
    """
    from pacebench.selection import joint_top_C
    n, T = X_valid.shape[0], Y.shape[1]
    tr = [j for j in range(n) if j != i]
    X_tr = X_valid[tr].astype(np.float64)
    preds_H_i = np.zeros(T)
    preds_D_i = np.zeros(T)
    sel_per_t = []
    for t_idx in range(T):
        if t_idx not in per_target_qC:
            sel_per_t.append((np.array([], dtype=np.int64),
                              np.array([], dtype=np.int64)))
            continue
        q, C = per_target_qC[t_idx]
        nc_h, nc_d = nc_h_per_t[t_idx], nc_d_per_t[t_idx]
        y_tr = Y[tr, t_idx]
        P_t  = P_target[t_idx][tr].astype(np.float64)
        h_idx, d_idx = joint_top_C(X_tr, y_tr, Vt_full, C, nc_d, q)
        sel_per_t.append((h_idx, d_idx))
        # H regression — same RNG seed as the standalone H path (matches
        # _fold_H_abs determinism so non-strict and strict are comparable
        # when no overlap-driven growth happens).
        if len(h_idx) > 0:
            rng_H = np.random.RandomState(seed + i * 997 + t_idx * 31)
            U_H_tr, u_H_te = h_embed(X_valid, i, X_tr, y_tr,
                                      len(h_idx), nc_h, pinned_sel=h_idx)
            U_pool, y_pool = pooled_absolute(U_H_tr, P_t, B, rng_H)
            preds_H_i[t_idx] = _fit_predict_abs(U_pool, y_pool, u_H_te)
        else:
            preds_H_i[t_idx] = float(P_t.mean())
        # D-pt regression — fresh RNG with same seed.
        if len(d_idx) > 0:
            rng_D = np.random.RandomState(seed + i * 997 + t_idx * 31)
            U_D_tr, u_D_te = dpt_embed(X_valid, i, X_tr, y_tr, Vt_full,
                                        len(d_idx), nc_d, pinned_sel=d_idx)
            U_pool, y_pool = pooled_absolute(U_D_tr, P_t, B, rng_D)
            preds_D_i[t_idx] = _fit_predict_abs(U_pool, y_pool, u_D_te)
        else:
            preds_D_i[t_idx] = float(P_t.mean())
    return preds_H_i, preds_D_i, sel_per_t


def fit_all_joint_abs(X_v, Y, Vt_full, per_target_qC, nc_h_per_t, nc_d_per_t,
                       P_target, B, seed):
    """Strict-budget fit-all (in-sample): |H ∪ D| = C per target, all n models
    used for both fit and prediction.

    Returns (preds_H, preds_D, selections):
      preds_H, preds_D : (n, T) ndarrays
      selections       : list[T] of (h_idx, d_idx) tuples (one fold = ALL n)
    """
    from pacebench.selection import joint_top_C
    from pacebench.embedding import h_embed_all, dpt_embed_all
    n, T = X_v.shape[0], Y.shape[1]
    X_all = X_v.astype(np.float64)
    preds_H = np.zeros((n, T))
    preds_D = np.zeros((n, T))
    selections = []
    for t_idx in range(T):
        if t_idx not in per_target_qC:
            selections.append((np.array([], dtype=np.int64),
                               np.array([], dtype=np.int64)))
            continue
        q, C = per_target_qC[t_idx]
        nc_h, nc_d = nc_h_per_t[t_idx], nc_d_per_t[t_idx]
        y_all = Y[:, t_idx]
        P_t   = P_target[t_idx].astype(np.float64)
        h_idx, d_idx = joint_top_C(X_all, y_all, Vt_full, C, nc_d, q)
        selections.append((h_idx, d_idx))
        # H regression (in-sample on all n models)
        if len(h_idx) > 0:
            rng_H = np.random.RandomState(seed + t_idx * 31)
            U_H_all = h_embed_all(X_all, y_all, len(h_idx), nc_h, pinned_sel=h_idx)
            U_pool, y_pool = pooled_absolute(U_H_all, P_t, B, rng_H)
            corrs = spearman_X_vs_Y(U_pool, y_pool.reshape(-1, 1))[:, 0]
            w = corrs / (np.abs(corrs).sum() + 1e-12)
            x_pool = U_pool @ w
            for i in range(n):
                preds_H[i, t_idx] = ols_1d(x_pool, y_pool, float(U_H_all[i] @ w))
        else:
            preds_H[:, t_idx] = float(P_t.mean())
        # D-pt regression (in-sample)
        if len(d_idx) > 0:
            rng_D = np.random.RandomState(seed + t_idx * 31)
            U_D_all = dpt_embed_all(X_all, y_all, Vt_full, len(d_idx), nc_d,
                                     pinned_sel=d_idx)
            U_pool, y_pool = pooled_absolute(U_D_all, P_t, B, rng_D)
            corrs = spearman_X_vs_Y(U_pool, y_pool.reshape(-1, 1))[:, 0]
            w = corrs / (np.abs(corrs).sum() + 1e-12)
            x_pool = U_pool @ w
            for i in range(n):
                preds_D[i, t_idx] = ols_1d(x_pool, y_pool, float(U_D_all[i] @ w))
        else:
            preds_D[:, t_idx] = float(P_t.mean())
    return preds_H, preds_D, selections


def loocv_joint_abs(X_v, Y, Vt_full, per_target_qC, nc_h_per_t, nc_d_per_t,
                     P_target, B, seed):
    """Strict-budget joint LOMO across n folds.

    Returns (preds_H, preds_D, selections) where:
      preds_H, preds_D : (n, T) ndarrays
      selections       : list of length n; each element is a list of length T
                         containing (h_idx, d_idx) tuples for that fold-target.
    """
    n = X_v.shape[0]
    res = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(_fold_joint_abs)(i, X_v, Y, Vt_full, per_target_qC,
                                  nc_h_per_t, nc_d_per_t, P_target, B, seed)
        for i in range(n))
    preds_H = np.array([r[0] for r in res])
    preds_D = np.array([r[1] for r in res])
    selections = [r[2] for r in res]
    return preds_H, preds_D, selections


# ──────────────────────────────────────────────────────────────
# Strict-budget AUTO-TUNE: full sweep over (q, nc_d, nc_h) at fixed C.
# Per-fold worker computes ALL combinations in one pass with shared y_pool
# (~10× speedup vs naive: y_pool depends only on (i, t_idx, B, seed) and is
# identical across all (q, nc_h, nc_d)).
# ──────────────────────────────────────────────────────────────

def _fold_strict_autotune(i, X_valid, Y, Vt_full, q_grid, nc_h_grid, nc_d_grid,
                           count, P_target, B, seed):
    """One fold worker: returns (h_records, d_records, sel_records).

    h_records   : dict[(t_idx, q, nc_h, nc_d)] → scalar pred
    d_records   : dict[(t_idx, q, nc_d)]       → scalar pred
    sel_records : dict[(t_idx, q, nc_d)]       → (h_idx, d_idx) ndarrays
    """
    from pacebench.selection import joint_top_C
    n, T = X_valid.shape[0], Y.shape[1]
    tr = [j for j in range(n) if j != i]
    X_tr = X_valid[tr].astype(np.float64)

    h_records, d_records, sel_records = {}, {}, {}
    for t_idx in range(T):
        y_tr = Y[tr, t_idx]
        P_t  = P_target[t_idx][tr].astype(np.float64)
        n_tr = P_t.shape[0]; n_tgt = P_t.shape[1]
        # Pre-compute y_pool ONCE: identical across all (q, nc_h, nc_d) at fixed
        # (i, t_idx) — same RNG seed as the standalone H/D-pt paths so results
        # match those paths when the same selection is chosen.
        rng_pool = np.random.RandomState(seed + i * 997 + t_idx * 31)
        y_pool = np.zeros(B * n_tr)
        for b in range(B):
            bi_t = rng_pool.choice(n_tgt, n_tgt, replace=True)
            y_pool[b*n_tr:(b+1)*n_tr] = P_t[:, bi_t].mean(axis=1)
        for q_val in q_grid:
            for nc_d in nc_d_grid:
                h_idx, d_idx = joint_top_C(X_tr, y_tr, Vt_full, count, nc_d, q_val)
                sel_records[(t_idx, q_val, nc_d)] = (h_idx, d_idx)
                # D-pt prediction (one per (q, nc_d))
                if len(d_idx) > 0:
                    U_D_tr, u_D_te = dpt_embed(X_valid, i, X_tr, y_tr, Vt_full,
                                                len(d_idx), nc_d, pinned_sel=d_idx)
                    U_pool = np.tile(U_D_tr, (B, 1))
                    d_records[(t_idx, q_val, nc_d)] = _fit_predict_abs(U_pool, y_pool, u_D_te)
                else:
                    d_records[(t_idx, q_val, nc_d)] = float(P_t.mean())
                # H predictions (one per (q, nc_h, nc_d))
                if len(h_idx) > 0:
                    for nc_h in nc_h_grid:
                        U_H_tr, u_H_te = h_embed(X_valid, i, X_tr, y_tr,
                                                  len(h_idx), nc_h, pinned_sel=h_idx)
                        U_pool = np.tile(U_H_tr, (B, 1))
                        h_records[(t_idx, q_val, nc_h, nc_d)] = _fit_predict_abs(
                            U_pool, y_pool, u_H_te)
                else:
                    fallback = float(P_t.mean())
                    for nc_h in nc_h_grid:
                        h_records[(t_idx, q_val, nc_h, nc_d)] = fallback
    return h_records, d_records, sel_records


def loocv_joint_abs_autotune(X_v, Y, Vt_full, q_grid, nc_h_grid, nc_d_grid,
                              count, P_target, B, seed):
    """Strict-budget LOMO across n folds, computing ALL (q, nc_h, nc_d) combos
    in a single pass per fold.

    Returns:
      H_cache  : dict[(q, nc_h, nc_d)] → (n, T) preds
      D_cache  : dict[(q, nc_d)]       → (n, T) preds
      sel_cache: dict[(q, nc_d)]       → list of length n; each list[T] of (h_idx, d_idx)
    """
    n, T = X_v.shape[0], Y.shape[1]
    res = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(_fold_strict_autotune)(i, X_v, Y, Vt_full, q_grid, nc_h_grid,
                                        nc_d_grid, count, P_target, B, seed)
        for i in range(n))

    H_cache = {(q_val, nc_h, nc_d): np.zeros((n, T))
               for q_val in q_grid for nc_h in nc_h_grid for nc_d in nc_d_grid}
    D_cache = {(q_val, nc_d): np.zeros((n, T))
               for q_val in q_grid for nc_d in nc_d_grid}
    sel_cache = {(q_val, nc_d): [[None] * T for _ in range(n)]
                 for q_val in q_grid for nc_d in nc_d_grid}

    for i, (h_rec, d_rec, sel_rec) in enumerate(res):
        for (t_idx, q_val, nc_h, nc_d), v in h_rec.items():
            H_cache[(q_val, nc_h, nc_d)][i, t_idx] = v
        for (t_idx, q_val, nc_d), v in d_rec.items():
            D_cache[(q_val, nc_d)][i, t_idx] = v
        for (t_idx, q_val, nc_d), v in sel_rec.items():
            sel_cache[(q_val, nc_d)][i][t_idx] = v
    return H_cache, D_cache, sel_cache


# ──────────────────────────────────────────────────────────────
# In-sample fit (all n models — no held-out). For analysis purposes only.
# ──────────────────────────────────────────────────────────────

def fit_all_H_abs(X_v, Y, P_target, nc, M, B, seed):
    """Task A in-sample fit with H embedding. Returns (n, T) predictions."""
    from pacebench.embedding import h_embed_all
    n, T = X_v.shape[0], Y.shape[1]
    X_all = X_v.astype(np.float64)
    preds = np.zeros((n, T))
    for t_idx in range(T):
        rng = np.random.RandomState(seed + t_idx * 31)
        y_all = Y[:, t_idx]
        P_t   = P_target[t_idx].astype(np.float64)
        U_all = h_embed_all(X_all, y_all, M, nc)
        U_pool, y_pool = pooled_absolute(U_all, P_t, B, rng)
        # Fit once, predict in-sample for all 14 models
        corrs = spearman_X_vs_Y(U_pool, y_pool.reshape(-1, 1))[:, 0]
        w = corrs / (np.abs(corrs).sum() + 1e-12)
        x_pool = U_pool @ w
        for i in range(n):
            preds[i, t_idx] = ols_1d(x_pool, y_pool, float(U_all[i] @ w))
    return preds


# ──────────────────────────────────────────────────────────────
# Custom split: arbitrary train / eval model sets
# Used by --train-models / --eval-models CLI mode AND (optionally) to dump selections
# ──────────────────────────────────────────────────────────────

def fit_split_H_abs(X_v, Y, P_target, train_idx, eval_idx, nc, M, B, seed):
    """
    Task A custom-split with H embedding.

    Returns:
      preds : (len(eval_idx), T)
      sel   : dict[t_idx] → 1-D array of selected column indices (into X_v)
    """
    from pacebench.embedding import h_embed_split
    n, T = X_v.shape[0], Y.shape[1]
    X_v64 = X_v.astype(np.float64)
    preds = np.zeros((len(eval_idx), T))
    sel_out = {}
    for t_idx in range(T):
        rng = np.random.RandomState(seed + t_idx * 31)
        y_tr = Y[train_idx, t_idx]
        P_t  = P_target[t_idx][train_idx].astype(np.float64)
        U_all, top_M_idx = h_embed_split(X_v64, train_idx, y_tr, M, nc)
        sel_out[t_idx] = top_M_idx
        U_tr = U_all[train_idx]
        U_pool, y_pool = pooled_absolute(U_tr, P_t, B, rng)
        corrs = spearman_X_vs_Y(U_pool, y_pool.reshape(-1, 1))[:, 0]
        w = corrs / (np.abs(corrs).sum() + 1e-12)
        x_pool = U_pool @ w
        for e_pos, e_i in enumerate(eval_idx):
            preds[e_pos, t_idx] = ols_1d(x_pool, y_pool, float(U_all[e_i] @ w))
    return preds, sel_out


def fit_split_Dpt_abs(X_v, Y, Vt_full, P_target, train_idx, eval_idx, nc, N, B, seed):
    """Task A custom-split with D-pt embedding. Same return shape as fit_split_H_abs."""
    from pacebench.embedding import dpt_embed_split
    n, T = X_v.shape[0], Y.shape[1]
    X_v64 = X_v.astype(np.float64)
    preds = np.zeros((len(eval_idx), T))
    sel_out = {}
    for t_idx in range(T):
        rng = np.random.RandomState(seed + t_idx * 31)
        y_tr = Y[train_idx, t_idx]
        P_t  = P_target[t_idx][train_idx].astype(np.float64)
        U_all, sel = dpt_embed_split(X_v64, train_idx, y_tr, Vt_full, N, nc)
        sel_out[t_idx] = sel
        U_tr = U_all[train_idx]
        U_pool, y_pool = pooled_absolute(U_tr, P_t, B, rng)
        corrs = spearman_X_vs_Y(U_pool, y_pool.reshape(-1, 1))[:, 0]
        w = corrs / (np.abs(corrs).sum() + 1e-12)
        x_pool = U_pool @ w
        for e_pos, e_i in enumerate(eval_idx):
            preds[e_pos, t_idx] = ols_1d(x_pool, y_pool, float(U_all[e_i] @ w))
    return preds, sel_out


# ──────────────────────────────────────────────────────────────
# LOMO selection collector — runs ONLY the selection step across the 14 folds,
# no regression. Cheap: ~14 × T Spearman computations per (strategy, nc). Used
# by --dump-selections when the CLI is in LOMO mode.
# ──────────────────────────────────────────────────────────────

def _ols_slope(x_tr, y_tr):
    """OLS slope with the same degeneracy guards as stats.ols_1d."""
    x_tr = np.asarray(x_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    xm = x_tr.mean(); ym = y_tr.mean()
    x_std = x_tr.std()
    if x_std < 1e-8 * max(abs(xm), 1e-12):
        return 0.0
    var = ((x_tr - xm) ** 2).sum()
    if var < 1e-12:
        return 0.0
    return float(((x_tr - xm) * (y_tr - ym)).sum() / var)


def _resolve_count_for_target(count, tname):
    """Accept scalar `count` or {target: count} dict. Returns int (0 → skip)."""
    if isinstance(count, dict):
        return int(count.get(tname, 0))
    return int(count)


def collect_lomo_weights_abs(X_v, Y, Vt_full, target_names, cfgs,
                              P_target, B, seed, count, strategy,
                              pinned_sel_by_fold_t=None):
    """
    Per-fold per-target per-instance weights in Task A's final linear predictor.

    For held-out model i at target t, the predictor is:
        ŷ(i, t) = m · (slope_H · Σ_s a_H[s] · X[i,s] + b_H)
              + (1-m) · (slope_D · Σ_s a_D[s] · X[i,s] + b_D)
    where
        a_H[s] = (V_k.T @ diag(1/S_k) @ w_dim)[s]        (for H)
        a_D[s] = (pinv(V_sel).T @ w_dim)[s]              (for D-pt)
    The `weight` below is the coefficient m_factor · slope · a[s] on X[i, s].

    `count` may be a scalar (applied to every target) or a {target: count} dict.
    `pinned_sel_by_fold_t` (optional): nested dict {fold_idx: {t_idx: ndarray}}.
    When provided, selection is skipped for that (fold, target) and the given
    indices are used directly; the m=0 skip is bypassed (weight=0 still emits).
    """
    from pacebench.selection import h_top_M, dpt_top_N
    from pacebench.bootstrap import pooled_absolute
    from pacebench.stats import spearman_X_vs_Y
    from numpy.linalg import svd as np_svd, pinv

    n = X_v.shape[0]
    X_v64 = X_v.astype(np.float64)
    rows = []
    for t_idx, tname in enumerate(target_names):
        if tname not in cfgs: continue
        cfg = cfgs[tname]
        m = cfg["m"]
        m_factor = m if strategy == "H" else (1.0 - m)
        # Need pinned to be present for at least one fold to bypass m=0 skip
        any_pinned_for_t = (pinned_sel_by_fold_t is not None and any(
            pinned_sel_by_fold_t.get(i, {}).get(t_idx) is not None for i in range(n)))
        if not any_pinned_for_t and m_factor == 0: continue
        cnt_t = _resolve_count_for_target(count, tname)
        if not any_pinned_for_t and cnt_t == 0: continue
        nc = cfg["nc_h"] if strategy == "H" else cfg["nc_d"]
        for i in range(n):
            tr = [k for k in range(n) if k != i]
            X_tr = X_v64[tr]
            y_tr = Y[tr, t_idx]
            P_t = P_target[t_idx][tr].astype(np.float64)
            rng = np.random.RandomState(seed + i * 997 + t_idx * 31)
            sel_pinned = (pinned_sel_by_fold_t.get(i, {}).get(t_idx)
                          if pinned_sel_by_fold_t is not None else None)
            if sel_pinned is not None and len(sel_pinned) == 0:
                continue
            if strategy == "H":
                sel = sel_pinned if sel_pinned is not None else h_top_M(X_tr, y_tr, cnt_t)
                _, S_w, Vt_w = np_svd(X_tr[:, sel], full_matrices=False)
                nc_ = min(nc, Vt_w.shape[0])
                S_k = S_w[:nc_] + 1e-12
                Vt_k = Vt_w[:nc_, :]
                PROJ = Vt_k.T * (1.0 / S_k)      # (|sel|, nc_)
            else:  # D-pt
                sel = (sel_pinned if sel_pinned is not None
                       else dpt_top_N(X_tr, y_tr, Vt_full, cnt_t, nc))
                V_sel = Vt_full[:nc, :][:, sel].T   # (|sel|, nc)
                PROJ = pinv(V_sel).T                # (|sel|, nc)
            U_tr = X_tr[:, sel] @ PROJ              # (n_tr, nc)
            U_pool, y_pool = pooled_absolute(U_tr, P_t, B, rng)
            corrs = spearman_X_vs_Y(U_pool, y_pool.reshape(-1, 1))[:, 0]
            w_dim = corrs / (np.abs(corrs).sum() + 1e-12)
            slope = _ols_slope(U_pool @ w_dim, y_pool)
            a_vec = PROJ @ w_dim                    # (|sel|,)
            weights = m_factor * slope * a_vec
            for rank, (col, w) in enumerate(zip(sel, weights)):
                rows.append(dict(held_out_idx=i, target=tname, strategy=strategy,
                                 nc=nc, rank=rank + 1, col_idx=int(col),
                                 weight=float(w)))
    return rows


def collect_fit_all_weights_abs(X_v, Y, Vt_full, target_names, cfgs,
                                 P_target, B, seed, count, strategy,
                                 pinned_sel_by_t=None):
    """Fit-all (all n models) variant of collect_lomo_weights_abs. fold = 'ALL'.

    `count` may be a scalar or a {target: count} dict (per-target budget).
    `pinned_sel_by_t` (optional): dict {t_idx: ndarray of col indices}. When
    provided for a target, the selection step is skipped and the given indices
    are used directly (e.g., from joint_top_C for strict-budget). The m=0 skip
    is bypassed for pinned targets so weight=0 rows still appear in the dump.
    """
    from pacebench.selection import h_top_M, dpt_top_N
    from pacebench.bootstrap import pooled_absolute
    from pacebench.stats import spearman_X_vs_Y
    from numpy.linalg import svd as np_svd, pinv

    X_v64 = X_v.astype(np.float64)
    rows = []
    for t_idx, tname in enumerate(target_names):
        if tname not in cfgs: continue
        cfg = cfgs[tname]
        m = cfg["m"]
        m_factor = m if strategy == "H" else (1.0 - m)
        sel_pinned = (pinned_sel_by_t.get(t_idx)
                      if pinned_sel_by_t is not None else None)
        # Skip target only if there's nothing to do (no pinned AND m_factor==0)
        if sel_pinned is None and m_factor == 0: continue
        cnt_t = _resolve_count_for_target(count, tname)
        if sel_pinned is None and cnt_t == 0: continue
        nc = cfg["nc_h"] if strategy == "H" else cfg["nc_d"]
        y_all = Y[:, t_idx]
        P_t = P_target[t_idx].astype(np.float64)
        rng = np.random.RandomState(seed + t_idx * 31)
        if strategy == "H":
            sel = sel_pinned if sel_pinned is not None else h_top_M(X_v64, y_all, cnt_t)
            _, S_w, Vt_w = np_svd(X_v64[:, sel], full_matrices=False)
            nc_ = min(nc, Vt_w.shape[0])
            S_k = S_w[:nc_] + 1e-12
            Vt_k = Vt_w[:nc_, :]
            PROJ = Vt_k.T * (1.0 / S_k)
        else:
            sel = (sel_pinned if sel_pinned is not None
                   else dpt_top_N(X_v64, y_all, Vt_full, cnt_t, nc))
            V_sel = Vt_full[:nc, :][:, sel].T
            PROJ = pinv(V_sel).T
        U_all = X_v64[:, sel] @ PROJ
        U_pool, y_pool = pooled_absolute(U_all, P_t, B, rng)
        corrs = spearman_X_vs_Y(U_pool, y_pool.reshape(-1, 1))[:, 0]
        w_dim = corrs / (np.abs(corrs).sum() + 1e-12)
        slope = _ols_slope(U_pool @ w_dim, y_pool)
        a_vec = PROJ @ w_dim
        weights = m_factor * slope * a_vec
        for rank, (col, w) in enumerate(zip(sel, weights)):
            rows.append(dict(held_out_idx=-1, target=tname, strategy=strategy,
                             nc=nc, rank=rank + 1, col_idx=int(col),
                             weight=float(w)))
    return rows


def collect_fit_all_selections(X_v, Y, Vt_full, target_names, cfgs, strategy, count):
    """Same as collect_lomo_selections but for fit-all mode: one selection per target
    trained on all n models (no held-out). Returns rows with fold='ALL'.

    `count` may be a scalar or a {target: count} dict (per-target budget).
    """
    from pacebench.selection import h_top_M, dpt_top_N
    X_v64 = X_v.astype(np.float64)
    rows = []
    for t_idx, tname in enumerate(target_names):
        cfg = cfgs[tname]
        if strategy == "H" and cfg["m"] == 0: continue
        if strategy == "D-pt" and cfg["m"] == 1: continue
        cnt_t = _resolve_count_for_target(count, tname)
        if cnt_t == 0: continue
        nc = cfg["nc_h"] if strategy == "H" else cfg["nc_d"]
        y_all = Y[:, t_idx]
        if strategy == "H":
            sel = h_top_M(X_v64, y_all, cnt_t)
        else:
            sel = dpt_top_N(X_v64, y_all, Vt_full, cnt_t, nc)
        for rank, col in enumerate(sel):
            rows.append(dict(held_out_idx=-1, target=tname, strategy=strategy,
                             nc=nc, rank=rank + 1, col_idx=int(col)))
    return rows


def collect_lomo_selections(X_v, Y, Vt_full, target_names, cfgs, strategy, count):
    """
    Return list of dicts: [{fold_model_idx, target, nc, rank, col_idx}, ...]

    cfgs : dict[target → {m, nc_h, nc_d, ...}]  (production per-target config)
    strategy : "H" or "D-pt"
    count : scalar (uniform) OR {target: count} dict (per-target budget)
            — pass count_L for H, count_G for D-pt to honor per-target q.
    """
    from pacebench.selection import h_top_M, dpt_top_N
    n = X_v.shape[0]
    X_v64 = X_v.astype(np.float64)
    rows = []
    for t_idx, tname in enumerate(target_names):
        if tname not in cfgs: continue
        cfg = cfgs[tname]
        if strategy == "H" and cfg["m"] == 0: continue
        if strategy == "D-pt" and cfg["m"] == 1: continue
        cnt_t = _resolve_count_for_target(count, tname)
        if cnt_t == 0: continue
        nc = cfg["nc_h"] if strategy == "H" else cfg["nc_d"]
        for i in range(n):
            tr = [k for k in range(n) if k != i]
            X_tr = X_v64[tr]
            y_tr = Y[tr, t_idx]
            if strategy == "H":
                sel = h_top_M(X_tr, y_tr, cnt_t)
            else:
                sel = dpt_top_N(X_tr, y_tr, Vt_full, cnt_t, nc)
            for rank, col in enumerate(sel):
                rows.append(dict(held_out_idx=i, target=tname, strategy=strategy,
                                 nc=nc, rank=rank + 1, col_idx=int(col)))
    return rows


def fit_all_Dpt_abs(X_v, Y, Vt_full, nc, N, P_target, B, seed):
    """Task A in-sample fit with D-pt embedding. Returns (n, T) predictions."""
    from pacebench.embedding import dpt_embed_all
    n, T = X_v.shape[0], Y.shape[1]
    X_all = X_v.astype(np.float64)
    preds = np.zeros((n, T))
    for t_idx in range(T):
        rng = np.random.RandomState(seed + t_idx * 31)
        y_all = Y[:, t_idx]
        P_t   = P_target[t_idx].astype(np.float64)
        U_all = dpt_embed_all(X_all, y_all, Vt_full, N, nc)
        U_pool, y_pool = pooled_absolute(U_all, P_t, B, rng)
        corrs = spearman_X_vs_Y(U_pool, y_pool.reshape(-1, 1))[:, 0]
        w = corrs / (np.abs(corrs).sum() + 1e-12)
        x_pool = U_pool @ w
        for i in range(n):
            preds[i, t_idx] = ols_1d(x_pool, y_pool, float(U_all[i] @ w))
    return preds
