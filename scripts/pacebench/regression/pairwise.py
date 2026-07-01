"""Task B: pairwise preference prediction.

Step 4 of the pipeline (regression): on pooled pair-diff data (D_pool, dy_pool),
either:
  * Pairwise logistic regression (L-BFGS, BCE + ridge) → logit margin
  * Spearman-weighted 1-D OLS on pair-diffs (no intercept) → score-scale margin

Per-target choice is encoded via TASK_B_CONFIG[target]["reg"] ∈ {"ols", "logit"}.

Fold + LOMO wrappers exist for both (H/D-pt) × (logit/ols) combinations.
"""

import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed

from pacebench.config import N_JOBS, DEFAULT_ALPHA
from pacebench.stats import sigmoid, pair_indices, spearman_X_vs_Y
from pacebench.embedding import h_embed, dpt_embed, expand_to_all_models
from pacebench.bootstrap import pooled_pairwise


# ──────────────────────────────────────────────────────────────
# Pairwise logistic fit
# ──────────────────────────────────────────────────────────────

def fit_pairwise_logistic(X, y, alpha=DEFAULT_ALPHA, max_iter=200):
    """
    Antisymmetric pairwise logistic regression, no intercept:
        P(i beats j) = sigmoid((u_i - u_j)ᵀ w)
    with BCE + (alpha/2)‖w‖² loss. Solved via L-BFGS-B with analytical gradient.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    def loss_and_grad(w):
        z  = X @ w
        lp = np.where(z >= 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
        loss = float(np.sum(lp - y * z) / n + 0.5 * alpha * np.dot(w, w))
        p = sigmoid(z)
        g = X.T @ (p - y) / n + alpha * w
        return loss, g
    w0 = np.zeros(d, dtype=np.float64)
    res = minimize(loss_and_grad, w0, jac=True, method='L-BFGS-B',
                   options={"maxiter": max_iter})
    return res.x


# ──────────────────────────────────────────────────────────────
# Per-fold functions
# ──────────────────────────────────────────────────────────────

def _fold_H_logit(i, X_valid, Y, P_target, nc, M, B, seed, alpha,
                  pair_idx_a, pair_idx_b, pinned_sel_by_t=None):
    n, T = X_valid.shape[0], Y.shape[1]
    tr = [k for k in range(n) if k != i]
    X_tr = X_valid[tr].astype(np.float64)
    margins = np.zeros((n, T))

    for t_idx in range(T):
        rng = np.random.RandomState(seed + i * 997 + t_idx * 31)
        y_tr = Y[tr, t_idx]
        P_t  = P_target[t_idx][tr].astype(np.float64)
        psel = pinned_sel_by_t.get(t_idx) if pinned_sel_by_t else None
        U_tr, u_te = h_embed(X_valid, i, X_tr, y_tr, M, nc, pinned_sel=psel)
        D_pool, dy_pool = pooled_pairwise(U_tr, P_t, B, rng, pair_idx_a, pair_idx_b)
        labels = (dy_pool > 0).astype(np.float64)
        if labels.sum() in (0, len(labels)):
            continue                                 # degenerate bootstrap
        w = fit_pairwise_logistic(D_pool, labels, alpha=alpha)
        u_all = expand_to_all_models(U_tr, u_te, tr, i, n)
        margins[:, t_idx] = (u_all[i:i+1] - u_all) @ w
    return margins


def _fold_Dpt_logit(i, X_valid, Y, Vt_full, P_target, nc, N, B, seed, alpha,
                    pair_idx_a, pair_idx_b, pinned_sel_by_t=None):
    n, T = X_valid.shape[0], Y.shape[1]
    tr = [k for k in range(n) if k != i]
    X_tr = X_valid[tr].astype(np.float64)
    margins = np.zeros((n, T))

    for t_idx in range(T):
        rng = np.random.RandomState(seed + i * 997 + t_idx * 31)
        y_tr = Y[tr, t_idx]
        P_t  = P_target[t_idx][tr].astype(np.float64)
        psel = pinned_sel_by_t.get(t_idx) if pinned_sel_by_t else None
        U_tr, u_te = dpt_embed(X_valid, i, X_tr, y_tr, Vt_full, N, nc, pinned_sel=psel)
        D_pool, dy_pool = pooled_pairwise(U_tr, P_t, B, rng, pair_idx_a, pair_idx_b)
        labels = (dy_pool > 0).astype(np.float64)
        if labels.sum() in (0, len(labels)):
            continue
        w = fit_pairwise_logistic(D_pool, labels, alpha=alpha)
        u_all = expand_to_all_models(U_tr, u_te, tr, i, n)
        margins[:, t_idx] = (u_all[i:i+1] - u_all) @ w
    return margins


# ──────────────────────────────────────────────────────────────
# LOMO parallel wrappers — 14 folds, returns (n, n, T) margins
# ──────────────────────────────────────────────────────────────

def lomo_H_logit(X_v, Y, nc, M, P_target, B, seed, alpha=DEFAULT_ALPHA,
                 pinned_sel_by_fold_t=None):
    """Task B LOMO with H embedding + pairwise logistic. Returns (n, n, T) logits."""
    n = X_v.shape[0]
    pair_idx_a, pair_idx_b = pair_indices(n - 1)
    res = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(_fold_H_logit)(i, X_v, Y, P_target, nc, M, B, seed,
                               alpha, pair_idx_a, pair_idx_b,
                               pinned_sel_by_fold_t.get(i) if pinned_sel_by_fold_t else None)
        for i in range(n))
    return np.stack(res, axis=0)


def lomo_Dpt_logit(X_v, Y, Vt, nc, N, P_target, B, seed, alpha=DEFAULT_ALPHA,
                   pinned_sel_by_fold_t=None):
    """Task B LOMO with D-pt embedding + pairwise logistic. Returns (n, n, T) logits."""
    n = X_v.shape[0]
    pair_idx_a, pair_idx_b = pair_indices(n - 1)
    res = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(_fold_Dpt_logit)(i, X_v, Y, Vt, P_target, nc, N, B, seed,
                                 alpha, pair_idx_a, pair_idx_b,
                                 pinned_sel_by_fold_t.get(i) if pinned_sel_by_fold_t else None)
        for i in range(n))
    return np.stack(res, axis=0)


# ──────────────────────────────────────────────────────────────
# Pair OLS (Spearman-weighted 1-D OLS on pair-diffs)
# ──────────────────────────────────────────────────────────────

def _fit_predict_pair_ols(D_pool, dy_pool, test_diffs):
    """
    Spearman-weighted 1-D projection + OLS through origin (antisymmetric).
    Returns prediction vector, one entry per row of test_diffs.
    """
    corrs = spearman_X_vs_Y(D_pool, dy_pool.reshape(-1, 1))[:, 0]
    w = corrs / (np.abs(corrs).sum() + 1e-12)
    x_tr = D_pool @ w
    var  = float((x_tr ** 2).sum())
    if var < 1e-12:
        return np.zeros(test_diffs.shape[0])
    slope = float((x_tr * dy_pool).sum() / var)
    return slope * (test_diffs @ w)


def _fold_H_pair_ols(i, X_valid, Y, P_target, nc, M, B, seed,
                     pair_idx_a, pair_idx_b, pinned_sel_by_t=None):
    n, T = X_valid.shape[0], Y.shape[1]
    tr = [k for k in range(n) if k != i]
    X_tr = X_valid[tr].astype(np.float64)
    margins = np.zeros((n, T))
    for t_idx in range(T):
        rng = np.random.RandomState(seed + i * 997 + t_idx * 31)
        y_tr = Y[tr, t_idx]
        P_t  = P_target[t_idx][tr].astype(np.float64)
        psel = pinned_sel_by_t.get(t_idx) if pinned_sel_by_t else None
        U_tr, u_te = h_embed(X_valid, i, X_tr, y_tr, M, nc, pinned_sel=psel)
        D_pool, dy_pool = pooled_pairwise(U_tr, P_t, B, rng, pair_idx_a, pair_idx_b)
        u_all = expand_to_all_models(U_tr, u_te, tr, i, n)
        margins[:, t_idx] = _fit_predict_pair_ols(D_pool, dy_pool,
                                                  u_all[i:i+1] - u_all)
    return margins


def _fold_Dpt_pair_ols(i, X_valid, Y, Vt_full, P_target, nc, N, B, seed,
                       pair_idx_a, pair_idx_b, pinned_sel_by_t=None):
    n, T = X_valid.shape[0], Y.shape[1]
    tr = [k for k in range(n) if k != i]
    X_tr = X_valid[tr].astype(np.float64)
    margins = np.zeros((n, T))
    for t_idx in range(T):
        rng = np.random.RandomState(seed + i * 997 + t_idx * 31)
        y_tr = Y[tr, t_idx]
        P_t  = P_target[t_idx][tr].astype(np.float64)
        psel = pinned_sel_by_t.get(t_idx) if pinned_sel_by_t else None
        U_tr, u_te = dpt_embed(X_valid, i, X_tr, y_tr, Vt_full, N, nc, pinned_sel=psel)
        D_pool, dy_pool = pooled_pairwise(U_tr, P_t, B, rng, pair_idx_a, pair_idx_b)
        u_all = expand_to_all_models(U_tr, u_te, tr, i, n)
        margins[:, t_idx] = _fit_predict_pair_ols(D_pool, dy_pool,
                                                  u_all[i:i+1] - u_all)
    return margins


def lomo_H_pair_ols(X_v, Y, nc, M, P_target, B, seed, pinned_sel_by_fold_t=None):
    """Task B LOMO: H embedding + pair-diff OLS (Spearman-weighted 1-D)."""
    n = X_v.shape[0]
    pair_idx_a, pair_idx_b = pair_indices(n - 1)
    res = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(_fold_H_pair_ols)(i, X_v, Y, P_target, nc, M, B, seed,
                                  pair_idx_a, pair_idx_b,
                                  pinned_sel_by_fold_t.get(i) if pinned_sel_by_fold_t else None)
        for i in range(n))
    return np.stack(res, axis=0)


def lomo_Dpt_pair_ols(X_v, Y, Vt, nc, N, P_target, B, seed, pinned_sel_by_fold_t=None):
    """Task B LOMO: D-pt embedding + pair-diff OLS (Spearman-weighted 1-D)."""
    n = X_v.shape[0]
    pair_idx_a, pair_idx_b = pair_indices(n - 1)
    res = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(_fold_Dpt_pair_ols)(i, X_v, Y, Vt, P_target, nc, N, B, seed,
                                    pair_idx_a, pair_idx_b,
                                    pinned_sel_by_fold_t.get(i) if pinned_sel_by_fold_t else None)
        for i in range(n))
    return np.stack(res, axis=0)


# ──────────────────────────────────────────────────────────────
# In-sample fit (all n models — no held-out). For analysis purposes.
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# Custom split: arbitrary train / eval model sets
# Pair (k, j) is populated iff (k ∈ eval_idx) OR (j ∈ eval_idx).
# All other cells are NaN (never used in metrics / dumps).
# ──────────────────────────────────────────────────────────────

def _populate_split_pair_matrix(U_all, w, eval_idx_set, n):
    """Compute margin matrix (n, n) with NaN where neither endpoint is in eval."""
    diff = U_all[:, None, :] - U_all[None, :, :]          # (n, n, nc)
    full = diff @ w                                         # (n, n)
    mask = np.zeros((n, n), dtype=bool)
    for k in range(n):
        for j in range(n):
            if k == j: continue
            if (k in eval_idx_set) or (j in eval_idx_set):
                mask[k, j] = True
    out = np.full((n, n), np.nan)
    out[mask] = full[mask]
    return out


def fit_split_H_pair(X_v, Y, P_target, train_idx, eval_idx, nc, M, B, seed,
                    reg="logit", alpha=DEFAULT_ALPHA, pinned_sel_by_t=None):
    """Task B custom-split with H embedding.

    Returns:
      margins : (n, n, T) with NaN outside (eval_idx × ∪ × eval_idx) pair filter
      sel     : dict[t_idx] → selected source column indices
    """
    from pacebench.embedding import h_embed_split
    n, T = X_v.shape[0], Y.shape[1]
    X_v64 = X_v.astype(np.float64)
    n_tr = len(train_idx)
    pair_idx_a, pair_idx_b = pair_indices(n_tr)
    eval_set = set(int(e) for e in eval_idx)
    margins = np.full((n, n, T), np.nan)
    sel_out = {}
    for t_idx in range(T):
        rng = np.random.RandomState(seed + t_idx * 31)
        y_tr = Y[train_idx, t_idx]
        P_t  = P_target[t_idx][train_idx].astype(np.float64)
        psel = pinned_sel_by_t.get(t_idx) if pinned_sel_by_t else None
        U_all, top_M_idx = h_embed_split(X_v64, train_idx, y_tr, M, nc, pinned_sel=psel)
        sel_out[t_idx] = top_M_idx
        U_tr = U_all[train_idx]
        D_pool, dy_pool = pooled_pairwise(U_tr, P_t, B, rng, pair_idx_a, pair_idx_b)
        if reg == "logit":
            labels = (dy_pool > 0).astype(np.float64)
            if labels.sum() in (0, len(labels)):
                continue
            w = fit_pairwise_logistic(D_pool, labels, alpha=alpha)
            margins[:, :, t_idx] = _populate_split_pair_matrix(U_all, w, eval_set, n)
        else:  # ols
            corrs = spearman_X_vs_Y(D_pool, dy_pool.reshape(-1, 1))[:, 0]
            w = corrs / (np.abs(corrs).sum() + 1e-12)
            x_tr = D_pool @ w
            var = float((x_tr ** 2).sum())
            if var < 1e-12: continue
            slope = float((x_tr * dy_pool).sum() / var)
            margins[:, :, t_idx] = slope * _populate_split_pair_matrix(U_all, w, eval_set, n)
    return margins, sel_out


def fit_split_Dpt_pair(X_v, Y, Vt_full, P_target, train_idx, eval_idx, nc, N, B, seed,
                       reg="logit", alpha=DEFAULT_ALPHA, pinned_sel_by_t=None):
    """Task B custom-split with D-pt embedding. Same return shape as fit_split_H_pair."""
    from pacebench.embedding import dpt_embed_split
    n, T = X_v.shape[0], Y.shape[1]
    X_v64 = X_v.astype(np.float64)
    n_tr = len(train_idx)
    pair_idx_a, pair_idx_b = pair_indices(n_tr)
    eval_set = set(int(e) for e in eval_idx)
    margins = np.full((n, n, T), np.nan)
    sel_out = {}
    for t_idx in range(T):
        rng = np.random.RandomState(seed + t_idx * 31)
        y_tr = Y[train_idx, t_idx]
        P_t  = P_target[t_idx][train_idx].astype(np.float64)
        psel = pinned_sel_by_t.get(t_idx) if pinned_sel_by_t else None
        U_all, sel = dpt_embed_split(X_v64, train_idx, y_tr, Vt_full, N, nc, pinned_sel=psel)
        sel_out[t_idx] = sel
        U_tr = U_all[train_idx]
        D_pool, dy_pool = pooled_pairwise(U_tr, P_t, B, rng, pair_idx_a, pair_idx_b)
        if reg == "logit":
            labels = (dy_pool > 0).astype(np.float64)
            if labels.sum() in (0, len(labels)):
                continue
            w = fit_pairwise_logistic(D_pool, labels, alpha=alpha)
            margins[:, :, t_idx] = _populate_split_pair_matrix(U_all, w, eval_set, n)
        else:  # ols
            corrs = spearman_X_vs_Y(D_pool, dy_pool.reshape(-1, 1))[:, 0]
            w = corrs / (np.abs(corrs).sum() + 1e-12)
            x_tr = D_pool @ w
            var = float((x_tr ** 2).sum())
            if var < 1e-12: continue
            slope = float((x_tr * dy_pool).sum() / var)
            margins[:, :, t_idx] = slope * _populate_split_pair_matrix(U_all, w, eval_set, n)
    return margins, sel_out


def collect_lomo_weights_pair(X_v, Y, Vt_full, target_names, cfgs,
                               P_target, B, seed, count, strategy,
                               pinned_sel_by_fold_t=None,
                               alpha=DEFAULT_ALPHA):
    """
    Per-fold per-target per-instance weights for Task B's pairwise-logit margin
    predictor.  For pair (k, j) at target t:
        ŝ(k, j, t) = (u_k - u_j) @ w_logit
                   = Σ_s (X[k,s] - X[j,s]) · a[s]
    where a[s] = (PROJ @ w_logit)[s].  The `weight` is m_factor · a[s].

    Only logit reg is supported here (matches paper's canonical setting).
    Respects `pinned_sel_by_fold_t` so weights are computed on the same
    instances as the actual regression.

    `count` may be a scalar or a {target: count} dict (per-target budget —
    pass count_L for H, count_G for D-pt to honor per-target q).

    Returns rows: held_out_idx, target, strategy, nc, rank, col_idx, weight.
    """
    from pacebench.selection import h_top_M, dpt_top_N
    from pacebench.regression.absolute import _resolve_count_for_target
    from numpy.linalg import svd as np_svd, pinv

    n = X_v.shape[0]
    X_v64 = X_v.astype(np.float64)
    pair_idx_a, pair_idx_b = pair_indices(n - 1)
    rows = []
    for t_idx, tname in enumerate(target_names):
        if tname not in cfgs: continue
        cfg = cfgs[tname]
        m = cfg["m"]
        m_factor = m if strategy == "H" else (1.0 - m)
        # When pinned is given for any fold of this target, dump even if m=0
        # (weight = m_factor * a_vec → 0; selection is still recorded for cost).
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
            pinned = (pinned_sel_by_fold_t.get(i, {}).get(t_idx)
                      if pinned_sel_by_fold_t else None)
            if pinned is not None and len(pinned) == 0:
                continue
            if strategy == "H":
                sel = pinned if pinned is not None else h_top_M(X_tr, y_tr, cnt_t)
                _, S_w, Vt_w = np_svd(X_tr[:, sel], full_matrices=False)
                nc_ = min(nc, Vt_w.shape[0])
                S_k = S_w[:nc_] + 1e-12
                Vt_k = Vt_w[:nc_, :]
                PROJ = Vt_k.T * (1.0 / S_k)
            else:
                sel = (pinned if pinned is not None
                       else dpt_top_N(X_tr, y_tr, Vt_full, cnt_t, nc))
                V_sel = Vt_full[:nc, :][:, sel].T
                PROJ = pinv(V_sel).T
            U_tr = X_tr[:, sel] @ PROJ
            D_pool, dy_pool = pooled_pairwise(U_tr, P_t, B, rng,
                                              pair_idx_a, pair_idx_b)
            labels = (dy_pool > 0).astype(np.float64)
            if labels.sum() in (0, len(labels)):
                a_vec = np.zeros(len(sel))
            else:
                w_logit = fit_pairwise_logistic(D_pool, labels, alpha=alpha)
                a_vec = PROJ @ w_logit
            weights = m_factor * a_vec
            for rank, (col, w) in enumerate(zip(sel, weights)):
                rows.append(dict(held_out_idx=i, target=tname, strategy=strategy,
                                 nc=nc, rank=rank + 1, col_idx=int(col),
                                 weight=float(w)))
    return rows


def collect_fit_all_weights_pair(X_v, Y, Vt_full, target_names, cfgs,
                                  P_target, B, seed, count, strategy,
                                  pinned_sel_by_t=None, alpha=DEFAULT_ALPHA):
    """Fit-all variant of collect_lomo_weights_pair. fold = 'ALL'.

    `count` may be a scalar or a {target: count} dict (per-target budget).
    """
    from pacebench.selection import h_top_M, dpt_top_N
    from pacebench.regression.absolute import _resolve_count_for_target
    from numpy.linalg import svd as np_svd, pinv

    n = X_v.shape[0]
    X_v64 = X_v.astype(np.float64)
    pair_idx_a, pair_idx_b = pair_indices(n)
    rows = []
    for t_idx, tname in enumerate(target_names):
        if tname not in cfgs: continue
        cfg = cfgs[tname]
        m = cfg["m"]
        m_factor = m if strategy == "H" else (1.0 - m)
        pinned = pinned_sel_by_t.get(t_idx) if pinned_sel_by_t else None
        # When pinned, dump the selection even if m=0 (weight=0 then) — keeps
        # full union visible for cost reporting under strict-budget.
        if pinned is None and m_factor == 0: continue
        cnt_t = _resolve_count_for_target(count, tname)
        if pinned is None and cnt_t == 0: continue
        nc = cfg["nc_h"] if strategy == "H" else cfg["nc_d"]
        y_all = Y[:, t_idx]
        P_t = P_target[t_idx].astype(np.float64)
        rng = np.random.RandomState(seed + t_idx * 31)
        if strategy == "H":
            sel = pinned if pinned is not None else h_top_M(X_v64, y_all, cnt_t)
            _, S_w, Vt_w = np_svd(X_v64[:, sel], full_matrices=False)
            nc_ = min(nc, Vt_w.shape[0])
            S_k = S_w[:nc_] + 1e-12
            Vt_k = Vt_w[:nc_, :]
            PROJ = Vt_k.T * (1.0 / S_k)
        else:
            sel = (pinned if pinned is not None
                   else dpt_top_N(X_v64, y_all, Vt_full, cnt_t, nc))
            V_sel = Vt_full[:nc, :][:, sel].T
            PROJ = pinv(V_sel).T
        U_all = X_v64[:, sel] @ PROJ
        D_pool, dy_pool = pooled_pairwise(U_all, P_t, B, rng,
                                          pair_idx_a, pair_idx_b)
        labels = (dy_pool > 0).astype(np.float64)
        if labels.sum() in (0, len(labels)):
            a_vec = np.zeros(len(sel))
        else:
            w_logit = fit_pairwise_logistic(D_pool, labels, alpha=alpha)
            a_vec = PROJ @ w_logit
        weights = m_factor * a_vec
        for rank, (col, w) in enumerate(zip(sel, weights)):
            rows.append(dict(held_out_idx=-1, target=tname, strategy=strategy,
                             nc=nc, rank=rank + 1, col_idx=int(col),
                             weight=float(w)))
    return rows


def fit_all_H_logit(X_v, Y, P_target, nc, M, B, seed, alpha=DEFAULT_ALPHA,
                    pinned_sel_by_t=None):
    """Task B in-sample fit with H embedding + pairwise logistic. Returns (n, n, T)."""
    from pacebench.embedding import h_embed_all
    from pacebench.bootstrap import pooled_pairwise
    n, T = X_v.shape[0], Y.shape[1]
    X_all = X_v.astype(np.float64)
    pair_idx_a, pair_idx_b = pair_indices(n)  # use all n models now
    margins = np.zeros((n, n, T))
    for t_idx in range(T):
        rng = np.random.RandomState(seed + t_idx * 31)
        y_all = Y[:, t_idx]
        P_t   = P_target[t_idx].astype(np.float64)
        psel = pinned_sel_by_t.get(t_idx) if pinned_sel_by_t else None
        U_all = h_embed_all(X_all, y_all, M, nc, pinned_sel=psel)
        D_pool, dy_pool = pooled_pairwise(U_all, P_t, B, rng, pair_idx_a, pair_idx_b)
        labels = (dy_pool > 0).astype(np.float64)
        if labels.sum() in (0, len(labels)):
            continue
        w = fit_pairwise_logistic(D_pool, labels, alpha=alpha)
        # In-sample: predict margin for every (k, j) pair
        for k in range(n):
            margins[k, :, t_idx] = (U_all[k:k+1] - U_all) @ w
    return margins


def fit_all_Dpt_logit(X_v, Y, Vt_full, nc, N, P_target, B, seed, alpha=DEFAULT_ALPHA,
                      pinned_sel_by_t=None):
    """Task B in-sample fit with D-pt embedding + pairwise logistic. Returns (n, n, T)."""
    from pacebench.embedding import dpt_embed_all
    from pacebench.bootstrap import pooled_pairwise
    n, T = X_v.shape[0], Y.shape[1]
    X_all = X_v.astype(np.float64)
    pair_idx_a, pair_idx_b = pair_indices(n)
    margins = np.zeros((n, n, T))
    for t_idx in range(T):
        rng = np.random.RandomState(seed + t_idx * 31)
        y_all = Y[:, t_idx]
        P_t   = P_target[t_idx].astype(np.float64)
        psel = pinned_sel_by_t.get(t_idx) if pinned_sel_by_t else None
        U_all = dpt_embed_all(X_all, y_all, Vt_full, N, nc, pinned_sel=psel)
        D_pool, dy_pool = pooled_pairwise(U_all, P_t, B, rng, pair_idx_a, pair_idx_b)
        labels = (dy_pool > 0).astype(np.float64)
        if labels.sum() in (0, len(labels)):
            continue
        w = fit_pairwise_logistic(D_pool, labels, alpha=alpha)
        for k in range(n):
            margins[k, :, t_idx] = (U_all[k:k+1] - U_all) @ w
    return margins


def fit_all_H_pair_ols(X_v, Y, P_target, nc, M, B, seed, pinned_sel_by_t=None):
    """Task B in-sample fit with H embedding + pair-diff OLS. Returns (n, n, T)."""
    from pacebench.embedding import h_embed_all
    from pacebench.bootstrap import pooled_pairwise
    n, T = X_v.shape[0], Y.shape[1]
    X_all = X_v.astype(np.float64)
    pair_idx_a, pair_idx_b = pair_indices(n)
    margins = np.zeros((n, n, T))
    for t_idx in range(T):
        rng = np.random.RandomState(seed + t_idx * 31)
        y_all = Y[:, t_idx]
        P_t   = P_target[t_idx].astype(np.float64)
        psel = pinned_sel_by_t.get(t_idx) if pinned_sel_by_t else None
        U_all = h_embed_all(X_all, y_all, M, nc, pinned_sel=psel)
        D_pool, dy_pool = pooled_pairwise(U_all, P_t, B, rng, pair_idx_a, pair_idx_b)
        for k in range(n):
            margins[k, :, t_idx] = _fit_predict_pair_ols(D_pool, dy_pool,
                                                         U_all[k:k+1] - U_all)
    return margins


def fit_all_Dpt_pair_ols(X_v, Y, Vt_full, nc, N, P_target, B, seed, pinned_sel_by_t=None):
    """Task B in-sample fit with D-pt embedding + pair-diff OLS. Returns (n, n, T)."""
    from pacebench.embedding import dpt_embed_all
    from pacebench.bootstrap import pooled_pairwise
    n, T = X_v.shape[0], Y.shape[1]
    X_all = X_v.astype(np.float64)
    pair_idx_a, pair_idx_b = pair_indices(n)
    margins = np.zeros((n, n, T))
    for t_idx in range(T):
        rng = np.random.RandomState(seed + t_idx * 31)
        y_all = Y[:, t_idx]
        P_t   = P_target[t_idx].astype(np.float64)
        psel = pinned_sel_by_t.get(t_idx) if pinned_sel_by_t else None
        U_all = dpt_embed_all(X_all, y_all, Vt_full, N, nc, pinned_sel=psel)
        D_pool, dy_pool = pooled_pairwise(U_all, P_t, B, rng, pair_idx_a, pair_idx_b)
        for k in range(n):
            margins[k, :, t_idx] = _fit_predict_pair_ols(D_pool, dy_pool,
                                                         U_all[k:k+1] - U_all)
    return margins
