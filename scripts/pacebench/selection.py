"""Per-target source-instance selection strategies (rigorous: training-fold only).

Two strategies share the same interface `(X_tr, y_tr, nc, count, ...) → indices`.

  H-strategy: top-C by |Spearman(X_tr[:, s], y_tr)|; target-relevance only.
  D-pt      : top-C by h_s × |Spearman(X_tr[:, s], y_tr)|, where
              h_s = Σ_c V_{c,s}^2 is the classical leverage score;
              combines SVD geometric importance (leverage) with target relevance.
"""

import os
import pickle
import numpy as np

from pacebench.stats import spearman_X_vs_Y


# Ablation hook: when PROXYBENCH_H_VARIANT is set, h_top_M dispatches to the
# named variant ("balanced" or "ability"), using side-channel metadata from
# PROXYBENCH_H_VARIANT_PICKLE (a pickle file containing a dict with
# {"bm_of_cols": [...], "bm_to_abilities": {...}, "abilities": [...]}).
# This design lets monkey-patch propagate through joblib/loky worker imports.
def _load_variant_metadata():
    path = os.environ.get("PROXYBENCH_H_VARIANT_PICKLE")
    if not path: return None
    with open(path, "rb") as f:
        return pickle.load(f)


def h_top_M(X_tr, y_tr, M):
    """Select top-M source instances by |Spearman| with training fold's target labels.

    If PROXYBENCH_H_VARIANT ∈ {"balanced", "ability"} is set, dispatch to the
    corresponding quota-based variant (used by scripts/ablation_quota.py).
    """
    variant = os.environ.get("PROXYBENCH_H_VARIANT", "")
    if variant == "balanced":
        meta = _load_variant_metadata()
        return h_top_C_balanced(X_tr, y_tr, meta["bm_of_cols"], C=M)
    if variant == "ability":
        meta = _load_variant_metadata()
        return h_top_C_ability(X_tr, y_tr, meta["bm_of_cols"],
                               meta["bm_to_abilities"], meta["abilities"], C=M)
    corrs = spearman_X_vs_Y(X_tr, y_tr.reshape(-1, 1))[:, 0]
    return np.argsort(np.abs(corrs))[::-1][:M]


def joint_top_C(X_tr, y_tr, Vt_full, C, nc_d, q):
    """
    Strict-budget joint H + D-pt selection: ``|h_idx ∪ d_idx| = C``.

    Initial allocation:
      h_idx = top ``round(q*C)``    by  |Spearman(X[:, s], y_tr)|
      d_idx = top ``C - round(q*C)`` by  leverage(s; nc_d) × |Spearman|

    If the two initial sets overlap, the union is short by ``X = C - |h ∪ d|``
    instances.  We then greedily extend each side's selection by next-best
    columns NOT yet in the union, with q-weighted rotation:

        at each step, pick whichever side is below its target ratio
        (target_h_adds = round(q*X), target_d_adds = X - target_h_adds);
        advance that side's pointer past columns already in union;
        add the first new column found.

    The growth phase preserves overlap = |h ∩ d| from the initial top-K
    (does NOT add overlapping columns to the OTHER side), so the final
    h_idx and d_idx satisfy:
        |h_idx| = round(q*C)    + h_adds       (h_adds = q-share of needed)
        |d_idx| = C - round(q*C) + d_adds      (d_adds = (1-q)-share)
        |h ∪ d| = C

    Returns (h_idx, d_idx) as ascending-sorted ndarrays. Falls back to
    available pool size (with warning suppressed) if C > |valid columns|.
    """
    n_S = X_tr.shape[1]
    if C > n_S:
        C = n_S   # cap at pool size

    corrs = spearman_X_vs_Y(X_tr, y_tr.reshape(-1, 1))[:, 0]
    abs_corrs = np.abs(corrs)
    h_order = np.argsort(abs_corrs)[::-1]

    V_nc = Vt_full[:nc_d, :].T
    lev = (V_nc ** 2).sum(axis=1)
    d_score = lev * abs_corrs
    d_order = np.argsort(d_score)[::-1]

    init_h = int(round(q * C))
    init_d = C - init_h
    h_set = {int(c) for c in h_order[:init_h]}
    d_set = {int(c) for c in d_order[:init_d]}
    union = h_set | d_set

    needed = C - len(union)
    if needed > 0:
        target_h_adds = int(round(q * needed))
        target_d_adds = needed - target_h_adds
        h_ptr, d_ptr = init_h, init_d
        h_adds, d_adds = 0, 0
        h_exhausted = d_exhausted = False
        while h_adds + d_adds < needed:
            if h_exhausted and d_exhausted:
                break
            if h_exhausted:
                advance = "D"
            elif d_exhausted:
                advance = "H"
            else:
                h_short = max(target_h_adds - h_adds, 0)
                d_short = max(target_d_adds - d_adds, 0)
                if h_short > d_short:
                    advance = "H"
                elif d_short > h_short:
                    advance = "D"
                else:
                    advance = "H" if h_adds <= d_adds else "D"
            if advance == "H":
                found = False
                while h_ptr < n_S:
                    col = int(h_order[h_ptr])
                    h_ptr += 1
                    if col not in union:
                        h_set.add(col); union.add(col); h_adds += 1
                        found = True; break
                if not found: h_exhausted = True
            else:
                found = False
                while d_ptr < n_S:
                    col = int(d_order[d_ptr])
                    d_ptr += 1
                    if col not in union:
                        d_set.add(col); union.add(col); d_adds += 1
                        found = True; break
                if not found: d_exhausted = True
    h_idx = np.array(sorted(h_set), dtype=np.int64)
    d_idx = np.array(sorted(d_set), dtype=np.int64)
    return h_idx, d_idx


def dpt_top_N(X_tr, y_tr, Vt_full, N, nc):
    """
    D-pt selection: top-N by h_s × |Spearman(X_tr[:, s], y_tr)|,
    where h_s = Σ_c V_{c,s}^2 is the classical leverage score.

    Vt_full is the source-matrix SVD right-singular-matrix (computed once outside fold,
    uses only X — non-leaky).
    """
    V_nc = Vt_full[:nc, :].T                      # (|S|, nc)
    lev = (V_nc ** 2).sum(axis=1)                 # (|S|,)  classical leverage score
    corrs = spearman_X_vs_Y(X_tr, y_tr.reshape(-1, 1))[:, 0]
    score = lev * np.abs(corrs)
    return np.array(sorted(np.argsort(score)[::-1][:N]))


def dpt_top_N_maxleverage(X_tr, y_tr, Vt_full, N, nc):
    """
    Ablation variant of dpt_top_N using max_c |V_{c,s}| (max-absolute loading)
    instead of the classical Σ_c V_{c,s}^2 leverage score.

    Produces identical top-N rankings to dpt_top_N on the current source pool
    (ablation confirms Δ = 0.00pp on LOMO metrics).
    """
    V_nc = Vt_full[:nc, :].T                      # (|S|, nc)
    abs_V_max = np.abs(V_nc).max(axis=1)          # (|S|,)
    corrs = spearman_X_vs_Y(X_tr, y_tr.reshape(-1, 1))[:, 0]
    score = abs_V_max * np.abs(corrs)
    return np.array(sorted(np.argsort(score)[::-1][:N]))


# ──────────────────────────────────────────────────────────────
# H-selection ablation variants for the instance-quota ablation
# (benchmark-balanced / per-ability quota). Both reuse |Spearman| as the
# within-group ranking score so they isolate ONLY the allocation strategy.
# ──────────────────────────────────────────────────────────────

def h_top_C_balanced(X_tr, y_tr, benchmarks_of_cols, C=100):
    """
    Benchmark-balanced quota: give each of the |B| source benchmarks an equal
    share of the C-instance budget.

    Let K = ceil(C / |B|).  Each benchmark contributes its top-K columns ranked
    by |Spearman(X_tr[:, col], y_tr)|; the union is then truncated to exactly C
    by the same global |Spearman| order.  A benchmark that cannot supply K
    columns contributes all of its columns.

    Parameters
    ----------
    X_tr : (n_tr, |S|) training source matrix.
    y_tr : (n_tr,) training-fold target labels.
    benchmarks_of_cols : sequence of length |S| with the benchmark name per column.
    C : int, total budget.
    """
    benchmarks_of_cols = np.asarray(benchmarks_of_cols)
    unique_bms = sorted(set(benchmarks_of_cols.tolist()))
    K = int(np.ceil(C / max(len(unique_bms), 1)))
    corrs = spearman_X_vs_Y(X_tr, y_tr.reshape(-1, 1))[:, 0]
    abs_corr = np.abs(corrs)
    chosen = set()
    for bm in unique_bms:
        bm_cols = np.where(benchmarks_of_cols == bm)[0]
        if bm_cols.size == 0:
            continue
        order = bm_cols[np.argsort(-abs_corr[bm_cols])]
        for c in order[:K]:
            chosen.add(int(c))
    chosen_arr = np.fromiter(chosen, dtype=np.int64)
    if chosen_arr.size <= C:
        return np.array(sorted(chosen_arr))
    # Truncate to exactly C by |Spearman|
    ord_by_corr = chosen_arr[np.argsort(-abs_corr[chosen_arr])]
    return np.array(sorted(ord_by_corr[:C].tolist()))


def h_top_C_ability(X_tr, y_tr, benchmarks_of_cols, bm_to_abilities, abilities, C=100):
    """
    Per-ability quota: give each of the |A| atomic abilities an equal share
    of the C-instance budget.

    Let K = ceil(C / |A|).  For each ability a ∈ abilities, take the top-K
    columns (by |Spearman|) among columns whose benchmark covers ability a
    (bm_to_abilities[bm] contains a).  An instance is assigned to every ability
    its benchmark covers (union, not exclusive).  Union of all ability top-K
    sets is truncated to exactly C by global |Spearman|.

    Parameters
    ----------
    X_tr : (n_tr, |S|) training source matrix.
    y_tr : (n_tr,) training-fold target labels.
    benchmarks_of_cols : sequence of length |S| with the benchmark name per column.
    bm_to_abilities : dict[benchmark → list of ability names].
    abilities : sequence of ability names (fixes the denominator |A|).
    C : int, total budget.
    """
    benchmarks_of_cols = np.asarray(benchmarks_of_cols)
    corrs = spearman_X_vs_Y(X_tr, y_tr.reshape(-1, 1))[:, 0]
    abs_corr = np.abs(corrs)
    K = int(np.ceil(C / max(len(abilities), 1)))
    # Precompute cols per ability
    chosen = set()
    for ab in abilities:
        # benchmarks covering this ability
        ab_bms = {bm for bm, abs_of_bm in bm_to_abilities.items() if ab in abs_of_bm}
        mask = np.array([bm in ab_bms for bm in benchmarks_of_cols])
        cols_a = np.where(mask)[0]
        if cols_a.size == 0:
            continue
        order = cols_a[np.argsort(-abs_corr[cols_a])]
        for c in order[:K]:
            chosen.add(int(c))
    chosen_arr = np.fromiter(chosen, dtype=np.int64)
    if chosen_arr.size <= C:
        return np.array(sorted(chosen_arr))
    ord_by_corr = chosen_arr[np.argsort(-abs_corr[chosen_arr])]
    return np.array(sorted(ord_by_corr[:C].tolist()))
