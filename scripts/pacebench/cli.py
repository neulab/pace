#!/usr/bin/env python3
"""Pacebench CLI — script-style entry point.

Usage:
  python scripts/pacebench/cli.py abs  [--B 300] [--seed 42]
  python scripts/pacebench/cli.py pair [--B 300] [--seed 42]

Produces per-target LOMO metrics + CSV + scatter plot for the selected task.
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import sys
from pathlib import Path

# Make `scripts/` importable so `from pacebench.xxx import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd
from numpy.linalg import svd as np_svd
import warnings
warnings.filterwarnings("ignore")

from pacebench.config import (
    MODELS, COUNT, DEFAULT_B, DEFAULT_SEED, DEFAULT_ALPHA, BASE_DIR,
    TARGET_DATASETS, EXCLUDE_DATASETS,
    get_task_a_config, get_task_b_config, get_task_b_config_pinned_a,
)
from pacebench.data import build_data, filter_valid_columns


# ──────────────────────────────────────────────────────────────
# Per-target config resolution — respects CLI overrides
# ──────────────────────────────────────────────────────────────

def _resolve_configs(task_config, target_names, override_m, override_nc_h,
                     override_nc_d, fallback_m, fallback_nc_h, fallback_nc_d,
                     fallback_reg=None):
    """
    Build a dict target → {m, nc_h, nc_d, reg, q} by:
      1. Starting from task_config entries for known targets.
      2. Falling back to defaults for unknown targets.
      3. Applying any CLI overrides (same value across all targets if given).

    `reg` is preserved from task_config (Task B) or set to fallback_reg.
    `q`   is preserved from task_config when present (per-target default split);
          absent → None (each strategy independently uses full `count`).
    """
    out = {}
    for t in target_names:
        base = task_config.get(t, dict(m=fallback_m, nc_h=fallback_nc_h,
                                       nc_d=fallback_nc_d, reg=fallback_reg))
        out[t] = dict(
            m    = base["m"]    if override_m    is None else override_m,
            nc_h = base["nc_h"] if override_nc_h is None else override_nc_h,
            nc_d = base["nc_d"] if override_nc_d is None else override_nc_d,
            reg  = base.get("reg", fallback_reg),
            q    = base.get("q"),     # None if absent
        )
    return out
from pacebench.metrics import (
    build_abs_metrics_df, build_abs_metrics_df_eval, compute_pair_accuracy_df,
    collect_margins_labels, print_abs_results, print_pair_results,
)
from pacebench.calibration import fit_platt, apply_platt
from pacebench.regression.absolute import (
    loocv_H_abs, loocv_Dpt_abs, loocv_joint_abs, loocv_joint_abs_autotune,
    fit_all_H_abs, fit_all_Dpt_abs, fit_all_joint_abs,
    fit_split_H_abs, fit_split_Dpt_abs,
    collect_lomo_selections,
    collect_lomo_weights_abs, collect_fit_all_weights_abs,
)
from pacebench.regression.pairwise import (
    lomo_H_logit, lomo_Dpt_logit,
    lomo_H_pair_ols, lomo_Dpt_pair_ols,
    fit_all_H_logit, fit_all_Dpt_logit,
    fit_all_H_pair_ols, fit_all_Dpt_pair_ols,
    fit_split_H_pair, fit_split_Dpt_pair,
    collect_lomo_weights_pair, collect_fit_all_weights_pair,
)
from pacebench.visualize import plot_abs_loocv_fit, plot_pair_scatter


# ──────────────────────────────────────────────────────────────
# Shared setup
# ──────────────────────────────────────────────────────────────

def _load_and_svd(sources=None, targets=None):
    target_set = set(targets) if targets else TARGET_DATASETS
    # If user specifies targets, any of those targets that were in EXCLUDE is
    # moved out of the exclude set; reverse: if user-specified target is also
    # in the user's source list, that's an error (can't be both).
    exclude = set(EXCLUDE_DATASETS) - target_set
    if sources and (set(sources) & target_set):
        overlap = sorted(set(sources) & target_set)
        raise ValueError(f"Dataset(s) {overlap} cannot be both source and target.")
    X, Y, target_names, P_target, source_col_info = build_data(
        target_datasets=target_set,
        exclude_datasets=exclude,
        source_datasets=sources,
    )
    if X.shape[1] == 0:
        raise ValueError("No source instances loaded — check --sources or data directory.")
    X_v, valid_mask = filter_valid_columns(X, return_mask=True)
    source_col_info_v = [info for info, keep in zip(source_col_info, valid_mask) if keep]
    print(f"\nX: {X.shape}  Valid: {X_v.shape[1]:,}  Y: {Y.shape}", flush=True)
    print("\nComputing SVD …", flush=True)
    _, _, Vt_full = np_svd(X_v, full_matrices=False)
    return X_v, Y, target_names, P_target, Vt_full, source_col_info_v


def _resolve_model_split(train_names, eval_names):
    """
    Resolve --train-models / --eval-models into (train_idx, eval_idx) index arrays.

    Rules:
      * Neither flag set → returns (None, None) → caller uses LOMO.
      * Only train set → eval = complement (must be non-empty).
      * Only eval  set → train = complement (must be non-empty).
      * Both set     → use as-is; overlap allowed (becomes partially in-sample).
    """
    if not train_names and not eval_names:
        return None, None
    model_to_idx = {m: i for i, m in enumerate(MODELS)}
    def resolve(names, label):
        idx = []
        for m in names:
            if m not in model_to_idx:
                raise ValueError(f"--{label}-models: unknown model '{m}'. "
                                 f"Known: {', '.join(MODELS)}")
            idx.append(model_to_idx[m])
        return sorted(set(idx))
    tr = resolve(train_names, "train") if train_names else None
    ev = resolve(eval_names,  "eval")  if eval_names  else None
    if tr is None:
        tr = sorted(set(range(len(MODELS))) - set(ev))
    if ev is None:
        ev = sorted(set(range(len(MODELS))) - set(tr))
    if not tr:
        raise ValueError("Custom split: training set is empty.")
    if not ev:
        raise ValueError("Custom split: eval set is empty.")
    return tr, ev


def _compute_task_a_selections_lomo(X_v, Y, Vt_full, target_names, count,
                                     P_target=None, B=None, seed=None,
                                     auto_tune=False, nc_max=10, m_grid=None,
                                     q=None):
    """Return (H_by_fold_t, Dpt_by_fold_t, a_cfgs) for pinning Task B to A.

    When `auto_tune=False`: use the frozen TASK_A_CONFIGS[count] per-target
    config (q honored per target if present in config).
    When `auto_tune=True`: sweep (q, m, nc_h, nc_d) per target — same objective
    as `cli.py abs --auto-tune` (minimize MAE_pp − Spearman) — so pinned A
    selections match what `abs --auto-tune` would produce. If user passed --q,
    that single q is used (no sweep). Requires P_target/B/seed in auto-tune mode.

    Each returned dict: {fold_i: {t_idx: np.array of column indices}}.
    """
    if auto_tune:
        if P_target is None or B is None or seed is None:
            raise ValueError("auto_tune requires P_target, B, and seed")
        from pacebench.regression.absolute import loocv_H_abs, loocv_Dpt_abs
        nc_range = list(range(1, nc_max + 1))
        if m_grid is None:
            m_grid = [round(i * 0.1, 1) for i in range(11)]
        # q sweep: explicit --q fixes it; otherwise sweep 0.0..1.0 step 0.1.
        if q is not None:
            q_list = [q]
            print(f"\n[pin-A auto-tune] sweeping nc × m at q={q} (fixed) …",
                  flush=True)
        else:
            q_list = [round(i * 0.1, 1) for i in range(11)]
            print(f"\n[pin-A auto-tune] sweeping q × nc × m to pick A's optimum …",
                  flush=True)
        # Build cache by (nc, M) for unique M values implied by q_list
        unique_M_L, unique_M_G = set(), set()
        for q_val in q_list:
            cL, cG = _split_count(count, q_val)
            if cL > 0: unique_M_L.add(cL)
            if cG > 0: unique_M_G.add(cG)
        unique_M_L, unique_M_G = sorted(unique_M_L), sorted(unique_M_G)
        H_cache_qm, D_cache_qm = {}, {}
        for nc_ in nc_range:
            for M in unique_M_L:
                print(f"    • H(nc={nc_}, M={M}) …", flush=True)
                H_cache_qm[(nc_, M)] = loocv_H_abs(X_v, Y, nc_, M, P_target, B, seed)
            for M in unique_M_G:
                print(f"    • D-pt(nc={nc_}, N={M}) …", flush=True)
                D_cache_qm[(nc_, M)] = loocv_Dpt_abs(X_v, Y, Vt_full, nc_, M,
                                                     P_target, B, seed)
        a_cfgs, _, _ = _autotune_task_a_q(H_cache_qm, D_cache_qm, Y, target_names,
                                          nc_range, m_grid, count, q_list)
        note = "(auto-tuned" + (f", q={q}" if q is not None else " w/ q-sweep") + " → used for pin-A)"
        _print_autotune_config("TASK_A", count, a_cfgs, extra_note=note)
    else:
        _a_tbl = get_task_a_config(count)
        a_cfgs = {t: _a_tbl[t] for t in target_names if t in _a_tbl}

    if not a_cfgs:
        return {}, {}, a_cfgs

    # Build pinned selections per target — each target uses its own q if
    # present in cfg, else falls back to the global q (or 2C if also None).
    from pacebench.selection import h_top_M, dpt_top_N
    n = X_v.shape[0]
    X_v64 = X_v.astype(np.float64)
    t_name_to_idx = {t: i for i, t in enumerate(target_names)}
    H_dict, D_dict = {}, {}
    for tname, cfg in a_cfgs.items():
        t_idx = t_name_to_idx[tname]
        q_t = cfg.get("q") if cfg.get("q") is not None else q
        cL, cG = _split_count(count, q_t)
        m_t = cfg["m"]
        for i in range(n):
            tr = [k for k in range(n) if k != i]
            X_tr = X_v64[tr]
            y_tr = Y[tr, t_idx]
            if cL > 0 and m_t > 0:
                H_dict.setdefault(i, {})[t_idx] = h_top_M(X_tr, y_tr, cL)
            if cG > 0 and m_t < 1:
                D_dict.setdefault(i, {})[t_idx] = dpt_top_N(
                    X_tr, y_tr, Vt_full, cG, cfg["nc_d"])
    return H_dict, D_dict, a_cfgs


def _compute_task_a_selections_lomo_strict(X_v, Y, Vt_full, target_names, count,
                                             P_target=None, B=None, seed=None,
                                             auto_tune=False, nc_max=10, m_grid=None,
                                             q=None):
    """Strict-budget version of _compute_task_a_selections_lomo.

    Selections satisfy |H ∪ D| = C per (fold, target) via joint_top_C.

    auto_tune=True: run the strict-budget Task A auto-tune (sweep q × nc_d ×
    nc_h × m, minimize MAE_pp − Spearman) to pick (q, nc_d) per target —
    matches what ``abs --strict-budget --auto-tune`` would produce. Pinned
    selections are extracted from the resulting sel_cache.
    """
    from pacebench.selection import joint_top_C
    n = X_v.shape[0]
    X_v64 = X_v.astype(np.float64)
    t_name_to_idx = {t: i for i, t in enumerate(target_names)}

    if auto_tune:
        if P_target is None or B is None or seed is None:
            raise ValueError("auto_tune requires P_target, B, and seed")
        nc_h_grid = list(range(1, nc_max + 1))
        nc_d_grid = list(range(1, nc_max + 1))
        if m_grid is None:
            m_grid = [round(i * 0.1, 1) for i in range(11)]
        q_list = [float(q)] if q is not None else [round(i * 0.1, 1) for i in range(11)]
        print(f"\n[pin-A strict-budget auto-tune] sweeping q ∈ {q_list}, "
              f"nc_h ∈ {nc_h_grid}, nc_d ∈ {nc_d_grid}, m ∈ {m_grid} …",
              flush=True)
        H_cache, D_cache, sel_cache = loocv_joint_abs_autotune(
            X_v, Y, Vt_full, q_list, nc_h_grid, nc_d_grid, count,
            P_target, B, seed)
        a_cfgs, _, _ = _autotune_task_a_strict_q(
            H_cache, D_cache, Y, target_names,
            q_list, nc_h_grid, nc_d_grid, m_grid, count)
        note = ("(strict-budget auto-tuned"
                + (f", q={q}" if q is not None else " w/ q-sweep")
                + " → pin-A)")
        _print_autotune_config("TASK_A", count, a_cfgs, extra_note=note)
        # Extract per-fold per-target pinned (h_idx, d_idx) from sel_cache
        H_dict, D_dict = {}, {}
        for tname, cfg in a_cfgs.items():
            t_idx = t_name_to_idx[tname]
            sel_qd = sel_cache[(cfg["q"], cfg["nc_d"])]
            for i in range(n):
                h_idx, d_idx = sel_qd[i][t_idx]
                if cfg["m"] > 0 and len(h_idx) > 0:
                    H_dict.setdefault(i, {})[t_idx] = h_idx
                if cfg["m"] < 1 and len(d_idx) > 0:
                    D_dict.setdefault(i, {})[t_idx] = d_idx
        return H_dict, D_dict, a_cfgs

    # Non-autotune: use frozen TASK_A_CONFIGS, recompute joint selection per fold
    _a_tbl = get_task_a_config(count)
    a_cfgs = {t: _a_tbl[t] for t in target_names if t in _a_tbl}
    if not a_cfgs:
        return {}, {}, a_cfgs
    H_dict, D_dict = {}, {}
    for tname, cfg in a_cfgs.items():
        t_idx = t_name_to_idx[tname]
        q_t = cfg.get("q") if cfg.get("q") is not None else q
        if q_t is None: q_t = 0.5
        for i in range(n):
            tr = [k for k in range(n) if k != i]
            X_tr = X_v64[tr]
            y_tr = Y[tr, t_idx]
            h_idx, d_idx = joint_top_C(X_tr, y_tr, Vt_full, count,
                                        cfg["nc_d"], q_t)
            # Always pin both halves regardless of A's m: the |union|=C cost
            # is paid (both h_idx and d_idx evaluated by joint_top_C); pair
            # regression skips a side internally if its own m makes it irrelevant.
            if len(h_idx) > 0:
                H_dict.setdefault(i, {})[t_idx] = h_idx
            if len(d_idx) > 0:
                D_dict.setdefault(i, {})[t_idx] = d_idx
    return H_dict, D_dict, a_cfgs


def _compute_task_a_selections_fit_all(X_v, Y, Vt_full, target_names, count):
    """Return (H_by_t, Dpt_by_t) using TASK_A_CONFIG's per-target nc on all n models."""
    from pacebench.selection import h_top_M, dpt_top_N
    _a_tbl = get_task_a_config(count)
    X_v64 = X_v.astype(np.float64)
    H_dict, D_dict = {}, {}
    for t_idx, tname in enumerate(target_names):
        if tname not in _a_tbl: continue
        a_cfg = _a_tbl[tname]
        y_all = Y[:, t_idx]
        H_dict[t_idx] = h_top_M(X_v64, y_all, count)
        D_dict[t_idx] = dpt_top_N(X_v64, y_all, Vt_full, count, a_cfg["nc_d"])
    return H_dict, D_dict


def _compute_task_a_selections_fit_all_strict(X_v, Y, Vt_full, target_names, count,
                                                 q=None):
    """Strict-budget fit-all variant of _compute_task_a_selections_fit_all.

    Uses joint_top_C(q, nc_d, count) on all n models per target, so |H ∪ D| = C.
    Per-target q taken from cfg["q"] unless `q` overrides.
    """
    from pacebench.selection import joint_top_C
    a_cfgs = {t: c for t, c in get_task_a_config(count).items() if t in target_names}
    X_v64 = X_v.astype(np.float64)
    H_dict, D_dict = {}, {}
    for t_idx, tname in enumerate(target_names):
        if tname not in a_cfgs: continue
        cfg = a_cfgs[tname]
        q_t = q if q is not None else cfg.get("q")
        if q_t is None: q_t = 0.5
        y_all = Y[:, t_idx]
        h_idx, d_idx = joint_top_C(X_v64, y_all, Vt_full, count, cfg["nc_d"], q_t)
        if len(h_idx) > 0: H_dict[t_idx] = h_idx
        if len(d_idx) > 0: D_dict[t_idx] = d_idx
    return H_dict, D_dict, a_cfgs


def _compute_task_a_selections_split(X_v, Y, Vt_full, target_names, count, train_idx):
    """Return (H_by_t, Dpt_by_t) using TASK_A_CONFIG's per-target nc on X_v[train_idx]."""
    from pacebench.selection import h_top_M, dpt_top_N
    _a_tbl = get_task_a_config(count)
    X_v64 = X_v.astype(np.float64)
    X_tr  = X_v64[train_idx]
    H_dict, D_dict = {}, {}
    for t_idx, tname in enumerate(target_names):
        if tname not in _a_tbl: continue
        a_cfg = _a_tbl[tname]
        y_tr = Y[train_idx, t_idx]
        H_dict[t_idx] = h_top_M(X_tr, y_tr, count)
        D_dict[t_idx] = dpt_top_N(X_tr, y_tr, Vt_full, count, a_cfg["nc_d"])
    return H_dict, D_dict


def _parse_m_grid(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_q_grid(s):
    """Parse a comma-separated q-grid into floats; entries must be in [0, 1]."""
    out = [float(x) for x in s.split(",") if x.strip()]
    for v in out:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"q-grid value {v} out of [0, 1]")
    return out


def _clamp_m_for_empty_split(count_L, count_G, override_m):
    """When one side of the joint budget is empty, force m to the surviving side
    (m = 0 → only Global; m = 1 → only Local).  Returns the new override_m."""
    if count_L == 0 and count_G > 0:
        if override_m not in (None, 0.0):
            print(f"(warning: count_L=0 → forcing m=0; ignoring --m {override_m})", flush=True)
        return 0.0
    if count_G == 0 and count_L > 0:
        if override_m not in (None, 1.0):
            print(f"(warning: count_G=0 → forcing m=1; ignoring --m {override_m})", flush=True)
        return 1.0
    return override_m


def _resolve_per_target_counts(cfgs, count, override_q):
    """
    Build per-target (count_L, count_G, q_used) using:
      * override_q (CLI --q) if provided — applied uniformly to all targets.
      * Otherwise, each target's cfg["q"] (None → no split, both strategies use full count).

    Also auto-clamps cfg["m"] when one side is empty (cL=0 → m=0, cG=0 → m=1).
    Returns:
      counts : dict {target: (count_L, count_G)}
      uses_split : True if any target has a non-None q, else False
    """
    counts = {}
    uses_split = False
    for tname, cfg in cfgs.items():
        q_t = override_q if override_q is not None else cfg.get("q")
        if q_t is not None:
            uses_split = True
        cL, cG = _split_count(count, q_t)
        counts[tname] = (cL, cG)
        # Per-target m clamp when one side is empty
        if cL == 0 and cG > 0 and cfg["m"] != 0.0:
            print(f"  (target {tname}: count_L=0 → forcing m=0; cfg had m={cfg['m']})",
                  flush=True)
            cfg["m"] = 0.0
        elif cG == 0 and cL > 0 and cfg["m"] != 1.0:
            print(f"  (target {tname}: count_G=0 → forcing m=1; cfg had m={cfg['m']})",
                  flush=True)
            cfg["m"] = 1.0
    return counts, uses_split


def _per_target_counts_for_dump(cfgs, count, override_q):
    """
    Build {target: count_L} and {target: count_G} dicts for selection-dump
    collectors, honoring per-target cfg["q"] (or the global override_q).
    Mirrors _resolve_per_target_counts but without mutating cfgs / re-printing.
    """
    counts_H, counts_G = {}, {}
    for tname, cfg in cfgs.items():
        q_t = override_q if override_q is not None else cfg.get("q")
        cL, cG = _split_count(count, q_t)
        counts_H[tname] = cL
        counts_G[tname] = cG
    return counts_H, counts_G


def _split_count(count, q):
    """
    Split a joint per-target instance budget `count` between Local-SVD and Global-SVD.

    When `q is None`: each strategy independently uses `count` instances (current
    default; effective budget up to `2 * count`).
    When `q in [0, 1]`: budget is jointly capped at `count`; Local-SVD gets
    `round(q * count)`, Global-SVD gets the remainder.
      * q = 0 → only Global-SVD (count_L = 0); callers should clamp m → 0.
      * q = 1 → only Local-SVD  (count_G = 0); callers should clamp m → 1.
      * q in (0, 1) → both strategies, joint cap at count.

    Returns (count_L, count_G).
    """
    if q is None:
        return count, count
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"--q must lie in [0, 1]; got {q}")
    cL = int(round(q * count))
    cG = count - cL
    if cL < 0 or cG < 0 or (cL == 0 and cG == 0):
        raise ValueError(f"--q={q} with count={count} yields total = 0 (degenerate)")
    return cL, cG


def _abs_score(met):
    """Task A optimization objective: minimize (MAE_pp − Spearman).
    MAE is rescaled to percentage points (×100) to put it on the same scale
    as Spearman ∈ [−1, 1]; lower score is better.

    Example: (MAE 3.6%, Sp 0.9) → 3.6 − 0.9 = 2.7  ←  preferred
             (MAE 3.5%, Sp 0.7) → 3.5 − 0.7 = 2.8
    A 0.2 Spearman gain offsets up to ~0.2pp MAE penalty (1:1 trade-off)."""
    return 100 * met["MAE"] - met["Spearman"]


def _autotune_task_a(H_cache, D_cache, Y, target_names, nc_range, m_grid):
    """
    Grid-search per-target (m, nc_h, nc_d) over cached H/D-pt preds, minimizing
    MAE − Spearman (see _abs_score). When one of H_cache / D_cache is empty
    (q = 0 or q = 1), the corresponding side is skipped and m is implicitly clamped.
    """
    from pacebench.metrics import compute_abs_metrics
    has_H, has_D = bool(H_cache), bool(D_cache)
    if not (has_H or has_D):
        raise ValueError("Both H_cache and D_cache are empty; nothing to tune.")
    ref = next(iter((H_cache or D_cache).values()))
    n, T = ref.shape
    cfgs = {}
    preds = np.zeros((n, T))
    grid_rows = []
    for t_idx, tname in enumerate(target_names):
        best = {"score": float("inf")}
        for m in m_grid:
            if m > 0 and not has_H: continue
            if m < 1 and not has_D: continue
            for nc_h in nc_range:
                for nc_d in nc_range:
                    if m == 0.0 and nc_h != nc_range[0]: continue
                    if m == 1.0 and nc_d != nc_range[0]: continue
                    h_part = m * H_cache[nc_h][:, t_idx]       if m > 0 else 0.0
                    d_part = (1 - m) * D_cache[nc_d][:, t_idx] if m < 1 else 0.0
                    p = h_part + d_part
                    met = compute_abs_metrics(Y[:, t_idx], p)
                    score = _abs_score(met)
                    grid_rows.append(dict(target=tname, m=m, nc_h=nc_h, nc_d=nc_d,
                                          score=score, **met))
                    if score < best["score"]:
                        best = {"score": score, "MAE": met["MAE"],
                                "Spearman": met["Spearman"],
                                "m": m, "nc_h": nc_h, "nc_d": nc_d, "preds": p}
        cfgs[tname] = dict(m=best["m"], nc_h=best["nc_h"], nc_d=best["nc_d"])
        preds[:, t_idx] = best["preds"]
    return cfgs, preds, grid_rows


def _autotune_task_a_q(H_cache_qm, D_cache_qm, Y, target_names, nc_range,
                       m_grid, count, q_grid):
    """
    Per-target sweep over (q, m, nc_h, nc_d), minimizing MAE − Spearman
    (see _abs_score).
      * H_cache_qm[(nc, M)] = (n, T) LOMO preds (build for unique M values)
      * D_cache_qm[(nc, M)] = (n, T) LOMO preds
      * For each q in q_grid: derive (cL, cG); skip configs whose required
        side is missing.
    Returns cfgs[target] = dict(m, nc_h, nc_d, q), preds (n, T), grid_rows.
    """
    from pacebench.metrics import compute_abs_metrics
    n = Y.shape[0]; T = Y.shape[1]
    cfgs = {}
    preds = np.zeros((n, T))
    grid_rows = []
    for t_idx, tname in enumerate(target_names):
        best = {"score": float("inf")}
        for q_val in q_grid:
            cL, cG = _split_count(count, q_val)
            has_H = (cL > 0) and any((nc_, cL) in H_cache_qm for nc_ in nc_range)
            has_D = (cG > 0) and any((nc_, cG) in D_cache_qm for nc_ in nc_range)
            for m in m_grid:
                if m > 0 and not has_H: continue
                if m < 1 and not has_D: continue
                for nc_h in nc_range:
                    for nc_d in nc_range:
                        if m == 0.0 and nc_h != nc_range[0]: continue
                        if m == 1.0 and nc_d != nc_range[0]: continue
                        h_part = m * H_cache_qm[(nc_h, cL)][:, t_idx]       if m > 0 else 0.0
                        d_part = (1 - m) * D_cache_qm[(nc_d, cG)][:, t_idx] if m < 1 else 0.0
                        p = h_part + d_part
                        met = compute_abs_metrics(Y[:, t_idx], p)
                        score = _abs_score(met)
                        grid_rows.append(dict(target=tname, q=q_val, m=m, nc_h=nc_h,
                                              nc_d=nc_d, score=score, **met))
                        if score < best["score"]:
                            best = {"score": score,
                                    "MAE": met["MAE"], "Spearman": met["Spearman"],
                                    "q": q_val, "m": m, "nc_h": nc_h, "nc_d": nc_d,
                                    "preds": p}
        cfgs[tname] = dict(m=best["m"], nc_h=best["nc_h"], nc_d=best["nc_d"],
                           q=best["q"])
        preds[:, t_idx] = best["preds"]
    return cfgs, preds, grid_rows


def _autotune_task_a_strict_q(H_cache, D_cache, Y, target_names,
                                q_grid, nc_h_grid, nc_d_grid, m_grid, count):
    """
    Strict-budget per-target sweep over (q, m, nc_h, nc_d), minimizing _abs_score.

      H_cache : dict[(q, nc_h, nc_d)] → (n, T) preds
      D_cache : dict[(q, nc_d)]       → (n, T) preds

    Returns cfgs[target] = dict(m, nc_h, nc_d, q), preds (n, T), grid_rows.
    Skips redundant combos: nc_h irrelevant when m=0; same nc_d still sweeps.
    """
    from pacebench.metrics import compute_abs_metrics
    n = Y.shape[0]; T = Y.shape[1]
    cfgs = {}
    preds = np.zeros((n, T))
    grid_rows = []
    for t_idx, tname in enumerate(target_names):
        best = {"score": float("inf")}
        for q_val in q_grid:
            has_H = q_val > 0
            has_D = q_val < 1
            for nc_d in nc_d_grid:
                d_pred = D_cache[(q_val, nc_d)][:, t_idx] if has_D else None
                for m in m_grid:
                    if m > 0 and not has_H: continue
                    if m < 1 and not has_D: continue
                    for nc_h in nc_h_grid:
                        if m == 0.0 and nc_h != nc_h_grid[0]: continue
                        h_pred = (H_cache[(q_val, nc_h, nc_d)][:, t_idx]
                                  if (has_H and m > 0) else None)
                        h_part = m * h_pred       if h_pred is not None else 0.0
                        d_part = (1 - m) * d_pred if d_pred is not None else 0.0
                        p = h_part + d_part
                        if np.isscalar(p):
                            continue
                        met = compute_abs_metrics(Y[:, t_idx], p)
                        score = _abs_score(met)
                        grid_rows.append(dict(target=tname, q=q_val, m=m,
                                              nc_h=nc_h, nc_d=nc_d,
                                              score=score, **met))
                        if score < best["score"]:
                            best = {"score": score, "MAE": met["MAE"],
                                    "Spearman": met["Spearman"], "q": q_val,
                                    "m": m, "nc_h": nc_h, "nc_d": nc_d, "preds": p}
        cfgs[tname] = dict(m=best["m"], nc_h=best["nc_h"], nc_d=best["nc_d"],
                           q=best["q"])
        preds[:, t_idx] = best["preds"]
    return cfgs, preds, grid_rows


def _autotune_task_b_q(H_cache_qm, D_cache_qm, Y, target_names, nc_range,
                        m_grid, count, q_grid):
    """Same as _autotune_task_a_q but for Task B pair-margin tensors and pair accuracy."""
    n = Y.shape[0]; T = Y.shape[1]
    cfgs = {}
    preds = np.full((n, n, T), 0.0)
    grid_rows = []
    eye = np.eye(n, dtype=bool)
    for t_idx, tname in enumerate(target_names):
        y_col = Y[:, t_idx]
        true_mat = y_col.reshape(-1, 1) - y_col.reshape(1, -1)
        mask_base = (~eye) & (np.abs(true_mat) >= 1e-9)
        best = {"acc": -1.0}
        for q_val in q_grid:
            cL, cG = _split_count(count, q_val)
            has_H = (cL > 0) and any((nc_, cL) in H_cache_qm for nc_ in nc_range)
            has_D = (cG > 0) and any((nc_, cG) in D_cache_qm for nc_ in nc_range)
            for m in m_grid:
                if m > 0 and not has_H: continue
                if m < 1 and not has_D: continue
                for nc_h in nc_range:
                    for nc_d in nc_range:
                        if m == 0.0 and nc_h != nc_range[0]: continue
                        if m == 1.0 and nc_d != nc_range[0]: continue
                        h_part = m * H_cache_qm[(nc_h, cL)][:, :, t_idx]       if m > 0 else 0.0
                        d_part = (1 - m) * D_cache_qm[(nc_d, cG)][:, :, t_idx] if m < 1 else 0.0
                        p = h_part + d_part
                        mask = mask_base & ~np.isnan(p)
                        n_total = int(mask.sum())
                        acc = (float(((p > 0) == (true_mat > 0)).astype(bool)[mask].sum())
                               / n_total) if n_total else 0.0
                        grid_rows.append(dict(target=tname, q=q_val, m=m, nc_h=nc_h,
                                              nc_d=nc_d, acc=acc, n_total=n_total))
                        if acc > best["acc"]:
                            best = {"acc": acc, "q": q_val, "m": m, "nc_h": nc_h,
                                    "nc_d": nc_d, "preds": p, "n_total": n_total}
        cfgs[tname] = dict(m=best["m"], nc_h=best["nc_h"], nc_d=best["nc_d"],
                           q=best["q"], reg="logit")
        preds[:, :, t_idx] = best["preds"]
    return cfgs, preds, grid_rows


def _autotune_task_b(H_cache, D_cache, Y, target_names, nc_range, m_grid):
    """
    Grid-search per-target (m, nc_h, nc_d) over cached H/D-pt LOMO margin tensors
    by maximum pair accuracy. (reg is fixed to logit for now.)
    Empty H_cache or D_cache (q = 0 / 1) is supported and clamps m accordingly.
    """
    has_H, has_D = bool(H_cache), bool(D_cache)
    if not (has_H or has_D):
        raise ValueError("Both H_cache and D_cache are empty; nothing to tune.")
    n = Y.shape[0]
    T = Y.shape[1]
    cfgs, grid_rows = {}, []
    preds = np.full((n, n, T), 0.0)
    eye = np.eye(n, dtype=bool)
    for t_idx, tname in enumerate(target_names):
        y_col = Y[:, t_idx]
        true_mat = y_col.reshape(-1, 1) - y_col.reshape(1, -1)
        mask_base = (~eye) & (np.abs(true_mat) >= 1e-9)
        best = {"acc": -1.0}
        for m in m_grid:
            if m > 0 and not has_H: continue
            if m < 1 and not has_D: continue
            for nc_h in nc_range:
                for nc_d in nc_range:
                    if m == 0.0 and nc_h != nc_range[0]: continue
                    if m == 1.0 and nc_d != nc_range[0]: continue
                    h_part = m * H_cache[nc_h][:, :, t_idx]       if m > 0 else 0.0
                    d_part = (1 - m) * D_cache[nc_d][:, :, t_idx] if m < 1 else 0.0
                    p = h_part + d_part
                    mask = mask_base & ~np.isnan(p)
                    n_total = int(mask.sum())
                    acc = (float(((p > 0) == (true_mat > 0)).astype(bool)[mask].sum())
                           / n_total) if n_total else 0.0
                    grid_rows.append(dict(target=tname, m=m, nc_h=nc_h, nc_d=nc_d,
                                          acc=acc, n_total=n_total))
                    if acc > best["acc"]:
                        best = {"acc": acc, "m": m, "nc_h": nc_h, "nc_d": nc_d,
                                "preds": p, "n_total": n_total}
        cfgs[tname] = dict(m=best["m"], nc_h=best["nc_h"], nc_d=best["nc_d"],
                           reg="logit")
        preds[:, :, t_idx] = best["preds"]
    return cfgs, preds, grid_rows


def _print_autotune_config(task_label, count, cfgs, extra_note=""):
    """Emit a paste-ready dict block for the user's config.py."""
    print(f"\n# ─────────────── paste into {task_label}_CONFIGS[{count}] "
          + (f"{extra_note} " if extra_note else "") + "───────────────")
    for tname, c in cfgs.items():
        tail_reg = (f', reg="{c["reg"]}"' if "reg" in c else "")
        tail_q   = (f', q={c["q"]:.1f}'   if c.get("q") is not None else "")
        print(f'    "{tname}": dict(m={c["m"]:.1f}, nc_h={c["nc_h"]:2d}, '
              f'nc_d={c["nc_d"]:2d}{tail_reg}{tail_q}),')
    print()


def _write_selection_rows(rows, source_col_info_v, out_path, *, task, mode, count):
    """Convert selection dicts (with col_idx; optional 'weight') + col info
    into long-format CSV(s).

    Output layout (out_path is treated as a base directory; ``count`` is the
    per-strategy budget C and is encoded in the file name):
        LOMO   :  <base>/<task>_loocv[_pinA]/<held-out-model>/selections_C<count>.csv
        FIT-ALL:  <base>/<task>_fit[_pinA]/selections_C<count>.csv
        SPLIT  :  <base>/<task>_split/selections_C<count>.csv
    A trailing ``.csv`` on out_path is silently stripped so users can keep
    passing the legacy ``selections.csv`` argument.
    """
    # Treat out_path as base directory; strip legacy trailing .csv
    base_str = str(out_path)
    if base_str.endswith(".csv"):
        base_str = base_str[:-4]
    base = Path(base_str)

    task_prefix = "abs" if task == "A" else "pair"
    if "LOMO" in mode:
        kind = "loocv"
    elif "FIT-ALL" in mode:
        kind = "fit"
    elif "SPLIT" in mode:
        kind = "split"
    else:
        kind = "other"
    sub = f"{task_prefix}_{kind}"
    if "pinA" in mode:
        sub += "_pinA"
    out_dir = base / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"selections_C{count}.csv"

    # Enrich rows with benchmark / subdir / instance_id (slim schema)
    enriched = []
    for r in rows:
        col = r["col_idx"]
        dname, sdir, iid = source_col_info_v[col]
        entry = dict(
            mode=mode, fold=r.get("fold", ""), target=r["target"],
            benchmark=dname, subdir=sdir, instance_id=iid, col_idx=col,
        )
        if "weight" in r:
            entry["weight"] = r["weight"]
        enriched.append(entry)
    df = pd.DataFrame(enriched)
    has_w = (not df.empty) and "weight" in df.columns

    if "LOMO" in mode and not df.empty:
        # One CSV per held-out model
        n_files = 0
        for fold_name, group in df.groupby("fold"):
            fold_dir = out_dir / str(fold_name)
            fold_dir.mkdir(parents=True, exist_ok=True)
            group.to_csv(fold_dir / fname, index=False)
            n_files += 1
        print(f"Selections dumped ({len(df):,} rows"
              + (", with weights" if has_w else "")
              + f") → {out_dir}/<fold>/{fname} ({n_files} fold dirs)",
              flush=True)
    else:
        out_file = out_dir / fname
        df.to_csv(out_file, index=False)
        print(f"Selections dumped ({len(df):,} rows"
              + (", with weights" if has_w else "")
              + f") → {out_file}", flush=True)


def _describe(cfg):
    m = cfg["m"]
    if m == 0.0: return f"D-pt(nc={cfg['nc_d']})"
    if m == 1.0: return f"H(nc={cfg['nc_h']})"
    return f"{m}·H(nc={cfg['nc_h']}) + {1-m:.1f}·D-pt(nc={cfg['nc_d']})"


# ──────────────────────────────────────────────────────────────
# Task A: absolute-score prediction
# ──────────────────────────────────────────────────────────────

def run_abs(B, seed, sources=None, targets=None, count=COUNT,
            override_m=None, override_nc_h=None, override_nc_d=None,
            train_models=None, eval_models=None, dump_selections=None,
            auto_tune=False, nc_max=10, m_grid="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
            q=None, strict_budget=False):
    count_L, count_G = _split_count(count, q)
    override_m = _clamp_m_for_empty_split(count_L, count_G, override_m)
    if strict_budget:
        if q is None:
            print(f"Task A config: C={count} STRICT-BUDGET (|H ∪ D| = {count} per target, "
                  f"per-target q from cfg), pooled B={B}, seed={seed}", flush=True)
        else:
            print(f"Task A config: C={count} STRICT-BUDGET, q={q} (|H ∪ D| = {count}, "
                  f"|H|/(|H|+|D|) ≈ {q}), pooled B={B}, seed={seed}", flush=True)
    elif q is None:
        print(f"Task A config: C={count} (per-strategy), pooled B={B}, seed={seed}", flush=True)
    else:
        print(f"Task A config: C_total={count}, q={q} → Local={count_L} / Global={count_G}, "
              f"pooled B={B}, seed={seed}", flush=True)
    X_v, Y, target_names, P_target, Vt_full, source_col_info_v = _load_and_svd(sources, targets)

    train_idx, eval_idx = _resolve_model_split(train_models, eval_models)
    out_dir = Path(BASE_DIR).parent.parent

    if auto_tune and train_idx is not None:
        raise ValueError("--auto-tune is incompatible with --train-models/--eval-models "
                         "(would leak eval into hyperparam selection).")

    cfgs = _resolve_configs(get_task_a_config(count), target_names,
                            override_m, override_nc_h, override_nc_d,
                            fallback_m=0.5, fallback_nc_h=2, fallback_nc_d=7)

    if train_idx is not None:
        # ── Custom train/eval split path ──────────────────────────
        print(f"\nCustom split:  train={len(train_idx)} models, eval={len(eval_idx)} models",
              flush=True)
        print(f"  train: {[MODELS[i] for i in train_idx]}", flush=True)
        print(f"  eval : {[MODELS[i] for i in eval_idx]}", flush=True)

        preds_eval = np.zeros((len(eval_idx), len(target_names)))
        sel_rows = []
        for tname, cfg in cfgs.items():
            t_idx = target_names.index(tname)
            m = cfg["m"]
            preds_H = preds_D = 0
            if m > 0:
                preds_H_all, sel_H = fit_split_H_abs(
                    X_v, Y, P_target, train_idx, eval_idx,
                    cfg["nc_h"], count_L, B, seed)
                preds_H = preds_H_all[:, t_idx]
                for rank, col in enumerate(sel_H[t_idx]):
                    sel_rows.append(dict(fold="SPLIT", target=tname, strategy="H",
                                         nc=cfg["nc_h"], rank=rank + 1, col_idx=int(col)))
            if m < 1:
                preds_D_all, sel_D = fit_split_Dpt_abs(
                    X_v, Y, Vt_full, P_target, train_idx, eval_idx,
                    cfg["nc_d"], count_G, B, seed)
                preds_D = preds_D_all[:, t_idx]
                for rank, col in enumerate(sel_D[t_idx]):
                    sel_rows.append(dict(fold="SPLIT", target=tname, strategy="D-pt",
                                         nc=cfg["nc_d"], rank=rank + 1, col_idx=int(col)))
            preds_eval[:, t_idx] = m * preds_H + (1 - m) * preds_D

        Y_eval = Y[eval_idx]
        df = build_abs_metrics_df_eval(Y_eval, preds_eval, target_names)

        print("\n" + "═" * 75, flush=True)
        print(f"Task A (absolute) CUSTOM SPLIT — per-target config at C={count}", flush=True)
        for tname, cfg in cfgs.items():
            print(f"  {tname:24s} → {_describe(cfg)}", flush=True)
        print("═" * 75, flush=True)
        print_abs_results(df, label=f"Task A SPLIT  C={count}, B={B}, seed={seed}")

        out_csv = out_dir / "abs_predictions_split.csv"
        rows = []
        for e_pos, e_i in enumerate(eval_idx):
            for t_idx, tname in enumerate(target_names):
                rows.append({
                    "eval_model": MODELS[e_i], "target": tname,
                    "predicted":  float(preds_eval[e_pos, t_idx]),
                    "actual":     float(Y_eval[e_pos, t_idx]),
                    "abs_error":  float(abs(preds_eval[e_pos, t_idx] - Y_eval[e_pos, t_idx])),
                })
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nPer-eval-model predictions saved → {out_csv}", flush=True)

        if dump_selections:
            _write_selection_rows(sel_rows, source_col_info_v, dump_selections,
                                  task="A", mode="SPLIT", count=count)
        return

    # ── LOMO path ──────────────────────────────────────────────
    if strict_budget and auto_tune:
        nc_h_grid = list(range(1, nc_max + 1))
        nc_d_grid = list(range(1, nc_max + 1))
        m_list   = _parse_m_grid(m_grid)
        if q is not None:
            q_list = [float(q)]
            print(f"\n[strict-budget auto-tune] sweeping nc_h ∈ {nc_h_grid}, "
                  f"nc_d ∈ {nc_d_grid}, m ∈ {m_list} (q = {q} fixed)", flush=True)
        else:
            q_list = [round(i * 0.1, 1) for i in range(11)]
            print(f"\n[strict-budget auto-tune] sweeping q ∈ {q_list}, "
                  f"nc_h ∈ {nc_h_grid}, nc_d ∈ {nc_d_grid}, m ∈ {m_list}",
                  flush=True)
        n_combos = len(q_list) * len(nc_h_grid) * len(nc_d_grid)
        print(f"  ({len(q_list)} q × {len(nc_h_grid)} nc_h × {len(nc_d_grid)} "
              f"nc_d = {n_combos} cache entries; "
              f"y_pool reused per (fold, target))", flush=True)
        H_cache, D_cache, sel_cache = loocv_joint_abs_autotune(
            X_v, Y, Vt_full, q_list, nc_h_grid, nc_d_grid, count,
            P_target, B, seed)
        cfgs, preds, _ = _autotune_task_a_strict_q(
            H_cache, D_cache, Y, target_names,
            q_list, nc_h_grid, nc_d_grid, m_list, count)
        note = "(strict-budget auto-tuned, q fixed)" if q is not None \
               else "(strict-budget auto-tuned w/ q-sweep)"
        _print_autotune_config("TASK_A", count, cfgs, extra_note=note)
        # Build joint_sels (per-fold per-target (h_idx, d_idx)) using the
        # auto-tuned (q, nc_d) per target — matches what dump-selections expects.
        n = X_v.shape[0]; T = len(target_names)
        joint_sels = [[None] * T for _ in range(n)]
        for t_idx, tname in enumerate(target_names):
            cfg = cfgs[tname]
            sel_qd = sel_cache[(cfg["q"], cfg["nc_d"])]
            for fold_i in range(n):
                joint_sels[fold_i][t_idx] = sel_qd[fold_i][t_idx]
    elif strict_budget:
        # Per-target (q, C, nc_h, nc_d) — q from CLI override or per-target cfg
        per_target_qC, nc_h_per_t, nc_d_per_t = {}, {}, {}
        for t_idx, tname in enumerate(target_names):
            cfg = cfgs[tname]
            q_t = q if q is not None else cfg.get("q")
            if q_t is None: q_t = 0.5   # uniform default
            per_target_qC[t_idx] = (float(q_t), int(count))
            nc_h_per_t[t_idx] = int(cfg["nc_h"])
            nc_d_per_t[t_idx] = int(cfg["nc_d"])
        print(f"\n[strict-budget] Per-target (q, C, nc_h, nc_d, m):", flush=True)
        for t_idx, tname in enumerate(target_names):
            q_t, _ = per_target_qC[t_idx]
            cfg = cfgs[tname]
            print(f"  {tname:24s}  q={q_t:.2f}  C={count}  nc_h={cfg['nc_h']}  "
                  f"nc_d={cfg['nc_d']}  m={cfg['m']}", flush=True)
        print(f"\nRunning joint LOMO (|H ∪ D| = {count} per fold-target) …", flush=True)
        preds_H, preds_D, joint_sels = loocv_joint_abs(
            X_v, Y, Vt_full, per_target_qC, nc_h_per_t, nc_d_per_t,
            P_target, B, seed)
        preds = np.zeros_like(preds_H)
        for tname, cfg in cfgs.items():
            t_idx = target_names.index(tname)
            m = cfg["m"]
            preds[:, t_idx] = m * preds_H[:, t_idx] + (1 - m) * preds_D[:, t_idx]
    elif auto_tune:
        nc_range = list(range(1, nc_max + 1))
        m_list   = _parse_m_grid(m_grid)
        # q sweep: if user explicitly passed --q, that single value is used;
        # otherwise auto-tune always sweeps q ∈ {0.0, 0.1, ..., 1.0} so each
        # target can pick its best joint-budget split.
        if q is not None:
            q_list = [q]
            print(f"\n[auto-tune] sweeping nc ∈ {nc_range}, m ∈ {m_list} "
                  f"(q = {q} fixed)", flush=True)
        else:
            q_list = [round(i * 0.1, 1) for i in range(11)]   # 0.0..1.0 step 0.1
            print(f"\n[auto-tune] sweeping q ∈ {q_list}, nc ∈ {nc_range}, "
                  f"m ∈ {m_list}", flush=True)

        # Build cache for each unique (count_L, count_G) implied by q_list.
        unique_M_L = set(); unique_M_G = set()
        for q_val in q_list:
            cL, cG = _split_count(count, q_val)
            if cL > 0: unique_M_L.add(cL)
            if cG > 0: unique_M_G.add(cG)
        unique_M_L, unique_M_G = sorted(unique_M_L), sorted(unique_M_G)

        H_cache_qm, D_cache_qm = {}, {}    # keyed by (nc, M)
        for nc_ in nc_range:
            for M in unique_M_L:
                # print(f"  • H(nc={nc_}, M={M}) …", flush=True)
                H_cache_qm[(nc_, M)] = loocv_H_abs(X_v, Y, nc_, M, P_target, B, seed)
        for nc_ in nc_range:
            for M in unique_M_G:
                # print(f"  • D-pt(nc={nc_}, N={M}) …", flush=True)
                D_cache_qm[(nc_, M)] = loocv_Dpt_abs(X_v, Y, Vt_full, nc_, M,
                                                     P_target, B, seed)
        cfgs, preds, _ = _autotune_task_a_q(H_cache_qm, D_cache_qm, Y, target_names,
                                            nc_range, m_list, count, q_list)
        note = "(auto-tuned, q fixed)" if q is not None else "(auto-tuned w/ q-sweep)"
        _print_autotune_config("TASK_A", count, cfgs, extra_note=note)
    else:
        # Per-target counts (cfg["q"] honored when --q is None)
        per_target_counts, uses_split = _resolve_per_target_counts(cfgs, count, q)
        nc_h_set = sorted({c["nc_h"] for c in cfgs.values() if c["m"] > 0})
        nc_d_set = sorted({c["nc_d"] for c in cfgs.values() if c["m"] < 1})
        unique_M_L = sorted({cL for tname, (cL, _) in per_target_counts.items()
                             if cL > 0 and cfgs[tname]["m"] > 0})
        unique_M_G = sorted({cG for tname, (_, cG) in per_target_counts.items()
                             if cG > 0 and cfgs[tname]["m"] < 1})
        if uses_split or q is not None:
            print(f"\nPer-target counts: " +
                  ", ".join(f"{t}=(L={cL},G={cG})" for t, (cL, cG) in per_target_counts.items()),
                  flush=True)
        print(f"\nBase predictors needed:  H(nc ∈ {nc_h_set}, M ∈ {unique_M_L}),  "
              f"D-pt(nc ∈ {nc_d_set}, N ∈ {unique_M_G})", flush=True)

        # Cache by (nc, M); same call cost as before when M is uniform across targets
        H_cache = {}     # {(nc, M): (n, T)}
        for nc_ in nc_h_set:
            for M in unique_M_L:
                print(f"  • H(nc={nc_}, M={M}) …", flush=True)
                H_cache[(nc_, M)] = loocv_H_abs(X_v, Y, nc_, M, P_target, B, seed)
        D_cache = {}
        for nc_ in nc_d_set:
            for M in unique_M_G:
                print(f"  • D-pt(nc={nc_}, N={M}) …", flush=True)
                D_cache[(nc_, M)] = loocv_Dpt_abs(X_v, Y, Vt_full, nc_, M, P_target, B, seed)

        idx = {t: i for i, t in enumerate(target_names)}
        ref_arr = (next(iter(H_cache.values())) if H_cache
                   else next(iter(D_cache.values())))
        preds = np.zeros_like(ref_arr)
        for tname, cfg in cfgs.items():
            t_idx = idx[tname]
            cL, cG = per_target_counts[tname]
            m = cfg["m"]
            h_part = m * H_cache[(cfg["nc_h"], cL)][:, t_idx] if (m > 0 and cL > 0) else 0
            d_part = (1 - m) * D_cache[(cfg["nc_d"], cG)][:, t_idx] if (m < 1 and cG > 0) else 0
            preds[:, t_idx] = h_part + d_part

    df = build_abs_metrics_df(preds, Y, target_names)

    print("\n" + "═" * 75, flush=True)
    print(f"Task A (absolute) — per-target config at C={count}", flush=True)
    for tname, cfg in cfgs.items():
        print(f"  {tname:24s} → {_describe(cfg)}", flush=True)
    print("═" * 75, flush=True)
    print_abs_results(df, label=f"Task A  C={count}, B={B}, seed={seed}")

    out_csv = out_dir / "abs_predictions.csv"
    rows = []
    for i, m in enumerate(MODELS):
        for t_idx, tname in enumerate(target_names):
            rows.append({
                "model":     m,
                "target":    tname,
                "predicted": float(preds[i, t_idx]),
                "actual":    float(Y[i, t_idx]),
                "abs_error": float(abs(preds[i, t_idx] - Y[i, t_idx])),
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nPer-model predictions saved → {out_csv}", flush=True)

    out_png = out_dir / "abs_loocv_fit.png"
    plot_abs_loocv_fit(Y, preds, target_names, df, out_path=str(out_png),
                       title=f"Task A LOMO: Predicted vs Actual  (C={COUNT}, B={B})")

    if dump_selections:
        if strict_budget:
            print(f"\nDumping strict-budget joint LOMO selections + weights "
                  f"(|H ∪ D| = {count} per fold-target) …", flush=True)
            # Build {fold_i: {t_idx: h_idx}} and same for d_idx for collectors
            n_models = len(joint_sels)
            h_pinned = {fold_i: {t_idx: hi for t_idx, (hi, _) in enumerate(fs) if len(hi) > 0}
                        for fold_i, fs in enumerate(joint_sels)}
            d_pinned = {fold_i: {t_idx: di for t_idx, (_, di) in enumerate(fs) if len(di) > 0}
                        for fold_i, fs in enumerate(joint_sels)}
            rows = []
            for r in collect_lomo_weights_abs(X_v, Y, Vt_full, target_names,
                                               cfgs, P_target, B, seed, count, "H",
                                               pinned_sel_by_fold_t=h_pinned):
                r["fold"] = MODELS[r.pop("held_out_idx")]
                rows.append(r)
            for r in collect_lomo_weights_abs(X_v, Y, Vt_full, target_names,
                                               cfgs, P_target, B, seed, count, "D-pt",
                                               pinned_sel_by_fold_t=d_pinned):
                r["fold"] = MODELS[r.pop("held_out_idx")]
                rows.append(r)
            _write_selection_rows(rows, source_col_info_v, dump_selections,
                                  task="A", mode="LOMO", count=count)
        else:
            print(f"\nCollecting LOMO selections + weights …", flush=True)
            counts_H, counts_G = _per_target_counts_for_dump(cfgs, count, q)
            rows = []
            for strat in ("H", "D-pt"):
                cnt = counts_H if strat == "H" else counts_G
                sel_rows = collect_lomo_weights_abs(X_v, Y, Vt_full, target_names,
                                                    cfgs, P_target, B, seed, cnt, strat)
                for r in sel_rows:
                    r["fold"] = MODELS[r.pop("held_out_idx")]
                    rows.append(r)
            _write_selection_rows(rows, source_col_info_v, dump_selections,
                                  task="A", mode="LOMO", count=count)


# ──────────────────────────────────────────────────────────────
# Task B: pairwise preference prediction
# ──────────────────────────────────────────────────────────────

def _describe_pair(cfg):
    reg = cfg.get("reg", "logit")
    suffix = "_logit" if reg == "logit" else "_ols"
    base = _describe(cfg).replace('H(', f'H{suffix}(').replace('D-pt(', f'D-pt{suffix}(')
    return base


def _lomo_pair_dispatch(strategy, reg, X_v, Y, Vt_full, nc, P_target, count, B, seed,
                         pinned_sel_by_fold_t=None):
    """Dispatch to the correct LOMO function by (strategy, reg) pair."""
    if strategy == "H":
        if reg == "ols":
            return lomo_H_pair_ols(X_v, Y, nc, count, P_target, B, seed,
                                   pinned_sel_by_fold_t=pinned_sel_by_fold_t)
        return lomo_H_logit(X_v, Y, nc, count, P_target, B, seed, DEFAULT_ALPHA,
                            pinned_sel_by_fold_t=pinned_sel_by_fold_t)
    else:  # D-pt
        if reg == "ols":
            return lomo_Dpt_pair_ols(X_v, Y, Vt_full, nc, count, P_target, B, seed,
                                     pinned_sel_by_fold_t=pinned_sel_by_fold_t)
        return lomo_Dpt_logit(X_v, Y, Vt_full, nc, count, P_target, B, seed, DEFAULT_ALPHA,
                              pinned_sel_by_fold_t=pinned_sel_by_fold_t)


def run_pair(B, seed, sources=None, targets=None, count=COUNT,
             override_m=None, override_nc_h=None, override_nc_d=None,
             train_models=None, eval_models=None, dump_selections=None,
             pin_task_a_selection=False,
             auto_tune=False, nc_max=10,
             m_grid="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
             q=None, strict_budget=False):
    count_L, count_G = _split_count(count, q)
    override_m = _clamp_m_for_empty_split(count_L, count_G, override_m)
    if strict_budget:
        if not pin_task_a_selection:
            raise ValueError("--strict-budget on `pair` requires --pin-task-a-selection "
                             "(joint selection has to come from Task A; pair-only joint "
                             "auto-tune is not implemented).")
        print(f"Task B config: C={count} STRICT-BUDGET (|H ∪ D| = {count} per target, "
              f"selection pinned to abs --strict-budget), pooled B={B}, seed={seed}",
              flush=True)
    elif q is None:
        print(f"Task B config: C={count} (per-strategy), pooled B={B}, seed={seed}", flush=True)
    else:
        print(f"Task B config: C_total={count}, q={q} → Local={count_L} / Global={count_G}, "
              f"pooled B={B}, seed={seed}", flush=True)
    X_v, Y, target_names, P_target, Vt_full, source_col_info_v = _load_and_svd(sources, targets)

    cfg_source = (get_task_b_config_pinned_a(count)
                  if pin_task_a_selection else get_task_b_config(count))
    cfgs = _resolve_configs(cfg_source, target_names,
                            override_m, override_nc_h, override_nc_d,
                            fallback_m=0.5, fallback_nc_h=2, fallback_nc_d=6,
                            fallback_reg="ols")

    train_idx, eval_idx = _resolve_model_split(train_models, eval_models)
    out_dir = Path(BASE_DIR).parent.parent

    if auto_tune and train_idx is not None:
        raise ValueError("--auto-tune is incompatible with --train-models/--eval-models "
                         "(would leak eval into hyperparam selection).")

    if pin_task_a_selection:
        print("\n[--pin-task-a-selection] Pinning Task B's selection to Task A's "
              "col indices (Task B keeps its own m, nc, reg for the predictor).",
              flush=True)

    if train_idx is not None:
        # ── Custom train/eval split path ──────────────────────────
        print(f"\nCustom split:  train={len(train_idx)} models, eval={len(eval_idx)} models",
              flush=True)
        print(f"  train: {[MODELS[i] for i in train_idx]}", flush=True)
        print(f"  eval : {[MODELS[i] for i in eval_idx]}", flush=True)
        print(f"  Pair rule: (k, j) populated iff k ∈ eval OR j ∈ eval", flush=True)

        pinned_H_by_t = pinned_D_by_t = None
        if pin_task_a_selection:
            pinned_H_by_t, pinned_D_by_t = _compute_task_a_selections_split(
                X_v, Y, Vt_full, target_names, count, train_idx)

        n = X_v.shape[0]
        preds = np.full((n, n, len(target_names)), np.nan)
        sel_rows = []
        for tname, cfg in cfgs.items():
            t_idx = target_names.index(tname)
            m, reg = cfg["m"], cfg["reg"]
            h_part = d_part = None
            if m > 0:
                psel_H = {t_idx: pinned_H_by_t[t_idx]} if pinned_H_by_t and t_idx in pinned_H_by_t else None
                preds_H, sel_H = fit_split_H_pair(
                    X_v, Y, P_target, train_idx, eval_idx,
                    cfg["nc_h"], count, B, seed, reg, DEFAULT_ALPHA,
                    pinned_sel_by_t=psel_H)
                h_part = preds_H[:, :, t_idx]
                for rank, col in enumerate(sel_H[t_idx]):
                    sel_rows.append(dict(fold="SPLIT", target=tname, strategy="H",
                                         nc=cfg["nc_h"], rank=rank + 1, col_idx=int(col)))
            if m < 1:
                psel_D = {t_idx: pinned_D_by_t[t_idx]} if pinned_D_by_t and t_idx in pinned_D_by_t else None
                preds_D, sel_D = fit_split_Dpt_pair(
                    X_v, Y, Vt_full, P_target, train_idx, eval_idx,
                    cfg["nc_d"], count, B, seed, reg, DEFAULT_ALPHA,
                    pinned_sel_by_t=psel_D)
                d_part = preds_D[:, :, t_idx]
                for rank, col in enumerate(sel_D[t_idx]):
                    sel_rows.append(dict(fold="SPLIT", target=tname, strategy="D-pt",
                                         nc=cfg["nc_d"], rank=rank + 1, col_idx=int(col)))
            # Mix: both parts share the same NaN mask (pair filter)
            if h_part is not None and d_part is not None:
                combined = m * h_part + (1 - m) * d_part
            elif h_part is not None:
                combined = h_part
            else:
                combined = d_part
            preds[:, :, t_idx] = combined

        df = compute_pair_accuracy_df(preds, Y, target_names)

        print("\n" + "═" * 80, flush=True)
        print(f"Task B (pairwise) CUSTOM SPLIT — per-target config at C={count}", flush=True)
        for tname, cfg in cfgs.items():
            print(f"  {tname:24s} → {_describe_pair(cfg)}", flush=True)
        print("═" * 80, flush=True)
        print_pair_results(df, label=f"Task B SPLIT  C={count}, B={B}, seed={seed}")

        # Per-pair CSV (only populated pairs)
        eval_set = set(int(e) for e in eval_idx)
        out_csv = out_dir / "pair_predictions_split.csv"
        rows = []
        for k in range(n):
            for j in range(n):
                if k == j: continue
                if not ((k in eval_set) or (j in eval_set)): continue
                for t_idx, tname in enumerate(target_names):
                    m_pred = float(preds[k, j, t_idx])
                    if not np.isfinite(m_pred): continue
                    true_margin = float(Y[k, t_idx] - Y[j, t_idx])
                    rows.append({
                        "k_model": MODELS[k], "j_model": MODELS[j], "target": tname,
                        "true_margin": true_margin,
                        "true_winner": MODELS[k] if true_margin > 0 else
                                       (MODELS[j] if true_margin < 0 else "tie"),
                        "pred_margin": m_pred,
                        "pred_winner": MODELS[k] if m_pred > 0 else MODELS[j],
                    })
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nPer-pair predictions saved → {out_csv}", flush=True)

        if dump_selections:
            _write_selection_rows(sel_rows, source_col_info_v, dump_selections,
                                  task="B", mode="SPLIT", count=count)
        return

    # ── LOMO path ──────────────────────────────────────────────
    pinned_H_lomo = pinned_D_lomo = None
    pin_a_cfgs_used = None   # A's per-target config actually used for pinning (frozen or auto-tuned)
    if pin_task_a_selection:
        a_select_fn = (_compute_task_a_selections_lomo_strict
                       if strict_budget else _compute_task_a_selections_lomo)
        pinned_H_lomo, pinned_D_lomo, pin_a_cfgs_used = a_select_fn(
            X_v, Y, Vt_full, target_names, count,
            P_target=P_target, B=B, seed=seed,
            auto_tune=auto_tune, nc_max=nc_max,
            m_grid=(_parse_m_grid(m_grid) if auto_tune else None),
            q=q)

    if auto_tune:
        nc_range = list(range(1, nc_max + 1))
        m_list   = _parse_m_grid(m_grid)
        # q sweep:
        #   * --q given        → fixed at that single value
        #   * pin-A on         → B's selection is fixed by A's pin; B's q does
        #                        not affect predictions (pinned_sel overrides M).
        #                        Skip the sweep (single dummy q).
        #   * else             → sweep q ∈ {0.0, 0.1, ..., 1.0}
        if q is not None:
            q_list = [q]
            print(f"\n[auto-tune] sweeping nc ∈ {nc_range}, m ∈ {m_list}, reg=logit "
                  f"(q = {q} fixed)", flush=True)
        elif pin_task_a_selection:
            q_list = [None]
            print(f"\n[auto-tune] sweeping nc ∈ {nc_range}, m ∈ {m_list}, reg=logit "
                  f"(q-sweep skipped: pinned selection overrides B's q)",
                  flush=True)
        else:
            q_list = [round(i * 0.1, 1) for i in range(11)]
            print(f"\n[auto-tune] sweeping q ∈ {q_list}, nc ∈ {nc_range}, "
                  f"m ∈ {m_list}, reg=logit", flush=True)

        unique_M_L = set(); unique_M_G = set()
        for q_val in q_list:
            cL, cG = _split_count(count, q_val)
            if cL > 0: unique_M_L.add(cL)
            if cG > 0: unique_M_G.add(cG)
        unique_M_L, unique_M_G = sorted(unique_M_L), sorted(unique_M_G)

        H_cache_qm, D_cache_qm = {}, {}
        for nc_ in nc_range:
            for M in unique_M_L:
                print(f"  • H_logit(nc={nc_}, M={M}) …", flush=True)
                H_cache_qm[(nc_, M)] = _lomo_pair_dispatch(
                    "H", "logit", X_v, Y, Vt_full, nc_, P_target, M, B, seed,
                    pinned_sel_by_fold_t=pinned_H_lomo)
        for nc_ in nc_range:
            for M in unique_M_G:
                print(f"  • D-pt_logit(nc={nc_}, N={M}) …", flush=True)
                D_cache_qm[(nc_, M)] = _lomo_pair_dispatch(
                    "D-pt", "logit", X_v, Y, Vt_full, nc_, P_target, M, B, seed,
                    pinned_sel_by_fold_t=pinned_D_lomo)
        cfgs, preds, _ = _autotune_task_b_q(H_cache_qm, D_cache_qm, Y, target_names,
                                            nc_range, m_list, count, q_list)
        note_parts = ["auto-tuned, q fixed" if q is not None else "auto-tuned w/ q-sweep"]
        if pin_task_a_selection:
            note_parts.append("pin-A")
        note = "(" + ", ".join(note_parts) + ")"
        label = "TASK_B_PINNED_A" if pin_task_a_selection else "TASK_B"
        _print_autotune_config(label, count, cfgs, extra_note=note)
    else:
        # Per-target counts (cfg["q"] honored when --q is None)
        per_target_counts, uses_split = _resolve_per_target_counts(cfgs, count, q)
        # Group base predictors needed: (strategy, reg, nc, M) → list of targets
        base_keys_H, base_keys_D = set(), set()
        for tname, cfg in cfgs.items():
            cL, cG = per_target_counts[tname]
            if cfg["m"] > 0 and cL > 0:
                base_keys_H.add((cfg["reg"], cfg["nc_h"], cL))
            if cfg["m"] < 1 and cG > 0:
                base_keys_D.add((cfg["reg"], cfg["nc_d"], cG))

        if uses_split or q is not None:
            print(f"\nPer-target counts: " +
                  ", ".join(f"{t}=(L={cL},G={cG})" for t, (cL, cG) in per_target_counts.items()),
                  flush=True)
        print(f"\nBase predictors needed:  H {sorted(base_keys_H)},  D-pt {sorted(base_keys_D)}",
              flush=True)

        H_cache, D_cache = {}, {}
        for reg, nc_, M in sorted(base_keys_H):
            print(f"  • H_{reg}(nc={nc_}, M={M}) …", flush=True)
            H_cache[(reg, nc_, M)] = _lomo_pair_dispatch("H", reg, X_v, Y, Vt_full,
                                                         nc_, P_target, M, B, seed,
                                                         pinned_sel_by_fold_t=pinned_H_lomo)
        for reg, nc_, M in sorted(base_keys_D):
            print(f"  • D-pt_{reg}(nc={nc_}, N={M}) …", flush=True)
            D_cache[(reg, nc_, M)] = _lomo_pair_dispatch("D-pt", reg, X_v, Y, Vt_full,
                                                         nc_, P_target, M, B, seed,
                                                         pinned_sel_by_fold_t=pinned_D_lomo)

        idx = {t: i for i, t in enumerate(target_names)}
        ref_arr = next(iter(H_cache.values())) if H_cache else next(iter(D_cache.values()))
        preds = np.zeros_like(ref_arr)
        for tname, cfg in cfgs.items():
            t_idx = idx[tname]
            cL, cG = per_target_counts[tname]
            m, reg = cfg["m"], cfg["reg"]
            h_part = m * H_cache[(reg, cfg["nc_h"], cL)][:, :, t_idx] if (m > 0 and cL > 0) else 0
            d_part = (1 - m) * D_cache[(reg, cfg["nc_d"], cG)][:, :, t_idx] if (m < 1 and cG > 0) else 0
            preds[:, :, t_idx] = h_part + d_part

    df = compute_pair_accuracy_df(preds, Y, target_names)

    print("\n" + "═" * 80, flush=True)
    print(f"Task B (pairwise) — per-target config at C={count}", flush=True)
    for tname, cfg in cfgs.items():
        print(f"  {tname:24s} → {_describe_pair(cfg)}", flush=True)
    print("═" * 80, flush=True)
    print_pair_results(df, label=f"Task B  C={count}, B={B}, seed={seed}")

    # Platt calibration: fit per target on all 728 decisions
    ml = collect_margins_labels(preds, Y, target_names)
    platt = {}
    for tname, (m_arr, l_arr) in ml.items():
        platt[tname] = fit_platt(m_arr, l_arr)

    out_csv = out_dir / "pair_predictions.csv"
    rows = []
    for k in range(len(MODELS)):
        for j in range(len(MODELS)):
            if j == k: continue
            for t_idx, tname in enumerate(target_names):
                a_pl, b_pl = platt[tname]
                true_margin = float(Y[k, t_idx] - Y[j, t_idx])
                m_pred = float(preds[k, j, t_idx])
                prob_k = float(apply_platt(np.array([m_pred]), a_pl, b_pl)[0])
                rows.append({
                    "held_out":               MODELS[k],
                    "other":                  MODELS[j],
                    "target":                 tname,
                    "true_margin":            true_margin,
                    "true_winner":            MODELS[k] if true_margin > 0 else
                                              (MODELS[j] if true_margin < 0 else "tie"),
                    "pred_margin":            m_pred,
                    "pred_prob_heldout_wins": prob_k,
                    "pred_winner":            MODELS[k] if m_pred > 0 else MODELS[j],
                })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nPer-pair predictions saved → {out_csv}", flush=True)

    print("\nGlobal Platt parameters  σ(a · logit + b):", flush=True)
    for tname, (a, b) in platt.items():
        print(f"  {tname:24s}  a={a:8.3f}  b={b:+8.3f}", flush=True)

    out_png = out_dir / "pair_loocv_scatter.png"
    plot_pair_scatter(preds, Y, target_names, out_path=str(out_png),
                      title=f"Task B LOMO: Predicted vs True Margin  (C={COUNT}, B={B})")

    if dump_selections:
        if strict_budget and pin_task_a_selection:
            # Strict-budget pin-A: dump the FULL pinned union (h_idx + d_idx)
            # regardless of pair's m, with pair-side weights from logit/ols fit.
            print(f"\nDumping strict-budget pin-A LOMO selections + weights "
                  f"(|H ∪ D| = {count} per fold-target, matches abs --strict-budget) …",
                  flush=True)
            rows = []
            for r in collect_lomo_weights_pair(
                    X_v, Y, Vt_full, target_names, cfgs,
                    P_target, B, seed, count, "H",
                    pinned_sel_by_fold_t=pinned_H_lomo):
                r["fold"] = MODELS[r.pop("held_out_idx")]
                rows.append(r)
            for r in collect_lomo_weights_pair(
                    X_v, Y, Vt_full, target_names, cfgs,
                    P_target, B, seed, count, "D-pt",
                    pinned_sel_by_fold_t=pinned_D_lomo):
                r["fold"] = MODELS[r.pop("held_out_idx")]
                rows.append(r)
            _write_selection_rows(rows, source_col_info_v, dump_selections,
                                  task="B", mode="LOMO-pinA", count=count)
        else:
            print(f"\nCollecting LOMO selections …", flush=True)
            counts_H, counts_G = _per_target_counts_for_dump(cfgs, count, q)
            rows = []
            if pin_task_a_selection:
                # Dump the A-pinned selections actually used by the regression.
                for strat in ("H", "D-pt"):
                    pinned = pinned_H_lomo if strat == "H" else pinned_D_lomo
                    cnt = counts_H if strat == "H" else counts_G
                    sel_rows = collect_lomo_weights_pair(
                        X_v, Y, Vt_full, target_names, cfgs,
                        P_target, B, seed, cnt, strat,
                        pinned_sel_by_fold_t=pinned)
                    for r in sel_rows:
                        r["fold"] = MODELS[r.pop("held_out_idx")]
                        rows.append(r)
            else:
                for strat in ("H", "D-pt"):
                    cnt = counts_H if strat == "H" else counts_G
                    sel_rows = collect_lomo_weights_pair(
                        X_v, Y, Vt_full, target_names, cfgs,
                        P_target, B, seed, cnt, strat)
                    for r in sel_rows:
                        r["fold"] = MODELS[r.pop("held_out_idx")]
                        rows.append(r)
            _write_selection_rows(rows, source_col_info_v, dump_selections,
                                  task="B", mode="LOMO" + ("-pinA" if pin_task_a_selection else ""),
                                  count=count)


# ──────────────────────────────────────────────────────────────
# Task A in-sample fit (all 14 models, no held-out) — analysis only
# ──────────────────────────────────────────────────────────────

def run_fit_abs(B, seed, sources=None, targets=None, count=COUNT,
                override_m=None, override_nc_h=None, override_nc_d=None,
                dump_selections=None, q=None, strict_budget=False):
    count_L, count_G = _split_count(count, q)
    override_m = _clamp_m_for_empty_split(count_L, count_G, override_m)
    if strict_budget:
        print(f"Task A  IN-SAMPLE fit config: C={count} STRICT-BUDGET "
              f"(|H ∪ D| = {count} per target), pooled B={B}, seed={seed}",
              flush=True)
    elif q is None:
        print(f"Task A  IN-SAMPLE fit config: C={count} (per-strategy), "
              f"pooled B={B}, seed={seed}", flush=True)
    else:
        print(f"Task A  IN-SAMPLE fit config: C_total={count}, q={q} → "
              f"Local={count_L} / Global={count_G}, pooled B={B}, seed={seed}",
              flush=True)
    print("  (all n models used both for fit and prediction — NOT out-of-sample)\n",
          flush=True)
    X_v, Y, target_names, P_target, Vt_full, source_col_info_v = _load_and_svd(sources, targets)

    cfgs = _resolve_configs(get_task_a_config(count), target_names,
                            override_m, override_nc_h, override_nc_d,
                            fallback_m=0.5, fallback_nc_h=2, fallback_nc_d=7)

    # ── Strict-budget branch (single joint-selection fit per target) ──
    if strict_budget:
        per_target_qC, nc_h_per_t, nc_d_per_t = {}, {}, {}
        for t_idx, tname in enumerate(target_names):
            cfg = cfgs[tname]
            q_t = q if q is not None else cfg.get("q")
            if q_t is None: q_t = 0.5
            per_target_qC[t_idx] = (float(q_t), int(count))
            nc_h_per_t[t_idx] = int(cfg["nc_h"])
            nc_d_per_t[t_idx] = int(cfg["nc_d"])
        print(f"[strict-budget] Per-target (q, C, nc_h, nc_d, m):", flush=True)
        for t_idx, tname in enumerate(target_names):
            q_t, _ = per_target_qC[t_idx]
            cfg = cfgs[tname]
            print(f"  {tname:24s}  q={q_t:.2f}  C={count}  nc_h={cfg['nc_h']}  "
                  f"nc_d={cfg['nc_d']}  m={cfg['m']}", flush=True)
        print(f"\nRunning joint fit-all (|H ∪ D| = {count} per target) …", flush=True)
        preds_H, preds_D, joint_sels = fit_all_joint_abs(
            X_v, Y, Vt_full, per_target_qC, nc_h_per_t, nc_d_per_t,
            P_target, B, seed)
        preds = np.zeros_like(preds_H)
        for tname, cfg in cfgs.items():
            t_idx = target_names.index(tname)
            m = cfg["m"]
            preds[:, t_idx] = m * preds_H[:, t_idx] + (1 - m) * preds_D[:, t_idx]

        df = build_abs_metrics_df(preds, Y, target_names)
        print("\n" + "═" * 75, flush=True)
        print(f"Task A IN-SAMPLE STRICT-BUDGET fit  — per-target config at C={count}",
              flush=True)
        for tname, cfg in cfgs.items():
            print(f"  {tname:24s} → {_describe(cfg)}", flush=True)
        print("═" * 75, flush=True)
        print_abs_results(df, label=f"Task A IN-SAMPLE STRICT  C={count}, B={B}, seed={seed}")

        out_dir = Path(BASE_DIR).parent.parent
        out_csv = out_dir / "abs_fit_all_predictions.csv"
        rows = []
        for i, mname in enumerate(MODELS):
            for t_idx, tname in enumerate(target_names):
                rows.append({"model": mname, "target": tname,
                    "predicted": float(preds[i, t_idx]),
                    "actual":    float(Y[i, t_idx]),
                    "residual":  float(preds[i, t_idx] - Y[i, t_idx]),
                    "abs_error": float(abs(preds[i, t_idx] - Y[i, t_idx])),
                })
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nPer-model in-sample predictions saved → {out_csv}", flush=True)

        if dump_selections:
            print(f"\nDumping strict-budget fit-all selections + weights "
                  f"(|H ∪ D| = {count} per target) …", flush=True)
            # Build {t_idx → h_idx} and {t_idx → d_idx} from joint_sels for collectors
            h_pinned = {t_idx: hi for t_idx, (hi, _) in enumerate(joint_sels)
                        if len(hi) > 0}
            d_pinned = {t_idx: di for t_idx, (_, di) in enumerate(joint_sels)
                        if len(di) > 0}
            sel_rows = []
            for r in collect_fit_all_weights_abs(X_v, Y, Vt_full, target_names,
                                                  cfgs, P_target, B, seed,
                                                  count, "H",
                                                  pinned_sel_by_t=h_pinned):
                r["fold"] = "ALL"
                r.pop("held_out_idx", None)
                sel_rows.append(r)
            for r in collect_fit_all_weights_abs(X_v, Y, Vt_full, target_names,
                                                  cfgs, P_target, B, seed,
                                                  count, "D-pt",
                                                  pinned_sel_by_t=d_pinned):
                r["fold"] = "ALL"
                r.pop("held_out_idx", None)
                sel_rows.append(r)
            _write_selection_rows(sel_rows, source_col_info_v, dump_selections,
                                  task="A", mode="FIT-ALL", count=count)
        return

    per_target_counts, uses_split = _resolve_per_target_counts(cfgs, count, q)
    nc_h_set = sorted({c["nc_h"] for c in cfgs.values() if c["m"] > 0})
    nc_d_set = sorted({c["nc_d"] for c in cfgs.values() if c["m"] < 1})
    unique_M_L = sorted({cL for tname, (cL, _) in per_target_counts.items()
                         if cL > 0 and cfgs[tname]["m"] > 0})
    unique_M_G = sorted({cG for tname, (_, cG) in per_target_counts.items()
                         if cG > 0 and cfgs[tname]["m"] < 1})
    if uses_split or q is not None:
        print(f"Per-target counts: " +
              ", ".join(f"{t}=(L={cL},G={cG})" for t, (cL, cG) in per_target_counts.items()),
              flush=True)

    H_cache = {}
    for nc_ in nc_h_set:
        for M in unique_M_L:
            print(f"  • H(nc={nc_}, M={M})  all-n fit …", flush=True)
            H_cache[(nc_, M)] = fit_all_H_abs(X_v, Y, P_target, nc_, M, B, seed)
    D_cache = {}
    for nc_ in nc_d_set:
        for M in unique_M_G:
            print(f"  • D-pt(nc={nc_}, N={M})  all-n fit …", flush=True)
            D_cache[(nc_, M)] = fit_all_Dpt_abs(X_v, Y, Vt_full, nc_, M, P_target, B, seed)

    idx = {t: i for i, t in enumerate(target_names)}
    ref_arr = next(iter(H_cache.values())) if H_cache else next(iter(D_cache.values()))
    preds = np.zeros_like(ref_arr)
    for tname, cfg in cfgs.items():
        t_idx = idx[tname]
        cL, cG = per_target_counts[tname]
        m = cfg["m"]
        h_part = m * H_cache[(cfg["nc_h"], cL)][:, t_idx] if (m > 0 and cL > 0) else 0
        d_part = (1 - m) * D_cache[(cfg["nc_d"], cG)][:, t_idx] if (m < 1 and cG > 0) else 0
        preds[:, t_idx] = h_part + d_part

    df = build_abs_metrics_df(preds, Y, target_names)

    print("\n" + "═" * 75, flush=True)
    print(f"Task A IN-SAMPLE fit (all n models)  — per-target config at C={count}",
          flush=True)
    for tname, cfg in cfgs.items():
        print(f"  {tname:24s} → {_describe(cfg)}", flush=True)
    print("═" * 75, flush=True)
    print_abs_results(df, label=f"Task A  IN-SAMPLE  C={count}, B={B}, seed={seed}")

    out_dir = Path(BASE_DIR).parent.parent
    out_csv = out_dir / "abs_fit_all_predictions.csv"
    rows = []
    for i, m in enumerate(MODELS):
        for t_idx, tname in enumerate(target_names):
            rows.append({
                "model":     m,
                "target":    tname,
                "predicted": float(preds[i, t_idx]),
                "actual":    float(Y[i, t_idx]),
                "residual":  float(preds[i, t_idx] - Y[i, t_idx]),
                "abs_error": float(abs(preds[i, t_idx] - Y[i, t_idx])),
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nPer-model in-sample predictions saved → {out_csv}", flush=True)

    # Per-model summary
    by_model = pd.DataFrame(rows).groupby("model")["abs_error"].mean().sort_values()
    print("\nPer-model mean |residual| across 4 targets (in-sample fit):", flush=True)
    for m, v in by_model.items():
        print(f"  {m:40s}  {v*100:5.2f}%", flush=True)

    if dump_selections:
        print(f"\nCollecting fit-all selections + weights …", flush=True)
        counts_H, counts_G = _per_target_counts_for_dump(cfgs, count, q)
        sel_rows = []
        for strat in ("H", "D-pt"):
            cnt = counts_H if strat == "H" else counts_G
            if not any(cnt.values()): continue
            for r in collect_fit_all_weights_abs(X_v, Y, Vt_full, target_names,
                                                  cfgs, P_target, B, seed, cnt, strat):
                r["fold"] = "ALL"
                r.pop("held_out_idx", None)
                sel_rows.append(r)
        _write_selection_rows(sel_rows, source_col_info_v, dump_selections,
                              task="A", mode="FIT-ALL", count=count)


# ──────────────────────────────────────────────────────────────
# Task B in-sample fit (all 14 models, no held-out) — analysis only
# ──────────────────────────────────────────────────────────────

def _fit_all_pair_dispatch(strategy, reg, X_v, Y, Vt_full, nc, P_target, count, B, seed,
                           pinned_sel_by_t=None):
    if strategy == "H":
        if reg == "ols":
            return fit_all_H_pair_ols(X_v, Y, P_target, nc, count, B, seed,
                                      pinned_sel_by_t=pinned_sel_by_t)
        return fit_all_H_logit(X_v, Y, P_target, nc, count, B, seed, DEFAULT_ALPHA,
                               pinned_sel_by_t=pinned_sel_by_t)
    else:
        if reg == "ols":
            return fit_all_Dpt_pair_ols(X_v, Y, Vt_full, nc, count, P_target, B, seed,
                                        pinned_sel_by_t=pinned_sel_by_t)
        return fit_all_Dpt_logit(X_v, Y, Vt_full, nc, count, P_target, B, seed, DEFAULT_ALPHA,
                                 pinned_sel_by_t=pinned_sel_by_t)


def run_fit_pair(B, seed, sources=None, targets=None, count=COUNT,
                 override_m=None, override_nc_h=None, override_nc_d=None,
                 dump_selections=None, pin_task_a_selection=False, q=None,
                 strict_budget=False):
    count_L, count_G = _split_count(count, q)
    override_m = _clamp_m_for_empty_split(count_L, count_G, override_m)
    if strict_budget:
        if not pin_task_a_selection:
            raise ValueError("--strict-budget on `fit-pair` requires "
                             "--pin-task-a-selection (selection comes from Task A's "
                             "joint top-C; pair-only joint fit is not implemented).")
        print(f"Task B  IN-SAMPLE fit config: C={count} STRICT-BUDGET "
              f"(|H ∪ D| = {count} per target, selection pinned to abs --strict-budget), "
              f"pooled B={B}, seed={seed}", flush=True)
    elif q is None:
        print(f"Task B  IN-SAMPLE fit config: C={count} (per-strategy), "
              f"pooled B={B}, seed={seed}", flush=True)
    else:
        print(f"Task B  IN-SAMPLE fit config: C_total={count}, q={q} → "
              f"Local={count_L} / Global={count_G}, pooled B={B}, seed={seed}",
              flush=True)
    print("  (all n models used both for fit and prediction — NOT out-of-sample)\n",
          flush=True)
    X_v, Y, target_names, P_target, Vt_full, source_col_info_v = _load_and_svd(sources, targets)

    cfg_source = (get_task_b_config_pinned_a(count)
                  if pin_task_a_selection else get_task_b_config(count))
    cfgs = _resolve_configs(cfg_source, target_names,
                            override_m, override_nc_h, override_nc_d,
                            fallback_m=0.5, fallback_nc_h=2, fallback_nc_d=6,
                            fallback_reg="ols")
    per_target_counts, uses_split = _resolve_per_target_counts(cfgs, count, q)
    if uses_split or q is not None:
        print(f"Per-target counts: " +
              ", ".join(f"{t}=(L={cL},G={cG})" for t, (cL, cG) in per_target_counts.items()),
              flush=True)
    base_keys_H, base_keys_D = set(), set()
    for tname, cfg in cfgs.items():
        cL, cG = per_target_counts[tname]
        if cfg["m"] > 0 and cL > 0:
            base_keys_H.add((cfg["reg"], cfg["nc_h"], cL))
        if cfg["m"] < 1 and cG > 0:
            base_keys_D.add((cfg["reg"], cfg["nc_d"], cG))

    pinned_H_all = pinned_D_all = None
    pin_a_cfgs_used = None
    if pin_task_a_selection:
        print("\n[--pin-task-a-selection] Pinning Task B's selection to Task A's "
              "col indices (all-n fit).", flush=True)
        if strict_budget:
            pinned_H_all, pinned_D_all, pin_a_cfgs_used = (
                _compute_task_a_selections_fit_all_strict(
                    X_v, Y, Vt_full, target_names, count, q=q))
        else:
            pinned_H_all, pinned_D_all = _compute_task_a_selections_fit_all(
                X_v, Y, Vt_full, target_names, count)

    H_cache, D_cache = {}, {}
    for reg, nc_, M in sorted(base_keys_H):
        print(f"  • H_{reg}(nc={nc_}, M={M})  all-n fit …", flush=True)
        H_cache[(reg, nc_, M)] = _fit_all_pair_dispatch("H", reg, X_v, Y, Vt_full,
                                                        nc_, P_target, M, B, seed,
                                                        pinned_sel_by_t=pinned_H_all)
    for reg, nc_, M in sorted(base_keys_D):
        print(f"  • D-pt_{reg}(nc={nc_}, N={M})  all-n fit …", flush=True)
        D_cache[(reg, nc_, M)] = _fit_all_pair_dispatch("D-pt", reg, X_v, Y, Vt_full,
                                                        nc_, P_target, M, B, seed,
                                                        pinned_sel_by_t=pinned_D_all)

    idx = {t: i for i, t in enumerate(target_names)}
    ref_arr = next(iter(H_cache.values())) if H_cache else next(iter(D_cache.values()))
    preds = np.zeros_like(ref_arr)
    for tname, cfg in cfgs.items():
        t_idx = idx[tname]
        cL, cG = per_target_counts[tname]
        m, reg = cfg["m"], cfg["reg"]
        h_part = m * H_cache[(reg, cfg["nc_h"], cL)][:, :, t_idx] if (m > 0 and cL > 0) else 0
        d_part = (1 - m) * D_cache[(reg, cfg["nc_d"], cG)][:, :, t_idx] if (m < 1 and cG > 0) else 0
        preds[:, :, t_idx] = h_part + d_part

    df = compute_pair_accuracy_df(preds, Y, target_names)

    print("\n" + "═" * 80, flush=True)
    print(f"Task B IN-SAMPLE fit (all n models) — per-target config at C={count}",
          flush=True)
    for tname, cfg in cfgs.items():
        print(f"  {tname:24s} → {_describe_pair(cfg)}", flush=True)
    print("═" * 80, flush=True)
    print_pair_results(df, label=f"Task B  IN-SAMPLE  C={count}, B={B}, seed={seed}")

    # Per-model summary: average accuracy of pairs involving each model
    n = preds.shape[0]
    rows = []
    for t_idx, tname in enumerate(target_names):
        true_mat = Y[:, t_idx:t_idx+1] - Y[:, t_idx:t_idx+1].T
        pred_mat = preds[:, :, t_idx]
        for k in range(n):
            for j in range(n):
                if k == j: continue
                y_m = float(true_mat[k, j])
                if abs(y_m) < 1e-9: continue
                rows.append({
                    "held_out":          MODELS[k],
                    "other":             MODELS[j],
                    "target":            tname,
                    "true_margin":       y_m,
                    "pred_margin":       float(pred_mat[k, j]),
                    "correct":           bool((pred_mat[k, j] > 0) == (y_m > 0)),
                })
    df_pairs = pd.DataFrame(rows)
    out_csv = Path(BASE_DIR).parent.parent / "pair_fit_all_predictions.csv"
    df_pairs.to_csv(out_csv, index=False)
    print(f"\nPer-pair in-sample predictions saved → {out_csv}", flush=True)

    # Per-model: accuracy of all pairs involving this model
    by_model = df_pairs.groupby("held_out")["correct"].mean().sort_values(ascending=False)
    print("\nPer-model pair accuracy (in-sample, averaged over all pairs involving each model):",
          flush=True)
    for m, v in by_model.items():
        print(f"  {m:40s}  {v*100:5.1f}%", flush=True)

    if dump_selections:
        if strict_budget and pin_task_a_selection:
            # Dump full pinned union + pair-side weights (matches abs --strict-budget cost)
            print(f"\nDumping strict-budget fit-pair pin-A selections + weights "
                  f"(|H ∪ D| = {count} per target) …", flush=True)
            sel_rows = []
            for r in collect_fit_all_weights_pair(
                    X_v, Y, Vt_full, target_names, cfgs,
                    P_target, B, seed, count, "H",
                    pinned_sel_by_t=pinned_H_all):
                r["fold"] = "ALL"
                r.pop("held_out_idx", None)
                sel_rows.append(r)
            for r in collect_fit_all_weights_pair(
                    X_v, Y, Vt_full, target_names, cfgs,
                    P_target, B, seed, count, "D-pt",
                    pinned_sel_by_t=pinned_D_all):
                r["fold"] = "ALL"
                r.pop("held_out_idx", None)
                sel_rows.append(r)
            _write_selection_rows(sel_rows, source_col_info_v, dump_selections,
                                  task="B", mode="FIT-ALL-pinA", count=count)
        else:
            print(f"\nCollecting fit-all selections + weights …", flush=True)
            counts_H, counts_G = _per_target_counts_for_dump(cfgs, count, q)
            sel_rows = []
            if pin_task_a_selection:
                for strat in ("H", "D-pt"):
                    cnt = counts_H if strat == "H" else counts_G
                    if not any(cnt.values()): continue
                    pinned = pinned_H_all if strat == "H" else pinned_D_all
                    for r in collect_fit_all_weights_pair(
                            X_v, Y, Vt_full, target_names, cfgs,
                            P_target, B, seed, cnt, strat,
                            pinned_sel_by_t=pinned):
                        r["fold"] = "ALL"
                        r.pop("held_out_idx", None)
                        sel_rows.append(r)
            else:
                for strat in ("H", "D-pt"):
                    cnt = counts_H if strat == "H" else counts_G
                    if not any(cnt.values()): continue
                    for r in collect_fit_all_weights_pair(
                            X_v, Y, Vt_full, target_names, cfgs,
                            P_target, B, seed, cnt, strat):
                        r["fold"] = "ALL"
                        r.pop("held_out_idx", None)
                        sel_rows.append(r)
            _write_selection_rows(sel_rows, source_col_info_v, dump_selections,
                                  task="B", mode="FIT-ALL" + ("-pinA" if pin_task_a_selection else ""),
                                  count=count)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pacebench LOMO predictor (Task A: absolute, Task B: pairwise).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", choices=["abs", "pair", "fit-abs", "fit-pair"],
                        help="Task:\n"
                             "  abs       — Task A LOMO (main results)\n"
                             "  pair      — Task B LOMO (main results)\n"
                             "  fit-abs   — Task A in-sample fit on all models (analysis only)\n"
                             "  fit-pair  — Task B in-sample fit on all models (analysis only)")
    parser.add_argument("--B", type=int, default=DEFAULT_B,
                        help=f"Pooled target-instance bootstrap count (default {DEFAULT_B})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default {DEFAULT_SEED})")
    parser.add_argument("--count", type=int, default=COUNT,
                        help=f"Source-instance count per target (default {COUNT})")
    parser.add_argument("--sources", nargs="+", default=None,
                        metavar="BENCH",
                        help="Restrict source benchmarks (default: all non-target, "
                             "non-excluded). Example: --sources mmmu humaneval")
    parser.add_argument("--targets", nargs="+", default=None,
                        metavar="BENCH",
                        help="Restrict target benchmarks (default: gaia swebench "
                             "swebench_multimodal swtbench). Example: --targets gaia")
    parser.add_argument("--m", type=float, default=None,
                        help="Override H-weight m ∈ [0, 1] for ALL targets "
                             "(default: use TASK_A_CONFIG / TASK_B_CONFIG per target)")
    parser.add_argument("--nc-h", type=int, default=None, dest="nc_h",
                        help="Override H embedding rank nc_h for ALL targets")
    parser.add_argument("--nc-d", type=int, default=None, dest="nc_d",
                        help="Override D-pt embedding rank nc_d for ALL targets")
    parser.add_argument("--train-models", nargs="+", default=None, metavar="MODEL",
                        help="Custom train-model subset (abs/pair only). If only this "
                             "is given, eval = complement. Example: --train-models "
                             "Claude-4.5-Opus GPT-5.2 …")
    parser.add_argument("--eval-models", nargs="+", default=None, metavar="MODEL",
                        help="Custom eval-model subset (abs/pair only). For pair: "
                             "(k, j) is predicted iff k or j is in this set.")
    parser.add_argument("--dump-selections", default="selections", metavar="DIR",
                        help="Base directory for selection dumps. Layout (filename "
                             "encodes --count as C<N>): "
                             "LOMO → <DIR>/<task>_loocv[_pinA]/<model>/selections_C<N>.csv "
                             "(one CSV per held-out model); FIT-ALL → "
                             "<DIR>/<task>_fit[_pinA]/selections_C<N>.csv; SPLIT → "
                             "<DIR>/<task>_split/selections_C<N>.csv. A trailing '.csv' on "
                             "the argument is silently stripped (legacy compat).")
    parser.add_argument("--pin-task-a-selection", action="store_true",
                        help="(pair / fit-pair only) Force Task B to train on "
                             "exactly the instances Task A would select per target "
                             "(using TASK_A_CONFIG's nc_d). Task B keeps its own "
                             "m, nc, and reg for the predictor.")
    parser.add_argument("--auto-tune", action="store_true",
                        help="Sweep m × nc_h × nc_d to find per-target best config, "
                             "then use it. Prints a paste-ready dict. Incompatible "
                             "with --train-models / --eval-models (eval leak).")
    parser.add_argument("--q", type=float, default=None,
                        help="Joint budget split: when given, Local-SVD uses round(q*C) "
                             "instances and Global-SVD uses C-round(q*C) (joint total = C). "
                             "Default (omitted): each strategy independently uses C "
                             "(effective budget up to 2*C). Must be in [0, 1].")
    parser.add_argument("--strict-budget", action="store_true",
                        help="Strict joint budget: |H ∪ D| = C per fold-target. "
                             "Initial allocation = (round(q*C) from H, C-round(q*C) "
                             "from D-pt); when overlap shrinks the union below C, "
                             "greedily extend each side's selection by next-best "
                             "non-overlapping instances, q-weighted. Compatible with "
                             "--auto-tune on `abs` only (sweeps q × nc_h × nc_d × m, "
                             "~6× slower). Supported on all 4 tasks; for `pair` and "
                             "`fit-pair` requires --pin-task-a-selection (selection "
                             "is pinned to abs's joint top-C).")
    parser.add_argument("--nc-max", type=int, default=10, dest="nc_max_at",
                        help="Upper bound for nc_h / nc_d when --auto-tune (default 10)")
    parser.add_argument("--m-grid", type=str,
                        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                        dest="m_grid_str",
                        help="Comma-separated m values to sweep when --auto-tune")
    args = parser.parse_args()

    kwargs = dict(
        B=args.B, seed=args.seed,
        sources=args.sources, targets=args.targets, count=args.count,
        override_m=args.m, override_nc_h=args.nc_h, override_nc_d=args.nc_d,
    )
    if args.task in ("abs", "pair"):
        kwargs["train_models"]    = args.train_models
        kwargs["eval_models"]     = args.eval_models
        kwargs["dump_selections"] = args.dump_selections
    else:  # fit-abs / fit-pair: dump-selections supported, train/eval not applicable
        kwargs["dump_selections"] = args.dump_selections
        if args.train_models or args.eval_models:
            print("(warning: --train-models / --eval-models are only applied to "
                  "`abs` and `pair`; ignored for fit-abs / fit-pair)", flush=True)
    if args.task in ("pair", "fit-pair"):
        kwargs["pin_task_a_selection"] = args.pin_task_a_selection
    elif args.pin_task_a_selection:
        print("(warning: --pin-task-a-selection only applies to pair / fit-pair; "
              "ignored)", flush=True)
    if args.task in ("abs", "pair"):  # auto-tune supported for LOMO paths
        kwargs["auto_tune"] = args.auto_tune
        kwargs["nc_max"]    = args.nc_max_at
        kwargs["m_grid"]    = args.m_grid_str
    elif args.auto_tune:
        print("(warning: --auto-tune not yet supported for fit-abs / fit-pair; "
              "ignored)", flush=True)
    kwargs["q"] = args.q   # supported by abs / pair / fit-abs / fit-pair
    kwargs["strict_budget"] = args.strict_budget   # all 4 tasks now support it
    {
        "abs":      run_abs,
        "pair":     run_pair,
        "fit-abs":  run_fit_abs,
        "fit-pair": run_fit_pair,
    }[args.task](**kwargs)


if __name__ == "__main__":
    main()
