"""Shared setup for predicting a NEW model from the Pace-Bench selection.

A new model is predicted by a regression fit on the calibration models only
(``config.MODELS``), with the frozen per-target configs. Everything that decides
*which* source instances are used -- the zero-variance column filter, the
Global-SVD basis and the per-target joint selection -- is computed on the
calibration models alone, so the new model's own scores never change the
selection. That selection must be exactly the shipped fit dump
(``selections/abs_fit/selections_C<count>.csv``), i.e. the instances
``evaluations/score_new_model.py`` scores; this is checked at run time.

The new model is appended as the last row, so the fold workers in
``pacebench.regression`` (which train on every row except the held-out one)
fit on the calibration models and project the new model.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.linalg import svd as np_svd

from pacebench import config
from pacebench.data import build_data, filter_valid_columns
from pacebench.selection import joint_top_C

SELECTIONS_DIR = Path(__file__).resolve().parents[1] / "selections" / "abs_fit"
# Largest value in the calibration source matrix is ~1.06 (a LIFBench score).
SCALE_MAX = 1.1


def _csv_ids(path, numeric_only=False):
    """Ids in a standardized CSV (optionally only rows with a finite numeric score)."""
    ids = set()
    if path.exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                if numeric_only:
                    try:
                        if not math.isfinite(float(r["score"])):
                            continue
                    except (TypeError, ValueError):
                        continue
                ids.add(str(r["id"]))
    return ids


def targets_with_ground_truth(model):
    """Agentic targets for which `model` has a score CSV under BASE_DIR."""
    base = Path(config.BASE_DIR)
    return {t for t in config.TARGET_DATASETS if any((base / t).rglob(f"{model}.csv"))}


def prepare(new_model):
    """Load the calibration models plus `new_model` (appended as the last row).

    Returns a dict with
      X_v        (n_cal + 1, |S_v|) source matrix on the calibration-valid columns
      Y          (n_cal + 1, T) target means; the new model's row is its ground
                 truth where available, else 0 (never used for fitting)
      P_target   list of (n_cal + 1, n_inst_t) target-instance matrices (new row = 0)
      Vt         Global-SVD basis of the calibration rows of X_v
      col_info   (benchmark, subdir, instance_id) per column of X_v
      target_names, cal_models, new_idx, gt_targets
    """
    cal_models = list(config.MODELS)
    # Compared case-insensitively: on case-insensitive filesystems 'gpt-5.2.csv'
    # would read GPT-5.2's own CSVs.
    clash = [m for m in cal_models if m.casefold() == new_model.casefold()]
    if clash:
        raise SystemExit(f"'{new_model}' matches calibration model '{clash[0]}'; "
                         "score the new model under a different --model-name-out.")

    # Calibration models alone: selection-relevant quantities (filter, SVD) and the
    # training labels come only from here.
    X_c, Y_c, target_names, P_c, info_c = build_data(models=cal_models, verbose=False)
    X_cv, mask = filter_valid_columns(X_c, return_mask=True)
    col_info = [c for c, keep in zip(info_c, mask) if keep]

    # The new model's scores, read on exactly those columns (absent instances -> 0).
    X_f, Y_f, target_names_f, _, info_f = build_data(
        models=cal_models + [new_model], verbose=False)
    if list(target_names_f) != list(target_names):
        raise RuntimeError("target order differs between data loads")
    pos = {c: j for j, c in enumerate(info_f)}
    new_idx = len(cal_models)
    x_new = np.array([X_f[new_idx, pos[c]] if c in pos else 0.0 for c in col_info],
                     dtype=np.float64)

    gt_targets = targets_with_ground_truth(new_model)
    base = Path(config.BASE_DIR)
    for t_idx, t in enumerate(target_names):
        if t in gt_targets:
            n_have = len(set().union(*(_csv_ids(p, numeric_only=True)
                                       for p in (base / t).rglob(f"{new_model}.csv"))))
            n_all = P_c[t_idx].shape[1]
            if n_have < n_all:
                print(f"WARNING: {t}: ground truth covers {n_have}/{n_all} target "
                      "instances; missing ones count as 0 in `actual`.", flush=True)
    y_new = np.array([Y_f[new_idx, t] if t_name in gt_targets else 0.0
                      for t, t_name in enumerate(target_names)])

    X_v = np.vstack([X_cv, x_new[None, :]])
    Y = np.vstack([Y_c, y_new[None, :]])
    P_target = [np.vstack([P, np.zeros((1, P.shape[1]), dtype=P.dtype)]) for P in P_c]
    _, _, Vt = np_svd(X_cv, full_matrices=False)
    return dict(X_v=X_v, Y=Y, P_target=P_target, Vt=Vt, col_info=col_info,
                target_names=list(target_names), cal_models=cal_models,
                new_idx=new_idx, gt_targets=gt_targets)


def task_a_configs(target_names, count):
    """Per-target Task A configs, resolved exactly as `cli.py abs` does."""
    import cli
    return cli._resolve_configs(config.get_task_a_config(count), target_names,
                                None, None, None, fallback_m=0.5, fallback_nc_h=2,
                                fallback_nc_d=7)


def plan_selection(d, count):
    """Per-target (h_idx, d_idx): the joint Local/Global selection fit on the
    calibration models with the frozen Task A config (= the shipped fit dump)."""
    cfgs = task_a_configs(d["target_names"], count)
    n_cal = d["new_idx"]
    X_cal = d["X_v"][:n_cal]
    sel = {}
    for t_idx, t in enumerate(d["target_names"]):
        cfg = cfgs[t]
        q = cfg.get("q") if cfg.get("q") is not None else 0.5
        sel[t_idx] = joint_top_C(X_cal, d["Y"][:n_cal, t_idx], d["Vt"], count,
                                 cfg["nc_d"], q)
    return sel


def _selected_keys(d, sel, t_idx):
    h_idx, d_idx = sel[t_idx]
    cols = sorted(set(h_idx.tolist()) | set(d_idx.tolist()))
    return [tuple(str(x) for x in d["col_info"][c]) for c in cols]


def check_against_shipped(d, sel, count):
    """Stop if the selection differs from the shipped selections_C<count>.csv: the
    new model was scored on the shipped instances, so a different selection would
    read unscored instances as 0."""
    path = SELECTIONS_DIR / f"selections_C{count}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found; generate it with "
                         f"`cli.py fit-abs --count {count} --B 300 --seed 42 --strict-budget "
                         "--dump-selections selections` and score the new model on it.")
    shipped = defaultdict(set)
    with open(path) as f:
        for r in csv.DictReader(f):
            shipped[r["target"]].add((r["benchmark"], r["subdir"], str(r["instance_id"])))
    bad = []
    for t_idx, t in enumerate(d["target_names"]):
        mine = set(_selected_keys(d, sel, t_idx))
        if mine != shipped[t]:
            bad.append(f"{t} ({len(mine & shipped[t])}/{len(shipped[t])} shared)")
    if bad:
        raise SystemExit(
            f"The selection implied by config.py and the calibration data differs from "
            f"{path.name} for: {', '.join(bad)}. Regenerate {path.name} with `cli.py "
            f"fit-abs --count {count} --B 300 --seed 42 --strict-budget --dump-selections "
            "selections` and re-score the new model on it.")


def coverage(d, sel, new_model):
    """Per target: (#selected instances with a numeric score for the new model, #selected)."""
    base = Path(config.BASE_DIR)
    scored = {}

    def ids_for(bench, subdir):
        key = (bench, subdir)
        if key not in scored:
            p = (base / bench / subdir if subdir else base / bench) / f"{new_model}.csv"
            scored[key] = _csv_ids(p, numeric_only=True)
        return scored[key]

    out = {}
    for t_idx, t in enumerate(d["target_names"]):
        keys = _selected_keys(d, sel, t_idx)
        have = sum(1 for (b, s, i) in keys if i in ids_for(b, s))
        out[t] = (have, len(keys))
    return out


def require_scores(cov, new_model):
    """Stop if the new model has no scores at all; warn about partial coverage."""
    if sum(h for h, _ in cov.values()) == 0:
        raise SystemExit(
            f"No scores found for '{new_model}' on any selected instance under "
            f"{config.BASE_DIR}. Check the name (it must match --model-name-out of "
            "evaluations/score_new_model.py).")
    partial = {t: hn for t, hn in cov.items() if hn[0] < hn[1]}
    if partial:
        print("WARNING: the new model is missing scores for some selected instances; "
              "they are read as 0:", flush=True)
        for t, (h, n) in partial.items():
            print(f"  {t:24s} {h}/{n} selected instances scored", flush=True)


def check_scale(d, sel):
    """Warn about selected new-model values outside the calibration scale (e.g. a
    metric written on 0-100 instead of 0-1)."""
    x = d["X_v"][d["new_idx"]]
    cols = sorted({c for h, dd in sel.values() for c in list(h) + list(dd)})
    bad = [d["col_info"][c] for c in cols if x[c] < 0 or x[c] > SCALE_MAX]
    if bad:
        print(f"WARNING: {len(bad)} selected score(s) of the new model are outside the "
              f"calibration range [0, {SCALE_MAX}], e.g. {bad[:3]}; check the "
              "standardized CSVs.", flush=True)
