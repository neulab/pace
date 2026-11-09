#!/usr/bin/env python3
"""Predict a NEW model's PAIRWISE win/loss vs the calibration models (Goal B).

Mirror of predict_new_model.py for Task B. The pairwise regressions are fit on the
calibration models only (config.MODELS) with the frozen per-target config
config.TASK_B_CONFIGS_PINNED_A[count], on the Goal A Pace-Bench selection
(= selections/abs_fit/selections_C<count>.csv). For each calibration model
`other` and each agentic target, the script reports the predicted margin, the
probability that the new model beats `other`, and the predicted winner.

P(win) is Platt-calibrated per target on a standard LOOCV `pair` run over the
calibration models only (saved under <out>/reference_without_new_model/).

Requirements
------------
The new model must already have per-instance score CSVs under BASE_DIR
(results/standardized_results/...), produced by evaluations/score_new_model.py.
Selected instances without a score are read as 0; per-target coverage is reported.

Outputs go to new_model_predictions/<NewModel>/ at the repo root (override with
$PROXYBENCH_OUT_DIR). `true_winner` is filled only for targets where the new model
has agentic ground truth.

Usage
-----
  python analysis/predict_new_model_pair.py --new-model My-New-Model
  python analysis/predict_new_model_pair.py --new-model My-New-Model --count 100 --B 300
  # add --summary for a per-target win-rate summary instead of every pair
"""
import os
# Same BLAS thread settings as cli.py, applied before numpy/pandas are imported:
# the thread count changes floating-point summation order.
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/pacebench/
sys.path.insert(0, str(Path(__file__).resolve().parent))      # analysis helpers

from pacebench import config  # noqa: E402
from pacebench.calibration import apply_platt, fit_platt  # noqa: E402
from pacebench.config import DEFAULT_ALPHA  # noqa: E402
from pacebench.stats import pair_indices  # noqa: E402
from pacebench.regression.pairwise import (  # noqa: E402
    _fold_H_logit, _fold_Dpt_logit, _fold_H_pair_ols, _fold_Dpt_pair_ols)
import _new_model as nm  # noqa: E402


def reference_platt(args, out_dir, log):
    """Per-target Platt (a, b) from a standard LOOCV `pair` run over the calibration
    models only (non-tied pairs, label = true margin > 0), as run_pair fits it."""
    import cli
    ref_dir = out_dir / "reference_without_new_model"
    prev = os.environ.get("PROXYBENCH_OUT_DIR")
    os.environ["PROXYBENCH_OUT_DIR"] = str(ref_dir)
    try:
        with contextlib.redirect_stdout(log):
            cli.run_pair(B=args.B, seed=args.seed, count=args.count,
                         strict_budget=True, auto_tune=False, pin_task_a_selection=True)
    finally:
        if prev is None:
            os.environ.pop("PROXYBENCH_OUT_DIR", None)
        else:
            os.environ["PROXYBENCH_OUT_DIR"] = prev
    ref = pd.read_csv(ref_dir / "pair_predictions.csv")
    ref = ref[(ref["true_margin"].abs() >= 1e-9) & np.isfinite(ref["pred_margin"])]
    return {t: fit_platt(g["pred_margin"].to_numpy(), (g["true_margin"] > 0).astype(int).to_numpy())
            for t, g in ref.groupby("target")}


def task_b_configs(target_names, count):
    """Per-target Task B (pinned-to-A) configs and (count_L, count_G), resolved
    exactly as `cli.py pair --pin-task-a-selection` does."""
    import cli
    cfgs = cli._resolve_configs(config.get_task_b_config_pinned_a(count), target_names,
                                None, None, None, fallback_m=0.5, fallback_nc_h=2,
                                fallback_nc_d=6, fallback_reg="ols")
    counts, _ = cli._resolve_per_target_counts(cfgs, count, None)
    return cfgs, counts


def new_model_margins(d, sel, count, B, seed):
    """(n, T) margins of the new model vs every row: margin[j, t] > 0 means the new
    model is predicted to beat row j on target t."""
    tn = d["target_names"]
    cfgs, counts = task_b_configs(tn, count)
    i, n = d["new_idx"], d["X_v"].shape[0]
    pa, pb = pair_indices(n - 1)
    pin_H = {t_idx: sel[t_idx][0] for t_idx in sel if len(sel[t_idx][0]) > 0}
    pin_D = {t_idx: sel[t_idx][1] for t_idx in sel if len(sel[t_idx][1]) > 0}
    for t_idx, t in enumerate(tn):
        cfg, (cL, cG) = cfgs[t], counts[t]
        if (cfg["m"] > 0 and cL > 0 and t_idx not in pin_H) or \
           (cfg["m"] < 1 and cG > 0 and t_idx not in pin_D):
            raise SystemExit(
                f"{t}: the Task B config uses a Local/Global side that is empty in the "
                "Task A selection; that would need instances outside the Pace-Bench "
                "selection, which this predictor does not support.")
    cache = {}

    def side(kind, reg, nc):
        key = (kind, reg, nc)
        if key not in cache:
            if kind == "H":
                cache[key] = (_fold_H_pair_ols(i, d["X_v"], d["Y"], d["P_target"], nc, count,
                                               B, seed, pa, pb, pinned_sel_by_t=pin_H)
                              if reg == "ols" else
                              _fold_H_logit(i, d["X_v"], d["Y"], d["P_target"], nc, count, B,
                                            seed, DEFAULT_ALPHA, pa, pb, pinned_sel_by_t=pin_H))
            else:
                cache[key] = (_fold_Dpt_pair_ols(i, d["X_v"], d["Y"], d["Vt"], d["P_target"],
                                                 nc, count, B, seed, pa, pb,
                                                 pinned_sel_by_t=pin_D)
                              if reg == "ols" else
                              _fold_Dpt_logit(i, d["X_v"], d["Y"], d["Vt"], d["P_target"], nc,
                                              count, B, seed, DEFAULT_ALPHA, pa, pb,
                                              pinned_sel_by_t=pin_D))
        return cache[key]

    margins = np.zeros((n, len(tn)))
    for t_idx, t in enumerate(tn):
        cfg, (cL, cG) = cfgs[t], counts[t]
        m, reg = cfg["m"], cfg["reg"]
        h = m * side("H", reg, int(cfg["nc_h"]))[:, t_idx] if (m > 0 and cL > 0) else 0.0
        dd = (1 - m) * side("D", reg, int(cfg["nc_d"]))[:, t_idx] if (m < 1 and cG > 0) else 0.0
        margins[:, t_idx] = h + dd
    return margins


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-model", required=True,
                    help="new model name; must have CSVs under results/standardized_results/")
    ap.add_argument("--count", type=int, default=100, help="source-instance budget C")
    ap.add_argument("--B", type=int, default=300, help="target-instance bootstrap replicates")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summary", action="store_true",
                    help="print per-target predicted win-rate instead of every pair")
    args = ap.parse_args()

    d = nm.prepare(args.new_model)
    sel = nm.plan_selection(d, args.count)
    nm.check_against_shipped(d, sel, args.count)
    cov = nm.coverage(d, sel, args.new_model)
    nm.require_scores(cov, args.new_model)
    nm.check_scale(d, sel)

    margins = new_model_margins(d, sel, args.count, args.B, args.seed)

    out_dir = Path(os.environ.get("PROXYBENCH_OUT_DIR") or
                   Path(config.BASE_DIR).parent.parent / "new_model_predictions" / args.new_model)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "pair_reference_run.log"
    print(f"Calibrating P(win) on a reference LOOCV run over the calibration models "
          f"(log: {log_path}) …", flush=True)
    with open(log_path, "w") as fh:
        platt = reference_platt(args, out_dir, fh)

    tn, new_idx, Y = d["target_names"], d["new_idx"], d["Y"]
    rows = []
    for t_idx, t in enumerate(tn):
        a, b = platt[t]
        for j, other in enumerate(d["cal_models"]):
            mg = float(margins[j, t_idx])
            row = {"new_model": args.new_model, "target": t, "other": other,
                   "pred_margin": mg,
                   "pred_prob_new_wins": float(apply_platt(np.array([mg]), a, b)[0]),
                   "pred_winner": args.new_model if mg > 0 else other}
            if t in d["gt_targets"]:
                diff = Y[new_idx, t_idx] - Y[j, t_idx]
                row["true_winner"] = (args.new_model if diff > 0 else
                                      other if diff < 0 else "tie")
            rows.append(row)
    sub = pd.DataFrame(rows)
    out_csv = out_dir / "pair_predictions.csv"
    sub.to_csv(out_csv, index=False)
    sub["new_model_wins"] = sub["pred_winner"] == args.new_model

    print("\n" + "=" * 60)
    print(f"Pairwise predictions for: {args.new_model}")
    print("=" * 60)
    if args.summary:
        print(f"\n{'target':24s} {'pred win-rate':>13s} {'mean P(win)':>12s}")
        print("-" * 52)
        for t, g in sub.groupby("target"):
            print(f"{t:24s} {g['new_model_wins'].mean():13.2%} "
                  f"{g['pred_prob_new_wins'].mean():12.3f}")
    else:
        cols = ["target", "other", "pred_prob_new_wins", "pred_winner"]
        if d["gt_targets"]:
            cols.append("true_winner")
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(sub[cols].to_string(index=False, na_rep="n/a"))
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
