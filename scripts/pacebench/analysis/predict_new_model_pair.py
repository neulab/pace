#!/usr/bin/env python3
"""Predict a NEW model's PAIRWISE win/loss vs existing models (Goal B).

Mirror of predict_new_model.py for Task B: registers the new model, runs the
standard leave-one-model-out (LOOCV) `pair` pipeline, and reads the held-out
model's predicted pairwise outcomes — i.e. for each existing model `other` and
each agentic target, the probability the new model beats `other` and the
predicted winner. The regression is fit on the other models only (proper
held-out), using the frozen config in config.TASK_B_CONFIGS[count].

Requirements
------------
The new model must already have per-instance score CSVs under BASE_DIR
(results/standardized_results/...), produced by running `evaluations/` on it.

Outputs go to new_model_predictions/<NewModel>/ at the repo root (override with
$PROXYBENCH_OUT_DIR), so the main LOOCV outputs at the repo root are never
overwritten.

Without agentic ground truth (the usual case for a new model) its target scores
are read as 0. The predicted margins / winners of its own fold are unaffected, but
the pooled Platt calibration and the other models' LOOCV metrics of that run would
use those zeros. So the calibration is refit on a reference LOOCV run over the
existing models only (saved under <out>/reference_without_new_model/), the run
logs go to <out>/pair_run.log, and `true_winner` is shown only for targets with
ground truth.

Usage
-----
  python analysis/predict_new_model_pair.py --new-model My-New-Model
  python analysis/predict_new_model_pair.py --new-model My-New-Model --count 100 --B 300
  # add --summary for a per-target win-rate summary instead of every pair
"""
import os
# Same BLAS thread settings as cli.py, applied before numpy/pandas are imported:
# the thread count changes floating-point summation order, so without this the
# predictions differ slightly from `cli.py abs` / `cli.py pair`.
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

from pacebench import config  # noqa: E402
from pacebench.calibration import apply_platt, fit_platt  # noqa: E402


def targets_with_ground_truth(model):
    """Agentic targets for which `model` has a score CSV under BASE_DIR."""
    base = Path(config.BASE_DIR)
    return {t for t in config.TARGET_DATASETS if any((base / t).rglob(f"{model}.csv"))}


def reference_platt(cli, args, out_dir, log):
    """Per-target Platt (a, b) from a standard LOOCV `pair` run over the existing
    models only — the same fit run_pair does (non-tied pairs, label = true margin > 0),
    but independent of the new model's unknown targets."""
    ref_dir = out_dir / "reference_without_new_model"
    config.MODELS.remove(args.new_model)
    os.environ["PROXYBENCH_OUT_DIR"] = str(ref_dir)
    try:
        with contextlib.redirect_stdout(log):
            print("\n\n##### reference run (existing models only) #####\n", flush=True)
            cli.run_pair(B=args.B, seed=args.seed, count=args.count,
                         strict_budget=True, auto_tune=False, pin_task_a_selection=True)
    finally:
        config.MODELS.append(args.new_model)
        os.environ["PROXYBENCH_OUT_DIR"] = str(out_dir)
    ref = pd.read_csv(ref_dir / "pair_predictions.csv")
    ref = ref[(ref["true_margin"].abs() >= 1e-9) & np.isfinite(ref["pred_margin"])]
    return {t: fit_platt(g["pred_margin"].to_numpy(), (g["true_margin"] > 0).astype(int).to_numpy())
            for t, g in ref.groupby("target")}


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

    if args.new_model not in config.MODELS:
        config.MODELS.append(args.new_model)
        config.N_MODELS = len(config.MODELS)

    out_dir = Path(os.environ.setdefault(
        "PROXYBENCH_OUT_DIR",
        str(Path(config.BASE_DIR).parent.parent / "new_model_predictions" / args.new_model)))
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_targets = targets_with_ground_truth(args.new_model)
    missing_gt = sorted(set(config.TARGET_DATASETS) - gt_targets)

    import cli  # after registration

    print(f"Running LOOCV pair over {len(config.MODELS)} models "
          f"(holding out each, incl. {args.new_model}) …", flush=True)
    run = lambda: cli.run_pair(B=args.B, seed=args.seed, count=args.count,  # noqa: E731
                               strict_budget=True, auto_tune=False, pin_task_a_selection=True)
    platt = None
    if missing_gt:
        log_path = out_dir / "pair_run.log"
        print(f"No agentic ground truth for {args.new_model} on: {', '.join(missing_gt)}.\n"
              "Its predicted winners are unaffected; P(win) is recalibrated on a reference run\n"
              f"over the existing models only. Run logs go to {log_path} only.", flush=True)
        with open(log_path, "w") as fh:
            with contextlib.redirect_stdout(fh):
                run()
            platt = reference_platt(cli, args, out_dir, fh)
    else:
        print()
        run()

    out_csv = out_dir / "pair_predictions.csv"
    df = pd.read_csv(out_csv)
    # rows where the new model is the held-out one: new model vs every `other`
    sub = df[df["held_out"] == args.new_model].copy()
    if sub.empty:
        raise SystemExit(f"No rows for '{args.new_model}' in {out_csv}.")

    if platt is not None:
        for t, (a, b) in platt.items():
            rows = sub["target"] == t
            sub.loc[rows, "pred_prob_heldout_wins"] = apply_platt(
                sub.loc[rows, "pred_margin"].to_numpy(), a, b)
        sub.loc[~sub["target"].isin(gt_targets), "true_winner"] = None
    sub["new_model_wins"] = sub["pred_winner"] == args.new_model

    print("\n" + "=" * 60)
    print(f"Pairwise predictions for: {args.new_model}  (as held-out)")
    print("=" * 60)

    if args.summary:
        print(f"\n{'target':24s} {'pred win-rate':>13s} {'mean P(win)':>12s}")
        print("-" * 52)
        for t, g in sub.groupby("target"):
            print(f"{t:24s} {g['new_model_wins'].mean():13.2%} "
                  f"{g['pred_prob_heldout_wins'].mean():12.3f}")
    else:
        cols = ["target", "other", "pred_prob_heldout_wins", "pred_winner"]
        if gt_targets:
            cols.append("true_winner")
        cols = [c for c in cols if c in sub.columns]
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(sub[cols].to_string(index=False, na_rep="n/a"))
    print(f"\nOutputs saved under {out_dir}")


if __name__ == "__main__":
    main()
