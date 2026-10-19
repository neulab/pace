#!/usr/bin/env python3
"""Predict a NEW model's agentic-target scores with the real Pace-Bench pipeline.

This does NOT hand-roll the prediction from the dumped selection weights (those
weights are an interpretability approximation, not an exact linear predictor).
Instead it registers the new model, runs the standard leave-one-model-out (LOOCV)
`abs` pipeline, and reads the new model's held-out prediction — i.e. the model is
predicted using a regression fit on all the *other* models, exactly as in the paper.

Requirements
------------
The new model must already have per-instance score CSVs under BASE_DIR
(results/standardized_results/<benchmark>/.../<NewModel>.csv), produced by running
`evaluations/` on it. Missing source instances are treated as 0 (same as the
pipeline). It uses the frozen hyper-parameters in config.TASK_A_CONFIGS[count]
(no --auto-tune, which would leak the eval model into tuning).

Outputs go to new_model_predictions/<NewModel>/ at the repo root (override with
$PROXYBENCH_OUT_DIR), so the main LOOCV outputs at the repo root are never
overwritten.

Without agentic ground truth (the usual case for a new model) its target scores
are read as 0. Its own held-out prediction is unaffected — that fold trains only on
the other models — but the other models' LOOCV metrics in the same run are not
meaningful, so the full run log goes to <out>/abs_run.log and only the new model's
predictions are printed (`actual` / `abs_error` only for targets with ground truth).

Usage
-----
  python analysis/predict_new_model.py --new-model My-New-Model
  python analysis/predict_new_model.py --new-model My-New-Model --count 100 --B 300 --seed 42
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

import pandas as pd

# make `pacebench` (parents[2]=scripts/) and `cli` (parents[1]=scripts/pacebench/) importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacebench import config  # noqa: E402


def targets_with_ground_truth(model):
    """Agentic targets for which `model` has a score CSV under BASE_DIR."""
    base = Path(config.BASE_DIR)
    return {t for t in config.TARGET_DATASETS if any((base / t).rglob(f"{model}.csv"))}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-model", required=True,
                    help="new model name; must have CSVs under results/standardized_results/")
    ap.add_argument("--count", type=int, default=100, help="source-instance budget C")
    ap.add_argument("--B", type=int, default=300, help="target-instance bootstrap replicates")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Register the new model so build_data loads it and LOOCV holds it out.
    # (config.MODELS is a shared list object; mutating it is seen everywhere.)
    if args.new_model not in config.MODELS:
        config.MODELS.append(args.new_model)
        config.N_MODELS = len(config.MODELS)

    out_dir = Path(os.environ.setdefault(
        "PROXYBENCH_OUT_DIR",
        str(Path(config.BASE_DIR).parent.parent / "new_model_predictions" / args.new_model)))
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_targets = targets_with_ground_truth(args.new_model)
    missing_gt = sorted(set(config.TARGET_DATASETS) - gt_targets)

    import cli  # imported after registration so it binds the mutated MODELS list

    print(f"Running LOOCV abs over {len(config.MODELS)} models "
          f"(holding out each, incl. {args.new_model}) …", flush=True)
    run = lambda: cli.run_abs(B=args.B, seed=args.seed, count=args.count,  # noqa: E731
                              strict_budget=True, auto_tune=False)
    if missing_gt:
        log_path = out_dir / "abs_run.log"
        print(f"No agentic ground truth for {args.new_model} on: {', '.join(missing_gt)}.\n"
              "Its held-out prediction is unaffected, but the other models' LOOCV metrics in\n"
              f"this run would count those scores as 0, so the run log goes to {log_path} only.",
              flush=True)
        with open(log_path, "w") as fh, contextlib.redirect_stdout(fh):
            run()
    else:
        print()
        run()

    out_csv = out_dir / "abs_predictions.csv"
    df = pd.read_csv(out_csv)
    sub = df[df["model"] == args.new_model].copy()
    if sub.empty:
        raise SystemExit(f"No rows for '{args.new_model}' in {out_csv}. "
                         "Check the model name and that its CSVs exist under BASE_DIR.")
    sub.loc[~sub["target"].isin(gt_targets), ["actual", "abs_error"]] = float("nan")

    print("\n" + "=" * 52)
    print(f"Held-out predictions for: {args.new_model}")
    print("=" * 52)
    cols = ["target", "predicted"] + (["actual", "abs_error"] if gt_targets else [])
    print(sub[cols].to_string(index=False, na_rep="n/a"))
    print(f"\nOutputs saved under {out_dir}")


if __name__ == "__main__":
    main()
