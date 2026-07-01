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

Usage
-----
  python analysis/predict_new_model.py --new-model My-New-Model
  python analysis/predict_new_model.py --new-model My-New-Model --count 100 --B 300 --seed 42
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# make `pacebench` (parents[2]=scripts/) and `cli` (parents[1]=scripts/pacebench/) importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacebench import config  # noqa: E402


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

    import cli  # imported after registration so it binds the mutated MODELS list

    print(f"Running LOOCV abs over {len(config.MODELS)} models "
          f"(holding out each, incl. {args.new_model}) …\n", flush=True)
    cli.run_abs(B=args.B, seed=args.seed, count=args.count,
                strict_budget=True, auto_tune=False)

    out_csv = Path(config.BASE_DIR).parent.parent / "abs_predictions.csv"
    df = pd.read_csv(out_csv)
    sub = df[df["model"] == args.new_model]
    if sub.empty:
        raise SystemExit(f"No rows for '{args.new_model}' in {out_csv}. "
                         "Check the model name and that its CSVs exist under BASE_DIR.")

    print("\n" + "=" * 52)
    print(f"Held-out predictions for: {args.new_model}")
    print("=" * 52)
    cols = [c for c in ("target", "predicted", "actual", "abs_error") if c in sub.columns]
    print(sub[cols].to_string(index=False))


if __name__ == "__main__":
    main()
