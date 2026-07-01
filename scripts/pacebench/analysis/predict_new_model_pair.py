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

Usage
-----
  python analysis/predict_new_model_pair.py --new-model My-New-Model
  python analysis/predict_new_model_pair.py --new-model My-New-Model --count 100 --B 300
  # add --summary for a per-target win-rate summary instead of every pair
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/pacebench/

from pacebench import config  # noqa: E402


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

    import cli  # after registration

    print(f"Running LOOCV pair over {len(config.MODELS)} models "
          f"(holding out each, incl. {args.new_model}) …\n", flush=True)
    cli.run_pair(B=args.B, seed=args.seed, count=args.count,
                 strict_budget=True, auto_tune=False, pin_task_a_selection=True)

    out_csv = Path(config.BASE_DIR).parent.parent / "pair_predictions.csv"
    df = pd.read_csv(out_csv)
    # rows where the new model is the held-out one: new model vs every `other`
    sub = df[df["held_out"] == args.new_model].copy()
    if sub.empty:
        raise SystemExit(f"No rows for '{args.new_model}' in {out_csv}.")

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
        cols = ["target", "other", "pred_prob_heldout_wins", "pred_winner",
                "true_winner"]
        cols = [c for c in cols if c in sub.columns]
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(sub[cols].to_string(index=False))


if __name__ == "__main__":
    main()
