#!/usr/bin/env python3
"""Predict a NEW model's agentic-target scores (Goal A) from Pace-Bench.

The regression is fit on the calibration models only (config.MODELS), with the
frozen per-target config config.TASK_A_CONFIGS[count], on the Pace-Bench
selection those models imply (= selections/abs_fit/selections_C<count>.csv, the
instances evaluations/score_new_model.py scores). The new model's scores on the
selected instances are then projected into the fitted model. Its own scores never
influence which instances are used, so only the selected instances matter.

Requirements
------------
The new model must already have per-instance score CSVs under BASE_DIR
(results/standardized_results/<benchmark>/.../<NewModel>.csv), produced by
evaluations/score_new_model.py. Selected instances without a score are read as 0;
the script reports per-target coverage.

Outputs go to new_model_predictions/<NewModel>/ at the repo root (override with
$PROXYBENCH_OUT_DIR). If the new model also has agentic ground-truth CSVs,
`actual` and `abs_error` are reported for those targets.

Usage
-----
  python analysis/predict_new_model.py --new-model My-New-Model
  python analysis/predict_new_model.py --new-model My-New-Model --count 100 --B 300 --seed 42
"""
import os
# Same BLAS thread settings as cli.py, applied before numpy/pandas are imported:
# the thread count changes floating-point summation order.
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import sys
from pathlib import Path

import pandas as pd

# make `pacebench` (parents[2]=scripts/), `cli` (parents[1]) and the analysis helpers importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pacebench import config  # noqa: E402
from pacebench.regression.absolute import _fold_joint_abs  # noqa: E402
import _new_model as nm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-model", required=True,
                    help="new model name; must have CSVs under results/standardized_results/")
    ap.add_argument("--count", type=int, default=100, help="source-instance budget C")
    ap.add_argument("--B", type=int, default=300, help="target-instance bootstrap replicates")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = nm.prepare(args.new_model)
    sel = nm.plan_selection(d, args.count)
    nm.check_against_shipped(d, sel, args.count)
    cov = nm.coverage(d, sel, args.new_model)
    nm.require_scores(cov, args.new_model)
    nm.check_scale(d, sel)

    out_dir = Path(os.environ.get("PROXYBENCH_OUT_DIR") or
                   Path(config.BASE_DIR).parent.parent / "new_model_predictions" / args.new_model)
    out_dir.mkdir(parents=True, exist_ok=True)

    tn = d["target_names"]
    cfgs = nm.task_a_configs(tn, args.count)
    per_target_qC, nc_h_per_t, nc_d_per_t = {}, {}, {}
    for t_idx, t in enumerate(tn):
        cfg = cfgs[t]
        q = cfg.get("q") if cfg.get("q") is not None else 0.5
        per_target_qC[t_idx] = (float(q), int(args.count))
        nc_h_per_t[t_idx] = int(cfg["nc_h"])
        nc_d_per_t[t_idx] = int(cfg["nc_d"])

    # One fold with the new model held out: trains on the calibration rows only.
    pred_H, pred_D, fold_sel = _fold_joint_abs(
        d["new_idx"], d["X_v"], d["Y"], d["Vt"], per_target_qC, nc_h_per_t,
        nc_d_per_t, d["P_target"], args.B, args.seed)

    rows = []
    for t_idx, t in enumerate(tn):
        # the fold's selection must be the plan (same rows, same filter and basis)
        h0, d0 = sel[t_idx]; h1, d1 = fold_sel[t_idx]
        assert list(h0) == list(h1) and list(d0) == list(d1), f"selection mismatch on {t}"
        m = cfgs[t]["m"]
        pred = m * pred_H[t_idx] + (1 - m) * pred_D[t_idx]
        have, n_sel = cov[t]
        row = {"model": args.new_model, "target": t, "predicted": float(pred),
               "scored_instances": have, "selected_instances": n_sel}
        if t in d["gt_targets"]:
            actual = float(d["Y"][d["new_idx"], t_idx])
            row.update(actual=actual, abs_error=abs(pred - actual))
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = out_dir / "abs_predictions.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 52)
    print(f"Predicted agentic scores for: {args.new_model}")
    print("=" * 52)
    cols = ["target", "predicted"] + (["actual", "abs_error"] if d["gt_targets"] else [])
    print(df[cols].to_string(index=False, na_rep="n/a"))
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
