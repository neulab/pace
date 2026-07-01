# `pacebench/` — \textsc{Pace}/\textsc{Pace-Bench} pipeline

Selection + regression code that produces \textsc{Pace-Bench} from a pool of
non-agentic source benchmarks and evaluates it under LOOCV. See the paper for
the full method; this README is just a reference for running each script.

## File layout

```
pacebench/
├── cli.py                 # main entry point: abs / pair / fit-abs / fit-pair
├── config.py              # constants + frozen TASK_A_CONFIG / TASK_B_CONFIG
├── data.py                # build_data() → (X, Y, target_names, P_target)
├── selection.py           # h_top_M, dpt_top_N, joint_top_C (strict-budget)
├── embedding.py           # h_embed, dpt_embed, expand_to_all_models
├── bootstrap.py           # pooled_absolute, pooled_pairwise  (env-gated)
├── regression/
│   ├── absolute.py        # Goal A folds + loocv + auto-tune
│   └── pairwise.py        # Goal B folds + loocv + auto-tune
├── calibration.py         # fit_platt, apply_platt
├── metrics.py             # MAE / Spearman / pair-acc + pretty printing
├── stats.py               # spearman_X_vs_Y, ols_1d, sigmoid, pair_indices
├── visualize.py           # plot_abs_loocv_fit, plot_pair_scatter
├── script.sh              # canonical reproduction commands
├── selections/            # selection dumps (CSV); abs_fit/selections_C100.csv is shipped
└── analysis/              # new-model prediction tools (see README_new_model.md)
    ├── emit_eval_plan.py       # which instances a new model must be scored on
    ├── score_new_model.py      # (in evaluations/) run a new model on the selection
    ├── predict_new_model.py    # Goal A: predict a new model's absolute scores
    └── predict_new_model_pair.py  # Goal B: predict pairwise win/loss
```

## Quick start

Reproduce the headline LOOCV numbers from the paper (`Pace-Bench`, $C=100$):

```bash
cd scripts/pacebench

# Goal A: absolute score prediction (MAE, Spearman, Pearson)
python cli.py abs  --count 100 --B 300 --seed 42 --auto-tune --strict-budget \
    > logs/abs_100.log

# Goal B: pairwise preference prediction (accuracy)
# `--pin-task-a-selection` reuses the same selected instances as Goal A
python cli.py pair --count 100 --B 300 --seed 42 --auto-tune --strict-budget \
    --pin-task-a-selection > logs/pair_100.log
```

Outputs (under repo root):
- `abs_predictions.csv`, `abs_loocv_fit.png`
- `pair_predictions.csv`, `pair_loocv_scatter.png`
- `selections/abs_loocv/<fold>/selections_C100.csv` (one per held-out fold)
- The log file contains per-target MAE / Spearman / Pair-Acc + the
  auto-tuned `(q, m, nc_h, nc_d)` block that gets pasted back into `config.py`.

## CLI reference (`cli.py`)

Four subcommands, all sharing the same flag surface:

| subcommand | Protocol | Goal |
|---|---|---|
| `abs`      | LOOCV (M-1 train, 1 eval) | Goal A: predict mean target score |
| `pair`     | LOOCV                     | Goal B: predict pairwise preference |
| `fit-abs`  | In-sample fit on all M models | Goal A analysis |
| `fit-pair` | In-sample fit on all M models | Goal B analysis |

Key flags:

```
--count INT          Source-instance budget C per target (default 100).
--B INT              Bootstrap replicates over target instances (default 300).
                     Set to 1 for "single fit"; see PROXYBENCH_NO_BOOTSTRAP below.
--seed INT           Random seed (default 42).
--auto-tune          Sweep (q, m, nc_h, nc_d) per target; otherwise read from
                     TASK_A_CONFIG / TASK_B_CONFIG in config.py.
--strict-budget      Enforce |H ∪ D| = C per target (the regime used in the
                     paper). Without this flag, |H| = |D| = C and the two
                     selection lists may overlap.
--pin-task-a-selection
                     (pair only) Reuse the instance subset selected by
                     the previous `abs` run, so Goal A and Goal B operate on
                     the same Pace-Bench instances.
--q FLOAT            Manual Local/Global mixing weight (default per-target).
--m FLOAT            Manual H-weight ∈ [0,1] (default per-target).
--nc-h, --nc-d INT   Manual SVD ranks (default per-target).
--sources BENCH …    Restrict source pool (default: 19 benchmarks).
--targets BENCH …    Restrict target list (default: gaia / swebench /
                     swebench_multimodal / swtbench).
--train-models, --eval-models MODEL …
                     Custom calibration / held-out splits.
--dump-selections DIR
                     Dump per-fold selection CSVs to DIR. Used to drive the
                     analysis scripts.
```

### Budget sweep

`script.sh` and `logs/abs_C.log` / `logs/pair_C.log` cover
`C ∈ {25, 50, 100, 200, 300, 400, 500}`. Re-run any of them with:

```bash
for c in 25 50 100 200 300 400 500; do
  python cli.py abs  --count $c --B 300 --seed 42 --auto-tune --strict-budget \
      > logs/abs_${c}.log
  python cli.py pair --count $c --B 300 --seed 42 --auto-tune --strict-budget \
      --pin-task-a-selection > logs/pair_${c}.log
done
```

### Bootstrap ablation

To disable target-instance bootstrap pooling (§5.2 of the paper):

```bash
PROXYBENCH_NO_BOOTSTRAP=1 python cli.py abs  --count 100 \
    --B 300 --seed 42 --auto-tune --strict-budget
```

The env var is read by `bootstrap.py::_no_bootstrap_env()` and forces
`pooled_absolute` / `pooled_pairwise` to return the raw target mean instead of
$B$ resamples. The flag is currently only respected on the non-autotune path
for `--strict-budget --auto-tune` — see `regression/absolute.py:248` if you
need to extend it there as well.

## New-model prediction (`analysis/`)

Tools to predict a brand-new model's agentic scores from the selection. Full
walk-through in `analysis/README_new_model.md`; in short:

```bash
# 1. (optional) preview which selected instances the new model needs
python analysis/emit_eval_plan.py --selections selections/abs_fit/selections_C100.csv --model <MODEL>

# 2. run the new model on the selection -> results/standardized_results/.../<NAME>.csv
python ../../evaluations/score_new_model.py --model <MODEL> --model-name-out <NAME> \
    --base-url "$BASE_URL" --selections selections/abs_fit/selections_C100.csv --workers 8

# 3. predict the 4 agentic-target scores (held-out)
python analysis/predict_new_model.py      --new-model <NAME>            # Goal A: absolute
python analysis/predict_new_model_pair.py --new-model <NAME> --summary   # Goal B: pairwise
```

## Pipeline (4 steps, shared between Goal A and Goal B)

| Step | Module | Goal A | Goal B |
|---|---|---|---|
| 1. Select instances | `selection.py` | `joint_top_C` (Local ∪ Global) | same |
| 2. Embed M models   | `embedding.py` | `h_embed`, `dpt_embed`         | same |
| 3. Pool over target instances | `bootstrap.py` | `pooled_absolute → (U_pool, y_pool)` | `pooled_pairwise → (D_pool, dy_pool)` |
| 4. Fit + predict    | `regression/`  | least-squares (1-D OLS, ridge) | logistic (L-BFGS + ridge)            |

Per-target `(q, m, nc_h, nc_d)` live in `config.py::TASK_A_CONFIG` /
`TASK_B_CONFIG` and are produced by `--auto-tune`. The auto-tune block in each
log file is the recommended way to update them after a re-run.

## No-Y-leakage invariants

The pipeline is structured to avoid information leakage in the LOOCV setting:

- SVD of $X$ uses only source features.
- Local/Global scoring (`h_top_M`, `dpt_top_N`) uses only training-fold $Y$.
- Bootstrap pooling samples target instances from training models only.
- Regression fits on training-model embeddings (or pair-diffs) only.
- `expand_to_all_models` places the held-out model at index $i$ — it
  participates in prediction (`u_k - u_j`) but never in fitting.
