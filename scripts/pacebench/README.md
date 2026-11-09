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
mkdir -p logs

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
--seed INT           Random seed (default 42).
--auto-tune          Sweep (q, m, nc_h, nc_d) per target; otherwise read from
                     TASK_A_CONFIG / TASK_B_CONFIG in config.py.
--strict-budget      Enforce |H ∪ D| = C per target (the regime used in the
                     paper). Without this flag, |H| = |D| = C and the two
                     selection lists may overlap.
--per-fold-svd       Recompute the Global-SVD basis on each fold's training
                     models. Applies to abs / pair with --strict-budget
                     --auto-tune, where it is the default.
--global-svd         Use one SVD basis computed over all loaded models instead
                     (always the case on the other paths).
--pin-task-a-selection
                     (pair only) Pin Goal B to the instances Goal A selects
                     (recomputed in the same run), so Goal A and Goal B operate
                     on the same Pace-Bench instances.
--q FLOAT            Manual Local budget share, C_L = round(q·C) (default per-target).
--m FLOAT            Manual Local/Global mixing weight ∈ [0,1] (default per-target).
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

To sweep `C ∈ {25, 50, 100, 200, 300, 400, 500}`:

```bash
mkdir -p logs
for c in 25 50 100 200 300 400 500; do
  python cli.py abs  --count $c --B 300 --seed 42 --auto-tune --strict-budget \
      > logs/abs_${c}.log
  python cli.py pair --count $c --B 300 --seed 42 --auto-tune --strict-budget \
      --pin-task-a-selection > logs/pair_${c}.log
done
```

## New-model prediction (`analysis/`)

Tools to predict a brand-new model's agentic scores from the selection. Full
walk-through in `analysis/README_new_model.md`; in short:

```bash
# 1. (optional) preview which selected instances the new model needs
python analysis/emit_eval_plan.py --selections selections/abs_fit/selections_C100.csv --model <MODEL>

# 2. run the new model on the selection -> results/standardized_results/.../<NAME>.csv
python ../../evaluations/score_new_model.py --model <MODEL> --model-name-out <NAME> \
    --base-url "$BASE_URL" --selections selections/abs_fit/selections_C100.csv --workers 8

# 3. predict the 4 agentic-target scores (fit on the calibration models)
python analysis/predict_new_model.py      --new-model <NAME>            # Goal A: absolute
python analysis/predict_new_model_pair.py --new-model <NAME> --summary   # Goal B: pairwise
```

## Pipeline (4 steps, shared between Goal A and Goal B)

| Step | Module | Goal A | Goal B |
|---|---|---|---|
| 1. Select instances | `selection.py` | `joint_top_C` (Local ∪ Global) | same |
| 2. Embed M models   | `embedding.py` | `h_embed`, `dpt_embed`         | same |
| 3. Pool over target instances | `bootstrap.py` | `pooled_absolute → (U_pool, y_pool)` | `pooled_pairwise → (D_pool, dy_pool)` |
| 4. Fit + predict    | `regression/`  | Spearman-weighted 1-D OLS      | logistic (L-BFGS, L2)                |

Per-target `(q, m, nc_h, nc_d)` live in `config.py::TASK_A_CONFIG` /
`TASK_B_CONFIG` and are produced by `--auto-tune`. The auto-tune block in each
log file is the recommended way to update them after a re-run. After changing
`TASK_A_CONFIGS`, regenerate `selections/abs_fit/selections_C<count>.csv` with
`fit-abs` (the new-model tools check that it matches).

## Per-fold computations

Within each LOOCV fold, these steps use the training models only:

- instance scoring (`h_top_M`, `dpt_top_N`, `joint_top_C` use training-fold $Y$);
- the Global-SVD basis (recomputed on the training models with `--per-fold-svd`,
  the default for `--strict-budget --auto-tune`);
- bootstrap pooling (target instances are resampled over training models);
- regression fitting (training-model embeddings or pair-diffs).

The held-out model is then projected into the fitted model: `expand_to_all_models`
places it at index $i$ for prediction (`u_k - u_j`).
