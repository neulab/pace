# Predicting a NEW model with Pace-Bench

Given a model that is **not** in the original pool, predict its scores on the four
agentic targets (gaia / swebench / swebench_multimodal / swtbench) from cheap
non-agentic evaluations — without running the full agentic benchmarks.

Three helper scripts (run from `scripts/pacebench/`):

| script | what it does | output |
|---|---|---|
| `analysis/emit_eval_plan.py` | list the exact source instances the new model must be scored on | `eval_plan.csv` + per-benchmark summary + `evaluations/run.py` commands |
| `analysis/predict_new_model.py` | **Goal A** — predict the 4 absolute target scores | `new_model_predictions/<NAME>/abs_predictions.csv` |
| `analysis/predict_new_model_pair.py` | **Goal B** — predict pairwise win/loss vs every calibration model | `new_model_predictions/<NAME>/pair_predictions.csv` |

## How it works

The prediction uses the real pipeline code, **not** a hand-rolled weighted sum of
the dumped selection weights (those weights are an interpretability approximation,
not an exact predictor). The regressions are fit on the calibration models only
(`config.MODELS`), with the frozen hyper-parameters in
`config.TASK_A_CONFIGS[count]` / `config.TASK_B_CONFIGS_PINNED_A[count]`, and the
new model is projected into the fitted model — the same fold computation as LOOCV,
with the new model as the held-out row.

Everything that decides which source instances are used (the zero-variance column
filter, the Global-SVD basis and the per-target selection) is computed on the
calibration models alone, so the new model's own scores never change it. The
selection is therefore exactly the fit-selection dump
(`selections/abs_fit/selections_C100.csv`) — the scripts check this — and that is
what `emit_eval_plan.py` and `evaluations/score_new_model.py` read. Only the
selected instances affect the prediction.

Each script reports, per target, how many selected instances the new model has
scores for, and stops if it has none (e.g. a misspelled `--new-model`).
`predict_new_model_pair.py` Platt-calibrates P(win) on a standard LOOCV `pair` run
over the calibration models only (`new_model_predictions/<NAME>/reference_without_new_model/`).
`actual` / `abs_error` (Goal A) and `true_winner` (Goal B) are filled only for
targets where the new model has agentic ground-truth CSVs.

## Workflow

```bash
cd scripts/pacebench

# 0. (once) produce the in-sample selection if you don't have it
python cli.py fit-abs --count 100 --B 300 --seed 42 --strict-budget \
    --dump-selections selections
#   → selections/abs_fit/selections_C100.csv

# 1. Which source instances does a new model need? (360 calls, not full benchmarks)
python analysis/emit_eval_plan.py \
    --selections selections/abs_fit/selections_C100.csv \
    --model azure_ai/gpt-5.2 --base-url https://cmu.litellm.ai --emit-commands
#   → eval_plan.csv (benchmark, subtask, instance_id) + ready-to-run commands

# 2. Evaluate the new model on exactly those instances AND write the standardized
#    CSVs pacebench reads — one command (from the repo root):
#      export API_KEY=...
#      python evaluations/score_new_model.py \
#          --model <API_MODEL_ID> --model-name-out <NewModel> \
#          --base-url https://cmu.litellm.ai \
#          --selections scripts/pacebench/selections/abs_fit/selections_C100.csv
#    -> results/standardized_results/<benchmark>/.../<NewModel>.csv  (resumable)

# 3a. Goal A — absolute score prediction for the 4 targets
python analysis/predict_new_model.py --new-model <NewModel> --count 100 --B 300

# 3b. Goal B — pairwise win/loss vs existing models
python analysis/predict_new_model_pair.py --new-model <NewModel> --count 100 --B 300 --summary
```

## Notes / caveats

- **Prereq for steps 3a/3b:** the new model must already have per-instance CSVs
  under `results/standardized_results/`. Selected instances without a score are
  treated as `0` (same as the pipeline), so cover the full `eval_plan.csv` for a
  faithful estimate; the predictors print per-target coverage.
- **No `--auto-tune`** in the predictors: the frozen `config.TASK_*_CONFIGS[count]`
  are used.
- **`emit_eval_plan.py` naming translation** (already handled): repobench subdirs
  carry a metric suffix that is stripped to the real subtask; lm_eval metric
  benchmarks (ifeval/logiqa/gpqa/…) drop the metric subdir. One evaluation call
  per instance, deduped.
- **Runtime:** a few seconds for Goal A; Goal B also runs the reference LOOCV
  `pair` run for the P(win) calibration (under a minute). `--B 300` matches the paper.
- **Cost:** step 2 evaluates 360 cheap non-agentic instances — well under 1% of a
  full agentic run, which is the whole point of Pace-Bench.
