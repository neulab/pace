# Predicting a NEW model with Pace-Bench

Given a model that is **not** in the original pool, predict its scores on the four
agentic targets (gaia / swebench / swebench_multimodal / swtbench) from cheap
non-agentic evaluations — without running the full agentic benchmarks.

Three helper scripts (run from `scripts/pacebench/`):

| script | what it does | output |
|---|---|---|
| `analysis/emit_eval_plan.py` | list the exact source instances the new model must be scored on | `eval_plan.csv` + per-benchmark summary + `evaluations/run.py` commands |
| `analysis/predict_new_model.py` | **Goal A** — predict the 4 absolute target scores | `abs_predictions.csv` (rows for the new model) |
| `analysis/predict_new_model_pair.py` | **Goal B** — predict pairwise win/loss vs every existing model | `pair_predictions.csv` (rows where the new model is held-out) |

## How it works

The prediction reuses the real leave-one-model-out (LOOCV) pipeline, **not** a
hand-rolled weighted sum of the dumped selection weights (those weights are an
interpretability approximation, not an exact predictor). The scripts register the
new model, run the standard `abs` / `pair` pipeline holding it out, and read its
prediction — the regression is fit only on the *other* models, so it is a proper
out-of-sample estimate, using the frozen hyper-parameters in
`config.TASK_A_CONFIGS[count]` / `config.TASK_B_CONFIGS_PINNED_A[count]`.

Because the held-out fold selects instances from the *other* models' scores, the
set of source instances the new model needs is exactly the fit-selection dump
(`selections/abs_fit/selections_C100.csv`). That is what `emit_eval_plan.py` reads.

## Workflow

```bash
cd scripts/pacebench

# 0. (once) produce the in-sample selection if you don't have it
python cli.py fit-abs --count 100 --B 300 --seed 0 --auto-tune --strict-budget \
    --dump-selections selections
#   → selections/abs_fit/selections_C100.csv

# 1. Which source instances does a new model need? (≈358 calls, not full benchmarks)
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
  under `results/standardized_results/`. Missing instances are treated as `0`
  (same as the pipeline), so cover the full `eval_plan.csv` for a faithful estimate.
- **No `--auto-tune`** in the predictors: auto-tuning would leak the eval model
  into hyper-parameter selection. The frozen `config.TASK_*_CONFIGS[count]` are used.
- **`emit_eval_plan.py` naming translation** (already handled): repobench subdirs
  carry a metric suffix that is stripped to the real subtask; lm_eval metric
  benchmarks (ifeval/logiqa/gpqa/…) drop the metric subdir. One evaluation call
  per instance, deduped.
- **Runtime:** the predictors run LOOCV over all models (existing + new). Use a
  smaller `--B` (e.g. 100) for a quick check; `--B 300` matches the paper.
- **Cost:** step 2 evaluates ~358 cheap non-agentic instances — well under 1% of a
  full agentic run, which is the whole point of Pace-Bench.
