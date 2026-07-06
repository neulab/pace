#!/usr/bin/env bash
# Run one benchmark instance (call model + grade) via evaluations/run.py.
#   bash evaluations/script.sh        # MODEL_NAME via env; credentials via .env
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root, so `python -m evaluations.run` resolves

# API_KEY / BASE_URL are read by run.py from .env (or the environment) — not here.
PYTHON="${PYTHON:-/usr/local/bin/python}"     # BFCL needs its own venv (see bottom)
MODEL_NAME="${MODEL_NAME:-azure_ai/gpt-5}"
export HF_HUB_ENABLE_HF_TRANSFER=0

run() {  # run <benchmark> <instance_id> [subtask]
  local bench="$1" iid="$2" subtask="${3:-}"
  echo ">>> model=$MODEL_NAME  benchmark=$bench  instance_id=$iid  subtask=${subtask:-<default>}"
  if [[ -n "$subtask" ]]; then
    "$PYTHON" -m evaluations.run --model_name "$MODEL_NAME" \
      --benchmark "$bench" --instance_id "$iid" --subtask "$subtask"
  else
    "$PYTHON" -m evaluations.run --model_name "$MODEL_NAME" \
      --benchmark "$bench" --instance_id "$iid"
  fi
}

# example
run infobench user_oriented_task_167

# ── Reference: uncomment one. Install its deps first: requirements/<benchmark>.txt ──
# run aime25          0
# run gpqa            0
# run ifeval          0
# run infobench user_oriented_task_167
# run logiqa          0_strict-match
# run humaneval_chat  0
# run mbpp_chat       0
# run mmlu_cot        abstract_algebra_0
# run acp_gen         0                       acp_app_gen
# run livecodebench   0                       codegeneration
# run repobench       repobench_0             repobench_xff_python
# run debugbench      "python3_condition error_the-kth-factor-of-n"   # needs keys.json (LeetCode)
# run lifbench        0_0_3                   onedoc-qa
# run planbench       2                       task_1_plan_generation
# run planbench       1                       task_3_plan_verification
# run mmmu            validation_Accounting_1
# run visualpuzzles   visualpuzzles_0         cot
# run visualwebbench  web_caption_0           web_caption
# BEIR_MAX_QUERIES=2 run beir_nfcorpus "NDCG@10"
# PYTHON=evaluations/.bfcl-venv/bin/python run bfcl parallel_multiple_0 non_live_parallel_multiple
