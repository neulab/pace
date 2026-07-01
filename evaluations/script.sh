#!/usr/bin/env bash
# =============================================================================
# evaluations/script.sh
#
# Run a single-instance evaluation: given model + API + benchmark + instance_id,
# call the model, grade it, and print the instance's JSON result (one raw_results row).
#
# Entry point evaluations/run.py must be run as a module from the repo root
# (this script cd's there automatically):
#     python -m evaluations.run ...
# So `bash evaluations/script.sh` works from any directory.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root, so `python -m evaluations.run` finds the package

# Which python: main env uses /usr/local/bin/python; BFCL needs its own venv (see bottom).
PYTHON="${PYTHON:-/usr/local/bin/python}"

# ─────────────────────────────────────────────────────────────────────────────
# *** CHANGE THE MODEL HERE ***  (base_url / api_key stay the same)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-azure_ai/gpt-5}"
#
# Verified-working models:
#     azure_ai/gpt-5.2                 (default)
#     anthropic/claude-sonnet-4-5
# List every available model on the proxy (~755):
#     curl -s "$BASE_URL/v1/models" -H "Authorization: Bearer $API_KEY" \
#       | python -c "import sys,json;[print(m['id']) for m in json.load(sys.stdin)['data']]"
#
# Three gotchas when switching models:
#   1. Some newer reasoning models (e.g. anthropic/claude-opus-4-8) reject
#      `temperature`; handlers hard-code temperature=0. The handlers now drop
#      temperature and retry automatically (evaluations/handlers/_compat.py).
#   2. Not all 755 are reachable: some routes (e.g. openai/gpt-5-chat-latest)
#      401 due to a misconfigured upstream key on the proxy side, not yours.
#   3. Multimodal benchmarks (mmmu / visualpuzzles / visualwebbench) need a
#      vision-capable model, or image inputs will error.
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL="${BASE_URL:-https://cmu.litellm.ai}"
API_KEY="${API_KEY:?set your API key first, e.g. export API_KEY=sk-...}"

# Needed for HF dataset downloads (else hf_transfer errors); lm_eval / multimodal download data.
export HF_HUB_ENABLE_HF_TRANSFER=0

run() {  # run <benchmark> <instance_id> [subtask]
  local bench="$1" iid="$2" subtask="${3:-}"
  echo "==================================================================="
  echo ">>> model=$MODEL_NAME  benchmark=$bench  instance_id=$iid  subtask=${subtask:-<default>}"
  echo "==================================================================="
  if [[ -n "$subtask" ]]; then
    "$PYTHON" -m evaluations.run --model_name "$MODEL_NAME" --base_url "$BASE_URL" \
      --api_key "$API_KEY" --benchmark "$bench" --instance_id "$iid" --subtask "$subtask"
  else
    "$PYTHON" -m evaluations.run --model_name "$MODEL_NAME" --base_url "$BASE_URL" \
      --api_key "$API_KEY" --benchmark "$bench" --instance_id "$iid"
  fi
}

# ── Default: run the most self-contained example (infobench) ─────────────────
run infobench user_oriented_task_167

# =============================================================================
# Reference commands for all 19 non-agentic benchmarks (default subtask + a real
# instance_id). Uncomment to run. Install that benchmark's deps first:
#     pip install -r evaluations/requirements/<benchmark>.txt
# or use the verified combo:  evaluations/requirements-verified.txt
# Pick another instance: find results/standardized_results/<benchmark> -name '*.csv' | head -1 | xargs cut -d, -f1 | head
# =============================================================================

# --- lm-evaluation-harness family (deps: requirements/lm_eval.txt) ---
# run aime25          0
# run gpqa            0
# run ifeval          0
# run logiqa          0_strict-match
# run humaneval_chat  0                       # (integer id, not "HumanEval/0")
# run mbpp_chat       0                       # (integer id, not "Mbpp/11")
# run mmlu_cot        abstract_algebra_0
# run acp_gen         0                       acp_app_gen

# --- code ---
# run livecodebench   0                       codegeneration        # deps: livecodebench.txt (first run downloads a big dataset)
# run repobench       repobench_0             repobench_xff_python  # deps: repobench.txt
# run debugbench      "python3_condition error_the-kth-factor-of-n" # deps: debugbench.txt + keys.json (LeetCode creds)

# --- instruction / planning ---
# run lifbench        0_0_3                   onedoc-qa
# run planbench       2                       task_1_plan_generation  # needs results/raw_results/planbench
# run planbench       1                       task_3_plan_verification  # task_3 yields a binary score

# --- multimodal (needs a vision model; deps: lmms_eval.txt / visualwebbench.txt) ---
# run mmmu            validation_Accounting_1
# run visualpuzzles   visualpuzzles_0         cot
# run visualwebbench  web_caption_0           web_caption

# --- retrieval ---
# BEIR_MAX_QUERIES=2 run beir_nfcorpus "NDCG@10"   # deps: beir.txt; unbounded reranks all queries (slow)

# --- tool calling (BFCL: must use its own venv; hard tree-sitter conflict with main env) ---
# PYTHON=/path/to/bfcl-venv/bin/python \
#   run bfcl parallel_multiple_0 non_live_parallel_multiple   # deps: requirements/bfcl.txt

# =============================================================================
# Notes:
#   1. Every instance is a real API call and costs money (generation + grading).
#   2. debugbench really submits to LeetCode using keys.json creds (shows in your submissions).
#   3. API_KEY / keys.json hold sensitive credentials; do not commit them to git.
# =============================================================================
