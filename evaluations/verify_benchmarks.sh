#!/usr/bin/env bash
# Verify all 19 non-agentic benchmarks run end-to-end on one example instance.
# Fault-tolerant: runs them ALL (does not stop on the first failure) and prints a
# PASS/FAIL summary at the end. Credentials come from .env / the environment (read
# by run.py). Run on Linux with Part 2 installed (bash evaluations/install.sh).
#
#   bash evaluations/verify_benchmarks.sh
#   MODEL_NAME=azure_ai/gpt-5.2 bash evaluations/verify_benchmarks.sh
#
# BFCL (the 19th) is auto-detected at evaluations/.bfcl-venv (created by install.sh);
# override its path with BFCL_VENV=... , or it is SKIP-ped if that venv is missing.
# Each PASS is a real API call (costs money). debugbench needs keys.json;
# planbench's non-verification tasks need VAL + Fast Downward.
set -uo pipefail                       # NOTE: no -e, we want to run past failures
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-azure_ai/gpt-5}"
export HF_HUB_ENABLE_HF_TRANSFER=0

PASS=0; FAIL=0; SUMMARY=""

check() {  # check <python> <benchmark> <instance_id> [subtask]
  local py="$1" bench="$2" iid="$3" subtask="${4:-}"
  local label="$bench${subtask:+/$subtask}"
  printf '\n=== %s ===\n' "$label"
  local out rc
  if [[ -n "$subtask" ]]; then
    out=$("$py" -m evaluations.run --model_name "$MODEL_NAME" \
          --benchmark "$bench" --instance_id "$iid" --subtask "$subtask" 2>&1); rc=$?
  else
    out=$("$py" -m evaluations.run --model_name "$MODEL_NAME" \
          --benchmark "$bench" --instance_id "$iid" 2>&1); rc=$?
  fi
  if [[ $rc -eq 0 && -n "$out" ]]; then
    PASS=$((PASS+1)); SUMMARY+=$(printf '  PASS  %s\n' "$label")
    echo "$out" | tail -8            # show the result JSON so you can eyeball the score
  else
    FAIL=$((FAIL+1)); SUMMARY+=$(printf '  FAIL  %s\n' "$label")
    echo "$out" | tail -6 | sed 's/^/    /'
  fi
}

# ── main env (18 benchmarks) ────────────────────────────────────────────────
check "$PYTHON" infobench       user_oriented_task_167
check "$PYTHON" aime25          0
check "$PYTHON" gpqa            0
check "$PYTHON" ifeval          0
check "$PYTHON" logiqa          0_strict-match
check "$PYTHON" humaneval_chat  0
check "$PYTHON" mbpp_chat       0
check "$PYTHON" mmlu_cot        abstract_algebra_0
check "$PYTHON" acp_gen         0                       acp_app_gen
check "$PYTHON" livecodebench   0                       codegeneration
check "$PYTHON" repobench       repobench_0             repobench_xff_python
check "$PYTHON" debugbench      "python3_condition error_the-kth-factor-of-n"
check "$PYTHON" lifbench        0_0_3                   onedoc-qa
check "$PYTHON" planbench       1                       task_3_plan_verification
check "$PYTHON" mmmu            validation_Accounting_1
check "$PYTHON" visualpuzzles   visualpuzzles_0         cot
check "$PYTHON" visualwebbench  web_caption_0           web_caption
BEIR_MAX_QUERIES=2 check "$PYTHON" beir_nfcorpus "NDCG@10"

# ── BFCL (19th): runs in its own venv; auto-detected at the install.sh path ──
BFCL_VENV="${BFCL_VENV:-evaluations/.bfcl-venv}"
if [[ -x "$BFCL_VENV/bin/python" ]]; then
  check "$BFCL_VENV/bin/python" bfcl parallel_multiple_0 non_live_parallel_multiple
else
  SUMMARY+=$(printf '  SKIP  bfcl (%s not found — run install.sh to create it)\n' "$BFCL_VENV")
fi

printf '\n===================== SUMMARY =====================\n%s' "$SUMMARY"
printf '\n  %d passed, %d failed\n' "$PASS" "$FAIL"
