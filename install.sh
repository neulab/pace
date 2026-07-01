#!/usr/bin/env bash
# =============================================================================
# evaluations/install.sh — ONE command to install everything for all 19 benchmarks.
#
#   bash evaluations/install.sh
#
# It sets up two environments because BFCL hard-conflicts with the rest on
# tree-sitter (BFCL needs 0.21.x, repobench/codebleu need 0.23.x):
#   1. the CURRENT python  → pacebench + 18 benchmarks (everything except BFCL)
#   2. evaluations/.bfcl-venv → BFCL only
#
# Override the interpreter with PYTHON=... and the BFCL venv path with BFCL_VENV=...
# The editable installs use --no-deps to skip GPU-only extras (e.g. vllm) that
# are not needed when generating via an API endpoint.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
PY="${PYTHON:-python}"
BFCL_VENV="${BFCL_VENV:-evaluations/.bfcl-venv}"

echo "== [1/2] main env: pacebench + 18 benchmarks =="
"$PY" -m pip install -r evaluations/requirements-main.txt
# codebleu 0.7.0 pins an older tree-sitter in metadata but works with 0.23.x at
# runtime (repobench); install this group with --no-deps to bypass the resolver.
"$PY" -m pip install --no-deps codebleu==0.7.0 tree-sitter==0.23.2 \
    tree-sitter-python==0.23.6 tree-sitter-java==0.23.5
# vendored packages (their heavy deps are already pinned above; --no-deps skips vllm etc.)
"$PY" -m pip install -e evaluations/benchmarks/lm-evaluation-harness --no-deps   # lm_eval: 8 benchmarks
"$PY" -m pip install -e evaluations/benchmarks/livecodebench          --no-deps   # lcb_runner
"$PY" -m pip install -e evaluations/benchmarks/lmms-eval              --no-deps   # lmms_eval: mmmu, visualpuzzles

echo "== [2/2] BFCL venv: $BFCL_VENV =="
"$PY" -m venv "$BFCL_VENV"
"$BFCL_VENV/bin/python" -m pip install -U pip
"$BFCL_VENV/bin/python" -m pip install -e \
    evaluations/benchmarks/BFCL/gorilla/berkeley-function-call-leaderboard --no-deps
"$BFCL_VENV/bin/python" -m pip install \
    "openai>=1.86.0" "tree-sitter==0.21.3" "tree-sitter-java==0.21.0" \
    "tree-sitter-javascript==0.21.4" pydantic numpy pandas tqdm tabulate \
    huggingface_hub tenacity overrides python-dotenv

cat <<EOF

Done.
  - 18 benchmarks: run with the current python, e.g.  bash evaluations/script.sh
  - BFCL: run with the venv python, e.g.
        PYTHON=$BFCL_VENV/bin/python bash evaluations/script.sh
    (then uncomment the bfcl line in evaluations/script.sh)

Reminders:
  * export HF_HUB_ENABLE_HF_TRANSFER=0   (HF dataset downloads)
  * planbench needs results/raw_results/planbench
  * debugbench grading needs LeetCode cookies in
    evaluations/benchmarks/debugbench/evaluation/keys.json
EOF
