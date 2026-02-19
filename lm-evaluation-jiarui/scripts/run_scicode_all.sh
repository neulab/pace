#!/bin/bash
# Run SciCode benchmark for all models
# Usage: ./run_scicode_all.sh [--limit N] [--split validation|test] [--with_background] [--evaluate] [--h5py_file_path PATH] [--max_parallel N]

set -e

export HF_TOKEN=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

# Default values
LIMIT=""
SPLIT="test"
WITH_BACKGROUND=""
OUTPUT_BASE="/home/jiaruil5/proxy_bench/proxy_bench_data/scicode"
EVALUATE="--evaluate"
H5PY_FILE_PATH="--h5py_file_path /home/jiaruil5/proxy_bench/proxy_bench_data/code/scicode/test_data.h5"
MAX_PARALLEL=4  # Limit concurrent jobs to avoid rate limits

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --with_background)
            WITH_BACKGROUND="--with_background"
            shift
            ;;
        --output_path)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --evaluate)
            EVALUATE="--evaluate"
            shift
            ;;
        --h5py_file_path)
            H5PY_FILE_PATH="--h5py_file_path $2"
            shift 2
            ;;
        --max_parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

models=(
  "azure/gpt-4o"
  "azure/gpt-5"
  "azure/o3"
  "azure/o4-mini"
  "azure/gpt-oss-120b"
  "gemini/gemini-2.0-flash"
  "gemini/gemini-2.5-flash"
  "gemini/gemini-2.5-pro"
  "gemini/gemini-3-pro-preview"
  "gemini/gemini-3-flash-preview"
  "neulab/claude-opus-4-5-20251101"
  "neulab/claude-sonnet-4-5-20250929"
  "neulab/claude-sonnet-4-20250514"
  "neulab/kimi-k2-0711-preview"
  "azure/Llama-4-Maverick-17B-128E-Instruct-FP8"
  # "neulab/qwen3-coder-480b-a35b-instruct"
)

BG_SUFFIX="without_background"
if [[ -n "$WITH_BACKGROUND" ]]; then
  BG_SUFFIX="with_background"
fi

EVAL_STATUS="generation only"
if [[ -n "$EVALUATE" ]]; then
  EVAL_STATUS="with evaluation"
fi

echo "Running SciCode benchmark (${SPLIT}/${BG_SUFFIX}) for ${#models[@]} models (${EVAL_STATUS})..."
echo "Output path: ${OUTPUT_BASE}/${SPLIT}_${BG_SUFFIX}"
echo "Max parallel jobs: ${MAX_PARALLEL}"
echo ""

running=0
for model in "${models[@]}"; do
  (
    # Set max_tokens based on model
    if [[ "$model" == "azure/gpt-4o" ]]; then
      MAX_TOKENS=16384
    else
      MAX_TOKENS=32768
    fi

    echo "Starting: ${model} (max_tokens=${MAX_TOKENS})"
    python run_scicode.py \
      --model "$model" \
      --split "$SPLIT" \
      --output_path "${OUTPUT_BASE}/${SPLIT}_${BG_SUFFIX}" \
      --apply_chat_template \
      --log_samples \
      --max_tokens $MAX_TOKENS \
      $WITH_BACKGROUND \
      $EVALUATE \
      $H5PY_FILE_PATH \
      $LIMIT
    echo "Completed: ${model}"
  ) &

  # Limit parallel jobs
  ((++running))
  if [[ $running -ge $MAX_PARALLEL ]]; then
    wait -n || true  # Wait for any one job to finish, ignore failures
    ((running--)) || true
  fi
done

# Wait for remaining jobs to complete
wait || true
echo "All SciCode evaluations completed."
