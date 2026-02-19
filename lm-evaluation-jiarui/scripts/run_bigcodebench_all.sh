#!/bin/bash
# Run BigCodeBench benchmark for all models
# Usage: ./run_bigcodebench_all.sh [--limit N] [--split instruct|complete] [--subset full|hard]

set -e


# Default values
LIMIT=""
SPLIT="instruct"
SUBSET="full"
OUTPUT_BASE="/home/jiaruil5/proxy_bench/proxy_bench_data/bigcodebench"

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
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_BASE="$2"
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

echo "Running BigCodeBench benchmark (${SPLIT}/${SUBSET}) for ${#models[@]} models..."
echo "Output path: ${OUTPUT_BASE}/${SPLIT}_${SUBSET}"
echo ""

for model in "${models[@]}"; do
  (
    # Set max_tokens based on model
    if [[ "$model" == "azure/gpt-4o" ]]; then
      MAX_TOKENS=16384
    else
      MAX_TOKENS=32768
    fi

    echo "Starting: ${model} (max_tokens=${MAX_TOKENS})"
    python run_bigcodebench.py \
      --model "$model" \
      --split "$SPLIT" \
      --subset "$SUBSET" \
      --output_path "${OUTPUT_BASE}/${SPLIT}_${SUBSET}" \
      --apply_chat_template \
      --log_samples \
      --max_tokens $MAX_TOKENS \
      --n_samples 1 \
      --evaluate \
      $LIMIT
    echo "Completed: ${model}"
  ) &
done

# Wait for all parallel jobs to complete
wait
echo "All BigCodeBench evaluations completed."
