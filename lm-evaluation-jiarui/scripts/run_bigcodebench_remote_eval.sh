#!/bin/bash
# Run remote BigCodeBench evaluation for all existing samples
# Usage: ./run_bigcodebench_remote_eval.sh [--input_path PATH] [--split instruct|complete] [--subset full|hard] [--timeout SECONDS]

set -e

# Default values
INPUT_BASE="/home/jiaruil5/proxy_bench/proxy_bench_data/bigcodebench/instruct_full"
SPLIT="instruct"
SUBSET="full"
TIMEOUT=3600  # 1 hour timeout for remote evaluation

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_path)
            INPUT_BASE="$2"
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
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running BigCodeBench remote evaluation"
echo "Input path: ${INPUT_BASE}"
echo "Split: ${SPLIT}"
echo "Subset: ${SUBSET}"
echo "Timeout: ${TIMEOUT}s ($((TIMEOUT / 60)) minutes)"
echo ""

# Find all model directories
model_dirs=$(find "${INPUT_BASE}" -maxdepth 1 -type d -name "*__*" | sort)

if [[ -z "$model_dirs" ]]; then
    echo "No model directories found in ${INPUT_BASE}"
    exit 1
fi

echo "Found model directories:"
echo "$model_dirs" | while read dir; do
    echo "  - $(basename "$dir")"
done
echo ""

# Process each model directory
for model_dir in $model_dirs; do
    model_name=$(basename "$model_dir")

    # Find the samples file in bigcodebench_format
    samples_file=$(find "${model_dir}/bigcodebench_format" -name "*--bigcodebench-${SUBSET}-${SPLIT}--0.0-1.jsonl" 2>/dev/null | head -1)

    if [[ -z "$samples_file" ]]; then
        echo "SKIP: No samples file found for ${model_name}"
        continue
    fi

    # Check if remote evaluation already exists
    remote_dir="${model_dir}/bigcodebench_format/bigcodebench_format_remote"
    if [[ -d "$remote_dir" ]] && [[ -n "$(ls -A "$remote_dir" 2>/dev/null)" ]]; then
        echo "SKIP: Remote evaluation already exists for ${model_name}"
        continue
    fi

    echo "Evaluating: ${model_name}"
    echo "  Samples: ${samples_file}"

    # Run remote evaluation sequentially (to avoid overwhelming Gradio endpoint)
    cd "${SCRIPT_DIR}"
    python run_bigcodebench.py \
        --split "${SPLIT}" \
        --subset "${SUBSET}" \
        --remote_timeout "${TIMEOUT}" \
        --evaluate_remote_only "${samples_file}" || {
        echo "ERROR: Failed to evaluate ${model_name}"
    }
    echo "Completed: ${model_name}"
    echo ""
done

echo "All remote evaluations completed."
