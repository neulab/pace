#!/bin/bash
# Bootstrap analysis pipeline for proxy-bench
#
# Goal: Find non-agentic benchmark instances that best predict agentic benchmark performance
#
# Agentic benchmarks (targets): commit0, gaia, swebench, swebench_bash_only, swebench_multimodal, swtbench
# Non-agentic benchmarks (sources): all other supported benchmarks

set -x  # Print commands as they execute

# Define benchmark groups
# AGENTIC_BENCHMARKS="commit0 gaia swebench swebench_multimodal swtbench"
AGENTIC_BENCHMARKS="commit0"
# NON_AGENTIC_BENCHMARKS="acp_gen aime25 beir_nfcorpus bfcl gpqa humaneval_chat ifeval infobench livecodebench logiqa mbpp_chat"
NON_AGENTIC_BENCHMARKS="bfcl"

# Common parameters
TRAIN_PCT=0.8       # 80% for training
EVAL_PCT=0.2        # 20% for evaluation
BOOT_SOURCE_K=1000
BOOT_TARGET_K=100
K_SOURCE=200
N_OUTER=1
N_RESTARTS=1

# Output directory
OUTPUT_DIR="../../analysis/agentic_proxy"

# Run 1: All non-agentic sources -> All agentic targets (merged)
# echo "========================================"
# echo "Running: All non-agentic -> All agentic (merged)"
# echo "========================================"
# python bootstrap_all.py \
#     --sources ${NON_AGENTIC_BENCHMARKS} \
#     --targets ${AGENTIC_BENCHMARKS} \
#     --train_pct ${TRAIN_PCT} \
#     --eval_pct ${EVAL_PCT} \
#     --boot_source_k ${BOOT_SOURCE_K} \
#     --boot_target_k ${BOOT_TARGET_K} \
#     --k_source ${K_SOURCE} \
#     --n_outer ${N_OUTER} \
#     --n_restarts ${N_RESTARTS} \
#     --output_dir "${OUTPUT_DIR}/all_to_all"

# Run 2: All non-agentic sources -> Each agentic target individually
for target in ${AGENTIC_BENCHMARKS}; do
    echo "========================================"
    echo "Running: All non-agentic -> ${target}"
    echo "========================================"
    python bootstrap_all.py \
        --sources ${NON_AGENTIC_BENCHMARKS} \
        --targets ${target} \
        --train_pct ${TRAIN_PCT} \
        --eval_pct ${EVAL_PCT} \
        --boot_source_k ${BOOT_SOURCE_K} \
        --boot_target_k ${BOOT_TARGET_K} \
        --k_source ${K_SOURCE} \
        --n_outer ${N_OUTER} \
        --n_restarts ${N_RESTARTS} \
        --output_dir "${OUTPUT_DIR}/to_${target}"
done

echo "========================================"
echo "All runs completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"
