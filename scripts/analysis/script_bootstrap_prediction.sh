#!/bin/bash
# Bootstrap prediction pipeline for proxy-bench
#
# Goal: Find non-agentic benchmark instances that best PREDICT agentic benchmark performance
# using MSE of linear regression instead of correlation
#
# Agentic benchmarks (targets): commit0, gaia, swebench, swebench_bash_only, swebench_multimodal, swtbench
# Non-agentic benchmarks (sources): all other supported benchmarks

set -x  # Print commands as they execute

# Define benchmark groups
# AGENTIC_BENCHMARKS="commit0 gaia swebench swebench_multimodal swtbench"
AGENTIC_BENCHMARKS="swebench"
NON_AGENTIC_BENCHMARKS="acp_gen aime25 beir_nfcorpus bfcl gpqa humaneval_chat ifeval infobench lifbench livecodebench logiqa mbpp_chat mmmu repobench visualpuzzles"

# Model splits for training and evaluation
# Training models: used for greedy selection of proxy instances (minimizing MSE)
TRAIN_MODELS="GPT-5.2-Codex MiniMax-M2.1 Gemini-3-Pro-Preview Claude-4.6-Opus Kimi-K2 Kimi-K2.5 Nemotron-3-Nano Qwen3-Coder-480B-A35B-Instruct GPT-5.2 Claude-4.5-Opus Gemini-3-Flash-Preview DeepSeek-V3.2 Claude-4.5-Sonnet MiniMax-M2.5 GLM-4.7"

# Evaluation models: used to test generalization of selected proxy instances
EVAL_MODELS="GPT-5.2-Codex MiniMax-M2.1"

# Common parameters
TRAIN_PCT=0.8       # 80% for training
EVAL_PCT=0.2        # 20% for evaluation
BOOT_SOURCE_K=1000
BOOT_TARGET_K=100
K_SOURCE=200
N_OUTER=1
N_RESTARTS=1

# Output directory
OUTPUT_DIR="../../analysis/agentic_proxy_prediction"

# Run 1: All non-agentic sources -> All agentic targets (merged)
# echo "========================================"
# echo "Running: All non-agentic -> All agentic (merged) [PREDICTION]"
# echo "========================================"
# python bootstrap_prediction.py \
#     --sources ${NON_AGENTIC_BENCHMARKS} \
#     --targets ${AGENTIC_BENCHMARKS} \
#     --train_models ${TRAIN_MODELS} \
#     --eval_models ${EVAL_MODELS} \
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
    echo "Running: All non-agentic -> ${target} [PREDICTION/MSE]"
    echo "========================================"
    python bootstrap_prediction.py \
        --sources ${NON_AGENTIC_BENCHMARKS} \
        --targets ${target} \
        --train_models ${TRAIN_MODELS} \
        --eval_models ${EVAL_MODELS} \
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
echo "All prediction runs completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"
