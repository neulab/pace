#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate proxy

# Use local lm-evaluation-harness
export PYTHONPATH=/home/jiaruil5/proxy_bench/proxy-bench/lm-evaluation-harness:$PYTHONPATH
export HF_ALLOW_CODE_EVAL="1"

# RunPod API key for nemotron (set in ~/.bashrc)
# export RUNPOD_API_KEY="..."

# --- New models via CMU LiteLLM proxy ---
models=(
  # Azure AI Foundry (free credits)
  "azure_ai/Kimi-K2.5"
  "azure_ai/Kimi-K2-Thinking"
  "azure_ai/DeepSeek-V3.2"
  # Azure
  "azure/gpt-5.2"
  # Fireworks AI
  "fireworks_ai/accounts/fireworks/models/minimax-m2p5"
  "fireworks_ai/accounts/fireworks/models/glm-4p7"
  "fireworks_ai/accounts/fireworks/models/minimax-m2p1"
  "fireworks_ai/accounts/fireworks/models/glm-5"
  # Budget-dependent
  "anthropic/claude-opus-4-6"
  "azure/gpt-5.2-codex"
  # "fireworks_ai/accounts/fireworks/models/qwen3-next-80b-a3b-thinking"
  # "fireworks_ai/accounts/fireworks/models/qwen3-coder-30b-a3b-instruct"
)

benchmarks=(
  "ifeval"
  "acp_gen_2shot"
  "mbpp_chat"
  "humaneval_chat"
  "gpqa_diamond_cot_zeroshot,gpqa_main_cot_zeroshot,gpqa_extended_cot_zeroshot"
  "aime25"
  "logiqa_cot_zeroshot"
)

# CMU proxy models
for model in "${models[@]}"; do
  (
    for benchmark in "${benchmarks[@]}"; do
      if [[ "$model" == azure/* ]]; then
        model_args="model=$model,base_url=https://cmu.litellm.ai/v1/chat/completions"
      else
        model_args="model=$model,base_url=https://cmu.litellm.ai/v1/chat/completions,disable_seed=true"
      fi

      if [[ "$model" == *"Thinking"* || "$model" == *"DeepSeek"* || "$model" == *"thinking"* || "$model" == *"deepseek"* ]]; then
        gen_kwargs="max_gen_toks=32768"
      else
        gen_kwargs="max_gen_toks=16384"
      fi

      echo "=== Testing $model on $benchmark ==="

      python -m lm_eval run \
        --model openai-chat-completions \
        --model_args $model_args \
        --tasks $benchmark \
        --output_path /home/jiaruil5/proxy_bench/proxy_bench_data/test/$benchmark \
        --apply_chat_template \
        --limit 2 \
        --confirm_run_unsafe_code \
        --log_samples \
        --gen_kwargs $gen_kwargs 2>&1

      if [ $? -eq 0 ]; then
        echo "=== SUCCESS: $model on $benchmark ==="
      else
        echo "=== FAILED: $model on $benchmark ==="
      fi
    done
  ) &
done

# RunPod nemotron (different base_url and api_key)
(
  model="nvidia/nvidia-nemotron-3-nano-30b-a3b-bf16"
  for benchmark in "${benchmarks[@]}"; do
    model_args="model=$model,base_url=https://api.runpod.ai/v2/c35bkyozx7erir/openai/v1/chat/completions,disable_seed=true"
    gen_kwargs="max_gen_toks=16384"

    echo "=== Testing $model on $benchmark (RunPod) ==="

    OPENAI_API_KEY="$RUNPOD_API_KEY" python -m lm_eval run \
      --model openai-chat-completions \
      --model_args $model_args \
      --tasks $benchmark \
      --output_path /home/jiaruil5/proxy_bench/proxy_bench_data/test/$benchmark \
      --apply_chat_template \
      --limit 2 \
      --confirm_run_unsafe_code \
      --log_samples \
      --gen_kwargs $gen_kwargs 2>&1

    if [ $? -eq 0 ]; then
      echo "=== SUCCESS: $model on $benchmark (RunPod) ==="
    else
      echo "=== FAILED: $model on $benchmark (RunPod) ==="
    fi
  done
) &

wait
echo "All model test evaluations completed."
