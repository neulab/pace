#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate proxy

# Use local lm-evaluation-harness
export PYTHONPATH=/home/jiaruil5/proxy_bench/proxy-bench/lm-evaluation-harness:$PYTHONPATH
export HF_ALLOW_CODE_EVAL="1"

# RunPod API key for nemotron (set in ~/.bashrc)
# export RUNPOD_API_KEY="..."

# --- Already evaluated models (commented out) ---
# "azure/gpt-4o"
# "azure/gpt-5"
# "azure/o3"
# "azure/o4-mini"
# "azure/gpt-oss-120b"                    # GPT-OSS-120B
# "gemini/gemini-2.0-flash"
# "gemini/gemini-2.5-flash"
# "gemini/gemini-2.5-pro"
# "gemini/gemini-3-pro-preview"            # Gemini-3-Pro
# "gemini/gemini-3-flash-preview"          # Gemini-3-Flash
# "neulab/claude-opus-4-5-20251101"        # claude-opus-4-5
# "neulab/claude-sonnet-4-5-20250929"      # claude-sonnet-4-5
# "neulab/claude-sonnet-4-20250514"
# "neulab/kimi-k2-0711-preview"
# "azure/Llama-4-Maverick-17B-128E-Instruct-FP8"
# "neulab/llama4-scout-instruct"
# "neulab/qwen3-coder-480b-a35b-instruct"  # Qwen3-Coder-480B

# --- New models for OpenHands Index alignment ---
models=(
  # Confirmed working - Azure AI Foundry (free credits)
  "azure_ai/Kimi-K2.5"
  "azure_ai/Kimi-K2-Thinking"
  "azure_ai/DeepSeek-V3.2"
  # Confirmed working - Azure
  "azure/gpt-5.2"
  # Confirmed working - Fireworks AI
  "fireworks_ai/accounts/fireworks/models/minimax-m2p5"          # MiniMax-M2.5
  "fireworks_ai/accounts/fireworks/models/glm-4p7"              # GLM-4.7
  "fireworks_ai/accounts/fireworks/models/minimax-m2p1"          # MiniMax-M2.1
  "fireworks_ai/accounts/fireworks/models/glm-5"                 # GLM-5
  # Needs budget increase to work
  "anthropic/claude-opus-4-6"                                    # claude-opus-4-6
  "azure/gpt-5.2-codex"                                          # GPT-5.2-Codex
  # "fireworks_ai/accounts/fireworks/models/qwen3-next-80b-a3b-thinking"   # Qwen3 Next 80B A3B Thinking
  # "fireworks_ai/accounts/fireworks/models/qwen3-coder-30b-a3b-instruct"  # Qwen3 Coder 30B A3B Instruct
)

# RunPod models (separate loop below due to different base_url/api_key)
runpod_models=(
  "nvidia/nvidia-nemotron-3-nano-30b-a3b-bf16"   # NVIDIA Nemotron Nano 3 30B A3B
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

for model in "${models[@]}"; do
  (
    for benchmark in "${benchmarks[@]}"; do
      # Set model_args based on provider
      if [[ "$model" == azure/* ]]; then
        model_args="model=$model,base_url=https://cmu.litellm.ai/v1/chat/completions"
      else
        model_args="model=$model,base_url=https://cmu.litellm.ai/v1/chat/completions,disable_seed=true"
      fi

      # Set gen_kwargs - higher token limits for thinking/reasoning models
      if [[ "$model" == *"Thinking"* || "$model" == *"DeepSeek"* || "$model" == *"thinking"* || "$model" == *"deepseek"* ]]; then
        gen_kwargs="max_gen_toks=32768"
      else
        gen_kwargs="max_gen_toks=16384"
      fi

      # Cache path for model responses (uses model name with / replaced by __)
      cache_dir="/home/jiaruil5/proxy_bench/proxy_bench_data/.cache"
      mkdir -p "$cache_dir"
      safe_model_name="${model//\//__}"
      cache_path="$cache_dir/${safe_model_name}"

      echo "[$(date)] Starting $model on $benchmark"

      python -m lm_eval run \
        --model openai-chat-completions \
        --model_args $model_args \
        --tasks $benchmark \
        --output_path /home/jiaruil5/proxy_bench/proxy_bench_data/$benchmark \
        --apply_chat_template \
        --confirm_run_unsafe_code \
        --log_samples \
        --gen_kwargs $gen_kwargs \
        --use_cache $cache_path

      echo "[$(date)] Finished $model on $benchmark (exit code: $?)"
    done
  ) &
done

# RunPod models (different base_url and api_key)
for model in "${runpod_models[@]}"; do
  (
    for benchmark in "${benchmarks[@]}"; do
      model_args="model=$model,base_url=https://api.runpod.ai/v2/c35bkyozx7erir/openai/v1/chat/completions,disable_seed=true"
      gen_kwargs="max_gen_toks=16384"

      cache_dir="/home/jiaruil5/proxy_bench/proxy_bench_data/.cache"
      mkdir -p "$cache_dir"
      safe_model_name="${model//\//__}"
      cache_path="$cache_dir/${safe_model_name}"

      echo "[$(date)] Starting $model on $benchmark (RunPod)"

      OPENAI_API_KEY="$RUNPOD_API_KEY" python -m lm_eval run \
        --model openai-chat-completions \
        --model_args $model_args \
        --tasks $benchmark \
        --output_path /home/jiaruil5/proxy_bench/proxy_bench_data/$benchmark \
        --apply_chat_template \
        --confirm_run_unsafe_code \
        --log_samples \
        --gen_kwargs $gen_kwargs \
        --use_cache $cache_path

      echo "[$(date)] Finished $model on $benchmark (exit code: $?)"
    done
  ) &
done

# Wait for all parallel jobs to complete
wait
echo "All model evaluations completed."
