#!/bin/bash

export HF_ALLOW_CODE_EVAL="1"

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
  "neulab/qwen3-coder-480b-a35b-instruct"
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
      # Set model_args based on whether model supports seed parameter
      # Azure/OpenAI models support seed, others (gemini, neulab) don't
      if [[ "$model" == azure/* ]]; then
        model_args="model=$model,base_url=https://cmu.litellm.ai/v1/chat/completions"
      else
        model_args="model=$model,base_url=https://cmu.litellm.ai/v1/chat/completions,disable_seed=true"
      fi

      # Set gen_kwargs for specific model + benchmark combinations
      # gpt-4o + aime25 needs higher max_gen_toks for long reasoning
      if [[ "$model" == "gemini/gemini-2.5-pro" ]]; then
        gen_kwargs="max_gen_toks=32768"
      elif [[ "$model" == "azure/gpt-4o" && "$benchmark" == "aime25" ]]; then
        gen_kwargs="max_gen_toks=16384"
      elif [[ "$benchmark" == "ifeval" || "$benchmark" == "acp_gen_2shot" || "$benchmark" == "gpqa_diamond_cot_zeroshot,gpqa_main_cot_zeroshot,gpqa_extended_cot_zeroshot" || "$benchmark" == "mbpp_chat" || "$benchmark" == "humaneval_chat" || "$benchmark" == "logiqa_cot_zeroshot" ]]; then
        gen_kwargs="max_gen_toks=16384"
      else
        gen_kwargs=""
      fi

      # test by running the first 2 samples
      if [[ -n "$gen_kwargs" ]]; then
        lm-eval run \
          --model openai-chat-completions \
          --model_args $model_args \
          --tasks $benchmark \
          --output_path /home/jiaruil5/proxy_bench/proxy_bench_data/test/$benchmark \
          --apply_chat_template \
          --limit 2 \
          --confirm_run_unsafe_code \
          --log_samples \
          --gen_kwargs $gen_kwargs
      else
        lm-eval run \
          --model openai-chat-completions \
          --model_args $model_args \
          --tasks $benchmark \
          --output_path /home/jiaruil5/proxy_bench/proxy_bench_data/test/$benchmark \
          --apply_chat_template \
          --limit 2 \
          --confirm_run_unsafe_code \
          --log_samples
      fi
    done
  ) &
done

# Wait for all parallel jobs to complete
wait
echo "All model evaluations completed."
