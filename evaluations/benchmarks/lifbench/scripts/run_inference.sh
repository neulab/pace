#!/bin/bash
# Run inference for multiple models on all 11 tasks

models=(
    # "azure_ai/gpt-4o"
    # "azure_ai/gpt-4o-mini"
    # "azure_ai/gpt-5"
    # "azure_ai/gpt-5.2"
    # "azure_ai/gpt-oss-120b"
    # "azure_ai/Llama-4-Maverick-17B-128E-Instruct-FP8"
    # "azure_ai/o3"
    # "azure_ai/o4-mini"
    # "azure_ai/DeepSeek-V3.2"
    # "azure_ai/gpt-5.2-codex"
    "azure_ai/Kimi-K2.5"
    # "anthropic/claude-opus-4-6"
    # "anthropic/claude-opus-4-5-20251101"
    # "anthropic/claude-sonnet-4-5-20250929"
    # "anthropic/claude-sonnet-4-20250514"
    # "anthropic/kimi-k2-0711-preview"
    # "gemini/gemini-2.0-flash"
    # "gemini/gemini-2.5-flash"
    # "gemini/gemini-2.5-pro"
    # "gemini/gemini-3-pro-preview"
    # "fireworks_ai/accounts/fireworks/models/glm-4p7"
    # "fireworks_ai/accounts/fireworks/models/glm-5"
    # "fireworks_ai/accounts/fireworks/models/minimax-m2p1"
    # "fireworks_ai/accounts/fireworks/models/nemotron-nano-3-30b-a3b"
    # "fireworks_ai/accounts/fireworks/models/qwen3-coder-30b-a3b-instruct"
    # "fireworks_ai/accounts/fireworks/models/qwen3-coder-480b-a35b-instruct"
    # "fireworks/accounts/fireworks/models/qwen3-next-80b-a3b-thinking"
)


task_filters=(
    "list-offset_query_element"
    "list-offset_query_id"
    "list-multi_query_id"
    "list-single_query_id"
    "list-blur_offset_query_element"
    "list-blur_offset_query_id"
    "multidoc-batch_label"
    "multidoc-find_dup_text"
    "onedoc-extract"
    "onedoc-qa"
    "onedoc-repeat"
)
for task_filter in "${task_filters[@]}"; do
    echo "=========================================="
    echo "Running inference for: $task_filter"
    echo "=========================================="
    for model in "${models[@]}"; do
        echo "=========================================="
        echo "Running inference for: $model"
        echo "=========================================="
        python ./evaluation/ApiInference.py \
            --model_name "$model" \
            --benchmark_base_dir './data' \
            --task_filter "$task_filter" 
    done
done
