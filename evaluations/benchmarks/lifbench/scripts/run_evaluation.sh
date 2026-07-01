#!/bin/bash
# Run evaluation for all models

python ./evaluation/Evaluator.py --data_dir './data' get_all --models '[
    # "Llama-4-Maverick-17B-128E-Instruct-FP8",
    # "claude-opus-4-5-20251101",
    # "claude-sonnet-4-20250514",
    # "claude-sonnet-4-5-20250929",
    # "gemini-2.0-flash",
    # "gemini-2.5-flash",
    # "gemini-2.5-pro",
    # "gemini-3-pro-preview",
    # "gemini-3-flash-preview",
    # "gpt-4o",
    # "gpt-5",
    # "gpt-5.2",
    # "gpt-oss-120b",
    # "kimi-k2-0711-preview",
    # "o3",
    # "o4-mini",
    # "qwen3-coder-480b-a35b-instruct",
    # "minimax-m2p1",
    # "minimax-m2p5",
    # "glm-4p7",
    # "nvidia-nemotron-3-nano-30b-a3b-bf16",
    # "glm-5",
    # "DeepSeek-V3.2",
    # "kimi-k2-thinking",
    # "claude-opus-4-6",
    # "gpt-5.2-codex",
    "Kimi-K2.5"
]'
