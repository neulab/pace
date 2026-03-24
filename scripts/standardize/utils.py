#!/usr/bin/env python3
"""
Shared utilities for standardizing eval score outputs across benchmarks.

This module contains helpers used by the standardize_{benchmark}.py scripts,
including:
- Model name parsing and normalization
- File processing for different benchmark formats
- CSV writing helpers
- Path resolution relative to the project root
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# =============================================================================
# MODEL NAME LOOKUP TABLE
# Maps eval_scores model names -> normalized names (swebench style)
# =============================================================================

MODEL_NAME_LOOKUP: Dict[str, str] = {
    # Azure/OpenAI models
    "azure--gpt-4o": "GPT-4o",
    "azure--gpt-5": "GPT-5",
    "azure--gpt-5-mini": "GPT-5-mini",
    "azure--gpt-5-nano": "GPT-5-nano",
    "azure--gpt-oss-120b": "GPT-OSS-120B",
    "azure--o3": "o3",
    "azure--o4-mini": "o4-mini",
    "azure--Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama-4-Maverick-Instruct",

    # Gemini models
    "gemini--gemini-2.0-flash": "Gemini-2.0-Flash",
    "gemini_gemini-2.0-flash": "Gemini-2.0-Flash",
    "gemini--gemini-2.0-flash-exp": "Gemini-2.0-Flash",
    "gemini--gemini-2.5-flash": "Gemini-2.5-Flash",
    "gemini--gemini-2.5-pro": "Gemini-2.5-Pro",
    "gemini--gemini-3-flash-preview": "Gemini-3-Flash-Preview",
    "gemini--gemini-3-pro-preview": "Gemini-3-Pro-Preview",

    # Anthropic Claude models (neulab)
    "neulab--claude-opus-4-5-20251101": "Claude-4.5-Opus",
    "neulab_claude-opus-4-5-20251101": "Claude-4.5-Opus",
    "neulab--claude-sonnet-4-20250514": "Claude-4-Sonnet",
    "anthropic_claude-sonnet-4-20250514": "Claude-4-Sonnet",
    "neulab_claude-sonnet-4-5-20250929": "Claude-4.5-Sonnet",
    "neulab--claude-sonnet-4-5-20250929": "Claude-4.5-Sonnet",

    # Other models (neulab)
    "neulab--kimi-k2-0711-preview": "Kimi-K2-Preview",
    "neulab_kimi-k2-0711-preview": "Kimi-K2-Preview",
    "neulab--qwen3-coder-480b-a35b-instruct": "Qwen3-Coder-480B-A35B-Instruct",
    "openai_qwen_qwen3-coder": "Qwen3-Coder-480B-A35B-Instruct",
    "neulab--gpt-4.1-mini-2025-04-14": "GPT-4.1-mini",
    "neulab--gpt-4o-2024-08-06": "GPT-4o",
    "neulab--llama4-maverick-instruct": "Llama-4-Maverick-Instruct",
    "Llama-4-Maverick-Instruct": "Llama-4-Maverick-Instruct",
    "neulab--llama4-scout-instruct": "Llama-4-Scout-Instruct",
    "neulab--qwen3-235b-a22b": "Qwen3-235B-A22B",
    "neulab--gpt-oss-120b": "GPT-OSS-120B",
    "neulab_gpt-oss-120b": "GPT-OSS-120B",

    # Standalone raw names (appear in some datasets)
    "gpt-oss-120b": "GPT-OSS-120B",
    "GPT-OSS-120B": "GPT-OSS-120B",
    "Gemini-3.1-Pro": "Gemini-3.1-Pro",
    "gemini-3-pro-preview": "Gemini-3-Pro-Preview",
    "Gemini-3-Pro-Preview": "Gemini-3-Pro-Preview",
    "Gemini-3-Pro": "Gemini-3-Pro-Preview",
    "Gemini-3-Flash": "Gemini-3-Flash-Preview",
    "Gemini-3-Flash-Preview": "Gemini-3-Flash-Preview",
    "Nemotron-3-Nano": "Nemotron-3-Nano",
    "nemotron-3-nano-30b-a3b": "Nemotron-3-Nano",
    "Nemotron-Nano-3-30B-A3B": "Nemotron-3-Nano",
    "Nvidia-Nemotron-3-Nano-30B-A3B": "Nemotron-3-Nano",
    "nvidia__nvidia-nemotron-3-nano-30b-a3b-bf16": "Nemotron-3-Nano",
    "gemini2.5flash": "Gemini-2.5-Flash",
    "gemini2.5pro": "Gemini-2.5-Pro",
    "gemini-2.0-flash": "Gemini-2.0-Flash",
    "Gemini-2.0-Flash": "Gemini-2.0-Flash",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
    "Gemini-2.5-Flash": "Gemini-2.5-Flash",
    "gemini-2.5-pro": "Gemini-2.5-Pro",
    "Gemini-2.5-Pro": "Gemini-2.5-Pro",
    "gpt4o": "GPT-4o",
    "gpt-4o": "GPT-4o",
    "GPT-4o": "GPT-4o",
    "GPT-4.1-mini": "GPT-4.1-mini",
    "gpt-5": "GPT-5",
    "openai_gpt-5": "GPT-5",
    "GPT-5": "GPT-5",
    "GPT-5-mini": "GPT-5-mini",
    "GPT-5-nano": "GPT-5-nano",
    "gpt-5.2": "GPT-5.2",
    "GPT-5.2": "GPT-5.2",
    "azure__gpt-5.2": "GPT-5.2",
    "GPT-5.2-Codex": "GPT-5.2-Codex",
    "azure__gpt-5.2-codex": "GPT-5.2-Codex",
    "o3": "o3",
    "o4-mini": "o4-mini",
    "Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama-4-Maverick-Instruct",
    "claude_opus": "Claude-4-Opus",
    "Claude-Sonnet-4": "Claude-4-Sonnet",
    "claude-sonnet-4-20250514": "Claude-4-Sonnet",
    "Claude-Sonnet-4.5": "Claude-4.5-Sonnet",
    "sonnet4.5": "Claude-4.5-Sonnet",
    "claude-sonnet-4-5-20250929": "Claude-4.5-Sonnet",
    "Claude-4.5-Sonnet": "Claude-4.5-Sonnet",
    "Claude-4.5-Opus": "Claude-4.5-Opus",
    "claude-sonnet-4-5": "Claude-4.5-Sonnet",
    "claude-sonnet-4-6": "Claude-4.6-Sonnet",
    "sonnet092925": "Claude-4.5-Sonnet",
    "kimik": "Kimi-K2",
    "kimi-k2-0711-preview": "Kimi-K2-Preview",
    "Kimi-K2-Thinking": "Kimi-K2",
    "fireworks_ai_accounts_fireworks_models_kimi-k2-thinking": "Kimi-K2",
    "Kimi-K2-Instruct": "Kimi-K2-Instruct",
    "Kimi-K2": "Kimi-K2",
    "Kimi-K2.5": "Kimi-K2.5",
    "azure_ai__Kimi-K2.5": "Kimi-K2.5",
    "fireworks_ai_accounts_fireworks_models_kimi-k2p5": "Kimi-K2.5",
    "DeepSeek-V3.2": "DeepSeek-V3.2",
    "azure_ai__DeepSeek-V3.2": "DeepSeek-V3.2",
    "deepseek-v3p2": "DeepSeek-V3.2",
    "fireworks_ai_accounts_fireworks_models_deepseek-v3p2": "DeepSeek-V3.2",
    "qwen3-coder-480b-a35b-instruct": "Qwen3-Coder-480B-A35B-Instruct",
    "Qwen3-Coder-480B-A35B-Instruct": "Qwen3-Coder-480B-A35B-Instruct",
    "Qwen3-Coder-480B": "Qwen3-Coder-480B-A35B-Instruct",
    "Qwen3-Coder-Next": "Qwen3-Coder-Next",
    "Qwen3.5-Flash": "Qwen3.5-Flash",
    "Claude-Opus-4.5": "Claude-4.5-Opus",
    "claude-opus-4-5": "Claude-4.5-Opus",
    "claude-opus-4-5-20251101": "Claude-4.5-Opus",
    "claude-opus-4-6": "Claude-4.6-Opus",
    "Claude-4.6-Opus": "Claude-4.6-Opus",
    "Claude-Opus-4.6": "Claude-4.6-Opus",
    "claude_opus_4.6": "Claude-4.6-Opus",
    "anthropic__claude-opus-4-6": "Claude-4.6-Opus",
    "Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B-Instruct",
    "MiniMax-M2.5": "MiniMax-M2.5",
    "fireworks_ai__accounts__fireworks__models__minimax-m2p5": "MiniMax-M2.5",
    "fireworks_ai_accounts_fireworks_models_minimax-m2p5": "MiniMax-M2.5",
    "MiniMax-M2.1": "MiniMax-M2.1",
    "fireworks_ai_accounts_fireworks_models_minimax-m2p1": "MiniMax-M2.1",
    "fireworks_ai__accounts__fireworks__models__minimax-m2p1": "MiniMax-M2.1",
    "GLM-4.7": "GLM-4.7",
    "fireworks_ai__accounts__fireworks__models__glm-4p7": "GLM-4.7",
    "fireworks_ai_accounts_fireworks_models_glm-4p7": "GLM-4.7",
    "GLM-5": "GLM-5",
    "fireworks_ai__accounts__fireworks__models__glm-5": "GLM-5",
    "DeepSeek-V3.2-Reasoner": "DeepSeek-V3.2",

    "azure__gpt-5.2": "GPT-5.2",
    "azure_ai__grok-4-fast-non-reasoning": "Grok-4-Fast",
    "azure_ai__grok-4-fast-reasoning": "Grok-4-Fast-Reasoning",
    "Grok-4-Fast": "Grok-4-Fast",
    "Grok-4-Fast-Reasoning": "Grok-4-Fast-Reasoning",
    "gemini__gemini-2.5-flash": "Gemini-2.5-Flash",
    "gemini__gemini-2.5-pro": "Gemini-2.5-Pro",
    "gemini__gemini-3-flash-preview": "Gemini-3-Flash-Preview",
    "gemini__gemini-3-pro-preview": "Gemini-3-Pro-Preview",
    "neulab__deepseek-v3p1": "DeepSeek-V3.1",
    "DeepSeek-V3.1": "DeepSeek-V3.1",
    "openai__gpt-5.2": "GPT-5.2",
    "openai__gpt-5.2-codex": "GPT-5.2-Codex",
    "gemini-3-flash-preview": "Gemini-3-Flash-Preview",
    "gemini3flash": "Gemini-3-Flash-Preview",
    "glm-4p7": "GLM-4.7",
    "glm-5": "GLM-5",
    "gpt-5.2-codex": "GPT-5.2-Codex",
    "kimi-k2-thinking": "Kimi-K2",
    "minimax-m2p1": "MiniMax-M2.1",
    "minimax-m2p5": "MiniMax-M2.5",
    "nvidia-nemotron-3-nano-30b-a3b-bf16": "Nemotron-3-Nano",
    "openai_nvidia_nvidia-nemotron-3-nano-30b-a3b-bf16": "Nemotron-3-Nano",
    "anthropic__claude-opus-4-5": "Claude-4.5-Opus",
    "anthropic__claude-sonnet-4-5": "Claude-4.5-Sonnet",
    "azure_ai__gpt-5.2": "GPT-5.2",
    "azure_ai__gpt-5.2-codex": "GPT-5.2-Codex",
    "azure_ai_gpt-5.2-codex": "GPT-5.2-Codex",
    "models__kimi-k2p5": "Kimi-K2.5",
    "neulab__claude-3-7-sonnet-20250219": "Claude-3.7-Sonnet",
}

def parse_model_name_from_filename(filename: str) -> Tuple[str, Optional[str]]:
    """
    Parse model name and optional reasoning level from a filename.

    Examples:
        azure--gpt-5-mini:reasoning:medium_openai_temp_1.0.eval_results.json
        -> ("azure--gpt-5-mini", "reasoning:medium")

        azure--o3_openai_temp_1.0.eval_results.json
        -> ("azure--o3", None)

    Returns: (model_name, reasoning_level)
    """
    name = filename
    for suffix in [
        ".eval_results.json",
        ".jsonl",
        ".raw.jsonl",
        "_openai_temp_1.0",
    ]:
        name = name.replace(suffix, "")

    match = re.search(r":reasoning:(low|medium|high)", name)
    if match:
        reasoning_level = match.group(0)[1:]  # remove leading ':'
        model_name = name[: match.start()]
    else:
        reasoning_level = None
        model_name = name

    return model_name, reasoning_level


def parse_model_dir_name(dirname: str) -> Tuple[str, Optional[str]]:
    """
    Parse model name and optional reasoning level from a directory name.

    Used for MMLU tasks where each model has its own subdirectory.
    """
    match = re.search(r":reasoning:(low|medium|high)", dirname)
    if match:
        reasoning_level = match.group(0)[1:]
        model_name = dirname[: match.start()]
    else:
        reasoning_level = None
        model_name = dirname
    return model_name, reasoning_level


def normalize_model_name(raw_name: str, reasoning_level: Optional[str] = None) -> str:
    """Normalize a raw model name to a standardized format.

    If a reasoning level is present, append it as a suffix separated by
    a double underscore, with colons replaced by underscores.
    """
    if raw_name in MODEL_NAME_LOOKUP:
        base_name = MODEL_NAME_LOOKUP[raw_name]
    elif raw_name.replace("__", "--") in MODEL_NAME_LOOKUP:
        base_name = MODEL_NAME_LOOKUP[raw_name.replace("__", "--")]
    elif raw_name.replace("__", "_") in MODEL_NAME_LOOKUP:
        base_name = MODEL_NAME_LOOKUP[raw_name.replace("__", "_")]
    elif raw_name.replace("_", "__") in MODEL_NAME_LOOKUP:
        base_name = MODEL_NAME_LOOKUP[raw_name.replace("_", "__")]
    elif raw_name.replace("--", "-") in MODEL_NAME_LOOKUP:
        base_name = MODEL_NAME_LOOKUP[raw_name.replace("--", "-")]
    elif raw_name.replace("__high", "") in MODEL_NAME_LOOKUP:
        base_name = MODEL_NAME_LOOKUP[raw_name.replace("__high", "")]
    else:
        raise ValueError(f"model not in MODEL_NAME_LOOKUP: {raw_name}")

    # if reasoning_level:
    #     return f"{base_name}__{reasoning_level.replace(':', '_')}"
    return base_name




def write_csv(output_csv: Path, metrics_data: List[Union[Tuple[float, str], Tuple[float, str, str]]]) -> None:
    """Write standardized metrics to a CSV file with a header.

    If items contain an id (triples), write header: id, score, metric_name.
    Otherwise, write header: score, metric_name.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "score", "metric_name"])
        for rid, score, metric_name in metrics_data:
            writer.writerow([rid, score, metric_name])


def resolve_paths(input_dir: Path, output_dir: Path) -> Tuple[Path, Path]:
    """
    Resolve input and output directories relative to the project root
    (proxy-bench directory) if they are not absolute.
    """
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    in_path = input_dir if input_dir.is_absolute() else (project_root / input_dir)
    out_path = output_dir if output_dir.is_absolute() else (project_root / output_dir)
    return in_path, out_path


def print_summary(stats: Dict[str, Dict[str, int]]) -> None:
    """Print a simple summary of processed files and entry counts."""
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)

    total_files = 0
    total_entries = 0

    for task in sorted(stats.keys()):
        print(f"\n{task}/")
        for model in sorted(stats[task].keys()):
            count = stats[task][model]
            print(f"  {model}.csv: {count} entries")
            total_files += 1
            total_entries += count

    print("\n" + "-" * 60)
    print(f"Total: {total_files} CSV files, {total_entries} total entries")
