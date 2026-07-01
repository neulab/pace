"""
Calculate estimated token costs for repobench inference results.

Since prompt text is not saved in results, prompt tokens are estimated from
the 'level' field (e.g. "8k" → 8192 tokens). Completion tokens are counted
from the saved 'pred' text.

The model name is derived from the result directory name by stripping the
language/level suffix (e.g. "gpt-5-python-8k" → "gpt-5"), then prepending
an optional prefix (e.g. "litellm_proxy/azure_ai/").

Usage:
    python calc_cost.py --prefix litellm_proxy/azure_ai/
    python calc_cost.py --prefix litellm_proxy/azure_ai/ --filter gpt-5-python-8k
    python calc_cost.py --results_dir ./results --prefix litellm_proxy/azure_ai/
"""

import os
import csv
import json
import argparse
from collections import defaultdict
import yaml
import litellm
from litellm import completion_cost, token_counter

litellm.api_key = os.getenv("LITELLM_API_KEY")
litellm.api_base = os.getenv("LITELLM_BASE_URL", "https://cmu.litellm.ai")

# Load custom pricing from YAML as a fallback for models not in litellm's pricing table.
# Keyed by model_name → (input_cost_per_token, output_cost_per_token).
_CUSTOM_PRICING: dict[str, tuple[float, float]] = {}
_PRICING_YAML = os.path.join(os.path.dirname(__file__), "custom_pricing.yaml")
if os.path.exists(_PRICING_YAML):
    with open(_PRICING_YAML) as f:
        _config = yaml.safe_load(f)
    for entry in _config.get("model_list", []):
        model_name = entry["model_name"]
        info = entry.get("model_info", {})
        _CUSTOM_PRICING[model_name] = (
            info.get("input_cost_per_token", 0),
            info.get("output_cost_per_token", 0),
        )

LEVEL_TO_TOKENS = {
    "2k":   2048,
    "4k":   4096,
    "8k":   8192,
    "12k":  12288,
    "16k":  16384,
    "24k":  24576,
    "32k":  32768,
    "64k":  65536,
    "128k": 131072,
}

# Known language suffixes to strip from directory names
LANGUAGE_SUFFIXES = ["python", "java"]

# Auto-detected prefix based on directory name prefix. Checked in order.
PREFIX_MAP = [
    ("claude",    "anthropic/"),
    ("gemini",    "gemini/"),
    ("Kimi",      "fireworks_ai/accounts/fireworks/models/"),
    ("qwen",      "fireworks_ai/accounts/fireworks/models/"),
    ("nemotron",  "fireworks_ai/accounts/fireworks/models/"),
    ("minimax",   "fireworks_ai/accounts/fireworks/models/"),
    ("glm",       "fireworks_ai/accounts/fireworks/models/"),
    ("deepseek",  "fireworks_ai/accounts/fireworks/models/"),
]
DEFAULT_PREFIX = "azure_ai/"


def dir_to_model(dir_name: str, prefix: str = "") -> str:
    """
    Derive litellm model name from a result directory name.

    e.g. "gpt-5-python-8k"        → azure_ai/gpt-5
         "claude-opus-4-6-python-8k" → anthropic/claude-opus-4-6
    """
    parts = dir_name.split("-")
    # Drop trailing level token (e.g. "8k")
    if parts and parts[-1] in LEVEL_TO_TOKENS:
        parts = parts[:-1]
    # Drop trailing language token
    if parts and parts[-1] in LANGUAGE_SUFFIXES:
        parts = parts[:-1]
    base = "-".join(parts)

    if prefix:
        return prefix + base

    for pattern, auto_prefix in PREFIX_MAP:
        if dir_name.startswith(pattern):
            return auto_prefix + base
    return DEFAULT_PREFIX + base


def calc_dir_cost(result_dir: str, prefix: str = "") -> dict:
    """Calculate costs for a single model result directory."""
    dir_name = os.path.basename(result_dir)
    model = dir_to_model(dir_name, prefix)  # prefix="" triggers auto-detection

    totals = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "items": 0, "errors": 0})

    for fname in sorted(os.listdir(result_dir)):
        if not fname.endswith(".jsonl"):
            continue
        subset = fname[:-6]  # strip .jsonl
        fpath = os.path.join(result_dir, fname)

        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                pred = record.get("pred", "")
                level = record.get("level", "8k")

                prompt_tokens = LEVEL_TO_TOKENS.get(level, 8192)

                try:
                    completion_tokens = token_counter(model=model, text=pred)
                except Exception:
                    completion_tokens = max(1, len(pred.split()))

                if model in _CUSTOM_PRICING:
                    in_price, out_price = _CUSTOM_PRICING[model]
                    cost = prompt_tokens * in_price + completion_tokens * out_price
                else:
                    try:
                        cost = completion_cost(
                            model=model,
                            prompt="x " * (prompt_tokens // 2),
                            completion=pred or " ",
                        )
                    except Exception as e:
                        cost = 0.0
                        totals[subset]["errors"] += 1
                        if totals[subset]["errors"] == 1:
                            print(f"  [error sample] {e}")

                t = totals[subset]
                t["prompt_tokens"] += prompt_tokens
                t["completion_tokens"] += completion_tokens
                t["cost"] += cost
                t["items"] += 1

    return {"model": model, "dir": dir_name, "subsets": dict(totals)}


def write_csv(results: list[dict], csv_path: str) -> None:
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "task", "total_cost", "num_samples", "avg_cost"])
        for r in results:
            for subset, s in sorted(r["subsets"].items()):
                avg = s["cost"] / s["items"] if s["items"] else 0.0
                writer.writerow([r["model"], subset, round(s["cost"], 6), s["items"], round(avg, 8)])
    print(f"Appended to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Estimate LLM inference costs from repobench results.")
    parser.add_argument("--results_dir", default="./results", help="Path to results directory (default: ./results)")
    parser.add_argument("--prefix", default="", help="Override auto-detected prefix for all models, e.g. 'azure_ai/'")
    parser.add_argument("--filter", default=None, metavar="DIR", help="Only process this subdirectory name")
    parser.add_argument("--output", default="costs.csv", help="Output CSV path (default: costs.csv)")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"Error: results directory not found: {results_dir}")
        return

    subdirs = sorted(os.listdir(results_dir))
    if args.filter:
        subdirs = [d for d in subdirs if d == args.filter]
        if not subdirs:
            print(f"Error: no directory named '{args.filter}' in {results_dir}")
            return

    all_results = []
    for d in subdirs:
        full_path = os.path.join(results_dir, d)
        if os.path.isdir(full_path):
            model = dir_to_model(d, args.prefix)
            print(f"Processing {d}  →  {model}")
            all_results.append(calc_dir_cost(full_path, args.prefix))

    write_csv(all_results, args.output)


if __name__ == "__main__":
    main()
