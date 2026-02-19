#!/usr/bin/env python3
"""
Oolong Benchmark Runner

Wrapper script for running Oolong benchmark with lm-eval-harness style interface.
Supports all models via the LiteLLM proxy.

Usage:
    python run_oolong.py --model azure/gpt-4o --dataset synth --output_path ./results --limit 10
"""

import argparse
import json
import logging
import os
import sys
import re
import ast
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, List, Any, Optional

from model_api import (
    ProxyModelAPI,
    ModelConfig,
    GenerationResult,
    ResultsLogger,
    apply_chat_template,
    SUPPORTED_MODELS,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============ Oolong-specific evaluation helpers ============

def synth_attempt_answer_parse(answer: str) -> tuple[str, str]:
    """Parse answer from model output for synth dataset."""
    parse_confidence = "low"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, parse_confidence
        else:
            return answer.split()[-1], parse_confidence

    candidate_answer = answer.split(":")[-1].strip()
    candidate_answer = candidate_answer.replace("*", "")  # OpenAI bolding
    candidate_answer = candidate_answer.replace("[", "").replace("]", "")  # Anthropic brackets

    parse_confidence = "med"
    if any(x in answer for x in ["User:", "Answer:", "Date:", "Label"]):
        parse_confidence = "high"
    if len(candidate_answer) < 20:
        parse_confidence = "vhigh"
    elif "more common" in candidate_answer:
        candidate_answer = "more common"
    elif "less common" in candidate_answer:
        candidate_answer = "less common"
    elif "same frequency" in candidate_answer:
        candidate_answer = "same frequency"

    return candidate_answer, parse_confidence


def synth_process_response(datapoint: Dict, output: str, model: str) -> Dict:
    """Process response for synth dataset."""
    try:
        import dateutil.parser
    except ImportError:
        dateutil = None

    score = 0
    try:
        if "datetime" not in datapoint["answer"]:
            gold = ast.literal_eval(datapoint["answer"])[0]
        else:
            gold = dt.strptime(datapoint["answer"], "[datetime.date(%Y, %m, %d)]")
    except:
        gold = datapoint["answer"]

    trimmed_output, parse_confidence = synth_attempt_answer_parse(output)

    if str(trimmed_output) == str(gold):
        score = 1
    elif str(trimmed_output) in ['more common', 'less common', 'same frequency']:
        if str(trimmed_output) in str(gold):
            score = 1
    elif datapoint.get("answer_type") == "ANSWER_TYPE.NUMERIC":
        try:
            trimmed_output_int = int(trimmed_output)
            gold_int = int(gold)
            score = 0.75 ** abs(gold_int - trimmed_output_int)
        except:
            parse_confidence = "low"
    elif datapoint.get("answer_type") == "ANSWER_TYPE.DATE" and dateutil:
        try:
            trimmed_output_date = dateutil.parser.parse(trimmed_output)
            score = float(trimmed_output_date == gold)
        except:
            parse_confidence = "low"

    return {
        "id": datapoint["id"],
        "context_window_id": datapoint.get("context_window_id", ""),
        "dataset": datapoint.get("dataset", "synth"),
        "model": model,
        "attempted_parse": str(trimmed_output),
        "parse_confidence": parse_confidence,
        "full_answer": output,
        "score": score,
        "context_len": datapoint.get("context_len", 0),
        "task_group": datapoint.get("task_group", ""),
        "task": datapoint.get("task", ""),
        "answer_type": datapoint.get("answer_type", ""),
        "answer": str(gold),
    }


def dnd_parse_answer(answer: str):
    """Parse the answer into int, str, or list of str."""
    try:
        return int(answer)
    except ValueError:
        pass
    if "," in answer:
        return [item.strip() for item in answer.split(",") if item.strip()]
    return answer


def dnd_parse_response(answer: str) -> tuple:
    """Parse response for DnD/real dataset."""
    match = re.search(r"\\boxed\{\\text\{([^}]*)\}\}", answer) or re.search(
        r"\\boxed[\{]+([^}]*)[\}]+", answer
    )
    if match:
        answer = match.group(1)
        return dnd_parse_answer(answer), "high"
    return answer, "low"


def dnd_process_response(datapoint: Dict, output: str, model: str) -> Dict:
    """Process response for DnD/real dataset."""
    gold = dnd_parse_answer(datapoint["answer"])
    trimmed_output, parse_confidence = dnd_parse_response(output)

    score = 0.0
    if isinstance(gold, int) and isinstance(trimmed_output, int):
        score = 0.75 ** abs(gold - trimmed_output)
    elif isinstance(gold, str) and isinstance(trimmed_output, str):
        score = float(gold.strip().lower() == trimmed_output.strip().lower())
    elif isinstance(gold, list) and isinstance(trimmed_output, list):
        overlap = set(gold) & set(trimmed_output)
        score = len(overlap) / len(gold) if gold else 0.0

    return {
        "id": datapoint["id"],
        "context_window_id": datapoint.get("context_window_id", ""),
        "model": model,
        "attempted_parse": trimmed_output,
        "parse_confidence": parse_confidence,
        "full_answer": output,
        "score": score,
        "answer": gold,
    }


def create_messages(datapoint: Dict, apply_template: bool = True) -> List[Dict]:
    """Create messages for the model from a datapoint."""
    context_text = datapoint.get("context_window_text", "")
    question = datapoint.get("question", "")

    if apply_template:
        # Use structured message format
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant.\n\n{context_text}"
            },
            {
                "role": "user",
                "content": question
            }
        ]
    else:
        # Simple concatenation
        messages = [
            {
                "role": "user",
                "content": f"{context_text}\n\n{question}"
            }
        ]

    return messages


def run_oolong(
    model: str,
    dataset: str,
    output_path: str,
    limit: Optional[int] = None,
    apply_chat_template_flag: bool = True,
    log_samples: bool = True,
    base_url: str = "https://cmu.litellm.ai/v1/chat/completions",
    max_tokens: int = 16384,
    temperature: float = 0.0,
    batch_size: int = 1,
    use_cache: bool = True,
    cache_dir: str = None,
):
    """Run Oolong benchmark evaluation."""
    from datasets import load_dataset
    import jsonlines

    logger.info(f"Running Oolong benchmark")
    logger.info(f"  Model: {model}")
    logger.info(f"  Dataset: {dataset}")
    logger.info(f"  Output path: {output_path}")
    logger.info(f"  Limit: {limit}")
    logger.info(f"  Apply chat template: {apply_chat_template_flag}")
    logger.info(f"  Cache enabled: {use_cache}")

    # Load dataset
    if dataset == "synth":
        data = load_dataset("oolongbench/oolong-synth")["test"]
        process_response = synth_process_response
    else:  # real
        data = load_dataset("oolongbench/oolong-real", "dnd")["test"]
        process_response = dnd_process_response

    logger.info(f"Loaded {len(data)} examples from {dataset} dataset")

    # Apply limit
    if limit is not None and limit > 0:
        data = data.select(range(min(limit, len(data))))
        logger.info(f"Limited to {len(data)} examples")

    # Initialize model API with caching
    config = ModelConfig(
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )
    api = ProxyModelAPI(config)

    # Initialize results logger
    results_logger = ResultsLogger(
        output_path=output_path,
        model_name=model,
        benchmark_name=f"oolong_{dataset}",
    )

    # Run evaluation
    all_outputs = []
    correct = 0
    total = 0

    from tqdm import tqdm

    for idx, datapoint in enumerate(tqdm(data, desc="Evaluating")):
        try:
            # Create messages
            messages = create_messages(
                dict(datapoint),
                apply_template=apply_chat_template_flag
            )

            # Generate response
            response = api.generate(messages, gen_kwargs={"max_tokens": max_tokens})

            # Process response
            output = process_response(dict(datapoint), response, model)

            # Track results
            correct += output["score"]
            total += 1
            all_outputs.append(output)

            # Log sample
            if log_samples:
                result = GenerationResult(
                    doc_id=datapoint["id"],
                    doc=dict(datapoint),
                    prompt=messages,
                    response=response,
                    metadata=output,
                )
                results_logger.add_sample(result)

            # Progress update
            if (idx + 1) % 10 == 0:
                logger.info(f"Progress: {idx + 1}/{len(data)}, Score: {correct/total:.4f}")

        except Exception as e:
            logger.error(f"Error on example {idx}: {e}")
            # Don't count errors towards total - they weren't actually evaluated
            # This allows re-running to fill in cached results
            all_outputs.append({
                "id": datapoint.get("id", idx),
                "error": str(e),
                "score": 0,
                "skipped": True,
            })

    # Calculate final metrics
    errors = sum(1 for o in all_outputs if o.get("skipped", False))
    final_score = correct / total if total > 0 else 0
    metrics = {
        "accuracy": final_score,
        "total_correct": correct,
        "total_examples": total,
        "total_errors": errors,
        "total_attempted": len(data),
        "cache_hits": api.cache_hits,
        "cache_misses": api.cache_misses,
    }

    logger.info(f"\nFinal Results:")
    logger.info(f"  Accuracy: {final_score:.4f}")
    logger.info(f"  Correct: {correct}/{total}")
    if errors > 0:
        logger.info(f"  Errors: {errors} (re-run to retry with cache)")
    logger.info(f"  Cache: {api.cache_hits} hits, {api.cache_misses} misses")

    # Save results
    if log_samples:
        results_logger.save_samples()
    results_logger.save_results(metrics)

    # Also save in Oolong's native format
    oolong_output_dir = Path(output_path) / results_logger.model_name_sanitized / "oolong_format"
    oolong_output_dir.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(oolong_output_dir / "full_output.jsonl", "w") as f:
        for line in all_outputs:
            f.write(line)

    with open(oolong_output_dir / "overall.txt", "w") as f:
        f.write(f"Overall score for {model} on {total} examples: {correct}/{total} = {final_score}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run Oolong benchmark with lm-eval-harness style interface"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., azure/gpt-4o, gemini/gemini-2.5-pro)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synth",
        choices=["synth", "real"],
        help="Dataset to evaluate on (default: synth)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results/oolong",
        help="Path to save results (default: ./results/oolong)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to evaluate (default: None = all)",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=True,
        help="Apply chat template to prompts (default: True)",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=True,
        help="Log individual samples to output (default: True)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://cmu.litellm.ai/v1/chat/completions",
        help="Base URL for LiteLLM proxy",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16384,
        help="Maximum tokens to generate (default: 16384)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation (default: 1)",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=True,
        help="Enable response caching to avoid redundant API calls (default: True)",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        default=False,
        help="Disable response caching",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory (default: ~/.cache/llm_responses)",
    )

    args = parser.parse_args()

    # Handle cache flag
    use_cache = args.use_cache and not args.no_cache

    metrics = run_oolong(
        model=args.model,
        dataset=args.dataset,
        output_path=args.output_path,
        limit=args.limit,
        apply_chat_template_flag=args.apply_chat_template,
        log_samples=args.log_samples,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        use_cache=use_cache,
        cache_dir=args.cache_dir,
    )

    return 0 if metrics else 1


if __name__ == "__main__":
    sys.exit(main())
