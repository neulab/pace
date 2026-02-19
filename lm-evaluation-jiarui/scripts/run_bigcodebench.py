#!/usr/bin/env python3
"""
BigCodeBench Benchmark Runner

Wrapper script for running BigCodeBench with lm-eval-harness style interface.
Supports all models via the LiteLLM proxy.

Usage:
    python run_bigcodebench.py --model azure/gpt-4o --split instruct --output_path ./results --limit 10
    python run_bigcodebench.py --model azure/gpt-4o --split instruct --output_path ./results --evaluate
"""

import argparse
import json
import logging
import os
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from model_api import (
    ProxyModelAPI,
    ModelConfig,
    GenerationResult,
    ResultsLogger,
    SUPPORTED_MODELS,
)

# Local evaluation module
try:
    from bigcodebench_eval import evaluate_samples, PASS, FAIL, TIMEOUT
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False

# Remote evaluation module (Gradio)
try:
    from gradio_client import Client, handle_file
    import httpx
    REMOTE_EVAL_AVAILABLE = True
except ImportError:
    REMOTE_EVAL_AVAILABLE = False

# Default Gradio endpoint for BigCodeBench evaluation
GRADIO_ENDPOINT = "https://bigcode-bigcodebench-evaluator.hf.space/"


def evaluate_remote(
    samples_path: str,
    split: str,
    subset: str,
    gradio_endpoint: str = GRADIO_ENDPOINT,
    pass_k: str = "1,5,10",
    parallel: int = -1,
    min_time_limit: float = 1,
    max_as_limit: int = 30 * 1024,
    max_data_limit: int = 30 * 1024,
    max_stack_limit: int = 10,
    calibrated: bool = True,
    no_gt: bool = False,
    max_retries: int = 5,
    timeout: int = 1800,  # 30 minutes default timeout
) -> Dict[str, Any]:
    """
    Evaluate samples using BigCodeBench remote Gradio endpoint.
    Faithful to original bigcodebench/evaluate.py gradio evaluation.
    """
    import time
    from concurrent.futures._base import CancelledError
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    if not REMOTE_EVAL_AVAILABLE:
        raise ImportError("gradio_client and httpx are required for remote evaluation. "
                         "Install with: pip install gradio_client httpx")

    logger.info(f"Connecting to Gradio endpoint: {gradio_endpoint}")
    logger.info(f"Timeout: {timeout}s ({timeout // 60} minutes)")

    for attempt in range(max_retries):
        try:
            # Create client with longer httpx timeout
            httpx_kwargs = {"timeout": httpx.Timeout(timeout, connect=60.0)}
            client = Client(gradio_endpoint, httpx_kwargs=httpx_kwargs)
            logger.info(f"Submitting samples for evaluation (attempt {attempt + 1}/{max_retries})...")

            # Use submit() with result(timeout) for longer timeout
            job = client.submit(
                split=split,
                subset=subset,
                samples=handle_file(samples_path),
                pass_k=pass_k,
                parallel=parallel,
                min_time_limit=min_time_limit,
                max_as_limit=max_as_limit,
                max_data_limit=max_data_limit,
                max_stack_limit=max_stack_limit,
                calibrated=calibrated,
                check_gt_only=False,
                no_gt=no_gt,
                selective_evaluate="",
                api_name="/predict"
            )

            # Wait for result with timeout
            results, pass_at_k = job.result(timeout=timeout)

            logger.info("Remote evaluation completed successfully!")
            return {
                "results": results,
                "pass_at_k": pass_at_k,
                "gt_pass_rate": pass_at_k.get("gt_pass_rate"),
                "failed_tasks": pass_at_k.get("failed_tasks", []),
            }

        except (httpx.ReadTimeout, httpx.ConnectTimeout, CancelledError, FuturesTimeoutError, TimeoutError) as e:
            logger.warning(f"Timeout error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = 4 * (attempt + 1)  # Exponential backoff
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            logger.error(f"Remote evaluation failed: {e}")
            raise

    raise RuntimeError("Remote evaluation failed after all retries")


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============ BigCodeBench-specific helpers ============

# EOS tokens for code generation
EOS_TOKENS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]

# Default instruction and response prefixes
DEFAULT_INSTRUCTION_PREFIX = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
DEFAULT_RESPONSE_PREFIX = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"


def extract_code_from_response(response: str) -> str:
    """Extract Python code from markdown code blocks or raw response."""
    # Try to extract from markdown code block
    code_block_pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # If no code block, try to find the longest valid Python code
    lines = response.split("\n")

    # Look for function definition start
    code_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("def ") or line.strip().startswith("import ") or line.strip().startswith("from "):
            code_start = i
            break

    if code_start >= 0:
        return "\n".join(lines[code_start:]).strip()

    return response.strip()


def sanitize_code(code: str, entry_point: str) -> str:
    """
    Code sanitization using bigcodebench.sanitize if available,
    otherwise falls back to basic extraction.
    """
    # First extract code from markdown if present
    code = extract_code_from_response(code)

    # Remove common EOS tokens
    for eos in EOS_TOKENS:
        if eos in code:
            code = code.split(eos)[0]

    # Try to use bigcodebench's proper tree-sitter based sanitization
    try:
        from bigcodebench.sanitize import sanitize
        return sanitize(code, entry_point)
    except ImportError:
        logger.warning(
            "bigcodebench package not installed. Using basic code extraction. "
            "For proper sanitization, install: pip install bigcodebench"
        )
        return code.strip()
    except Exception as e:
        # If sanitization fails, return the basic extracted code
        logger.warning(f"Sanitization failed: {e}. Using basic extraction.")
        return code.strip()


def create_prompt(
    task: Dict,
    split: str,
    apply_template: bool = True,
    instruction_prefix: str = DEFAULT_INSTRUCTION_PREFIX,
    response_prefix: str = DEFAULT_RESPONSE_PREFIX,
) -> List[Dict[str, str]]:
    """Create messages for the model from a task."""
    # Get the appropriate prompt based on split
    if split == "complete":
        task_prompt = task.get("complete_prompt", "")
        # For complete split, wrap in code block
        user_content = f"{instruction_prefix}\n```\n{task_prompt.strip()}\n```"
    else:  # instruct
        task_prompt = task.get("instruct_prompt", "")
        user_content = f"{instruction_prefix}\n{task_prompt.strip()}"

    if apply_template:
        messages = [
            {"role": "user", "content": user_content},
        ]
        # For prefill mode, we could add assistant message prefix
        # but for simplicity, we just let the model generate freely
    else:
        messages = [{"role": "user", "content": task_prompt}]

    return messages


def run_bigcodebench(
    model: str,
    split: str,
    subset: str,
    output_path: str,
    limit: Optional[int] = None,
    apply_chat_template_flag: bool = True,
    log_samples: bool = True,
    base_url: str = "https://cmu.litellm.ai/v1/chat/completions",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    n_samples: int = 1,
    instruction_prefix: str = DEFAULT_INSTRUCTION_PREFIX,
    response_prefix: str = DEFAULT_RESPONSE_PREFIX,
    evaluate: bool = False,
    evaluate_remote: bool = False,
    eval_parallel: int = -1,
    calibrated: bool = True,
    gradio_endpoint: str = GRADIO_ENDPOINT,
    remote_timeout: int = 1800,
):
    """Run BigCodeBench benchmark evaluation."""
    from datasets import load_dataset

    logger.info(f"Running BigCodeBench benchmark")
    logger.info(f"  Model: {model}")
    logger.info(f"  Split: {split}")
    logger.info(f"  Subset: {subset}")
    logger.info(f"  Output path: {output_path}")
    logger.info(f"  Limit: {limit}")
    logger.info(f"  Apply chat template: {apply_chat_template_flag}")

    # Load dataset
    # BigCodeBench uses different HF paths for different subsets
    if subset == "full":
        hf_path = "bigcode/bigcodebench"
    else:
        hf_path = f"bigcode/bigcodebench-{subset}"

    try:
        # Try loading from HuggingFace
        data = load_dataset(hf_path, split="v0.1.4")
        logger.info(f"Loaded {len(data)} tasks from {hf_path}")
    except Exception as e:
        logger.warning(f"Could not load from HF: {e}")
        # Try loading from local repo
        try:
            from bigcodebench.data import get_bigcodebench
            data_dict = get_bigcodebench(subset=subset)
            # Convert dict to list for iteration
            data = [{"task_id": k, **v} for k, v in data_dict.items()]
            logger.info(f"Loaded {len(data)} tasks from local cache")
        except Exception as e2:
            logger.error(f"Could not load BigCodeBench data: {e2}")
            return None

    # Convert to list if it's a dict
    if isinstance(data, dict):
        data = [{"task_id": k, **v} for k, v in data.items()]

    # Apply limit
    if limit is not None and limit > 0:
        data = data[:limit] if isinstance(data, list) else list(data)[:limit]
        logger.info(f"Limited to {len(data)} tasks")
    else:
        data = list(data)

    # Initialize model API
    config = ModelConfig(
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    api = ProxyModelAPI(config)

    # Initialize results logger
    benchmark_name = f"bigcodebench_{subset}_{split}"
    results_logger = ResultsLogger(
        output_path=output_path,
        model_name=model,
        benchmark_name=benchmark_name,
    )

    # Prepare output path for BigCodeBench native format
    model_sanitized = model.replace("/", "--")
    native_output_path = Path(output_path) / results_logger.model_name_sanitized / "bigcodebench_format"
    native_output_path.mkdir(parents=True, exist_ok=True)
    native_output_file = native_output_path / f"{model_sanitized}--bigcodebench-{subset}-{split}--{temperature}-{n_samples}.jsonl"

    # Run generation
    all_samples = []
    from tqdm import tqdm

    for idx, task in enumerate(tqdm(data, desc="Generating")):
        task_dict = dict(task) if hasattr(task, 'items') else task
        task_id = task_dict.get("task_id", f"task_{idx}")
        entry_point = task_dict.get("entry_point", "solution")

        try:
            # Create prompt
            messages = create_prompt(
                task_dict,
                split=split,
                apply_template=apply_chat_template_flag,
                instruction_prefix=instruction_prefix,
                response_prefix=response_prefix,
            )

            # Generate samples
            for sample_idx in range(n_samples):
                response = api.generate(
                    messages,
                    gen_kwargs={
                        "max_tokens": max_tokens,
                        "stop": EOS_TOKENS[:4],  # API typically allows max 4 stop sequences
                    }
                )

                # Sanitize code
                sanitized_code = sanitize_code(response, entry_point)

                # Create sample record
                sample = {
                    "task_id": task_id,
                    "solution": sanitized_code,
                    "raw_solution": response,
                    "sample_idx": sample_idx,
                }
                all_samples.append(sample)

                # Log sample
                if log_samples:
                    result = GenerationResult(
                        doc_id=f"{task_id}_{sample_idx}",
                        doc=task_dict,
                        prompt=messages,
                        response=response,
                        metadata={
                            "sanitized_solution": sanitized_code,
                            "entry_point": entry_point,
                        },
                    )
                    results_logger.add_sample(result)

            # Progress update
            if (idx + 1) % 10 == 0:
                logger.info(f"Progress: {idx + 1}/{len(data)}")

        except Exception as e:
            logger.error(f"Error on task {task_id}: {e}")
            all_samples.append({
                "task_id": task_id,
                "solution": "",
                "raw_solution": f"ERROR: {str(e)}",
                "sample_idx": 0,
            })

    # Save results in BigCodeBench native format
    with open(native_output_file, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    logger.info(f"Saved {len(all_samples)} samples to {native_output_file}")

    # Calculate metrics (generation metrics only - execution needs separate step)
    metrics = {
        "num_tasks": len(data),
        "num_samples": len(all_samples),
        "samples_per_task": n_samples,
        "generation_complete": True,
    }

    # Run evaluation if requested
    if evaluate:
        if not EVAL_AVAILABLE:
            logger.error("Evaluation module not available. Install dependencies or check bigcodebench_eval.py")
            metrics["note"] = "Evaluation skipped - module not available"
        else:
            logger.info("\n" + "="*50)
            logger.info("Running local evaluation...")
            logger.info("="*50)

            # Build problems dict from data
            problems = {}
            for task in data:
                task_dict = dict(task) if hasattr(task, 'items') else task
                task_id = task_dict.get("task_id")
                if task_id:
                    problems[task_id] = task_dict

            try:
                eval_results = evaluate_samples(
                    samples_path=str(native_output_file),
                    problems=problems,
                    split=split,
                    subset=subset,
                    pass_k=[1] if n_samples == 1 else [1, 5, 10],
                    parallel=eval_parallel,
                    calibrated=calibrated,
                    verbose=True,
                )

                # Add evaluation metrics to results
                metrics["evaluation"] = {
                    "num_passed": eval_results["num_passed"],
                    "num_failed": eval_results["num_failed"],
                    "num_timeout": eval_results["num_timeout"],
                    "calibrated": eval_results["calibrated"],
                }

                # Add pass@k metrics
                for key in eval_results:
                    if key.startswith("pass@"):
                        metrics["evaluation"][key] = eval_results[key]

                # Save detailed eval results separately
                eval_output_file = native_output_path / f"{model_sanitized}--bigcodebench-{subset}-{split}--eval_results.json"
                with open(eval_output_file, "w") as f:
                    # Save without the large detailed_results for the summary
                    eval_summary = {k: v for k, v in eval_results.items() if k != "detailed_results"}
                    json.dump(eval_summary, f, indent=2)
                logger.info(f"Saved evaluation summary to {eval_output_file}")

                # Save full detailed results
                detailed_output_file = native_output_path / f"{model_sanitized}--bigcodebench-{subset}-{split}--eval_detailed.json"
                with open(detailed_output_file, "w") as f:
                    json.dump(eval_results["detailed_results"], f, indent=2)
                logger.info(f"Saved detailed results to {detailed_output_file}")

            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                metrics["evaluation_error"] = str(e)

    elif evaluate_remote:
        if not REMOTE_EVAL_AVAILABLE:
            logger.error("Remote evaluation requires gradio_client and httpx. "
                        "Install with: pip install gradio_client httpx")
            metrics["note"] = "Remote evaluation skipped - dependencies not available"
        else:
            logger.info("\n" + "="*50)
            logger.info("Running remote evaluation via Gradio...")
            logger.info("="*50)

            # Prepare output path for remote evaluation results
            remote_output_path = Path(output_path) / results_logger.model_name_sanitized / "bigcodebench_format_remote"
            remote_output_path.mkdir(parents=True, exist_ok=True)

            try:
                eval_results = evaluate_remote(
                    samples_path=str(native_output_file),
                    split=split,
                    subset=subset,
                    gradio_endpoint=gradio_endpoint,
                    pass_k="1" if n_samples == 1 else "1,5,10",
                    parallel=eval_parallel,
                    calibrated=calibrated,
                    timeout=remote_timeout,
                )

                # Add evaluation metrics to results
                metrics["evaluation_remote"] = {
                    "pass_at_k": eval_results.get("pass_at_k", {}),
                    "gt_pass_rate": eval_results.get("gt_pass_rate"),
                    "failed_tasks": eval_results.get("failed_tasks", []),
                }

                # Save remote eval results
                remote_eval_file = remote_output_path / f"{model_sanitized}--bigcodebench-{subset}-{split}--remote_eval_results.json"
                with open(remote_eval_file, "w") as f:
                    json.dump(eval_results, f, indent=2)
                logger.info(f"Saved remote evaluation results to {remote_eval_file}")

                # Also save pass@k summary
                pass_at_k_file = remote_output_path / f"{model_sanitized}--bigcodebench-{subset}-{split}--pass_at_k.json"
                with open(pass_at_k_file, "w") as f:
                    json.dump(eval_results.get("pass_at_k", {}), f, indent=2)
                logger.info(f"Saved pass@k summary to {pass_at_k_file}")

            except Exception as e:
                logger.error(f"Remote evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                metrics["evaluation_remote_error"] = str(e)

    else:
        metrics["note"] = "Run with --evaluate or --evaluate_remote flag for pass@k metrics"

    # Save results
    if log_samples:
        results_logger.save_samples()
    results_logger.save_results(metrics)

    logger.info(f"\nGeneration Complete!")
    logger.info(f"  Tasks: {len(data)}")
    logger.info(f"  Samples: {len(all_samples)}")
    logger.info(f"  Output: {native_output_file}")

    if not evaluate and not evaluate_remote:
        logger.info(f"\nTo evaluate, run:")
        logger.info(f"  Local:  python run_bigcodebench.py --model {model} --split {split} --subset {subset} --evaluate")
        logger.info(f"  Remote: python run_bigcodebench.py --model {model} --split {split} --subset {subset} --evaluate_remote")
        logger.info(f"  OR")
        logger.info(f"  bigcodebench.evaluate {split} {subset} --samples {native_output_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run BigCodeBench benchmark with lm-eval-harness style interface"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., azure/gpt-4o, gemini/gemini-2.5-pro). Required unless using --evaluate_only or --evaluate_remote_only.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="instruct",
        choices=["complete", "instruct"],
        help="Task split (default: instruct)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="full",
        choices=["full", "hard"],
        help="Dataset subset (default: full)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results/bigcodebench",
        help="Path to save results (default: ./results/bigcodebench)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to evaluate (default: None = all)",
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
        default=2048,
        help="Maximum tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of samples per task (default: 1)",
    )
    parser.add_argument(
        "--instruction_prefix",
        type=str,
        default=DEFAULT_INSTRUCTION_PREFIX,
        help="Instruction prefix for prompts",
    )
    parser.add_argument(
        "--response_prefix",
        type=str,
        default=DEFAULT_RESPONSE_PREFIX,
        help="Response prefix for prompts",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Run local evaluation after generation (default: False)",
    )
    parser.add_argument(
        "--eval_parallel",
        type=int,
        default=-1,
        help="Number of parallel workers for evaluation (-1 for auto, default: -1)",
    )
    parser.add_argument(
        "--calibrated",
        action="store_true",
        default=True,
        help="Use calibrated evaluation (prepend code_prompt to solutions, default: True)",
    )
    parser.add_argument(
        "--no_calibrated",
        action="store_true",
        default=False,
        help="Disable calibrated evaluation",
    )
    parser.add_argument(
        "--evaluate_only",
        type=str,
        default=None,
        help="Path to existing samples file to evaluate (skips generation)",
    )
    parser.add_argument(
        "--evaluate_remote",
        action="store_true",
        default=False,
        help="Run remote evaluation via Gradio endpoint after generation (default: False)",
    )
    parser.add_argument(
        "--gradio_endpoint",
        type=str,
        default=GRADIO_ENDPOINT,
        help=f"Gradio endpoint for remote evaluation (default: {GRADIO_ENDPOINT})",
    )
    parser.add_argument(
        "--evaluate_remote_only",
        type=str,
        default=None,
        help="Path to existing samples file to evaluate remotely (skips generation)",
    )
    parser.add_argument(
        "--remote_timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for remote Gradio evaluation (default: 1800 = 30 minutes)",
    )

    args = parser.parse_args()

    # Handle calibrated flag
    calibrated = args.calibrated and not args.no_calibrated

    # Validate --model is required unless using evaluation-only modes
    if not args.evaluate_only and not args.evaluate_remote_only and not args.model:
        parser.error("--model is required unless using --evaluate_only or --evaluate_remote_only")

    # Evaluate-only mode
    if args.evaluate_only:
        if not EVAL_AVAILABLE:
            logger.error("Evaluation module not available")
            return 1

        logger.info(f"Running evaluation-only mode on: {args.evaluate_only}")

        # Load problems from dataset
        from datasets import load_dataset
        if args.subset == "full":
            hf_path = "bigcode/bigcodebench"
        else:
            hf_path = f"bigcode/bigcodebench-{args.subset}"

        try:
            data = load_dataset(hf_path, split="v0.1.4")
        except Exception as e:
            logger.error(f"Could not load BigCodeBench data: {e}")
            return 1

        problems = {}
        for task in data:
            task_dict = dict(task)
            task_id = task_dict.get("task_id")
            if task_id:
                problems[task_id] = task_dict

        eval_results = evaluate_samples(
            samples_path=args.evaluate_only,
            problems=problems,
            split=args.split,
            subset=args.subset,
            pass_k=[1, 5, 10],
            parallel=args.eval_parallel,
            calibrated=calibrated,
            verbose=True,
        )

        # Save results
        output_file = args.evaluate_only.replace(".jsonl", "_eval_results.json")
        eval_summary = {k: v for k, v in eval_results.items() if k != "detailed_results"}
        with open(output_file, "w") as f:
            json.dump(eval_summary, f, indent=2)
        logger.info(f"Saved evaluation results to {output_file}")

        return 0

    # Remote evaluate-only mode
    if args.evaluate_remote_only:
        if not REMOTE_EVAL_AVAILABLE:
            logger.error("Remote evaluation requires gradio_client and httpx. "
                        "Install with: pip install gradio_client httpx")
            return 1

        logger.info(f"Running remote evaluation-only mode on: {args.evaluate_remote_only}")

        # Create output directory for remote eval results
        samples_path = Path(args.evaluate_remote_only)
        remote_output_path = samples_path.parent / "bigcodebench_format_remote"
        remote_output_path.mkdir(parents=True, exist_ok=True)

        try:
            eval_results = evaluate_remote(
                samples_path=args.evaluate_remote_only,
                split=args.split,
                subset=args.subset,
                gradio_endpoint=args.gradio_endpoint,
                pass_k="1,5,10",
                parallel=args.eval_parallel,
                calibrated=calibrated,
                timeout=args.remote_timeout,
            )

            # Save remote eval results
            output_file = remote_output_path / samples_path.name.replace(".jsonl", "--remote_eval_results.json")
            with open(output_file, "w") as f:
                json.dump(eval_results, f, indent=2)
            logger.info(f"Saved remote evaluation results to {output_file}")

            # Also save pass@k summary
            pass_at_k_file = remote_output_path / samples_path.name.replace(".jsonl", "--pass_at_k.json")
            with open(pass_at_k_file, "w") as f:
                json.dump(eval_results.get("pass_at_k", {}), f, indent=2)
            logger.info(f"Saved pass@k summary to {pass_at_k_file}")

            # Print results
            logger.info("\nRemote Evaluation Results:")
            logger.info(f"  pass@k: {eval_results.get('pass_at_k', {})}")
            logger.info(f"  gt_pass_rate: {eval_results.get('gt_pass_rate')}")

        except Exception as e:
            logger.error(f"Remote evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

        return 0

    metrics = run_bigcodebench(
        model=args.model,
        split=args.split,
        subset=args.subset,
        output_path=args.output_path,
        limit=args.limit,
        apply_chat_template_flag=args.apply_chat_template,
        log_samples=args.log_samples,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        n_samples=args.n_samples,
        instruction_prefix=args.instruction_prefix,
        response_prefix=args.response_prefix,
        evaluate=args.evaluate,
        evaluate_remote=args.evaluate_remote,
        eval_parallel=args.eval_parallel,
        calibrated=calibrated,
        gradio_endpoint=args.gradio_endpoint,
        remote_timeout=args.remote_timeout,
    )

    return 0 if metrics else 1


if __name__ == "__main__":
    sys.exit(main())
