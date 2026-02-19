#!/usr/bin/env python3
"""
Run remote BigCodeBench evaluation via Gradio endpoint.
This is the fastest evaluation method according to BigCodeBench docs.

Usage:
    # Single file
    python run_bigcodebench_remote_eval.py --samples /path/to/samples.jsonl

    # All models in a directory
    python run_bigcodebench_remote_eval.py --input_path /path/to/results

    # With custom timeout
    python run_bigcodebench_remote_eval.py --samples /path/to/samples.jsonl --timeout 3600
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default Gradio endpoint
GRADIO_ENDPOINT = "https://bigcode-bigcodebench-evaluator.hf.space/"

# Check for required dependencies
try:
    import httpx
    from gradio_client import Client, handle_file
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("gradio_client and httpx not available. Install with: pip install gradio_client httpx")


def evaluate_remote(
    samples_path: str,
    split: str = "instruct",
    subset: str = "full",
    gradio_endpoint: str = GRADIO_ENDPOINT,
    pass_k: str = "1",
    parallel: int = -1,
    min_time_limit: float = 1,
    max_as_limit: int = 30 * 1024,
    max_data_limit: int = 30 * 1024,
    max_stack_limit: int = 10,
    calibrated: bool = True,
    no_gt: bool = False,
    max_retries: int = 5,
    timeout: int = 3600,
) -> Dict[str, Any]:
    """
    Evaluate samples using BigCodeBench remote Gradio endpoint.

    Args:
        samples_path: Path to the samples JSONL file
        split: Task split (instruct or complete)
        subset: Dataset subset (full or hard)
        gradio_endpoint: Gradio endpoint URL
        pass_k: Pass@k values, comma-separated
        parallel: Number of parallel workers (-1 for auto)
        calibrated: Whether to use calibrated evaluation
        timeout: Timeout in seconds for the evaluation

    Returns:
        Dictionary with evaluation results
    """
    import time
    from concurrent.futures._base import CancelledError
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    if not GRADIO_AVAILABLE:
        raise ImportError("gradio_client and httpx are required. "
                         "Install with: pip install gradio_client httpx")

    logger.info(f"Connecting to Gradio endpoint: {gradio_endpoint}")
    logger.info(f"Samples: {samples_path}")
    logger.info(f"Split: {split}, Subset: {subset}")
    logger.info(f"Timeout: {timeout}s ({timeout // 60} minutes)")

    for attempt in range(max_retries):
        try:
            # Create client with longer httpx timeout
            httpx_kwargs = {"timeout": httpx.Timeout(timeout, connect=120.0)}
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
            logger.info("Waiting for evaluation results...")
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
                wait_time = 10 * (attempt + 1)  # Exponential backoff
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            logger.error(f"Remote evaluation failed: {e}")
            raise

    raise RuntimeError("Remote evaluation failed after all retries")


def evaluate_single(
    samples_path: str,
    split: str = "instruct",
    subset: str = "full",
    timeout: int = 3600,
    calibrated: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a single samples file and save results."""

    samples_file = Path(samples_path)
    if not samples_file.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    # Determine output directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = samples_file.parent / "remote_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    eval_results = evaluate_remote(
        samples_path=samples_path,
        split=split,
        subset=subset,
        timeout=timeout,
        calibrated=calibrated,
    )

    # Save full results
    results_file = out_dir / f"{samples_file.stem}_remote_eval_results.json"
    with open(results_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Saved full results to {results_file}")

    # Save pass@k summary
    pass_at_k_file = out_dir / f"{samples_file.stem}_remote_pass_at_k.json"
    pass_at_k_data = eval_results.get("pass_at_k", {})
    pass_at_k_data["samples_file"] = str(samples_file)
    pass_at_k_data["split"] = split
    pass_at_k_data["subset"] = subset
    pass_at_k_data["calibrated"] = calibrated
    with open(pass_at_k_file, "w") as f:
        json.dump(pass_at_k_data, f, indent=2)
    logger.info(f"Saved pass@k summary to {pass_at_k_file}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for key, value in pass_at_k_data.items():
        if key not in ["failed_tasks", "samples_file"]:
            logger.info(f"  {key}: {value}")

    return eval_results


def evaluate_all(
    input_path: str,
    split: str = "instruct",
    subset: str = "full",
    timeout: int = 3600,
    calibrated: bool = True,
    skip_existing: bool = True,
):
    """Evaluate all samples in a results directory."""
    input_base = Path(input_path)

    # Find all model directories
    model_dirs = sorted([
        d for d in input_base.iterdir()
        if d.is_dir() and "__" in d.name
    ])

    if not model_dirs:
        logger.error(f"No model directories found in {input_base}")
        return

    logger.info(f"Found {len(model_dirs)} model directories")

    results_summary = []

    for model_dir in model_dirs:
        model_name = model_dir.name

        # Find samples file
        bcb_format_dir = model_dir / "bigcodebench_format"
        if not bcb_format_dir.exists():
            logger.warning(f"SKIP: No bigcodebench_format dir for {model_name}")
            continue

        samples_files = list(bcb_format_dir.glob(f"*--bigcodebench-{subset}-{split}--0.0-1.jsonl"))
        if not samples_files:
            logger.warning(f"SKIP: No samples file for {model_name}")
            continue

        samples_file = samples_files[0]

        # Check if evaluation already exists
        remote_eval_dir = bcb_format_dir / "remote_eval"
        pass_at_k_file = remote_eval_dir / f"{samples_file.stem}_remote_pass_at_k.json"

        if skip_existing and pass_at_k_file.exists():
            logger.info(f"SKIP: Remote evaluation exists for {model_name}")
            # Load existing results for summary
            with open(pass_at_k_file) as f:
                existing = json.load(f)
                results_summary.append({
                    "model": model_name,
                    "pass@1": existing.get("pass@1", "N/A"),
                    "gt_pass_rate": existing.get("gt_pass_rate", "N/A"),
                })
            continue

        logger.info("=" * 60)
        logger.info(f"Evaluating: {model_name}")
        logger.info("=" * 60)

        try:
            eval_results = evaluate_single(
                samples_path=str(samples_file),
                split=split,
                subset=subset,
                timeout=timeout,
                calibrated=calibrated,
                output_dir=str(remote_eval_dir),
            )

            pass_at_k = eval_results.get("pass_at_k", {})
            results_summary.append({
                "model": model_name,
                "pass@1": pass_at_k.get("pass@1", "N/A"),
                "gt_pass_rate": pass_at_k.get("gt_pass_rate", "N/A"),
            })

        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                "model": model_name,
                "pass@1": "ERROR",
                "gt_pass_rate": "ERROR",
                "error": str(e),
            })
            continue

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Model':<50} {'pass@1':<10} {'GT Rate':<10}")
    logger.info("-" * 70)
    for r in sorted(results_summary, key=lambda x: x.get("pass@1", 0) if isinstance(x.get("pass@1"), float) else 0, reverse=True):
        pass1 = r["pass@1"]
        if isinstance(pass1, float):
            pass1 = f"{pass1:.1%}"
        gt = r["gt_pass_rate"]
        if isinstance(gt, float):
            gt = f"{gt:.1%}"
        logger.info(f"{r['model']:<50} {pass1:<10} {gt:<10}")

    # Save summary
    summary_file = input_base / f"remote_eval_summary_{split}_{subset}.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"\nSaved summary to {summary_file}")

    logger.info("\nAll remote evaluations completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Run BigCodeBench remote evaluation via Gradio endpoint"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to results directory (evaluates all models)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help="Path to a single samples file to evaluate",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="instruct",
        choices=["instruct", "complete"],
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
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for remote evaluation (default: 3600 = 1 hour)",
    )
    parser.add_argument(
        "--no_calibrated",
        action="store_true",
        default=False,
        help="Disable calibrated evaluation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-evaluation even if results exist",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: samples_dir/remote_eval)",
    )

    args = parser.parse_args()

    if not GRADIO_AVAILABLE:
        logger.error("Required packages not installed. Run: pip install gradio_client httpx")
        return 1

    calibrated = not args.no_calibrated

    if args.samples:
        evaluate_single(
            samples_path=args.samples,
            split=args.split,
            subset=args.subset,
            timeout=args.timeout,
            calibrated=calibrated,
            output_dir=args.output_dir,
        )
    elif args.input_path:
        evaluate_all(
            input_path=args.input_path,
            split=args.split,
            subset=args.subset,
            timeout=args.timeout,
            calibrated=calibrated,
            skip_existing=not args.force,
        )
    else:
        parser.error("Either --input_path or --samples is required")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
