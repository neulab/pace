#!/usr/bin/env python3
"""
Run local BigCodeBench evaluation for all existing samples.
Uses bigcodebench.evaluate directly to avoid Fire CLI issues.

Usage:
    python run_bigcodebench_local_eval.py --input_path /path/to/results
    python run_bigcodebench_local_eval.py --samples /path/to/samples.jsonl --split instruct --subset full
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_single(
    samples_path: str,
    split: str = "instruct",
    subset: str = "full",
    pass_k: str = "1",
    parallel: int = -1,
    calibrated: bool = True,
    force: bool = False,
):
    """Evaluate a single samples file using bigcodebench."""
    import os
    from bigcodebench.evaluate import evaluate

    logger.info(f"Evaluating: {samples_path}")
    logger.info(f"  Split: {split}, Subset: {subset}")
    logger.info(f"  Pass@k: {pass_k}, Parallel: {parallel}, Calibrated: {calibrated}")

    # If force=True, delete existing result files to avoid interactive prompt
    # bigcodebench.evaluate prompts "Press [Y/N] to overwrite" if results exist
    if force:
        samples_file = Path(samples_path)
        result_files = [
            samples_file.parent / (samples_file.stem + "_eval_results.json"),
            samples_file.parent / (samples_file.stem + "_pass_at_k.json"),
            samples_file.parent / "pass_at_k.txt",
        ]
        for rf in result_files:
            if rf.exists():
                logger.info(f"Removing existing result file: {rf}")
                os.remove(rf)

    # Call evaluate directly to avoid Fire CLI issues
    evaluate(
        split=split,
        subset=subset,
        samples=samples_path,
        execution="local",
        pass_k=pass_k,
        parallel=parallel,
        calibrated=calibrated,
    )

    logger.info(f"Completed: {samples_path}")


def evaluate_all(
    input_path: str,
    split: str = "instruct",
    subset: str = "full",
    pass_k: str = "1",
    parallel: int = -1,
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
        # Note: samples files are like "model--bigcodebench-full-instruct--0.0-1.jsonl"
        # Eval results are saved as "model--bigcodebench-full-instruct--0.0-1_eval_results.json"
        eval_file = samples_file.parent / (samples_file.stem + "_eval_results.json")
        if skip_existing and eval_file.exists():
            logger.info(f"SKIP: Evaluation exists for {model_name}")
            continue

        logger.info("=" * 60)
        logger.info(f"Evaluating: {model_name}")
        logger.info("=" * 60)

        try:
            evaluate_single(
                samples_path=str(samples_file),
                split=split,
                subset=subset,
                pass_k=pass_k,
                parallel=parallel,
                calibrated=calibrated,
                force=not skip_existing,
            )
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("All evaluations completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Run local BigCodeBench evaluation"
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
        "--pass_k",
        type=str,
        default="1",
        help="Pass@k values, comma-separated (default: 1)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=-1,
        help="Number of parallel workers (-1 for auto, default: -1)",
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

    args = parser.parse_args()

    calibrated = not args.no_calibrated

    if args.samples:
        evaluate_single(
            samples_path=args.samples,
            split=args.split,
            subset=args.subset,
            pass_k=args.pass_k,
            parallel=args.parallel,
            calibrated=calibrated,
            force=args.force,
        )
    elif args.input_path:
        evaluate_all(
            input_path=args.input_path,
            split=args.split,
            subset=args.subset,
            pass_k=args.pass_k,
            parallel=args.parallel,
            calibrated=calibrated,
            skip_existing=not args.force,
        )
    else:
        parser.error("Either --input_path or --samples is required")


if __name__ == "__main__":
    main()
