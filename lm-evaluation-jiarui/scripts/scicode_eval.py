#!/usr/bin/env python3
"""
SciCode Local Evaluation Module

Local implementation of SciCode evaluation functions.
Faithful to: https://github.com/scicode-bench/SciCode
Based on: eval/scripts/test_generated_code.py
"""

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Constants from original SciCode
PROB_NUM = 80
DEV_PROB_NUM = 15
STEP_NUM = 288
DEV_STEP_NUM = 50
DEFAULT_TIMEOUT = 1800  # 30 minutes per test, same as original


def get_background_dir(with_background: bool) -> str:
    """Get directory suffix based on background mode."""
    return "with_background" if with_background else "without_background"


def run_script(script_path: Path, timeout: int = DEFAULT_TIMEOUT) -> Tuple[int, str]:
    """
    Run a Python script and return status code with error details.
    Faithful to original test_generated_code.py run_script function.

    Returns:
        Tuple of (status_code, error_details):
        - 0: pass
        - 1: fail (CalledProcessError)
        - 2: timeout
    """
    try:
        subprocess.run(
            ['python', str(script_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return 0, ""
    except subprocess.CalledProcessError as e:
        logger.debug(f"Error running script {script_path}: {e}")
        logger.debug(f"Output: {e.output}")
        error_details = f"=== STDOUT ===\n{e.stdout}\n=== STDERR ===\n{e.stderr}"
        return 1, error_details
    except subprocess.TimeoutExpired as e:
        logger.debug(f"Timeout running script {script_path}: {e}")
        return 2, f"Timeout after {timeout} seconds"


def create_test_file(
    code_content: str,
    step_id: str,
    test_cases: List[str],
    output_path: Path,
    h5py_file_path: str = "eval/data/test_data.h5",
) -> Path:
    """
    Create a test file by appending test cases to generated code.
    Faithful to original test_generated_code.py logic.
    """
    test_file = output_path / f"{step_id}.py"

    with open(test_file, 'w', encoding='utf-8') as f:
        # Write the generated code
        f.write(code_content)

        # Append the test harness (faithful to original)
        f.write(f"""

from scicode.parse.parse import process_hdf5_to_tuple

""")
        # Get targets for all test cases
        f.write(f"targets = process_hdf5_to_tuple('{step_id}', {len(test_cases)}, '{h5py_file_path}')\n")

        # Write each test case
        for idx in range(len(test_cases)):
            f.write(f"target = targets[{idx}]\n\n")
            for line in test_cases[idx].split('\n'):
                f.write(line + '\n')

    return test_file


def test_generated_code(
    code_dir: Path,
    scicode_data: List[Dict],
    split: str = "test",
    with_background: bool = False,
    h5py_file_path: str = "eval/data/test_data.h5",
    log_dir: Optional[Path] = None,
    timeout: int = DEFAULT_TIMEOUT,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Test generated code against SciCode test cases.
    Faithful to original test_generated_code.py test_code function.

    Args:
        code_dir: Directory containing generated code files (e.g., 1.1.py, 1.2.py)
        scicode_data: List of problem data from SciCode dataset
        split: Dataset split ("validation" or "test")
        with_background: Whether background prompts were used
        h5py_file_path: Path to the HDF5 file with test data
        log_dir: Optional directory to cache test results
        timeout: Timeout per test in seconds (default: 1800)
        verbose: Whether to print progress

    Returns:
        Dictionary with evaluation results
    """
    # Build lookup dictionaries (faithful to original)
    json_dct = {}  # problem_id -> num_steps
    json_idx = {}  # problem_id -> index in scicode_data

    for idx, prob_data in enumerate(scicode_data):
        prob_id = prob_data['problem_id']
        json_dct[prob_id] = len(prob_data['sub_steps'])
        json_idx[prob_id] = idx

    start_time = time.time()

    # Create temporary directory for test files
    tmp_dir = Path(f'tmp_{start_time}')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Create log directory if specified
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Collect all code files and create test files
        code_files = list(code_dir.glob("*.py"))

        if verbose:
            logger.info(f"Found {len(code_files)} code files to test")

        for file_path in code_files:
            if not file_path.is_file():
                continue

            file_name = file_path.stem  # e.g., "1.1" from "1.1.py"
            parts = file_name.split(".")
            if len(parts) != 2:
                continue

            prob_id = parts[0]
            step_num = parts[1]

            if prob_id not in json_idx:
                logger.warning(f"Problem {prob_id} not found in dataset, skipping {file_name}")
                continue

            # Read generated code
            code_content = file_path.read_text(encoding='utf-8')

            # Get test cases from dataset
            prob_data = scicode_data[json_idx[prob_id]]
            step_idx = int(step_num) - 1

            if step_idx >= len(prob_data["sub_steps"]):
                logger.warning(f"Step {step_num} not found for problem {prob_id}")
                continue

            step_data = prob_data["sub_steps"][step_idx]
            step_id = step_data["step_number"]  # e.g., "1.1"
            test_cases = step_data["test_cases"]

            # Create test file
            create_test_file(
                code_content=code_content,
                step_id=step_id,
                test_cases=test_cases,
                output_path=tmp_dir,
                h5py_file_path=h5py_file_path,
            )

        # Run all test files (faithful to original logic)
        correct_prob = np.zeros(PROB_NUM)
        tot_prob = np.zeros(PROB_NUM)
        correct_step = []
        correct_dict = {f'{i+1}': [] for i in range(PROB_NUM)}

        test_files = list(tmp_dir.glob("*.py"))

        if verbose:
            logger.info(f"Running {len(test_files)} tests...")

        for file_path in test_files:
            if not file_path.is_file():
                continue

            func_id = file_path.stem  # e.g., "1.1"
            prob_id = func_id.split('.')[0]

            if verbose:
                logger.info(f'Testing function {func_id} ...')

            prob_idx = int(prob_id) - 1
            if prob_idx < 0 or prob_idx >= PROB_NUM:
                continue

            tot_prob[prob_idx] += 1

            # Check cache if log_dir specified (faithful to original)
            if log_dir:
                log_file = log_dir / f'{func_id}.txt'
                if log_file.exists():
                    content = log_file.read_text().splitlines()
                    if content and content[0] == 'pass':
                        correct_prob[prob_idx] += 1
                        correct_step.append(func_id)
                        correct_dict[prob_id].append(func_id)
                    continue

            # Run the test
            ret, error_details = run_script(file_path, timeout=timeout)

            if ret == 0:  # pass
                correct_prob[prob_idx] += 1
                correct_step.append(func_id)
                correct_dict[prob_id].append(func_id)
                if log_dir:
                    (log_dir / f'{func_id}.txt').write_text('pass')
            elif ret == 1:  # fail
                if log_dir:
                    (log_dir / f'{func_id}.txt').write_text(f'fail\n{error_details}')
            else:  # timeout
                if log_dir:
                    (log_dir / f'{func_id}.txt').write_text(f'time out\n{error_details}')

        test_time = time.time() - start_time

        # Calculate metrics (faithful to original)
        correct_prob_num = sum(
            1 for i in range(PROB_NUM)
            if correct_prob[i] == tot_prob[i] and tot_prob[i] != 0
        )

        # Determine expected totals based on split
        if split == "validation":
            expected_probs = DEV_PROB_NUM
            expected_steps = DEV_STEP_NUM
        else:
            expected_probs = PROB_NUM - DEV_PROB_NUM
            expected_steps = STEP_NUM

        # Build results
        results = {
            "split": split,
            "with_background": with_background,
            "correct_problems": correct_prob_num,
            "total_problems": expected_probs,
            "problem_accuracy": correct_prob_num / expected_probs if expected_probs > 0 else 0,
            "correct_steps": len(correct_step),
            "total_steps": int(tot_prob.sum()),
            "expected_steps": expected_steps,
            "step_accuracy": len(correct_step) / tot_prob.sum() if tot_prob.sum() > 0 else 0,
            "test_duration_seconds": test_time,
            "correct_step_ids": correct_step,
            "correct_by_problem": {k: v for k, v in correct_dict.items() if v},
            "problems_all_correct": [
                i + 1 for i in range(PROB_NUM)
                if correct_prob[i] == tot_prob[i] and tot_prob[i] != 0
            ],
        }

        if verbose:
            logger.info(f"\nEvaluation Results:")
            logger.info(f"  Correct problems: {correct_prob_num}/{expected_probs}")
            logger.info(f"  Correct steps: {len(correct_step)}/{int(tot_prob.sum())}")
            logger.info(f"  Problem accuracy: {results['problem_accuracy']:.4f}")
            logger.info(f"  Step accuracy: {results['step_accuracy']:.4f}")
            logger.info(f"  Duration: {test_time:.2f}s")

        return results

    finally:
        # Cleanup temporary directory (faithful to original)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def evaluate_scicode(
    code_dir: Path,
    split: str = "test",
    with_background: bool = False,
    h5py_file_path: Optional[str] = None,
    log_dir: Optional[Path] = None,
    timeout: int = DEFAULT_TIMEOUT,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Main evaluation function for SciCode.
    Loads dataset and runs evaluation.

    Args:
        code_dir: Directory containing generated code files
        split: Dataset split ("validation" or "test")
        with_background: Whether background prompts were used
        h5py_file_path: Path to HDF5 test data file (required)
        log_dir: Optional directory to cache test results
        timeout: Timeout per test in seconds
        verbose: Whether to print progress

    Returns:
        Dictionary with evaluation results
    """
    # Try to import scicode for data loading
    try:
        from scicode.parse.parse import read_from_hf_dataset, H5PY_FILE

        # Use default H5PY file path if not specified
        if h5py_file_path is None:
            h5py_file_path = H5PY_FILE

    except ImportError:
        # Fallback to datasets directly
        from datasets import load_dataset

        if h5py_file_path is None:
            raise ValueError(
                "h5py_file_path must be specified when scicode package is not installed. "
                "Install with: pip install scicode"
            )

        def read_from_hf_dataset(split):
            return load_dataset('SciCode1/SciCode', split=split)

    # Check H5PY file exists
    if not Path(h5py_file_path).exists():
        raise FileNotFoundError(
            f"Test data file not found: {h5py_file_path}\n"
            "Please download the numeric test results before testing generated code.\n"
            "See: https://github.com/scicode-bench/SciCode#evaluation"
        )

    # Load dataset
    if verbose:
        logger.info(f"Loading SciCode dataset (split={split})...")

    scicode_data = list(read_from_hf_dataset(split))

    if verbose:
        logger.info(f"Loaded {len(scicode_data)} problems")

    # Run evaluation
    return test_generated_code(
        code_dir=code_dir,
        scicode_data=scicode_data,
        split=split,
        with_background=with_background,
        h5py_file_path=h5py_file_path,
        log_dir=log_dir,
        timeout=timeout,
        verbose=verbose,
    )
