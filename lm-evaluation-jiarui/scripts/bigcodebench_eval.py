#!/usr/bin/env python3
"""
BigCodeBench Local Evaluation Module

Local implementation of BigCodeBench evaluation functions.
Adapted from: https://github.com/bigcode-project/bigcodebench
"""

# The MIT License
# Copyright (c) OpenAI (https://openai.com)
# Adapted for local use

import contextlib
import faulthandler
import io
import itertools
import json
import multiprocessing
import os
import platform
import signal
import subprocess
import tempfile
import time
import types
import unittest
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Value
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

# ============ Constants ============

TIMEOUT_LIMIT = 240.0

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


# ============ Utility Functions ============

class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def swallow_subprocess_output():
    """Context manager to swallow stdout and stderr for subprocesses."""
    original_popen = subprocess.Popen
    original_run = subprocess.run

    def _popen_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
        return original_popen(*args, **kwargs)

    def _run_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
        return original_run(*args, **kwargs)

    subprocess.Popen = _popen_patch
    subprocess.run = _run_patch
    try:
        yield
    finally:
        subprocess.Popen = original_popen
        subprocess.run = original_run


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                with swallow_subprocess_output():
                    yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


@contextlib.contextmanager
def safe_environment():
    """Safe environment that prevents harmful operations during code execution."""
    original_kill = os.kill
    original_killpg = os.killpg
    original_system = os.system
    original_subprocess_call = subprocess.call
    original_subprocess_check_output = subprocess.check_output
    original_subprocess_run = subprocess.run
    original_subprocess_popen = subprocess.Popen
    original_os_popen = os.popen
    original_os_execv = os.execv
    original_os_execvp = os.execvp
    original_os_execvpe = os.execvpe

    current_pid = os.getpid()
    current_pgid = os.getpgid(current_pid)
    manager = multiprocessing.Manager()
    child_pids = manager.list()

    def safe_kill(pid, sig):
        try:
            if pid == current_pid or pid in child_pids:
                original_kill(pid, sig)
        except ProcessLookupError:
            pass

    def safe_killpg(pgid, sig):
        if pgid == current_pgid or pgid in {os.getpgid(pid) for pid in child_pids}:
            original_killpg(pgid, sig)

    def safe_system(command):
        if 'kill' in command or 'killall' in command:
            return 0
        return original_system(command)

    def safe_subprocess_call(command, *args, **kwargs):
        if 'kill' in command or 'killall' in command:
            return 0
        return original_subprocess_call(command, *args, **kwargs)

    def safe_subprocess_check_output(command, *args, **kwargs):
        if 'ps' in command:
            return b""
        return original_subprocess_check_output(command, *args, **kwargs)

    def safe_subprocess_run(*args, **kwargs):
        if 'kill' in args[0] or 'killall' in args[0]:
            return subprocess.CompletedProcess(args, 0, b'', b'')
        return original_subprocess_run(*args, **kwargs)

    class SafePopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            kwargs['preexec_fn'] = os.setsid
            super().__init__(*args, **kwargs)
            child_pids.append(self.pid)

        def communicate(self, *args, **kwargs):
            try:
                return super().communicate(*args, **kwargs)
            except subprocess.TimeoutExpired:
                return None, None

        def kill(self):
            safe_kill(self.pid, signal.SIGTERM)

        def terminate(self):
            safe_kill(self.pid, signal.SIGTERM)

    def safe_os_popen(command):
        if 'kill' in command or 'killall' in command:
            return os.popen('echo Intercepted')
        return original_os_popen(command)

    def safe_exec(*args, **kwargs):
        pass

    os.kill = safe_kill
    os.killpg = safe_killpg
    os.system = safe_system
    subprocess.call = safe_subprocess_call
    subprocess.check_output = safe_subprocess_check_output
    subprocess.run = safe_subprocess_run
    subprocess.Popen = SafePopen
    os.popen = safe_os_popen
    os.execv = safe_exec
    os.execvp = safe_exec
    os.execvpe = safe_exec

    try:
        yield
    finally:
        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(10):
                    time.sleep(0.1)
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                else:
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                pass

        os.kill = original_kill
        os.killpg = original_killpg
        os.system = original_system
        subprocess.call = original_subprocess_call
        subprocess.check_output = original_subprocess_check_output
        subprocess.run = original_subprocess_run
        subprocess.Popen = original_subprocess_popen
        os.popen = original_os_popen
        os.execv = original_os_execv
        os.execvp = original_os_execvp
        os.execvpe = original_os_execvpe


def reliability_guard(max_as_limit, max_data_limit, max_stack_limit):
    """
    Disables various destructive functions and prevents generated code
    from interfering with the test.
    """
    os.environ['TZ'] = 'UTC'
    time.tzset()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

    if max_as_limit and max_data_limit and max_stack_limit:
        import resource

        max_as_limit = max_as_limit * 1024 * 1024
        max_data_limit = max_data_limit * 1024 * 1024
        max_stack_limit = max_stack_limit * 1024 * 1024

        resource.setrlimit(resource.RLIMIT_AS, (max_as_limit, max_as_limit))
        resource.setrlimit(resource.RLIMIT_DATA, (max_data_limit, max_data_limit))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (max_stack_limit, max_stack_limit))

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except ImportError:
        pass


# ============ Core Evaluation Functions ============

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    Unbiased estimator from https://github.com/openai/human-eval
    """

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def unsafe_execute(
    entry_point: str,
    code: str,
    test_code: str,
    timeout: float,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    stat,
    details,
):
    """Execute code in a sandboxed environment."""
    with safe_environment(), create_tempdir():
        import os
        import shutil
        import builtins
        import sys

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir_func = os.chdir
        reliability_guard(max_as_limit, max_data_limit, max_stack_limit)
        module_name = "__test__"
        new_module = types.ModuleType(module_name)
        new_module.__dict__.update({
            '__builtins__': builtins,
            '__file__': f"{module_name}.py",
            '__package__': None,
            '__doc__': None,
            'sys': sys,
            'os': os,
            'environ': os.environ,
        })

        try:
            full_code = code + "\n" + test_code

            with swallow_io():
                exec(compile(full_code, f"{module_name}.py", 'exec'), new_module.__dict__)
                sys.modules[module_name] = new_module
                TestCases = getattr(new_module, 'TestCases')
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(TestCases)
                test_result = unittest.TestResult()
                with time_limit(timeout):
                    suite.run(test_result)

            issues = test_result.failures + test_result.errors
            for test, trace in issues:
                details[test.id().split(".")[-1]] = trace
            stat.value = _SUCCESS
        except BaseException as e:
            details["ALL"] = str(e)
            stat.value = _FAILED
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir_func


def untrusted_check(
    code: str,
    test_code: str,
    entry_point: str,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    min_time_limit: float = 10,
    gt_time_limit: float = 60
) -> Tuple[str, Dict]:
    """Run untrusted code check in a subprocess with timeout."""
    min_time_limit = max(min_time_limit, gt_time_limit)
    timeout = max(os.getenv("BIGCODEBENCH_TIMEOUT_PER_TASK", TIMEOUT_LIMIT), min_time_limit) + 1
    stat = Value("i", _UNKNOWN)
    manager = Manager()
    details = manager.dict()

    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            entry_point,
            code,
            test_code,
            timeout,
            max_as_limit,
            max_data_limit,
            max_stack_limit,
            stat,
            details,
        ),
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    stat = _mapping[stat.value]
    details = dict(details)

    if not stat:
        stat = TIMEOUT
    if stat == PASS:
        if details:
            stat = FAIL

    return stat, details


def check_correctness(
    completion_id: int,
    problem: Dict[str, Any],
    solution: str,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    identifier: Optional[str] = None,
    min_time_limit: float = 0.1,
    gt_time_limit: float = 2.0,
) -> Dict[str, Any]:
    """Check if a solution is correct against the test cases."""
    ret = {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
    }
    ret["base"] = untrusted_check(
        solution,
        problem["test"],
        problem["entry_point"],
        max_as_limit,
        max_data_limit,
        max_stack_limit,
        min_time_limit,
        gt_time_limit,
    )
    return ret


# ============ Data Loading Functions ============

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """Parses each jsonl line and yields it as a dictionary."""
    import gzip

    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def load_solutions(sample_path: str) -> Iterable[Dict]:
    """Load solutions from a jsonl file or directory."""
    if os.path.isfile(sample_path):
        for i, sample in enumerate(stream_jsonl(sample_path)):
            assert (
                "completion" in sample or "solution" in sample
            ), "No completion or solution found in sample!"
            sample["_identifier"] = (
                sample["task_id"] + f" (line {i+1} in {sample_path})"
            )
            yield sample
    else:
        for task_id in os.listdir(sample_path):
            task_path = os.path.join(sample_path, task_id)
            if not os.path.isdir(task_path):
                continue

            for solution_id in os.listdir(task_path):
                solution_path = os.path.join(task_path, solution_id)
                if os.path.isfile(solution_path) and solution_path.endswith(".py"):
                    with open(solution_path, "r") as f:
                        completion = f.read()
                    yield {
                        "_identifier": solution_path,
                        "_path": solution_path,
                        "task_id": task_id.replace("_", "/"),
                        "solution": completion,
                    }


# ============ Main Evaluation Function ============

def evaluate_samples(
    samples_path: str,
    problems: Dict[str, Dict],
    split: str = "instruct",
    subset: str = "full",
    pass_k: List[int] = [1],
    parallel: int = -1,
    min_time_limit: float = 1,
    max_as_limit: int = 30 * 1024,
    max_data_limit: int = 30 * 1024,
    max_stack_limit: int = 10,
    calibrated: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate generated samples against test cases.

    Args:
        samples_path: Path to the jsonl file with generated samples
        problems: Dictionary of problems from BigCodeBench dataset
        split: "complete" or "instruct"
        subset: "full" or "hard"
        pass_k: List of k values for pass@k calculation
        parallel: Number of parallel workers (-1 for auto)
        min_time_limit: Minimum time limit per test
        max_as_limit: Maximum address space limit in MB
        max_data_limit: Maximum data segment limit in MB
        max_stack_limit: Maximum stack limit in MB
        calibrated: Whether to prepend code_prompt to solutions
        verbose: Whether to print progress

    Returns:
        Dictionary with evaluation results including pass@k metrics
    """
    from datetime import datetime
    from tqdm import tqdm

    if parallel < 1:
        n_workers = max(1, multiprocessing.cpu_count() // 2)
    else:
        n_workers = parallel

    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "eval": {},
    }

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        eval_results = defaultdict(list)

        if verbose:
            print("Reading samples...")

        samples_list = list(load_solutions(samples_path))

        for sample in (tqdm(samples_list, desc="Submitting") if verbose else samples_list):
            task_id = sample["task_id"]

            if task_id not in problems:
                continue

            solution = (
                sample["solution"]
                if "solution" in sample
                else problems[task_id]["complete_prompt"] + sample["completion"]
            )
            if calibrated:
                solution = problems[task_id]["code_prompt"] + "\n    pass\n" + solution

            args = (
                completion_id[task_id],
                problems[task_id],
                solution,
                max_as_limit,
                max_data_limit,
                max_stack_limit,
                sample["_identifier"],
                min_time_limit,
                20,  # default gt_time_limit
            )
            futures.append(executor.submit(check_correctness, *args))
            completion_id[task_id] += 1
            n_samples += 1

        if verbose:
            print(f"Evaluating {n_samples} samples...")

        for future in (tqdm(as_completed(futures), total=n_samples, desc="Evaluating") if verbose else as_completed(futures)):
            result = future.result()
            eval_results[result["task_id"]].append(result)

        for task_id, task_results in eval_results.items():
            task_results.sort(key=lambda x: x["completion_id"])
            results["eval"][task_id] = []
            for res in task_results:
                stat, details = res["base"]
                results["eval"][task_id].append(
                    {
                        "task_id": task_id,
                        "solution": res["solution"],
                        "status": stat,
                        "details": details,
                    }
                )

    # Calculate pass@k
    total = np.array([len(r) for k, r in results["eval"].items() if k in problems])
    base_correct = []

    for key, res in results["eval"].items():
        if key not in problems:
            continue
        bc = sum([r["status"] == PASS for r in res])
        base_correct.append(bc)

    base_correct = np.array(base_correct)

    pass_at_k_results = {}
    for k in pass_k:
        if total.min() >= k:
            pass_at_k_results[f"pass@{k}"] = float(estimate_pass_at_k(total, base_correct, k).mean())

    # Summary statistics
    num_passed = sum(1 for res_list in results["eval"].values()
                     for res in res_list if res["status"] == PASS)
    num_failed = sum(1 for res_list in results["eval"].values()
                     for res in res_list if res["status"] == FAIL)
    num_timeout = sum(1 for res_list in results["eval"].values()
                      for res in res_list if res["status"] == TIMEOUT)

    evaluation_results = {
        "split": split,
        "subset": subset,
        "calibrated": calibrated,
        "num_tasks": len(problems),
        "num_samples_evaluated": n_samples,
        "num_passed": num_passed,
        "num_failed": num_failed,
        "num_timeout": num_timeout,
        **pass_at_k_results,
        "detailed_results": results,
    }

    if verbose:
        print(f"\nEvaluation Results:")
        print(f"  Tasks: {len(problems)}")
        print(f"  Samples: {n_samples}")
        print(f"  Passed: {num_passed}")
        print(f"  Failed: {num_failed}")
        print(f"  Timeout: {num_timeout}")
        for k, v in pass_at_k_results.items():
            print(f"  {k}: {v:.4f}")

    return evaluation_results
