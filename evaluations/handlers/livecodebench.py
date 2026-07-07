"""LiveCodeBench handler for ProxyBench."""

import contextlib
import os
import sys

_HANDLERS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVALUATIONS_DIR = os.path.dirname(_HANDLERS_DIR)
BENCHMARKS_DIR = os.path.join(_EVALUATIONS_DIR, "benchmarks")

LCB_DIR = os.path.join(BENCHMARKS_DIR, "livecodebench")

LCB_SUBTASKS = ("codegeneration", "selfrepair", "testoutputprediction", "codeexecution")
LCB_DEFAULT_SUBTASK = "codegeneration"


@contextlib.contextmanager
def _in_lcb_dir():
    """Temporarily chdir to LCB_DIR so module-level file opens in lcb_runner succeed."""
    prev = os.getcwd()
    try:
        os.chdir(LCB_DIR)
        yield
    finally:
        os.chdir(prev)


def _run_livecodebench(
    model_name: str,
    base_url: str,
    api_key: str,
    subtask: str,
    instance_id: str,
) -> list:
    """Run a single LiveCodeBench instance: generate output then evaluate.

    instance_id is the 0-based integer index in the sorted problem list.
    """
    if subtask not in LCB_SUBTASKS:
        raise ValueError(
            f"LiveCodeBench subtask '{subtask}' not recognized. "
            f"Valid subtasks: {LCB_SUBTASKS}"
        )

    idx = int(instance_id)

    if LCB_DIR not in sys.path:
        sys.path.insert(0, LCB_DIR)

    with _in_lcb_dir():
        from lcb_runner.lm_styles import LMStyle
        from lcb_runner.utils.extraction_utils import (
            extract_code,
            extract_test_output_code,
            extract_execution_code,
        )

        # NOTE: the standardized results were produced with "release_latest"
        # (lcb_runner's parser default), giving 1055 codegen/selfrepair problems.
        # "release_v1" has only ~400 problems, so high indices (e.g. 974) went out
        # of range — the instance ids index into the release_latest ordering.
        if subtask in ("codegeneration", "selfrepair"):
            from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
            from lcb_runner.prompts.code_generation import format_prompt_generation
            dataset = sorted(load_code_generation_dataset("release_latest"), key=lambda x: x.question_id)
            problem = dataset[idx]
            messages = format_prompt_generation(problem, LMStyle.OpenAIChat)

        elif subtask == "testoutputprediction":
            from lcb_runner.benchmarks.test_output_prediction import load_test_prediction_dataset
            from lcb_runner.prompts.test_output_prediction import format_prompt_test_output
            dataset = sorted(load_test_prediction_dataset("release_latest"), key=lambda x: (x.question_id, x.test_id))
            problem = dataset[idx]
            messages = format_prompt_test_output(problem, LMStyle.OpenAIChat)

        elif subtask == "codeexecution":
            from lcb_runner.benchmarks.code_execution import load_code_execution_dataset
            from lcb_runner.prompts.code_execution import format_prompt_execution_cot
            dataset = sorted(load_code_execution_dataset("release_latest"), key=lambda x: int(x.id.split("_")[1]))
            problem = dataset[idx]
            messages = format_prompt_execution_cot(problem, LMStyle.OpenAIChat)

    from openai import OpenAI
    from evaluations.handlers._compat import chat_completion
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = chat_completion(
        client,
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=4096,
        timeout=300,
    )
    output = response.choices[0].message.content or ""

    with _in_lcb_dir():
        if subtask in ("codegeneration", "selfrepair"):
            code = extract_code(output, LMStyle.OpenAIChat)
            sample = problem.get_evaluation_sample()
            from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
            try:
                graded, _ = check_correctness(sample, code, timeout=6)
            except Exception as e:
                print(f"Warning: check_correctness failed: {e}", file=sys.stderr)
                graded = []
            graded_list = [bool(g) for g in graded] if graded else [False]
            result = problem.insert_output_evaluation([output], [code], graded_list)

        elif subtask == "testoutputprediction":
            pred = extract_test_output_code(output, LMStyle.OpenAIChat)
            expected = problem.get_evaluation_sample()["output"]
            from lcb_runner.evaluation.compute_test_output_prediction_metrics import check_testcase_output
            passed = bool(check_testcase_output(pred, expected))
            result = problem.insert_output_evaluation([output], [pred], [passed])

        elif subtask == "codeexecution":
            pred = extract_execution_code(output, LMStyle.OpenAIChat, cot=True)
            sample = problem.get_evaluation_sample()
            from lcb_runner.evaluation.utils_execute import BASE_IMPORTS, check_correctness as ce_check
            code_to_execute = f"{BASE_IMPORTS}\n{sample['code']}\nassert {sample['output']} == {pred}"
            try:
                passed = ce_check(code_to_execute, 3)
            except Exception as e:
                print(f"Warning: code execution check failed: {e}", file=sys.stderr)
                passed = False
            result = problem.insert_output_evaluation([output], [pred], [bool(passed)])

    result["instance_id"] = idx
    result["subtask"] = subtask
    return [result]
