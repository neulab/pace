"""LIFBench handler for ProxyBench."""

import json
import os
import sys

_HANDLERS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVALUATIONS_DIR = os.path.dirname(_HANDLERS_DIR)
BENCHMARKS_DIR = os.path.join(_EVALUATIONS_DIR, "benchmarks")

LIFBENCH_DIR = os.path.join(BENCHMARKS_DIR, "lifbench")
LIFBENCH_PROMPTS_DIR = os.path.join(LIFBENCH_DIR, "data", "prompts")
LIFBENCH_EVAL_DIR = os.path.join(LIFBENCH_DIR, "evaluation")

LIFBENCH_FUNC_MAP = {
    "onedoc-repeat":                  "judge_onedoc_repeat",
    "onedoc-qa":                      "judge_onedoc_qa",
    "onedoc-extract":                 "judge_onedoc_extract",
    "list-single_query_id":           "judge_label_equal_output",
    "list-multi_query_id":            "judge_labels_equal_outputs",
    "list-offset_query_id":           "judge_label_equal_output",
    "list-offset_query_element":      "judge_label_equal_output",
    "list-blur_offset_query_id":      "judge_list_input_blur_offset_query",
    "list-blur_offset_query_element": "judge_list_input_blur_offset_query",
    "multidoc-batch_label":           "judge_multidoc_batch_label",
    "multidoc-find_dup_text":         "judge_multidoc_find_dup_text",
}

LIFBENCH_DEFAULT_SUBTASK = "onedoc-qa"


def _run_lifbench(
    model_name: str,
    base_url: str,
    api_key: str,
    subtask: str,
    instance_id: str,
) -> list:
    """Run a single LIFBench instance: generate output then score with EvaluateFunc.

    instance_id format: "{ins_id}_{param_id}_{length}", e.g. "0_0_3".
    """
    from openai import OpenAI

    if subtask not in LIFBENCH_FUNC_MAP:
        raise ValueError(
            f"LIFBench subtask '{subtask}' not recognized. "
            f"Valid subtasks: {list(LIFBENCH_FUNC_MAP.keys())}"
        )

    parts = instance_id.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"LIFBench instance_id must be in '{{ins_id}}_{{param_id}}_{{length}}' format, "
            f"got: '{instance_id}'"
        )
    try:
        length_str = parts[2].split("#")[0]
        ins_id = int(parts[0])
        param_id = int(parts[1])
        length = int(length_str)
    except ValueError:
        raise ValueError(
            f"LIFBench instance_id components must be integers, got: '{instance_id}'"
        )

    prompt_file = os.path.join(LIFBENCH_PROMPTS_DIR, f"{subtask}.json")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"LIFBench prompt file not found: {prompt_file}")

    with open(prompt_file) as f:
        prompts = json.load(f)

    entry = None
    for item in prompts:
        if (item.get("ins_id") == ins_id and
                item.get("param_id") == param_id and
                item.get("length") == length):
            entry = dict(item)
            break

    if entry is None:
        raise ValueError(
            f"LIFBench instance not found: ins_id={ins_id}, param_id={param_id}, "
            f"length={length} in {prompt_file}"
        )

    from evaluations.handlers._compat import chat_completion
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = chat_completion(
        client,
        model=model_name,
        messages=[{"role": "user", "content": entry["prompt"]}],
        temperature=0,
        max_tokens=2048,
        timeout=300,
    )
    output = response.choices[0].message.content or ""

    if LIFBENCH_EVAL_DIR not in sys.path:
        sys.path.insert(0, LIFBENCH_EVAL_DIR)

    import importlib
    eval_mod = importlib.import_module("EvaluateFunc")
    evaluate_func = getattr(eval_mod, LIFBENCH_FUNC_MAP[subtask])

    entry["output"] = output
    entry["score_dict"] = evaluate_func(entry)

    return [entry]
