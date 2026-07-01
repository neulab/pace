"""PlanBench handler for ProxyBench."""

import json
import os
import re

_HANDLERS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.normpath(os.path.join(_HANDLERS_DIR, "..", ".."))

PLANBENCH_RAW_DIR = os.path.join(_ROOT_DIR, "results", "raw_results", "planbench")

PLANBENCH_TASKS = (
    "task_1_plan_generation",
    "task_2_plan_optimality",
    "task_3_plan_verification",
    "task_4_plan_reuse",
    "task_5_plan_generalization",
    "task_6_replanning",
    "task_7_plan_execution",
    "task_8_1_goal_shuffling",
    "task_8_2_full_to_partial",
    "task_8_3_partial_to_full",
)

PLANBENCH_DEFAULT_SUBTASK = "task_1_plan_generation"


def _run_planbench(
    model_name: str,
    base_url: str,
    api_key: str,
    subtask: str,
    instance_id: str,
) -> list:
    """Run a single PlanBench instance.

    instance_id is the integer instance_id stored in each JSON entry.
    Query prompts are loaded from any existing raw-result file for the given subtask.

    Evaluation:
      task_3_plan_verification: binary valid/invalid string match.
      All other tasks: returns llm_correct=None (requires VAL + Fast Downward).
    """
    from openai import OpenAI
    from evaluations.handlers._compat import chat_completion

    if subtask not in PLANBENCH_TASKS:
        raise ValueError(
            f"PlanBench subtask '{subtask}' not recognized. "
            f"Valid subtasks: {PLANBENCH_TASKS}"
        )

    target_id = int(instance_id)

    instance_data = None
    for provider in sorted(os.listdir(PLANBENCH_RAW_DIR)):
        provider_dir = os.path.join(PLANBENCH_RAW_DIR, provider)
        if not os.path.isdir(provider_dir):
            continue
        for model_dir in sorted(os.listdir(provider_dir)):
            task_file = os.path.join(provider_dir, model_dir, f"{subtask}.json")
            if not os.path.exists(task_file):
                continue
            with open(task_file) as f:
                data = json.load(f)
            for inst in data.get("instances", []):
                if inst.get("instance_id") == target_id:
                    instance_data = {
                        "instance_id":       inst["instance_id"],
                        "query":             inst["query"],
                        "ground_truth_plan": inst.get("ground_truth_plan", ""),
                        "task":              data.get("task", subtask),
                        "domain":            data.get("domain", ""),
                    }
                    for k in ("parsed_ground_truth_plan", "example_instance_ids"):
                        if k in inst:
                            instance_data[k] = inst[k]
                    break
            if instance_data:
                break
        if instance_data:
            break

    if instance_data is None:
        raise ValueError(
            f"PlanBench instance not found: subtask='{subtask}', instance_id={target_id}. "
            f"Check that raw results exist under {PLANBENCH_RAW_DIR}."
        )

    query = instance_data["query"]
    ground_truth_plan = instance_data["ground_truth_plan"]

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = chat_completion(
        client,
        model=model_name,
        messages=[{"role": "user", "content": query}],
        temperature=0,
        max_tokens=2048,
        timeout=300,
    )
    llm_raw_response = response.choices[0].message.content or ""

    result = dict(instance_data)
    result["llm_raw_response"] = llm_raw_response
    result["extracted_llm_plan"] = None

    if subtask == "task_3_plan_verification":
        def _parse_validity(text: str):
            t = text.lower()
            if re.search(r'\bthe (above )?plan is (not valid|invalid)\b', t):
                return False
            if re.search(r'\bthe (above )?plan is valid\b', t):
                return True
            if re.search(r'\binvalid\b', t):
                return False
            if re.search(r'\bvalid\b', t):
                return True
            return None

        gt_valid = _parse_validity(ground_truth_plan)
        llm_valid = _parse_validity(llm_raw_response)
        if gt_valid is not None and llm_valid is not None:
            result["llm_correct_binary"] = (gt_valid == llm_valid)
        else:
            result["llm_correct_binary"] = None
        result["llm_correct"] = None
    else:
        result["llm_correct"] = None

    return [result]
