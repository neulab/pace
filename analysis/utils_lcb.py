import os
import json
from typing import List, Dict, Tuple

DEFAULT_LCB_DIR = "/home/yueqis/LiveCodeBench/output/"

# -------------------------------------------------
# Mapping between LCB model dir names and canonical model names
# (Align canonical names with SWE / MMLU naming)
# -------------------------------------------------

LCB_TO_CANON = {
    "GPT-5__high": "GPT-5",
    "GPT-5.2__high": "GPT-5.2",
    # "GPT-5-mini__high": "GPT-5-mini",
    "o3__high": "o3",
    "o4-mini__high": "o4-mini",
    "GPT-OSS-120B": "GPT-OSS-120B",
    "Claude-Sonnet-4": "Claude-Sonnet-4",
    "Claude-Sonnet-4.5": "Claude-Sonnet-4.5",
    "Gemini-3-Pro-Preview": "Gemini-3-Pro-Preview",
    "Gemini-2.5-Pro": "Gemini-2.5-Pro",
    "Qwen3-Coder-480B-A35B-Instruct": "Qwen3-Coder-480B-A35B-Instruct",
    "Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B-Instruct",
    "Kimi-K2-Instruct": "Kimi-K2-Instruct",

}

CANON_TO_LCB = {v: k for k, v in LCB_TO_CANON.items()}

# -------------------------------------------------
# LCB tasks may include:
# codegeneration, codeexecution, selfrepair, testoutputprediction
# -------------------------------------------------


def _get_task_key(task: str, line: Dict) -> str:
    if task == "codeexecution":
        return str(line.get("id"))
    if task in ("codegeneration", "selfrepair"):
        return str(line.get("question_id"))
    if task == "testoutputprediction":
        return f"{line.get('question_id')}_{line.get('test_id')}"
    return str(line.get("id", ""))


def _load_model_results(model_dir_name: str, task: str, base_dir: str) -> Dict[str, float]:
    model_dir = os.path.join(base_dir, model_dir_name)
    if not os.path.exists(model_dir):
        return {}

    output: Dict[str, float] = {}

    # only use the first iteration (as in prior code)
    i = 0

    if task == "codeexecution":
        file_path = os.path.join(model_dir, f"{i}/{task}_1_cot_eval_all.json")
    else:
        file_path = os.path.join(model_dir, f"{i}/{task}_1_eval_all.json")

    if not os.path.exists(file_path):
        return {}

    with open(file_path) as f:
        data = json.load(f)

    for line in data:
        key = _get_task_key(task, line)

        # assume presence of numeric pass@1-like field; fall back to 0
        val = line.get("pass@1", line.get("accuracy", 0))

        try:
            output[str(key)] = float(val)
        except Exception:
            output[str(key)] = 0.0

    return output


# -------------------------------------------------
# Public API (aligned with utils_mmlu.py style)
# -------------------------------------------------


def list_canonical_models(lcb_dir: str = DEFAULT_LCB_DIR) -> List[str]:
    if not os.path.isdir(lcb_dir):
        return []

    models = []

    for d in os.listdir(lcb_dir):
        p = os.path.join(lcb_dir, d)
        if os.path.isdir(p) and d in LCB_TO_CANON:
            models.append(LCB_TO_CANON[d])

    models.sort()
    return models


def load_model_outputs_for_models(
    model_names: List[str],
    task: str,
    lcb_dir: str = DEFAULT_LCB_DIR
) -> Tuple[List[str], List[Dict[str, float]]]:

    kept_models: List[str] = []
    dicts: List[Dict[str, float]] = []

    for canon in model_names:
        lcb_name = CANON_TO_LCB.get(canon)
        if not lcb_name:
            continue

        d = _load_model_results(lcb_name, task, lcb_dir)

        if not d:
            continue

        dicts.append(d)
        kept_models.append(canon)

    return kept_models, dicts
