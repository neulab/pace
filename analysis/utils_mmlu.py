import os
import json
from typing import List, Dict, Tuple

DEFAULT_MMLU_BASE_DIR = "/workspace/project/proxy-bench/data/eval_scores"

# Mapping between MMLU model dir names and canonical model names (aligned to SWE CSV basenames)
MMLU_TO_CANON = {
    "azure--o3": "o3",
    "azure--o4-mini": "o4-mini",
    "azure--gpt-5": "GPT-5",
    "neulab--gpt-oss-120b": "GPT-OSS-120B",
    "neulab--qwen3-coder-480b-a35b-instruct": "Qwen3-Coder-480B-A35B-Instruct",
    "azure--gpt-5-mini": "GPT-5-mini",
    "neulab--claude-sonnet-4-20250514": "Claude-Sonnet-4",
    "neulab--claude-sonnet-4-5-20250929": "Claude-Sonnet-4.5",
}
CANON_TO_MMLU = {v: k for k, v in MMLU_TO_CANON.items()}


def _load_mmlu_model_results(model_dir_name: str, mmlu_dir: str = DEFAULT_MMLU_BASE_DIR, mmlu_task: str = "mmlu_electrical_engineering") -> Dict[str, float]:
    model_dir = os.path.join(mmlu_dir, mmlu_task, model_dir_name)
    file_path = os.path.join(model_dir, "output.jsonl")
    if not os.path.exists(file_path):
        return {}
    out: Dict[str, float] = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            sid = j.get("sample_id", j.get("idx"))
            if sid is None:
                continue
            acc = j.get("accuracy")
            if isinstance(acc, list):
                acc_val = float(acc[0]) if len(acc) > 0 else 0.0
            elif acc is None:
                ev = j.get("eval")
                acc2 = ev.get("accuracy") if isinstance(ev, dict) else None
                if isinstance(acc2, list):
                    acc_val = float(acc2[0]) if len(acc2) > 0 else 0.0
                elif acc2 is not None:
                    acc_val = float(acc2)
                else:
                    acc_val = 0.0
            else:
                acc_val = float(acc)
            out[str(sid)] = acc_val
    return out


def list_canonical_models(mmlu_dir: str = DEFAULT_MMLU_BASE_DIR, mmlu_task: str = "mmlu_electrical_engineering") -> List[str]:
    task_dir = os.path.join(mmlu_dir, mmlu_task)
    if not os.path.isdir(task_dir):
        return []
    models = []
    for d in os.listdir(task_dir):
        p = os.path.join(task_dir, d)
        if os.path.isdir(p) and d in MMLU_TO_CANON:
            models.append(MMLU_TO_CANON[d])
    models.sort()
    return models


def load_model_outputs_for_models(model_names: List[str], mmlu_dir: str = DEFAULT_MMLU_BASE_DIR, mmlu_task: str = "mmlu_electrical_engineering") -> Tuple[List[str], List[Dict[str, float]]]:
    kept_models: List[str] = []
    dicts: List[Dict[str, float]] = []
    for canon in model_names:
        mmlu_name = CANON_TO_MMLU.get(canon)
        if not mmlu_name:
            continue
        d = _load_mmlu_model_results(mmlu_name, mmlu_dir=mmlu_dir, mmlu_task=mmlu_task)
        if not d:
            continue
        dicts.append(d)
        kept_models.append(canon)
    return kept_models, dicts
