"""RepoBench handler for ProxyBench."""

import os
import sys

_HANDLERS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVALUATIONS_DIR = os.path.dirname(_HANDLERS_DIR)
BENCHMARKS_DIR = os.path.join(_EVALUATIONS_DIR, "benchmarks")

REPOBENCH_DIR = os.path.join(BENCHMARKS_DIR, "repobench")
REPOBENCH_EVAL_DIR = os.path.join(REPOBENCH_DIR, "evaluation")

REPOBENCH_SUBTASK_MAP = {
    "repobench_xff_python": "cross_file_first",
    "repobench_xfr_python": "cross_file_random",
    "repobench_if_python":  "in_file",
}
REPOBENCH_DEFAULT_SUBTASK = "repobench_xff_python"

REPOBENCH_DATASET_NAME = "tianyang/repobench_python_v1.1"
REPOBENCH_START_DATE = "2023-12-01"
REPOBENCH_END_DATE   = "2023-12-31"
REPOBENCH_LEVEL      = "8k"


def _first_line_not_comment(code: str) -> str:
    """Return the first non-comment, non-empty line of generated Python code."""
    code = code.lstrip("\n")
    lines = code.split("\n")
    in_multiline = False
    if lines and lines[0].strip().startswith("```python"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    for line in lines:
        if not line.strip():
            continue
        if not in_multiline and (line.strip().startswith('"""') or line.strip().startswith("'''")):
            in_multiline = True
            continue
        if in_multiline and (line.strip().endswith('"""') or line.strip().endswith("'''")):
            in_multiline = False
            continue
        if in_multiline:
            continue
        if line.strip().startswith("#"):
            continue
        return line
    return lines[0] if lines else ""


def _run_repobench(
    model_name: str,
    base_url: str,
    api_key: str,
    subtask: str,
    instance_id: str,
) -> list:
    """Run a single RepoBench instance.

    instance_id format: 'repobench_{idx}' where idx is the 0-based index
    within the filtered dataset (date: 2023-12-01 to 2023-12-31, level: 8k).
    subtask: one of repobench_xff_python, repobench_xfr_python, repobench_if_python.
    """
    import pandas as pd
    from datasets import load_dataset

    if instance_id.startswith("repobench_"):
        idx_str = instance_id[len("repobench_"):]
    else:
        idx_str = instance_id
    try:
        idx = int(idx_str)
    except ValueError:
        raise ValueError(
            f"Invalid RepoBench instance_id '{instance_id}'. "
            f"Expected format: 'repobench_{{integer}}'"
        )

    if subtask not in REPOBENCH_SUBTASK_MAP:
        raise ValueError(
            f"Unknown RepoBench subtask '{subtask}'. "
            f"Valid subtasks: {list(REPOBENCH_SUBTASK_MAP.keys())}"
        )
    subset_name = REPOBENCH_SUBTASK_MAP[subtask]

    if REPOBENCH_DIR not in sys.path:
        sys.path.insert(0, REPOBENCH_DIR)
    from data.utils import construct_prompt

    dataset = load_dataset(REPOBENCH_DATASET_NAME)

    start_dt = pd.to_datetime(REPOBENCH_START_DATE).tz_localize("UTC")
    end_dt = pd.to_datetime(REPOBENCH_END_DATE).tz_localize("UTC")
    df = pd.DataFrame(dataset[subset_name])
    df["created_at"] = pd.to_datetime(df["created_at"])
    mask = (df["created_at"] >= start_dt) & (df["created_at"] <= end_dt)
    df = df[mask]
    df = df[df["level"] == REPOBENCH_LEVEL].reset_index(drop=True)

    if idx >= len(df):
        raise IndexError(
            f"Instance idx={idx} out of range for RepoBench subtask '{subtask}' "
            f"(filtered dataset has {len(df)} entries)."
        )

    item = df.iloc[idx].to_dict()
    prompt = construct_prompt(item, language="python", tokenizer=None, max_token_nums=15800)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a code completion assistant. Continue the provided code "
                    "snippet directly and exactly from where it ends. Output only the "
                    "next line of code, nothing else. No explanation, no markdown, no preamble."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=128,
        timeout=300,
    )
    raw_output = response.choices[0].message.content or ""

    pred = _first_line_not_comment(raw_output)
    gt = item["next_line"]

    if REPOBENCH_EVAL_DIR not in sys.path:
        sys.path.insert(0, REPOBENCH_EVAL_DIR)
    from metrics import exact_match_score, edit_similarity_score, codebleu_score

    em = exact_match_score([pred], [gt])
    es = edit_similarity_score([pred], [gt])
    cb = codebleu_score([pred], [gt], language="python")

    return [{
        "instance_id": instance_id,
        "subtask": subtask,
        "idx": idx,
        "level": item.get("level", REPOBENCH_LEVEL),
        "pred": pred,
        "gt": gt,
        "raw_output": raw_output,
        "exact_match": em,
        "edit_similarity": es,
        "codebleu": cb,
    }]
