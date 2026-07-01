"""VisualPuzzles handler for ProxyBench."""

import base64
import io
import os
import re
import sys

_HANDLERS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVALUATIONS_DIR = os.path.dirname(_HANDLERS_DIR)
BENCHMARKS_DIR = os.path.join(_EVALUATIONS_DIR, "benchmarks")

VP_DIR = os.path.join(BENCHMARKS_DIR, "lmms-eval", "lmms_eval", "tasks", "VisualPuzzles")
VP_DATASET_NAME = "neulab/VisualPuzzles"
VP_DEFAULT_SUBTASK = "cot"

VP_PROMPTS = {
    "cot": (
        "Solve the multiple-choice question and then answer with the option letter "
        "from the given choices. The last line of your response should be of the "
        "following format: 'Answer: $LETTER' (without quotes) where LETTER is one "
        "of options. Think step by step before answering."
    ),
    "direct": "Answer the question with the option's letter from the given choices directly.",
}
VP_MAX_TOKENS = {"cot": 4096, "direct": 512}


def _parse_response(response: str, all_choices, index2ans) -> str:
    """Parse model response to extract the predicted choice letter."""
    for pattern, _ in [
        (r"Answer:\s*\(([A-Za-z])\)", True),
        (r"(?<!Final )Answer:\s*([A-Za-z])", False),
        (r"Answer:\s*([A-Za-z])", False),
        (r"\s*\(([A-Za-z])\)", True),
    ]:
        matches = re.findall(pattern, response)
        for m in reversed(matches):
            if m in all_choices or m.upper() in all_choices:
                return m.upper()

    response_sp = " " + response.strip()
    for pattern in [r"\s*([A-Za-z])\)", r"\s*\{([A-Za-z])\}", r"\s*\$([A-Za-z])\$", r" ([A-Da-d])\."]:
        matches = re.findall(pattern, response_sp)
        for m in reversed(matches):
            if m in all_choices or m.upper() in all_choices:
                return m.upper()

    matches = re.findall(r" ([A-Da-d])", response_sp)
    if matches and len(response) <= 5:
        for m in reversed(matches):
            if m in all_choices or m.upper() in all_choices:
                return m.upper()

    if index2ans:
        for idx in all_choices:
            ans = index2ans[idx]
            if f"answer: {ans}" in response.lower() or f"answer:{ans}" in response.lower():
                return idx
        last_found, last_pos = None, -1
        for idx in all_choices:
            pos = response.rfind(index2ans[idx])
            if pos > last_pos:
                last_found, last_pos = idx, pos
        if last_found:
            return last_found

    return ""


def _run_visualpuzzles(
    model_name: str,
    base_url: str,
    api_key: str,
    subtask: str,
    instance_id: str,
) -> list:
    """Run a single VisualPuzzles instance.

    instance_id format: 'visualpuzzles_{doc_id}' (0-based index in train split).
    subtask: 'cot' or 'direct'.
    """
    import datasets as hf_datasets

    if subtask not in VP_PROMPTS:
        raise ValueError(
            f"Unknown VisualPuzzles subtask '{subtask}'. Valid: {list(VP_PROMPTS.keys())}"
        )

    prefix = "visualpuzzles_"
    idx_str = instance_id[len(prefix):] if instance_id.startswith(prefix) else instance_id
    try:
        doc_id = int(idx_str)
    except ValueError:
        raise ValueError(
            f"Invalid VisualPuzzles instance_id '{instance_id}'. "
            f"Expected format: 'visualpuzzles_{{integer}}'"
        )

    dataset = hf_datasets.load_dataset(VP_DATASET_NAME, split="train", token=True)

    if doc_id >= len(dataset):
        raise IndexError(
            f"doc_id={doc_id} out of range (dataset has {len(dataset)} instances)."
        )
    doc = dataset[doc_id]

    question = "Question: " + doc["question"].strip()
    options = doc.get("options")
    if options is not None:
        question += (
            f"\nOptions:\n(A) {options[0]}\n(B) {options[1]}"
            f"\n(C) {options[2]}\n(D) {options[3]}"
        )
    else:
        question += "\nOptions: Choose from (A) (B) (C) (D) in the image."
    question += "\n" + VP_PROMPTS[subtask]

    img = doc["image"]
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }],
        max_tokens=VP_MAX_TOKENS[subtask],
        timeout=300,
    )
    raw_output = response.choices[0].message.content or ""

    all_choices = ["A", "B", "C", "D"]
    index2ans = (
        {all_choices[i]: options[i] for i in range(4)} if options is not None else None
    )
    parsed_pred = _parse_response(raw_output, all_choices, index2ans)
    answer = doc["answer"]
    correct = parsed_pred.upper() == answer.upper() if parsed_pred else False
    score = 1.0 if correct else 0.0

    return [{
        "instance_id": instance_id,
        "subtask": subtask,
        "doc_id": doc_id,
        "question": doc["question"],
        "options": options,
        "answer": answer,
        "category": doc.get("category"),
        "difficulty": doc.get("difficulty"),
        "raw_output": raw_output,
        "parsed_pred": parsed_pred,
        "correct": correct,
        "exact_match": score,
    }]
