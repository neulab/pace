"""MMMU handler for ProxyBench."""

import ast
import base64
import io
import os
import re
import sys

_HANDLERS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVALUATIONS_DIR = os.path.dirname(_HANDLERS_DIR)
BENCHMARKS_DIR = os.path.join(_EVALUATIONS_DIR, "benchmarks")

MMMU_LMMS_EVAL_DIR = os.path.join(BENCHMARKS_DIR, "lmms-eval")
MMMU_DATASET_NAME = "lmms-lab/MMMU"
MMMU_MC_PROMPT = "Answer with the option's letter from the given choices directly."
MMMU_OE_PROMPT = "Answer the question using a single word or phrase."

MMMU_SUBJECTS = {
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine",
    "Computer_Science", "Design", "Diagnostics_and_Laboratory_Medicine", "Economics",
    "Electronics", "Energy_and_Power", "Finance", "Geography", "History",
    "Literature", "manage", "Manage", "Marketing", "Materials",
    "Math", "Mechanical_Engineering", "Music", "Pharmacy", "Physics",
    "Psychology", "Public_Health", "Sociology",
}


def _run_mmmu(
    model_name: str,
    base_url: str,
    api_key: str,
    instance_id: str,
) -> list:
    """Run a single MMMU instance.

    instance_id format: '{split}_{Subject}_{idx}', e.g. 'validation_Accounting_1'.
    """
    import datasets as hf_datasets

    parts = instance_id.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid MMMU instance_id '{instance_id}'. "
            f"Expected format: 'validation_{{Subject}}_{{n}}'"
        )
    split_name = parts[0]
    subject = "_".join(parts[1:-1])

    if MMMU_LMMS_EVAL_DIR not in sys.path:
        sys.path.insert(0, MMMU_LMMS_EVAL_DIR)

    from lmms_eval.tasks._task_utils.mmmu_mcq_utils import (
        get_multi_choice_info,
        parse_mmmu_multi_choice_response,
    )

    dataset = hf_datasets.load_dataset(MMMU_DATASET_NAME, "default")[split_name]

    doc = None
    for item in dataset:
        if item.get("id") == instance_id:
            doc = item
            break
    if doc is None:
        raise ValueError(
            f"MMMU instance '{instance_id}' not found in split '{split_name}'."
        )

    question = doc["question"]
    if doc["question_type"] == "multiple-choice":
        options = ast.literal_eval(doc["options"])
        choices_str = "\n".join(
            f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)
        )
        prompt_text = f"{question}\n{choices_str}\n\n{MMMU_MC_PROMPT}"
    else:
        prompt_text = f"{question}\n\n{MMMU_OE_PROMPT}"

    for i in range(1, 8):
        prompt_text = prompt_text.replace(f"<image {i}>", "<image>")

    raw_question = doc["question"]
    referenced = sorted(set(re.findall(r"<image (\d+)>", raw_question)))
    if not referenced and doc.get("image_1") is not None:
        referenced = ["1"]

    def encode_image(pil_img):
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    images_b64 = [encode_image(doc[f"image_{ref}"]) for ref in referenced if doc.get(f"image_{ref}") is not None]

    segments = prompt_text.split("<image>")
    content = []
    for i, seg in enumerate(segments):
        if seg.strip():
            content.append({"type": "text", "text": seg.strip()})
        if i < len(images_b64):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{images_b64[i]}"},
            })
    if len(segments) == 1 and images_b64:
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images_b64[0]}"}},
        ] + content

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        max_tokens=512,
        timeout=300,
    )
    raw_output = response.choices[0].message.content or ""

    if doc["question_type"] == "multiple-choice":
        options = ast.literal_eval(doc["options"])
        index2ans, all_choices = get_multi_choice_info(options)
        parsed_pred = parse_mmmu_multi_choice_response(raw_output, all_choices, index2ans)
        answer = doc["answer"]
        correct = (parsed_pred == answer)
    else:
        def _normalize_str(s):
            s = s.strip()
            try:
                return [round(float(s.replace(",", "")), 2)]
            except ValueError:
                s = s.lower()
                return [" " + s, s + " "] if len(s) == 1 else [s]

        def _extract_numbers(s):
            nums = []
            for pat in [r"-?\b\d{1,3}(?:,\d{3})+\b", r"-?\d+(?:\.\d+)?[eE][+-]?\d+",
                        r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"]:
                nums.extend(re.findall(pat, s))
            return nums

        def _parse_open(text):
            text = text.strip().strip(".").lower()
            subs = re.split(r"\.\s(?=[A-Z])|\n", text)
            indicators = ["could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result "]
            key = []
            for idx2, sub in enumerate(subs):
                if idx2 == len(subs) - 1:
                    indicators = indicators + ["="]
                shortest = None
                for ind in indicators:
                    if ind in sub:
                        cand = sub.split(ind)[-1].strip()
                        if shortest is None or len(cand) < len(shortest):
                            shortest = cand
                if shortest and shortest not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                    key.append(shortest)
            if not key:
                key = [text]
            preds = list(key)
            for r in key:
                preds.extend(_extract_numbers(r))
            norm = []
            for p in preds:
                norm.extend(_normalize_str(str(p)))
            return list(set(norm))

        parsed_pred_list = _parse_open(raw_output)
        gold = doc["answer"]
        gold_norm = []
        for g in ([gold] if isinstance(gold, str) else gold):
            try:
                gold_norm.append(round(float(g.replace(",", "")), 2))
            except ValueError:
                gold_norm.extend([g.lower(), " " + g.lower(), g.lower() + " "])
        correct = False
        for p in parsed_pred_list:
            if isinstance(p, float):
                correct = p in gold_norm
            elif isinstance(p, str):
                correct = any(isinstance(g, str) and g in p for g in gold_norm)
            if correct:
                break
        parsed_pred = parsed_pred_list

    score = 1.0 if correct else 0.0

    return [{
        "instance_id": instance_id,
        "subject": subject,
        "question_type": doc["question_type"],
        "question": doc["question"],
        "answer": doc["answer"],
        "raw_output": raw_output,
        "parsed_pred": parsed_pred if isinstance(parsed_pred, list) else [parsed_pred],
        "correct": correct,
        "mmmu_acc": score,
    }]
