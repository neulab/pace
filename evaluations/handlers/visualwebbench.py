"""VisualWebBench handler for ProxyBench."""

import io
import os
import re
import sys
import base64

_HANDLERS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVALUATIONS_DIR = os.path.dirname(_HANDLERS_DIR)
BENCHMARKS_DIR = os.path.join(_EVALUATIONS_DIR, "benchmarks")

VWB_DIR = os.path.join(BENCHMARKS_DIR, "visualwebbench")
VWB_DATASET_NAME = "webbench/WebBench"

VWB_TASKS = (
    "web_caption",
    "heading_ocr",
    "webqa",
    "element_ocr",
    "element_ground",
    "action_prediction",
    "action_ground",
)
VWB_DEFAULT_SUBTASK = "web_caption"

VWB_TASK_METRIC = {
    "web_caption":       ("rouge_l", 1 / 100),
    "heading_ocr":       ("rouge_l", 1 / 100),
    "webqa":             ("f1",      1 / 100),
    "element_ocr":       ("rouge_l", 1 / 100),
    "element_ground":    ("correct", 1.0),
    "action_prediction": ("correct", 1.0),
    "action_ground":     ("correct", 1.0),
}


def _run_visualwebbench(
    model_name: str,
    base_url: str,
    api_key: str,
    subtask: str,
    instance_id: str,
) -> list:
    """Run a single VisualWebBench instance.

    instance_id format: '{task}_{idx}', e.g. 'web_caption_0'.
    """
    import datasets as hf_datasets

    # The instance_id encodes the task as '{task}_{idx}' (e.g. 'heading_ocr_3'),
    # so derive the subtask from it — callers need not pass the matching one.
    m = re.match(r"^(.*)_(\d+)$", instance_id)
    if m and m.group(1) in VWB_TASKS:
        subtask = m.group(1)
        idx = int(m.group(2))
    elif subtask in VWB_TASKS:
        # Fallback: trust the given subtask; id is the bare index or '{subtask}_{idx}'.
        prefix = f"{subtask}_"
        idx_str = instance_id[len(prefix):] if instance_id.startswith(prefix) else instance_id
        try:
            idx = int(idx_str)
        except ValueError:
            raise ValueError(
                f"Invalid VisualWebBench instance_id '{instance_id}'. "
                f"Expected '{{task}}_{{integer}}' with task in {VWB_TASKS}."
            )
    else:
        raise ValueError(
            f"Cannot resolve VisualWebBench task from instance_id '{instance_id}' "
            f"(subtask='{subtask}'). Valid tasks: {VWB_TASKS}."
        )

    vwb_utils_dir = os.path.join(VWB_DIR, "utils")
    if vwb_utils_dir not in sys.path:
        sys.path.insert(0, vwb_utils_dir)
    if VWB_DIR not in sys.path:
        sys.path.insert(0, VWB_DIR)

    from utils.constants import (
        CAPTION_TASK, HEADING_OCR_TASK, WEBQA_TASK, ELEMENT_OCR_TASK,
        ELEMENT_GROUND_TASK, ACTION_PREDICTION_TASK, ACTION_GROUND_TASK,
    )
    from utils.eval_utils import (
        eval_web_caption, eval_heading_ocr, eval_webqa, eval_element_ocr,
        eval_element_ground, eval_action_prediction, eval_action_ground,
    )
    from utils.prompts import DEFAULT_PROMPTS

    eval_func_map = {
        CAPTION_TASK:           eval_web_caption,
        HEADING_OCR_TASK:       eval_heading_ocr,
        WEBQA_TASK:             eval_webqa,
        ELEMENT_OCR_TASK:       eval_element_ocr,
        ELEMENT_GROUND_TASK:    eval_element_ground,
        ACTION_PREDICTION_TASK: eval_action_prediction,
        ACTION_GROUND_TASK:     eval_action_ground,
    }

    dataset = hf_datasets.load_dataset(VWB_DATASET_NAME, subtask)["test"]

    if idx >= len(dataset):
        raise IndexError(
            f"Instance idx={idx} out of range for VisualWebBench subtask '{subtask}' "
            f"(dataset has {len(dataset)} instances)."
        )

    sample = dataset[idx]

    prompt_key = f"{subtask}_prompt"
    prompt_template = DEFAULT_PROMPTS[prompt_key]

    if subtask in (CAPTION_TASK, HEADING_OCR_TASK):
        prompt = prompt_template
    elif subtask == WEBQA_TASK:
        prompt = prompt_template.format(question=sample["question"])
    elif subtask == ELEMENT_OCR_TASK:
        prompt = prompt_template.format(bbox_ratio=sample["bbox"])
    elif subtask == ELEMENT_GROUND_TASK:
        prompt = prompt_template.format(element_desc=sample["elem_desc"])
    elif subtask == ACTION_PREDICTION_TASK:
        prompt = prompt_template.format(bbox_ratio=sample["bbox"], choices_text=sample["options"])
    elif subtask == ACTION_GROUND_TASK:
        prompt = prompt_template.format(instruction=sample["instruction"])
    else:
        prompt = prompt_template

    image = sample["image"]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }],
        max_tokens=512,
        timeout=300,
    )
    raw_output = response.choices[0].message.content or ""

    if subtask == CAPTION_TASK:
        pattern = re.compile(r'<meta name="description" content="(.*)">')
        m = re.findall(pattern, raw_output)
        pred = m[0] if m else raw_output
    elif subtask == ELEMENT_OCR_TASK:
        if ":" in raw_output:
            pred = ":".join(raw_output.split(":")[1:]).strip().strip('"').strip("'")
        else:
            pred = raw_output
    else:
        pred = raw_output

    gold = sample["answer"]
    eval_func = eval_func_map[subtask]
    _, per_instance = eval_func([pred], [gold])

    instance_score = per_instance[0]
    metric_key, scale = VWB_TASK_METRIC[subtask]
    raw_score = instance_score.get(metric_key, 0.0)
    normalized_score = max(0.0, min(1.0, float(raw_score) * scale))

    return [{
        "instance_id": instance_id,
        "subtask": subtask,
        "idx": idx,
        "prompt": prompt,
        "pred": pred,
        "gold": gold,
        "raw_output": raw_output,
        "instance_score": instance_score,
        metric_key: raw_score,
        f"{metric_key}_normalized": normalized_score,
    }]
