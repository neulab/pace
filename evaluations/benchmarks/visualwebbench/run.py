import os
import json
import yaml
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import datasets
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import model_adapters
from utils import DEFAULT_PROMPTS
from utils import (
    eval_web_caption,
    eval_heading_ocr,
    eval_element_ocr,
    eval_action_prediction,
    eval_element_ground,
    eval_action_ground,
    eval_webqa,
)
from utils.constants import *


eval_metric = {
    CAPTION_TASK: eval_web_caption,
    HEADING_OCR_TASK: eval_heading_ocr,
    WEBQA_TASK: eval_webqa,
    ELEMENT_OCR_TASK: eval_element_ocr,
    ELEMENT_GROUND_TASK: eval_element_ground,
    ACTION_PREDICTION_TASK: eval_action_prediction,
    ACTION_GROUND_TASK: eval_action_ground,
}


def build_prompt(prompt, sample, task_type):
    if task_type in [CAPTION_TASK, HEADING_OCR_TASK]:
        return prompt
    elif task_type == WEBQA_TASK:
        return prompt.format(question=sample['question'])
    elif task_type == ELEMENT_OCR_TASK:
        return prompt.format(bbox_ratio=sample['bbox'])
    elif task_type == ELEMENT_GROUND_TASK:
        return prompt.format(element_desc=sample['elem_desc'])
    elif task_type == ACTION_PREDICTION_TASK:
        return prompt.format(bbox_ratio=sample['bbox'], choices_text=sample['options'])
    elif task_type == ACTION_GROUND_TASK:
        return prompt.format(instruction=sample['instruction'])
    else:
        raise NotImplementedError(f"Task type {task_type} not implemented.")


def evaluate(
    model_adapter: model_adapters.BaseAdapter,
    prompt: str,
    dataset: datasets.Dataset,
    task_type: str,
    num_workers: int = 1,
    **kwargs,
):
    print('='*80)
    print('Prompt: ', prompt)
    data_size = len(dataset)

    def process_one(idx_):
        sample = dataset[idx_]
        cur_prompt = build_prompt(prompt, sample, task_type)
        response = model_adapter(cur_prompt, sample['image'], task_type=task_type)
        return idx_, response, sample['answer']

    results = [None] * data_size
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, i): i for i in range(data_size)}
        for future in tqdm(as_completed(futures), total=data_size, desc=task_type):
            idx_, response, answer = future.result()
            results[idx_] = (response, answer)

    preds = [r[0] for r in results]
    golds = [r[1] for r in results]
    scores, per_instance_scores = eval_metric[task_type](preds, golds)
    return scores, per_instance_scores, preds, golds


def main(args):
    model_config = yaml.load(open(f"configs/{args.model_name}.yaml"), Loader=yaml.FullLoader)
    model_path = model_config.get('model_path')
    tokenizer_path = model_config.get('tokenizer_path', model_path)
    adapter_name = model_config['model_adapter']

    device = f"cuda:{args.gpus}"

    if adapter_name == 'OpenAIAdapter':
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model_adapter = getattr(model_adapters, adapter_name)(client, model_path)

    elif adapter_name == 'ClaudeAdapter':
        from anthropic import Anthropic

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        model_adapter = getattr(model_adapters, adapter_name)(client, model_path)

    elif adapter_name == 'GeminiAdapter':
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(model_path)
        model_adapter = getattr(model_adapters, adapter_name)(model)

    elif adapter_name == 'LlavaAdapter':
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from llava.mm_utils import get_model_name_from_path

        raw_model_name = get_model_name_from_path(model_path)
        disable_torch_init()
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, raw_model_name, device_map=None, device=device,
        )
        model_adapter = getattr(model_adapters, adapter_name)(
            model, tokenizer, context_len, image_processor, model_config['conv_mode']
        )

    else:
        # Generic local HuggingFace model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model_adapter = getattr(model_adapters, adapter_name)(model, tokenizer)

    if ',' in args.task_type:
        task_types = [item.strip() for item in args.task_type.split(',')]
    else:
        task_types = [args.task_type]

    def run_task(task_type):
        prompt = model_config.get(f"{task_type}_prompt", DEFAULT_PROMPTS[f"{task_type}_prompt"])
        dataset = datasets.load_dataset(args.dataset_name_or_path, task_type)['test']
        scores, per_instance_scores, preds, golds = evaluate(
            model_adapter=model_adapter,
            prompt=prompt,
            dataset=dataset,
            task_type=task_type,
            num_workers=args.num_workers,
        )
        score_str = ', '.join([f"{k}: {v:.2f}" for k, v in scores.items()])
        print(f"Model: {args.model_name}, Task: {task_type}, Scores: {score_str}")
        output_res = [{"score": score_str}] + [
            {"pred": pred, "gold": gold, "instance_score": inst_score}
            for pred, gold, inst_score in zip(preds, golds, per_instance_scores)
        ]
        with open(os.path.join(args.output_path, f"{task_type}.json"), "w") as f:
            json.dump(output_res, f, indent=2)

    with ThreadPoolExecutor(max_workers=len(task_types)) as executor:
        futures = [executor.submit(run_task, t) for t in task_types]
        for future in futures:
            future.result()  # re-raise any exception


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name_or_path',
        default="webbench/WebBench",
        type=str,
    )
    parser.add_argument(
        '--model_name',
        default='qwen_vl',
        type=str,
        choices=[file[:-5] for file in os.listdir("configs") if file.endswith(".yaml")],
    )
    parser.add_argument(
        '--task_type',
        default='web_caption',
        type=str,
        help="Task type can be one of web_caption, heading_ocr, element_ocr, action_prediction, element_ground, action_ground, webqa. Or several tasks separated by comma.",
    )
    parser.add_argument(
        '--output_path', 
        default='output', 
        type=str
    )
    parser.add_argument(
        "--gpus",
        default="0",
        type=str,
        help="A single GPU like 1 or multiple GPUs like 0,2",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of parallel threads for API calls within each task.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    args.output_path = os.path.join(args.output_path, args.model_name)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print(args)

    main(args)
