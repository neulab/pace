import json
import os
import sys
import tempfile

import tiktoken
from tqdm import tqdm
import torch
from LLMApi import llmApi

BENCHMARK_BASE_DIR = './data'
MAX_GENERATE_TOKENS = 512
MODEL_NAMES = [
    'gpt-4o-2024-08-06',  # 128 g16
    'gpt-4-0125-preview'
    ]


def get_model_seq_len():
    return 125 * 1024 - 8


def inference(model_name='', token_limit=32 * 1024, benchmark_base_dir=BENCHMARK_BASE_DIR, append_mode=True, tensor_parallel_size=torch.cuda.device_count(), task_filter='', max_samples=0):
    # basic info
    model_seq_len = get_model_seq_len()
    try:
        tokenizer = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback for models not in tiktoken (e.g., azure/gpt-5)
        tokenizer = tiktoken.encoding_for_model('gpt-4o')
    print(f'******************************** api load model: {model_name} ********************************')
    try:
        llmapi = llmApi(model_name)
    except OSError:
        sys.exit(0)
    except Exception as e:
        print(f'******************************** Error: {e}, Model:{model_name}')
        sys.exit(11)

    def llm_generate_one(prompt, max_tokens, stream=True):
        if stream:
            llm_output = llmapi.get_by_stream(prompt, max_new_tokens=max_tokens, max_tries=3)
        else:
            llm_output = llmapi.get(prompt, max_new_tokens=max_tokens, max_tries=3)
        return llm_output

    def benchmark_inference(base_path, model_name):
        task_path = os.path.join(base_path, 'prompts')
        output_model_name = model_name.split('/')[-1]
        outputs_path = os.path.join(base_path, 'outputs', output_model_name)
        if not os.path.exists(outputs_path):
            os.makedirs(outputs_path)

        def get_answers(task_file):
            if 'list-' in task_file and 'multi_query_id' not in task_file:
                max_generate_tokens = 100
            elif 'find_dup_text' in task_file or 'batch_label' in task_file:
                max_generate_tokens = 4096
            else:
                max_generate_tokens = MAX_GENERATE_TOKENS
            query_path = os.path.join(task_path, task_file)
            output_file_path = os.path.join(outputs_path, task_file)
            if os.path.exists(output_file_path) and append_mode:
                print(f'******************************** loaded output****** max_gen: {max_generate_tokens}---')
                with open(output_file_path, 'r') as f:
                    datas = json.load(f)
            else:
                print(f'******************************** new output----- max_gen: {max_generate_tokens}---')
                with open(query_path, 'r') as f:
                    datas = json.load(f)
            assert isinstance(datas, list)
            if max_samples > 0:
                seen_ids = set()
                filtered = []
                for d_ in datas:
                    iid = d_.get('ins_id')
                    if iid not in seen_ids:
                        if len(seen_ids) >= max_samples:
                            break
                        seen_ids.add(iid)
                        filtered.append(d_)
                datas = filtered
            save_count = 0
            for d in tqdm(datas):
                if 'output' in d and d['output'] and len(tokenizer.encode(d['output'])) != 512: continue
                # if 'output' in d and d['output']: continue
                prompt_len = len(tokenizer.encode(d['prompt']))
                max_generate_tokens_ = max_generate_tokens
                max_prompt_len = min(token_limit, model_seq_len - max_generate_tokens)
                if prompt_len >= max_prompt_len:
                    truncated_prompt = tokenizer.decode(tokenizer.encode(d['prompt'])[:max_prompt_len])
                    print(f"max_generate_tokens: {max_generate_tokens_}")
                    output = llm_generate_one(truncated_prompt, max_generate_tokens_)
                else:
                    output = llm_generate_one(d['prompt'], max_generate_tokens_)
                d['output'] = output
                save_count += 1
                if save_count % 1 == 0:
                    final_path = os.path.join(outputs_path, task_file)
                    tmp_fd, tmp_path = tempfile.mkstemp(dir=outputs_path, suffix='.tmp')
                    with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                        json.dump(datas, f, indent=4)
                    os.replace(tmp_path, final_path)
            final_path = os.path.join(outputs_path, task_file)
            tmp_fd, tmp_path = tempfile.mkstemp(dir=outputs_path, suffix='.tmp')
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                json.dump(datas, f, indent=4)
            os.replace(tmp_path, final_path)

        task_file_names = os.listdir(task_path)
        task_file_names.sort()
        if task_filter:
            task_file_names = [t for t in task_file_names if task_filter in t]
        print(task_file_names)
        for task_file in task_file_names:
            if task_file.split('.')[-1] != 'json':
                continue
            print(task_file)
            get_answers(task_file)

    model_name_ = list(filter(lambda x: x, model_name.split('/')))[-1]
    benchmark_inference(benchmark_base_dir, model_name_)
    sys.exit(0)


def main(model_name='', token_limit=128 * 1024, benchmark_base_dir=BENCHMARK_BASE_DIR, task_filter='', max_samples=0, append_mode=True):
    inference(model_name=model_name, token_limit=token_limit, benchmark_base_dir=benchmark_base_dir, append_mode=append_mode, task_filter=task_filter, max_samples=max_samples)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)

