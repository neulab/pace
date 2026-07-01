import json
import os
import sys

import random

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM
from vllm import SamplingParams
import torch

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(0)

def truncate_prompt(d, tokenizer: AutoTokenizer, target_len=120 * 1024):
    '''return truncated prompt and truncation len '''
    # instruction flag: \nInstruction:  \n\nInstruction: \n\n Instructions:
    assert d['length'] == 120, f'Error: truncate prompt for length {d["length"]}'
    prompt = d['prompt']
    if prompt.count('\n\n Instructions:') == 1:
        ins_sep = '\n\n Instructions:'
    elif prompt.count('\n\nInstruction:') == 1:
        ins_sep = '\n\nInstruction:'
    elif prompt.count('\nInstruction:') == 1:
        ins_sep = '\nInstruction:'
    else:
        raise 'Error: No ins_sep.'
    ftext, ptext = prompt.split(ins_sep)
    encode_idx = tokenizer.encode(ftext)
    rtn_prompt = tokenizer.decode(encode_idx[:target_len]) + ins_sep + ptext
    return rtn_prompt, len(encode_idx) - target_len


def get_max_seq_len(model_name):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if 'Qwen2.5' in model_name:
        return 50 * 1024
    try:
        msl = config.model_max_length
    except:
        try:
            msl = config.max_sequence_length
        except:
            try:
                msl = config.seq_length
            except:
                msl = config.max_position_embeddings
    return msl


BENCHMARK_BASE_DIR = '.'
MAX_GENERATE_TOKENS = 512


def inference(model_name='', max_len_limit=150, seqlen_reduce_ratio=0, benchmark_base_dir=BENCHMARK_BASE_DIR, append_mode=True, tensor_parallel_size=torch.cuda.device_count()):
    # basic info
    model_max_seq_len = get_max_seq_len(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # if apply chat_template
    use_chattpl = True
    try:
        prompt = [{"role": "user", "content": 'testtest'}]
        inputs = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    except:
        use_chattpl = False
    print('********************************\n',
          f'run Benchmark for model: {model_name}.'
          f'\nmax_seq_len: {model_max_seq_len // 1024}k / {model_max_seq_len} token'
          f'\napply chat_template: {use_chattpl}'
          '\n********************************')

    # try to load vllm
    model_seq_len = model_max_seq_len if model_max_seq_len <= max_len_limit * 1024 else max_len_limit * 1024
    for _ in range(seqlen_reduce_ratio):
        model_seq_len = model_seq_len // 2
    print(f'******************************** vllm load: max_seq_len:  {model_seq_len // 1024}k / {model_seq_len} token ********************************')
    try:
        llm = LLM(model=model_name, gpu_memory_utilization=0.98, trust_remote_code=True, max_model_len=model_seq_len, tensor_parallel_size=tensor_parallel_size)
    except OSError:
        sys.exit(0)
    except Exception as e:
        print(f'******************************** Error: {e}, Model:{model_name}, seq_len:  {model_seq_len // 1024}k / {model_seq_len} token')
        if model_max_seq_len:
            sys.exit(seqlen_reduce_ratio + 1)

    # settings for each LLM models
    if 'glm-4' in model_name:
        stop_token_ids = [151329, 151336, 151338]
        sampling_params_dict = {'temperature': 0, 'stop_token_ids': stop_token_ids}
    else:
        sampling_params_dict = {'temperature': 0}

    def llm_generate_one(prompt, max_tokens):
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params_dict)
        if use_chattpl:
            prompt = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            outputs = llm.generate(prompts=inputs, sampling_params=sampling_params, use_tqdm=False)[0]
        else:
            outputs = llm.generate(prompts=prompt + '\noutput: ', sampling_params=sampling_params, use_tqdm=False)[0]
        return outputs.outputs[0].text

    def benchmark_inference(base_path, model_name):
        task_path = os.path.join(base_path, 'prompts')
        outputs_path = os.path.join(base_path, 'outputs', model_name)
        if not os.path.exists(outputs_path):
            os.mkdir(outputs_path)

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
            save_count = 0
            for d in tqdm(datas):
                if 'output' in d and d['output']:
                    if ('find_dup_text' in task_file or 'batch_label' in task_file) and d['length'] in [60, 120] and len(tokenizer.encode(
                            d['output'])) <= 1024:
                        pass  # do a inference
                    else:
                        continue
                max_generate_tokens_ = max_generate_tokens
                prompt = d['prompt']
                prompt_len = len(tokenizer.encode(d['prompt']))
                if prompt_len >= model_seq_len - max_generate_tokens_:
                    max_generate_tokens_ = model_seq_len - prompt_len
                    if max_generate_tokens_ <= 1024:
                        prompt, trunc_len = truncate_prompt(d, tokenizer)
                        d['truncate'] = trunc_len
                        max_generate_tokens_ = max_generate_tokens
                output = llm_generate_one(prompt, max_generate_tokens_)
                d[f'output'] = output
                save_count += 1
                if save_count % 20 == 0:
                    with open(os.path.join(outputs_path, task_file), 'w', encoding='utf-8') as f:
                        json.dump(datas, f, indent=4)
            with open(os.path.join(outputs_path, task_file), 'w', encoding='utf-8') as f:
                json.dump(datas, f, indent=4)

        task_file_names = os.listdir(task_path)
        task_file_names.sort()
        print(task_file_names)
        for task_file in task_file_names:
            if task_file.split('.')[-1] != 'json':
                continue
            print(task_file)
            get_answers(task_file)

    model_name_ = list(filter(lambda x: x, model_name.split('/')))[-1]
    benchmark_inference(benchmark_base_dir, model_name_)
    sys.exit(0)


def main(model_name='', max_len_limit=256, seqlen_reduce_ratio=0, benchmark_base_dir=BENCHMARK_BASE_DIR):
    inference(model_name, max_len_limit, seqlen_reduce_ratio, benchmark_base_dir)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
