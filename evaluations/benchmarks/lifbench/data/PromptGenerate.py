import itertools
import os
from typing import List
import json
import random

import nltk
import tiktoken

random.seed(0)

BENCHMAKR_BASE = ''
class PromptGenerator():
    def __init__(self):
        self.tokenizer = tiktoken.encoding_for_model('gpt-4o')
        self.task_dict = json.load(open(os.path.join(BENCHMAKR_BASE, 'data/meta/tasks.json'), 'r'))
        self.base_dir = os.path.join(BENCHMAKR_BASE, 'data')
        self.task_sampleFunc_map = {
            "repeat": combine_sample,
            'qa': qa_sample,
            'extract': combine_sample,
            'single_query_id': element_sample,
            'multi_query_id': int_arraye_sample,
            'offset_query_id': element_sample,
            'offset_query_element': element_sample,
            'blur_offset_query_element': element_sample,
            'blur_offset_query_id': element_sample,
            'find_dup_text': direct_sample_dup,
            'batch_label': direct_sample_label,
            }

    def construct_prompts(self, sys_instructions, meta: dict, inputs: List[dict], sample_func) -> List[dict]:
        prompts = []
        instructions = meta['instructions']
        params_dict = meta.get('params', {})
        input_info = inputs['info']
        for input_data in inputs['inputs']:
            for ins_id, instruction in enumerate(instructions):
                parmas = sample_func(instruction, params_dict, input_data, input_info)
                for param_id, param in enumerate(parmas):
                    if isinstance(instruction, list):
                        instruction = instruction[0]
                    instruction_ = instruction.format(**param)
                    prompt = sys_instructions.format(**{"instruction": instruction_, "input": input_data['input_text']})
                    line = {
                        "prompt": prompt,
                        "label": input_data['label'],
                        "param": param.copy(),
                        "length": input_data['length'],
                        "ins_id": ins_id,
                        "param_id": param_id,
                        }
                    prompts.append(line)
        return prompts

    def generate_prompts(self, main_tasks=[]):
        print('Generating prompts...')
        main_tasks = list(self.task_dict.keys()) if not main_tasks else main_tasks
        for main_task_name in main_tasks:
            sys_instructions = self.task_dict[main_task_name]['sys_instruction']
            inputs = json.load(open(os.path.join(self.base_dir, 'meta', f'{main_task_name}_input.json'), 'r'))
            for sub_task_name, meta in self.task_dict[main_task_name]['subtasks'].items():
                prompts = self.construct_prompts(sys_instructions, meta, inputs, self.task_sampleFunc_map[sub_task_name])
                json.dump(prompts, open(os.path.join(self.base_dir, 'prompts', f'{main_task_name}-{sub_task_name}.json'), 'w'), indent=2)
                print(f'{main_task_name}-{sub_task_name} done. num_prompts: {len(prompts)}')


# sample funcs: standard_input (instructions, params_dict, input_data, num_param_sample) ) -> List[dict]
def sample_cartesian_product(params: dict, n: int) -> List[dict]:
    """
    输入多个列表，计算它们的笛卡尔积，并随机采样n个结果组合。

    参数：
    - params: 输入的参数
    - n: 需要采样的组合数量

    返回：
    - 一个包含n个随机组合的列表
    """
    lists = list(params.values())
    # 计算所有输入列表的笛卡尔积
    product = list(itertools.product(*lists))

    # 如果采样数量n大于笛卡尔积的总组合数，抛出异常
    if n > len(product):
        raise ValueError("The number of samples n is greater than the total number of combinations of the Cartesian product.")

    # 随机采样n个组合
    sample_results = random.sample(product, n)
    return [{k: v for k, v in zip(list(params.keys()), sample)} for sample in sample_results]


def combine_sample(instruction, params_dict, input_data, input_info, num_param_sample=5):
    return sample_cartesian_product(params_dict, num_param_sample)


def qa_sample(instruction, params_dict, input_data, input_info, num_param_sample=6):
    '''
    :param num_param_sample: if 6: 2 error, 2 not evidence, remain correct
    :return:
    '''
    options = params_dict['options']
    true_evidences = []
    error_evidences = []
    not_evidence_evidences = []
    for v in input_data['label'].values():
        for evidence, _, _, is_correct in list(v):
            if is_correct:
                true_evidences.append(evidence)
            else:
                error_evidences.append(evidence)
    # sample
    true_evidences = random.sample(true_evidences, 2)
    error_evidences = random.sample(error_evidences, 2)

    # not evidence
    sentences = nltk.sent_tokenize(input_data['input_text'])
    n = num_param_sample - 4
    assert n > 0
    while len(not_evidence_evidences) < n:
        t = random.sample(sentences, 1)[0]
        if t not in true_evidences + error_evidences:
            not_evidence_evidences.append(t)
    start_i = random.randint(0, len(options))
    return [{'evidence': evidence, 'options': options[i % len(options)]} for i, evidence in enumerate(true_evidences + error_evidences + not_evidence_evidences, start=start_i)]


def int_arraye_sample(instruction, params_dict, input_data, input_info, num_param_sample=5):
    # 2, 10 个元素随机挑5个把
    random.seed(0)
    k = int(input_data['label'])
    assert k > 20

    sample_arrays = []
    sample_nums = sorted(random.sample(range(2, 11), num_param_sample))
    for n in sample_nums:
        sample_array = random.sample(range(0, k), n)
        sample_arrays.append(sample_array)

    sample_results = []
    for sample_array in sample_arrays:
        sample_results.append({'id_arr': [i + 1 for i in sample_array], 'elements': [input_info[i] for i in sample_array]})
    return sample_results


def number_to_ordinal(n):
    """
    Convert an integer to its ordinal representation.

    :param n: Integer to convert
    :return: String ordinal representation of the integer
    """
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return str(n) + suffix


def element_sample(instruction, params_dict, input_data, input_info):
    # 0 ~ 10 抽5个 k-10到k抽5个， 中间抽10个
    # 0 ~ 10 抽2个 k-10到k抽2个， 中间抽2个
    random.seed(0)
    k = int(input_data['label'])
    assert k > 20

    sample_ints = []
    sample_ints += random.sample(range(1, 11), 2)
    sample_ints += random.sample(range(11, k - 10), 2)
    sample_ints += random.sample(range(k - 10, k - 1), 2)
    sample_results = []
    for n in sample_ints:
        line = {
            'id': n + 1,
            'ordid': number_to_ordinal(n + 1),
            'element': input_info[n],
            'pre_element': input_info[n - 1],
            'post_element': input_info[n + 1]
            }
        if isinstance(instruction, list):
            line['bias'] = instruction[1]
        sample_results.append(line)
    return sample_results


def direct_sample_dup(instruction, params_dict, input_data, input_info):
    k = input_data['label']
    dup_idxs = input_info['dupli_idxs']
    doc_list = input_info['doc_list'][:k]
    dup_idxs_ = []
    for i, di in enumerate(dup_idxs):
        di_ = list(filter(lambda x: x < k, di))
        if len(di_) > 1:
            dup_idxs_.append(di_)
    # print(dup_idxs_)
    params = []
    for pd in params_dict:
        info_keys = pd['info_keys'].split(' ')
        pd['dup_idxs'] = dup_idxs_
        dup_infos = []
        for dup_idx in dup_idxs_:
            dup_infos.append([{key: doc_list[i].get(key, None) for key in info_keys} for i in dup_idx])
        pd['dup_infos'] = dup_infos
        params.append(pd)
        # print(pd)
    # raise "123332123123"
    return params


def direct_sample_label(instruction, params_dict, input_data, input_info):
    k = input_data['label']
    doc_list = input_info['doc_list'][:k]
    keys = []
    for doc in doc_list:
        keys.append(list(doc.keys()))
    params = []
    for pd in params_dict:
        pd['doc_keys'] = keys
        params.append(pd)
    return params

def main(base_dir: str):
    BENCHMAKR_BASE  = base_dir

    prompt_generator = PromptGenerator()
    prompt_generator.generate_prompts()

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
