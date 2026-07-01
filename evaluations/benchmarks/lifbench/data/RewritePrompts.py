import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.LLMApi import llmApi
import json
import os.path
from collections import defaultdict




def get_prompts_dict(input_file):
    metadata = json.load(open(input_file, 'r'))
    subtask_original_instruction = {}
    for task in metadata.values():
        subtasks = task['subtasks']
        for subtask, d in subtasks.items():
            ori_instruction = d['instructions'][0]
            if isinstance(ori_instruction, list):
                ori_instruction = ori_instruction[0]
            subtask_original_instruction[subtask] = ori_instruction
    return subtask_original_instruction


def rewrite_prompt(model, input_prompt):
    prompt = '''
    Please rewrite the given prompt according to the following requirements:

    1. The rewritten prompt must retain the same meaning as the original without altering the intent.
    2. Try your best to use different vocabulary or sentence structures.
    3. Ensure that the rewritten prompt is clear and accurate, avoiding any expressions that could lead to ambiguity.
    4. Please keep the placeholders in the prompt (i.e. “{{}}” and the contents therein) exactly as they are during rewriting.
    5. Please keep the example in the prompt, but you can make some small changes according to the prompt while keeping the original meaning.
    6. Output the result in Json List format, without anything else.
    7. Please generate 20 different rewrites at once. 

    prompt: {}'''.format(input_prompt)

    llmapi = llmApi(model)
    for i in range(3):
        llm_output = llmapi.get(prompt, max_tries=3)
        try:
            l_idx, r_idx = llm_output.find('['), llm_output.rfind(']')
            parse_results = json.loads(llm_output[l_idx:r_idx + 1])
            break
        except:
            parse_results = ""
            print(f"Error parsing, retrieving {i + 1}.")
    return llm_output, parse_results


def main(
        input_file='./data/meta/tasks.json',
        output_dir='./data/meta/rewrite/',
        models=[],
        ):
    prompts_dict = get_prompts_dict(input_file)
    for sub_task, ori_prompt in prompts_dict.items():
        # breakpoint handling
        try:
            print("Load existent rewrites.")
            prompts_rewites_results = json.load(open(os.path.join(output_dir, f'{sub_task}_rewrite.json'), 'r'))
            prompts_rewites_results['ori'] = ori_prompt
        except:
            print("Create New Rewrites.")
            prompts_rewites_results = defaultdict(dict)
            prompts_rewites_results['ori'] = ori_prompt
        for model in models:
            if model in prompts_rewites_results:
                parse_result = prompts_rewites_results[model]['parse_result']
                if isinstance(parse_result, list) and all([isinstance(sentence, str) for sentence in parse_result]):
                    continue
            print(f"Model: {model} Sub Task: {sub_task}")
            llm_output, parse_result = rewrite_prompt(model, ori_prompt)
            print(f"parse result: {'OK' if parse_result else 'ERROR'}")
            if not llm_output: continue
            print(llm_output)
            prompts_rewites_results[model] = {"llm_output": llm_output, "parse_result": parse_result}
            json.dump(prompts_rewites_results, open(os.path.join(output_dir, f'{sub_task}_rewrite.json'), 'w'), indent=2)


def parse_check(): # Check if the rewrite can be parsed
    prompts_dict = get_prompts_dict()
    for sub_task, ori_prompt in prompts_dict.items():
        prompts_rewites_results = json.load(open(f'/data/xdwu/data/LI_tasks/Benchmarkbase1/meta/rewrite/{sub_task}_rewrite.json', 'r'))
        for model in prompts_rewites_results:
            llm_output, parse_result = prompts_rewites_results[model].values()
            if parse_result and isinstance(parse_result, list):
                print(f"Model: {model} Sub Task: {sub_task} Parse Result: OK, len: {len(parse_result)}")
                continue
            l_idx, r_idx = llm_output.find('['), llm_output.rfind(']')
            try:
                if l_idx == -1 or r_idx == -1:
                    raise 'cannot find [] in llm_output'
                parse_result = json.loads(llm_output[l_idx: r_idx + 1])
                prompts_rewites_results[model]['parse_result'] = parse_result
                print('parse ok: {sub_task}-{model}.')
            except:
                print(f"parse failed: {sub_task}-{model}.")
                continue
            json.dump(prompts_rewites_results, open(f'/data/xdwu/data/LI_tasks/Benchmarkbase1/meta/rewrite/{sub_task}_rewrite.json', 'w'), indent=2)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
