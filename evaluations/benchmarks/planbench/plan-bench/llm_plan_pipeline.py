import random
import argparse
import os
import json
import yaml
from prompt_generation import PromptGenerator
from response_evaluation import ResponseEvaluator
from response_generation import ResponseGenerator


def is_task_complete(config_file, engine, task_name):
    """Check if all instances already have responses and evaluation results."""
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    domain = data['domain_name']
    # Check responses
    resp_path = f"responses/{domain}/{engine}/{task_name}.json"
    if not os.path.exists(resp_path):
        return False
    try:
        with open(resp_path, 'r') as f:
            resp_data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return False
    instances = resp_data.get('instances', [])
    if not instances:
        return False
    all_responded = all(inst.get('llm_raw_response') for inst in instances)
    if not all_responded:
        return False
    # Check results exist and have actual evaluation data (not all None)
    results_path = f"results/{domain}/{engine}/{task_name}.json"
    if not os.path.exists(results_path):
        return False
    try:
        with open(results_path, 'r') as f:
            results_data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return False
    result_instances = results_data.get('instances', [])
    if not result_instances:
        return False
    # At least one instance must have a non-None is_correct value
    return any(inst.get('is_correct') is not None for inst in result_instances)


if __name__=="__main__":
    random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Task to run \
    \n t1 = Plan Generation\
    \n t2 = Optimal Planning \
    \n t3 = Plan Verification \
    \n t4 = Plan Reuse\
    \n t5 = Plan Generalization\
    \n t6 = Replanning \
    \n t7 = Reasoning about Plan Execution \
    \n t8_1 = Goal Reformulation (Goal shuffling) \
    \n t8_2 = Goal Reformulation (Full -> Partial) \
    \n t8_3 = Goal Reformulation (Partial -> Full) \
    ')
    #config
    parser.add_argument('--config', type=str, required=True, help='Config file name (no need to add .yaml)')
    
    parser.add_argument('--engine', type=str, required=True, help='Engine to use \
                        \n gpt-4_chat = GPT-4 \
                        \n bloom = Bloom \
                        \n gpt-3.5-turbo_chat = GPT-3.5 Turbo \
                        \n davinci = GPT-3 Davinci \
                        \n curie = GPT-3 Curie \
                        \n babbage = GPT-3 Babbage \
                        \n ada = GPT-3 Ada \
                        ')
    
    parser.add_argument('--run_till_completion', type=str, default="True", help='Run till completion')
    parser.add_argument('--verbose', type=str, default="False", help='Verbose')
    parser.add_argument('--ignore_existing', action='store_true', help='Ignore existing output')
    parser.add_argument('--specific_instances', nargs='+', type=int, default=[], help='List of instances to run')
    parser.add_argument('--random_example', type=str, default="False", help='Random example')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_workers', type=int, default=3, help='Number of parallel workers')
    parser.add_argument('--api_base', type=str, default=None, help='Custom API base URL (bypasses CMU proxy)')
    parser.add_argument('--api_key', type=str, default=None, help='API key for custom endpoint')
    parser.add_argument('--engine_alias', type=str, default=None, help='Clean name for output dirs (defaults to engine)')
    args = parser.parse_args()
    task = args.task
    config = args.config
    engine = args.engine
    verbose = eval(args.verbose)
    specified_instances = args.specific_instances
    seed=args.seed
    ignore_existing = args.ignore_existing
    max_workers = args.max_workers
    random_example = eval(args.random_example)
    run_till_completion = eval(args.run_till_completion)
    # print(task, config, verbose, specified_instances, random_example)
    config_file = f'./configs/{config}.yaml'


    # ========================= Prompt Generation =========================
    # Build task_name early so we can check if prompts already exist
    task_dict = {
        't1': 'task_1_plan_generation',
        't2': 'task_2_plan_optimality',
        't3': 'task_3_plan_verification',
        't4': 'task_4_plan_reuse',
        't5': 'task_5_plan_generalization',
        't6': 'task_6_replanning',
        't7': 'task_7_plan_execution',
        't8_1': 'task_8_1_goal_shuffling',
        't8_2': 'task_8_2_full_to_partial',
        't8_3': 'task_8_3_partial_to_full',
    }
    try:
        task_name = task_dict[task]
    except:
        raise ValueError("Invalid task name")

    with open(config_file, 'r') as f:
        _cfg = yaml.safe_load(f)
    prompt_file = f"prompts/{_cfg['domain_name']}/{task_name}.json"

    if os.path.exists(prompt_file):
        print(f"[SKIP] Prompt file already exists: {prompt_file}")
    else:
        prompt_generator = PromptGenerator(config_file, verbose, ignore_existing, seed)
        if task == 't1':
            prompt_generator.task_1_plan_generation(specified_instances, random_example)
        elif task == 't2':
            prompt_generator.task_2_plan_optimality(specified_instances, random_example)
        elif task == 't3':
            prompt_generator.task_3_plan_verification(specified_instances)
        elif task == 't4':
            prompt_generator.task_4_plan_reuse(specified_instances)
        elif task == 't5':
            prompt_generator.task_5_plan_generalization(specified_instances, random_example)
        elif task == 't6':
            prompt_generator.task_6_replanning(specified_instances, random_example)
        elif task == 't7':
            prompt_generator.task_7_plan_execution(specified_instances, random_example)
        elif task == 't8_1':
            prompt_generator.task_8_1_goal_shuffling(specified_instances)
        elif task == 't8_2':
            prompt_generator.task_8_2_full_to_partial(specified_instances)
        elif task == 't8_3':
            prompt_generator.task_8_3_partial_to_full(specified_instances)
    
    # ========================= Response Generation + Evaluation =========================
    engine_alias = args.engine_alias or engine
    response_generator = ResponseGenerator(config_file, engine, verbose, ignore_existing,
                                           api_base=args.api_base, api_key=args.api_key,
                                           engine_alias=engine_alias)

    if not ignore_existing and is_task_complete(config_file, engine_alias, task_name):
        print(f"[SKIP] {task_name} for {engine_alias} already complete (responses + results exist). Use --ignore_existing to re-run.")
        exit(0)

    # Set up evaluator early so we can run it incrementally after each response wave
    response_evaluator = ResponseEvaluator(config_file, engine_alias, specified_instances, verbose, ignore_existing)
    eval_plan_dict = {
        't1': 'task_1_plan_generation',
        't2': 'task_2_plan_optimality',
        't4': 'task_4_plan_reuse',
        't5': 'task_5_plan_generalization',
        't6': 'task_6_replanning',
        't8_1': 'task_8_1_goal_shuffling',
        't8_2': 'task_8_2_full_to_partial',
        't8_3': 'task_8_3_partial_to_full',
    }
    eval_state_dict = {
        't7': 'task_7_plan_execution'
    }
    eval_verification_dict = {
        't3': 'task_3_plan_verification'
    }

    def run_eval():
        """Run evaluation on whatever responses are available so far."""
        print(f"[EVAL] Running incremental evaluation for {task_name}...")
        if task in eval_plan_dict:
            response_evaluator.evaluate_plan(task_name)
        elif task in eval_state_dict:
            response_evaluator.evaluate_state(task_name)
        elif task in eval_verification_dict:
            response_evaluator.evaluate_verification(task_name)

    response_generator.get_responses(task_name, run_till_completion=run_till_completion,
                                     max_workers=max_workers, on_wave_complete=run_eval)

    # ========================= Final Evaluation Pass =========================
    with open(config_file, 'r') as f:
        _cfg2 = yaml.safe_load(f)
    resp_file = f"responses/{_cfg2['domain_name']}/{engine_alias}/{task_name}.json"
    if not os.path.exists(resp_file):
        print(f"[SKIP] No response file found at {resp_file} — skipping evaluation.")
        exit(1)

    print(f"[EVAL] Final evaluation pass for {task_name}...")
    run_eval()


