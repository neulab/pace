import os
import random

import yaml
from Executor import Executor
from utils import *
from pathlib import Path
from tarski.io import PDDLReader
import argparse
import time
import json
np.random.seed(42)
import copy
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
class ResponseGenerator:
    def __init__(self, config_file, engine, verbose, ignore_existing,
                 api_base=None, api_key=None, engine_alias=None):
        self.engine = engine
        self.engine_alias = engine_alias or engine
        self.verbose = verbose
        self.ignore_existing = ignore_existing
        self.max_gpt_response_length = 4096
        self.data = self.read_config(config_file)
        self.model = None
        self.api_base = api_base
        self.api_key = api_key
    def read_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def get_responses(self, task_name, specified_instances=[], run_till_completion=False, max_workers=10, on_wave_complete=None):
        output_dir = f"responses/{self.data['domain_name']}/{self.engine_alias}/"
        os.makedirs(output_dir, exist_ok=True)
        output_json = output_dir + f"{task_name}.json"
        save_lock = threading.Lock()

        while True:
            if os.path.exists(output_json):
                with open(output_json, 'r') as file:
                    structured_output = json.load(file)
            else:
                prompt_dir = f"prompts/{self.data['domain_name']}/"
                assert os.path.exists(prompt_dir + f"{task_name}.json")
                with open(prompt_dir + f"{task_name}.json", 'r') as file:
                    structured_output = json.load(file)
                structured_output['engine'] = self.engine

            # Collect instances that need processing
            pending = []
            for instance in structured_output["instances"]:
                if "llm_raw_response" in instance:
                    if instance["llm_raw_response"] and not self.ignore_existing:
                        continue
                if len(specified_instances) > 0:
                    if instance['instance_id'] not in specified_instances:
                        continue
                    else:
                        specified_instances.remove(instance['instance_id'])
                pending.append(instance)

            if not pending:
                break

            stop_statement = "[STATEMENT]"
            if 'caesar' in self.data['domain_name']:
                stop_statement = caesar_encode(stop_statement)

            failed_instances = []

            def process_instance(instance):
                query = instance["query"]
                return send_query(query, self.engine, self.max_gpt_response_length, model=self.model, stop=stop_statement,
                                  api_base=self.api_base, api_key=self.api_key)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_instance = {executor.submit(process_instance, inst): inst for inst in pending}
                for future in tqdm(as_completed(future_to_instance), total=len(future_to_instance)):
                    instance = future_to_instance[future]
                    llm_response = future.result()
                    if not llm_response:
                        failed_instances.append(instance['instance_id'])
                        print(f"Failed instance: {instance['instance_id']}")
                        continue
                    if self.verbose:
                        print(f"Instance {instance['instance_id']}: {llm_response[:80]}...")
                    instance["llm_raw_response"] = llm_response
                    with save_lock:
                        with open(output_json, 'w') as file:
                            json.dump(structured_output, file, indent=4)

            # Run incremental evaluation after each wave
            if on_wave_complete:
                try:
                    on_wave_complete()
                except Exception as e:
                    print(f"[!] Incremental evaluation error (non-fatal): {e}")

            if run_till_completion:
                if len(failed_instances) == 0:
                    break
                elif len(failed_instances) == len(pending):
                    print(f"[!] All {len(pending)} instances failed — model likely unavailable. Aborting.")
                    break
                else:
                    print(f"Retrying {len(failed_instances)} failed instances: {failed_instances}")
                    time.sleep(10)
            else:
                break
        
            
        
        
        
            

    


if __name__=="__main__":
    random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Task to run \
    \n t1 = Plan Generation\
    \n t2 = Optimal Planning \
    \n t3 = Plan Verification \
    \n t4 = Plan Reuse\
    \n t5 = Plan Generalization\
    \n t6 = Replanning (easier) \
    \n t7 = Reasoning about Plan Execution \
    \n t8_1 = Goal Reformulation (Goal shuffling) \
    \n t8_2 = Goal Reformulation (Full -> Partial) \
    \n t8_3 = Goal Reformulation (Partial -> Full) \
    ')
    parser.add_argument('--engine', type=str, required=True, help='Engine to use \
                        \n gpt-4_chat = GPT-4 \
                        \n bloom = Bloom \
                        \n gpt-3.5-turbo_chat = GPT-3.5 Turbo \
                        \n davinci = GPT-3 Davinci \
                        \n curie = GPT-3 Curie \
                        \n babbage = GPT-3 Babbage \
                        \n ada = GPT-3 Ada \
                        ')
                        
    parser.add_argument('--verbose', type=str, default="False", help='Verbose')
    #config
    parser.add_argument('--config', type=str, required=True, help='Config file name (no need to add .yaml)')
    parser.add_argument('--run_till_completion', type=str, default="True", help='Run till completion')
    parser.add_argument('--specific_instances', nargs='+', type=int, default=[], help='List of instances to run')
    parser.add_argument('--ignore_existing', action='store_true', help='Ignore existing output')
    parser.add_argument('--max_workers', type=int, default=3, help='Number of parallel workers')
    parser.add_argument('--api_base', type=str, default=None, help='Custom API base URL (bypasses CMU proxy)')
    parser.add_argument('--api_key', type=str, default=None, help='API key for custom endpoint')
    parser.add_argument('--engine_alias', type=str, default=None, help='Clean name for output dirs (defaults to engine)')
    args = parser.parse_args()
    task = args.task
    engine = args.engine
    config = args.config
    specified_instances = args.specific_instances
    verbose = eval(args.verbose)
    run_till_completion = eval(args.run_till_completion)
    ignore_existing = args.ignore_existing
    max_workers = args.max_workers
    print(f"Task: {task}, Engine: {engine}, Config: {config}, Verbose: {verbose}, Run till completion: {run_till_completion}, Workers: {max_workers}")
    config_file = f'./configs/{config}.yaml'
    response_generator = ResponseGenerator(config_file, engine, verbose, ignore_existing,
                                           api_base=args.api_base, api_key=args.api_key,
                                           engine_alias=args.engine_alias)
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
    response_generator.get_responses(task_name, specified_instances, run_till_completion=run_till_completion, max_workers=max_workers)





