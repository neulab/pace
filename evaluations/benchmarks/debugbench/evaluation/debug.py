import os
import sys
import json
import tqdm
import time
from leetcode_oj import LeetCodeTester, LeetCodeTesterPool
from debugger import GPT4Responser, TurboResponser, IODebugger, LiteLLMResponser
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser
SETTING = "debug"

WORK_DIR = "evaluation"
SRC_DIR = f"benchmark"

Responser = LiteLLMResponser
# {'gpt-4': GPT4Responser, 'gpt-35-turbo': TurboResponser}[MODEL]


def load_bug_data():
    """ load data with different languages and bug types """
    res = {
        # 'cpp': {},
        # 'java': {},
        'python3': {},
    }
    files = os.listdir(SRC_DIR)
    for file in files:
        if "python3" not in file:
            continue
        file_name = os.path.splitext(file)[0]
        lang = file_name[:file_name.find('_')]
        bug_type = file_name[file_name.find('_') + 1:]
        res[lang][bug_type] = json.load(open(os.path.join(SRC_DIR, file)))
    return res


def debug_case(case, debugger, tester, lang):
    fixed_code, fixing_exp = debugger.debug(lang=lang, code=case['buggy_code'])
    case['fixed_code'] = fixed_code
    case['fixing_exp'] = fixing_exp
    rw, res_dict = tester.test(code=fixed_code, language=lang, task_id=case['slug'])
    case['test_result_bool'] = rw
    case['test_result_dict'] = res_dict
    return case

def main(args):
    global MODEL, SAVE_DIR
    MODEL = args.model
    SAVE_DIR = f"{WORK_DIR}/res/{MODEL}/{SETTING}"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    responser = Responser(model=MODEL)
    debugger = IODebugger(responser)
    leetcode_sessions = []
    csrf_tokens = []
    with open("evaluation/keys.json", 'r') as f:
        keys = json.load(f)
        import random
        random.shuffle(keys)
        for key in keys[args.start_idx : (args.start_idx + args.cnt)]:
            leetcode_sessions.append(key['leetcode_session'])
            csrf_tokens.append(key['csrf_token'])
    print(len(leetcode_sessions), len(csrf_tokens))
    tester_pool = LeetCodeTesterPool(leetcode_sessions=leetcode_sessions, csrf_tokens=csrf_tokens, cooldown=args.timeout)

    bug_data = load_bug_data()
    for lang in bug_data.keys():
        bug_types = list(bug_data[lang].keys())
        import random
        random.shuffle(bug_types)
        # print(bug_types)
        for bug_type in bug_types:
            save_dir = os.path.join(SAVE_DIR, f"{lang}_{bug_type}.json")
            if not os.path.exists(save_dir):
                bug_data_split = bug_data[lang][bug_type]
                # if len(bug_data_split) > 100:
                #     continue
                res = []
                import random
                random.shuffle(bug_data_split)
                for case in tqdm.tqdm(bug_data_split, desc=f"{lang}_{bug_type}"):
                    try:
                        fixed_code, fixing_exp = debugger.debug(lang=lang, code=case['buggy_code'])
                        case['fixed_code'] = fixed_code
                        case['fixing_exp'] = fixing_exp
                        rw, res_dict = tester_pool.test(code=fixed_code, language=lang, task_id=case['slug'])
                        case['test_result_bool'] = rw
                        case['test_result_dict'] = res_dict
                    except Exception as e:
                        # if RuntimeError from tester_pool occurred, then raise, else ignore
                        if isinstance(e, RuntimeError) and "All testers failed consecutively" in str(e):
                            raise e
                        case["test_result_bool"] = False
                        case['test_result_dict'] = {}
                        case['fixed_code'] = ""
                        case['fixing_exp'] = ""
                        print(f"error", flush=True)
                    except:
                        case["test_result_bool"] = False
                        case['test_result_dict'] = {}
                        case['fixed_code'] = ""
                        case['fixing_exp'] = ""
                        print("error2", flush=True)
                    res.append(case)
                    import time
                    time.sleep(20)
                with open(save_dir, 'w') as f:
                    json.dump(res, f, indent=4)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0, help="start index of the keys to use")
    parser.add_argument("--cnt", type=int, default=1, help="number of keys to use")
    parser.add_argument("--timeout", type=int, default=25, help="cooldown time for each submission in seconds")
    args = parser.parse_args()
    timeout_duration = 10*60 # 10 minutes
    err_cnt = 0
    while True:
        try:
            main(args)
            break
        except:
            print(f"Error occurred. Retrying...", flush=True)
            err_cnt += 1
            final_timeout = min(timeout_duration*err_cnt, 1*60*60) # cap the timeout to 1 hour
            time.sleep(final_timeout)
