import warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

import json
import os
import time
import tiktoken
import litellm

from os.path import join, exists
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
encoder = tiktoken.get_encoding("cl100k_base")

SYS_MSG = "Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:\n\n- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration. consider a question that asks. \"Does each sentence in the generated text use a second person?\" If even one sentence does not use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question\n\n- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks. \"Is the second sentence in the generated text a compound sentence?\" and the generated text only has one sentence. it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.'''"


def load_jsonl(file_path):
    _data = []
    with open(file_path, 'r') as f:
        for data in f:
            jline = json.loads(data)
            _data.append(jline)
    return _data


def is_valid_output(entry):
    """
    Only evaluate samples where generation succeeded.
    """
    if entry.get("output") is None:
        return False

    if entry.get("prompt_tokens") is None or entry.get("completion_tokens") is None:
        return False

    if isinstance(entry.get("output"), str) and entry["output"].startswith("ERROR"):
        return False

    return True


def bool_ratio(fpath):
    _data = load_jsonl(fpath)

    count = {"true": 0, "false": 0}

    for entry in _data:

        if entry.get("eval", None) is None:
            print("Wrong output")
            print(entry['id'])

        if len(entry['decomposed_questions']) != len(entry['eval']):
            print("Wrong length")
            print(entry['id'])

        if None in entry['eval']:
            print("None in eval")
            print(entry['id'])

        for eva_value in entry['eval']:
            if eva_value:
                count["true"] += 1
            else:
                count["false"] += 1

    print("-------- True False Table --------")
    print(count)

    if sum(count.values()) > 0:
        print(f"Percentage of True: {count['true']/sum(count.values())}")


def run_evaluation(api_key, base_url, in_path, o_dir, eval_model="litellm_proxy/gpt-4.1", temperature=0, max_context_tokens=128000):

    _data = load_jsonl(in_path)

    # ========================================
    # FILTER OUT FAILED GENERATIONS
    # ========================================
    original_count = len(_data)
    _data = [e for e in _data if is_valid_output(e)]
    skipped = original_count - len(_data)

    print(f"Skipped {skipped} samples with generation errors")
    print(f"Remaining valid samples: {len(_data)}")

    _model_name = os.path.splitext(os.path.basename(in_path))[0].replace('test_output_', '')

    _o_dir = join(o_dir, eval_model.replace('/', '_'))
    if not exists(_o_dir):
        os.makedirs(_o_dir, exist_ok=True)

    _opath = join(_o_dir, f"{_model_name}_DecomposeEval.json")

    if os.path.exists(_opath):
        _exist = load_jsonl(_opath)
        _exist_ids = [i['id'] for i in _exist]

        for pos, instance in enumerate(_data):
            if instance['id'] in _exist_ids:
                _data[pos] = _exist[_exist_ids.index(instance['id'])]

    result_writer = open(_opath, 'w')

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_api_calls = 0
    samples_processed = 0
    skipped_too_long = 0

    MAX_PROMPT_TOKENS = max_context_tokens - 1000

    print(f"--------Evaluating output from {in_path}--------")
    print(f"--------Evaluation Using {eval_model}--------")
    print(f"Total samples to process: {len(_data)}")
    print(f"Max prompt tokens: {MAX_PROMPT_TOKENS:,}")
    print()

    for entry_idx, entry in enumerate(tqdm(_data, desc="Processing samples")):

        input_task = entry['input']
        output = entry['output']

        sample_id = entry.get('id', f'sample_{entry_idx}')
        num_questions = len(entry['decomposed_questions'])

        input_tokens = len(encoder.encode(input_task)) if input_task else 0
        output_tokens = len(encoder.encode(output))

        tqdm.write(f"\n📝 Sample {entry_idx+1}/{len(_data)}: {sample_id}")
        tqdm.write(f"   Questions to evaluate: {num_questions}")
        tqdm.write(f"   Input length: {input_tokens:,} tokens, Output length: {output_tokens:,} tokens")

        answer = ""

        for q_idx, question in enumerate(
            tqdm(entry['decomposed_questions'], desc="  Evaluating questions", leave=False)
        ):

            if input_task:
                content = f"{SYS_MSG}\n\nInput:\n\"{input_task}\"\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
            else:
                content = f"{SYS_MSG}\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"

            single_message = [{"role": "user", "content": content}]

            prompt_token_count = len(encoder.encode(content))

            success = False
            retry_count = 0

            while not success:

                try:

                    start_time = time.time()

                    completion = litellm.completion(
                        api_key=api_key or os.environ.get("LITELLM_API_KEY"),
                        base_url=base_url,
                        model=eval_model,
                        messages=single_message,
                        temperature=temperature,
                        timeout=300,
                        max_tokens=500
                    )

                    elapsed = time.time() - start_time

                    generation = completion.choices[0].message.content

                    total_api_calls += 1

                    usage = getattr(completion, 'usage', None)

                    if usage:
                        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                        completion_tokens = getattr(usage, 'completion_tokens', 0)

                        total_prompt_tokens += prompt_tokens
                        total_completion_tokens += completion_tokens
                    else:
                        prompt_tokens = prompt_token_count
                        completion_tokens = len(encoder.encode(generation))

                        total_prompt_tokens += prompt_tokens
                        total_completion_tokens += completion_tokens

                    tqdm.write(
                        f"      ✓ Q{q_idx+1}/{num_questions}: {generation[:30]}... "
                        f"({prompt_tokens + completion_tokens} tokens, {elapsed:.1f}s)"
                    )

                    if generation.lower().startswith("yes"):
                        answer += "Yes\n"
                    elif generation.lower().startswith("no"):
                        answer += "No\n"
                    else:

                        if "YES" in generation and "NO" not in generation:
                            answer += "Yes\n"
                        elif "NO" in generation and "YES" not in generation:
                            answer += "No\n"
                        else:
                            tqdm.write(f"      ⚠️ Ambiguous answer: {generation}")
                            answer += "None\n"

                    success = True

                except Exception as e:

                    retry_count += 1

                    tqdm.write(f"      ❌ Error on Q{q_idx+1} (attempt {retry_count}): {e}")

                    if retry_count >= 3:
                        tqdm.write("      ⛔ Max retries reached")
                        answer += "None\n"
                        break

                    tqdm.write("      ⏳ Retrying in 20 seconds...")
                    time.sleep(20)

        answer = answer[:-1] if answer else ""

        bool_results = []

        for i in answer.split('\n'):
            if i == "Yes":
                bool_results.append(True)
            elif i == "No":
                bool_results.append(False)
            else:
                bool_results.append(None)

        entry['eval'] = bool_results

        result_writer.write(json.dumps(entry) + '\n')
        result_writer.flush()

        samples_processed += 1

        correct = sum([1 for r in bool_results if r])
        total = len(bool_results)

        if total > 0:
            tqdm.write(f"   ✅ Sample complete: {correct}/{total} correct ({correct/total*100:.1f}%)")
        else:
            tqdm.write("   ⚠️ Sample complete: No valid answers")

    result_writer.close()

    print("\n" + "="*60)
    print("📊 Token Usage Summary")
    print("="*60)

    print(f"Samples processed:       {samples_processed}")
    print(f"Questions skipped:       {skipped_too_long}")
    print(f"Total API calls:         {total_api_calls:,}")
    print(f"Total prompt tokens:     {total_prompt_tokens:,}")
    print(f"Total completion tokens: {total_completion_tokens:,}")
    print(f"Total tokens:            {(total_prompt_tokens + total_completion_tokens):,}")

    if total_api_calls > 0:
        print(f"Avg tokens per call:     {(total_prompt_tokens + total_completion_tokens)/total_api_calls:.1f}")

    print("="*60)
    print()

    bool_ratio(_opath)

    return _opath


if __name__ == "__main__":

    API_KEY = os.environ.get("LITELLM_API_KEY")

    BASE_URL = "https://cmu.litellm.ai"

    MODEL_NAME = "litellm_proxy/azure/gpt-4.1"

    MAX_CONTEXT = 128000

    INPUT_FILE = "../output_files/test_output_nemotron-3-nano-30b-a3b_free.jsonl"

    OUTPUT_DIR = "../evaluation/"

    TEMPERATURE = 0

    print("="*60)
    print("Starting InFoBench Evaluation")
    print("="*60)

    print(f"Input file:  {INPUT_FILE}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print(f"Model:       {MODEL_NAME}")
    print(f"Max context: {MAX_CONTEXT:,} tokens")
    print(f"Temperature: {TEMPERATURE}")

    print("="*60)
    print()

    if not exists(INPUT_FILE):
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        exit(1)

    run_evaluation(
        api_key=API_KEY,
        base_url=BASE_URL,
        in_path=INPUT_FILE,
        o_dir=OUTPUT_DIR,
        eval_model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_context_tokens=MAX_CONTEXT
    )

    print("\n✅ Evaluation complete!")