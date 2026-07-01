import warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

import json
import os
import time
from datasets import load_dataset
from tqdm import tqdm
import litellm
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# RETRY FUNCTION
# =========================================================
def call_with_retry(completion_params, max_retries=6, backoff_base=2):

    last_exception = None

    for attempt in range(max_retries):
        try:
            response = litellm.completion(**completion_params)
            return response

        except Exception as e:
            last_exception = e

            if attempt < max_retries - 1:
                sleep_time = backoff_base ** attempt
                print(f"⚠️ Attempt {attempt+1} failed. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                break

    raise last_exception


# =========================================================
# SAVE FUNCTION
# =========================================================
def save_results(results, output_file):

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


# =========================================================
# MODEL RUNNER
# =========================================================
def run_model(
    model_name,
    dataset,
    api_key,
    base_url,
    max_samples,
    temperature,
    max_tokens
):

    clean_model_name = model_name.split("/")[-1]
    output_file = f"outputs_{clean_model_name}.jsonl"

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    results = []
    errors = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    print("\n" + "=" * 60)
    print(f"🚀 Running model: {clean_model_name}")
    print("=" * 60)

    for idx, sample in enumerate(tqdm(dataset, desc=clean_model_name)):

        instruction = sample["instruction"]
        input_text = sample.get("input", "")

        if input_text and input_text.strip():
            prompt = f"Instruction: {instruction}\nInput: {input_text}"
        else:
            prompt = f"Instruction: {instruction}"

        messages = [{"role": "user", "content": prompt}]

        completion_params = {
            "api_key": api_key,
            "base_url": base_url,
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": 300,
        }

        try:
            response = call_with_retry(completion_params)

            output = response.choices[0].message.content

            usage = getattr(response, "usage", None)

            if usage:
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
            else:
                prompt_tokens = None
                completion_tokens = None

        except Exception as e:

            errors += 1
            output = f"ERROR: {str(e)}"
            prompt_tokens = None
            completion_tokens = None

        result = {key: sample[key] for key in sample.keys()}
        result["output"] = output
        result["prompt_tokens"] = prompt_tokens
        result["completion_tokens"] = completion_tokens

        results.append(result)

        if (idx + 1) % 10 == 0:
            save_results(results, output_file)

        # small delay to avoid proxy throttling
        time.sleep(0.2)

    save_results(results, output_file)

    stats = {
        "model": clean_model_name,
        "total": len(results),
        "successful": len(results) - errors,
        "errors": errors,
        "tokens": total_prompt_tokens + total_completion_tokens
    }

    print(f"\n✅ Finished {clean_model_name}")

    return stats


# =========================================================
# ENTRYPOINT
# =========================================================
if __name__ == "__main__":

    API_KEY = os.environ.get("LITELLM_API_KEY")
    BASE_URL = "https://cmu.litellm.ai"

    MODELS = [
        "litellm_proxy/gemini/gemini-2.5-flash",
    ]

    MAX_SAMPLES = None
    TEMPERATURE = 1
    MAX_TOKENS = 30000

    print("=" * 60)
    print("SEQUENTIAL INFOBENCH RUNNER")
    print("=" * 60)
    print(f"Total models: {len(MODELS)}")
    print("=" * 60)

    dataset = load_dataset("kqsong/InFoBench")["train"]

    start_time = time.time()
    stats = []

    for model in MODELS:
        stats.append(
            run_model(
                model,
                dataset,
                API_KEY,
                BASE_URL,
                MAX_SAMPLES,
                TEMPERATURE,
                MAX_TOKENS
            )
        )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for s in stats:

        success_rate = (s["successful"] / s["total"]) * 100 if s["total"] else 0

        print(f"\n{s['model']}")
        print(f"  Success: {s['successful']}/{s['total']} ({success_rate:.1f}%)")
        print(f"  Errors: {s['errors']}")
        print(f"  Tokens: {s['tokens']:,}")

    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    print("=" * 60)
    print("✅ ALL MODELS COMPLETE")
    print("=" * 60)