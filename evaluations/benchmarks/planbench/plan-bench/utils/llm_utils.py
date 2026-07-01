import os
import time
import litellm

litellm.drop_params = True

MAX_RETRIES = 5
INITIAL_BACKOFF = 2  # seconds


def send_query(query, engine, max_tokens, model=None, stop="[STATEMENT]",
               api_base=None, api_key=None):
    messages = [
        {"role": "system", "content": "You are the planner assistant who comes up with correct plans."},
        {"role": "user", "content": query}
    ]

    for attempt in range(MAX_RETRIES):
        try:
            if api_base:
                # Custom endpoint (e.g. RunPod, OpenRouter)
                response = litellm.completion(
                    api_key=api_key,
                    base_url=api_base,
                    model=engine,
                    messages=messages,
                    max_tokens=max_tokens,
                )
            else:
                # Default: CMU LiteLLM proxy
                response = litellm.completion(
                    api_key=os.environ.get("LITELLM_PROXY_API_KEY"),
                    api_base="https://cmu.litellm.ai",
                    model=f"litellm_proxy/{engine}",
                    messages=messages,
                    max_tokens=max_tokens,
                )
            text_response = response.choices[0].message.content or ""
            return text_response.strip()
        except Exception as e:
            error_str = str(e).lower()
            is_retryable = any(kw in error_str for kw in [
                "rate_limit", "ratelimit", "429", "too many requests",
                "timeout", "timed out", "connection", "server_error", "500", "502", "503",
            ])
            if is_retryable and attempt < MAX_RETRIES - 1:
                wait = INITIAL_BACKOFF * (2 ** attempt)
                print(f"[!] Retryable error (attempt {attempt+1}/{MAX_RETRIES}), waiting {wait}s: {e}")
                time.sleep(wait)
            else:
                import traceback
                print("[-]: Failed LLM query execution: {}".format(e))
                traceback.print_exc()
                return ""
    return ""
