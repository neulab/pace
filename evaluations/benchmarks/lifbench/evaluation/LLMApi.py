import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from openai import OpenAI
import tiktoken
import os

API_BASE = os.environ.get("LLM_BASE_URL", "")
API_KEY = os.environ.get("LLM_API_KEY", "")

# Reasoning models that don't support streaming and use max_completion_tokens
REASONING_MODELS = ['o1', 'o1-mini', 'o1-preview']

# OpenAI thinking models: reasoning_effort="low", no max_tokens
THINKING_MODELS = ['o3', 'o4', 'o3-mini', 'o4-mini', 'gpt-5']

# gemini-2.5-flash: can disable thinking (thinking_budget=0 → reasoning_effort="none")
GEMINI_NO_THINK_MODELS = ['gemini-2.5-flash']

# gemini-2.5-pro: min thinking_budget=128 → reasoning_effort="low"
GEMINI_25PRO_MODELS = ['gemini-2.5-pro']

# gemini-3-pro: min thinking_level="low" → reasoning_effort="low"
GEMINI_3_MODELS = ['gemini-3']

# Models that don't support streaming (use non-streaming directly)
NO_STREAMING_MODELS = ['gpt-5', 'gpt-oss', 'minimax', 'glm-4p7', 'deepseek', 'kimi', 'nemotron', 'glm-5', 'gemini-3']


def is_no_streaming_model(model_name):
    model_lower = model_name.lower()
    for m in NO_STREAMING_MODELS:
        if m in model_lower:
            return True
    return False


def is_reasoning_model(model_name):
    """Check if model is a reasoning model (o1 series only, no way to disable thinking)"""
    model_lower = model_name.lower()
    for rm in REASONING_MODELS:
        if rm in model_lower:
            return True
    return False


def is_thinking_model(model_name):
    model_lower = model_name.lower()
    for tm in THINKING_MODELS:
        if tm in model_lower:
            return True
    return False


def is_gemini_no_think(model_name):
    if model_name.startswith('gemini/'):
        return False
    model_lower = model_name.lower()
    for m in GEMINI_NO_THINK_MODELS:
        if m in model_lower:
            return True
    return False


def is_gemini_think_low(model_name):
    if model_name.startswith('gemini/'):
        return False
    model_lower = model_name.lower()
    for m in GEMINI_25PRO_MODELS + GEMINI_3_MODELS:
        if m in model_lower:
            return True
    return False


class llmApi():
    def __init__(self, model="gpt-4o-2024-08-06", api_key="", api_base=""):
        if api_base or API_BASE:
            self.client = OpenAI(
                api_key=api_key if api_key else API_KEY,
                base_url=api_base if api_base else API_BASE,
            )
        else:
            self.client = OpenAI(api_key=API_KEY)
        self.model = model
        self.is_reasoning = is_reasoning_model(model)
        self.is_thinking = is_thinking_model(model)
        self.is_gemini_no_think = is_gemini_no_think(model)
        self.is_gemini_think_low = is_gemini_think_low(model)
        self.tokenizer = tiktoken.encoding_for_model('gpt-4o-2024-08-06')

    def _gemini_request(self, prompt, max_new_tokens):
        if self.is_gemini_no_think:
            return self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                timeout=120,
                extra_body={"thinking_config": {"thinking_budget": 0}}
            )
        else:
            if is_gemini_no_think(self.model) is False and 'gemini-3' in self.model.lower():
                extra = {"thinking_config": {"thinking_level": "low"}}
            else:
                extra = {"thinking_config": {"thinking_budget": 128}}
            return self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                timeout=120,
                extra_body=extra
            )

    def get(self, prompt, max_new_tokens=512, max_tries=3):
        for i in range(max_tries):
            try:
                if self.is_gemini_no_think or self.is_gemini_think_low:
                    executor = ThreadPoolExecutor(max_workers=1)
                    future = executor.submit(self._gemini_request, prompt, max_new_tokens)
                    try:
                        response = future.result(timeout=300)
                    except FuturesTimeoutError:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    executor.shutdown(wait=False)
                elif self.is_reasoning:
                    response = self.client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model,
                        max_completion_tokens=max_new_tokens,
                        timeout=300
                    )
                elif self.is_thinking:
                    response = self.client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model,
                        reasoning_effort="low",
                        timeout=300
                    )
                else:
                    is_thinking_litellm = 'thinking' in self.model.lower() or 'reasoner' in self.model.lower() or 'gemini-3' in self.model.lower() or 'k2.5' in self.model.lower()
                    if is_thinking_litellm:
                        thinking_tokens = max(max_new_tokens * 32, 16384)
                        if 'azure_ai/' in self.model:
                            response = self.client.chat.completions.create(
                                messages=[{"role": "user", "content": prompt}],
                                model=self.model,
                                max_completion_tokens=thinking_tokens,
                                timeout=900
                            )
                        else:
                            response = self.client.chat.completions.create(
                                messages=[{"role": "user", "content": prompt}],
                                model=self.model,
                                max_tokens=thinking_tokens,
                                timeout=900
                            )
                    else:
                        response = self.client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model=self.model,
                            max_tokens=max_new_tokens,
                            timeout=300
                        )
                msg = response.choices[0].message
                content = msg.content
                if not content:
                    print(f"API returned empty content for {self.model}")
                    print(f"  finish_reason: {response.choices[0].finish_reason}")
                    print(f"  message keys: {vars(msg).keys()}")
                    print(f"  message: {msg}")
                return content or ""
            except FuturesTimeoutError:
                print(f"Wall-clock timeout (300s) for {self.model}, skipping")
                return ""
            except Exception as e:
                if self.is_gemini_no_think or self.is_gemini_think_low:
                    print(f"API Error: {e}, skipping (gemini no retry)")
                    return ""
                if 'timed out' in str(e).lower() or 'timeout' in str(e).lower():
                    print(f"API timeout for {self.model}, skipping")
                    return ""
                time.sleep(30)
                print(f"API Error: {e}, retry: {i + 1}")
        return ""

    def get_by_stream(self, prompt, max_new_tokens=512, max_tries=3):
        # Reasoning models and no-streaming models use regular get instead
        if self.is_reasoning or self.is_thinking or self.is_gemini_no_think or self.is_gemini_think_low or is_no_streaming_model(self.model):
            return self.get(prompt, max_new_tokens, max_tries)

        for i in range(max_tries):
            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    max_tokens=max_new_tokens,
                    stream=True,
                    timeout=300
                )

                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content

                if not full_response:
                    print(f"Streaming returned empty for {self.model}")
                return full_response

            except Exception as e:
                time.sleep(30)
                print(f"API Error: {e}, retry: {i + 1}")
        return ""


def api_get(prompt, model="gpt-4o-2024-05-13", maxtries=3):
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
        )
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
                }
            ],
        model=model,
        )
    return response.choices[0].message.content

