"""
Unified Model API module for proxy benchmarks.
Mimics lm-eval-harness model calling pattern using the LiteLLM proxy.
Supports all models: azure/*, gemini/*, neulab/*
"""

import os
import json
import logging
import asyncio
import copy
import time
import hashlib
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import aiohttp
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    Simple file-based cache for LLM responses.
    Caches successful responses to avoid redundant API calls.
    """

    def __init__(self, cache_dir: str = None, enabled: bool = True):
        self.enabled = enabled
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/llm_responses")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, str] = {}

    def _make_key(self, model: str, messages: List[Dict], gen_kwargs: Optional[Dict] = None) -> str:
        """Create a unique cache key from request parameters."""
        key_data = {
            "model": model,
            "messages": messages,
            "gen_kwargs": gen_kwargs or {},
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_cache_file(self, model: str) -> Path:
        """Get cache file path for a model."""
        safe_model = model.replace("/", "__").replace(":", "_")
        return self.cache_dir / f"{safe_model}_cache.jsonl"

    def get(self, model: str, messages: List[Dict], gen_kwargs: Optional[Dict] = None) -> Optional[str]:
        """Get cached response if available."""
        if not self.enabled:
            return None

        key = self._make_key(model, messages, gen_kwargs)

        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check file cache
        cache_file = self._get_cache_file(model)
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    for line in f:
                        entry = json.loads(line)
                        if entry.get("key") == key:
                            response = entry.get("response")
                            self._memory_cache[key] = response
                            return response
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        return None

    def set(self, model: str, messages: List[Dict], response: str, gen_kwargs: Optional[Dict] = None):
        """Cache a successful response."""
        if not self.enabled:
            return

        key = self._make_key(model, messages, gen_kwargs)
        self._memory_cache[key] = response

        # Append to file cache
        cache_file = self._get_cache_file(model)
        try:
            entry = {
                "key": key,
                "model": model,
                "response": response,
                "timestamp": datetime.now().isoformat(),
            }
            with open(cache_file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def load_model_cache(self, model: str):
        """Pre-load all cache entries for a model into memory."""
        if not self.enabled:
            return

        cache_file = self._get_cache_file(model)
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    for line in f:
                        entry = json.loads(line)
                        key = entry.get("key")
                        if key and key not in self._memory_cache:
                            self._memory_cache[key] = entry.get("response")
                logger.info(f"Loaded {len(self._memory_cache)} cached responses for {model}")
            except Exception as e:
                logger.warning(f"Cache load error: {e}")


# Global cache instance
_response_cache: Optional[ResponseCache] = None


def get_response_cache(cache_dir: str = None, enabled: bool = True) -> ResponseCache:
    """Get or create the global response cache."""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache(cache_dir=cache_dir, enabled=enabled)
    return _response_cache


@dataclass
class ModelConfig:
    """Configuration for model API calls."""
    model: str
    base_url: str = "https://cmu.litellm.ai/v1/chat/completions"
    max_tokens: int = 16384
    temperature: float = 0.0
    seed: int = 1234
    disable_seed: bool = False  # For non-Azure models that don't support seed
    timeout: int = 300
    max_retries: int = 5  # Increased from 3 for better rate limit handling
    num_concurrent: int = 1
    use_cache: bool = True  # Enable response caching
    cache_dir: str = None  # Cache directory (default: ~/.cache/llm_responses)

    def __post_init__(self):
        # Auto-detect if seed should be disabled based on model prefix
        if not self.model.startswith("azure/"):
            self.disable_seed = True


@dataclass
class GenerationResult:
    """Result from a single generation."""
    doc_id: Union[str, int]
    doc: Dict[str, Any]
    prompt: Union[str, List[Dict]]
    response: str
    raw_response: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProxyModelAPI:
    """
    Model API wrapper that uses the LiteLLM proxy.
    Compatible with lm-eval-harness model calling pattern.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = config.model
        self.base_url = config.base_url
        self._header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        # Initialize cache
        self.cache = get_response_cache(
            cache_dir=config.cache_dir,
            enabled=config.use_cache
        )
        # Pre-load cache for this model
        if config.use_cache:
            self.cache.load_model_cache(self.model)
        # Track cache stats
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def api_key(self) -> str:
        """Get API key from environment."""
        # Try multiple env vars
        for key_name in ["LITELLM_API_KEY", "OPENAI_API_KEY", "PROXY_API_KEY"]:
            key = os.environ.get(key_name)
            if key:
                return key
        raise ValueError(
            "API key not found. Please set LITELLM_API_KEY, OPENAI_API_KEY, or PROXY_API_KEY."
        )

    def _create_payload(
        self,
        messages: List[Dict[str, str]],
        gen_kwargs: Optional[Dict] = None,
    ) -> Dict:
        """Create the request payload."""
        gen_kwargs = gen_kwargs or {}

        max_tokens = gen_kwargs.pop("max_tokens", self.config.max_tokens)
        temperature = gen_kwargs.pop("temperature", self.config.temperature)
        stop = gen_kwargs.pop("stop", None)

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **gen_kwargs,
        }

        if stop:
            if not isinstance(stop, list):
                stop = [stop]
            payload["stop"] = stop[:4]  # Max 4 stop sequences

        # Add seed only if not disabled
        if not self.config.disable_seed:
            payload["seed"] = self.config.seed

        # Handle reasoning models (o1, o3, o4, gpt-5)
        model_lower = self.model.lower()
        if any(x in model_lower for x in ["o1", "o3", "o4", "gpt-5"]):
            payload.pop("stop", None)
            payload["temperature"] = 1
            # Use max_completion_tokens for reasoning models
            if "max_tokens" in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")

        return payload

    @staticmethod
    def parse_response(response_json: Dict) -> str:
        """Parse the response to extract generated text."""
        try:
            choices = response_json.get("choices", [])
            if not choices:
                return ""

            message = choices[0].get("message", {})
            content = message.get("content")

            # For reasoning models, try reasoning_content if content is empty
            if not content:
                content = message.get("reasoning_content", "")

            return content if content else ""
        except Exception as e:
            logger.warning(f"Could not parse response: {e}")
            return ""

    def generate(
        self,
        messages: List[Dict[str, str]],
        gen_kwargs: Optional[Dict] = None,
    ) -> str:
        """Synchronous generation with caching and retry logic."""
        # Check cache first
        if self.config.use_cache:
            cached = self.cache.get(self.model, messages, gen_kwargs)
            if cached is not None:
                self.cache_hits += 1
                return cached
            self.cache_misses += 1

        payload = self._create_payload(messages, gen_kwargs)

        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    json=payload,
                    headers=self._header,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                response_json = response.json()
                result = self.parse_response(response_json)

                # Retry if response is empty (potential transient error)
                if not result and attempt < self.config.max_retries - 1:
                    logger.warning(f"Empty response on attempt {attempt + 1}, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue

                # Cache successful response
                if self.config.use_cache and result:
                    self.cache.set(self.model, messages, result, gen_kwargs)

                return result
            except requests.exceptions.HTTPError as e:
                last_exception = e
                status_code = e.response.status_code if e.response else None

                # Special handling for rate limits (429) and server errors (5xx)
                if status_code == 429:
                    # Longer backoff for rate limits: 5, 15, 45, 135, 405 seconds
                    wait_time = 5 * (3 ** attempt)
                    logger.warning(
                        f"Rate limited (429) on attempt {attempt + 1}/{self.config.max_retries}. "
                        f"Waiting {wait_time}s before retry..."
                    )
                    time.sleep(wait_time)
                    continue
                elif status_code in (500, 502, 503, 504):
                    # Server errors: moderate backoff
                    wait_time = 2 * (2 ** attempt)
                    logger.warning(
                        f"Server error ({status_code}) on attempt {attempt + 1}/{self.config.max_retries}. "
                        f"Waiting {wait_time}s before retry..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.warning(f"API request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"API request failed after {self.config.max_retries} attempts: {e}")
                        raise
            except Exception as e:
                last_exception = e
                logger.warning(f"API request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    logger.error(f"API request failed after {self.config.max_retries} attempts: {e}")
                    raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        return ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=10),
        reraise=True,
    )
    async def agenerate(
        self,
        session: aiohttp.ClientSession,
        messages: List[Dict[str, str]],
        gen_kwargs: Optional[Dict] = None,
    ) -> tuple[str, Dict]:
        """Async generation with retry."""
        payload = self._create_payload(copy.deepcopy(gen_kwargs) if gen_kwargs else None)
        payload["messages"] = messages

        async with session.post(
            self.base_url,
            json=payload,
            headers=self._header,
        ) as response:
            if not response.ok:
                error_text = await response.text()
                logger.warning(f"API request failed: {response.status}, {error_text}")
            response.raise_for_status()
            response_json = await response.json()
            return self.parse_response(response_json), response_json

    async def batch_generate_async(
        self,
        messages_list: List[List[Dict[str, str]]],
        gen_kwargs: Optional[Dict] = None,
    ) -> List[tuple[str, Dict]]:
        """Batch async generation."""
        connector = aiohttp.TCPConnector(limit=self.config.num_concurrent)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        ) as session:
            tasks = [
                asyncio.create_task(self.agenerate(session, msgs, gen_kwargs))
                for msgs in messages_list
            ]
            results = await tqdm_asyncio.gather(*tasks, desc="Generating")
            return results

    def batch_generate(
        self,
        messages_list: List[List[Dict[str, str]]],
        gen_kwargs: Optional[Dict] = None,
        use_async: bool = True,
    ) -> List[str]:
        """Batch generation (sync wrapper for async)."""
        if use_async and len(messages_list) > 1:
            results = asyncio.run(
                self.batch_generate_async(messages_list, gen_kwargs)
            )
            return [r[0] for r in results]
        else:
            # Sync fallback
            results = []
            for messages in tqdm(messages_list, desc="Generating"):
                result = self.generate(messages, gen_kwargs)
                results.append(result)
            return results


def apply_chat_template(
    prompt: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Apply a simple chat template to convert prompt to messages format."""
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    return messages


class ResultsLogger:
    """
    Logger for saving results in lm-eval-harness format.
    """

    def __init__(
        self,
        output_path: str,
        model_name: str,
        benchmark_name: str,
    ):
        self.output_path = Path(output_path)
        self.model_name = model_name
        self.model_name_sanitized = self._sanitize_model_name(model_name)
        self.benchmark_name = benchmark_name
        self.start_time = datetime.now()
        self.samples: List[Dict] = []

        # Create output directory
        self.results_dir = self.output_path / self.model_name_sanitized
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_model_name(name: str) -> str:
        """Sanitize model name for filesystem."""
        return name.replace("/", "__").replace(":", "_")

    def add_sample(self, result: GenerationResult):
        """Add a sample result."""
        sample = {
            "doc_id": result.doc_id,
            "doc": result.doc,
            "prompt": result.prompt,
            "response": result.response,
            "raw_response": result.raw_response,
            "metadata": result.metadata,
        }
        self.samples.append(sample)

    def save_samples(self):
        """Save samples to JSONL file."""
        timestamp = self.start_time.isoformat().replace(":", "-")
        samples_file = self.results_dir / f"samples_{self.benchmark_name}_{timestamp}.jsonl"

        with open(samples_file, "w") as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")

        logger.info(f"Saved {len(self.samples)} samples to {samples_file}")
        return samples_file

    def save_results(self, metrics: Dict[str, Any]):
        """Save aggregated results to JSON file."""
        timestamp = self.start_time.isoformat().replace(":", "-")
        end_time = datetime.now()

        results = {
            "model_name": self.model_name,
            "benchmark": self.benchmark_name,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_evaluation_time_seconds": (end_time - self.start_time).total_seconds(),
            "num_samples": len(self.samples),
            "metrics": metrics,
        }

        results_file = self.results_dir / f"results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Saved results to {results_file}")
        return results_file


# Supported models list
SUPPORTED_MODELS = [
    "azure/gpt-4o",
    "azure/gpt-5",
    "azure/o3",
    "azure/o4-mini",
    "azure/gpt-oss-120b",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-pro",
    "gemini/gemini-3-pro-preview",
    "gemini/gemini-3-flash-preview",
    "neulab/claude-opus-4-5-20251101",
    "neulab/claude-sonnet-4-5-20250929",
    "neulab/claude-sonnet-4-20250514",
    "neulab/kimi-k2-0711-preview",
    "azure/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "neulab/qwen3-coder-480b-a35b-instruct",
]


def get_model_api(
    model: str,
    base_url: str = "https://cmu.litellm.ai/v1/chat/completions",
    **kwargs,
) -> ProxyModelAPI:
    """Factory function to create a model API instance."""
    config = ModelConfig(
        model=model,
        base_url=base_url,
        **kwargs,
    )
    return ProxyModelAPI(config)
