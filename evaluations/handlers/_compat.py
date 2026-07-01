"""Shared OpenAI-compatible helpers for benchmark handlers.

`chat_completion` wraps `client.chat.completions.create` and transparently
handles two things that break when swapping in newer models:

1. temperature — some models (reasoning models such as gpt-5 / claude-opus-4-8)
   reject `temperature` with a 400 ("temperature is deprecated ..."). The
   handlers hard-code `temperature=0`; on that error we drop it and retry once.

2. token budget — reasoning models spend the `max_tokens` budget on hidden
   reasoning, so a small budget can return an empty, length-truncated message
   (finish_reason="length", content=""). When that happens we retry once with a
   much larger budget so the actual answer fits.

Both fallbacks only fire on the failure signature, so non-reasoning models keep
their original single call with no extra cost.
"""

# Floor for the enlarged retry budget when a reasoning model truncates.
_REASONING_BUDGET_FLOOR = 16384
_BUDGET_KEYS = ("max_completion_tokens", "max_tokens")


def _create_with_temperature_fallback(client, **kwargs):
    try:
        from openai import BadRequestError
    except Exception:  # pragma: no cover - openai always present at call time
        BadRequestError = Exception

    try:
        return client.chat.completions.create(**kwargs)
    except BadRequestError as exc:
        if "temperature" in kwargs and "temperature" in str(exc).lower():
            kwargs.pop("temperature", None)
            return client.chat.completions.create(**kwargs)
        raise


def _truncated_empty(response) -> bool:
    """True if the response was cut off for length with no usable content."""
    try:
        choice = response.choices[0]
        content = (getattr(choice.message, "content", None) or "").strip()
        return getattr(choice, "finish_reason", None) == "length" and not content
    except Exception:
        return False


def chat_completion(client, *, reasoning_budget_retry=True, **kwargs):
    """`client.chat.completions.create(**kwargs)` with model-portability fallbacks.

    - Drops `temperature` and retries if the model rejects it.
    - If the reply is empty because the token budget was exhausted by reasoning,
      retries once with an enlarged max_tokens / max_completion_tokens budget.
    """
    response = _create_with_temperature_fallback(client, **kwargs)

    if reasoning_budget_retry and _truncated_empty(response):
        budget_keys = [k for k in _BUDGET_KEYS if k in kwargs]
        if budget_keys:
            bumped = dict(kwargs)
            for k in budget_keys:
                bumped[k] = max((kwargs[k] or 0) * 8, _REASONING_BUDGET_FLOOR)
            response = _create_with_temperature_fallback(client, **bumped)

    return response
