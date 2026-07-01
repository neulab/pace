# Proxy Bench — lm-evaluation-harness Usage

How we use this (patched) fork of `lm-evaluation-harness` to evaluate a fixed set of models across a fixed set of benchmarks for the Proxy Bench project.

All models are accessed over an OpenAI-compatible chat completions API. You can point `base_url` at any OpenAI-compatible endpoint — e.g. a local proxy (LiteLLM, vLLM, SGLang) or a hosted inference provider — as long as it accepts the standard `/v1/chat/completions` interface.

## Models (11)

- `Kimi-K2.5`
- `Kimi-K2-Thinking`
- `DeepSeek-V3.2`
- `gpt-5.2`
- `gpt-5.2-codex`
- `MiniMax-M2.5`
- `MiniMax-M2.1`
- `GLM-4.7`
- `GLM-5`
- `claude-opus-4-6`
- `nvidia-nemotron-3-nano-30b-a3b-bf16`

The exact string you pass as `model=...` in `--model_args` is whatever identifier your OpenAI-compatible endpoint expects — typically `<namespace>/<model-name>` if you are using a multi-provider proxy.

## Benchmarks (8)

```
ifeval
acp_gen_2shot
mbpp_chat
humaneval_chat
mmlu_cot
gpqa_diamond_cot_zeroshot,gpqa_main_cot_zeroshot,gpqa_extended_cot_zeroshot
aime25
logiqa_cot_zeroshot
```

## Environment

- Install this fork (`pip install -e .`) — it contains patches required for running through chat-completions proxies and reasoning models (see "Local patches" below).
- `export OPENAI_API_KEY=<your-endpoint-key>`
- `export HF_ALLOW_CODE_EVAL=1` — required for `mbpp_chat` / `humaneval_chat`, which execute generated code.

## Example invocation

```bash
python -m lm_eval run \
  --model openai-chat-completions \
  --model_args "model=<model-id>,base_url=https://<your-endpoint>/v1/chat/completions,num_concurrent=20" \
  --tasks ifeval \
  --output_path ./results/ifeval \
  --apply_chat_template \
  --confirm_run_unsafe_code \
  --log_samples \
  --gen_kwargs max_gen_toks=16384 \
  --use_cache ./.cache/<safe-model-name>
```

Fixed flags on every run:
- `--model openai-chat-completions` — all models speak the OpenAI chat API.
- `--apply_chat_template` — apply the task's chat template (required for `*_chat` tasks and for correct instruction formatting).
- `--confirm_run_unsafe_code` — acknowledges `mbpp_chat` / `humaneval_chat` execute model-generated code.
- `--log_samples` — write per-sample outputs for later analysis.
- `--use_cache <dir>` — SQLite response cache, one directory per model (we replace `/` in the model name with `__` to make a safe path). Lets re-runs skip already-completed prompts.
- `num_concurrent=20` in `model_args` — 20 in-flight requests per run.

## Per-model / per-benchmark differences

Only two things change across runs:

**1. `disable_seed` in `model_args`** — some backends reject the `seed` parameter and will error out if it's sent. If your endpoint accepts `seed`, leave it out; otherwise add `disable_seed=true`.

```
# seed supported
model=<...>,base_url=<...>,num_concurrent=20

# seed not supported — add disable_seed
model=<...>,base_url=<...>,disable_seed=true,num_concurrent=20
```

**2. `gen_kwargs max_gen_toks` — larger budget for reasoning/thinking models.**

```
max_gen_toks=32768   # model name contains "Thinking" or "DeepSeek" (case-insensitive)
max_gen_toks=16384   # everything else
```

Benchmarks themselves are not varied per model — the same 7 tasks run for every model; only `--tasks` and `--output_path` change across benchmarks.

If a given model is served on a different endpoint than the rest, point `base_url` at that endpoint and set `OPENAI_API_KEY` to that endpoint's key for the run.

## Local patches in this fork

This fork contains fixes that matter when running through OpenAI-compatible proxies and for reasoning models:

- **Async + sync response cache**: don't cache empty responses (`lm_eval/models/api_models.py`, `lm_eval/api/model.py`).
- **`asyncio.gather` poison**: one failed request no longer wipes the whole batch — wrapped in a per-request `safe_call` (`api_models.py`).
- **Stop sequences**: whitespace-only stops filtered out (some providers reject them) (`lm_eval/models/openai_completions.py`).
- **Reasoning-model detection**: narrowed to `gpt-5` / `o1` / `o3` / `o4` so it doesn't falsely match names like `K2.5`, `GLM-5`, `M2.5` (`openai_completions.py`).
- **Nemotron handling**: `temperature=1.0`, `repetition_penalty=1.1`, strip `</think>` from outputs (`openai_completions.py`).
- **GPQA `\boxed{}` extraction** for reasoning-model answers (`lm_eval/filters/extraction.py`).

## Running multiple models / benchmarks

The simplest pattern is two nested loops (models × benchmarks) in a shell script. Within one model, run benchmarks sequentially so they share the same cache file cleanly; across models, you can run in parallel (background each model's inner loop and `wait`). Each model should get its own `--use_cache` directory.
