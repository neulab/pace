"""lm-evaluation-harness handler for ProxyBench."""

import os
import sys

_HANDLERS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVALUATIONS_DIR = os.path.dirname(_HANDLERS_DIR)
BENCHMARKS_DIR = os.path.join(_EVALUATIONS_DIR, "benchmarks")

LM_EVAL_HARNESS_DIR = os.path.join(BENCHMARKS_DIR, "lm-evaluation-harness")

LM_EVAL_TASK_PATHS = {
    "acp_gen": os.path.join(LM_EVAL_HARNESS_DIR, "lm_eval", "tasks", "acpbench"),
    "aime25": os.path.join(LM_EVAL_HARNESS_DIR, "lm_eval", "tasks", "aime"),
    "gpqa": os.path.join(LM_EVAL_HARNESS_DIR, "lm_eval", "tasks", "gpqa", "cot_zeroshot"),
    "humaneval_chat": os.path.join(LM_EVAL_HARNESS_DIR, "lm_eval", "tasks", "humaneval"),
    "ifeval": os.path.join(LM_EVAL_HARNESS_DIR, "lm_eval", "tasks", "ifeval"),
    "logiqa": os.path.join(LM_EVAL_HARNESS_DIR, "lm_eval", "tasks", "logiqa", "cot_zeroshot"),
    "mbpp_chat": os.path.join(LM_EVAL_HARNESS_DIR, "lm_eval", "tasks", "mbpp"),
    "mmlu_cot": os.path.join(LM_EVAL_HARNESS_DIR, "lm_eval", "tasks", "mmlu", "cot_generative"),
}

LM_EVAL_DEFAULT_SUBTASK = {
    "acp_gen": "acp_app_gen",
    "aime25": "aime25",
    "gpqa": "gpqa_diamond_cot_zeroshot",
    "humaneval_chat": "humaneval_chat",
    "ifeval": "ifeval",
    "logiqa": "logiqa_cot_zeroshot",
    "mbpp_chat": "mbpp_chat",
    "mmlu_cot": "mmlu_abstract_algebra_cot_generative",
}


def standardized_id_to_doc_id(benchmark: str, subtask: str | None, instance_id: str | int) -> int:
    """Convert a standardized_results id string to the raw doc_id integer.

    The standardized id format is defined by scripts/standardize/standardize_*.py.
    Most benchmarks use doc_id directly as the id; a few embed extra context:

      logiqa:    "{doc_id}_{filter}"  e.g. "0_strict-match" → 0
      mmlu_cot:  "{subject}_{doc_id}" e.g. "abstract_algebra_3" → 3

    For all other benchmarks the id is just str(doc_id), so int() suffices.
    """
    if isinstance(instance_id, int):
        return instance_id

    s = str(instance_id)

    if benchmark == "logiqa":
        return int(s.split("_")[0])

    if benchmark == "mmlu_cot":
        return int(s.split("_")[-1])

    return int(s)


def _run_lm_eval(
    model_name: str,
    base_url: str,
    api_key: str,
    benchmark: str,
    subtask: str,
    instance_id: str | int,
) -> list:
    """Run a single instance through lm-evaluation-harness."""
    if LM_EVAL_HARNESS_DIR not in sys.path:
        sys.path.insert(0, LM_EVAL_HARNESS_DIR)

    from lm_eval.tasks import TaskManager
    from lm_eval.evaluator import simple_evaluate

    task_dir = LM_EVAL_TASK_PATHS[benchmark]
    tm = TaskManager(include_path=task_dir, include_defaults=False)

    if subtask not in tm.all_tasks:
        raise ValueError(
            f"Subtask '{subtask}' not found in benchmark '{benchmark}'. "
            f"Available: {sorted(tm.all_tasks)}"
        )

    is_reasoning = any(kw in model_name.lower() for kw in ("thinking", "deepseek"))
    max_gen_toks = 32768 if is_reasoning else 16384

    disable_seed = "disable_seed=true"
    doc_id = standardized_id_to_doc_id(benchmark, subtask, instance_id)

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    chat_completions_url = base_url.rstrip("/") + "/v1/chat/completions"

    def _evaluate(gen_kwargs):
        return simple_evaluate(
            model="openai-chat-completions",
            model_args=(
                f"model={model_name},"
                f"base_url={chat_completions_url},"
                f"num_concurrent=1,"
                f"{disable_seed}"
            ),
            tasks=[subtask],
            samples={subtask: [doc_id]},
            log_samples=True,
            apply_chat_template=True,
            gen_kwargs=gen_kwargs,
            bootstrap_iters=0,
            task_manager=tm,
            confirm_run_unsafe_code=True,
        )

    # Some models reject `temperature` (e.g. reasoning models return a 400).
    # Try with temperature=0 for determinism; on a temperature error, drop it.
    try:
        results = _evaluate(f"max_gen_toks={max_gen_toks},temperature=0.0")
    except Exception as exc:
        if "temperature" not in str(exc).lower():
            raise
        results = _evaluate(f"max_gen_toks={max_gen_toks}")

    all_samples = results.get("samples", {}).get(subtask, [])
    if not all_samples:
        raise RuntimeError(
            f"No samples returned for subtask '{subtask}' instance {instance_id}"
        )

    matched = [s for s in all_samples if s.get("doc_id") == doc_id]
    return matched if matched else [all_samples[0]]
