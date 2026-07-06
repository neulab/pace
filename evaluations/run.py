"""
Unified single-instance evaluation runner for ProxyBench.

Usage:
    python run.py \\
        --model_name  azure_ai/gpt-5.2 \\
        --base_url    https://cmu.litellm.ai \\
        --api_key     sk-... \\
        --benchmark   lifbench \\
        --subtask     onedoc-qa \\
        --instance_id 0_0_3

Python API:
    from evaluations.run import run_instance
    result = run_instance(
        model_name="azure_ai/gpt-5.2",
        base_url="https://cmu.litellm.ai",
        api_key="sk-...",
        benchmark="lifbench",
        subtask="onedoc-qa",
        instance_id="0_0_3",
    )
"""

import argparse
import json
import os
import sys

from evaluations._env import load_env

from evaluations.handlers.lm_eval import (
    _run_lm_eval,
    LM_EVAL_DEFAULT_SUBTASK,
    standardized_id_to_doc_id,
)
from evaluations.handlers.beir import _run_beir
from evaluations.handlers.bfcl import _run_bfcl, BFCL_DEFAULT_SUBTASK
from evaluations.handlers.debugbench import _run_debugbench
from evaluations.handlers.infobench import _run_infobench
from evaluations.handlers.lifbench import _run_lifbench, LIFBENCH_DEFAULT_SUBTASK
from evaluations.handlers.livecodebench import _run_livecodebench, LCB_DEFAULT_SUBTASK
from evaluations.handlers.planbench import _run_planbench, PLANBENCH_DEFAULT_SUBTASK
from evaluations.handlers.repobench import _run_repobench, REPOBENCH_DEFAULT_SUBTASK
from evaluations.handlers.visualwebbench import _run_visualwebbench, VWB_DEFAULT_SUBTASK
from evaluations.handlers.mmmu import _run_mmmu
from evaluations.handlers.visualpuzzles import _run_visualpuzzles, VP_DEFAULT_SUBTASK

# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

BENCHMARK_HANDLERS = {
    "acp_gen":          "lm_eval",
    "aime25":           "lm_eval",
    "gpqa":             "lm_eval",
    "humaneval_chat":   "lm_eval",
    "ifeval":           "lm_eval",
    "logiqa":           "lm_eval",
    "mbpp_chat":        "lm_eval",
    "mmlu_cot":         "lm_eval",
    "beir_nfcorpus":    "beir",
    "bfcl":             "bfcl",
    "debugbench":       "debugbench",
    "infobench":        "infobench",
    "lifbench":         "lifbench",
    "livecodebench":    "livecodebench",
    "planbench":        "planbench",
    "repobench":        "repobench",
    "visualwebbench":   "visualwebbench",
    "mmmu":             "mmmu",
    "visualpuzzles":    "visualpuzzles",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_instance(
    model_name: str,
    base_url: str,
    api_key: str,
    benchmark: str,
    subtask: str | None = None,
    instance_id: str | int = 0,
) -> list:
    """Run a single benchmark instance and return the result as a list of dicts.

    Parameters
    ----------
    model_name:   Model identifier, e.g. "azure_ai/gpt-5.2"
    base_url:     API base URL, e.g. "https://cmu.litellm.ai"
    api_key:      API key for the endpoint
    benchmark:    Benchmark name, e.g. "lifbench"
    subtask:      Optional sub-task name. If omitted, the benchmark default is used.
    instance_id:  Instance identifier in standardized_results format.

    Returns
    -------
    List of dicts, one per filter, matching the raw results schema.
    """
    if benchmark not in BENCHMARK_HANDLERS:
        raise ValueError(
            f"Benchmark '{benchmark}' is not supported. "
            f"Supported: {list(BENCHMARK_HANDLERS.keys())}"
        )

    handler = BENCHMARK_HANDLERS[benchmark]

    if handler == "lm_eval":
        if subtask is None:
            subtask = LM_EVAL_DEFAULT_SUBTASK.get(benchmark)
            if subtask is None:
                raise ValueError(
                    f"No default subtask for benchmark '{benchmark}'. "
                    "Please provide one explicitly."
                )
        return _run_lm_eval(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            benchmark=benchmark,
            subtask=subtask,
            instance_id=instance_id,
        )

    if handler == "beir":
        return _run_beir(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            benchmark=benchmark,
            instance_id=str(instance_id),
        )

    if handler == "bfcl":
        if subtask is None:
            subtask = BFCL_DEFAULT_SUBTASK
        return _run_bfcl(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            subtask=subtask,
            instance_id=str(instance_id),
        )

    if handler == "debugbench":
        return _run_debugbench(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            instance_id=str(instance_id),
        )

    if handler == "infobench":
        return _run_infobench(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            instance_id=str(instance_id),
        )

    if handler == "lifbench":
        if subtask is None:
            subtask = LIFBENCH_DEFAULT_SUBTASK
        return _run_lifbench(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            subtask=subtask,
            instance_id=str(instance_id),
        )

    if handler == "livecodebench":
        if subtask is None:
            subtask = LCB_DEFAULT_SUBTASK
        return _run_livecodebench(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            subtask=subtask,
            instance_id=str(instance_id),
        )

    if handler == "planbench":
        if subtask is None:
            subtask = PLANBENCH_DEFAULT_SUBTASK
        return _run_planbench(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            subtask=subtask,
            instance_id=str(instance_id),
        )

    if handler == "repobench":
        if subtask is None:
            subtask = REPOBENCH_DEFAULT_SUBTASK
        return _run_repobench(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            subtask=subtask,
            instance_id=str(instance_id),
        )

    if handler == "visualwebbench":
        if subtask is None:
            subtask = VWB_DEFAULT_SUBTASK
        return _run_visualwebbench(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            subtask=subtask,
            instance_id=str(instance_id),
        )

    if handler == "mmmu":
        return _run_mmmu(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            instance_id=str(instance_id),
        )

    if handler == "visualpuzzles":
        if subtask is None:
            subtask = VP_DEFAULT_SUBTASK
        return _run_visualpuzzles(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            subtask=subtask,
            instance_id=str(instance_id),
        )

    raise NotImplementedError(f"Handler '{handler}' is not implemented.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    load_env()  # populate env from repo-root .env (real env vars still win)
    parser = argparse.ArgumentParser(
        description="Run a single benchmark instance and print the result as JSON."
    )
    parser.add_argument("--model_name", required=True,
                        help="Model identifier, e.g. azure_ai/gpt-5.2")
    parser.add_argument("--base_url", default=os.environ.get("BASE_URL", "https://cmu.litellm.ai"),
                        help="API base URL (default: $BASE_URL or https://cmu.litellm.ai)")
    parser.add_argument("--api_key", default=os.environ.get("API_KEY"),
                        help="API key for the endpoint (default: $API_KEY, e.g. from .env)")
    parser.add_argument("--benchmark", required=True,
                        help="Benchmark name, e.g. lifbench")
    parser.add_argument("--subtask", default=None,
                        help="Optional subtask name, e.g. onedoc-qa")
    parser.add_argument("--instance_id", type=str, default="0",
                        help="Instance id in standardized_results format (default: 0)")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("no API key: pass --api_key, export API_KEY, or set it in .env")

    results = run_instance(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        benchmark=args.benchmark,
        subtask=args.subtask,
        instance_id=args.instance_id,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
