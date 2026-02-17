import importlib
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np

try:
    from .bootstrap import (
        dicts_to_matrix,
        run_bootstrap_pipeline,
    )
except ImportError:
    # Allow running as a script: python proxy-bench/analysis/run.py
    from bootstrap import (
        dicts_to_matrix,
        run_bootstrap_pipeline,
    )

# Utilities registry
_UTILS_REGISTRY = {
    "swebench": ".utils_swebench",
    "mmlu": ".utils_mmlu",
    "lcb": ".utils_lcb",
}


def _load_utils(bench: str):
    if bench not in _UTILS_REGISTRY:
        raise ValueError(f"Unknown benchmark: {bench}. Available: {list(_UTILS_REGISTRY)}")
    mod = _UTILS_REGISTRY[bench]
    if __package__:
        return importlib.import_module(mod, package=__package__)
    # running as script: import from current dir
    return importlib.import_module(mod.lstrip("."))


def _intersect_models(a: List[str], b: List[str]) -> List[str]:
    aset = set(a)
    inter = [m for m in a if m in aset and m in b]
    # ensure stable deterministic order: alphabetical over intersection
    return sorted(set(inter))


def run_pair(
    source_bench: str,
    target_bench: str,
    source_kwargs: Dict = None,
    target_kwargs: Dict = None,
    title_prefix: str = "",
    plot_dir: str = "plots",
    annotate: bool = True,
    # bootstrap config
    k_source: int = None,
    target_train_size: int = None,
    target_eval_size: int = None,
    n_outer: int = None,
    n_restarts_inner: int = None,
    boot_source_k: int = None,
    boot_target_k: int = None,
    boot_seed: int = None,
    split_seed: int = None,
):
    source_kwargs = source_kwargs or {}
    target_kwargs = target_kwargs or {}

    src_utils = _load_utils(source_bench)
    tgt_utils = _load_utils(target_bench)

    src_models = src_utils.list_canonical_models(**{k: v for k, v in source_kwargs.items() if k in ("mmlu_dir", "lcb_dir", "swe_csv_dir")})
    tgt_models = tgt_utils.list_canonical_models(**{k: v for k, v in target_kwargs.items() if k in ("mmlu_dir", "lcb_dir", "swe_csv_dir")})

    models = _intersect_models(src_models, tgt_models)
    if not models:
        raise RuntimeError("No overlapping canonical model names between the two benchmarks")

    # Load outputs in the same (intersection) order
    kept_src, src_dicts = src_utils.load_model_outputs_for_models(models, **source_kwargs)
    kept_tgt, tgt_dicts = tgt_utils.load_model_outputs_for_models(models, **target_kwargs)

    # ensure both returned the same model ordering
    if kept_src != kept_tgt:
        # reconcile to the common intersection order
        model_set = set(kept_src).intersection(set(kept_tgt))
        models = [m for m in models if m in model_set]
        kept_src, src_dicts = src_utils.load_model_outputs_for_models(models, **source_kwargs)
        kept_tgt, tgt_dicts = tgt_utils.load_model_outputs_for_models(models, **target_kwargs)

    model_names = kept_src

    # Build matrices
    A_src, src_ids = dicts_to_matrix(src_dicts, fill_value=0.0)
    A_tgt, tgt_ids = dicts_to_matrix(tgt_dicts, fill_value=0.0)

    # Run generic pipeline
    tag = title_prefix or f"{source_bench}_vs_{target_bench}"
    kwargs = dict(plot_dir=plot_dir, title_prefix=tag, annotate=annotate)
    if k_source is not None: kwargs["k_source"] = k_source
    if target_train_size is not None: kwargs["target_train_size"] = target_train_size
    if target_eval_size is not None: kwargs["target_eval_size"] = target_eval_size
    if n_outer is not None: kwargs["n_outer"] = n_outer
    if n_restarts_inner is not None: kwargs["n_restarts_inner"] = n_restarts_inner
    if boot_source_k is not None: kwargs["boot_source_k"] = boot_source_k
    if boot_target_k is not None: kwargs["boot_target_k"] = boot_target_k
    if boot_seed is not None: kwargs["boot_seed"] = boot_seed
    if split_seed is not None: kwargs["split_seed"] = split_seed

    return run_bootstrap_pipeline(
        model_names,
        A_src,
        A_tgt,
        src_ids,
        tgt_ids,
        **kwargs,
    )



def run_mmlu_vs_swebench(
    mmlu_dir: str = "/workspace/project/proxy-bench/data/eval_scores",
    mmlu_task: str = "mmlu_electrical_engineering",
    swe_csv_dir: str = "swebench",
    title_prefix: str = "mmlu_vs_swebench",
    plot_dir: str = "plots",
    annotate: bool = True,
    **kwargs,
):
    return run_pair(
        source_bench="mmlu",
        target_bench="swebench",
        source_kwargs={"mmlu_dir": mmlu_dir, "mmlu_task": mmlu_task},
        target_kwargs={"swe_csv_dir": swe_csv_dir},
        title_prefix=title_prefix,
        plot_dir=plot_dir,
        annotate=annotate,
        **kwargs,
    )


def run_lcb_vs_swebench(
    lcb_task: str,
    lcb_dir: str,
    swe_csv_dir: str = "swebench",
    title_prefix: str = None,
    plot_dir: str = "plots",
    annotate: bool = True,
    **kwargs,
):
    if title_prefix is None:
        title_prefix = f"lcb_vs_swebench_{lcb_task}"
    return run_pair(
        source_bench="lcb",
        target_bench="swebench",
        source_kwargs={"lcb_dir": lcb_dir, "task": lcb_task},
        target_kwargs={"swe_csv_dir": swe_csv_dir},
        title_prefix=title_prefix,
        plot_dir=plot_dir,
        annotate=annotate,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run bootstrapping between source and target benchmarks.")
    parser.add_argument("--source", choices=["mmlu", "lcb"], required=True, help="Source benchmark to optimize subset from")
    parser.add_argument("--target", choices=["swebench"], default="swebench", help="Target benchmark for correlation")
    parser.add_argument("--title_prefix", default="", help="Prefix for output filenames and titles")
    parser.add_argument("--plot_dir", default="plots", help="Directory to save plots and JSON")
    parser.add_argument("--no_annotate", action="store_true", help="Disable point annotations in scatter plots")

    # MMLU options
    parser.add_argument("--mmlu_dir", default="/home/yueqis/proxybench/mmlu/", help="Base directory containing MMLU task folders")
    parser.add_argument("--mmlu_task", required=False, choices=['mmlu_high_school_geography', 'mmlu_college_physics', 'mmlu_moral_disputes', 'mmlu_professional_medicine', 'mmlu_college_computer_science', 'mmlu_nutrition', 'mmlu_marketing', 'mmlu_college_medicine', 'mmlu_formal_logic', 'mmlu_business_ethics', 'mmlu_high_school_chemistry', 'mmlu_us_foreign_policy', 'mmlu_college_mathematics', 'mmlu_moral_scenarios', 'mmlu_computer_security', 'mmlu_world_religions', 'mmlu_high_school_microeconomics', 'mmlu_miscellaneous', 'mmlu_clinical_knowledge', 'mmlu_management', 'mmlu_logical_fallacies', 'mmlu_high_school_computer_science', 'mmlu_high_school_psychology', 'mmlu_conceptual_physics', 'mmlu_professional_accounting', 'mmlu_international_law', 'mmlu_human_sexuality', 'mmlu_econometrics', 'mmlu_high_school_european_history', 'mmlu_machine_learning', 'mmlu_high_school_physics', 'mmlu_professional_psychology', 'mmlu_astronomy', 'mmlu_elementary_mathematics', 'mmlu_high_school_biology', 'mmlu_high_school_world_history', 'mmlu_anatomy', 'mmlu_high_school_macroeconomics', 'mmlu_virology', 'mmlu_security_studies', 'mmlu_high_school_mathematics', 'mmlu_prehistory', 'mmlu_professional_law', 'mmlu_human_aging', 'mmlu_high_school_government_and_politics', 'mmlu_high_school_statistics', 'mmlu_public_relations', 'mmlu_medical_genetics', 'mmlu_electrical_engineering', 'mmlu_sociology', 'mmlu_global_facts', 'mmlu_philosophy', 'mmlu_high_school_us_history', 'mmlu_jurisprudence', 'mmlu_college_chemistry', 'mmlu_abstract_algebra', 'mmlu_college_biology'], help="MMLU task folder name (e.g., mmlu_anatomy)")

    # LCB options
    parser.add_argument("--lcb_dir", default="/home/yueqis/proxybench/LiveCodeBench/output/", help="Base directory containing LCB outputs")
    parser.add_argument("--lcb_task", choices=["codegeneration", "codeexecution", "selfrepair", "testoutputprediction"], help="LCB task (required if source=lcb)")

    # SWE options
    parser.add_argument("--swe_csv_dir", default="/home/yueqis/proxybench/swebench", help="Directory containing SWE CSVs")

    # Bootstrapping hyperparameters (optional)
    parser.add_argument("--k_source", type=int)
    parser.add_argument("--target_train_size", type=int)
    parser.add_argument("--target_eval_size", type=int)
    parser.add_argument("--n_outer", type=int)
    parser.add_argument("--n_restarts_inner", type=int)
    parser.add_argument("--boot_source_k", type=int)
    parser.add_argument("--boot_target_k", type=int)
    parser.add_argument("--boot_seed", type=int)
    parser.add_argument("--split_seed", type=int)

    args = parser.parse_args()
    annotate = not args.no_annotate

    if args.source == "mmlu":
        res = run_mmlu_vs_swebench(
            mmlu_dir=args.mmlu_dir,
            mmlu_task=args.mmlu_task,
            swe_csv_dir=args.swe_csv_dir,
            title_prefix=args.title_prefix or "mmlu_vs_swebench",
            plot_dir=f"/home/yueqis/proxybench/analysis/plots/mmlu/{args.mmlu_task}/",
            annotate=annotate,
            k_source=args.k_source,
            target_train_size=args.target_train_size,
            target_eval_size=args.target_eval_size,
            n_outer=args.n_outer,
            n_restarts_inner=args.n_restarts_inner,
            boot_source_k=args.boot_source_k,
            boot_target_k=args.boot_target_k,
            boot_seed=args.boot_seed,
            split_seed=args.split_seed,
        )
    else:
        if not args.lcb_task:
            parser.error("--lcb_task is required when --source lcb")
        res = run_lcb_vs_swebench(
            lcb_task=args.lcb_task,
            lcb_dir=args.lcb_dir,
            swe_csv_dir=args.swe_csv_dir,
            title_prefix=args.title_prefix or f"lcb_vs_swebench_{args.lcb_task}",
            plot_dir=args.plot_dir,
            annotate=annotate,
            k_source=args.k_source,
            target_train_size=args.target_train_size,
            target_eval_size=args.target_eval_size,
            n_outer=args.n_outer,
            n_restarts_inner=args.n_restarts_inner,
            boot_source_k=args.boot_source_k,
            boot_target_k=args.boot_target_k,
            boot_seed=args.boot_seed,
            split_seed=args.split_seed,
        )

    # print a concise JSON summary to stdout
    def _to(obj):
        try:
            import numpy as _np
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except Exception:
            pass
        return obj

    print(json.dumps({k: _to(v) for k, v in res.items()}, indent=2))


# Example usage:
# result = run_pair(
#     source_bench="mmlu",
#     target_bench="swebench",
#     source_kwargs={"mmlu_dir": "proxy-bench/data/eval_scores/mmlu_electrical_engineering"},
#     target_kwargs={"swe_csv_dir": "swebench"},
#     title_prefix="mmlu_ee vs swebench",
# )
