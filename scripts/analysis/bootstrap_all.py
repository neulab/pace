#!/usr/bin/env python3
"""
Run bootstrap pipeline on merged source and target benchmarks.

This script merges all instances from multiple source benchmarks into a single
combined source dataset, and similarly merges all instances from multiple target
benchmarks into a single combined target dataset, then runs the bootstrap pipeline
once on the merged data.

Example usage:
    python bootstrap_all.py \
        --sources humaneval mbpp livecodebench \
        --targets swebench gaia \
        --train_size 200 --eval_size 200 \
        --boot_source_k 100 --boot_target_k 100

This will:
1. Merge all instances from humaneval + mbpp + livecodebench into one source
2. Merge all instances from swebench + gaia into one target
3. Run bootstrap once on merged_source -> merged_target
"""

import argparse
import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Import utilities from bootstrap.py
from bootstrap import (
    STD_BASE,
    BenchDataset,
    discover_benchmark,
    dicts_to_matrix,
    compute_mean_over_indices,
    two_sided_bootstrap_max_sample,
    vote_source_instances_over_target_bootstraps,
    pearson_corr,
    spearman_corr,
)


@dataclass
class MergedBenchDataset:
    """Represents a merged dataset from multiple benchmarks."""
    
    bench_names: List[str]
    label: str
    model_dicts: Dict[str, Dict[str, float]]  # model_name -> {instance_id -> score}
    
    def list_models(self) -> List[str]:
        return sorted(self.model_dicts.keys())
    
    def get_model_dict(self, model_name: str) -> Dict[str, float]:
        return self.model_dicts.get(model_name, {})


def load_benchmark_data(
    bench_name: str,
    base_dir: Path,
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Load all model data for a benchmark.
    
    Returns:
        (list of model names, dict of model_name -> {prefixed_instance_id -> score})
    """
    bench_full, _ = discover_benchmark(bench_name, base_dir=base_dir)
    models = bench_full.list_models()
    
    model_dicts: Dict[str, Dict[str, float]] = {}
    
    for model in models:
        kept, dicts = bench_full.load_model_outputs_for_models([model])
        if kept and dicts:
            # Prefix instance IDs with benchmark name for uniqueness
            prefixed_dict = {
                f"{bench_name}::{k}": v 
                for k, v in dicts[0].items()
            }
            model_dicts[model] = prefixed_dict
    
    return models, model_dicts


def merge_benchmarks(
    bench_names: List[str],
    base_dir: Path,
    label: str,
) -> MergedBenchDataset:
    """Merge multiple benchmarks into a single dataset.
    
    Instance IDs are prefixed with benchmark name to ensure uniqueness.
    Only models that exist in ALL benchmarks are kept.
    """
    print(f"\nMerging benchmarks for {label}: {bench_names}")
    
    all_model_sets: List[set] = []
    bench_data: Dict[str, Dict[str, Dict[str, float]]] = {}  # bench -> model -> dict
    
    for bench_name in bench_names:
        models, model_dicts = load_benchmark_data(bench_name, base_dir)
        all_model_sets.append(set(models))
        bench_data[bench_name] = model_dicts
        print(f"  - {bench_name}: {len(models)} models, "
              f"{sum(len(d) for d in model_dicts.values())} total instances")
    
    # Find common models across all benchmarks
    common_models = set.intersection(*all_model_sets) if all_model_sets else set()
    print(f"  Common models across all benchmarks: {len(common_models)}")
    
    # Merge data for common models
    merged_model_dicts: Dict[str, Dict[str, float]] = {}
    
    for model in sorted(common_models):
        merged_dict: Dict[str, float] = {}
        for bench_name in bench_names:
            if model in bench_data[bench_name]:
                merged_dict.update(bench_data[bench_name][model])
        merged_model_dicts[model] = merged_dict
    
    total_instances = sum(len(d) for d in merged_model_dicts.values()) // len(common_models) if common_models else 0
    print(f"  Merged dataset: {len(common_models)} models, ~{total_instances} instances per model")
    
    return MergedBenchDataset(
        bench_names=bench_names,
        label=label,
        model_dicts=merged_model_dicts,
    )


def run_pipeline_for_merged(
    src_merged: MergedBenchDataset,
    tgt_merged: MergedBenchDataset,
    omit_models: Optional[List[str]] = None,
    plot_dir: Path = Path("plots"),
    title_prefix: str = "",
    k_source: Optional[int] = None,
    target_train_size: int = 400,
    target_eval_size: int = 100,
    n_outer: int = 10,
    n_restarts_inner: int = 10,
    boot_source_k: int = 200,
    boot_target_k: int = 400,
    n_boot: int = 50,
    boot_seed: int = 0,
    split_seed: int = 0,
    swap_passes: int = 10,
    swap_sample_in: int = 300,
    candidate_cap: Optional[int] = None,
    annotate: bool = True,
) -> Dict:
    """Run the bootstrap pipeline on merged source and target datasets."""
    
    omit_models = set(omit_models or [])
    tag = title_prefix or f"{'_'.join(src_merged.bench_names)}_to_{'_'.join(tgt_merged.bench_names)}"
    
    # 1) Find common models
    src_models = set(src_merged.list_models()) - omit_models
    tgt_models = set(tgt_merged.list_models()) - omit_models
    models = sorted(src_models.intersection(tgt_models))
    
    if not models:
        raise RuntimeError(f"No overlapping models between merged source and target")
    
    print(f"\nRunning pipeline with {len(models)} models")
    
    # 2) Build model output dicts in same order
    src_dicts = [src_merged.get_model_dict(m) for m in models]
    tgt_dicts = [tgt_merged.get_model_dict(m) for m in models]
    
    # 3) Build matrices
    A_src, src_ids = dicts_to_matrix(src_dicts, fill_value=0.0)
    A_tgt, tgt_ids = dicts_to_matrix(tgt_dicts, fill_value=0.0)
    
    print(f"Source matrix: {A_src.shape} (models x instances)")
    print(f"Target matrix: {A_tgt.shape} (models x instances)")
    
    # 4) Split target into train/eval
    N_tgt = A_tgt.shape[1]
    if target_train_size + target_eval_size > N_tgt:
        target_train_size = min(target_train_size, N_tgt // 2)
        target_eval_size = min(target_eval_size, N_tgt - target_train_size)
        print(f"[WARN] Adjusted target split to train={target_train_size}, eval={target_eval_size} (N={N_tgt})")
    
    rng_split = np.random.default_rng(split_seed)
    perm = rng_split.permutation(N_tgt)
    tgt_train_cols = perm[:target_train_size].tolist()
    tgt_eval_cols = perm[target_train_size:target_train_size + target_eval_size].tolist()
    
    y_tgt_eval = compute_mean_over_indices(A_tgt, tgt_eval_cols) if target_eval_size > 0 else compute_mean_over_indices(A_tgt, tgt_train_cols)
    y_tgt_train = compute_mean_over_indices(A_tgt, tgt_train_cols)
    
    # 5) Two-sided bootstrap max corr sample
    x_boot_max, y_boot_max, boot_max_corr = two_sided_bootstrap_max_sample(
        A_src, A_tgt, tgt_train_cols,
        k_src=boot_source_k, k_tgt=boot_target_k,
        n_boot=n_boot, seed=boot_seed,
    )
    
    # 6) Voting-based subset selection
    if k_source is None:
        k_source = boot_source_k
    
    S_vote, vote_counts, eval_corrs, boot_corrs = vote_source_instances_over_target_bootstraps(
        A_src,
        A_tgt,
        tgt_train_cols,
        y_tgt_eval,
        k_src=k_source,
        tgt_boot_k=boot_target_k,
        n_outer=n_outer,
        n_restarts_inner=n_restarts_inner,
        seed=boot_seed,
        swap_passes=swap_passes,
        swap_sample_in=swap_sample_in,
        candidate_cap=candidate_cap,
    )
    
    # 7) Evaluate final subset
    x_vote = compute_mean_over_indices(A_src, S_vote)
    corr_train_p = pearson_corr(x_vote, y_tgt_train)
    corr_train_s = spearman_corr(x_vote, y_tgt_train)
    corr_eval_p = pearson_corr(x_vote, y_tgt_eval)
    corr_eval_s = spearman_corr(x_vote, y_tgt_eval)
    
    print(f"\nFinal correlations:")
    print(f"  Train: Pearson={corr_train_p:.4f}, Spearman={corr_train_s:.4f}")
    print(f"  Eval:  Pearson={corr_eval_p:.4f}, Spearman={corr_eval_s:.4f}")
    
    # 8) Create output directory and save plots
    out_dir = plot_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Scatter plot: voted subset vs eval
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_vote, y_tgt_eval, alpha=0.7)
    if annotate:
        for i, m in enumerate(models):
            ax.annotate(m, (x_vote[i], y_tgt_eval[i]), fontsize=6, alpha=0.7)
    ax.set_xlabel(f"Source (merged): {', '.join(src_merged.bench_names)}")
    ax.set_ylabel(f"Target (merged): {', '.join(tgt_merged.bench_names)}")
    ax.set_title(f"{tag}\nEval Pearson={corr_eval_p:.4f}, Spearman={corr_eval_s:.4f}")
    
    # Add regression line
    if len(x_vote) > 1:
        z = np.polyfit(x_vote, y_tgt_eval, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x_vote), max(x_vote), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_eval_scatter.png", dpi=150)
    plt.close(fig)
    
    # Correlation per iteration plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(eval_corrs) + 1), eval_corrs, 'b-o', label='Eval corr')
    ax.plot(range(1, len(boot_corrs) + 1), boot_corrs, 'g--s', label='Boot corr')
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Pearson correlation")
    ax.set_title(f"{tag}: Correlation per iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_corr_per_iter.png", dpi=150)
    plt.close(fig)
    
    # Get selected instance IDs
    top_ids = [src_ids[j] for j in S_vote]
    
    # Group selected instances by benchmark
    instances_by_bench: Dict[str, List[str]] = {}
    for inst_id in top_ids:
        bench = inst_id.split("::")[0]
        if bench not in instances_by_bench:
            instances_by_bench[bench] = []
        instances_by_bench[bench].append(inst_id)
    
    # Save JSON results
    out_json = out_dir / f"{tag}_results.json"
    results_dict = {
        "title": tag,
        "source_benchmarks": src_merged.bench_names,
        "target_benchmarks": tgt_merged.bench_names,
        "models_used": models,
        "params": {
            "k_source": k_source,
            "target_train_size": target_train_size,
            "target_eval_size": target_eval_size,
            "n_outer": n_outer,
            "n_restarts_inner": n_restarts_inner,
            "boot_source_k": boot_source_k,
            "boot_target_k": boot_target_k,
            "n_boot": n_boot,
            "boot_seed": boot_seed,
            "split_seed": split_seed,
        },
        "source_matrix_shape": list(A_src.shape),
        "target_matrix_shape": list(A_tgt.shape),
        "corr_train": {"pearson": float(corr_train_p), "spearman": float(corr_train_s)},
        "corr_eval": {"pearson": float(corr_eval_p), "spearman": float(corr_eval_s)},
        "boot_max_corr": float(boot_max_corr),
        "eval_corrs_per_iter": [float(x) for x in eval_corrs],
        "boot_corrs_per_iter": [float(x) for x in boot_corrs],
        "selected_source_instances": {
            "total": len(top_ids),
            "by_benchmark": {k: len(v) for k, v in instances_by_bench.items()},
            "ids": top_ids,
        },
    }
    
    with open(out_json, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nSaved results to: {out_json}")
    print(f"Selected {len(top_ids)} source instances:")
    for bench, ids in sorted(instances_by_bench.items()):
        print(f"  - {bench}: {len(ids)} instances")
    
    return results_dict


def main():
    parser = argparse.ArgumentParser(
        description="Run bootstrap pipeline on merged source and target benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge multiple source benchmarks and multiple target benchmarks
  python bootstrap_all.py \\
      --sources humaneval mbpp livecodebench \\
      --targets swebench gaia \\
      --train_size 200 --eval_size 200 \\
      --boot_source_k 100 --boot_target_k 100

  # Find proxy instances from code benchmarks for SWE-bench
  python bootstrap_all.py \\
      --sources humaneval mbpp humaneval_chat mbpp_chat \\
      --targets swebench \\
      --train_size 150 --eval_size 150 \\
      --boot_source_k 100 --boot_target_k 100 \\
      --k_source 200

  # Find proxy instances from reasoning benchmarks for multiple targets
  python bootstrap_all.py \\
      --sources gpqa logiqa mmlu \\
      --targets swebench gaia \\
      --train_size 200 --eval_size 200
        """
    )
    
    # Benchmark selection
    parser.add_argument("--sources", nargs="+", required=True,
                        help="List of source benchmark names to merge")
    parser.add_argument("--targets", nargs="+", required=True,
                        help="List of target benchmark names to merge")
    parser.add_argument("--base_dir", default=str(STD_BASE),
                        help="Base directory for standardized_results")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="../../analysis/merged",
                        help="Output directory for results")
    
    # Size parameters
    parser.add_argument("--train_size", type=int, default=200,
                        help="Target train size (default: 200)")
    parser.add_argument("--eval_size", type=int, default=200,
                        help="Target eval size (default: 200)")
    parser.add_argument("--boot_source_k", type=int, default=100,
                        help="Source bootstrap sample size (default: 100)")
    parser.add_argument("--boot_target_k", type=int, default=100,
                        help="Target bootstrap sample size (default: 100)")
    parser.add_argument("--k_source", type=int, default=None,
                        help="Final source subset size (default: boot_source_k)")
    
    # Algorithm parameters
    parser.add_argument("--n_outer", type=int, default=10,
                        help="Outer iterations (default: 10)")
    parser.add_argument("--n_restarts", type=int, default=10,
                        help="Greedy+swap restarts per outer iteration (default: 10)")
    parser.add_argument("--n_bootstraps", type=int, default=50,
                        help="Number of bootstrap samples (default: 50)")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--boot_seed", type=int, default=0)
    parser.add_argument("--swap_passes", type=int, default=10)
    parser.add_argument("--swap_sample_in", type=int, default=300)
    parser.add_argument("--candidate_cap", type=int, default=None)
    
    # Misc
    parser.add_argument("--omit_models", type=str, default="",
                        help="Comma-separated list of model names to omit")
    parser.add_argument("--annotate", action="store_true",
                        help="Annotate points in scatter plots")
    
    args = parser.parse_args()
    
    omit_models = [x.strip() for x in args.omit_models.split(",") if x.strip()]
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    
    print("="*70)
    print("BOOTSTRAP ON MERGED BENCHMARKS")
    print("="*70)
    print(f"Sources: {args.sources}")
    print(f"Targets: {args.targets}")
    
    # Merge source benchmarks
    src_merged = merge_benchmarks(
        args.sources, base_dir, 
        label="source_" + "_".join(args.sources)
    )
    
    # Merge target benchmarks
    tgt_merged = merge_benchmarks(
        args.targets, base_dir,
        label="target_" + "_".join(args.targets)
    )
    
    # Create tag for output
    src_tag = "+".join(args.sources)
    tgt_tag = "+".join(args.targets)
    tag = f"{src_tag}_TO_{tgt_tag}"
    
    # Run pipeline
    try:
        results = run_pipeline_for_merged(
            src_merged,
            tgt_merged,
            omit_models=omit_models,
            plot_dir=output_dir,
            title_prefix=tag,
            k_source=args.k_source,
            target_train_size=args.train_size,
            target_eval_size=args.eval_size,
            n_outer=args.n_outer,
            n_restarts_inner=args.n_restarts,
            boot_source_k=args.boot_source_k,
            boot_target_k=args.boot_target_k,
            n_boot=args.n_bootstraps,
            boot_seed=args.boot_seed,
            split_seed=args.split_seed,
            swap_passes=args.swap_passes,
            swap_sample_in=args.swap_sample_in,
            candidate_cap=args.candidate_cap,
            annotate=args.annotate,
        )
        
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"Source benchmarks: {args.sources}")
        print(f"Target benchmarks: {args.targets}")
        print(f"Models used: {len(results['models_used'])}")
        print(f"Source instances: {results['source_matrix_shape'][1]}")
        print(f"Target instances: {results['target_matrix_shape'][1]}")
        print(f"\nCorrelations:")
        print(f"  Train: Pearson={results['corr_train']['pearson']:.4f}, "
              f"Spearman={results['corr_train']['spearman']:.4f}")
        print(f"  Eval:  Pearson={results['corr_eval']['pearson']:.4f}, "
              f"Spearman={results['corr_eval']['spearman']:.4f}")
        print(f"\nSelected {results['selected_source_instances']['total']} proxy instances:")
        for bench, count in results['selected_source_instances']['by_benchmark'].items():
            print(f"  - {bench}: {count}")
        
    except Exception as e:
        traceback.print_exc()
        print(f"\nERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    main()
