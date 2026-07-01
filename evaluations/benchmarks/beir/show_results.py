"""
Aggregate all beir_results/*/metrics.json and print a comparison table.
Usage: python show_results.py [--dataset nfcorpus]
"""
import argparse
import json
import os
import pathlib

def short_name(engine: str) -> str:
    """Shorten long model paths for display."""
    engine = engine.replace("fireworks_ai/accounts/fireworks/models/", "fw/")
    engine = engine.replace("anthropic/", "")
    engine = engine.replace("openai/", "")
    engine = engine.replace("gemini/", "")
    engine = engine.replace("neulab/", "nl/")
    return engine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--metric", default="NDCG@10",
                        help="Primary sort metric (default: NDCG@10)")
    args = parser.parse_args()

    base_dir = pathlib.Path(__file__).parent
    results_base = base_dir / "beir_results" / args.dataset

    rows = []
    for model_dir in sorted(results_base.iterdir()):
        if not model_dir.is_dir():
            continue
        metrics_file = model_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file) as f:
            m = json.load(f)
        rows.append({
            "model":     short_name(m.get("engine", model_dir.name)),
            "NDCG@5":    m["ndcg"].get("NDCG@5",  0),
            "NDCG@10":   m["ndcg"].get("NDCG@10", 0),
            "NDCG@20":   m["ndcg"].get("NDCG@20", 0),
            "MAP@10":    m["map"].get("MAP@10",   0),
            "Recall@20": m["recall"].get("Recall@20", 0),
        })

    if not rows:
        print(f"No results found in {results_base}")
        return

    # Sort by primary metric descending
    rows.sort(key=lambda r: r.get(args.metric, 0), reverse=True)

    col_w = max(len(r["model"]) for r in rows) + 2
    header = f"{'#':>3}  {'Model':<{col_w}} {'NDCG@5':>8} {'NDCG@10':>8} {'NDCG@20':>8} {'MAP@10':>8} {'R@20':>8}"
    sep    = "-" * len(header)

    print(f"\nBEIR — {args.dataset}  (sorted by {args.metric})")
    print(sep)
    print(header)
    print(sep)
    for rank, r in enumerate(rows, 1):
        print(f"{rank:>3}.  {r['model']:<{col_w}} "
              f"{r['NDCG@5']:>8.4f} "
              f"{r['NDCG@10']:>8.4f} "
              f"{r['NDCG@20']:>8.4f} "
              f"{r['MAP@10']:>8.4f} "
              f"{r['Recall@20']:>8.4f}")
    print(sep)
    print()

if __name__ == "__main__":
    main()
