"""BEIR reranking handler for ProxyBench."""

import json
import os
import sys

_HANDLERS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVALUATIONS_DIR = os.path.dirname(_HANDLERS_DIR)
BENCHMARKS_DIR = os.path.join(_EVALUATIONS_DIR, "benchmarks")

BEIR_DIR = os.path.join(BENCHMARKS_DIR, "beir")

BEIR_DATASET_INFO = {
    "beir_nfcorpus": ("nfcorpus", "nfcorpus"),
}


def _run_beir(
    model_name: str,
    base_url: str,
    api_key: str,
    benchmark: str,
    instance_id: str,
) -> list:
    """Run the full BEIR reranking pipeline and return aggregate metrics.

    instance_id is a metric name from standardized_results (e.g. "NDCG@10").
    Requires cached first_stage_results.json.
    """
    if BEIR_DIR not in sys.path:
        sys.path.insert(0, BEIR_DIR)

    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir_llm_rerank import sliding_window_rerank, WINDOW_SIZE, STRIDE

    FIRST_STAGE_MODEL = "Alibaba-NLP/gte-modernbert-base"
    TOP_K = 20
    VALID_METRICS = {
        f"{cat}@{k}"
        for cat in ("NDCG", "MAP", "Recall", "P")
        for k in (1, 5, 10, 20)
    }
    if instance_id not in VALID_METRICS:
        raise ValueError(
            f"instance_id '{instance_id}' is not a valid BEIR metric. "
            f"Expected one of: {sorted(VALID_METRICS)}"
        )

    dataset_name, results_subdir = BEIR_DATASET_INFO[benchmark]

    data_path = os.path.join(BEIR_DIR, "datasets", dataset_name)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    first_stage_path = os.path.join(
        BEIR_DIR, "beir_results", results_subdir, "first_stage_results.json"
    )
    if not os.path.exists(first_stage_path):
        raise FileNotFoundError(
            f"Cached first-stage results not found: {first_stage_path}\n"
            "Run the full BEIR pipeline first to generate this cache."
        )
    with open(first_stage_path) as f:
        first_stage_results = json.load(f)

    api_base = base_url.rstrip("/")

    # instance_id selects which metric to report; run at most 5 queries so
    # a smoke-test completes quickly without running the whole dataset.
    MAX_QUERIES_SMOKE = int(os.environ.get("BEIR_MAX_QUERIES", len(queries)))
    reranked_results = {}
    for qid, query_text in list(queries.items())[:MAX_QUERIES_SMOKE]:
        doc_scores = first_stage_results.get(qid, {})
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        doc_list = [
            (doc_id, corpus[doc_id].get("title", ""), corpus[doc_id].get("text", ""))
            for doc_id, _ in sorted_docs
            if doc_id in corpus
        ]
        ranked = sliding_window_rerank(
            engine=model_name,
            query_text=query_text,
            doc_list=doc_list,
            api_base=api_base,
            api_key=api_key,
        )
        n = len(ranked)
        reranked_results[qid] = {doc_id: n - rank for rank, (doc_id, _, _) in enumerate(ranked)}

    k_values = [1, 5, 10, 20]
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, reranked_results, k_values)

    return [{
        "dataset": dataset_name,
        "engine": model_name,
        "top_k_reranked": TOP_K,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "first_stage_model": FIRST_STAGE_MODEL,
        "num_queries": len(queries),
        "num_corpus": len(corpus),
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precision": precision,
    }]
