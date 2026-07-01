"""
BEIR LLM Reranking Pipeline — Listwise (RankGPT / sliding window)
------------------------------------------------------------------
1. Dense retrieval (SentenceBERT) → top-K candidates
2. LLM reranks top-K via listwise sliding-window (RankGPT style)
3. Evaluate with BEIR metrics (NDCG, MAP, Recall, Precision)
"""

import argparse
import json
import logging
import os
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
# beir.retrieval.models and DenseRetrievalExactSearch are only needed for the
# standalone first-stage retrieval script, not for LLM reranking.
try:
    from beir.retrieval import models
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
except (ImportError, RuntimeError):
    models = None
    DRES = None

_STATUS_INTERVAL = 10

WINDOW_SIZE = 20
STRIDE = 10
MAX_DOC_CHARS = 500
MAX_RETRIES = 5
INITIAL_BACKOFF = 2


def _write_status(path: str, data: dict):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        pass


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.WARNING,
    handlers=[LoggingHandler()],
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PUBLIC_DATASETS = [
    "scifact", "nfcorpus", "fiqa", "arguana", "webis-touche2020",
    "dbpedia-entity", "scidocs", "climate-fever", "fever",
    "hotpotqa", "nq", "quora", "msmarco", "cqadupstack",
]

LISTWISE_PROMPT = """I will provide you with {num_docs} passages, each indicated by a number identifier []. Rank the passages based on their relevance to the search query: {query}

{passages}

Search Query: {query}
Rank the {num_docs} passages above based on their relevance to the search query. Your response must be a single JSON array of integers, most relevant first. Example for 4 passages: [3, 1, 4, 2]. Output only the JSON array with no explanation, no markdown, no extra text."""


def build_prompt(query, doc_list):
    passages = []
    for i, (doc_id, title, text) in enumerate(doc_list, 1):
        content = f"{title} {text}".strip()[:MAX_DOC_CHARS]
        passages.append(f"[{i}] {content}")
    return LISTWISE_PROMPT.format(
        num_docs=len(doc_list),
        query=query,
        passages="\n\n".join(passages),
    )


def parse_ranking(response_text, num_docs):
    """Parse JSON array from model response. Returns 0-based index list."""
    m = re.search(r'\[[\d,\s]+\]', response_text)
    if not m:
        return list(range(num_docs))
    try:
        indices = json.loads(m.group())
    except Exception:
        return list(range(num_docs))
    seen = set()
    ranked = []
    for idx in indices:
        i = int(idx) - 1  # 1-based → 0-based
        if 0 <= i < num_docs and i not in seen:
            ranked.append(i)
            seen.add(i)
    ranked += [i for i in range(num_docs) if i not in seen]
    return ranked


def rank_window(engine, query_text, doc_list, api_base=None, api_key=None):
    """Send one window to LLM; return doc_list reordered most→least relevant."""
    prompt = build_prompt(query_text, doc_list)
    messages = [{"role": "user", "content": prompt}]
    _api_key = api_key or os.environ.get("LITELLM_PROXY_API_KEY") or os.environ.get("OPENAI_API_KEY")
    _base_url = api_base or "https://cmu.litellm.ai"
    _client = OpenAI(api_key=_api_key, base_url=_base_url)
    for attempt in range(MAX_RETRIES):
        try:
            response = _client.chat.completions.create(
                model=engine,
                messages=messages,
                max_tokens=2048,
            )
            text = response.choices[0].message.content or ""
            ranked_idx = parse_ranking(text, len(doc_list))
            return [doc_list[i] for i in ranked_idx]
        except Exception as e:
            error_str = str(e).lower()
            is_retryable = any(kw in error_str for kw in [
                "rate_limit", "ratelimit", "429", "too many requests",
                "timeout", "timed out", "connection", "server_error",
                "500", "502", "503",
            ])
            if is_retryable and attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_BACKOFF * (2 ** attempt))
            else:
                logger.warning(f"rank_window failed: {e}")
                return doc_list  # fallback: original order
    return doc_list


def sliding_window_rerank(engine, query_text, doc_list, api_base=None, api_key=None):
    """
    Bottom-up sliding window reranking (RankGPT style).
    doc_list ordered best-first by first-stage retriever.
    Returns reordered doc_list, most relevant first.
    """
    n = len(doc_list)
    if n <= WINDOW_SIZE:
        return rank_window(engine, query_text, doc_list, api_base=api_base, api_key=api_key)

    start = n - WINDOW_SIZE
    while True:
        end = start + WINDOW_SIZE
        window = doc_list[start:end]
        reranked_window = rank_window(engine, query_text, window, api_base=api_base, api_key=api_key)
        doc_list = doc_list[:start] + reranked_window + doc_list[end:]
        if start == 0:
            break
        start = max(0, start - STRIDE)

    return doc_list


def llm_rerank(engine, corpus, queries, first_stage_results, top_k, max_workers,
               status_file=None, api_base=None, api_key=None):
    """Rerank all queries using listwise sliding-window LLM reranking."""
    query_list = list(queries.items())
    total = len(query_list)
    logger.info(f"Listwise reranking {total} queries | engine={engine} top_k={top_k} workers={max_workers}")

    start_time = time.time()
    if status_file:
        _write_status(status_file, {
            "model": engine, "stage": "reranking",
            "completed": 0, "total": total,
            "started_at": start_time, "updated_at": start_time,
            "status": "running",
        })

    def _rerank_query(qid_query):
        qid, query_text = qid_query
        doc_scores = first_stage_results.get(qid, {})
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        doc_list = [
            (doc_id, corpus[doc_id].get("title", ""), corpus[doc_id].get("text", ""))
            for doc_id, _ in sorted_docs
        ]
        ranked = sliding_window_rerank(engine, query_text, doc_list, api_base=api_base, api_key=api_key)
        n = len(ranked)
        return qid, {doc_id: n - rank for rank, (doc_id, _, _) in enumerate(ranked)}

    reranked = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_rerank_query, item): item for item in query_list}
        for future in tqdm(as_completed(futures), total=total, desc="Reranking queries"):
            qid, scores = future.result()
            reranked[qid] = scores
            completed += 1
            if status_file and completed % _STATUS_INTERVAL == 0:
                _write_status(status_file, {
                    "model": engine, "stage": "reranking",
                    "completed": completed, "total": total,
                    "started_at": start_time, "updated_at": time.time(),
                    "status": "running",
                })

    if status_file:
        _write_status(status_file, {
            "model": engine, "stage": "reranking",
            "completed": total, "total": total,
            "started_at": start_time, "updated_at": time.time(),
            "status": "running",
        })
    return reranked


def run_dataset(dataset_name, engine, top_k, max_workers, first_stage_model,
                base_dir, results_base, skip_existing, status_file=None,
                api_base=None, api_key=None):
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {dataset_name} | Engine: {engine} | top_k: {top_k}")
    logger.info(f"{'='*60}")

    safe_engine = engine.replace("/", "_")
    result_file = os.path.join(results_base, dataset_name, safe_engine, "metrics.json")
    if skip_existing and os.path.exists(result_file):
        logger.info(f"[SKIP] Results already exist: {result_file}")
        return None

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = os.path.join(base_dir, "datasets", dataset_name)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, os.path.join(base_dir, "datasets"))

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    logger.info(f"Loaded {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")

    first_stage_file = os.path.join(results_base, dataset_name, "first_stage_results.json")
    if os.path.exists(first_stage_file):
        logger.info(f"Loading cached first-stage results from {first_stage_file}")
        with open(first_stage_file) as f:
            first_stage_results = json.load(f)
    else:
        logger.info(f"Running dense retrieval with {first_stage_model}...")
        model = DRES(models.SentenceBERT(first_stage_model), batch_size=64)
        retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=[100])
        first_stage_results = retriever.retrieve(corpus, queries)
        os.makedirs(os.path.dirname(first_stage_file), exist_ok=True)
        with open(first_stage_file, "w") as f:
            json.dump(first_stage_results, f)
        logger.info(f"Cached first-stage results to {first_stage_file}")

    reranked_file = os.path.join(results_base, dataset_name, safe_engine, "reranked.json")
    os.makedirs(os.path.dirname(reranked_file), exist_ok=True)

    if os.path.exists(reranked_file) and skip_existing:
        logger.info(f"Loading cached reranked results from {reranked_file}")
        with open(reranked_file) as f:
            reranked_results = json.load(f)
    else:
        reranked_results = llm_rerank(
            engine, corpus, queries, first_stage_results, top_k, max_workers,
            status_file=status_file, api_base=api_base, api_key=api_key,
        )
        with open(reranked_file, "w") as f:
            json.dump(reranked_results, f)

    k_values = [1, 5, 10, 20]
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, reranked_results, k_values
    )

    metrics = {
        "dataset": dataset_name,
        "engine": engine,
        "top_k_reranked": top_k,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "first_stage_model": first_stage_model,
        "num_queries": len(queries),
        "num_corpus": len(corpus),
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precision": precision,
    }

    with open(result_file, "w") as f:
        json.dump(metrics, f, indent=2)

    if status_file:
        _write_status(status_file, {
            "model": engine, "stage": "done",
            "ndcg10": ndcg.get("NDCG@10", 0),
            "started_at": time.time(), "updated_at": time.time(),
            "status": "done",
        })

    logger.info(f"Results for {dataset_name}:")
    logger.info(f"  NDCG@10: {ndcg.get('NDCG@10', 'N/A'):.4f}")
    logger.info(f"  MAP@10:  {_map.get('MAP@10', 'N/A'):.4f}")
    logger.info(f"  Saved to {result_file}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="BEIR Listwise LLM Reranking Pipeline")
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", default=["nfcorpus"])
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--first_stage_model", type=str,
                        default="Alibaba-NLP/gte-modernbert-base")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--ignore_existing", action="store_true")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Custom API base URL (bypasses CMU LiteLLM proxy)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for custom endpoint")
    args = parser.parse_args()

    if args.ignore_existing:
        args.skip_existing = False

    base_dir = str(pathlib.Path(__file__).parent.absolute())
    results_base = os.path.join(base_dir, "beir_results")
    os.makedirs(results_base, exist_ok=True)

    safe_engine = args.engine.replace("/", "_")
    status_file = os.path.join(results_base, "_status", f"{safe_engine}.json")

    all_metrics = []
    for dataset_name in args.datasets:
        try:
            m = run_dataset(
                dataset_name, args.engine, args.top_k, args.max_workers,
                args.first_stage_model, base_dir, results_base, args.skip_existing,
                status_file=status_file,
                api_base=args.api_base, api_key=args.api_key,
            )
            if m:
                all_metrics.append(m)
        except Exception as e:
            logger.error(f"Failed on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_metrics:
        print("\n" + "=" * 70)
        print(f"{'Dataset':<25} {'NDCG@5':>8} {'NDCG@10':>8} {'NDCG@20':>8} {'MAP@10':>8} {'R@20':>8}")
        print("-" * 80)
        for m in all_metrics:
            print(f"{m['dataset']:<25} "
                  f"{m['ndcg'].get('NDCG@5', 0):>8.4f} "
                  f"{m['ndcg'].get('NDCG@10', 0):>8.4f} "
                  f"{m['ndcg'].get('NDCG@20', 0):>8.4f} "
                  f"{m['map'].get('MAP@10', 0):>8.4f} "
                  f"{m['recall'].get('Recall@20', 0):>8.4f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
