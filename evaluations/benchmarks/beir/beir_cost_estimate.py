"""
Cost estimation for listwise BEIR reranking (RankGPT style).
Runs --num_queries queries against a model and extrapolates total cost.

Usage:
  python beir_cost_estimate.py --engine anthropic/claude-opus-4-6 --num_queries 5
"""
import argparse
import json
import os
import pathlib
import re

import litellm
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

litellm.drop_params = True
litellm.suppress_debug_info = True

WINDOW_SIZE = 20
MAX_DOC_CHARS = 500

LISTWISE_PROMPT = """I will provide you with {num_docs} passages, each indicated by a number identifier []. Rank the passages based on their relevance to the search query: {query}

{passages}

Search Query: {query}
Rank the {num_docs} passages above based on their relevance to the search query. Your response must be a single JSON array of integers, most relevant first. Example for 4 passages: [3, 1, 4, 2]. Output only the JSON array with no explanation, no markdown, no extra text."""


def build_prompt(query, docs):
    passages = []
    for i, (doc_id, title, text) in enumerate(docs, 1):
        content = f"{title} {text}".strip()[:MAX_DOC_CHARS]
        passages.append(f"[{i}] {content}")
    return LISTWISE_PROMPT.format(
        num_docs=len(docs),
        query=query,
        passages="\n\n".join(passages),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="LiteLLM model name")
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--num_queries", type=int, default=3,
                        help="Number of queries to sample (default: 3)")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Docs per query to rerank (default: 20)")
    args = parser.parse_args()

    base_dir = str(pathlib.Path(__file__).parent.absolute())

    # Load dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
    data_path = os.path.join(base_dir, "datasets", args.dataset)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, os.path.join(base_dir, "datasets"))
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    print(f"Loaded {len(corpus)} docs, {len(queries)} queries\n")

    # Load or compute first-stage results
    results_base = os.path.join(base_dir, "beir_results")
    first_stage_file = os.path.join(results_base, args.dataset, "first_stage_results.json")
    if os.path.exists(first_stage_file):
        with open(first_stage_file) as f:
            first_stage_results = json.load(f)
        print(f"Loaded cached first-stage results from {first_stage_file}\n")
    else:
        print("Running first-stage dense retrieval (will be cached)...")
        model = DRES(models.SentenceBERT("Alibaba-NLP/gte-modernbert-base"), batch_size=64)
        retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=[100])
        first_stage_results = retriever.retrieve(corpus, queries)
        os.makedirs(os.path.dirname(first_stage_file), exist_ok=True)
        with open(first_stage_file, "w") as f:
            json.dump(first_stage_results, f)
        print(f"Cached to {first_stage_file}\n")

    # Sample queries
    query_ids = list(queries.keys())[:args.num_queries]
    total_queries = len(queries)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    successes = 0

    for i, qid in enumerate(query_ids, 1):
        query_text = queries[qid]
        doc_scores = first_stage_results.get(qid, {})
        top_docs_raw = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:args.top_k]
        doc_list = [
            (doc_id, corpus[doc_id].get("title", ""), corpus[doc_id].get("text", ""))
            for doc_id, _ in top_docs_raw
        ]

        prompt = build_prompt(query_text, doc_list)
        prompt_chars = len(prompt)

        print(f"[{i}/{args.num_queries}] Query: {query_text[:70]}")
        print(f"  Prompt length: {prompt_chars} chars (~{prompt_chars // 4} tokens est.)")

        try:
            response = litellm.completion(
                api_key=os.environ.get("LITELLM_PROXY_API_KEY"),
                api_base="https://cmu.litellm.ai",
                model=f"litellm_proxy/{args.engine}",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            usage = response.usage
            pt = usage.prompt_tokens or 0
            ct = usage.completion_tokens or 0

            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                cost = 0.0

            total_prompt_tokens += pt
            total_completion_tokens += ct
            total_cost += cost
            successes += 1

            resp_text = response.choices[0].message.content or ""
            print(f"  Prompt tokens:     {pt}")
            print(f"  Completion tokens: {ct}")
            print(f"  Cost (USD):        ${cost:.6f}")
            print(f"  Response:          {resp_text[:120]}")

        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    if successes == 0:
        print("All queries failed — cannot estimate cost.")
        return

    avg_prompt = total_prompt_tokens // successes
    avg_completion = total_completion_tokens // successes
    avg_cost = total_cost / successes

    # Extrapolate to full dataset
    # top_k=20: 1 call/query. top_k>20: needs sliding window → more calls
    if args.top_k <= WINDOW_SIZE:
        calls_per_query = 1
    else:
        stride = WINDOW_SIZE // 2
        calls_per_query = (args.top_k - WINDOW_SIZE) // stride + 1

    estimated_total_cost = avg_cost * calls_per_query * total_queries
    estimated_total_calls = calls_per_query * total_queries

    print("=" * 60)
    print(f"COST ESTIMATE — {args.engine}")
    print(f"  Dataset:         {args.dataset} ({total_queries} queries total)")
    print(f"  Method:          Listwise, top-{args.top_k}, window={WINDOW_SIZE}")
    print(f"  Calls/query:     {calls_per_query}")
    print(f"  Total API calls: {estimated_total_calls}")
    print(f"  Avg prompt tok:  {avg_prompt}")
    print(f"  Avg output tok:  {avg_completion}")
    print(f"  Cost per call:   ${avg_cost:.6f}")
    print(f"  ESTIMATED TOTAL: ${estimated_total_cost:.4f} USD")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    finally:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
        except Exception:
            pass
