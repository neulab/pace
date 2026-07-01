#!/usr/bin/env python3
"""Score a NEW model on the selected source instances and write standardized CSVs.

Closes the loop between `evaluations/` and `pacebench/`:

    selections CSV  --(this script: run each instance, extract score)-->
        results/standardized_results/<benchmark>/<subdir>/<NewModel>.csv
    --> scripts/pacebench/analysis/predict_new_model.py

For every row in the selection CSV it calls the benchmark handler once per unique
(benchmark, subtask, instance_id), extracts the metric that matches the row's
subdir, and appends `(instance_id, score, metric_name)` to the standardized CSV
that pacebench reads.

Usage
-----
  export API_KEY=...   # OpenAI-compatible endpoint that serves the new model
  python evaluations/score_new_model.py \
      --model azure_ai/gpt-5.2 \
      --model-name-out My-New-Model \
      --base-url https://cmu.litellm.ai \
      --selections scripts/pacebench/selections/abs_fit/selections_C100.csv

`--model` is the id sent to the API; `--model-name-out` is the filename written
under standardized_results (defaults to `--model`). Re-running resumes: instances
already present in the output CSVs are skipped.

Support / limitations
---------------------
Fully scored by the handler: visualpuzzles, mmmu, livecodebench, repobench, bfcl,
lifbench, ifeval, logiqa, acp_gen, aime25, gpqa, humaneval_chat, mbpp_chat,
mmlu_cot, infobench, and debugbench (needs LeetCode cookies in keys.json).
planbench: only `task_3_plan_verification` is auto-scored; the other tasks need
VAL + Fast Downward (see benchmarks/planbench/plan-bench/install_val.py) and are
skipped with a warning.
"""
import argparse
import csv
import os
import re
import sys
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from evaluations.run import run_instance  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_STD = _REPO / "results" / "standardized_results"
_REPOBENCH_METRIC = re.compile(r"_(codebleu|edit_similarity|exact_match)$")
SUBDIR_IS_SUBTASK = {"planbench", "bfcl", "livecodebench", "lifbench", "acp_gen",
                     "visualwebbench"}
METRIC_SUBDIR = {"ifeval", "logiqa", "gpqa", "aime25", "mmlu_cot",
                 "humaneval_chat", "mbpp_chat"}
# Handlers that are NOT thread-safe (os.chdir, lm_eval global harness state,
# os.environ mutation) — serialized behind a lock; everything else runs concurrently.
SERIAL_BENCHMARKS = {"livecodebench", "bfcl", "debugbench",
                     "acp_gen", "aime25", "gpqa", "humaneval_chat", "ifeval",
                     "logiqa", "mbpp_chat", "mmlu_cot"}


def _num(v):
    """bool/None/number -> float or None."""
    if v is True:
        return 1.0
    if v is False:
        return 0.0
    if v is None:
        return None
    return float(v)


def eval_call(benchmark, subdir, instance_id):
    """(benchmark, subdir, id) -> (subtask_or_None, instance_id) for run_instance."""
    if benchmark == "repobench":
        return _REPOBENCH_METRIC.sub("", subdir), instance_id
    if benchmark in METRIC_SUBDIR:
        return None, instance_id
    if benchmark in SUBDIR_IS_SUBTASK and subdir:
        return subdir, instance_id
    return None, instance_id


def extract(benchmark, subdir, result):
    """Return (metric_name, score) for this row's subdir, or None if unscorable."""
    r = result[0] if isinstance(result, list) and result else result
    if not isinstance(r, dict):
        return None
    b = benchmark
    if b == "visualpuzzles":
        return ("exact_match", _num(r.get("correct")))
    if b == "mmmu":
        return ("mmmu_acc", _num(r.get("correct")))
    if b == "livecodebench":
        return ("pass@1", _num(r.get("pass@1", r.get("pass_at_1", 0.0))))
    if b == "bfcl":
        return ("accuracy", _num(r.get("valid")))
    if b == "lifbench":
        return ("total_score", _num((r.get("score_dict") or {}).get("total_score", 0.0)))
    if b == "logiqa":
        return ("exact_match", _num(r.get("exact_match", 0.0)))
    if b == "acp_gen":
        return ("score", _num(r.get("score", 0.0)))
    if b in ("aime25", "gpqa", "mmlu_cot"):
        return ("exact_match", _num(r.get("exact_match", 0.0)))
    if b == "humaneval_chat":
        return ("pass@1", _num(r.get("pass@1", r.get("pass_at_1", 0.0))))
    if b == "mbpp_chat":
        return ("pass_at_1", _num(r.get("pass_at_1", r.get("pass@1", 0.0))))
    if b == "infobench":
        return ("accuracy", _num((r.get("accuracy_percent") or 0.0) / 100.0))
    if b == "debugbench":
        return ("debug_accuracy", _num(r.get("test_result_bool")))
    if b == "repobench":
        for metric in ("codebleu", "edit_similarity", "exact_match"):
            if subdir.endswith("_" + metric):
                return (metric, _num(r.get(metric, 0.0)))
        return None
    if b == "ifeval":
        v = r.get(subdir)
        if isinstance(v, list):
            v = (sum(1.0 for x in v if x) / len(v)) if v else 0.0
        return (subdir, _num(v))
    if b == "planbench":
        if subdir == "task_3_plan_verification":
            return None if r.get("llm_correct_binary") is None else \
                ("llm_correct", _num(r["llm_correct_binary"]))
        return None  # other tasks need VAL
    return None


def _load_existing(path):
    if not path.exists():
        return set()
    return {row["id"] for row in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selections", required=True)
    ap.add_argument("--model", required=True, help="model id sent to the API")
    ap.add_argument("--model-name-out", default=None,
                    help="filename under standardized_results (default: --model)")
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "https://cmu.litellm.ai"))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY"))
    ap.add_argument("--limit", type=int, default=0, help="only process the first N rows (debug)")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent API calls (default 8; unsafe handlers stay serialized)")
    args = ap.parse_args()
    if not args.api_key:
        ap.error("set --api-key or export API_KEY")
    out_name = args.model_name_out or args.model

    rows = list(csv.DictReader(open(args.selections)))
    if args.limit:
        rows = rows[:args.limit]

    def flush(written):
        """Merge accumulated triples into the standardized CSVs (dedup by id)."""
        for (b, subdir), triples in written.items():
            if not triples:
                continue
            out_dir = _STD / b / subdir if subdir else _STD / b
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / f"{out_name}.csv"
            existing = {}
            if out_csv.exists():
                for r in csv.DictReader(open(out_csv)):
                    existing[r["id"]] = (r["score"], r["metric_name"])
            for iid, score, metric in triples:
                existing[iid] = (score, metric)
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "score", "metric_name"])
                for iid, (score, metric) in existing.items():
                    w.writerow([iid, score, metric])

    # skip instances already scored in a previous run (resume)
    def already_done(b, subdir, iid):
        out_csv = (_STD / b / subdir if subdir else _STD / b) / f"{out_name}.csv"
        return out_csv.exists() and str(iid) in _load_existing(out_csv)

    # Group rows by unique API call (dedup: repobench's metric rows share one call).
    rows_by_key = defaultdict(list)
    for row in rows:
        b, subdir, iid = row["benchmark"], row["subdir"], row["instance_id"]
        if already_done(b, subdir, iid):
            continue
        subtask, call_iid = eval_call(b, subdir, iid)
        rows_by_key[(b, subtask, call_iid)].append((b, subdir, iid))

    serial_lock = threading.Lock()

    def run_one(key):
        b, subtask, call_iid = key
        try:
            if b in SERIAL_BENCHMARKS:   # not thread-safe (chdir / global state / env)
                with serial_lock:
                    r = run_instance(model_name=args.model, base_url=args.base_url,
                                     api_key=args.api_key, benchmark=b,
                                     subtask=subtask, instance_id=call_iid)
            else:
                r = run_instance(model_name=args.model, base_url=args.base_url,
                                 api_key=args.api_key, benchmark=b,
                                 subtask=subtask, instance_id=call_iid)
            return key, r, None
        except Exception as e:
            return key, None, e

    written = defaultdict(list)   # (benchmark, subdir) -> [(id, score, metric)]
    skipped = defaultdict(int)
    errors = defaultdict(int)
    total = len(rows_by_key)
    done = 0
    # Workers do only the (I/O-bound) API calls; extraction + CSV writes stay on
    # the main thread here, so no locking is needed for `written`/flush.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_one, k) for k in rows_by_key]
        for fut in as_completed(futs):
            key, result, err = fut.result()
            done += 1
            b = key[0]
            if err is not None:
                errors[b] += 1
                print(f"[{done}/{total}] ERROR {b}/{key[2]}: {err}", file=sys.stderr, flush=True)
                continue
            if result is None:
                continue
            for (bb, subdir, iid) in rows_by_key[key]:
                ex = extract(bb, subdir, result)
                if ex is None or ex[1] is None:
                    skipped[bb] += 1
                    continue
                metric, score = ex
                written[(bb, subdir)].append((iid, score, metric))
                print(f"[{done}/{total}] {bb}/{subdir}/{iid} -> {metric}={score}", flush=True)
            if done % 15 == 0:        # incremental flush: crash-safe + resumable
                flush(written)
                written = defaultdict(list)

    flush(written)               # final flush

    print(f"\nWrote scores to CSVs under {_STD}/<benchmark>/.../{out_name}.csv")
    if skipped:
        print("Skipped (unscorable, e.g. planbench needs VAL): " +
              ", ".join(f"{k}={v}" for k, v in skipped.items()))
    if errors:
        print("Errors: " + ", ".join(f"{k}={v}" for k, v in errors.items()))


if __name__ == "__main__":
    main()
