#!/usr/bin/env python3
"""From a dumped selection, list exactly which source instances a NEW model must
be evaluated on (so you don't run whole benchmarks — only the selected instances).

The held-out fold for a new model selects on the *other* models' scores, which is
identical to the all-models fit dump. So the instances in the fit selection CSV
(selections/abs_fit/selections_C100.csv) ARE the set the new model needs scored.

Standardized-results (benchmark, subdir, instance_id) are translated to the
evaluations `--benchmark/--subtask/--instance_id` form, then de-duplicated to
one row per evaluation call:
  * repobench : subdir carries a metric suffix (_codebleu/_edit_similarity/
                _exact_match) → stripped to the real subtask; one run yields all
                metrics, so those rows collapse to one.
  * lm_eval metric benchmarks (ifeval/logiqa/gpqa/aime25/mmlu_cot/humaneval_chat/
                mbpp_chat) : subdir is a metric name → dropped (default subtask),
                deduped by instance_id.
  * planbench/bfcl/livecodebench/lifbench/acp_gen/visualwebbench : subdir == subtask.
  * mmmu/visualpuzzles/debugbench/infobench/beir_nfcorpus : no subtask.

Outputs: a per-benchmark summary, eval_plan.csv, and (optional) runnable commands.

Usage
-----
  python analysis/emit_eval_plan.py --selections selections/abs_fit/selections_C100.csv
  python analysis/emit_eval_plan.py --selections ... --emit-commands \
      --model azure_ai/gpt-5.2 --base-url https://cmu.litellm.ai
"""
import argparse
import csv
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

SUBDIR_IS_SUBTASK = {"planbench", "bfcl", "livecodebench", "lifbench", "acp_gen",
                     "visualwebbench"}
METRIC_SUBDIR = {"ifeval", "logiqa", "gpqa", "aime25", "mmlu_cot",
                 "humaneval_chat", "mbpp_chat"}
_REPOBENCH_METRIC = re.compile(r"_(codebleu|edit_similarity|exact_match)$")


def translate(benchmark, subdir, instance_id):
    """(benchmark, subdir, id) -> (benchmark, subtask_or_None, id) for evaluations."""
    if benchmark == "repobench":
        return benchmark, _REPOBENCH_METRIC.sub("", subdir), instance_id
    if benchmark in METRIC_SUBDIR:
        return benchmark, None, instance_id           # drop metric subdir
    if benchmark in SUBDIR_IS_SUBTASK and subdir:
        return benchmark, subdir, instance_id
    return benchmark, None, instance_id               # no subtask


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selections", required=True)
    ap.add_argument("--out", default="eval_plan.csv")
    ap.add_argument("--model", default="<MODEL>")
    ap.add_argument("--base-url", default="<BASE_URL>")
    ap.add_argument("--emit-commands", action="store_true")
    args = ap.parse_args()

    # translate + dedup to one row per (benchmark, subtask, instance_id)
    plan = OrderedDict()
    for r in csv.DictReader(open(args.selections)):
        b, sub, iid = translate(r["benchmark"], r["subdir"], r["instance_id"])
        plan.setdefault((b, sub or "", iid), None)

    by_bench = defaultdict(list)
    for (b, sub, iid) in plan:
        by_bench[b].append((sub, iid))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "subtask", "instance_id"])
        for (b, sub, iid) in plan:
            w.writerow([b, sub, iid])

    print(f"New model must be scored on {len(plan)} unique evaluation calls "
          f"across {len(by_bench)} benchmarks.")
    print(f"Full list → {args.out}\n")
    print(f"{'benchmark':16s} {'#instances':>10s}  {'#subtasks':>9s}")
    print("-" * 40)
    for b in sorted(by_bench, key=lambda x: -len(by_bench[x])):
        subs = {s for s, _ in by_bench[b] if s}
        print(f"{b:16s} {len(by_bench[b]):10d}  {len(subs):9d}")

    if args.emit_commands:
        print("\n# ── evaluations/run.py commands (run from repo root) ──")
        for (b, sub, iid) in plan:
            cmd = (f"python -m evaluations.run --model_name {args.model} "
                   f"--base_url {args.base_url} --api_key $API_KEY "
                   f"--benchmark {b} --instance_id {iid!r}")
            if sub:
                cmd += f" --subtask {sub}"
            print(cmd)


if __name__ == "__main__":
    main()
