#!/usr/bin/env python3
"""Validate the authentic PlanBench grading against the ORIGINAL stored grades.

The PlanBench handler (evaluations/handlers/planbench.py) now grades non-task_3
tasks by reconstructing the instance from the query and running the original VAL
pipeline. This script proves that reconstruction is faithful: it re-grades every
instance in results/raw_results/planbench/<provider>/<model>/<task>.json with the
handler's _planbench_grade() and compares to the `llm_correct` value the ORIGINAL
PlanBench evaluation stored in the same JSON. If they match everywhere, the
handler reproduces the standardized results exactly.

Requires VAL (env var $VAL pointing at a dir with the `validate` binary) — run on
Linux after `python3 evaluations/benchmarks/planbench/plan-bench/install_val.py`
(and symlink `validate` -> `Validate` if needed). Run from the repo root:

    export VAL="$HOME/.planutils/packages/val/bin"
    python evaluations/_validate_planbench.py
    python evaluations/_validate_planbench.py --task task_5_plan_generalization  # one task
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from evaluations.handlers.planbench import _planbench_grade  # noqa: E402

RAW = _REPO / "results" / "raw_results" / "planbench"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="only this task (default: all non-task_3)")
    ap.add_argument("--limit", type=int, default=0, help="max instances per (model,task)")
    args = ap.parse_args()
    if not os.environ.get("VAL"):
        print("WARNING: $VAL is not set — plan tasks will fail to validate.", file=sys.stderr)

    files = sorted(glob.glob(str(RAW / "*" / "*" / "*.json")))
    per_task = defaultdict(lambda: [0, 0])   # task -> [match, total]
    mismatches = []
    for path in files:
        task = Path(path).stem
        if task == "task_3_plan_verification":
            continue
        if args.task and task != args.task:
            continue
        data = json.load(open(path))
        insts = data.get("instances", [])
        seen = 0
        for inst in insts:
            resp = inst.get("llm_raw_response")
            orig = inst.get("llm_correct")
            if not resp or orig is None:
                continue
            if args.limit and seen >= args.limit:
                break
            seen += 1
            try:
                got, _, _ = _planbench_grade(
                    task, inst.get("query", ""), resp, inst.get("ground_truth_plan", ""))
            except Exception as e:
                got = f"ERR:{type(e).__name__}"
            ok = (got == int(bool(orig)))
            per_task[task][1] += 1
            if ok:
                per_task[task][0] += 1
            elif len(mismatches) < 20:
                mismatches.append((task, inst.get("instance_id"), Path(path).parent.name,
                                   f"orig={int(bool(orig))} got={got}"))

    print(f"\n{'task':32} {'match/total':>14} {'rate':>7}")
    print("-" * 58)
    tm = tt = 0
    for task in sorted(per_task):
        m, t = per_task[task]
        tm += m; tt += t
        print(f"{task:32} {f'{m}/{t}':>14} {(m/t*100 if t else 0):6.1f}%")
    print("-" * 58)
    print(f"{'TOTAL':32} {f'{tm}/{tt}':>14} {(tm/tt*100 if tt else 0):6.1f}%")
    if mismatches:
        print("\nSample mismatches (first 20):")
        for task, iid, model, detail in mismatches:
            print(f"  {task}/{iid} [{model}]  {detail}")
    print("\n=> 100% match means the handler reproduces the original PlanBench grades.")


if __name__ == "__main__":
    main()
