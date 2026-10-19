#!/usr/bin/env python3
"""
Export PACE-Bench as per-target JSONL from the fitted selections.

For each of the 4 agentic targets, emit one JSONL file whose rows are the
C=100 selected *source* instances, each line:

    instance_id, source_benchmark, subdir, input, answer, metric, weight, images

- input / answer are resolved from local raw-results, vendored data, or the HF
  datasets cache (offline). answer is null when a benchmark has no single gold
  reference (e.g. ifeval checker-based scoring) — the metric field documents how
  it is scored.
- Multimodal benchmarks (mmmu / visualpuzzles / visualwebbench) store their image
  files under images/<benchmark>/ and put the relative paths in `images`.

Instances whose content cannot be resolved get input=answer=null and
content_status="unresolved:<reason>", so the release is honest about coverage.

Requires results/raw_results/ (git-ignored, ~15 GB; host it separately) and
overwrites release/pace-bench/ in place.

Run from the repo root with the pace conda env:
    .../envs/pace/bin/python scripts/pacebench/export_pace_bench.py
"""
from __future__ import annotations
import os, sys, csv, json, ast, glob, re, io
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
SEL_CSV = REPO / "scripts/pacebench/selections/abs_fit/selections_C100.csv"
STD = REPO / "results/standardized_results"
RAW = REPO / "results/raw_results"
BENCH = REPO / "evaluations/benchmarks"
OUT = REPO / "release/pace-bench"
IMG_DIR = OUT / "images"

if not os.environ.get("PACE_ALLOW_NET"):
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ── metric lookup: (benchmark, subdir) -> {instance_id: metric_name} ──────────
_metric_cache: dict = {}
def metric_for(benchmark: str, subdir: str, instance_id: str):
    key = (benchmark, subdir)
    if key not in _metric_cache:
        d = STD / benchmark / subdir if subdir else STD / benchmark
        m = {}
        csvs = sorted(d.glob("*.csv")) if d.is_dir() else []
        if csvs:
            with open(csvs[0]) as f:
                for r in csv.DictReader(f):
                    if "metric_name" in r:
                        m[str(r["id"])] = r.get("metric_name", "")
        _metric_cache[key] = m
    return _metric_cache[key].get(str(instance_id))


# ── HF dataset cache (offline) ────────────────────────────────────────────────
_hf_cache: dict = {}
def hf(name, config=None, split=None):
    key = (name, config, split)
    if key not in _hf_cache:
        from datasets import load_dataset
        _hf_cache[key] = load_dataset(name, config, split=split)
    return _hf_cache[key]


def save_image(pil_img, benchmark, iid, n):
    d = IMG_DIR / benchmark
    d.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.-]", "_", str(iid))
    p = d / f"{safe}_{n}.png"
    if pil_img.mode not in ("RGB", "RGBA", "L"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(p)
    return str(p.relative_to(OUT))


# ── per-benchmark resolvers: (subdir, instance_id) -> (input, answer, images) ──
# Each returns (input:str|None, answer:str|None, images:list[str]). Raise to mark
# unresolved.

def r_ifeval(subdir, iid):
    # subdir is the metric; input from any model's samples_*.jsonl by doc_id
    files = glob.glob(str(RAW / "ifeval" / "*" / "samples_*.jsonl"))
    if not files:
        raise FileNotFoundError("no ifeval samples")
    want = int(iid)
    with open(files[0]) as f:
        for line in f:
            d = json.loads(line)
            if d.get("doc_id") == want:
                return d["doc"]["prompt"], None, []   # checker-based -> answer null
    raise KeyError(f"ifeval doc_id {iid}")

def r_planbench(subdir, iid):
    # any model's task json under raw_results/planbench/*/<subdir>.json
    files = glob.glob(str(RAW / "planbench" / "**" / f"{subdir}.json"), recursive=True)
    if not files:
        raise FileNotFoundError(f"no planbench {subdir}")
    d = json.load(open(files[0]))
    for inst in d.get("instances", []):
        if str(inst.get("instance_id")) == str(iid):
            return inst.get("query"), inst.get("ground_truth_plan"), []
    raise KeyError(f"planbench {subdir}/{iid}")

def r_logiqa(subdir, iid):
    # iid like '29_strict-match' -> row index 29 into test split
    idx = int(str(iid).split("_")[0])
    ds = hf("EleutherAI/logiqa", "logiqa", "test")
    row = ds[idx]
    opts = row["options"]
    if isinstance(opts, str):
        try: opts = ast.literal_eval(opts)
        except Exception: opts = [opts]
    letters = "abcd"
    choices = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))
    inp = f"{row['context']}\n\n{row['question']}\n{choices}"
    return inp, str(row.get("label")), []

def r_acp(subdir, iid):
    ds = hf("ibm-research/acp_bench", subdir, "test")
    row = ds[int(iid)]                       # id field is a hash -> use row index
    inp = f"{row.get('context','')}\n\n{row.get('question','')}".strip()
    return inp, str(row.get("answer")), []

def r_mmmu(subdir, iid):
    split = str(iid).split("_")[0]
    ds = hf("lmms-lab/MMMU", None, split)
    if not hasattr(r_mmmu, "_idx"):
        r_mmmu._idx = {}
    if split not in r_mmmu._idx:
        r_mmmu._idx[split] = {row["id"]: i for i, row in enumerate(ds)}
    i = r_mmmu._idx[split].get(iid)
    if i is None:
        raise KeyError(f"mmmu {iid}")
    row = ds[i]
    q = row["question"]
    opts = row.get("options")
    if opts and str(opts) not in ("[]", "None"):
        try: opts = ast.literal_eval(opts) if isinstance(opts, str) else opts
        except Exception: opts = []
        if opts:
            q += "\n" + "\n".join(f"{chr(65+j)}. {o}" for j, o in enumerate(opts))
    imgs = []
    for k in range(1, 8):
        im = row.get(f"image_{k}")
        if im is not None:
            imgs.append(save_image(im, "mmmu", iid, k))
    return q, str(row.get("answer")), imgs

def r_visualpuzzles(subdir, iid):
    num = int(re.sub(r"\D", "", str(iid)))   # 'visualpuzzles_1115' -> 1115
    ds = hf("neulab/VisualPuzzles", None, "train")
    if not hasattr(r_visualpuzzles, "_idx"):
        r_visualpuzzles._idx = {int(row["id"]): i for i, row in enumerate(ds)}
    i = r_visualpuzzles._idx.get(num)
    if i is None:
        raise KeyError(f"visualpuzzles {iid}")
    row = ds[i]
    q = row["question"]
    opts = row.get("options")
    if opts:
        try: opts = ast.literal_eval(opts) if isinstance(opts, str) else opts
        except Exception: opts = None
        if opts:
            q += "\n" + "\n".join(str(o) for o in opts)
    imgs = [save_image(row["image"], "visualpuzzles", iid, 1)] if row.get("image") else []
    return q, str(row.get("answer")), imgs

def r_debugbench(subdir, iid):
    # iid = '<lang>_<category>_<slug>', file = benchmark/<lang>_<category>.json
    m = re.match(r"(.+?_[^_]+(?: [^_]+)*)_(.+)$", str(iid))
    # robust: file is longest prefix that matches an existing json
    d = BENCH / "debugbench" / "benchmark"
    fname = None; slug = None
    for jf in d.glob("*.json"):
        stem = jf.stem
        if str(iid).startswith(stem + "_"):
            fname = jf; slug = str(iid)[len(stem) + 1:]; break
    if fname is None:
        raise KeyError(f"debugbench file for {iid}")
    data = json.load(open(fname))
    for rec in data:
        if rec.get("slug") == slug:
            inp = f"{rec.get('description','')}\n\n# Buggy code:\n{rec.get('buggy_code','')}"
            return inp, rec.get("oracle_code"), []
    raise KeyError(f"debugbench slug {slug}")

BFCL_ROOT = REPO / "evaluations/benchmarks/BFCL/gorilla/berkeley-function-call-leaderboard"
BFCL_SUBTASK_TO_CAT = {
    "non_live_simple_python": "simple_python", "non_live_simple_java": "simple_java",
    "non_live_simple_javascript": "simple_javascript", "non_live_multiple": "multiple",
    "non_live_parallel": "parallel", "non_live_parallel_multiple": "parallel_multiple",
    "non_live_irrelevance": "irrelevance",
    "live_simple": "live_simple", "live_multiple": "live_multiple",
    "live_parallel": "live_parallel", "live_parallel_multiple": "live_parallel_multiple",
    "live_irrelevance": "live_irrelevance", "live_relevance": "live_relevance",
    "multi_turn_base": "multi_turn_base", "multi_turn_long_context": "multi_turn_long_context",
    "multi_turn_miss_func": "multi_turn_miss_func", "multi_turn_miss_param": "multi_turn_miss_param",
    "agentic_memory_kv": "memory_kv", "agentic_memory_vector": "memory_vector",
    "agentic_memory_rec_sum": "memory_rec_sum", "agentic_web_search_base": "web_search_base",
    "agentic_web_search_no_snippet": "web_search_no_snippet",
}
_bfcl_cache: dict = {}
def _bfcl_load(cat):
    if cat not in _bfcl_cache:
        if str(BFCL_ROOT) not in sys.path:
            sys.path.insert(0, str(BFCL_ROOT))
        from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry
        es = load_dataset_entry(cat, include_prereq=False, include_language_specific_hint=False)
        try:
            gt = {str(g["id"]): g.get("ground_truth") for g in load_ground_truth_entry(cat)}
        except Exception:
            gt = None
        _bfcl_cache[cat] = (es, gt)
    return _bfcl_cache[cat]

def r_bfcl(subdir, iid):
    # selection id '<cat>_<N>'; N is the POSITION in load_dataset_entry(category)
    cat = BFCL_SUBTASK_TO_CAT.get(subdir)
    if cat is None:
        raise KeyError(f"bfcl subtask {subdir}")
    n = int(re.findall(r"\d+", str(iid))[-1])
    es, gt = _bfcl_load(cat)
    if n >= len(es):
        raise KeyError(f"bfcl idx {n} >= {len(es)}")
    e = es[n]
    parts = []
    q = e.get("question")
    if isinstance(q, list):
        for turn in q:
            for msg in (turn if isinstance(turn, list) else [turn]):
                if isinstance(msg, dict):
                    parts.append(f"{msg.get('role','')}: {msg.get('content','')}")
    inp = "\n".join(parts)
    if e.get("function"):
        inp += "\n\nAvailable functions:\n" + json.dumps(e["function"], ensure_ascii=False)
    ans = None
    if gt is not None:
        g = gt.get(str(e["id"]))
        if g is not None:
            ans = json.dumps(g, ensure_ascii=False)
    return inp, ans, []

def r_livecodebench(subdir, iid):
    if subdir in ("codegeneration", "selfrepair"):
        # selfrepair reuses the code_generation problems (repair a wrong solution)
        ds = hf("livecodebench/code_generation_lite", None, "test")
        row = ds[int(iid)]
        inp = row.get("question_content") or row.get("question_title", "")
        return inp, None, []                 # tests as answer -> null
    if subdir == "codeexecution":
        ds = hf("livecodebench/execution-v2", None, "test")
        row = ds[int(iid)]
        inp = f"Code:\n{row.get('code','')}\n\nInput: {row.get('input','')}"
        return inp, (str(row["output"]) if row.get("output") is not None else None), []
    raise NotImplementedError(f"lcb {subdir}")

def r_visualwebbench(subdir, iid):
    # iid = '<task>_<n>', task is the HF builder config; ds 'id' matches iid exactly
    task = re.sub(r"_\d+$", "", str(iid))
    ds = hf("webbench/WebBench", task, "test")
    if not hasattr(r_visualwebbench, "_idx"):
        r_visualwebbench._idx = {}
    if task not in r_visualwebbench._idx:
        r_visualwebbench._idx[task] = {row["id"]: i for i, row in enumerate(ds)}
    i = r_visualwebbench._idx[task].get(iid)
    if i is None:
        raise KeyError(f"visualwebbench {iid}")
    row = ds[i]
    txt = ""
    for k in ("question", "instruction", "text"):
        if row.get(k):
            txt = str(row[k]); break
    inp = f"[{row.get('task_type', task)}] {row.get('website','')}".strip()
    if txt:
        inp += "\n" + txt
    imgs = [save_image(row["image"], "visualwebbench", iid, 1)] if row.get("image") else []
    return inp, (str(row["answer"]) if row.get("answer") is not None else None), imgs

_RB_SPLIT = {"xff": "cross_file_first", "xfr": "cross_file_random", "if": "in_file"}
def r_repobench(subdir, iid):
    # subdir 'repobench_<xff|xfr|if>_python_<metric>'; iid 'repobench_<row idx>'
    m = re.search(r"repobench_(xff|xfr|if)_", subdir)
    if not m:
        raise KeyError(f"repobench split from {subdir}")
    split = _RB_SPLIT[m.group(1)]
    idx = int(re.sub(r"\D", "", str(iid)))
    ds = hf("tianyang/repobench_python_v1.1", "default")[split]
    if idx >= len(ds):
        raise KeyError(f"repobench idx {idx} >= {len(ds)}")
    row = ds[idx]
    inp = (f"# File: {row.get('file_path','')}\n{row.get('import_statement','')}\n\n"
           f"{row.get('cropped_code','')}")
    return inp, row.get("next_line"), []

_lif_cache: dict = {}
def r_lifbench(subdir, iid):
    # id '{ins_id}_{param_id}_{length}'; prompt pre-generated in data/prompts/<subtask>.json
    p = BENCH / "lifbench" / "data" / "prompts" / f"{subdir}.json"
    if subdir not in _lif_cache:
        _lif_cache[subdir] = json.load(open(p)) if p.exists() else []
    parts = str(iid).split("_")
    ins_id, param_id = int(parts[0]), int(parts[1])
    length = int(parts[2].split("#")[0])
    for it in _lif_cache[subdir]:
        if (it.get("ins_id") == ins_id and it.get("param_id") == param_id
                and it.get("length") == length):
            return it.get("prompt"), (str(it["label"]) if it.get("label") is not None else None), []
    raise KeyError(f"lifbench {subdir}/{iid}")

RESOLVERS = {
    "ifeval": r_ifeval, "planbench": r_planbench, "logiqa": r_logiqa,
    "lifbench": r_lifbench,
    "acp_gen": r_acp, "mmmu": r_mmmu, "visualpuzzles": r_visualpuzzles,
    "debugbench": r_debugbench, "livecodebench": r_livecodebench,
    "visualwebbench": r_visualwebbench, "repobench": r_repobench,
    "bfcl": r_bfcl,
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(SEL_CSV)))
    by_target = defaultdict(list)
    for r in rows:
        by_target[r["target"]].append(r)

    cov = defaultdict(lambda: [0, 0])   # benchmark -> [resolved, total]
    for target, recs in sorted(by_target.items()):
        out_path = OUT / f"{target}.jsonl"
        with open(out_path, "w") as f:
            for r in recs:
                bm, subdir, iid = r["benchmark"], r["subdir"], r["instance_id"]
                metric = metric_for(bm, subdir, iid)
                inp = ans = None; imgs = []; status = "ok"
                resolver = RESOLVERS.get(bm)
                cov[bm][1] += 1
                if resolver is None:
                    status = "unresolved:no_resolver"
                else:
                    try:
                        inp, ans, imgs = resolver(subdir, iid)
                        cov[bm][0] += 1
                    except Exception as e:
                        status = f"unresolved:{type(e).__name__}:{str(e)[:60]}"
                line = {
                    "instance_id": iid,
                    "source_benchmark": bm,
                    "subdir": subdir,
                    "input": inp,
                    "answer": ans,
                    "metric": metric,
                    "weight": float(r["weight"]),
                    "images": imgs,
                    "content_status": status,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"  wrote {out_path.relative_to(REPO)}  ({len(recs)} rows)")

    print("\n=== coverage by benchmark (resolved / total) ===")
    for bm in sorted(cov):
        got, tot = cov[bm]
        flag = "" if got == tot else "  <-- partial" if got else "  <-- NONE"
        print(f"  {bm:16s} {got:3d}/{tot:3d}{flag}")
    tot_got = sum(v[0] for v in cov.values()); tot_all = sum(v[1] for v in cov.values())
    print(f"  {'TOTAL':16s} {tot_got:3d}/{tot_all:3d}")


if __name__ == "__main__":
    main()
