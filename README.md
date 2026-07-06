# Pace / Pace-Bench

Predict a model's performance on **expensive agentic benchmarks**
(GAIA, SWE-Bench Verified, SWE-Bench Multimodal, SWT-Bench) from its performance
on a small, carefully selected set of **cheap non-agentic instances** — at well
under 1% of the cost of a full agentic run.

This README is the from-scratch guide: everything a new person needs to set up
and run, in order.

## Repository layout

```
pace/
├── results/
│   ├── standardized_results/   # per-model, per-instance scores (INPUT to pacebench)
│   └── raw_results/            # full model outputs
├── scripts/pacebench/          # the Pace-Bench pipeline (selection + regression)
│   ├── cli.py                  # entry point: abs / pair / fit-abs / fit-pair
│   ├── config.py               # MODELS, targets, frozen hyperparameters, BASE_DIR
│   ├── analysis/               # figures, tables, and new-model prediction tools
│   └── README.md               # detailed pipeline reference
├── evaluations/                # runs models on the 19 non-agentic benchmarks
│   ├── run.py                  # entry point: score one instance
│   ├── handlers/               # one adapter per benchmark
│   ├── benchmarks/             # vendored third-party benchmark repos
│   ├── requirements-main.txt   # what install.sh installs (main env: 18 benchmarks)
│   ├── requirements/           # per-benchmark deps, to set up ONE benchmark
│   ├── requirements-verified.txt  # frozen known-good pins (reference only; not installed)
│   └── script.sh               # runnable examples for every benchmark
└── requirements.txt            # CORE deps for Part 1 (pacebench + handler layer)
```

There are two independent things you can run:

- **Part 1 — `scripts/pacebench/`**: the prediction pipeline. Uses the scores
  already in `results/` — **no API key, no GPU**. Start here.
- **Part 2 — `evaluations/`**: (re)generate those scores by running models via an
  API. Needs an API key.

## 0. Setup (Part 1)

```bash
conda create -n pace python=3.11 -y
conda activate pace
pip install -r requirements.txt
```

## Part 1 — run the prediction pipeline (no API key)

The input data lives in `results/standardized_results/`, and
`config.py` locates it automatically (you can override it with `PROXYBENCH_BASE_DIR`).

```bash
cd scripts/pacebench

# Goal A: predict absolute agentic scores (MAE / Spearman / Pearson), LOOCV
python cli.py abs  --count 100 --B 300 --seed 42 --auto-tune --strict-budget

# Goal B: predict pairwise model ranking (accuracy), LOOCV
python cli.py pair --count 100 --B 300 --seed 42 --auto-tune --strict-budget --pin-task-a-selection
```

Outputs land at the repo root: `abs_predictions.csv`, `pair_predictions.csv`. See `scripts/pacebench/README.md` for all flags.

### The Pace-Bench selection (the 100 selected instances)

Pace-Bench itself **is** the set of source instances the pipeline selects per
target (100 per target at `--count 100`). It ships ready-made at
[`scripts/pacebench/selections/abs_fit/selections_C100.csv`](scripts/pacebench/selections/abs_fit/selections_C100.csv)
— one row per selected `(target, benchmark, subdir, instance_id)` with the
regression `weight`. The `abs`/`pair` commands above pick instances internally;
this file is that selection materialized (used by the analysis and new-model
tools).

To (re)generate it, or to produce it for a different budget `C`:

```bash
cd scripts/pacebench
python cli.py fit-abs --count 100 --B 300 --seed 42 --strict-budget \
    --dump-selections selections
#   -> selections/abs_fit/selections_C100.csv   (change --count for C=25, 200, ...)
```

`fit-abs` fits on all models in-sample (it is the selection/analysis path, not a
held-out prediction). 

### Reproduce the paper

The two commands above **are** the headline result — they print the LOOCV table
(MAE / Spearman / Pearson for `abs`, pairwise accuracy for `pair`) on the shipped
data with the frozen config, so they reproduce the paper's core numbers
deterministically (seed 42). See `scripts/pacebench/README.md` for the budget
sweep (`C ∈ {25..500}`) and all flags.

## Part 2 — generate scores by running models (needs an API key)

### Install everything in one command

```bash
bash evaluations/install.sh
```

This installs the verified, pinned deps for **all 19 benchmarks**. Because BFCL
hard-conflicts with the rest on `tree-sitter`, it sets up two environments:

- the **current python** → pacebench + 18 benchmarks (`evaluations/requirements-main.txt`
  plus the editable vendored packages lm_eval / lcb_runner / lmms_eval);
- **`evaluations/.bfcl-venv`** → BFCL only.

Override the interpreter with `PYTHON=...` and the BFCL venv with `BFCL_VENV=...`.
(A single flat requirements file for all 19 is impossible — the tree-sitter
conflict cannot be satisfied in one environment.)

**What `install.sh` actually does with the requirements files.** You normally only
run `install.sh`; it orchestrates the files below, so you rarely touch them
directly:

| file | role | who uses it |
|---|---|---|
| `evaluations/requirements-main.txt` | pinned deps for the main env (18 benchmarks, no BFCL) | `install.sh` pip-installs it first |
| `evaluations/benchmarks/{lm-evaluation-harness,livecodebench,lmms-eval}` | vendored packages needing an editable install | `install.sh` runs `pip install -e ... --no-deps` after the above |
| `evaluations/requirements/<benchmark>.txt` | deps for **one** benchmark in isolation | you, only if setting up a single benchmark instead of all |
| `evaluations/requirements-verified.txt` | a frozen record of a known-good version combo | reference / reproducing an exact env — **not installed by `install.sh`** |

So `install.sh` = `requirements-main.txt` + the three editable installs + a
separate `.bfcl-venv`. `requirements-verified.txt` is documentation, not an input
to the installer; `requirements/` is only for the one-benchmark path below.

### Credentials: API key + endpoint

Running models needs an **OpenAI-compatible endpoint** (`base_url`) and an
**API key** for it. You supply your own — nothing is bundled.

- **What endpoint?** Any OpenAI-compatible URL that serves the models you want:
  a [LiteLLM](https://github.com/BerriAI/litellm) proxy (the examples use CMU's
  `https://cmu.litellm.ai`), the OpenAI API (`https://api.openai.com/v1`), Azure
  OpenAI, etc. `--model_name` must be a model id that endpoint exposes
  (e.g. `azure_ai/gpt-5.2`, `anthropic/claude-sonnet-4-5`). List a LiteLLM
  proxy's models with `curl -s "$BASE_URL/v1/models" -H "Authorization: Bearer $API_KEY"`.
- **Where to put it?** The entry points read `API_KEY` / `BASE_URL` from the
  environment. **Do not hardcode keys in tracked files or commit them**
  (`keys.json` and `.env` are git-ignored). Two ways:

  **(recommended) a repo-root `.env` file** — set it once, no re-exporting each
  shell. Copy the template and fill in your key:

  ```bash
  cp .env.example .env     # then edit .env and put your key after API_KEY=
  ```

  The `evaluations/` entry points auto-load this `.env` (`evaluations/_env.py`),
  so once it's filled in you just run them — no flags, no re-exporting.

  Precedence: an explicit `--api_key` / `--base_url` flag **>** a real env var
  (`export`) **>** `.env`. A bare `.env` (no flags) is enough for the **Python**
  entry points `run.py` and `score_new_model.py` — they auto-load it — and you can
  still override per-run with `export` or a flag. **`evaluations/script.sh` is a
  shell script and does not read `.env`**: either `export` the vars, or source the
  file first with `set -a; source .env; set +a`.

### Run one instance

```bash
# API_KEY / BASE_URL come from .env (or the environment); run from the repo root.
python -m evaluations.run \
    --model_name azure_ai/gpt-5.2 \
    --benchmark  infobench \
    --instance_id user_oriented_task_167
```

(You can still pass `--api_key` / `--base_url` explicitly to override `.env` for a
single run.)

`evaluations/script.sh` has ready-to-run examples for all 19 benchmarks (change
the model at the top; run BFCL with `PYTHON=evaluations/.bfcl-venv/bin/python`).

The 19 `--benchmark` names:

| handler | benchmarks |
|---|---|
| lm_eval (editable install) | acp_gen, aime25, gpqa, humaneval_chat, ifeval, logiqa, mbpp_chat, mmlu_cot |
| infobench / lifbench / planbench | infobench, lifbench, planbench |
| livecodebench (editable) / repobench / debugbench | livecodebench, repobench, debugbench |
| lmms_eval (editable) / visualwebbench | mmmu, visualpuzzles, visualwebbench |
| beir / bfcl (separate venv) | beir_nfcorpus, bfcl |

To set up just one benchmark instead of all of them, its deps are in
`evaluations/requirements/<benchmark>.txt`; note lm_eval / livecodebench /
lmms_eval also need `pip install -e evaluations/benchmarks/<repo> --no-deps`.

### Verify all benchmarks (optional)

After install, check that every benchmark runs end-to-end. This runs all 19 on one
example instance each and prints a PASS/FAIL summary — fault-tolerant, so one
failure doesn't stop the rest. **It makes ~19 real API calls (costs money).**

```bash
bash evaluations/verify_benchmarks.sh
```

BFCL (the 19th) is auto-detected at `evaluations/.bfcl-venv`; it shows `SKIP` if that
venv is missing (run `install.sh` to create it). Expected failures without extra
setup: `debugbench` needs `keys.json` (below), and `planbench`'s non-verification
tasks need VAL + Fast Downward.

### debugbench: extra one-time LeetCode setup

debugbench grades fixed code by submitting it to the **LeetCode online judge**,
so it needs your LeetCode session cookies (installed deps `python-leetcode` and
`gym` are already covered by `install.sh`). 

To get the leetcode tokens, do the following steps:

1. Go to leetcode.com and login. If you haven't verified your email (for new accounts), do that before next step.
2. Make sure the URL on the webpage is exactly "https://leetcode.com/". Right click and press inspect. Go to Application -> Storage -> Cookies -> leetcode.com
3. Copy the strings for the LEETCODE_SESSION and csrftoken and paste them here. **Note that you need to keep your leetcode webpage open in your browser all the time while running evaluation.**
4. Put them in `evaluations/benchmarks/debugbench/evaluation/keys.json`
   (a `keys.example.json` template sits next to it):

   ```json
   [{"leetcode_session": "<LEETCODE_SESSION value>", "csrf_token": "<csrftoken value>"}]
   ```

Then debugbench runs like any other benchmark. **Each run makes a real submission
to your LeetCode account** (rate-limited to one per ~25s). Never commit `keys.json`.

## Predict a new model's agentic performance (the main use case)

Goal: get a **new model's predicted scores on the 4 agentic targets** by running
it only on the ~385 cheap Pace-Bench instances — never on the agentic benchmarks
themselves. Two steps: **(1) score the model on the selection**, **(2) predict**.
This uses **Part 2** (to run the model) plus the **Part 1** prediction tools.

**You do not need to build Pace-Bench yourself** — the selection
`scripts/pacebench/selections/abs_fit/selections_C100.csv` ships with the repo, so
you can skip the whole selection-generation pipeline (the `fit-abs` step in Part 1)
and go straight to the two steps below.

Prerequisites: **Part 2 set up above** (to run the model) + credentials exported
(`API_KEY` / `BASE_URL`). `<MODEL>` is the model id on your endpoint
(e.g. `neulab/gemini-2.5-pro`); `<NAME>` is a label for the output files
(e.g. `Gemini-2.5-Pro`).

```bash
# 1. Run the model on every selected instance and write the standardized CSVs
#    it needs, to results/standardized_results/<benchmark>/.../<NAME>.csv.
#    Parallel + resumable; ~30-60 min for the full selection. (from repo root)
python evaluations/score_new_model.py \
    --model <MODEL> --model-name-out <NAME> \
    --selections scripts/pacebench/selections/abs_fit/selections_C100.csv \
    --workers 8   # API_KEY / BASE_URL come from .env (or the environment)

# 2. Predict the 4 agentic-target scores (held-out: fit on the other models).
cd scripts/pacebench
python analysis/predict_new_model.py      --new-model <NAME>            # absolute scores
python analysis/predict_new_model_pair.py --new-model <NAME> --summary   # pairwise win/loss
```

Step 2 prints a table of `predicted` (and, if `<NAME>` also has agentic
ground-truth CSVs, `actual` + `abs_error`) for gaia / swebench /
swebench_multimodal / swtbench. That prediction **is** the new model's estimated
agentic performance, at ~1% of the cost of running the agentic benchmarks.

Preview which instances will be run (optional, no API calls):

```bash
python scripts/pacebench/analysis/emit_eval_plan.py \
    --selections scripts/pacebench/selections/abs_fit/selections_C100.csv --model <MODEL>
```

**Coverage matters.** The prediction is a weighted sum over the selected
instances, so instances you do *not* score count as 0 and pull the prediction
toward the training-set mean. For a faithful estimate, score the whole selection.
`planbench` grading needs VAL + Fast Downward, which require **Linux**, so run the
scoring step on a Linux machine. See
`scripts/pacebench/analysis/README_new_model.md` for details.

## Notes

- **Run as a module.** `python -m evaluations.run ...` from the repo root; and run
  the pipeline from `scripts/pacebench/` (`python cli.py ...`).
- **openai version.** Handlers use the new SDK (`from openai import OpenAI`), so
  `openai>=1.0` is required — an older 0.28 will fail to import.
- **HuggingFace downloads.** Some benchmarks download datasets; export
  `HF_HUB_ENABLE_HF_TRANSFER=0` unless you install `hf_transfer`.
- **BFCL.** install
  it in its own venv (`evaluations/requirements/bfcl.txt`) and point the runner at
  that venv's python.
- **Reasoning models.** Some (e.g. gpt-5) reject `temperature`; the handlers now
  drop it and retry, and enlarge the token budget when a reasoning model truncates
  (`evaluations/handlers/_compat.py`).
- **planbench / debugbench.** planbench reads prompts from
  `results/raw_results/planbench`; debugbench grades via the LeetCode online judge
  and needs cookies in `evaluations/benchmarks/debugbench/evaluation/keys.json`
  (do not commit those).
- **Secrets.** Do not commit API keys or `keys.json`.
