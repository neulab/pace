## Summary: `script.sh` and `bootstrap_all.py`

### Flowchart

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           script.sh                                      │
│  Orchestrates bootstrap analysis for proxy benchmark discovery          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Define benchmark groups:                                                │
│    SOURCES (non-agentic): acp_gen, aime25, bfcl, gpqa, humaneval_chat,  │
│                           ifeval, livecodebench, logiqa, mbpp_chat      │
│    TARGETS (agentic):     swtbench (or swebench, gaia, commit0, etc.)   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────┐
         │  For each TARGET benchmark:                       │
         │    Run: bootstrap_all.py                          │
         │      --sources [all non-agentic]                  │
         │      --targets [single agentic target]            │
         └──────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        bootstrap_all.py                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        ▼                                                       ▼
┌───────────────────┐                               ┌───────────────────┐
│ MERGE SOURCES     │                               │ MERGE TARGETS     │
│ humaneval_chat +  │                               │ swebench          │
│ gpqa + bfcl + ... │                               │ (instances from   │
│                   │                               │  target benchmark)│
│ → Prefixed IDs:   │                               │ → Prefixed IDs:   │
│ "gpqa::q_123"     │                               │ "swebench::t_456" │
│ "bfcl::f_789"     │                               │                   │
└───────────────────┘                               └───────────────────┘
        │                                                       │
        └───────────────────────┬───────────────────────────────┘
                                ▼
                ┌───────────────────────────────┐
                │ Find COMMON MODELS            │
                │ (models present in all        │
                │  source AND target benchmarks)│
                └───────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │ Build matrices:               │
                │  A_src: [models × src_insts]  │
                │  A_tgt: [models × tgt_insts]  │
                └───────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │ Split target instances:       │
                │  train_cols (400 instances)   │
                │  eval_cols  (100 instances)   │
                └───────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────────┐
        │ GREEDY SUBSET SELECTION (vote over bootstraps)    │
        │                                                   │
        │ For n_outer iterations:                           │
        │   1. Bootstrap sample target train instances      │
        │   2. Greedy select k_source SOURCE instances      │
        │      that maximize correlation with target        │
        │   3. Vote: count how often each source instance   │
        │      gets selected                                │
        │                                                   │
        │ Output: Top k_source instances by vote count      │
        └───────────────────────────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │ EVALUATE on held-out eval set │
                │  Pearson & Spearman corr      │
                └───────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │ OUTPUT:                       │
                │  - Scatter plots              │
                │  - JSON with selected IDs     │
                │  - Correlation metrics        │
                └───────────────────────────────┘
```

---

### Example Run

```bash
# script.sh calls:
python bootstrap_all.py \
    --sources acp_gen aime25 bfcl gpqa humaneval_chat ifeval livecodebench logiqa mbpp_chat \
    --targets swtbench \
    --train_size 400 --eval_size 100 \
    --boot_source_k 1000 --boot_target_k 100 \
    --k_source 200
```

**What happens:**

1. **Merge Sources**: Combine ~50,000 instances from 9 non-agentic benchmarks
   - IDs become: `gpqa::q_0`, `bfcl::simple_python_13`, `humaneval_chat::HumanEval_42`, etc.

2. **Load Target**: Load ~500 instances from swtbench
   - IDs: `swtbench::django__django-12345`, etc.

3. **Find Common Models**: e.g., `{GPT-5.2, Claude-4.5-Opus, Gemini-2.5-Flash, ...}` (maybe 15 models)

4. **Greedy Selection**: Find 200 source instances whose average score best correlates with swtbench performance

5. **Output**: `selected_source_instances.json` containing:
   ```json
   {
     "selected_source_instance_ids": [
       "livecodebench::LC_problem_42",
       "gpqa::diamond_q_17", 
       "bfcl::multiple_35",
       ...
     ],
     "corr_eval": {"pearson": 0.85, "spearman": 0.82}
   }
   ```

**Key Insight**: The selected 200 instances from non-agentic benchmarks can serve as a "proxy" to predict model performance on agentic tasks like swtbench.