"""Global constants and frozen per-target production configurations."""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Data paths
# ──────────────────────────────────────────────────────────────

# Per-instance standardized score CSVs live under <repo>/results/standardized_results.
# Override with the PROXYBENCH_BASE_DIR env var; otherwise locate the bundled
# results dir relative to this file (config.py is at <repo>/scripts/pacebench/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(
    os.environ.get(
        "PROXYBENCH_BASE_DIR",
        _REPO_ROOT / "results" / "standardized_results",
    )
)

# ──────────────────────────────────────────────────────────────
# Models and benchmarks
# ──────────────────────────────────────────────────────────────

MODELS = [
    "Claude-4.5-Opus", "Claude-4.5-Sonnet", "Claude-4.6-Opus",
    "DeepSeek-V3.2", "GLM-4.7", "GPT-5.2", "GPT-5.2-Codex",
    "Gemini-3-Flash-Preview", "Gemini-3-Pro-Preview", "Kimi-K2",
    "Kimi-K2.5", "MiniMax-M2.1", "MiniMax-M2.5",
    "Qwen3-Coder-480B-A35B-Instruct",
]
N_MODELS = len(MODELS)

TARGET_DATASETS  = {"gaia", "swebench", "swebench_multimodal", "swtbench"}
EXCLUDE_DATASETS = {"commit0", "swebench_bash_only"}

# ──────────────────────────────────────────────────────────────
# Default hyperparameters
# ──────────────────────────────────────────────────────────────

N_JOBS        = 14          # LOMO folds run in parallel
COUNT         = 100         # uniform source-instance count per target
DEFAULT_B     = 300         # pooled target-instance bootstrap draws
DEFAULT_SEED  = 42
DEFAULT_ALPHA = 0.1         # ridge regularization for pairwise logistic

# ──────────────────────────────────────────────────────────────
# Frozen per-target production configurations, indexed by per-strategy
# instance count C.  Each entry is the best (m, nc_h, nc_d) — plus `reg` for
# Task B — found via exhaustive sweep at that count.
#
# Format: prediction = m · H(nc_h) + (1 - m) · D-pt(nc_d)
# Sweep space: m ∈ {0.0, 0.1, ..., 1.0};  nc_h, nc_d ∈ {1, ..., 10}
#              (Task B additionally picks reg ∈ {ols, logit}).
# Source pool: 19 fully-evaluated benchmarks (humaneval / mbpp / mmlu removed).
#
# To add a new count C: run `python scripts/pacebench/sweep_count_task_a.py
# --count C` and paste the printed dict here.  (Analogous sweep for Task B.)
# ──────────────────────────────────────────────────────────────

# Per-target dict schema:
#   m, nc_h, nc_d : required (mixing weight, embedding dims for L/G strategies)
#   reg          : Task B only — "logit" or "ols"
#   q            : OPTIONAL float in [0, 1]. If present and the user does not
#                  pass --q on the CLI, this target uses q · count instances
#                  for Local-SVD and (1 - q) · count for Global-SVD (joint
#                  budget). If absent, both strategies use full `count`
#                  (effective up to 2·count, the default behavior). q can vary
#                  per target; the runtime caches LOMO predictions by (nc, M)
#                  so different targets with different q add at most |unique q
#                  values| extra LOMO calls per nc.

TASK_A_CONFIGS = {
    100: {   # previous canonical
        "gaia": dict(m=0.2, nc_h= 1, nc_d= 5, q=0.2),
        "swebench": dict(m=0.1, nc_h= 7, nc_d= 7, q=0.8),
        "swebench_multimodal": dict(m=0.9, nc_h=10, nc_d= 4, q=0.9),
        "swtbench": dict(m=0.7, nc_h= 8, nc_d= 3, q=0.9),
    },
    150: {   # budget-sweep optimum (new canonical)
        "gaia":                dict(m=0.3, nc_h=10, nc_d=5),    # MAE 6.11% / Sp 0.849
        "swebench":            dict(m=0.0, nc_h=1,  nc_d=2),    # MAE 2.04% / Sp 0.407
        "swebench_multimodal": dict(m=0.2, nc_h=10, nc_d=8),    # MAE 2.21% / Sp 0.870
        "swtbench":            dict(m=0.9, nc_h=6,  nc_d=3),    # MAE 4.71% / Sp 0.925
    },
    50: {
        "gaia":                dict(m=0.5, nc_h=2,  nc_d=5),    # MAE 6.13% / Sp 0.752
        "swebench":            dict(m=0.7, nc_h=1,  nc_d=2),    # MAE 2.45% / Sp 0.559
        "swebench_multimodal": dict(m=0.7, nc_h=8,  nc_d=2),    # MAE 2.53% / Sp 0.764
        "swtbench":            dict(m=0.7, nc_h=4,  nc_d=7),    # MAE 6.27% / Sp 0.741
    },
}

TASK_B_CONFIGS = {
    100: {   # canonical / paper numbers — all logit
        "gaia":                dict(m=0.9, nc_h=7,  nc_d=3,  reg="logit"),   # 87.78%
        "swebench":            dict(m=0.8, nc_h=10, nc_d=2,  reg="logit"),   # 85.39%
        "swebench_multimodal": dict(m=0.8, nc_h=9,  nc_d=4,  reg="logit"),   # 85.96%
        "swtbench":            dict(m=0.7, nc_h=5,  nc_d=10, reg="logit"),   # 88.46%
    },
    # 50: {...}  # TODO: populate via a Task B sweep at C=50
}

# Task B pinned-to-A variant: optimal (m, nc_h, nc_d, reg) when Task B's selection
# is locked to Task A's (H selection coincides; D-pt selection uses
# TASK_A_CONFIGS[count][target]["nc_d"] as the selector). Use via
# `pair --pin-task-a-selection` in cli.py. Tuned on LOMO pair accuracy.
TASK_B_CONFIGS_PINNED_A = {
    100: {
        "gaia":                dict(m=0.7, nc_h=3,  nc_d=7,  reg="logit"),   # 87.22%
        "swebench":            dict(m=0.9, nc_h=1,  nc_d=10, reg="logit"),   # 83.71%
        "swebench_multimodal": dict(m=0.9, nc_h=9,  nc_d=3,  reg="logit"),   # 84.83%
        "swtbench":            dict(m=0.2, nc_h=6,  nc_d=10, reg="logit"),   # 87.36%
    },
    150: {   # budget-sweep optimum (new canonical)
        "gaia":                dict(m=0.8, nc_h=9,  nc_d=6,  reg="logit"),   # 85.56%
        "swebench":            dict(m=0.0, nc_h=1,  nc_d=3,  reg="logit"),   # 84.83%
        "swebench_multimodal": dict(m=0.9, nc_h=10, nc_d=9,  reg="logit"),   # 89.89%
        "swtbench":            dict(m=0.3, nc_h=7,  nc_d=1,  reg="logit"),   # 90.11%
    },
    50: {
        "gaia":                dict(m=0.9, nc_h=1,  nc_d=8,  reg="logit"),   # 87.22%
        "swebench":            dict(m=0.1, nc_h=3,  nc_d=2,  reg="logit"),   # 88.20%
        "swebench_multimodal": dict(m=0.9, nc_h=10, nc_d=2,  reg="logit"),   # 84.27%
        "swtbench":            dict(m=0.9, nc_h=6,  nc_d=3,  reg="logit"),   # 92.31%
    },
}


def _resolve_count_config(configs_by_count, count, label):
    """Return configs_by_count[count]; on miss, fall back to nearest count with a warning."""
    if count in configs_by_count:
        return configs_by_count[count]
    if not configs_by_count:
        raise ValueError(f"{label}_CONFIGS is empty")
    nearest = min(configs_by_count.keys(), key=lambda c: (abs(c - count), c))
    print(f"(warning: no tuned {label} config for count={count}; "
          f"using nearest count={nearest})", flush=True)
    return configs_by_count[nearest]


def get_task_a_config(count=COUNT):
    """Look up the Task A per-target config tuned for the given count."""
    return _resolve_count_config(TASK_A_CONFIGS, count, "TASK_A")


def get_task_b_config(count=COUNT):
    """Look up the Task B per-target config tuned for the given count."""
    return _resolve_count_config(TASK_B_CONFIGS, count, "TASK_B")


def get_task_b_config_pinned_a(count=COUNT):
    """Look up the Task B (pinned-to-A) per-target config tuned for the given count."""
    return _resolve_count_config(TASK_B_CONFIGS_PINNED_A, count, "TASK_B_PINNED_A")


# Backwards-compat aliases (= the canonical COUNT=100 table). Callers that still
# import TASK_A_CONFIG / TASK_B_CONFIG keep working; new code should use
# get_task_?_config(count) to pick the count-specific config.
TASK_A_CONFIG = TASK_A_CONFIGS[COUNT]
TASK_B_CONFIG = TASK_B_CONFIGS[COUNT]
