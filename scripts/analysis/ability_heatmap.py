#!/usr/bin/env python3
"""
Per-target ability allocation heatmap.

Rows: 4 paper targets. Cols: 11 atomic abilities (from paper/tables/benchmarks.tex).
Cells: how many selected source instances cover that ability.

A single instance contributes once to EACH ability its source benchmark covers
(e.g. a PlanBench instance covers IF + Plan + Reas + Ver → +1 to all four).
Row sums therefore exceed 100 (= |union| × avg-abilities-per-instance).

Usage:
  python scripts/pacebench/analysis/ability_heatmap.py \
      --csv scripts/pacebench/selections/abs_fit/selections_C100.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# Ability codes & display names (from tex header column order)
# ──────────────────────────────────────────────────────────────────────
ABILITIES = [
    ("IF",   "IF"),
    ("LCA",  "LCA"),
    ("ER",   "ER"),
    ("Plan", "Plan"),
    ("Code", "Code"),
    ("IR",   "IR"),
    ("CS",   "CS"),
    ("TC",   "TC"),
    ("Reas", "Reasoning"),
    ("MM",   "MM"),
    ("Ver",  "Verification"),
]
ABILITY_KEYS = [k for k, _ in ABILITIES]

# Source benchmark → set of abilities (from paper/tables/benchmarks.tex bullets).
# "Verification" subsumes the former "Test Understanding" since every Test-annotated
# benchmark in our pool also has Verification (Test ⊂ Ver).
SOURCE_ABILITIES = {
    "acp_gen":        {"IF", "Plan", "Reas", "Ver"},                       # ACPBench
    "aime25":         {"IF", "Reas"},                                      # AIME 2025
    "beir_nfcorpus":  {"IF", "IR"},                                        # BEIR (NFCorpus)
    "bfcl":           {"IF", "TC", "Reas", "Ver"},                         # BFCL
    "debugbench":     {"IF", "ER", "Code", "Reas", "Ver"},                 # DebugBench
    "gpqa":           {"IF", "Reas"},                                      # GPQA
    "humaneval_chat": {"IF", "Code", "Reas", "Ver"},                       # HumanEval
    "ifeval":         {"IF", "Reas"},                                      # IFEval
    "infobench":      {"IF", "Reas"},                                      # InFoBench
    "lifbench":       {"IF", "LCA", "Reas"},                               # LIFBench
    "livecodebench":  {"IF", "ER", "Plan", "Code", "Reas", "Ver"},         # LiveCodeBench
    "logiqa":         {"IF", "Reas"},                                      # LogiQA
    "mbpp_chat":      {"IF", "Code", "Reas", "Ver"},                       # MBPP
    "mmlu_cot":       {"IF", "Reas"},                                      # MMLU
    "mmmu":           {"IF", "Reas", "MM"},                                # MMMU
    "planbench":      {"IF", "Plan", "Reas", "Ver"},                       # PlanBench
    "repobench":      {"IF", "LCA", "Code", "CS", "Reas"},                 # RepoBench-R
    "visualpuzzles":  {"IF", "Reas", "MM"},                                # VisualPuzzles
    "visualwebbench": {"IF", "IR", "Reas", "MM"},                          # VisualWebBench
}

TARGET_ORDER = [
    ("gaia",                "GAIA"),
    ("swebench_multimodal", "SWE-Bench\nMultimodal"),
    ("swebench",            "SWE-Bench\nVerified"),
    ("swtbench",            "SWT-Bench"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="scripts/pacebench/selections/abs_fit/selections_C100.csv",
                    help="Selections CSV (default: selections_C100.csv)")
    ap.add_argument("--out", default="ability_heatmap.pdf",
                    help="Output figure path (default: ability_heatmap.pdf)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["target", "col_idx"])  # dedup H/D-pt overlap

    tgt_keys = [k for k, _ in TARGET_ORDER]
    tgt_disp = [d for _, d in TARGET_ORDER]
    ab_disp  = [d for _, d in ABILITIES]

    # Per-(target, ability) instance count — instances counted once per ability covered
    M = np.zeros((len(tgt_keys), len(ABILITY_KEYS)), dtype=int)
    unmapped = set()
    for ti, t in enumerate(tgt_keys):
        sub = df[df["target"] == t]
        for src, n_inst in sub.groupby("benchmark").size().items():
            abilities = SOURCE_ABILITIES.get(src)
            if abilities is None:
                unmapped.add(src); continue
            for ab in abilities:
                M[ti, ABILITY_KEYS.index(ab)] += int(n_inst)
    if unmapped:
        print(f"(warning: source benchmarks without ability mapping: {sorted(unmapped)})",
              flush=True)

    # ── Plot (mirror allocation_heatmap.py style) ─────────────────
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    masked = np.ma.masked_where(M == 0, M)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="white")
    vmax = int(M.max()) if M.max() > 0 else 1
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(ab_disp)))
    ax.set_xticklabels(ab_disp, fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(tgt_disp)))
    ax.set_yticklabels(tgt_disp, fontsize=10)
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", which="both", length=0)

    threshold = vmax * 0.55
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v == 0: continue
            color = "white" if v >= threshold else "black"
            weight = "bold" if v >= max(int(vmax * 0.5), 1) else "normal"
            ax.text(j, i, f"{v}", ha="center", va="center",
                    color=color, fontsize=10, fontweight=weight)

    for x in np.arange(-0.5, len(ab_disp), 1): ax.axvline(x, color="#888", lw=0.3)
    for y in np.arange(-0.5, len(tgt_disp), 1): ax.axhline(y, color="#888", lw=0.3)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, shrink=0.85)
    cbar.set_label("# selected instances covering ability", fontsize=10)

    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, bbox_inches="tight")
    print(f"Heatmap saved → {out.resolve()}")
    print("\nRow sums (= total ability-instance assignments per target):", flush=True)
    for ti, t in enumerate(tgt_disp):
        print(f"  {t.replace(chr(10), ' '):24s}  {M[ti].sum()}")


if __name__ == "__main__":
    main()
