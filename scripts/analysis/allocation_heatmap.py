#!/usr/bin/env python3
"""
Per-target source-benchmark allocation heatmap.

Rows: 4 paper targets. Cols: source benchmarks. Cells: # unique instances
selected for that (target, source) pair (deduped across H/D-pt).

Usage:
  python scripts/pacebench/analysis/allocation_heatmap.py \
      --csv scripts/pacebench/selections/abs_fit/selections_C100.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ──────────────────────────────────────────────────────────────────────
# Display ordering (matches paper/tables/benchmarks.tex)
# ──────────────────────────────────────────────────────────────────────
SOURCE_ORDER = [   # (csv_key, display_name)
    ("acp_gen",        "ACPBench"),
    ("aime25",         "AIME"),
    ("beir_nfcorpus",  "BEIR"),
    ("bfcl",           "BFCL"),
    ("debugbench",     "DebugBench"),
    ("gpqa",           "GPQA"),
    ("humaneval_chat", "HumanEval"),
    ("ifeval",         "IFEval"),
    ("infobench",      "InFoBench"),
    ("lifbench",       "LIFBench"),
    ("livecodebench",  "LiveCodeBench"),
    ("logiqa",         "LogiQA"),
    ("mbpp_chat",      "MBPP"),
    ("mmlu_cot",       "MMLU"),
    ("mmmu",           "MMMU"),
    ("planbench",      "PlanBench"),
    ("repobench",      "RepoBench"),
    ("visualpuzzles",  "VisualPuzzles"),
    ("visualwebbench", "VisualWebBench"),
]
TARGET_ORDER = [   # (csv_key, display_name)
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
    ap.add_argument("--out", default="allocation_heatmap.pdf",
                    help="Output figure path (default: allocation_heatmap.pdf)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["target", "col_idx"])

    # Build matrix [n_targets × n_sources]
    src_keys = [k for k, _ in SOURCE_ORDER]
    src_disp = [d for _, d in SOURCE_ORDER]
    tgt_keys = [k for k, _ in TARGET_ORDER]
    tgt_disp = [d for _, d in TARGET_ORDER]
    M = np.zeros((len(tgt_keys), len(src_keys)), dtype=int)
    for ti, t in enumerate(tgt_keys):
        for si, s in enumerate(src_keys):
            M[ti, si] = int(((df["target"] == t) & (df["benchmark"] == s)).sum())

    # ── Plot ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13.5, 3.5))
    # Use Blues colormap; mask zero cells (don't color them, leave white)
    masked = np.ma.masked_where(M == 0, M)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="white")
    vmax = int(M.max()) if M.max() > 0 else 1
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(src_disp)))
    ax.set_xticklabels(src_disp, rotation=35, ha="left", fontsize=10)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(tgt_disp)))
    ax.set_yticklabels(tgt_disp, fontsize=10)
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", which="both", length=0)

    # Cell annotations: skip zeros; flip text color for dark cells
    threshold = vmax * 0.55
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v == 0: continue
            color = "white" if v >= threshold else "black"
            weight = "bold" if v >= max(int(vmax * 0.5), 1) else "normal"
            ax.text(j, i, f"{v}", ha="center", va="center",
                    color=color, fontsize=10, fontweight=weight)

    # Hairline grid between cells
    for x in np.arange(-0.5, len(src_disp), 1): ax.axvline(x, color="#888", lw=0.3)
    for y in np.arange(-0.5, len(tgt_disp), 1): ax.axhline(y, color="#888", lw=0.3)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, shrink=0.85)
    cbar.set_label("# selected instances", fontsize=10)

    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, bbox_inches="tight")
    print(f"Heatmap saved → {out.resolve()}")


if __name__ == "__main__":
    main()
