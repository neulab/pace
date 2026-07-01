"""LOMO plots: Task A predicted-vs-actual scatter, Task B pair scatter."""

from pathlib import Path

import numpy as np

from pacebench.config import MODELS


_SHORT_NAMES = [
    m.replace("Claude-", "Cl-").replace("Gemini-3-", "Ge-")
     .replace("DeepSeek-", "DS-").replace("MiniMax-", "MM-")
     .replace("Qwen3-Coder-480B-A35B-Instruct", "Qwen3-480B")
     .replace("-Preview", "") for m in MODELS
]


def plot_abs_loocv_fit(Y_true, Y_pred, target_names, metrics_df, out_path, title):
    """Task A scatter: 2×2 grid of predicted vs actual for each target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    T = len(target_names); nrows = (T + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 9))
    axes = axes.flatten()
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for t_idx, tname in enumerate(target_names):
        ax = axes[t_idx]
        y  = Y_true[:, t_idx]; yh = Y_pred[:, t_idx]
        for j, (yj, yhj, name) in enumerate(zip(y, yh, _SHORT_NAMES)):
            ax.scatter(yhj, yj, color=palette[j % len(palette)], s=55, zorder=3)
            ax.annotate(name, (yhj, yj), textcoords="offset points",
                        xytext=(4, 3), fontsize=6.5,
                        color=palette[j % len(palette)])
        xmin, xmax = yh.min(), yh.max()
        pad = (xmax - xmin) * 0.12 or 0.02
        xs = np.linspace(xmin - pad, xmax + pad, 100)
        xm, ym = yh.mean(), y.mean()
        var_x = ((yh - xm) ** 2).sum()
        beta  = ((yh - xm) * (y - ym)).sum() / (var_x + 1e-12)
        ax.plot(xs, beta * (xs - xm) + ym, color="steelblue", linewidth=1.8, label="OLS fit")
        lo = min(xmin - pad, y.min() - pad); hi = max(xmax + pad, y.max() + pad)
        ax.plot([lo, hi], [lo, hi], color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
        row = metrics_df[metrics_df["target"] == tname].iloc[0]
        txt = (f"MAE={row['MAE']*100:.2f}%\nSpearman={row['Spearman']:.3f}\n"
               f"Pearson={row['Pearson']:.3f}\nR²={row['R2']:.3f}")
        ax.text(0.04, 0.97, txt, transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))
        ax.set_xlabel("Predicted score", fontsize=9)
        ax.set_ylabel("Actual score", fontsize=9)
        ax.set_title(tname, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    for k in range(T, len(axes)):
        axes[k].set_visible(False)

    legend_els = [Line2D([0], [0], color="steelblue", lw=1.8, label="OLS fit"),
                  Line2D([0], [0], color="gray", lw=1.0, ls="--", label="y = x")]
    fig.legend(handles=legend_els, loc="lower right", fontsize=9,
               bbox_to_anchor=(0.98, 0.02))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {out_path}", flush=True)


def plot_pair_scatter(pair_margins, Y, target_names, out_path, title):
    """Task B scatter: 2×2 grid of predicted margin vs true margin, coloured by correctness."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(target_names); n = pair_margins.shape[0]
    nrows = (T + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 9))
    axes = axes.flatten()

    for t_idx, tname in enumerate(target_names):
        ax = axes[t_idx]
        pred = pair_margins[:, :, t_idx]
        true = Y[:, t_idx:t_idx+1] - Y[:, t_idx:t_idx+1].T
        mask = (~np.eye(n, dtype=bool)) & (np.abs(true) >= 1e-9)
        y_m = true[mask]; p_m = pred[mask]
        ok = ((p_m > 0) == (y_m > 0))
        ax.scatter(y_m[ok]*100,  p_m[ok],  s=12, c='tab:green', alpha=0.6,
                   label=f"correct ({ok.sum()})")
        ax.scatter(y_m[~ok]*100, p_m[~ok], s=20, c='tab:red', alpha=0.85,
                   edgecolor='black', linewidth=0.4,
                   label=f"wrong ({(~ok).sum()})")
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.axvspan(-2, 2, alpha=0.08, color='orange')
        acc = ok.mean()
        ax.set_title(f"{tname}  Acc={acc*100:.1f}%  N={mask.sum()}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("True margin (%)", fontsize=9)
        ax.set_ylabel("Predicted margin (logit)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)

    for k in range(T, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPair scatter saved → {out_path}", flush=True)
