"""Per-target metric computation for Task A (absolute) and Task B (pairwise)."""

import numpy as np
import pandas as pd
from scipy import stats


# ──────────────────────────────────────────────────────────────
# Task A — absolute-score metrics
# ──────────────────────────────────────────────────────────────

def compute_abs_metrics(y_true, y_pred):
    """Per-target scalar metrics for one benchmark: MAE, Spearman, Pearson, R²."""
    mae = float(np.mean(np.abs(y_pred - y_true)))
    sp  = float(stats.spearmanr(y_pred, y_true).correlation)
    pe  = float(stats.pearsonr(y_pred, y_true)[0])
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"MAE": mae, "Spearman": sp, "Pearson": pe, "R2": r2}


def build_abs_metrics_df(preds_all, Y, target_names):
    """DataFrame of per-target absolute metrics."""
    rows = []
    for t_idx, tname in enumerate(target_names):
        m = compute_abs_metrics(Y[:, t_idx], preds_all[:, t_idx])
        m["target"] = tname
        rows.append(m)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Task B — pairwise metrics
# ──────────────────────────────────────────────────────────────

def compute_pair_accuracy_df(pair_margins, Y, target_names):
    """Vectorized pair-sign accuracy per target.

    NaN entries in `pair_margins` (used by the custom-split mode to mark pairs
    that don't involve any eval-set model) are excluded from the count.
    """
    n, _, T = pair_margins.shape
    rows = []
    for t_idx, tname in enumerate(target_names):
        pred_mat = pair_margins[:, :, t_idx]
        true_mat = Y[:, t_idx:t_idx+1] - Y[:, t_idx:t_idx+1].T
        mask = (~np.eye(n, dtype=bool)) & (np.abs(true_mat) >= 1e-9) & ~np.isnan(pred_mat)
        correct = ((pred_mat > 0) == (true_mat > 0)) & mask
        rows.append({
            "target":      tname,
            "acc_overall": float(correct.sum()) / max(int(mask.sum()), 1),
            "n_total":     int(mask.sum()),
        })
    return pd.DataFrame(rows)


def build_abs_metrics_df_eval(y_true_eval, preds_eval, target_names):
    """Same as build_abs_metrics_df but scoped to an arbitrary eval-model subset."""
    rows = []
    for t_idx, tname in enumerate(target_names):
        m = compute_abs_metrics(y_true_eval[:, t_idx], preds_eval[:, t_idx])
        m["target"] = tname
        rows.append(m)
    return pd.DataFrame(rows)


def collect_margins_labels(pair_margins, Y, target_names):
    """Per-target (margins, binary labels) for Platt calibration.  Skips NaN cells."""
    n, _, T = pair_margins.shape
    out = {}
    for t_idx, tname in enumerate(target_names):
        true_mat = Y[:, t_idx:t_idx+1] - Y[:, t_idx:t_idx+1].T
        pred_mat = pair_margins[:, :, t_idx]
        m_list, l_list = [], []
        for k in range(n):
            for j in range(n):
                if j == k: continue
                y_m = float(true_mat[k, j])
                if abs(y_m) < 1e-9: continue
                m_pred = float(pred_mat[k, j])
                if not np.isfinite(m_pred): continue
                m_list.append(m_pred)
                l_list.append(1 if y_m > 0 else 0)
        out[tname] = (np.array(m_list), np.array(l_list))
    return out


# ──────────────────────────────────────────────────────────────
# Pretty-printing
# ──────────────────────────────────────────────────────────────

def print_abs_results(df, label=""):
    """Format abs metrics table to stdout."""
    print(f"\n─────────────────────────────────────────────────────────────────", flush=True)
    print(f"  {label}", flush=True)
    print(f"─────────────────────────────────────────────────────────────────", flush=True)
    hdr = f"{'Target':28s}  {'MAE%':>6}  {'Spearman':>8}  {'Pearson':>7}  {'R²':>6}"
    print(hdr, flush=True); print("─" * len(hdr), flush=True)
    for _, row in df.iterrows():
        print(f"{row['target']:28s}  {row['MAE']*100:5.2f}%  {row['Spearman']:8.3f}"
              f"  {row['Pearson']:7.3f}  {row['R2']:6.3f}", flush=True)
    print("─" * len(hdr), flush=True)
    avg = {c: df[c].mean() for c in ["MAE", "Spearman", "Pearson", "R2"]}
    print(f"{'AVERAGE':28s}  {avg['MAE']*100:5.2f}%  {avg['Spearman']:8.3f}"
          f"  {avg['Pearson']:7.3f}  {avg['R2']:6.3f}", flush=True)


def print_pair_results(df, label=""):
    """Format pair-accuracy table to stdout."""
    print(f"\n─────────────────────────────────────────────────────────────────", flush=True)
    print(f"  {label}", flush=True)
    print(f"─────────────────────────────────────────────────────────────────", flush=True)
    hdr = f"{'Target':28s}  {'Accuracy':>10}  {'N':>5s}"
    print(hdr, flush=True); print("─" * len(hdr), flush=True)
    for _, row in df.iterrows():
        print(f"{row['target']:28s}  {row['acc_overall']*100:9.2f}%  {row['n_total']:5d}",
              flush=True)
    print("─" * len(hdr), flush=True)
    print(f"{'AVERAGE':28s}  {df['acc_overall'].mean()*100:9.2f}%", flush=True)
