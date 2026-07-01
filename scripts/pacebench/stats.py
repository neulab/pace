"""Numerical utilities: Spearman, OLS, sigmoid, pair-index helpers."""

import numpy as np
from scipy.stats import rankdata


def spearman_X_vs_Y(X, Y):
    """
    Spearman correlation between each column of X and each column of Y.

    X : (n, p) array
    Y : (n, q) array
    → returns (p, q) float32 matrix of pairwise rank correlations.
    """
    n = X.shape[0]
    Xr = rankdata(X, axis=0).astype(np.float64)
    Yr = rankdata(Y, axis=0).astype(np.float64)
    Xr -= Xr.mean(axis=0); Yr -= Yr.mean(axis=0)
    Xr /= (Xr.std(axis=0) + 1e-12); Yr /= (Yr.std(axis=0) + 1e-12)
    return (Xr.T @ Yr / n).astype(np.float32)


def ols_1d(x_tr, y_tr, x_te):
    """
    Univariate OLS with intercept: fit on (x_tr, y_tr), predict scalar at x_te.

    Degenerate cases (near-zero x_tr variance, or fit that produces a wild
    extrapolation at x_te) fall back to y_tr.mean(). This guards against
    numerical blow-up in small-sample / low-signal regimes (e.g. when the
    caller restricts to a single source benchmark with few informative dims).
    """
    x_tr = np.asarray(x_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    xm, ym = x_tr.mean(), y_tr.mean()
    x_std = x_tr.std()
    # Scale-aware degeneracy check: relative std below 1e-8 → treat as constant.
    if x_std < 1e-8 * max(abs(xm), 1e-12):
        return float(ym)
    var = ((x_tr - xm) ** 2).sum()
    if var < 1e-12:
        return float(ym)
    a = ((x_tr - xm) * (y_tr - ym)).sum() / var
    pred = float(a * (x_te - xm) + ym)
    # Sanity guard: clamp wild extrapolations to a generous Y-range envelope.
    y_min = float(y_tr.min()); y_max = float(y_tr.max())
    y_range = max(y_max - y_min, 1e-6)
    lo = y_min - 2.0 * y_range
    hi = y_max + 2.0 * y_range
    if not np.isfinite(pred):
        return float(ym)
    if pred < lo or pred > hi:
        return float(ym)
    return pred


def sigmoid(z):
    """Numerically stable sigmoid on a 1-D array."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z     = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def pair_indices(n_tr):
    """Return (pair_a, pair_b) index arrays for all directed off-diagonal pairs i ≠ j."""
    a, b = [], []
    for i in range(n_tr):
        for j in range(n_tr):
            if i != j:
                a.append(i); b.append(j)
    return np.array(a), np.array(b)
