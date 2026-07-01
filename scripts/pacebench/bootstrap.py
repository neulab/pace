"""Pooled target-instance bootstrap (shared by Task A and Task B).

Key idea: instead of bootstrapping over the 13 training models, we bootstrap over
target-benchmark instances — accounting for Y-label sampling noise without adding
variance to an already-stable n=13 estimator.

Two variants:
  pooled_absolute : returns (U_pool, y_pool) for absolute-score regression.
  pooled_pairwise : returns (D_pool, dy_pool) for pair-margin classification.

Ablation hook: setting environment variable PROXYBENCH_NO_BOOTSTRAP=1 globally
disables bootstrapping (returns plain mean instead of B resamples). Default
behavior is unchanged.
"""

import os
import numpy as np


def _no_bootstrap_env():
    return os.environ.get("PROXYBENCH_NO_BOOTSTRAP", "").strip() in ("1", "true", "True")


def pooled_absolute(U_tr, P_t, B, rng, bootstrap=True):
    """
    U_tr : (n_tr, nc) training-model embeddings.
    P_t  : (n_tr, n_tgt_inst) training models' per-instance target scores.
    bootstrap : if False, ignore B and use the true mean P_t.mean(axis=1) once
                (no resampling), giving (U_pool, y_pool) of shape (n_tr, ·).

    Returns:
      U_pool : (B · n_tr, nc)  — U_tr tiled B times (or n_tr × nc if bootstrap=False).
      y_pool : (B · n_tr,)     — B bootstraps of the n_tr-dim Y_{Train_k, t}.
    """
    n_tr  = U_tr.shape[0]
    n_tgt = P_t.shape[1]
    if (not bootstrap) or _no_bootstrap_env():
        return U_tr.copy(), P_t.mean(axis=1)
    U_pool = np.tile(U_tr, (B, 1))
    y_pool = np.zeros(B * n_tr)
    for b in range(B):
        bi_t = rng.choice(n_tgt, n_tgt, replace=True)
        y_pool[b*n_tr:(b+1)*n_tr] = P_t[:, bi_t].mean(axis=1)
    return U_pool, y_pool


def pooled_pairwise(U_tr, P_t, B, rng, pair_idx_a, pair_idx_b, bootstrap=True):
    """
    U_tr       : (n_tr, nc) training-model embeddings.
    P_t        : (n_tr, n_tgt_inst) training models' per-instance target scores.
    pair_idx_a, pair_idx_b : flat index arrays enumerating all directed pairs i ≠ j.
    bootstrap  : if False, ignore B and use true pair-margins P_t.mean - P_t.mean once.

    Returns:
      D_pool  : (B · n_pairs, nc)  — pair-diff matrix tiled B times.
      dy_pool : (B · n_pairs,)     — B bootstraps of pair-margin labels.
    """
    n_pairs = len(pair_idx_a)
    n_tgt = P_t.shape[1]
    D_pairs = U_tr[pair_idx_a] - U_tr[pair_idx_b]
    if (not bootstrap) or _no_bootstrap_env():
        y_true = P_t.mean(axis=1)
        return D_pairs.copy(), y_true[pair_idx_a] - y_true[pair_idx_b]
    D_pool = np.tile(D_pairs, (B, 1))
    dy_pool = np.zeros(B * n_pairs)
    for b in range(B):
        bi_t = rng.choice(n_tgt, n_tgt, replace=True)
        y_b = P_t[:, bi_t].mean(axis=1)
        dy_pool[b*n_pairs:(b+1)*n_pairs] = y_b[pair_idx_a] - y_b[pair_idx_b]
    return D_pool, dy_pool
