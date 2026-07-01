"""Per-fold embedding construction for H and D-pt strategies.

Both strategies produce a `(U_tr, u_te)` pair where:
  U_tr : (n_tr, nc) — embeddings of the training models
  u_te : (nc,)      — embedding of the held-out model projected onto the same basis

The two strategies differ only in how the subspace is constructed:
  * H  : per-fold SVD on the training models' sub-matrix over selected instances.
  * D-pt: pseudoinverse against the global SVD basis Vt_full on selected instances.
"""

import numpy as np
from numpy.linalg import svd as np_svd, pinv

from pacebench.selection import h_top_M, dpt_top_N


def h_embed(X_valid, i, X_tr, y_tr, M, nc, pinned_sel=None):
    """
    H embedding: per-target top-M selection → per-fold SVD → project held-out.

    If `pinned_sel` is provided, selection is skipped and the given column indices
    are used directly (useful for Task B reusing Task A's selections).

    Returns (U_tr, u_te) with nc_effective = min(nc, n_tr).
    """
    top_M_idx = pinned_sel if pinned_sel is not None else h_top_M(X_tr, y_tr, M)
    X_sub = X_tr[:, top_M_idx]
    U_w, S_w, Vt_w = np_svd(X_sub, full_matrices=False)
    nc_   = min(nc, U_w.shape[1])
    U_tr  = U_w[:, :nc_]
    S_k   = S_w[:nc_] + 1e-12
    Vt_k  = Vt_w[:nc_, :]
    u_te  = X_valid[i, top_M_idx].astype(np.float64) @ Vt_k.T / S_k
    return U_tr, u_te


def dpt_embed(X_valid, i, X_tr, y_tr, Vt_full, N, nc, pinned_sel=None):
    """
    D-pt embedding: per-target score-based selection → pseudoinverse → embed all 14 models.

    If `pinned_sel` is provided, selection is skipped; the given indices are used
    with the supplied `nc` (which now controls only the embedding subspace, not
    the selection score).

    Returns (U_tr, u_te). U_tr is rows of the pseudoinverse embedding restricted to
    training models; u_te is the held-out model's row.
    """
    sel = pinned_sel if pinned_sel is not None else dpt_top_N(X_tr, y_tr, Vt_full, N, nc)
    V_sel = Vt_full[:nc, :][:, sel].T
    U_hat = X_valid[:, sel].astype(np.float64) @ pinv(V_sel).T   # (n, nc)
    tr = [j for j in range(X_valid.shape[0]) if j != i]
    return U_hat[tr], U_hat[i]


def expand_to_all_models(U_tr, u_te, tr, i, n):
    """
    Reassemble a (n, nc) embedding table: training rows at tr, held-out row at i.
    Useful for pair prediction where we need u_k - u_j across all j.
    """
    nc = U_tr.shape[1]
    u_all = np.zeros((n, nc))
    u_all[tr] = U_tr
    u_all[i]  = u_te
    return u_all


# ──────────────────────────────────────────────────────────────
# "Fit-all" variants: use all n models (no held-out) for in-sample analysis
# ──────────────────────────────────────────────────────────────

def h_embed_all(X_all, y_all, M, nc, pinned_sel=None):
    """H embedding from all n models. Returns U_all of shape (n, nc_effective)."""
    from pacebench.selection import h_top_M
    top_M_idx = pinned_sel if pinned_sel is not None else h_top_M(X_all, y_all, M)
    X_sub = X_all[:, top_M_idx]
    U_w, _, _ = np_svd(X_sub, full_matrices=False)
    nc_ = min(nc, U_w.shape[1])
    return U_w[:, :nc_]


def dpt_embed_all(X_all, y_all, Vt_full, N, nc, pinned_sel=None):
    """D-pt embedding from all n models. Returns U_all of shape (n, nc)."""
    from pacebench.selection import dpt_top_N
    sel = pinned_sel if pinned_sel is not None else dpt_top_N(X_all, y_all, Vt_full, N, nc)
    V_sel = Vt_full[:nc, :][:, sel].T
    return X_all[:, sel].astype(np.float64) @ pinv(V_sel).T


# ──────────────────────────────────────────────────────────────
# Split variants: arbitrary train set → embed ALL models + return selection
# ──────────────────────────────────────────────────────────────

def h_embed_split(X_valid, train_idx, y_tr, M, nc, pinned_sel=None):
    """
    H embedding with an arbitrary training set. Uses X_valid[train_idx] for selection
    (unless pinned_sel given) and SVD basis; projects ALL n models onto that basis
    via the pseudoinverse formula  X[:, sel] @ Vt_k.T / S_k.

    Returns (U_all, top_M_idx) where U_all has shape (n, nc_effective).
    """
    X_tr = X_valid[train_idx].astype(np.float64)
    top_M_idx = pinned_sel if pinned_sel is not None else h_top_M(X_tr, y_tr, M)
    X_sub = X_tr[:, top_M_idx]
    _, S_w, Vt_w = np_svd(X_sub, full_matrices=False)
    nc_  = min(nc, Vt_w.shape[0])
    S_k  = S_w[:nc_] + 1e-12
    Vt_k = Vt_w[:nc_, :]
    U_all = X_valid[:, top_M_idx].astype(np.float64) @ Vt_k.T / S_k
    return U_all, top_M_idx


def dpt_embed_split(X_valid, train_idx, y_tr, Vt_full, N, nc, pinned_sel=None):
    """
    D-pt embedding with an arbitrary training set. Uses X_valid[train_idx] + Vt_full
    for selection (unless pinned_sel given); projects ALL n models via pinv(V_sel).

    Returns (U_all, sel).
    """
    from pacebench.selection import dpt_top_N
    X_tr = X_valid[train_idx].astype(np.float64)
    sel = pinned_sel if pinned_sel is not None else dpt_top_N(X_tr, y_tr, Vt_full, N, nc)
    V_sel = Vt_full[:nc, :][:, sel].T
    U_all = X_valid[:, sel].astype(np.float64) @ pinv(V_sel).T
    return U_all, sel
