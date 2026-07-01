"""Platt scaling — affine recalibration of margin scores to calibrated probabilities."""

import numpy as np
from scipy.optimize import minimize

from pacebench.stats import sigmoid


def fit_platt(margins, labels):
    """
    Fit 2-parameter Platt scaling: sigmoid(a · margin + b) ≈ P(y = 1).

    Returns (a, b) via L-BFGS-B minimizing negative log-likelihood.
    """
    margins = np.asarray(margins, dtype=np.float64)
    labels  = np.asarray(labels,  dtype=np.float64)
    def nll(p):
        a, b = p
        z  = a * margins + b
        lp = np.where(z >= 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
        return float(np.sum(lp - labels * z))
    res = minimize(nll, x0=[1.0, 0.0], method='L-BFGS-B')
    return float(res.x[0]), float(res.x[1])


def apply_platt(margins, a, b):
    """Convert margin scores → calibrated probabilities via sigmoid(a · margins + b)."""
    return sigmoid(a * np.asarray(margins, dtype=np.float64) + b)
