"""Identifiability metrics: mean correlation coefficient and Amari distance.

Both are invariant to the permutation and per-coordinate scaling (and sign)
ambiguity that interventional identifiability leaves, so they measure recovery
up to the identifiable affine equivalence class.

Pure numpy / scipy (no torch).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def mean_correlation_coefficient(z_true: np.ndarray, z_hat: np.ndarray) -> float:
    """Mean absolute correlation under the best matching of recovered to true sources.

    1.0 means perfect recovery up to permutation, sign, and scale.
    """
    d = z_true.shape[1]
    corr = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            c = np.corrcoef(z_true[:, i], z_hat[:, j])[0, 1]
            corr[i, j] = 0.0 if not np.isfinite(c) else abs(c)
    row, col = linear_sum_assignment(-corr)
    return float(corr[row, col].mean())


def amari_distance(w_true: np.ndarray, w_hat: np.ndarray) -> float:
    """Amari distance between two mixing matrices (0 = equal up to perm/scale).

    Standard ICA metric on P = W_hat^{-1} W_true.
    """
    d = w_true.shape[0]
    p = np.abs(np.linalg.inv(w_hat) @ w_true)
    row_term = (p.sum(axis=1) / (p.max(axis=1) + 1e-12) - 1.0).sum()
    col_term = (p.sum(axis=0) / (p.max(axis=0) + 1e-12) - 1.0).sum()
    return float((row_term + col_term) / (2 * d * (d - 1)))
