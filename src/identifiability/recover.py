"""Recover the mixing and latents from interventional data.

Exact base case (independent latent sources, B = 0, perfect single-gene hard
interventions). With Y = W Z and Z independent, the observational covariance is
C0 = W diag(var Z) W^T. A hard intervention that changes only the variance of
source i gives

    C_i - C0 = (var_i' - var_i) w_i w_i^T,

a rank-one matrix whose leading eigenvector is the i-th column of W (up to sign
and scale). Collecting the columns recovers W up to permutation and scaling,
hence the latents Z = W^{-1} Y up to the same affine equivalence. This is the
identifiable equivalence class the project targets.

The general dense-SCM and soft-combinatorial cases are the open research targets
(a visible xfail-strict test tracks the gap).

Pure numpy (no torch).
"""

from __future__ import annotations

import numpy as np


def _top_eig(symmetric: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the (signed) largest-magnitude eigenvalue and its eigenvector."""
    vals, vecs = np.linalg.eigh(symmetric)
    k = int(np.argmax(np.abs(vals)))
    return float(vals[k]), vecs[:, k]


def recover_mixing(y_obs: np.ndarray, y_interventions: dict[int, np.ndarray]) -> np.ndarray:
    """Recover W [d, d] from observational and per-node interventional samples.

    ``y_interventions[i]`` are observations under a hard intervention on node i
    (changing its source variance). Column i of the returned W is recovered up
    to sign and scale.
    """
    d = y_obs.shape[1]
    c0 = np.cov(y_obs, rowvar=False)
    w = np.zeros((d, d))
    for i in range(d):
        if i not in y_interventions:
            raise ValueError(f"missing interventional samples for node {i}")
        delta = np.cov(y_interventions[i], rowvar=False) - c0
        val, vec = _top_eig(delta)
        # Scale the column by sqrt|eigenvalue| so W is well-conditioned; sign and
        # scale are unidentifiable anyway (handled by MCC / Amari).
        w[:, i] = vec * np.sqrt(abs(val) + 1e-12)
    return w


def recover_latents(y: np.ndarray, w_hat: np.ndarray) -> np.ndarray:
    """Recover latents Z_hat [n, d] = (W_hat^{-1} Y^T)^T."""
    return y @ np.linalg.inv(w_hat).T


def recover_observation_map(w_hat: np.ndarray, p_alpha: np.ndarray) -> np.ndarray:
    """Recover G_hat = W_hat P_alpha^{-1} using the KNOWN P_alpha.

    G_hat equals G up to the residual per-column sign/scale ambiguity in W_hat
    transported through P_alpha^{-1}; the clean invariant is the latent MCC.
    """
    return w_hat @ np.linalg.inv(p_alpha)
