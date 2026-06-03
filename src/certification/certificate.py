"""Matrix-Bernstein certified L2 region for the diffusion-propagated prediction.

The certified ball is centered at the plug-in mean-graph prediction
y_plugin = P_alpha(A_bar) x0 (computed by :func:`certification.smoothed`), where
A_bar = E[A] is the STRING confidence matrix. This module computes the certified
radius around that center.

Edge decomposition (un-normalized case S(A) = A). The structure perturbation
A - A_bar decomposes into independent zero-mean self-adjoint per-edge matrices:
for undirected edge e = (i, j), toggling it contributes

    W_e = (A_ij - c_ij) (E_ij + E_ji),

with E[A_ij - c_ij] = 0 and Var(A_ij) = c_ij (1 - c_ij). Using the identity
(E_ij + E_ji)^2 = E_ii + E_jj, the matrix-variance proxy is

    v = || sum_e c_ij (1-c_ij) (E_ij + E_ji)^2 ||_op
      = max_i  sum_{j ~ i} c_ij (1 - c_ij),

a diagonal operator, and the uniform per-edge operator-norm bound is
L = max_e || W_e ||_op = max_e max(c_ij, 1 - c_ij) (since || E_ij + E_ji ||_op = 1).

Matrix Bernstein. With probability at least 1 - delta,

    || A - A_bar ||_op <= b(delta),   b(delta) = sqrt(2 v log(2n/delta)) + (2L/3) log(2n/delta),

the standard closed-form upper bound for the inverse of the Bernstein tail
2 n exp( - b^2 / (2 (v + L b / 3)) ) = delta.

Resolvent conversion (correct constant). The algebraic resolvent identity

    P_alpha(A) - P_alpha(A_bar)
        = ((1-alpha)/alpha) P_alpha(A) (S(A) - S(A_bar)) P_alpha(A_bar),

together with || P_alpha ||_op = 1 (the eigenvalues of P_alpha lie in
[alpha/(2-alpha), 1] when the operator is spectrally bounded by 1), gives

    radius(delta) = ((1-alpha)/alpha) * b(delta) * || x0 ||_2.

The two operator-norm factors are each 1 and are already accounted for, so the
constant is (1-alpha)/alpha, NOT (2-alpha)/alpha.

A Monte-Carlo certificate (:func:`monte_carlo_certificate`) gives a practically
computable confidence interval on the empirical smoothed estimator y_bar, in the
style of randomized smoothing (Cohen, Rosenfeld, Kolter 2019).

Pure numpy / scipy (no torch).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.stats import binom

from .noise_model import EdgeBernoulliNoise


def edge_variance_contributions(
    noise: EdgeBernoulliNoise,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Per-edge variance c_ij (1 - c_ij) over the candidate (upper-triangle) edges."""
    triu = sp.triu(noise.C, k=1).tocoo()
    edges = list(zip(triu.row.tolist(), triu.col.tolist(), strict=True))
    c = triu.data
    return edges, c * (1.0 - c)


def variance_proxy(noise: EdgeBernoulliNoise) -> tuple[float, np.ndarray]:
    """Matrix-variance proxy v and the per-node variance accumulation.

    Since (E_ij + E_ji)^2 = E_ii + E_jj, the variance matrix is diagonal with
    node entry d_i = sum_{j ~ i} c_ij (1 - c_ij), and v = ||diag(d)||_op = max_i d_i.
    """
    n = noise.n_nodes
    triu = sp.triu(noise.C, k=1).tocoo()
    var_e = triu.data * (1.0 - triu.data)
    node_var = np.zeros(n)
    np.add.at(node_var, triu.row, var_e)
    np.add.at(node_var, triu.col, var_e)
    v = float(node_var.max()) if n > 0 and node_var.size else 0.0
    return v, node_var


def uniform_edge_bound(noise: EdgeBernoulliNoise) -> float:
    """Uniform per-edge operator-norm bound L = max_e max(c_ij, 1 - c_ij)."""
    c = sp.triu(noise.C, k=1).tocoo().data
    if c.size == 0:
        return 0.0
    return float(np.maximum(c, 1.0 - c).max())


def p_alpha_eigenvalue_range(alpha: float) -> tuple[float, float]:
    """Eigenvalue range of P_alpha for a spectrally-bounded operator: [alpha/(2-alpha), 1]."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha / (2.0 - alpha), 1.0


def bernstein_b(delta: float, v: float, L: float, n: int) -> float:
    """Closed-form matrix-Bernstein high-probability bound on ||A - A_bar||_op."""
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    log_term = math.log(2.0 * n / delta)
    return math.sqrt(2.0 * v * log_term) + (2.0 * L / 3.0) * log_term


def bernstein_radius(
    delta: float, v: float, L: float, n: int, alpha: float, x0_norm: float
) -> float:
    """Certified L2 radius ((1-alpha)/alpha) b(delta) ||x0|| (correct constant)."""
    return (1.0 - alpha) / alpha * bernstein_b(delta, v, L, n) * x0_norm


@dataclass(frozen=True)
class BernsteinCertificate:
    """Matrix-Bernstein certified radius and its components.

    The certified ball is { y : || y - y_plugin ||_2 <= radius }, centered at the
    plug-in mean-graph prediction y_plugin = P_alpha(A_bar) x0.
    """

    radius: float
    b: float
    v: float
    L: float
    eig_min: float
    eig_max: float
    p_alpha_op_norm: float  # = eig_max = 1.0
    per_edge_variance: np.ndarray
    edges: list[tuple[int, int]]
    node_variance: np.ndarray
    n_nodes: int
    alpha: float
    delta: float
    x0_norm: float

    def b_at(self, delta: float) -> float:
        """b(delta) as a function of the confidence level."""
        return bernstein_b(delta, self.v, self.L, self.n_nodes)

    def radius_at(self, delta: float) -> float:
        """radius(delta) as a function of the confidence level."""
        return bernstein_radius(delta, self.v, self.L, self.n_nodes, self.alpha, self.x0_norm)


def bernstein_certificate(
    x0: np.ndarray, noise: EdgeBernoulliNoise, alpha: float, delta: float
) -> BernsteinCertificate:
    """Build the matrix-Bernstein certificate for the random-subgraph noise model.

    Returns the certified ``radius`` (and ``radius_at`` / ``b_at`` as functions of
    delta), the Bernstein bound ``b``, the variance proxy ``v``, the uniform edge
    bound ``L``, the P_alpha eigenvalue range, and the per-edge variance
    contributions.
    """
    n = noise.n_nodes
    edges, per_edge_var = edge_variance_contributions(noise)
    v, node_var = variance_proxy(noise)
    L = uniform_edge_bound(noise)
    eig_min, eig_max = p_alpha_eigenvalue_range(alpha)
    x0_norm = float(np.linalg.norm(np.asarray(x0, dtype=float)))

    b = bernstein_b(delta, v, L, n)
    radius = (1.0 - alpha) / alpha * b * x0_norm

    return BernsteinCertificate(
        radius=radius,
        b=b,
        v=v,
        L=L,
        eig_min=eig_min,
        eig_max=eig_max,
        p_alpha_op_norm=eig_max,
        per_edge_variance=per_edge_var,
        edges=edges,
        node_variance=node_var,
        n_nodes=n,
        alpha=alpha,
        delta=delta,
        x0_norm=x0_norm,
    )


# ---------------------------------------------------------------------------
# Monte-Carlo certificate for the empirical smoothed estimator (Cohen et al. 2019).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MonteCarloCertificate:
    """Confidence intervals on the empirical smoothed estimator from N samples."""

    y_bar: np.ndarray  # Monte-Carlo mean
    y_median: np.ndarray  # Monte-Carlo median
    mean_ci_low: np.ndarray  # Hoeffding interval on the mean (union over coords)
    mean_ci_high: np.ndarray
    mean_halfwidth: np.ndarray
    median_ci_low: np.ndarray  # nonparametric (binomial / Clopper-Pearson) median CI
    median_ci_high: np.ndarray
    n_samples: int
    delta: float
    value_radius: float | None


def monte_carlo_certificate(
    samples: np.ndarray, delta: float, value_radius: float | None = None
) -> MonteCarloCertificate:
    """Confidence intervals on y_bar / y_median from N ensemble samples.

    ``samples`` is [N, d] with each row a draw y(A) = P_alpha(S(A)) x0. The mean
    interval is a Hoeffding bound (union-bounded over the d coordinates at joint
    confidence 1 - delta); ``value_radius`` is a bound on |y_i| giving the
    per-sample support [-value_radius, value_radius] (when omitted, the empirical
    per-coordinate range is used, which is a practical approximation). The median
    interval is the distribution-free order-statistic (sign-test / Clopper-Pearson
    style) interval. Mirrors the practically-computable certificates of
    randomized smoothing (Cohen, Rosenfeld, Kolter 2019).
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    samples = np.asarray(samples, dtype=float)
    n, d = samples.shape
    if n < 2:
        raise ValueError("need at least 2 samples for a confidence interval")

    y_bar = samples.mean(axis=0)
    y_median = np.median(samples, axis=0)

    # Hoeffding interval on the mean (Bonferroni union bound over coordinates).
    log_term = math.log(2.0 * d / delta)
    if value_radius is not None:
        rng_width = np.full(d, 2.0 * value_radius)
    else:
        rng_width = samples.max(axis=0) - samples.min(axis=0)
    halfwidth = rng_width * math.sqrt(log_term / (2.0 * n))
    mean_ci_low = y_bar - halfwidth
    mean_ci_high = y_bar + halfwidth

    # Nonparametric two-sided CI on the median via binomial order statistics.
    lo_idx = int(np.clip(binom.ppf(delta / 2.0, n, 0.5), 0, n - 1))
    hi_idx = int(np.clip(binom.isf(delta / 2.0, n, 0.5), 0, n - 1))
    ordered = np.sort(samples, axis=0)
    median_ci_low = ordered[lo_idx]
    median_ci_high = ordered[hi_idx]

    return MonteCarloCertificate(
        y_bar=y_bar,
        y_median=y_median,
        mean_ci_low=mean_ci_low,
        mean_ci_high=mean_ci_high,
        mean_halfwidth=halfwidth,
        median_ci_low=median_ci_low,
        median_ci_high=median_ci_high,
        n_samples=n,
        delta=delta,
        value_radius=value_radius,
    )
