"""Latent linear structural causal model with hard and soft interventions.

The concept latents Z follow a linear SCM Z = B Z + e, with B strictly lower
triangular (a DAG in topological order) and independent noise e. The reduced
form is Z = (I - B)^{-1} e, and (I - B)^{-1} is the total-effect matrix.

Interventions (the interventional data for identifiability):

- hard single-gene do(Z_i = noise): a perfect intervention that cuts the
  incoming edges of node i (zeros row i of B) and resets its noise term. The
  base identifiable case is B = 0, where Z is a vector of independent sources.
- soft combinatorial: shift the mean and/or scale the variance of the noise on
  a subset of nodes, leaving the structure B intact.

Pure numpy (no torch).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class LatentLinearSCM:
    """Linear SCM Z = B Z + e with independent Gaussian noise.

    ``B`` is (d, d) strictly lower triangular. ``noise_std`` and ``noise_mean``
    are length-d. Sampling returns Z with rows as samples.
    """

    B: np.ndarray
    noise_std: np.ndarray
    noise_mean: np.ndarray | None = None

    @property
    def dim(self) -> int:
        return self.B.shape[0]

    def total_effect_matrix(self) -> np.ndarray:
        """Return (I - B)^{-1}, the linear total causal effect matrix."""
        d = self.dim
        return np.linalg.inv(np.eye(d) - self.B)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` latent samples Z [n, d]."""
        d = self.dim
        mean = np.zeros(d) if self.noise_mean is None else self.noise_mean
        e = mean + self.noise_std * rng.standard_normal((n, d))
        f = self.total_effect_matrix()
        return e @ f.T

    def intervene_hard(
        self, node: int, value_std: float = 1.0, value_mean: float = 0.0
    ) -> LatentLinearSCM:
        """Perfect do(Z_node = N(value_mean, value_std^2)): cut parents, reset noise."""
        b = self.B.copy()
        b[node, :] = 0.0  # remove incoming edges to the intervened node
        std = self.noise_std.copy()
        std[node] = value_std
        mean = np.zeros(self.dim) if self.noise_mean is None else self.noise_mean.copy()
        mean[node] = value_mean
        return replace(self, B=b, noise_std=std, noise_mean=mean)

    def intervene_soft(
        self,
        nodes: Sequence[int],
        mean_shift: float = 1.0,
        var_scale: float = 1.0,
    ) -> LatentLinearSCM:
        """Soft (structure-preserving) intervention: shift / scale the noise on ``nodes``.

        Combinatorial interventions pass several nodes. B is left intact.
        """
        std = self.noise_std.copy()
        mean = np.zeros(self.dim) if self.noise_mean is None else self.noise_mean.copy()
        for node in nodes:
            std[node] *= np.sqrt(var_scale)
            mean[node] += mean_shift
        return replace(self, B=self.B.copy(), noise_std=std, noise_mean=mean)


def random_dag_scm(
    d: int,
    rng: np.random.Generator,
    edge_density: float = 0.0,
    coef_scale: float = 0.8,
    noise_std_range: tuple[float, float] = (0.5, 1.5),
) -> LatentLinearSCM:
    """Random linear SCM. ``edge_density`` 0 gives independent sources (B = 0)."""
    b = np.zeros((d, d))
    if edge_density > 0:
        for i in range(d):
            for j in range(i):  # strictly lower triangular => DAG
                if rng.random() < edge_density:
                    b[i, j] = coef_scale * rng.standard_normal()
    lo, hi = noise_std_range
    noise_std = rng.uniform(lo, hi, size=d)
    return LatentLinearSCM(B=b, noise_std=noise_std)
