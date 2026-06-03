"""Observation model: mix latents by the KNOWN P_alpha, then an unknown map G.

The concept latents Z are mixed by the known invertible propagation operator
P_alpha = alpha (I - (1-alpha) S)^{-1} (reused from the CASCADE core, never
re-implemented here) and then observed through an unknown invertible linear map
G. The observation is

    Y = G P_alpha Z = W Z,   W = G P_alpha.

Because P_alpha is known, recovering the mixing W (up to permutation and
scaling, the identifiable equivalence class) isolates the unknown-G content.

Pure numpy (no torch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from cascade.verify.containment import dense_propagation_operator


def random_invertible(d: int, rng: np.random.Generator, scale: float = 0.4) -> np.ndarray:
    """A well-conditioned random invertible matrix (I + scale * noise)."""
    for _ in range(100):
        g = np.eye(d) + scale * rng.standard_normal((d, d))
        if np.linalg.cond(g) < 1e6:
            return g
    raise RuntimeError("failed to draw a well-conditioned invertible matrix")


@dataclass(frozen=True)
class ObservationModel:
    """Y = G P_alpha Z. Holds the known P_alpha and the unknown G; W = G P_alpha."""

    P_alpha: np.ndarray  # KNOWN invertible mixing (from the CASCADE operator)
    G: np.ndarray  # unknown invertible observation map

    @property
    def W(self) -> np.ndarray:
        return self.G @ self.P_alpha

    def observe(self, Z: np.ndarray) -> np.ndarray:
        """Map latents Z [n, d] to observations Y [n, d] = (W Z^T)^T."""
        return Z @ self.W.T

    @classmethod
    def from_graph(
        cls,
        S: sp.spmatrix,
        alpha: float,
        rng: np.random.Generator,
        g_scale: float = 0.4,
    ) -> ObservationModel:
        """Build the known P_alpha from a graph S and draw a random unknown G."""
        p_alpha = dense_propagation_operator(S, alpha)
        g = random_invertible(p_alpha.shape[0], rng, scale=g_scale)
        return cls(P_alpha=p_alpha, G=g)
