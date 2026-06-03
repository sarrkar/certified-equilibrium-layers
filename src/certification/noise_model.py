"""Confidence-calibrated random-edge noise model on the inferred GRN.

Each edge of the symmetric-normalized adjacency S is perturbed by Gaussian noise
whose scale is calibrated to that edge's confidence: high-confidence edges
(large weight, e.g. a high STRING combined score) barely move, low-confidence
edges are noisy. The perturbed adjacency is kept symmetric, nonnegative, and
spectral-norm-clipped to ||S'||_2 <= 1 (reusing the CASCADE clip), so the
perturbed propagation operator stays contained by construction.

Pure numpy / scipy (no torch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from cascade.verify.containment import spectral_clip


def upper_edges(S: sp.spmatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (row, col, weight) of the strict upper triangle of symmetric S."""
    coo = sp.triu(S, k=1).tocoo()
    return coo.row, coo.col, coo.data


def confidence_from_weights(weights: np.ndarray) -> np.ndarray:
    """Map edge weights to a confidence in [0, 1] (relative to the strongest edge)."""
    mx = float(weights.max()) if weights.size else 1.0
    return weights / (mx + 1e-12)


@dataclass(frozen=True)
class RandomEdgeNoise:
    """Per-edge Gaussian perturbation with edge std = base_std * (1 - confidence).

    ``edge_confidence`` (aligned to the strict-upper-triangle edge order of the
    graph passed to :meth:`perturb`) overrides the weight-derived confidence.
    """

    base_std: float = 0.1
    nonneg: bool = True
    edge_confidence: np.ndarray | None = None

    def perturb(self, S: sp.spmatrix, rng: np.random.Generator) -> sp.csr_matrix:
        """Draw a perturbed, symmetric, spectral-clipped adjacency S'."""
        row, col, w = upper_edges(S)
        conf = (
            self.edge_confidence if self.edge_confidence is not None else confidence_from_weights(w)
        )
        conf = np.clip(np.asarray(conf, dtype=float), 0.0, 1.0)
        std = self.base_std * (1.0 - conf)
        wp = w + std * rng.standard_normal(w.shape[0])
        if self.nonneg:
            wp = np.clip(wp, 0.0, None)
        n = S.shape[0]
        data = np.concatenate([wp, wp])
        rows = np.concatenate([row, col])
        cols = np.concatenate([col, row])
        sp_perturbed = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        clipped, _ = spectral_clip(sp_perturbed)
        return clipped
