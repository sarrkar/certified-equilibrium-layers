"""Synthetic random-graph ensembles for coverage / tightness (role: synthetic_coverage).

These carry no biological claim. They are the confidence-matrix ensembles the
AAAI certification experiments use to verify coverage (matchings), tightness
(stars), and the symmetric-normalized breakdown (Erdos-Renyi). Each builder
returns a :class:`SyntheticCoverageDataset` tagged with the synthetic_coverage
role so it can never be confused with the measured efficacy or off-target sets.

Pure numpy / scipy (no torch).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .roles import SYNTHETIC_COVERAGE

KINDS = ("synthetic_matching", "synthetic_star", "synthetic_erdos_renyi")


@dataclass(frozen=True)
class SyntheticCoverageDataset:
    """A synthetic confidence matrix (mean adjacency) for coverage experiments."""

    name: str
    confidence: sp.csr_matrix
    role: str = field(default=SYNTHETIC_COVERAGE)

    @property
    def n_nodes(self) -> int:
        return self.confidence.shape[0]

    @property
    def n_edges(self) -> int:
        return int(self.confidence.nnz // 2)


def _symmetric(rows, cols, data, n) -> sp.csr_matrix:
    r = np.concatenate([rows, cols])
    c = np.concatenate([cols, rows])
    d = np.concatenate([data, data])
    return sp.coo_matrix((d, (r, c)), shape=(n, n)).tocsr()


def make_synthetic_coverage(
    kind: str,
    n_edges: int = 8,
    confidence: float = 0.5,
    edge_prob: float = 0.5,
    lo: float = 0.4,
    hi: float = 1.0,
    seed: int = 0,
) -> SyntheticCoverageDataset:
    """Build a synthetic confidence matrix of the requested ``kind``.

    - ``synthetic_matching``: ``n_edges`` disjoint edges (every binary subgraph
      has spectral norm <= 1; the coverage regime).
    - ``synthetic_star``: ``n_edges`` edges all incident to one center node (the
      tightness regime where the variance proxy equals the edge-variance sum).
    - ``synthetic_erdos_renyi``: random graph on ``2 * n_edges`` nodes with edge
      probability ``edge_prob`` and confidences ~ U(lo, hi) (the breakdown regime).
    """
    rng = np.random.default_rng(seed)
    if kind == "synthetic_matching":
        n = 2 * n_edges
        rows = np.arange(0, n, 2)
        cols = rows + 1
        data = np.full(n_edges, float(confidence))
        return SyntheticCoverageDataset(kind, _symmetric(rows, cols, data, n))
    if kind == "synthetic_star":
        n = n_edges + 1
        rows = np.zeros(n_edges, dtype=int)
        cols = np.arange(1, n_edges + 1)
        data = np.full(n_edges, float(confidence))
        return SyntheticCoverageDataset(kind, _symmetric(rows, cols, data, n))
    if kind == "synthetic_erdos_renyi":
        n = 2 * n_edges
        iu = np.triu_indices(n, 1)
        present = rng.random(iu[0].shape[0]) < edge_prob
        conf = rng.uniform(lo, hi, iu[0].shape[0]) * present
        keep = conf > 0
        return SyntheticCoverageDataset(kind, _symmetric(iu[0][keep], iu[1][keep], conf[keep], n))
    raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
