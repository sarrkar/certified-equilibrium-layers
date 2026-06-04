"""Tests for the synthetic-coverage ensembles (role: synthetic_coverage)."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.data import SYNTHETIC_COVERAGE, make_synthetic_coverage


def test_matching_is_disjoint() -> None:
    ds = make_synthetic_coverage("synthetic_matching", n_edges=4, confidence=0.6)
    assert ds.role == SYNTHETIC_COVERAGE
    assert ds.n_nodes == 8 and ds.n_edges == 4
    deg = np.asarray(ds.confidence.sum(axis=1)).ravel()
    # A matching: every node touches at most one edge (max degree value = one conf).
    assert np.allclose(deg, 0.6)


def test_star_centers_all_edges() -> None:
    ds = make_synthetic_coverage("synthetic_star", n_edges=5, confidence=0.5)
    assert ds.n_nodes == 6 and ds.n_edges == 5
    deg = np.asarray(ds.confidence.sum(axis=1)).ravel()
    assert deg[0] == pytest.approx(5 * 0.5)  # center holds every edge
    assert np.allclose(deg[1:], 0.5)  # leaves have one edge each


def test_erdos_renyi_builds_symmetric() -> None:
    ds = make_synthetic_coverage("synthetic_erdos_renyi", n_edges=10, edge_prob=0.5, seed=1)
    assert ds.role == SYNTHETIC_COVERAGE
    assert ds.n_nodes == 20
    dense = ds.confidence.toarray()
    assert np.allclose(dense, dense.T)
    assert dense.min() >= 0.0 and dense.max() <= 1.0


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        make_synthetic_coverage("synthetic_bogus")
