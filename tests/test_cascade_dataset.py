"""Tests for the CASCADE dataset assembly (pure numpy, no torch)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cascade.features import concept_layout
from cascade.synthetic_dataset import build_cascade_data

FIXTURE = Path(__file__).parent / "fixtures" / "string_tiny.txt"


def _fixture_cfg(cache_root: Path, mode: str = "standard") -> dict:
    return {
        "graph": str(FIXTURE),
        "threshold": 700,
        "cache": str(cache_root),
        "synthetic": True,
        "synthetic_alpha": 0.2,
        "synthetic_noise": 0.05,
        "split": {"mode": mode, "val_frac": 0.25, "test_frac": 0.25, "n_genes": 2},
    }


def test_build_on_fixture_graph(cache_root: Path) -> None:
    data = build_cascade_data(_fixture_cfg(cache_root), seed=0)
    assert data.num_nodes == 4
    assert data.num_features == concept_layout().total_dim == 37
    assert data.feature_names == concept_layout().feature_names
    assert data.labels.shape == (4,)
    assert np.isfinite(data.labels).all()
    assert np.isfinite(data.features).all()


def test_standard_split_is_a_partition(cache_root: Path) -> None:
    data = build_cascade_data(_fixture_cfg(cache_root), seed=0)
    tr, va, te = data.train_mask, data.val_mask, data.test_mask
    # Every node assigned exactly once.
    assert np.all((tr.astype(int) + va.astype(int) + te.astype(int)) == 1)
    assert tr.any() and va.any() and te.any()


def test_held_out_gene_split_is_gene_disjoint() -> None:
    cfg = {
        "graph": "synthetic",
        "n_nodes": 24,
        "edge_p": 0.35,
        "synthetic": True,
        "split": {"mode": "held_out_gene", "val_frac": 0.25, "test_frac": 0.25, "n_genes": 6},
    }
    data = build_cascade_data(cfg, seed=1)
    g = data.gene
    train_g = set(g[data.train_mask].tolist())
    val_g = set(g[data.val_mask].tolist())
    test_g = set(g[data.test_mask].tolist())
    # No gene leaks across splits.
    assert train_g.isdisjoint(val_g)
    assert train_g.isdisjoint(test_g)
    assert val_g.isdisjoint(test_g)
    # Partition over nodes.
    assert np.all(
        (data.train_mask.astype(int) + data.val_mask.astype(int) + data.test_mask.astype(int)) == 1
    )


def test_reproducible_for_same_seed(cache_root: Path) -> None:
    a = build_cascade_data(_fixture_cfg(cache_root), seed=7)
    b = build_cascade_data(_fixture_cfg(cache_root), seed=7)
    assert np.array_equal(a.labels, b.labels)
    assert np.array_equal(a.features, b.features)
    assert np.array_equal(a.train_mask, b.train_mask)


def test_non_synthetic_without_table_raises(cache_root: Path) -> None:
    cfg = _fixture_cfg(cache_root)
    cfg["synthetic"] = False
    cfg["node_table"] = None
    with pytest.raises(ValueError, match="node_table"):
        build_cascade_data(cfg, seed=0)
