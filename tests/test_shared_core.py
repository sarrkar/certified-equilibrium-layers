"""Shared-core contract check (CLAUDE.md "Shared core contract").

The two theory projects depend on these imports from ``cascade``. This module
confirms the implemented pieces import and run, and marks each tracked gap with
a visible xfail-strict test: the test stays XFAIL while the gap is open and
flips to XPASS (a strict failure that forces marker removal) the moment the gap
is implemented, so a gap cannot close silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

FIXTURE = Path(__file__).parent / "fixtures" / "string_tiny.txt"


# --------------------------------------------------------------------------
# Implemented core: must import and work.
# --------------------------------------------------------------------------
def test_p_alpha_operator_numpy_importable_and_invertible() -> None:
    from cascade.verify.containment import dense_propagation_operator, propagate

    s = sp.csr_matrix(np.array([[0.0, 0.5], [0.5, 0.0]]))
    p = dense_propagation_operator(s, alpha=0.2)
    assert p.shape == (2, 2)
    assert abs(np.linalg.det(p)) > 0.0  # KNOWN invertible mixing operator
    # The dense operator equals the converged power-iteration application.
    h = np.array([1.0, 0.0])
    assert np.allclose(p @ h, propagate(s, h, alpha=0.2, num_steps=200), atol=1e-6)


def test_p_alpha_operator_torch_importable() -> None:
    pytest.importorskip("torch")
    from cascade.models.propagation_head import PropagationHead

    assert PropagationHead is not None


def test_string_graph_builder_works(cache_root: Path) -> None:
    from cascade.data import load_string_grn

    grn = load_string_grn(FIXTURE, cache_root=cache_root, threshold=700)
    assert grn.num_nodes() == 4
    assert sp.issparse(grn.S)


def test_gse184929_loader_importable() -> None:
    from cascade.data import load_gse184929

    assert callable(load_gse184929)


def test_seedmatch_logic_importable() -> None:
    from cascade.features.seedmatch import find_sites

    assert callable(find_sites)


# --------------------------------------------------------------------------
# Tracked gaps: visible xfail-strict markers (remove when implemented).
# --------------------------------------------------------------------------
@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="GEARS-based Replogle 2022 loader not wired yet (cascade.data.load_replogle2022)",
)
def test_gap_gears_replogle2022() -> None:
    from cascade.data import load_replogle2022

    ds = load_replogle2022()
    assert len(ds) > 0


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="GEARS-based Norman 2019 loader not wired yet (cascade.data.load_norman2019)",
)
def test_gap_gears_norman2019() -> None:
    from cascade.data import load_norman2019

    ds = load_norman2019()
    assert len(ds) > 0


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="Reactome graph builder not implemented yet (cascade.data.load_reactome_graph)",
)
def test_gap_reactome_graph_builder() -> None:
    from cascade.data import load_reactome_graph

    grn = load_reactome_graph()
    assert grn.num_nodes() > 0
