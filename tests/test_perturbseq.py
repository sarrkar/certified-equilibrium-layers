"""Tests for the Perturb-seq stub loaders."""

from __future__ import annotations

import pytest

from cascade.data import PerturbSeqDataset, load_norman2019, load_replogle2022


def test_dataclass_shape() -> None:
    ds = PerturbSeqDataset(name="replogle2022")
    assert ds.name == "replogle2022"
    assert len(ds) == 0
    assert ds.expression is None


def test_replogle_stub_raises() -> None:
    with pytest.raises(NotImplementedError, match="follow-up commit"):
        load_replogle2022()


def test_norman_stub_raises() -> None:
    with pytest.raises(NotImplementedError, match="follow-up commit"):
        load_norman2019()
