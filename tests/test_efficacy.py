"""Tests for the unified OligoFormer efficacy loader (role: efficacy_backbone)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cascade.data import EFFICACY_BACKBONE, EfficacyDataset, load_oligoformer_efficacy
from cascade.data.efficacy import COLUMNS


def _raw(fixtures_dir: Path) -> Path:
    return fixtures_dir / "oligoformer"


def test_load_unified_suite(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_oligoformer_efficacy(_raw(fixtures_dir), cache_root=cache_root)
    assert isinstance(ds, EfficacyDataset)
    assert ds.role == EFFICACY_BACKBONE
    assert list(ds.frame.columns) == COLUMNS
    assert len(ds) == 8  # 3 huesken + 2 katoh + 2 mix + 1 shabalina
    # Pooled provenance: Huesken, Katoh, the two mix studies, and Shabalina.
    assert {
        "Huesken 2005",
        "Katoh (Taka)",
        "Reynolds 2004",
        "Ui-Tei 2004",
        "Shabalina 2006",
    } <= set(ds.sources())
    assert "H1299" in ds.cell_lines()


def test_efficacy_is_normalized_to_unit_interval(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_oligoformer_efficacy(_raw(fixtures_dir), cache_root=cache_root)
    eff = ds.frame["efficacy"].to_numpy()
    assert eff.min() >= 0.0 and eff.max() <= 1.0
    # Katoh fixture is percent-scaled (85.0 -> 0.85).
    katoh = ds.frame[ds.frame["source_study"] == "Katoh (Taka)"]["efficacy"].to_numpy()
    assert np.isclose(katoh.max(), 0.85)


def test_subset_sources(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_oligoformer_efficacy(_raw(fixtures_dir), cache_root=cache_root, sources=["huesken"])
    assert ds.sources() == ["Huesken 2005"]
    assert len(ds) == 3


def test_filter_study(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_oligoformer_efficacy(_raw(fixtures_dir), cache_root=cache_root)
    only = ds.filter_study("Reynolds 2004")
    assert len(only) == 1
    assert only.sources() == ["Reynolds 2004"]


def test_cache_is_idempotent(fixtures_dir: Path, cache_root: Path) -> None:
    first = load_oligoformer_efficacy(_raw(fixtures_dir), cache_root=cache_root)
    second = load_oligoformer_efficacy(_raw(fixtures_dir), cache_root=cache_root)
    assert first.frame.equals(second.frame)
    assert list(cache_root.glob("efficacy/*/frame.parquet"))


def test_missing_dir_raises(cache_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_oligoformer_efficacy("/no/such/dir", cache_root=cache_root)
