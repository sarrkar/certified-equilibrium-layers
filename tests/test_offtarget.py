"""Tests for the off-target loaders (role: offtarget_certification)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade.data import (
    OFFTARGET_CERTIFICATION,
    OffTargetDataset,
    load_birmingham2006,
    load_jackson2003,
    load_jackson2006,
    load_seedseq,
)
from cascade.data import offtarget as ot
from cascade.data.offtarget import COLUMNS


def _ot(fixtures_dir: Path, name: str) -> Path:
    return fixtures_dir / "offtarget" / f"{name}.csv"


def test_jackson2003_seed_derived_from_antisense(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_jackson2003(_ot(fixtures_dir, "jackson2003"), cache_root=cache_root)
    assert isinstance(ds, OffTargetDataset)
    assert ds.role == OFFTARGET_CERTIFICATION
    assert ds.assay == "microarray"
    assert list(ds.frame.columns) == COLUMNS
    assert len(ds) == 3
    # No seed column => derived from antisense positions 2-8.
    assert ds.frame["seed"].iloc[0] == "GAGGUAG"
    assert "supplement" in ds.provenance
    assert ds.sources() == ["jackson2003"]  # defaulted to the dataset name


def test_jackson2006_uses_explicit_seed(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_jackson2006(_ot(fixtures_dir, "jackson2006"), cache_root=cache_root)
    assert set(ds.frame["seed"]) == {"GAGGUAG"}
    assert ds.transcripts() == ["NM_000010", "NM_000011"]


def test_birmingham2006_source_column(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_birmingham2006(_ot(fixtures_dir, "birmingham2006"), cache_root=cache_root)
    assert ds.sources() == ["birmingham2006"]
    assert len(ds) == 2


def test_seedseq_aggregated_has_source_flags(fixtures_dir: Path, cache_root: Path) -> None:
    ds = load_seedseq(_ot(fixtures_dir, "seedseq"), cache_root=cache_root)
    assert ds.sources() == ["birmingham2006", "jackson2003", "jackson2006"]
    assert len(ds) == 3


def test_seedseq_union_fallback(
    fixtures_dir: Path, cache_root: Path, tmp_path, monkeypatch
) -> None:
    # No aggregated seedseq table => build the union from available members,
    # tagging each call with its source.
    def fake_default(name: str) -> Path:
        if name == "seedseq":
            return tmp_path / "absent_seedseq.csv"
        return _ot(fixtures_dir, name)

    monkeypatch.setattr(ot, "_default_path", fake_default)
    ds = load_seedseq(cache_root=cache_root)
    assert set(ds.sources()) == {"jackson2003", "jackson2006", "birmingham2006"}
    assert "built from available members" in ds.provenance


def test_missing_file_raises(cache_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_jackson2003("/no/such/jackson.csv", cache_root=cache_root)
