"""RNA-FM embedding extraction on real Huesken 2005 guide sequences.

The full extraction test needs the rna-fm ('fm') package + torch and is skipped when they
are absent (the lightweight import path never requires them); the validation test runs
without RNA-FM. Guide sequences are read from data/huesken2005.csv (real Huesken 2005
guides, the antisense / silencing strands derived from the OligoFormer Hu.csv).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cascade.embeddings import RNAFM_DIM, extract_embeddings, write_embeddings  # noqa: E402
from cascade.embeddings.rnafm_extract import _clean_sequences  # noqa: E402

HUESKEN_CSV = REPO / "data" / "huesken2005.csv"


def _three_real_guides() -> list[str]:
    if not HUESKEN_CSV.exists():
        pytest.skip(f"real Huesken 2005 guides not present at {HUESKEN_CSV}")
    with HUESKEN_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    guides = [r["guide"].strip().upper() for r in rows[:3]]
    assert len(guides) == 3 and all(19 <= len(g) <= 23 for g in guides)
    return guides


def test_clean_sequences_raises_on_missing_or_invalid() -> None:
    # No synthetic fallback: a missing / empty / non-RNA sequence must RAISE.
    with pytest.raises(ValueError):
        _clean_sequences(["ACGU", ""])
    with pytest.raises(ValueError):
        _clean_sequences(["ACGU", None])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        _clean_sequences(["ACGZ"])  # Z is not an RNA base
    # A valid guide passes and is normalized (T mapped to U, uppercased).
    assert _clean_sequences(["acgt"]) == ["ACGU"]


def test_rnafm_embeddings_on_real_huesken_guides(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("fm")  # the rna-fm package; skip the model test when absent

    guides = _three_real_guides()
    per_nt, pooled = extract_embeddings(guides, labels=["h0", "h1", "h2"])

    assert pooled.shape == (3, RNAFM_DIM)
    assert pooled.dtype == np.float32 and np.all(np.isfinite(pooled))
    assert len(per_nt) == 3
    for emb, g in zip(per_nt, guides, strict=True):
        assert emb.shape == (len(g), RNAFM_DIM)  # one 640-d vector per nucleotide
        assert np.allclose(emb.mean(axis=0), pooled[guides.index(g)], atol=1e-5)

    # Sidecar + .npy round-trip.
    per_nt2, pooled2 = write_embeddings(guides, tmp_path / "huesken_rnafm")
    import json

    side = json.loads((tmp_path / "huesken_rnafm.json").read_text())
    assert side["embedding_dim"] == RNAFM_DIM and side["n_sequences"] == 3
    loaded = np.load(tmp_path / "huesken_rnafm.npy", allow_pickle=True)
    assert loaded.shape == (3,) and loaded[0].shape == (len(guides[0]), RNAFM_DIM)
    assert np.allclose(np.array(side["mean_pooled"], dtype=np.float32), pooled2, atol=1e-5)
