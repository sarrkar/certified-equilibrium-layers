"""Regression test for the backbone efficacy benchmark.

The experiment module is import-safe without torch (torch is imported only inside
the train function), so the helpers and the committed JSON are checked in CI; the
end-to-end run is torch-gated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "aaai"))

import exp_backbone_efficacy as ex  # noqa: E402

BACKBONE_JSON = REPO / "results" / "aaai" / "backbone_efficacy.json"


def test_reported_baselines_are_cited_not_fabricated() -> None:
    # Every baseline carries a source citation; only transcribed values are set,
    # the rest are null (not invented).
    for b in ex.REPORTED_BASELINES:
        assert b["source"] and isinstance(b["source"], str)
        assert b["reported_value"] is None or isinstance(b["reported_value"], int | float)
        assert b["metric"] == "pearson"


def test_target_defaults_to_perfect_complement() -> None:
    from cascade.features.seedmatch import revcomp

    g = "UGAGGUAGUAGGUUGUAUAGU"
    assert ex._target_for(g, "") == revcomp(g)  # on-target site when no sense given
    assert ex._target_for(g, "ACGU") == "ACGU"  # provided sense used as-is


def test_synthetic_datasets_have_expected_keys_and_finite_features() -> None:
    feats, labels, provenance = ex._synthetic_datasets(np.random.default_rng(0))
    assert provenance == "synthetic_proxy"
    assert set(feats) == {"huesken_train", *ex.DATASETS}
    for key, x in feats.items():
        assert x.shape[1] == ex.concept_layout().total_dim
        assert np.isfinite(x).all()
        assert labels[key].min() >= 0.0 and labels[key].max() <= 1.0


def test_committed_backbone_json() -> None:
    d = json.loads(BACKBONE_JSON.read_text())
    assert d["provenance"] in {"synthetic_proxy", "real_oligoformer"}
    assert set(d["our_model"]) == set(ex.DATASETS)
    # Pearson reported for every dataset.
    for key in ex.DATASETS:
        assert "pearson" in d["our_model"][key]
    acc = d["acceptance"]
    assert acc["flag"] in {"PASS", "FAIL", "PENDING"}
    # On the synthetic proxy the comparison to real SOTA is not valid -> PENDING.
    if d["provenance"] == "synthetic_proxy":
        assert acc["valid_comparison"] is False
        assert acc["flag"] == "PENDING"
    # BIOPREDsi is the transcribed reference point.
    names = {b["name"]: b["reported_value"] for b in d["reported_baselines"]}
    assert names["BIOPREDsi (Huesken ANN)"] == 0.66


def test_end_to_end_smoke(tmp_path) -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    # Tiny featurized datasets so the smoke run is fast.
    feats, labels = {}, {}
    from cascade.features.seedmatch import revcomp

    for key in ["huesken_train", *ex.DATASETS]:
        guides = ex._random_guides(20, 0.0, rng)
        feats[key] = ex._featurize(guides, [revcomp(g) for g in guides])
        labels[key] = rng.random(20)
    res = ex._train_and_eval(feats, labels, seed=0, epochs=5, lr=0.01)
    assert "per_dataset" in res
    assert set(res["per_dataset"]) == set(ex.DATASETS)
