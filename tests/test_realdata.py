"""Regression tests for the AAAI off-target certification experiment.

Pure numpy/sklearn; imports the experiment module (no torch) and checks the
committed per-dataset JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "aaai"))

import exp_realdata as ex  # noqa: E402

REALDATA_JSON = REPO / "results" / "aaai" / "realdata.json"


def test_runs_over_all_offtarget_datasets() -> None:
    from cascade.data import OFFTARGET_CERTIFICATION, datasets_with_role

    assert set(ex.DATASET_CONFIG) == set(datasets_with_role(OFFTARGET_CERTIFICATION))


def test_seed_repression_tracks_measured_labels() -> None:
    rng = np.random.default_rng(0)
    measured = np.zeros(120, dtype=bool)
    measured[:30] = True
    x0 = ex.seed_repression_x0(measured, plant_p=0.95, bg_q=0.02, rng=rng, utr_len=200)
    assert x0.shape == (120,)
    assert x0.min() >= 0.0 and x0.max() <= 1.0
    # Planted off-targets carry much more seed signal than the background.
    assert x0[measured].mean() > x0[~measured].mean()


def test_subsample_and_inject_change_edges() -> None:
    rng = np.random.default_rng(1)
    c = ex.synthetic_confidence(40, 0.6, 0.4, 1.0, rng)
    n = c.nnz // 2
    assert ex.subsample_edges(c, 0.5, np.random.default_rng(2)).nnz // 2 < n
    assert ex.inject_false_edges(c, 1.0, 0.4, 1.0, np.random.default_rng(3)).nnz // 2 > n


def test_evaluate_dataset_meaningful_on_rna_seq() -> None:
    rng = np.random.default_rng(0)
    res = ex.evaluate_dataset(
        "gse184929", n_nodes=150, alpha=0.5, delta=0.1, base_thr=0.7, rng=rng, string_links=None
    )
    c = res["containment"]
    # Measured off-targets are mostly NOT contained; non-off-targets are.
    assert c["offtarget_contained_fraction"] < c["background_contained_fraction"]
    assert c["auc_predicted_vs_measured"] > 0.55
    assert res["meaningful"] is True
    # Radius curve has the three STRING thresholds.
    assert [d["threshold"] for d in res["radius_vs_threshold"]] == [0.4, 0.7, 0.9]


def test_committed_realdata_json() -> None:
    d = json.loads(REALDATA_JSON.read_text())
    assert d["theorem4_radius"] == "((1-alpha)/alpha) * b(delta) * ||x0||"
    assert set(d["per_dataset"]) == set(ex.DATASET_CONFIG)
    # Each dataset reports containment, radius-vs-threshold, and degradation.
    for name, r in d["per_dataset"].items():
        assert "containment" in r and "radius_vs_threshold" in r and "degradation" in r
        assert r["assay"] in {"rna_seq", "microarray", "microarray_aggregated"}
    # Cross-assay: RNA-seq separates measured off-targets at least as cleanly as microarray.
    ca = d["cross_assay"]
    assert ca["rna_seq_mean_auc"] >= ca["microarray_mean_auc"]
    # Acceptance: GSE184929 RNA-seq and at least one microarray are meaningful.
    acc = d["acceptance"]
    assert acc["flag"] in {"PASS", "FAIL"}
    assert acc["gse184929_meaningful"] is True
    assert acc["any_microarray_meaningful"] is True
    assert acc["flag"] == "PASS"
