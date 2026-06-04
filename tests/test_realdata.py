"""Regression tests for the AAAI real-data certification experiment.

Pure numpy, runs in CI. Imports the experiment module (no torch, matplotlib is
lazy) and checks the committed JSON.
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


def test_seedmatch_x0_in_unit_interval_with_off_targets() -> None:
    rng = np.random.default_rng(0)
    x0 = ex.seedmatch_x0(120, ex.GUIDE, rng, utr_len=300, plant_frac=0.2)
    assert x0.shape == (120,)
    assert x0.min() >= 0.0 and x0.max() <= 1.0  # SeedMatchR site grades
    assert np.count_nonzero(x0) > 0  # planted seeds give an off-target set


def test_subsample_and_inject_change_edge_count() -> None:
    rng = np.random.default_rng(1)
    c = ex.synthetic_confidence(40, 0.6, 0.4, 1.0, rng)
    n_edges = c.nnz // 2
    fewer = ex.subsample_edges(c, 0.5, np.random.default_rng(2)).nnz // 2
    more = ex.inject_false_edges(c, 1.0, 0.4, 1.0, np.random.default_rng(3)).nnz // 2
    assert fewer < n_edges
    assert more > n_edges


def test_containment_fraction_helper() -> None:
    y = np.array([0.0, 0.5, 2.0])
    assert ex.containment_fraction(y, radius=1.0, tau=0.5) == 0.0  # radius alone exceeds tau
    assert ex.containment_fraction(y, radius=1.0, tau=10.0) == 1.0  # all contained
    assert ex.containment_fraction(y, radius=float("inf"), tau=10.0) == 0.0  # invalid cert


def test_containment_fraction_monotone_in_tolerance() -> None:
    rng = np.random.default_rng(0)
    c = ex.synthetic_confidence(40, 0.85, 0.7, 0.97, rng)
    x0 = ex.seedmatch_x0(40, ex.GUIDE, rng, utr_len=300, plant_frac=0.3)
    res = ex.part_containment(c, x0, alpha=0.5, delta=0.1)
    frac = np.array(res["certified_contained_fraction"])
    assert np.all(np.diff(frac) >= -1e-12)  # non-decreasing in tau
    assert frac[0] == 0.0 and frac[-1] == 1.0


def test_mc_cost_grows_as_delta_shrinks() -> None:
    rng = np.random.default_rng(0)
    x0 = ex.seedmatch_x0(60, ex.GUIDE, rng, plant_frac=0.3)
    mc = ex.part_mc_cost(x0, 60, eps_rel=0.1)
    ns = [r["n_samples"] for r in mc["per_delta"]]  # deltas listed descending
    assert all(np.diff(ns) >= 0)  # N increases as delta decreases


def test_committed_realdata_json() -> None:
    result = json.loads(REALDATA_JSON.read_text())
    assert "provenance" in result and "x0_source" in result["provenance"]

    thr = result["radius_vs_threshold"]["per_threshold"]
    assert [d["threshold"] for d in thr] == [0.4, 0.7, 0.9]
    rhos = [d["degree_rel_fluctuation"] for d in thr]
    assert all(np.diff(rhos) >= -1e-9)  # sparser graph => larger rho

    deg = result["degradation"]
    assert deg["breakdown_subsample_fraction"] is not None
    sub_rho = [d["degree_rel_fluctuation"] for d in deg["subsample"]]
    assert all(np.diff(sub_rho) >= -1e-9)  # more edges dropped => larger rho
    assert deg["subsample"][0]["assumption_holds"] is True
    assert deg["subsample"][-1]["assumption_holds"] is False

    ns = [r["n_samples"] for r in result["monte_carlo_cost"]["per_delta"]]
    assert all(np.diff(ns) >= 0)

    cont = result["containment"]
    assert cont["tau_for_90pct_certified"] is not None
    assert 0.0 <= min(cont["certified_contained_fraction"]) <= 1.0
