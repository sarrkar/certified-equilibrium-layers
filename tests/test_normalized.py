"""Tests for the symmetric-normalized (C5) certificate. Pure numpy, runs in CI."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from certification import (
    EdgeBernoulliNoise,
    expected_degrees,
    min_degree_diagnostic,
    normalized_certificate,
)
from certification.noise_model import normalize_adjacency

NORMALIZED_JSON = Path(__file__).resolve().parents[1] / "results" / "aaai" / "normalized.json"


def _synth(n: int, p: float, lo: float, hi: float, rng) -> sp.csr_matrix:
    iu = np.triu_indices(n, 1)
    present = rng.random(iu[0].shape[0]) < p
    conf = rng.uniform(lo, hi, iu[0].shape[0]) * present
    rows = np.concatenate([iu[0][present], iu[1][present]])
    cols = np.concatenate([iu[1][present], iu[0][present]])
    data = np.concatenate([conf[present], conf[present]])
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _threshold(C: sp.csr_matrix, cutoff: float) -> sp.csr_matrix:
    co = C.tocoo()
    keep = co.data >= cutoff
    return sp.coo_matrix((co.data[keep], (co.row[keep], co.col[keep])), shape=C.shape).tocsr()


def test_expected_degrees_are_row_sums() -> None:
    c = _synth(20, 0.5, 0.5, 0.9, np.random.default_rng(0))
    assert np.allclose(expected_degrees(c), np.asarray(c.sum(axis=1)).ravel())


def test_certificate_valid_and_covers_on_dense_graph() -> None:
    rng = np.random.default_rng(0)
    n, delta, alpha = 40, 0.1, 0.5
    noise = EdgeBernoulliNoise(C=_synth(n, 0.85, 0.75, 0.97, rng))
    x0 = rng.standard_normal(n)
    cert = normalized_certificate(x0, noise, alpha, delta)
    assert cert.assumption_holds  # dense graph: degrees concentrate, rho < 1
    assert cert.degree_rel_fluctuation < 1.0
    assert math.isfinite(cert.structure_bound) and cert.structure_bound > 0
    assert math.isfinite(cert.radius_sym) and cert.radius_sym > 0

    s_bar = normalize_adjacency(noise.mean_adjacency(), "sym")
    devs = np.array(
        [
            np.linalg.norm(
                (normalize_adjacency(noise.sample_adjacency(rng), "sym") - s_bar).toarray(), 2
            )
            for _ in range(600)
        ]
    )
    # The degree-concentration structure bound holds with prob >= 1 - delta.
    assert float(np.mean(devs <= cert.structure_bound)) >= 1.0 - delta
    assert cert.structure_bound >= float(np.quantile(devs, 1.0 - delta))


def test_certificate_invalid_when_too_sparse() -> None:
    # A near-isolated graph fails the minimum-expected-degree assumption.
    c = sp.csr_matrix(np.array([[0.0, 0.9, 0.0], [0.9, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    cert = normalized_certificate(np.ones(3), EdgeBernoulliNoise(C=c), alpha=0.5, delta=0.1)
    assert cert.assumption_holds is False
    assert cert.degree_rel_fluctuation >= 1.0
    assert math.isinf(cert.radius_sym)


def test_breakdown_is_monotone_and_occurs() -> None:
    rng = np.random.default_rng(1)
    c_full = _synth(200, 0.55, 0.4, 1.0, rng)
    cutoffs = np.arange(0.70, 0.951, 0.05)
    rhos, min_degs, valid = [], [], []
    for cut in cutoffs:
        diag = min_degree_diagnostic(EdgeBernoulliNoise(C=_threshold(c_full, cut)), delta=0.1)
        rhos.append(diag["degree_rel_fluctuation"])
        min_degs.append(diag["expected_min_degree"])
        valid.append(diag["assumption_holds"])
    # Raising the cutoff sparsifies: min degree drops, rho grows, validity is lost.
    assert all(np.diff(min_degs) <= 1e-9)  # non-increasing min expected degree
    assert all(np.diff(rhos) >= -1e-9)  # non-decreasing relative fluctuation
    assert valid[0] and not valid[-1]  # valid when dense, breaks when sparse


def test_committed_normalized_json() -> None:
    result = json.loads(NORMALIZED_JSON.read_text())
    cov = result["coverage"]
    assert cov["structure_no_violation"] is True
    assert cov["prediction_no_violation"] is True
    assert cov["degree_rel_fluctuation"] < 1.0

    bd = result["breakdown"]
    assert bd["breakdown_cutoff"] is not None
    assert 0.70 <= bd["breakdown_cutoff"] <= 0.95
    rhos = [d["degree_rel_fluctuation"] for d in bd["per_cutoff"]]
    assert all(np.diff(rhos) >= -1e-9)  # rho non-decreasing in the cutoff
    # Validity holds at the loosest cutoff and is lost by the tightest.
    assert bd["per_cutoff"][0]["assumption_holds"] is True
    assert bd["per_cutoff"][-1]["assumption_holds"] is False
