"""Regression tests for the AAAI tightness analysis. Pure numpy, runs in CI."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from certification import EdgeBernoulliNoise, bernstein_certificate

TIGHTNESS_JSON = Path(__file__).resolve().parents[1] / "results" / "aaai" / "tightness.json"


def _star(confidences: np.ndarray) -> sp.csr_matrix:
    e = len(confidences)
    rows, cols, data = [], [], []
    for k, c in enumerate(confidences):
        rows += [0, k + 1]
        cols += [k + 1, 0]
        data += [float(c), float(c)]
    return sp.coo_matrix((data, (rows, cols)), shape=(e + 1, e + 1)).tocsr()


def test_variance_proxy_is_sum_of_edge_variances_for_star() -> None:
    # All uncertain edges incident to the center, so v = sum_e c(1-c).
    confs = np.array([0.2, 0.5, 0.8, 0.6, 0.3])
    cert = bernstein_certificate(
        np.ones(6), EdgeBernoulliNoise(C=_star(confs)), alpha=0.5, delta=0.05
    )
    assert cert.v == np.sum(confs * (1.0 - confs))


def test_variance_term_maximized_at_half_and_sqrt_scaling() -> None:
    # The radius variance term sqrt(2 v log) is proportional to sqrt(c(1-c)),
    # so it is maximized at c = 0.5 and vanishes as c -> 0 or 1.
    e, alpha, delta = 32, 0.5, 0.05
    cs = np.linspace(0.05, 0.95, 19)
    var_terms = []
    for c in cs:
        cert = bernstein_certificate(
            np.ones(e + 1), EdgeBernoulliNoise(C=_star(np.full(e, c))), alpha, delta
        )
        log_term = math.log(2.0 * cert.n_nodes / delta)
        var_terms.append(math.sqrt(2.0 * cert.v * log_term))
    var_terms = np.array(var_terms)
    assert abs(cs[int(var_terms.argmax())] - 0.5) < 1e-9  # peak at coin-flip confidence
    assert var_terms[0] < var_terms[len(cs) // 2]  # tighter near 0
    assert var_terms[-1] < var_terms[len(cs) // 2]  # tighter near 1
    # var_term / sqrt(c(1-c)) is constant => the sqrt(sum c(1-c)) scaling law.
    ratio = var_terms / np.sqrt(cs * (1.0 - cs))
    assert ratio.std() / ratio.mean() < 1e-9


def test_lower_bound_achieves_variance_at_half() -> None:
    # At c = 0.5 every drawn perturbation attains ||A - A_bar||_op = sqrt(v).
    e = 32
    noise = EdgeBernoulliNoise(C=_star(np.full(e, 0.5)))
    cert = bernstein_certificate(np.ones(e + 1), noise, alpha=0.5, delta=0.05)
    a_bar = noise.mean_adjacency().toarray()
    rng = np.random.default_rng(0)
    ops = np.array(
        [np.linalg.norm(noise.sample_adjacency(rng).toarray() - a_bar, 2) for _ in range(500)]
    )
    assert np.allclose(ops, math.sqrt(cert.v))  # variance achieved deterministically
    # Bernstein b exceeds the achieved sqrt(v) only by a bounded (log) factor.
    assert cert.b >= math.sqrt(cert.v)
    assert 1.0 < cert.b / math.sqrt(cert.v) < 10.0


def test_probabilistic_tighter_than_deterministic_by_sqrt_e() -> None:
    alpha, delta, c = 0.5, 0.05, 0.5
    edge_grid = [4, 8, 16, 32, 64, 128]
    ratios = []
    for e in edge_grid:
        cert = bernstein_certificate(
            np.ones(e + 1), EdgeBernoulliNoise(C=_star(np.full(e, c))), alpha, delta
        )
        det_lipschitz = e * max(c, 1.0 - c)  # sum_e ||W_e||_op (deterministic envelope)
        ratios.append(det_lipschitz / cert.b)
    ratios = np.array(ratios)
    assert np.all(np.diff(ratios) > 0)  # deterministic loses ground as E grows
    exponent = float(np.polyfit(np.log(edge_grid), np.log(ratios), 1)[0])
    assert 0.45 <= exponent <= 0.6  # ratio ~ sqrt(E)


def test_committed_tightness_json() -> None:
    result = json.loads(TIGHTNESS_JSON.read_text())
    a = result["A_variance_scaling"]
    assert a["argmax_c_variance_term"] == 0.5
    assert a["variance_term_over_sqrt_c1mc_cv"] < 1e-6
    # Full-radius peak converges to 0.5 as the uncertain-edge count grows.
    argmax_vs_e = a["full_radius_argmax_vs_e"]
    assert argmax_vs_e["4096"] >= argmax_vs_e["16"]
    assert argmax_vs_e["4096"] >= 0.48

    b = result["B_matching_lower_bound"]["c=0.5"]
    assert b["rms_matches_sqrt_v"] is True
    assert b["bernstein_b"] >= b["sqrt_v"]

    c = result["C_probabilistic_vs_deterministic"]
    assert c["approx_sqrt_e"] is True
    assert 0.45 <= c["fitted_ratio_exponent"] <= 0.6
