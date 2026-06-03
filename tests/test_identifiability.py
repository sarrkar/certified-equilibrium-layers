"""Tests for Project 1 (identifiability). Pure numpy, runs in CI."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from identifiability.metrics import amari_distance, mean_correlation_coefficient
from identifiability.mixing import ObservationModel
from identifiability.recover import recover_latents, recover_mixing, recover_observation_map
from identifiability.scm import random_dag_scm


def _graph(d: int, rng: np.random.Generator) -> sp.csr_matrix:
    w = rng.random((d, d)) * (rng.random((d, d)) < 0.5)
    w = np.triu(w, 1)
    for i in range(d):
        w[i, (i + 1) % d] = max(w[i, (i + 1) % d], 0.5)
    w = np.triu(w, 1)
    w = w + w.T
    deg = w.sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return sp.csr_matrix(dinv[:, None] * w * dinv[None, :])


def test_scm_sample_shape_and_total_effect() -> None:
    rng = np.random.default_rng(0)
    scm = random_dag_scm(5, rng, edge_density=0.5)
    z = scm.sample(100, rng)
    assert z.shape == (100, 5)
    f = scm.total_effect_matrix()
    assert np.allclose(f @ (np.eye(5) - scm.B), np.eye(5))  # (I-B)^{-1} (I-B) = I


def test_hard_intervention_cuts_parents() -> None:
    rng = np.random.default_rng(0)
    scm = random_dag_scm(5, rng, edge_density=0.9, coef_scale=0.9)
    intervened = scm.intervene_hard(2, value_std=2.0)
    assert np.allclose(intervened.B[2, :], 0.0)  # incoming edges removed
    assert intervened.noise_std[2] == 2.0


def test_observation_is_invertible() -> None:
    rng = np.random.default_rng(1)
    obs = ObservationModel.from_graph(_graph(5, rng), alpha=0.2, rng=rng)
    z = rng.standard_normal((50, 5))
    y = obs.observe(z)
    z_back = recover_latents(y, obs.W)  # known W inverts exactly
    assert np.allclose(z_back, z, atol=1e-8)


def test_mcc_is_permutation_and_scale_invariant() -> None:
    rng = np.random.default_rng(2)
    z = rng.standard_normal((2000, 4))
    assert mean_correlation_coefficient(z, z) == pytest.approx(1.0, abs=1e-6)
    shuffled = z[:, [3, 0, 2, 1]] * np.array([-2.0, 0.5, 3.0, -1.0])
    assert mean_correlation_coefficient(z, shuffled) == pytest.approx(1.0, abs=1e-6)


def test_exact_recovery_independent_sources() -> None:
    # Identifiable base case: B = 0, hard single-gene (variance) interventions.
    rng = np.random.default_rng(0)
    d, n = 6, 30000
    obs = ObservationModel.from_graph(_graph(d, rng), alpha=0.2, rng=rng)
    scm = random_dag_scm(d, rng, edge_density=0.0)
    z = scm.sample(n, rng)
    y = obs.observe(z)
    y_int = {i: obs.observe(scm.intervene_hard(i, value_std=3.0).sample(n, rng)) for i in range(d)}

    w_hat = recover_mixing(y, y_int)
    z_hat = recover_latents(y, w_hat)
    assert mean_correlation_coefficient(z, z_hat) > 0.97
    assert amari_distance(obs.W, w_hat) < 0.05
    # Known P_alpha lets us name the recovered observation map.
    g_hat = recover_observation_map(w_hat, obs.P_alpha)
    assert g_hat.shape == obs.G.shape


@pytest.mark.xfail(
    strict=True,
    reason="exact recovery on a dense latent SCM is the open research target; "
    "the rank-one cov-difference recovery only handles independent sources",
)
def test_gap_exact_recovery_dense_scm() -> None:
    rng = np.random.default_rng(0)
    d, n = 6, 30000
    obs = ObservationModel.from_graph(_graph(d, rng), alpha=0.2, rng=rng)
    scm = random_dag_scm(d, rng, edge_density=0.8, coef_scale=0.9)
    z = scm.sample(n, rng)
    y = obs.observe(z)
    y_int = {i: obs.observe(scm.intervene_hard(i, value_std=3.0).sample(n, rng)) for i in range(d)}
    z_hat = recover_latents(y, recover_mixing(y, y_int))
    # Near-exact identification on a dense SCM is not yet achieved.
    assert mean_correlation_coefficient(z, z_hat) > 0.95


def test_committed_identifiability_json() -> None:
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1] / "results" / "iclr" / "identifiability_synthetic.json"
    )
    result = json.loads(path.read_text())
    assert result["identified_base_case"] is True
    assert result["regimes"]["independent_hard"]["mcc"] > 0.95
