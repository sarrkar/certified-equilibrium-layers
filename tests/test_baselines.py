"""Tests for the linear baselines and the GO/NO-GO gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cascade.baselines.linear import (
    AdditiveLinearBaseline,
    MeanBaseline,
    regression_metrics,
)


def test_mean_baseline_predicts_train_mean() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 4))
    y = rng.standard_normal(20) + 3.0
    model = MeanBaseline().fit(x, y)
    pred = model.predict(rng.standard_normal((5, 4)))
    assert np.allclose(pred, y.mean())
    assert pred.shape == (5,)


def test_additive_linear_recovers_linear_signal() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 3))
    w_true = np.array([1.5, -2.0, 0.5])
    y = x @ w_true + 0.7
    model = AdditiveLinearBaseline().fit(x, y)
    assert np.allclose(model.coef_, w_true, atol=1e-6)
    assert abs(model.intercept_ - 0.7) < 1e-6
    assert regression_metrics(model.predict(x), y)["mse"] < 1e-10


def test_additive_linear_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="must be fit"):
        AdditiveLinearBaseline().predict(np.zeros((2, 3)))


def test_regression_metrics_constant_pred_pearson_nan() -> None:
    m = regression_metrics(np.ones(5), np.arange(5.0))
    assert m["mse"] > 0
    assert np.isnan(m["pearson"])  # constant prediction has undefined correlation


def test_committed_gonogo_json_is_valid_and_go() -> None:
    # The committed (multi-seed) gate result must parse and record a GO verdict.
    path = Path(__file__).resolve().parents[1] / "results" / "baseline_gonogo.json"
    result = json.loads(path.read_text())
    assert result["go"] is True
    assert result["n_go"] == result["n_seeds"]
    assert result["cascade_test_mse_mean"] <= result["best_baseline_test_mse_mean"] * (
        1.0 + result["tie_band"]
    )
    # Every seed individually ties or beats the best baseline within the tie band.
    for r in result["per_seed"]:
        assert r["cascade"]["test_mse"] <= r["best_baseline_test_mse"] * (1.0 + result["tie_band"])


def test_gonogo_runs_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from cascade.eval.gonogo import run_gonogo

    cfg = {
        "seed": 0,
        "device": "cpu",
        "model": {
            "hidden_dim": 16,
            "concept_emb_dim": 16,
            "n_layers": 2,
            "head_hidden": 16,
            "dropout": 0.0,
            "prop_steps": 8,
            "alpha_init": 0.15,
            "learnable_alpha": True,
        },
        "dataset": {
            "graph": "synthetic",
            "n_nodes": 40,
            "edge_p": 0.3,
            "synthetic": True,
            "synthetic_alpha": 0.2,
            "synthetic_noise": 0.05,
            "split": {"mode": "standard", "val_frac": 0.2, "test_frac": 0.2},
        },
        "training": {"epochs": 60, "lr": 0.02, "log_every": 30},
    }
    result = run_gonogo(cfg=cfg, seed=0, out_dir=tmp_path, run_dir=tmp_path / "run")
    assert "go" in result
    assert (tmp_path / "baseline_gonogo.json").exists()
    # JSON is valid (no NaN tokens).
    json.loads((tmp_path / "baseline_gonogo.json").read_text())
