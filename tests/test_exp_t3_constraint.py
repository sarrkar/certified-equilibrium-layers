"""Tests for experiments/aaai/exp_t3_constraint.py (torch + ViennaRNA gated)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "aaai"))
sys.path.insert(0, str(REPO / "src"))

import exp_t3_constraint as ex  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "efficacy"


def _result(residual, unc_p, con_p, newton_p=0.3):
    return {
        "constraint_residual": {"constrained_max": residual},
        "accuracy": {
            "unconstrained": {"pearson": unc_p},
            "constrained_kkt": {"pearson": con_p},
            "exact_occupancy_newton": {"pearson": newton_p},
        },
    }


def test_acceptance_logic() -> None:
    # Residual at machine precision and no accuracy regression -> PASS.
    assert ex.acceptance(_result(1e-12, 0.50, 0.50))["flag"] == "PASS"
    # Residual above tolerance -> FAIL.
    assert ex.acceptance(_result(1e-3, 0.50, 0.50))["flag"] == "FAIL"
    # Constraint costs accuracy -> FAIL, flagged.
    cost = ex.acceptance(_result(1e-12, 0.50, 0.30))
    assert cost["flag"] == "FAIL" and cost["constraint_costs_accuracy"] is True
    # Pearson undefined (tiny data) -> still PASS on the residual alone.
    assert ex.acceptance(_result(1e-12, None, None))["flag"] == "PASS"


def test_real_inputs_raises_without_data(tmp_path) -> None:
    pytest.importorskip("RNA")
    with pytest.raises(FileNotFoundError):
        ex.real_inputs(tmp_path / "absent", tmp_path / "cache")


def test_full_path_on_fixtures(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("RNA")
    res = ex.run_t3(FIXTURES, tmp_path / "cache", seed=0, epochs=40, lr=0.01)
    # The KKT-projected output satisfies the mass-action constraint at machine precision.
    assert res["constraint_residual"]["constrained_max"] < 1e-8
    # The unconstrained prediction clearly violates the constraint (it is non-trivial).
    assert res["constraint_residual"]["unconstrained_max"] > 1e-3
    # The mass-action slope is the (negative) Boltzmann slope.
    assert res["mass_action_line"]["b"] < 0
    acc = ex.acceptance(res)
    assert acc["flag"] in {"PASS", "FAIL"}
    assert acc["constraint_residual_below_tol"] is True


def test_multiseed_aggregation_is_conservative() -> None:
    def run(res, unc_p, con_p):
        def metric(p):
            return {"pearson": p, "spearman": None, "mse": 0.1, "mae": 0.1}

        return {
            "n_train": 3,
            "n_test": 2,
            "mass_action_line": {"a": -10.0, "b": -1.0},
            "constraint_residual": {
                "constrained_max": res,
                "unconstrained_max": 1e5,
                "exact_occupancy_newton_max": 0.1,
                "tolerance": 1e-8,
            },
            "accuracy": {
                "unconstrained": metric(unc_p),
                "constrained_kkt": metric(con_p),
                "exact_occupancy_newton": metric(0.2),
            },
        }

    # Residual below tol every seed AND constrained drop within one std of the
    # unconstrained Pearson -> PASS; worst residual kept.
    runs = [run(1e-10, 0.50, 0.50), run(2e-10, 0.55, 0.52), run(1e-10, 0.60, 0.55)]
    agg = ex.aggregate_results(runs, [0, 1, 2])
    assert agg["constraint_residual"]["constrained_max"] == 2e-10
    acc = ex.aggregate_acceptance(agg, [ex.acceptance(r) for r in runs])
    assert acc["flag"] == "PASS" and acc["n_seeds"] == 3
    for k in (
        "mean_unconstrained_pearson",
        "std_unconstrained_pearson",
        "mean_constrained_pearson",
        "std_constrained_pearson",
    ):
        assert acc[k] is not None  # the four reported stats are present
    assert acc["accuracy_regression_beyond_one_std"] is False

    # Constrained drop LARGER than one std of unconstrained -> FAIL (accuracy gate).
    runs2 = [run(1e-10, 0.50, 0.30), run(1e-10, 0.50, 0.30), run(1e-10, 0.50, 0.30)]
    acc2 = ex.aggregate_acceptance(
        ex.aggregate_results(runs2, [0, 1, 2]), [ex.acceptance(r) for r in runs2]
    )
    assert acc2["flag"] == "FAIL" and acc2["accuracy_regression_beyond_one_std"] is True

    # Residual above tol on one seed -> FAIL regardless of accuracy (the theorem gate).
    runs3 = [run(1e-10, 0.50, 0.50), run(1e-3, 0.50, 0.50), run(1e-10, 0.50, 0.50)]
    acc3 = ex.aggregate_acceptance(
        ex.aggregate_results(runs3, [0, 1, 2]), [ex.acceptance(r) for r in runs3]
    )
    assert acc3["flag"] == "FAIL" and acc3["constraint_residual_below_tol_all_seeds"] is False
