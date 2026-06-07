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
