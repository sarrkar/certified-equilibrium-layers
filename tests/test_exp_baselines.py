"""Tests for experiments/aaai/exp_baselines.py (REAL-data baseline comparison).

REAL DATA ONLY: with no real efficacy / SeedMatchR data present the sections go
pending with their source. The full path (CCMP + ridge + mean + acceptance) is
exercised on the committed REAL-slice fixtures and is torch / sklearn gated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "aaai"))

import exp_baselines as ex  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "efficacy"


def test_reported_baselines_cited_not_fabricated() -> None:
    names = {b["name"] for b in ex.REPORTED_BASELINES}
    # The mandatory simple baselines (mean, ridge) are computed, not "reported";
    # the published deep / seed models are cited with a source.
    assert {
        "OligoFormer",
        "siRNADiscovery (GNN)",
        "TargetScan context++",
        "scanMiR",
        "PITA",
    } <= names
    for b in ex.REPORTED_BASELINES:
        assert b["source"] and isinstance(b["source"], str)
        assert b["reported_value"] is None or isinstance(b["reported_value"], int | float)
    cited = {b["name"]: b["reported_value"] for b in ex.REPORTED_BASELINES}
    assert cited["BIOPREDsi (Huesken ANN)"] == 0.66  # transcribed reference point


def test_efficacy_section_raises_without_real_huesken(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ex.efficacy_section(tmp_path / "absent", tmp_path / "cache", seed=0, epochs=1, lr=0.01)


def test_offtarget_section_pending_without_export(tmp_path) -> None:
    sec = ex.offtarget_section(tmp_path / "absent", tmp_path / "cache")
    assert sec["status"] == "pending_real_data"
    # Every Schlegel contrast is pending with its source, none fabricated.
    assert set(sec["pending"]) and not sec["per_contrast"]


def test_acceptance_logic() -> None:
    def mk(cascade_p, cascade_mse, ridge_p, mean_mse):
        return {
            "status": "ok",
            "our_models": {
                "cascade_ccmp": {"huesken_test": {"pearson": cascade_p, "mse": cascade_mse}},
                "ridge_linear": {"huesken_test": {"pearson": ridge_p}},
                "mean": {"huesken_test": {"mse": mean_mse}},
            },
        }

    passing = ex._acceptance(mk(0.55, 0.10, 0.54, 0.20))
    assert passing["flag"] == "PASS"
    assert passing["beats_linear"] and passing["beats_mean"]

    lose_mean = ex._acceptance(mk(0.55, 0.30, 0.54, 0.20))  # worse MSE than the mean
    assert lose_mean["flag"] == "FAIL" and lose_mean["beats_mean"] is False

    lose_linear = ex._acceptance(mk(0.40, 0.10, 0.60, 0.20))  # below ridge Pearson
    assert lose_linear["flag"] == "FAIL" and lose_linear["beats_linear"] is False

    assert (
        ex._acceptance({"status": "ok", "our_models": {"cascade_ccmp": None}})["flag"] == "PENDING"
    )
    assert ex._acceptance({"status": "pending_real_data"})["flag"] == "PENDING"


def test_full_path_on_fixtures(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    sec = ex.efficacy_section(FIXTURES, tmp_path / "cache", seed=0, epochs=20, lr=0.01)
    assert sec["status"] == "ok"
    assert set(sec["our_models"]) == {"mean", "ridge_linear", "cascade_ccmp"}
    # Mandatory simple baselines computed on the same split as CASCADE.
    for model in ("mean", "ridge_linear", "cascade_ccmp"):
        assert "huesken_test" in sec["our_models"][model]
    acc = ex._acceptance(sec)
    assert acc["flag"] in {"PASS", "FAIL"}
    assert isinstance(acc["beats_mean"], bool) and isinstance(acc["beats_linear"], bool)
    # Cross-datasets: katoh resolves from the fixture master; mix is absent, so it is
    # recorded unavailable, never faked.
    assert "mix" in set(sec["cross_datasets_unavailable"])


def test_aggregate_acceptance_is_conservative() -> None:
    def acc(flag, beats_linear=True, beats_mean=True):
        if flag == "PENDING":
            return {"flag": "PENDING"}
        return {
            "flag": flag,
            "beats_linear": beats_linear,
            "beats_mean": beats_mean,
            "cascade_pearson": 0.55,
            "ridge_pearson": 0.54,
            "cascade_mse": 0.10,
            "mean_mse": 0.20,
        }

    # Every seed passes -> PASS, with the across-seed means reported.
    agg = ex._aggregate_acceptance([acc("PASS")] * 3, [0, 1, 2])
    assert agg["flag"] == "PASS" and agg["n_seeds"] == 3
    assert agg["beats_linear_all"] and agg["beats_mean_all"]
    assert agg["cascade_pearson_mean"] == 0.55

    # One seed fails the mean bar -> conservative FAIL.
    agg = ex._aggregate_acceptance(
        [acc("PASS"), acc("FAIL", beats_mean=False), acc("PASS")], [0, 1, 2]
    )
    assert agg["flag"] == "FAIL" and agg["beats_mean_all"] is False

    # Any pending seed -> PENDING aggregate.
    agg = ex._aggregate_acceptance([acc("PASS"), acc("PENDING")], [0, 1])
    assert agg["flag"] == "PENDING"


def test_multiseed_on_fixtures(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    seeds = [0, 1, 2]
    section, accs = ex.efficacy_multiseed(FIXTURES, tmp_path / "cache", seeds, epochs=15, lr=0.01)
    assert section["status"] == "ok"
    assert len(section["per_seed"]) == len(seeds)  # CCMP retrained per seed
    assert set(section["deterministic_baselines"]) == {"mean", "ridge_linear"}  # computed once
    agg = ex._aggregate_acceptance(accs, seeds)
    assert agg["flag"] in {"PASS", "FAIL"} and agg["n_seeds"] == 3
