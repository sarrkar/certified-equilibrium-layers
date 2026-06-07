"""Tests for experiments/aaai/exp_t1_leakage.py.

The full audit (three trained CBMs) is torch-gated and runs on the committed
real-slice efficacy fixtures; the acceptance logic and the pending path are
checked without heavy training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "aaai"))
sys.path.insert(0, str(REPO / "src"))

import exp_t1_leakage as ex  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "efficacy"


def test_acceptance_logic() -> None:
    def mk(cross, gap=0.0, comp=0.5):
        return {
            "max_cross_channel_change": cross,
            "leakage_mi": {"mean_leakage_auc_gap": gap},
            "completeness": {"completeness": comp},
        }

    passing = {
        "disjoint_ccmp": mk(0.0),
        "shared_encoder_cbm": mk(0.5),
        "concept_embedding_model": mk(0.3),
    }
    acc = ex.acceptance(passing)
    assert acc["flag"] == "PASS"
    assert acc["disjoint_machine_zero"] and acc["shared_encoder_leaks"] and acc["cem_leaks"]

    # Disjoint leaks -> FAIL.
    leaky_disjoint = dict(passing, disjoint_ccmp=mk(0.1))
    assert ex.acceptance(leaky_disjoint)["flag"] == "FAIL"

    # Shared encoder does not leak (degenerate baseline) -> FAIL.
    no_shared_leak = dict(passing, shared_encoder_cbm=mk(0.0))
    assert ex.acceptance(no_shared_leak)["flag"] == "FAIL"


def test_efficacy_matrix_raises_without_real_data(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ex.efficacy_matrix(tmp_path / "absent", tmp_path / "cache")


def test_full_audit_on_fixtures(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    audit = ex.run_audit(FIXTURES, tmp_path / "cache", seed=0, epochs=30, lr=0.01)
    m = audit["models"]
    assert set(m) == {"disjoint_ccmp", "shared_encoder_cbm", "concept_embedding_model"}
    # The disjoint backbone leaves every other channel exactly unchanged (L_k = 0).
    assert m["disjoint_ccmp"]["max_cross_channel_change"] == 0.0
    # The shared-encoder baselines leak across concepts.
    assert m["shared_encoder_cbm"]["max_cross_channel_change"] > 1e-3
    assert m["concept_embedding_model"]["max_cross_channel_change"] > 1e-3
    acc = ex.acceptance(m)
    assert acc["flag"] == "PASS" and acc["disjoint_machine_zero"] is True
    assert len(audit["concepts"]) == 9  # the K = 9 concept channels
