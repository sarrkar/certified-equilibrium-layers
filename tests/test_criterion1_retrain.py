"""Criterion-1 retrain gate: pre-registration is fixed and the verdict is self-consistent.

Reads the committed results/cascade_ot/criterion1_retrain.json (the GPU retrain is not re-run
here). Verifies the gate was pre-registered (the >= 0.05 mean-margin threshold and the
both-studies / D4<D1 criteria) and that the PASS/FAIL verdict follows the pre-registered rule
exactly: PASS iff all three criteria hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / "results" / "cascade_ot" / "criterion1_retrain.json"


def _load() -> dict:
    if not RESULT.exists():
        pytest.skip("criterion1_retrain.json not built yet")
    return json.loads(RESULT.read_text())


def test_gate_is_pre_registered() -> None:
    j = _load()
    pre = j["pre_registered"]
    assert pre["criterion_a_min_mean_margin"] == 0.05
    assert "BOTH" in pre["criterion_b"]
    assert "D4" in pre["criterion_c"] and "D1" in pre["criterion_c"]
    assert pre["decision_rule"].startswith("PASS only if (a) AND (b) AND (c)")


def test_verdict_follows_pre_registered_rule() -> None:
    j = _load()
    cr = j["criteria"]
    m = j["margins"]
    # (a) pooled margin meets threshold.
    assert cr["a_pooled_margin"] == (m["pooled_mean"] >= 0.05)
    # (b) both studies strictly positive.
    assert cr["b_both_studies_positive"] == (m["schlegel_mean"] > 0 and m["patel_mean"] > 0)
    # (c) D4 < D1.
    assert cr["c_d4_lt_d1"] == (j["d4_lt_d1"]["E_D4"] < j["d4_lt_d1"]["E_D1"])
    # verdict is exactly the conjunction.
    expected = (
        "PASS"
        if (cr["a_pooled_margin"] and cr["b_both_studies_positive"] and cr["c_d4_lt_d1"])
        else "FAIL"
    )
    assert j["verdict"] == expected


def test_d4_lt_d1_discrimination_holds() -> None:
    # The modification-aware discrimination is the methods result that stands regardless of the gate.
    j = _load()
    assert j["d4_lt_d1"]["holds"] is True
    assert j["d4_lt_d1"]["E_D4"] < j["d4_lt_d1"]["E_D1"]


def test_evaluation_is_zero_shot_real_studies() -> None:
    j = _load()
    studies = {c["study"] for c in j["conditions"].values()}
    assert {"Schlegel", "Patel"} <= studies
    assert j["training"]["corpus"].startswith("McGeary")
