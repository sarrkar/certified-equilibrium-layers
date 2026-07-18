"""Paper result-set manifest: criterion 2 PASS with certificate, criterion 1 FAIL twice.

Reads the committed results/cascade_ot/paper_result_set.json and verifies it faithfully mirrors
its four source gate results (no drift) and states the demarcation: criterion 2 PASSES with a
certificate, criterion 1 FAILS twice pre-registered, the generator stays gated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CO = REPO / "results" / "cascade_ot"
MANIFEST = CO / "paper_result_set.json"


def _load() -> dict:
    if not MANIFEST.exists():
        pytest.skip("paper_result_set.json not built yet")
    return json.loads(MANIFEST.read_text())


def test_criterion_2_pass_with_certificate() -> None:
    m = _load()
    c2 = m["results"]["criterion_2_modification_discrimination"]
    assert c2["verdict"] == "PASS"
    assert c2["with_certificate"] is True
    sd = c2["evidence"]["subgate_d4d1"]
    assert sd["D4_lt_D1"] is True and sd["monotone_E_D4"] < sd["monotone_E_D1"]
    assert sd["non_monotone_control_D4_lt_D1"] is False  # count-only cannot
    assert "linear in occupancy" in sd["certificate"]
    ip = c2["evidence"]["ift_propagation"]
    assert ip["verdict"] == "PASS" and ip["unique_fixed_point"] is True
    assert ip["ift_grad_vs_finite_diff_max_rel_error"] < 1e-5


def test_criterion_1_fails_twice_pre_registered() -> None:
    m = _load()
    c1 = m["results"]["criterion_1_transfer"]
    assert c1["verdict"] == "FAIL"
    assert c1["failed_twice_pre_registered"] is True
    assert c1["attempts"]["transfer_gate_count_based"]["verdict"] == "FAIL"
    assert c1["attempts"]["criterion1_retrain_learned"]["verdict"] == "FAIL"
    # both carry a pre-registered threshold block.
    assert "decision_rule" in c1["attempts"]["transfer_gate_count_based"]["pre_registered"]
    assert "decision_rule" in c1["attempts"]["criterion1_retrain_learned"]["pre_registered"]


def test_manifest_matches_sources() -> None:
    m = _load()
    # Every quoted verdict must match the live source file (no drift).
    sub = json.loads((CO / "subgate_d4d1.json").read_text())
    ift = json.loads((CO / "ift_check.json").read_text())
    tg = json.loads((CO / "transfer_gate.json").read_text())
    cr = json.loads((CO / "criterion1_retrain.json").read_text())
    c2 = m["results"]["criterion_2_modification_discrimination"]["evidence"]
    assert c2["subgate_d4d1"]["subgate"] == sub["subgate"]
    assert c2["ift_propagation"]["verdict"] == ift["verdict"]
    att = m["results"]["criterion_1_transfer"]["attempts"]
    assert att["transfer_gate_count_based"]["verdict"] == tg["verdict"]
    assert att["criterion1_retrain_learned"]["verdict"] == cr["verdict"]
    assert att["criterion1_retrain_learned"]["margins"] == cr["margins"]


def test_demarcation_and_generator_gated() -> None:
    m = _load()
    d = m["demarcation"]
    assert "PASS with certificate" in d["criterion_2_modification_discrimination"]
    assert "FAIL twice" in d["criterion_1_transfer"]
    assert "independent-study count" in d["binding_constraint"]
    assert "not model" in d["binding_constraint"]
    assert m["stage_g_generator"]["status"] == "GATED"
