"""Regression test for the AAAI four-theorem paper-numbers aggregation.

Pure stdlib, runs in CI. The export has four theorem sections (T1-T4) plus a real
efficacy baselines block; every section carries a provenance field naming the real
dataset, and a section whose real data is absent is recorded as pending with the
exact source, never a fabricated number.
"""

from __future__ import annotations

import json
from pathlib import Path

AAAI = Path(__file__).resolve().parents[1] / "results" / "aaai"
SECTIONS = (
    "T1_leakage_free_identifiability",
    "T2_contraction_stability",
    "T3_mass_action_constraint",
    "T4_certified_offtarget_bound",
    "baselines_efficacy",
)


def _pn() -> dict:
    return json.loads((AAAI / "paper_numbers.json").read_text())


def test_four_theorem_sections_plus_baselines_present() -> None:
    pn = _pn()
    for section in SECTIONS:
        assert section in pn
    assert "data_policy" in pn and "REAL DATA ONLY" in pn["data_policy"]


def test_every_section_has_provenance() -> None:
    pn = _pn()
    for section in SECTIONS:
        block = pn[section]
        assert block.get("provenance"), f"{section} missing provenance"
        assert isinstance(block["provenance"], str) and block["provenance"]
        # Each section is either real ('ok') or honestly pending, never fabricated.
        assert block["status"] in {"ok", "pending_real_data", "absent"}


def test_t2_real_numbers_from_trrust() -> None:
    t2 = _pn()["T2_contraction_stability"]
    assert t2["status"] == "ok"
    assert "TRRUST" in t2["provenance"]  # the real signed/directed GRN
    assert abs(t2["rho_S"] - 1.0) < 1e-6  # row-normalized transition: rho <= 1
    assert t2["operator_2norm_S"] > t2["rho_S"]  # non-normal: ||S||_2 >> rho
    assert t2["certified_contraction_constant_l2_rho"] > 0
    assert t2["inf_norm_nonexpansive"] is True  # the inf-norm certificate holds
    assert len(t2["alpha_sweep"]) >= 5
    assert "signed_spectral_gap" in t2 and "dirichlet_energies" in t2


def test_pending_sections_name_their_source() -> None:
    pn = _pn()
    # T1, T3, baselines carry the exact fetch source when pending.
    for section in (
        "T1_leakage_free_identifiability",
        "T3_mass_action_constraint",
        "baselines_efficacy",
    ):
        block = pn[section]
        if block["status"] == "pending_real_data":
            assert block.get("source")
    t4 = pn["T4_certified_offtarget_bound"]
    if t4["status"] == "pending_real_data":
        # Each real off-target dataset is named with its assay and source.
        assert {"gse184929", "jackson2006", "birmingham2006"} <= set(t4["pending"])
        for d in t4["pending"].values():
            assert d["assay"] and d["source"]


def test_summary_headline_leads_with_the_four_items() -> None:
    captions = json.loads((AAAI / "figures" / "captions.json").read_text())
    assert all(isinstance(v, str) and v for v in captions.values())

    summary = (AAAI / "SUMMARY.md").read_text()
    assert summary.startswith("# AAAI")
    head = "\n".join(summary.splitlines()[:8])
    # The four required headline items: baselines, T2 constant, T3 residual, T4 binds.
    assert "Baselines (real Huesken efficacy)" in head
    assert "T2 (real TRRUST GRN)" in head
    assert "T3 (real Huesken + ViennaRNA)" in head
    assert "T4 (real GSE184929 RNA-seq)" in head
