"""Regression test for the AAAI paper-numbers aggregation. Pure stdlib, runs in CI."""

from __future__ import annotations

import json
from pathlib import Path

AAAI = Path(__file__).resolve().parents[1] / "results" / "aaai"


def test_three_sections_present() -> None:
    pn = json.loads((AAAI / "paper_numbers.json").read_text())
    assert "certificate_soundness_synthetic" in pn
    assert "application_real_offtarget" in pn
    assert "backbone_competitiveness_efficacy" in pn


def test_certificate_soundness_synthetic() -> None:
    pn = json.loads((AAAI / "paper_numbers.json").read_text())
    sound = pn["certificate_soundness_synthetic"]
    cert = sound["certificate"]
    assert cert["radius_constant_one_minus_alpha_over_alpha"] == (1.0 - pn["alpha"]) / pn["alpha"]
    assert cert["p_alpha_operator_norm"] == 1.0
    lo, hi = cert["p_alpha_eigenvalue_range"]
    assert lo == pn["alpha"] / (2.0 - pn["alpha"]) and hi == 1.0
    for key in ("v", "L", "b_delta"):
        assert cert["bernstein_example"][key] > 0
    # C3 coverage at or above nominal; C4 tightness; C5 breakdown.
    cov = sound["coverage"]
    assert cov["verdict"] == "PASS"
    for d in cov["nominal_vs_empirical"]:
        assert d["min_empirical_coverage"] >= d["nominal_coverage"]
    assert sound["tightness"]["sqrt_c_one_minus_c"]["variance_term_argmax_c"] == 0.5
    assert sound["tightness"]["probabilistic_vs_deterministic"]["approx_sqrt_e"] is True
    assert sound["min_degree_breakdown"]["breakdown_cutoff_rho_ge_1"] is not None


def test_application_real_offtarget() -> None:
    pn = json.loads((AAAI / "paper_numbers.json").read_text())
    app = pn["application_real_offtarget"]
    assert set(app["per_dataset"]) == {
        "gse184929",
        "jackson2003",
        "jackson2006",
        "birmingham2006",
        "seedseq",
    }
    for r in app["per_dataset"].values():
        # Per-dataset certified-containment fraction by class.
        assert "offtarget_contained_fraction" in r
        assert "background_contained_fraction" in r
        assert "auc_predicted_vs_measured" in r
    ca = app["cross_assay"]
    assert ca["rna_seq_mean_auc"] >= ca["microarray_mean_auc"]  # reported gap
    assert app["acceptance"]["flag"] in {"PASS", "FAIL"}


def test_backbone_competitiveness_efficacy() -> None:
    pn = json.loads((AAAI / "paper_numbers.json").read_text())
    back = pn["backbone_competitiveness_efficacy"]
    assert set(back["per_dataset"]) == {"huesken_test", "taka", "mix", "shabalina"}
    for r in back["per_dataset"].values():
        assert "pearson" in r and "spearman" in r
    assert back["acceptance"]["flag"] in {"PASS", "FAIL", "PENDING"}
    # Baselines are cited, not fabricated.
    assert any(b["name"].startswith("BIOPREDsi") for b in back["reported_baselines"])


def test_summary_leads_with_three_lines() -> None:
    captions = json.loads((AAAI / "figures" / "captions.json").read_text())
    assert len(captions) >= 6
    assert all(isinstance(v, str) and v for v in captions.values())
    assert "backbone_efficacy.png" in captions

    summary = (AAAI / "SUMMARY.md").read_text()
    assert summary.startswith("# AAAI")
    head = "\n".join(summary.splitlines()[:8])
    # The three required headline lines.
    assert "Certificate soundness (synthetic)" in head
    assert "Application (real off-target data)" in head
    assert "Backbone competitiveness (efficacy)" in head
