"""Regression test for the AAAI paper-numbers aggregation. Pure stdlib, runs in CI.

Under the hard data policy, the certificate-soundness section (random-graph math
verification) is always present, while the application (real off-target) and
backbone (efficacy) sections appear ONLY when their real-data experiment has
produced a JSON. Until the real data is wired they are pending, not fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path

AAAI = Path(__file__).resolve().parents[1] / "results" / "aaai"


def test_soundness_present_application_and_backbone_pending() -> None:
    pn = json.loads((AAAI / "paper_numbers.json").read_text())
    # Certificate soundness is always present (math verification on random graphs).
    assert "certificate_soundness_synthetic" in pn
    # No real-data results are reported until the real experiments run; the policy
    # forbids a generated-data stand-in for these sections.
    assert "application_real_offtarget" not in pn
    assert "backbone_competitiveness_efficacy" not in pn


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


def test_summary_leads_with_three_lines() -> None:
    captions = json.loads((AAAI / "figures" / "captions.json").read_text())
    assert all(isinstance(v, str) and v for v in captions.values())
    # The certificate-soundness figures are present and captioned.
    assert "coverage.png" in captions

    summary = (AAAI / "SUMMARY.md").read_text()
    assert summary.startswith("# AAAI")
    head = "\n".join(summary.splitlines()[:8])
    # The three required headline lines.
    assert "Certificate soundness" in head
    assert "Application (real off-target data)" in head
    assert "Backbone competitiveness (efficacy)" in head
    # Sections 2 and 3 are honestly marked pending real data.
    assert "Application (real off-target data): PENDING real data" in summary
    assert "Backbone competitiveness (efficacy): PENDING real data" in summary
