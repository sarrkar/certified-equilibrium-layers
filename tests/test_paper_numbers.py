"""Regression test for the AAAI paper-numbers aggregation. Pure stdlib, runs in CI."""

from __future__ import annotations

import json
from pathlib import Path

AAAI = Path(__file__).resolve().parents[1] / "results" / "aaai"


def test_paper_numbers_aggregate() -> None:
    pn = json.loads((AAAI / "paper_numbers.json").read_text())
    # Certificate constants.
    cert = pn["certificate"]
    assert cert["radius_constant_one_minus_alpha_over_alpha"] == (1.0 - pn["alpha"]) / pn["alpha"]
    assert cert["p_alpha_operator_norm"] == 1.0
    lo, hi = cert["p_alpha_eigenvalue_range"]
    assert lo == pn["alpha"] / (2.0 - pn["alpha"]) and hi == 1.0
    for key in ("v", "L", "b_delta"):
        assert cert["bernstein_example"][key] > 0

    # Coverage: empirical at or above nominal, verdict PASS.
    cov = pn["coverage"]
    assert cov["verdict"] == "PASS"
    for d in cov["nominal_vs_empirical"]:
        assert d["min_empirical_coverage"] >= d["nominal_coverage"]

    # Tightness and breakdown headline numbers.
    assert pn["tightness"]["sqrt_c_one_minus_c"]["variance_term_argmax_c"] == 0.5
    assert pn["tightness"]["probabilistic_vs_deterministic"]["approx_sqrt_e"] is True
    assert pn["min_degree_breakdown"]["breakdown_cutoff_rho_ge_1"] is not None

    # Real-data containment and degradation present.
    cont = pn["real_data"]["certified_containment"]
    assert cont["tau_for_90pct_certified"] is not None
    assert pn["real_data"]["degradation"]["breakdown_subsample_fraction"] is not None
    assert len(pn["real_data"]["monte_carlo_cost"]["per_delta"]) > 0


def test_figure_captions_and_summary_exist() -> None:
    captions = json.loads((AAAI / "figures" / "captions.json").read_text())
    assert len(captions) >= 5
    assert all(isinstance(v, str) and v for v in captions.values())  # plain-string captions
    summary = (AAAI / "SUMMARY.md").read_text()
    assert summary.startswith("# AAAI")
    # Leads with the empirical coverage result and the certified-containment fraction.
    head = "\n".join(summary.splitlines()[:6])
    assert "Empirical coverage" in head
    assert "containment fraction" in head
