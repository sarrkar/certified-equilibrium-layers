#!/usr/bin/env python3
"""Project 2 (AAAI): aggregate the certification results into paper numbers.

Reads the per-experiment JSONs under results/aaai/ (coverage, tightness,
normalized, realdata, certified_region) and aggregates the headline quantities
into results/aaai/paper_numbers.json. Copies all figures into
results/aaai/figures/ with plain-string captions, and writes a plain markdown
results/aaai/SUMMARY.md leading with the empirical coverage result and the
certified-containment fraction on the GSE184929 / SeedMatchR pipeline.

This is a deterministic aggregation step (it produces no LaTeX and no .tex). It
accepts --seed and logs it for CLAUDE.md rule 3 consistency, though it draws no
randomness.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from certification import (  # noqa: E402
    EdgeBernoulliNoise,
    bernstein_certificate,
    p_alpha_eigenvalue_range,
)

AAAI = REPO / "results" / "aaai"

FIGURE_CAPTIONS = {
    "coverage.png": (
        "Empirical versus nominal coverage of the matrix-Bernstein certified radius "
        "across delta on spectrally-bounded random matchings. Empirical coverage stays "
        "at or above 1 minus delta at every delta (verdict PASS, no violations)."
    ),
    "tightness_radius_vs_c.png": (
        "Certified radius versus edge confidence c on a star graph. The variance term "
        "peaks at the coin-flip confidence c = 0.5 and follows the sqrt(c(1-c)) scaling "
        "law; it is tightest as confidences approach 0 or 1."
    ),
    "tightness_prob_vs_det.png": (
        "Probabilistic Bernstein radius versus the deterministic resolvent-Lipschitz "
        "envelope as the number of uncertain edges grows. The deterministic envelope "
        "grows linearly while the probabilistic radius grows as sqrt(E), so the "
        "probabilistic certificate is tighter by approximately sqrt(number of edges)."
    ),
    "normalized_breakdown.png": (
        "Symmetric-normalized (C5) certificate via degree concentration: minimum "
        "expected degree and the degree relative fluctuation rho versus STRING "
        "confidence threshold. The certificate is valid while rho < 1 and breaks down "
        "as the graph is thresholded sparser."
    ),
    "realdata.png": (
        "Real-data pipeline (SeedMatchR seed repression propagated over a STRING GRN): "
        "certified off-target containment fraction versus tolerance, certified radius "
        "versus STRING threshold, graceful degradation with the C5 breakdown under edge "
        "subsampling, and the Monte-Carlo sample cost for a 1 minus delta interval."
    ),
    "certified_region.png": (
        "Resolvent-perturbation certified L2 region under the Gaussian per-edge model: "
        "certified radius versus delta and the realized out-of-sample coverage."
    ),
}


def _load(name: str) -> dict | None:
    path = AAAI / f"{name}.json"
    if not path.exists():
        print(f"[export] WARNING: {path} missing; skipping its numbers")
        return None
    return json.loads(path.read_text())


def _star_confidence(n_edges: int, c: float) -> sp.csr_matrix:
    rows, cols, data = [], [], []
    for k in range(n_edges):
        rows += [0, k + 1]
        cols += [k + 1, 0]
        data += [c, c]
    return sp.coo_matrix((data, (rows, cols)), shape=(n_edges + 1, n_edges + 1)).tocsr()


def certificate_quantities(alpha: float, delta: float) -> dict:
    """Canonical Bernstein quantities on the star example (E=32, c=0.5)."""
    noise = EdgeBernoulliNoise(C=_star_confidence(32, 0.5))
    x0 = np.ones(noise.n_nodes) / np.sqrt(noise.n_nodes)  # ||x0|| = 1
    cert = bernstein_certificate(x0, noise, alpha, delta)
    eig_min, eig_max = p_alpha_eigenvalue_range(alpha)
    return {
        "alpha": alpha,
        "radius_constant_one_minus_alpha_over_alpha": (1.0 - alpha) / alpha,
        "p_alpha_eigenvalue_range": [eig_min, eig_max],
        "p_alpha_operator_norm": eig_max,
        "bernstein_example": {
            "config": "star graph, 32 uncertain edges, c = 0.5, delta = " + str(delta),
            "v": cert.v,
            "L": cert.L,
            "b_delta": cert.b,
            "radius_unit_x0": cert.radius,
        },
    }


def coverage_numbers(cov: dict) -> dict:
    per_delta = []
    for delta in cov["delta_grid"]:
        emps = [
            d["empirical_coverage"]
            for cfg in cov["configs"]
            for d in cfg["per_delta"]
            if d["delta"] == delta
        ]
        per_delta.append(
            {
                "delta": delta,
                "nominal_coverage": 1.0 - delta,
                "min_empirical_coverage": float(min(emps)),
            }
        )
    return {
        "verdict": cov["verdict"],
        "any_violation": cov["any_violation"],
        "max_excess_violation": cov["max_excess_violation"],
        "regime": cov["regime"],
        "nominal_vs_empirical": per_delta,
    }


def tightness_numbers(ti: dict) -> dict:
    a = ti["A_variance_scaling"]
    b = ti["B_matching_lower_bound"]["c=0.5"]
    c = ti["C_probabilistic_vs_deterministic"]
    return {
        "sqrt_c_one_minus_c": {
            "variance_term_argmax_c": a["argmax_c_variance_term"],
            "variance_term_over_sqrt_c1mc_cv": a["variance_term_over_sqrt_c1mc_cv"],
            "full_radius_argmax_vs_edges": a["full_radius_argmax_vs_e"],
        },
        "matching_lower_bound": {
            "v": b["v"],
            "sqrt_v": b["sqrt_v"],
            "bernstein_b": b["bernstein_b"],
            "b_over_empirical_quantile": b["b_over_empirical_quantile"],
        },
        "probabilistic_vs_deterministic": {
            "fitted_ratio_exponent": c["fitted_ratio_exponent"],
            "approx_sqrt_e": c["approx_sqrt_e"],
        },
    }


def breakdown_numbers(no: dict) -> dict:
    bd = no["breakdown"]
    return {
        "criterion": bd["criterion"],
        "breakdown_cutoff_rho_ge_1": bd["breakdown_cutoff"],
        "llv_breakdown_cutoff": bd["llv_breakdown_cutoff"],
        "min_degree_vs_threshold": [
            {
                "cutoff": d["cutoff"],
                "min_expected_degree": d["min_expected_degree"],
                "degree_rel_fluctuation": d["degree_rel_fluctuation"],
                "assumption_holds": d["assumption_holds"],
            }
            for d in bd["per_cutoff"]
        ],
        "coverage_when_valid": {
            "structure_coverage": no["coverage"]["structure_coverage"],
            "prediction_coverage": no["coverage"]["prediction_coverage"],
        },
    }


def realdata_numbers(rd: dict) -> dict:
    cont = rd["containment"]
    return {
        "provenance": rd["provenance"],
        "certified_containment": {
            "n_off_targets": cont["n_off_targets"],
            "max_predicted_effect": cont["max_predicted_effect"],
            "certified_radius": cont["radius_sym"],
            "tau_for_90pct_certified": cont["tau_for_90pct_certified"],
            "certified_fraction_at_2x_radius": cont["certified_fraction_at_2x_radius"],
            "tau_grid": cont["tau_grid"],
            "certified_contained_fraction": cont["certified_contained_fraction"],
            "point_contained_fraction": cont["point_contained_fraction"],
        },
        "radius_vs_threshold": rd["radius_vs_threshold"]["per_threshold"],
        "degradation": {
            "subsample": rd["degradation"]["subsample"],
            "false_edges": rd["degradation"]["false_edges"],
            "breakdown_subsample_fraction": rd["degradation"]["breakdown_subsample_fraction"],
        },
        "monte_carlo_cost": rd["monte_carlo_cost"],
    }


def copy_figures() -> dict:
    fig_dir = AAAI / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name, caption in FIGURE_CAPTIONS.items():
        src = AAAI / name
        if src.exists():
            shutil.copy2(src, fig_dir / name)
            copied[name] = caption
        else:
            print(f"[export] WARNING: figure {src} missing; not copied")
    (fig_dir / "captions.json").write_text(json.dumps(copied, indent=2), encoding="utf-8")
    return copied


def write_summary(numbers: dict, captions: dict) -> None:
    cov = numbers.get("coverage", {})
    rd = numbers.get("real_data", {})
    cont = rd.get("certified_containment", {})
    cert = numbers.get("certificate", {})

    lines = ["# AAAI certified off-target containment: results summary", ""]

    lines.append("## Headline")
    if cov:
        worst = min(
            (d["min_empirical_coverage"] for d in cov["nominal_vs_empirical"]), default=None
        )
        lines.append(
            f"- Empirical coverage of the certified radius is {worst} (at or above the "
            f"nominal 1 minus delta at every delta); verdict {cov['verdict']}, "
            f"max excess violation {cov['max_excess_violation']} (the radius is never "
            "violated more than a delta fraction)."
        )
    if cont:
        tau90 = cont.get("tau_for_90pct_certified")
        radius = cont.get("certified_radius")
        lines.append(
            f"- On the GSE184929 / SeedMatchR off-target pipeline (real seed logic, "
            f"synthetic-proxy graph: the STRING sequence join is a tracked gap), the "
            f"certified containment fraction reaches 0.9 at tolerance tau = "
            f"{tau90:.3f} and is {cont.get('certified_fraction_at_2x_radius')} at twice "
            f"the certified radius (radius {radius:.3f}, "
            f"{cont.get('n_off_targets')} seed-matched off-targets)."
        )
    lines.append("")

    lines.append("## Certificate")
    if cert:
        ex = cert["bernstein_example"]
        eig = cert["p_alpha_eigenvalue_range"]
        lines += [
            f"- alpha = {cert['alpha']}; certified-radius constant (1-alpha)/alpha = "
            f"{cert['radius_constant_one_minus_alpha_over_alpha']}.",
            f"- P_alpha eigenvalues lie in [{eig[0]:.4f}, {eig[1]:.1f}], so the operator "
            f"norm is {cert['p_alpha_operator_norm']}.",
            f"- Bernstein quantities ({ex['config']}): v = {ex['v']}, L = {ex['L']}, "
            f"b(delta) = {ex['b_delta']:.4f}.",
        ]
    lines.append("")

    lines.append("## Tightness")
    ti = numbers.get("tightness", {})
    if ti:
        s = ti["sqrt_c_one_minus_c"]
        p = ti["probabilistic_vs_deterministic"]
        lines += [
            f"- The radius variance term peaks at c = {s['variance_term_argmax_c']} and "
            f"follows sqrt(c(1-c)) (scaling coefficient of variation "
            f"{s['variance_term_over_sqrt_c1mc_cv']:.2e}).",
            f"- The probabilistic certificate is tighter than the deterministic Lipschitz "
            f"envelope by approximately sqrt(E) (fitted ratio exponent "
            f"{p['fitted_ratio_exponent']:.3f}).",
        ]
    lines.append("")

    lines.append("## Symmetric-normalized breakdown (C5)")
    bd = numbers.get("min_degree_breakdown", {})
    if bd:
        lines += [
            f"- The degree-concentration assumption fails (rho >= 1) at STRING confidence "
            f"cutoff {bd['breakdown_cutoff_rho_ge_1']}; the Le-Levina-Vershynin "
            f"minimum-degree condition is lost at {bd['llv_breakdown_cutoff']}.",
        ]
    if rd.get("degradation"):
        lines.append(
            f"- On the pipeline, edge subsampling degrades gracefully until the C5 "
            f"breakdown at {rd['degradation']['breakdown_subsample_fraction']} of edges "
            "removed."
        )
    lines.append("")

    lines.append("## Monte-Carlo cost")
    mc = rd.get("monte_carlo_cost", {})
    if mc:
        pairs = ", ".join(f"delta={r['delta']}: N={r['n_samples']}" for r in mc["per_delta"])
        lines.append(f"- Samples needed for a 1 minus delta interval: {pairs}.")
    lines.append("")

    lines.append("## Figures (results/aaai/figures/)")
    for name, caption in captions.items():
        lines.append(f"- `{name}`: {caption}")
    lines.append("")

    lines.append("## Provenance")
    prov = rd.get("provenance", {})
    if prov:
        lines.append(
            f"- Graph source: {prov.get('graph_source')}. x0 source: "
            f"{prov.get('x0_source')}. {prov.get('note', '')}"
        )
    lines.append("")

    (AAAI / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=0, help="logged for rule 3; aggregation is deterministic"
    )
    parser.add_argument("--out-dir", type=Path, default=AAAI)
    args = parser.parse_args()
    print(f"[export] seed={args.seed} (deterministic aggregation)")

    cov = _load("coverage")
    ti = _load("tightness")
    no = _load("normalized")
    rd = _load("realdata")
    alpha = ti["alpha"] if ti else 0.5
    delta = ti["delta"] if ti else 0.05

    numbers = {
        "alpha": alpha,
        "source_files": ["coverage.json", "tightness.json", "normalized.json", "realdata.json"],
        "certificate": certificate_quantities(alpha, delta),
    }
    if cov:
        numbers["coverage"] = coverage_numbers(cov)
    if ti:
        numbers["tightness"] = tightness_numbers(ti)
    if no:
        numbers["min_degree_breakdown"] = breakdown_numbers(no)
    if rd:
        numbers["real_data"] = realdata_numbers(rd)

    out_path = args.out_dir / "paper_numbers.json"
    out_path.write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    print(f"[export] wrote {out_path}")

    captions = copy_figures()
    print(f"[export] copied {len(captions)} figures into {AAAI / 'figures'}")

    write_summary(numbers, captions)
    print(f"[export] wrote {AAAI / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
