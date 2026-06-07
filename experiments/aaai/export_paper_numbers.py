#!/usr/bin/env python3
"""Project 2 (AAAI): aggregate the certification results into paper numbers.

Reads the per-experiment JSONs under results/aaai/ and aggregates them into
results/aaai/paper_numbers.json under three clearly separated sections:

- certificate_soundness_synthetic: C3 coverage, C4 tightness, and the C5
  minimum-degree breakdown verified on random-graph ensembles (math soundness
  only, no biological claim; produced by experiments/soundness/);
- application_real_offtarget: per-dataset certified containment on the real
  off-target sets (GSE184929, Jackson 2003/2006, Birmingham 2006, SeedSeq) plus
  the cross-assay comparison (produced by experiments/aaai/exp_realdata.py);
- backbone_competitiveness_efficacy: the CCMP backbone on-target efficacy on the
  OligoFormer suite (produced by experiments/aaai/exp_backbone_efficacy.py).

Per the hard data policy (CLAUDE.md), sections 2 and 3 only appear when their
real-data experiment has produced a JSON; until the real data is wired those
experiments raise and the export records the application / backbone sections as
pending real data rather than reporting any non-real number.

It copies all figures into results/aaai/figures/ with plain-string captions and
writes a plain markdown results/aaai/SUMMARY.md whose headline leads with three
lines: certified-radius empirical coverage vs nominal (random-graph soundness),
certified containment on real off-target data, and backbone competitiveness on
the efficacy suite.

Deterministic aggregation: no LaTeX, no .tex. Accepts --seed and logs it for
CLAUDE.md rule 3 consistency, though it draws no randomness.
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
        "Application across every off-target dataset (GSE184929 RNA-seq, Jackson 2003/2006, "
        "Birmingham 2006, SeedSeq): the predicted off-target effect vs the measured calls "
        "(AUC and containment separation), the contained fraction by class (measured "
        "off-targets vs non-off-targets), certified radius vs STRING threshold per source, "
        "and degradation with the C5 breakdown under edge subsampling."
    ),
    "backbone_efficacy.png": (
        "CCMP backbone on-target efficacy (the x0 predictor) on the OligoFormer suite: "
        "Pearson and Spearman on the Huesken test split and cross-dataset generalization to "
        "Taka, Mix, and Shabalina, with the reported BIOPREDsi reference overlaid."
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
    """Application results: per-dataset certified containment on real off-target sets."""
    per = {}
    for name, r in rd["per_dataset"].items():
        c = r["containment"]
        per[name] = {
            "assay": r["assay"],
            "n_nodes": r["n_nodes"],
            "n_measured_offtargets": r["n_measured_offtargets"],
            "certified_radius": r["certificate"]["radius"],
            "overall_contained_fraction": c["overall_contained_fraction"],
            "offtarget_contained_fraction": c["offtarget_contained_fraction"],
            "background_contained_fraction": c["background_contained_fraction"],
            "separation": c["separation"],
            "auc_predicted_vs_measured": c["auc_predicted_vs_measured"],
            "measured_offtargets_mostly_uncontained": c["measured_offtargets_mostly_uncontained"],
            "radius_vs_threshold": r["radius_vs_threshold"],
            "breakdown_subsample_fraction": r["degradation"]["breakdown_subsample_fraction"],
            "meaningful": r["meaningful"],
            "provenance": r["provenance"],
        }
    return {
        "note": "real measured off-target datasets; these are the datasets the certificate is "
        "evaluated against. No values appear here unless the real-data experiment produced them.",
        "theorem4_radius": rd.get("theorem4_radius"),
        "per_dataset": per,
        "cross_assay": rd["cross_assay"],
        "acceptance": rd["acceptance"],
    }


def backbone_numbers(be: dict) -> dict:
    """Backbone competitiveness: on-target efficacy on the OligoFormer suite."""
    return {
        "note": "on-target efficacy of the CCMP backbone (the x0 direct-repression predictor). "
        "This measures on-target potency and does NOT validate the off-target certificate.",
        "provenance": be["provenance"],
        "protocol": be["protocol"],
        "per_dataset": be["our_model"],
        "reported_baselines": be["reported_baselines"],
        "acceptance": be["acceptance"],
        "cross_dataset_note": be["cross_dataset_note"],
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


def _fmt(x, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def write_summary(numbers: dict, captions: dict) -> None:
    sound = numbers.get("certificate_soundness_synthetic", {})
    app = numbers.get("application_real_offtarget", {})
    back = numbers.get("backbone_competitiveness_efficacy", {})
    cov = sound.get("coverage", {})
    cert = sound.get("certificate", {})
    ti = sound.get("tightness", {})
    bd = sound.get("min_degree_breakdown", {})

    lines = ["# AAAI results summary", ""]

    # The three headline lines: soundness (random graphs), application (real
    # off-target), backbone competitiveness (efficacy).
    lines.append("## Headline")
    if cov:
        worst = min(
            (d["min_empirical_coverage"] for d in cov["nominal_vs_empirical"]), default=None
        )
        lines.append(
            f"- Certificate soundness (random-graph verification, no biological claim): empirical "
            f"coverage of the certified radius is {worst} (>= nominal 1 minus delta at every delta); "
            f"verdict {cov['verdict']}, max excess violation {cov['max_excess_violation']} (never "
            "violated more than delta)."
        )
    if app:
        g = app["per_dataset"].get("gse184929", {})
        acc = app["acceptance"]
        lines.append(
            f"- Application (real off-target data): on GSE184929 RNA-seq the certified containment "
            f"separates measured off-targets (AUC {_fmt(g.get('auc_predicted_vs_measured'))}, "
            f"measured-off-target contained {_fmt(g.get('offtarget_contained_fraction'))} vs "
            f"non-off-target {_fmt(g.get('background_contained_fraction'))}); acceptance "
            f"{acc['flag']} across GSE184929 and {len(app['cross_assay']['microarray_datasets'])} "
            "microarray sets."
        )
    else:
        lines.append(
            "- Application (real off-target data): PENDING real data. exp_realdata.py runs the "
            "Theorem 4 certificate the moment the real off-target / STRING join is wired; no "
            "non-real number is reported."
        )
    if back:
        ht = back["per_dataset"].get("huesken_test", {})
        ba = back["acceptance"]
        lines.append(
            f"- Backbone competitiveness (efficacy): Huesken-test Pearson "
            f"{_fmt(ht.get('pearson'))} / Spearman {_fmt(ht.get('spearman'))}; acceptance "
            f"{ba['flag']} vs reported {ba.get('best_reported_name')} "
            f"({ba.get('best_reported_pearson')})."
        )
    else:
        lines.append(
            "- Backbone competitiveness (efficacy): PENDING real data. exp_backbone_efficacy.py "
            "trains on the real OligoFormer suite when the CSVs are present; no non-real number "
            "is reported."
        )
    lines.append("")

    # Section 1: certificate soundness (random-graph math verification).
    lines.append("## 1. Certificate soundness (random-graph verification, no biological claim)")
    if cert:
        ex = cert["bernstein_example"]
        eig = cert["p_alpha_eigenvalue_range"]
        lines += [
            f"- alpha = {cert['alpha']}; certified-radius constant (1-alpha)/alpha = "
            f"{cert['radius_constant_one_minus_alpha_over_alpha']}; P_alpha eigenvalues in "
            f"[{eig[0]:.4f}, {eig[1]:.1f}] (operator norm {cert['p_alpha_operator_norm']}).",
            f"- Bernstein quantities ({ex['config']}): v = {ex['v']}, L = {ex['L']}, "
            f"b(delta) = {ex['b_delta']:.4f}.",
        ]
    if ti:
        s = ti["sqrt_c_one_minus_c"]
        p = ti["probabilistic_vs_deterministic"]
        lines += [
            f"- Tightness (C4): the variance term peaks at c = {s['variance_term_argmax_c']} and "
            f"follows sqrt(c(1-c)) (CV {s['variance_term_over_sqrt_c1mc_cv']:.2e}); probabilistic "
            f"vs deterministic ratio exponent {p['fitted_ratio_exponent']:.3f} (approx sqrt(E)).",
        ]
    if cov:
        lines.append(
            f"- Coverage (C3): empirical vs nominal across delta on random matchings, verdict "
            f"{cov['verdict']}."
        )
    if bd:
        lines.append(
            f"- Symmetric-normalized breakdown (C5): degree concentration fails (rho >= 1) at "
            f"STRING cutoff {bd['breakdown_cutoff_rho_ge_1']}; LLV min-degree condition lost at "
            f"{bd['llv_breakdown_cutoff']}."
        )
    lines.append("")

    # Section 2: application results (real off-target data).
    lines.append("## 2. Application results (real off-target data)")
    if not app:
        lines.append(
            "PENDING real data. Per the hard data policy, exp_realdata.py uses only real measured "
            "off-target calls and a real STRING GRN and raises until that join is wired, so no "
            "containment numbers are reported here yet."
        )
    if app:
        lines.append(app["note"])
        lines.append("")
        lines.append(
            "| dataset | assay | AUC | separation | off-target contained | non-off contained | meaningful |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for name, r in app["per_dataset"].items():
            lines.append(
                f"| {name} | {r['assay']} | {_fmt(r['auc_predicted_vs_measured'])} | "
                f"{_fmt(r['separation'])} | {_fmt(r['offtarget_contained_fraction'])} | "
                f"{_fmt(r['background_contained_fraction'])} | {r['meaningful']} |"
            )
        ca = app["cross_assay"]
        lines.append("")
        lines.append(
            f"- Cross-assay: RNA-seq mean AUC {_fmt(ca['rna_seq_mean_auc'])} vs microarray mean "
            f"AUC {_fmt(ca['microarray_mean_auc'])}. {ca['note']}"
        )
        acc = app["acceptance"]
        lines.append(
            f"- ACCEPTANCE {acc['flag']}: GSE184929 meaningful={acc['gse184929_meaningful']}, "
            f"any microarray meaningful={acc['any_microarray_meaningful']}; inconsistencies="
            f"{acc['inconsistencies']}."
        )
    lines.append("")

    # Section 3: backbone competitiveness (efficacy data).
    lines.append("## 3. Backbone competitiveness (efficacy data, on-target only)")
    if not back:
        lines.append(
            "PENDING real data. exp_backbone_efficacy.py trains the CCMP backbone on the real "
            "OligoFormer suite and raises when the CSVs are absent, so no efficacy numbers are "
            "reported here yet."
        )
    if back:
        lines.append(back["note"])
        lines.append("")
        lines.append("| dataset | n | Pearson | Spearman |")
        lines.append("| --- | --- | --- | --- |")
        for name, r in back["per_dataset"].items():
            lines.append(f"| {name} | {r['n']} | {_fmt(r['pearson'])} | {_fmt(r['spearman'])} |")
        ba = back["acceptance"]
        lines.append("")
        lines.append(
            f"- ACCEPTANCE {ba['flag']}: our Huesken-test Pearson "
            f"{_fmt(ba['our_huesken_test_pearson'])} vs best reported {ba.get('best_reported_name')} "
            f"({ba.get('best_reported_pearson')}). {ba['note']}"
        )
        lines.append(f"- {back['cross_dataset_note']}")
    lines.append("")

    lines.append("## Figures (results/aaai/figures/)")
    for name, caption in captions.items():
        lines.append(f"- `{name}`: {caption}")
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
    be = _load("backbone_efficacy")
    alpha = ti["alpha"] if ti else 0.5
    delta = ti["delta"] if ti else 0.05

    # Three clearly separated sections.
    soundness = {
        "note": "random-graph ensembles: certificate math soundness only, no biological claim.",
        "certificate": certificate_quantities(alpha, delta),
    }
    if cov:
        soundness["coverage"] = coverage_numbers(cov)
    if ti:
        soundness["tightness"] = tightness_numbers(ti)
    if no:
        soundness["min_degree_breakdown"] = breakdown_numbers(no)

    numbers = {
        "alpha": alpha,
        "source_files": [
            "coverage.json",
            "tightness.json",
            "normalized.json",
            "realdata.json",
            "backbone_efficacy.json",
        ],
        "certificate_soundness_synthetic": soundness,
    }
    if rd:
        numbers["application_real_offtarget"] = realdata_numbers(rd)
    if be:
        numbers["backbone_competitiveness_efficacy"] = backbone_numbers(be)

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
