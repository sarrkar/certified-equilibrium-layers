"""1B decision-relevant capability example on REAL measured data (Candidate 1: miR-122 titration).

The decision: two competitor-loading conditions A and B on the SAME miR-122 reporter (same focal-target
site, same guide pool), differing only in how many competing target sites are added to the cell.
  A = 3e4 added sites/cell (the MEASURED effective endogenous target-site abundance, below threshold);
  B = 4.5e5 added sites/cell (the MEASURED derepression IC50, half-maximal by definition).
An independent per-target scorer sees only (guide, reporter); it takes no competitor-abundance argument,
so its prediction is IDENTICAL for A and B and its d(pred)/d(x_competitor) is exactly 0 for both. The
conservation-law operator couples the reporter to the competitor through the shared free guide f*, so it
predicts B is more derepressed than A. The MEASURED Denzler biology resolves the ordering.

FRAMING (honest): this is a BLINDNESS-FILLED capability confirmation, NOT a passed falsification test.
The operator's reporter derepression is monotone increasing in competitor abundance by the M-matrix
property itself (the reduced residual Jacobian D = 1 + sum x sigma'(f) >= 1 > 0), so the operator ordering
CANNOT disagree with a monotone measured titration. The value is that the independent scorer is
structurally blind (D = 0 for A and B) while the operator plus the measured biology both say B > A. The
only pair that could have CONTRADICTED the operator is the cross-system miR-122 vs miR-92/25 comparison
(Candidate 2), which needs a fictional shared reporter and is therefore recorded as a NON-GROUNDED note,
not claimed.

REUSED, NOT REBUILT: solve_f (offtarget_phase_probe.py) for the single-pool equilibrium; occ_langmuir
(theory_realdata_verification.py) for theta and sigma'; BASELINES + load_systems (cerna_derepression.py)
for the independent scorers and the VERIFY-marked measured config (loader RAISES on any missing field).

REAL DATA ONLY (Denzler 2014/2016, Bosson 2014; every measured value cites its study). Deterministic
(bisection, no random, CPU); bitwise-identical on rerun. No em-dashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, "/home/shadi/mirna_baselines/scripts")
from offtarget_phase_probe import solve_f  # noqa: E402 (REUSED bisection solver)

from cerna_derepression import (  # noqa: E402 (REUSED scorers + config loader)
    BASELINES,
    load_systems,
)
from theory_realdata_verification import occ_langmuir  # noqa: E402 (REUSED Langmuir Jacobian)

CONFIG = REPO / "scripts/cerna_derepression_config.json"
OUT_JSON = REPO / "results/decision_example.json"

# A/B competitor abundances (copies/cell), BOTH taken from Denzler MEASURED anchors (not constructed).
X_A = 3.0e4
X_B = 4.5e5
SRC_A = (
    "Denzler et al. 2016 Mol Cell 64:565 (PMID 27871486) Fig 7A: effective endogenous target-site "
    "abundance ~3e4 copies/cell, which is BELOW the threshold-like release at >=1.5e5 added sites "
    "measured in Denzler et al. 2014 Mol Cell 54:766 (PMID 24793693)"
)
SRC_B = (
    "Denzler et al. 2016 Mol Cell 64:565 (PMID 27871486): miR-122 derepression IC50 = 4.5e5 "
    "sites/cell, i.e. half-maximal derepression (fraction ~0.5) at this added-site abundance"
)


def _sites(system: dict, x_comp: float) -> tuple[np.ndarray, np.ndarray]:
    xs = [x for x, _ in system["background"]] + [system["reporter_x"]]
    ks = [k for _, k in system["background"]] + [system["reporter_K"]]
    if x_comp > 0.0:
        xs.append(x_comp)
        ks.append(system["competitor_K"])
    return np.asarray(xs, dtype=float), np.asarray(ks, dtype=float)


def reporter_state(system: dict, x_comp: float) -> dict:
    """Reporter occupancy theta_rep and the reduced residual Jacobian D at the shared-pool equilibrium."""
    xs, ks = _sites(system, x_comp)
    f = float(solve_f(system["M"], xs, ks))  # REUSED solver
    theta_rep, _sig_rep = occ_langmuir(f, system["reporter_K"])  # REUSED: theta = f/(K+f)
    _theta_all, sig_all = occ_langmuir(f, ks)  # REUSED: sigma'(f) = K/(K+f)^2
    d_jac = 1.0 + float(np.sum(xs * sig_all))  # M-matrix reduced Jacobian, >= 1 > 0
    return {"free_guide": f, "theta_rep": float(theta_rep), "D_jacobian": d_jac}


def derepression(system: dict, x_comp: float) -> dict:
    base = reporter_state(system, 0.0)
    now = reporter_state(system, x_comp)
    return {
        "x_competitor": x_comp,
        "theta_rep": now["theta_rep"],
        "free_guide": now["free_guide"],
        "D_jacobian": now["D_jacobian"],
        "derepression_fraction": 1.0 - now["theta_rep"] / base["theta_rep"],
    }


def scorer_predictions(system: dict) -> dict:
    """The independent per-target scorers are functions of (guide, reporter) ONLY; identical for A and B
    and with d(score)/d(x_competitor) = 0 exactly, since none of them takes a competitor argument."""
    return {name: float(fn(system)) for name, fn in BASELINES.items()}


def main() -> int:
    config = json.loads(CONFIG.read_text())
    systems, _defs = load_systems(config)  # REUSED loader (RAISES on missing field)
    by_name = {s["name"]: s for s in systems}
    mir122 = by_name["miR-122"]
    mir9225 = by_name["miR-92/25"]

    # ---- (i) independent scorer: identical for A and B, competitor-derivative exactly 0 ----
    scorer = scorer_predictions(mir122)
    scorer_A, scorer_B = scorer, dict(scorer)
    scorer_identical = scorer_A == scorer_B
    scorer_dpred_dx = {name: 0.0 for name in scorer}  # no competitor argument -> exact 0

    # ---- (ii) operator: derepression of the miR-122 reporter at A and B ----
    op_A = derepression(mir122, X_A)
    op_B = derepression(mir122, X_B)
    op_orders_B_over_A = op_B["derepression_fraction"] > op_A["derepression_fraction"]

    # ---- (iii) measured ordering from Denzler ----
    measured = {
        "A": {
            "x_competitor": X_A,
            "derepression": "sub-threshold (below the >=1.5e5 release threshold); "
            "far below half-maximal",
            "value_bound": "< 0.5",
            "source": SRC_A,
        },
        "B": {
            "x_competitor": X_B,
            "derepression": "half-maximal by definition of the measured IC50",
            "value": 0.5,
            "source": SRC_B,
        },
        "ordering": "derepression(B) > derepression(A)",
        "resolved": True,
    }
    measured_orders_B_over_A = True  # 0.5 (IC50) > sub-threshold (below 1.5e5)

    match = op_orders_B_over_A and measured_orders_B_over_A
    verdict = "WIN" if match else ("CONTRADICTION" if measured["resolved"] else "INCONCLUSIVE")

    # ---- Candidate 2: cross-system prediction, preserved as an explicitly NON-GROUNDED note ----
    candidate2 = {
        "label": "NON_GROUNDED (requires a shared-reporter construct; NOT claimed as grounded)",
        "prediction": "At a FIXED added-competitor abundance the operator predicts miR-92/25 (small guide "
        "pool) is derepressed while miR-122 (large guide pool) is not, because derepression turns on when "
        "added sites approach the guide pool. This is the FALSIFIABLE cross-system ordering.",
        "measured_thresholds": {
            "miR-92/25": {
                "threshold_sites": mir9225["measured_threshold"],
                "source": mir9225["measured_threshold_source"],
            },
            "miR-122": {
                "threshold_sites": mir122["measured_threshold"],
                "source": mir122["measured_threshold_source"],
            },
            "measured_ordering": "miR-92/25 threshold (~3000 sites) << miR-122 threshold (~1.5e5 sites), "
            "so at a fixed added-site level between them only miR-92/25 is derepressed (matches operator)",
        },
        "why_not_grounded": "Requirement (i) needs A and B to share the SAME focal-target sites so the "
        "independent scorer is indifferent. miR-122 and miR-92/25 are different guides and cannot share a "
        "reporter site; the identical reporter in the config is a modeling construct, so this is recorded "
        "for a future camera-ready that finds a shared-reporter dataset, and is NOT counted as evidence.",
    }

    result = {
        "example": "1B decision-relevant capability (Candidate 1: miR-122 competitor titration, grounded)",
        "reused_components": {
            "solver": "solve_f(M, x, K) [/home/shadi/mirna_baselines/scripts/offtarget_phase_probe.py]",
            "jacobian": "occ_langmuir(f, K) -> (f/(K+f), K/(K+f)**2) [scripts/theory_realdata_verification.py]",
            "scorers_and_loader": "BASELINES, load_systems [scripts/cerna_derepression.py] (RAISES on missing)",
        },
        "system": {
            "name": mir122["name"],
            "source": mir122["source"],
            "guide_total_M": mir122["M"],
            "reporter_K": mir122["reporter_K"],
            "reporter_x": mir122["reporter_x"],
            "competitor_K": mir122["competitor_K"],
            "background_sites": [{"abundance": x, "K": k} for x, k in mir122["background"]],
        },
        "AB_definition": {
            "shared": "identical miR-122 reporter site and guide pool; A and B differ ONLY in added "
            "competitor abundance, so an independent per-target scorer receives identical inputs",
            "A": {
                "x_competitor": X_A,
                "meaning": "measured effective endogenous target-site abundance",
                "source": SRC_A,
            },
            "B": {
                "x_competitor": X_B,
                "meaning": "measured derepression IC50 (half-maximal)",
                "source": SRC_B,
            },
        },
        "independent_scorer": {
            "predictions_A": scorer_A,
            "predictions_B": scorer_B,
            "identical_for_A_and_B": scorer_identical,
            "dpred_dx_competitor": scorer_dpred_dx,
            "note": "each scorer is a function of (guide, reporter) only; it has no competitor-abundance "
            "argument, so it is blind to the A/B distinction (derivative exactly 0)",
        },
        "operator": {
            "A": op_A,
            "B": op_B,
            "predicted_ordering": "derepression(B) > derepression(A)"
            if op_orders_B_over_A
            else "derepression(B) <= derepression(A)",
            "monotone_by_construction": "D_jacobian >= 1 > 0 at both A and B (reduced residual Jacobian, "
            "M-matrix); reporter derepression is monotone increasing in competitor abundance by this "
            "property, so the operator ordering is not a risky prediction",
        },
        "measured": measured,
        "verdict": verdict,
        "verdict_nature": "BLINDNESS-FILLED capability confirmation, NOT a passed falsification test. The "
        "operator ordering is monotone by construction and so cannot disagree with a monotone measured "
        "titration; the decision-relevant content is that the independent scorer is structurally blind "
        "(D = 0 for both A and B) while the operator and the measured Denzler biology agree B > A.",
        "candidate2_cross_system_nongrounded": candidate2,
        "determinism": "bisection solver, no randomness, CPU; JSON bitwise-identical on rerun (md5 reported)",
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True))
    md5 = hashlib.md5(OUT_JSON.read_bytes()).hexdigest()

    print("1B decision-relevant capability example (Candidate 1: miR-122 titration, GROUNDED)")
    print("=" * 92)
    print(f"  system: {mir122['name']} guide_total_M={mir122['M']:g}  ({mir122['source'][:60]}...)")
    print(
        f"  A = {X_A:g} added sites/cell (measured effective endogenous TSA, Denzler 2016 Fig 7A)"
    )
    print(f"  B = {X_B:g} added sites/cell (measured IC50, Denzler 2016)")
    print("\n  (i) INDEPENDENT SCORER (function of guide,reporter only):")
    for name in scorer_A:
        print(f"      {name:22s} A={scorer_A[name]:.6g}  B={scorer_B[name]:.6g}  d/dx_competitor=0")
    print(f"      identical for A and B = {scorer_identical}  (blind to the A/B distinction)")
    print("\n  (ii) OPERATOR derepression of the miR-122 reporter:")
    print(
        f"      A: derepression={op_A['derepression_fraction']:.4f}  f*={op_A['free_guide']:.4g}  "
        f"D_jac={op_A['D_jacobian']:.4g}"
    )
    print(
        f"      B: derepression={op_B['derepression_fraction']:.4f}  f*={op_B['free_guide']:.4g}  "
        f"D_jac={op_B['D_jacobian']:.4g}"
    )
    print(f"      operator ordering: derepression(B) > derepression(A) = {op_orders_B_over_A}")
    print(
        "\n  (iii) MEASURED ordering (Denzler): B at IC50 = half-maximal (~0.5) > A sub-threshold "
        "(below >=1.5e5) -> B > A"
    )
    print(f"\n  VERDICT: {verdict}  ({result['verdict_nature'][:64]}...)")
    print(
        "  Candidate 2 (cross-system, miR-92/25 3000 vs miR-122 1.5e5): recorded NON-GROUNDED "
        "(needs shared-reporter construct)"
    )
    print(f"\n  wrote {OUT_JSON.relative_to(REPO)}  md5={md5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
