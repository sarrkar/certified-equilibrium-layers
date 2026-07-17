"""Competitive-derepression (ceRNA) comparative statics for the conservation-law equilibrium operator.

Demonstrates a CAPABILITY difference, not an accuracy difference: the shared-pool equilibrium
operator expresses competitive derepression (adding a competing transcript for a shared guide
raises free guide and derepresses a reporter), while independent per-target scorers (seed count,
seed Tm, fixed-free-level occupancy, TargetScan context) are functions of (guide, reporter) ONLY,
have no competitor-abundance argument, and are therefore identically flat (D = 0 for every x). The
zero is an input-signature property, not a tuning artifact.

House rules: no synthetic data (anchors carry sources and VERIFY notes in the config; the loader
RAISES on any missing field); deterministic and CPU only (no random, bitwise-identical on rerun);
equilibrium solves use BISECTION on the scalar monotone residual g(f)=f+sum_i x_i f/(K_i+f)-M over
[0,M] (never the Picard map, which is not a contraction); no em-dashes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "cerna_derepression_config.json"
DEFAULT_OUTDIR = REPO / "results"
DEFAULT_SCALES = [0.01, 0.1, 1.0, 10.0, 100.0]


# ---------------------------------------------------------------------------------------------
# Config loading (RAISES on any missing required field or missing VERIFY marker).
# ---------------------------------------------------------------------------------------------
def _num(node, path: str) -> float:
    if not isinstance(node, dict) or "value" not in node or "verify" not in node:
        raise ValueError(f"{path}: numeric anchor must be an object with 'value' and 'verify' (VERIFY) fields")
    return float(node["value"])


def load_systems(config: dict):
    if "definitions" not in config or "systems" not in config:
        raise ValueError("config missing required top-level field: 'definitions' or 'systems'")
    defs = config["definitions"]
    if "threshold_target" not in defs:
        raise ValueError("definitions missing required field: 'threshold_target'")
    required = ("name", "source", "guide_total_M", "background_sites", "reporter",
                "competitor_K", "measured_threshold_sites", "measured_threshold_source")
    systems = []
    for i, s in enumerate(config["systems"]):
        for field in required:
            if field not in s:
                raise ValueError(f"system[{i}] missing required field: {field}")
        for cite in ("source", "measured_threshold_source"):
            if not (isinstance(s[cite], str) and s[cite].strip()):
                raise ValueError(f"system[{i}] ({s.get('name', '?')}) has empty or non-string '{cite}'; "
                                 "every anchor must cite its primary source")
        if not s["background_sites"]:
            raise ValueError(f"system[{i}] ({s['name']}) has empty background_sites")
        background = []
        for j, e in enumerate(s["background_sites"]):
            if "abundance" not in e or "K" not in e:
                raise ValueError(f"system[{i}].background_sites[{j}] missing 'abundance' or 'K'")
            background.append((_num(e["abundance"], f"system[{i}].bg[{j}].abundance"),
                               _num(e["K"], f"system[{i}].bg[{j}].K")))
        rep = s["reporter"]
        if "abundance" not in rep or "K" not in rep:
            raise ValueError(f"system[{i}].reporter missing 'abundance' or 'K'")
        sys_d = {
            "name": s["name"],
            "source": s["source"],
            # added_competing_sites (Denzler, Bosson) systems are ranked; other axes (e.g. Mukherji
            # target_production) are support only and never enter the ranking / Table B / sweep.
            "threshold_axis": s.get("threshold_axis", "added_competing_sites"),
            "M": _num(s["guide_total_M"], f"system[{i}].guide_total_M"),
            "background": background,
            "reporter_x": _num(rep["abundance"], f"system[{i}].reporter.abundance"),
            "reporter_K": _num(rep["K"], f"system[{i}].reporter.K"),
            "competitor_K": _num(s["competitor_K"], f"system[{i}].competitor_K"),
            "measured_threshold": _num(s["measured_threshold_sites"], f"system[{i}].measured_threshold_sites"),
            "measured_threshold_source": s["measured_threshold_source"],
            "measured_background": None,
            "measured_background_source": None,
            "measured_background_circular": False,
        }
        # Optional per-system measured background sink (Prompt 3). Requires its own non-empty source.
        mb = s.get("measured_background_sites")
        if mb is not None:
            for f in ("abundance", "K", "source"):
                if f not in mb:
                    raise ValueError(f"system[{i}] ({s['name']}) measured_background_sites missing '{f}'")
            if not (isinstance(mb["source"], str) and mb["source"].strip()):
                raise ValueError(f"system[{i}] ({s['name']}) measured_background_sites has empty 'source'")
            sys_d["measured_background"] = (_num(mb["abundance"], f"system[{i}].measured_background_sites.abundance"),
                                           _num(mb["K"], f"system[{i}].measured_background_sites.K"))
            sys_d["measured_background_source"] = mb["source"]
            # A background whose value is derived from the system's own measured threshold is
            # circular: it must not produce a "tightening" predicted-over-measured ratio.
            sys_d["measured_background_circular"] = mb.get("circular_with") == "measured_threshold_sites"
        systems.append(sys_d)
    return systems, defs


def active_background(system: dict, mode: str):
    """Site sink for this system under the background mode: the common nominal background_sites, or
    (in per_system mode, when present) the single measured effective-target-site-abundance sink."""
    if mode == "per_system" and system.get("measured_background") is not None:
        return [system["measured_background"]]
    return system["background"]


# ---------------------------------------------------------------------------------------------
# Equilibrium: bisection on the monotone residual (house rule, NOT the Picard map).
# ---------------------------------------------------------------------------------------------
def solve_free_guide(M: float, sites, max_iter: int = 200, tol: float = 1e-12) -> float:
    """Solve M = f + sum_i x_i f/(K_i+f) for f in [0,M] by bisection on g(f)=RHS-M.

    g is strictly increasing with g(0)=-M<=0 and g(M)=sum x_i M/(K_i+M)>=0, so the bracket [0,M]
    is unconditional. No Picard iteration: |T'|=sum x_i K_i/(K_i+f)^2 can exceed 1 and diverge.
    """
    def g(f: float) -> float:
        return f + math.fsum(x * f / (K + f) for x, K in sites) - M

    lo, hi = 0.0, M
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * (1.0 + hi):
            break
    return 0.5 * (lo + hi)


def reporter_occupancy(system: dict, x_comp: float, k_scale: float, mode: str = "common") -> float:
    """Assemble sites (background + reporter + competitor if x_comp>0, all K scaled by k_scale),
    solve the shared-pool equilibrium, and return the reporter occupancy f/(K_rep*k_scale+f).

    The background sink is selected by `mode` (common nominal or per_system measured)."""
    sites = [(x, K * k_scale) for x, K in active_background(system, mode)]
    sites.append((system["reporter_x"], system["reporter_K"] * k_scale))
    if x_comp > 0.0:
        sites.append((x_comp, system["competitor_K"] * k_scale))
    f = solve_free_guide(system["M"], sites)
    k_rep = system["reporter_K"] * k_scale
    return f / (k_rep + f)


def derepression_fraction(system: dict, x_comp: float, k_scale: float, mode: str = "common") -> float:
    """D(x) = 1 - theta_rep(x)/theta_rep(0). Positive means the reporter is derepressed."""
    t0 = reporter_occupancy(system, 0.0, k_scale, mode)
    tx = reporter_occupancy(system, x_comp, k_scale, mode)
    return 1.0 - tx / t0


def derepression_threshold(system: dict, target: float, k_scale: float, mode: str = "common", x_max: float = 1e14) -> float:
    """Smallest competitor abundance x with D(x)>=target: geometric bracketing then bisection."""
    def d(x: float) -> float:
        return derepression_fraction(system, x, k_scale, mode)

    if d(0.0) >= target:
        return 0.0
    x_hi = 1.0
    while d(x_hi) < target:
        x_hi *= 10.0
        if x_hi > x_max:
            return float("inf")
    x_lo = x_hi / 10.0
    for _ in range(200):
        mid = 0.5 * (x_lo + x_hi)
        if d(mid) >= target:
            x_hi = mid
        else:
            x_lo = mid
        if x_hi - x_lo <= 1e-6 * (1.0 + x_hi):
            break
    return 0.5 * (x_lo + x_hi)


# ---------------------------------------------------------------------------------------------
# Baselines: independent scorers of (guide, reporter) ONLY. No competitor-abundance argument.
# ---------------------------------------------------------------------------------------------
def _seed_count_score(system: dict) -> float:
    """Number of high-affinity reporter sites (a property of guide:reporter, not competition)."""
    return 1.0


def _seed_tm_score(system: dict) -> float:
    """Seed:reporter pairing stability proxy (monotone in -log K_rep); competitor-independent."""
    return -math.log(system["reporter_K"])


def _independent_occupancy_score(system: dict) -> float:
    """Occupancy at a FIXED free-guide level (the no-competitor baseline free guide)."""
    f_fixed = solve_free_guide(system["M"],
                               [(x, K) for x, K in system["background"]]
                               + [(system["reporter_x"], system["reporter_K"])])
    return f_fixed / (system["reporter_K"] + f_fixed)


def _targetscan_context_score(system: dict) -> float:
    """A context-type score of guide:reporter (nominal, competitor-independent)."""
    return system["reporter_x"] / (system["reporter_K"] + system["reporter_x"])


BASELINES = {
    "seed_count": _seed_count_score,
    "seed_Tm": _seed_tm_score,
    "independent_occupancy": _independent_occupancy_score,
    "targetscan_context": _targetscan_context_score,
}


def baseline_derepression(system: dict, x_grid):
    """D-curves for the independent scorers. Each score is a function of (guide, reporter) with NO
    competitor-abundance argument, so D(x) = 1 - score/score = 0 for every x. The identical zero is
    an INPUT-SIGNATURE property (the scorer cannot see the competitor), not a tuning artifact."""
    curves = {}
    for name, fn in BASELINES.items():
        _ = fn(system)  # evaluated once; independent of x by construction
        curves[name] = [0.0 for _ in x_grid]
    return curves


# ---------------------------------------------------------------------------------------------
# Curve analysis.
# ---------------------------------------------------------------------------------------------
def _loglog_slope(x_grid, d_curve) -> float:
    """Least-squares slope of log10(D) vs log10(x) over the rising region D in (1e-9, 0.999).

    Returns 0.0 when there is no rising region (a flat baseline), which is exact for D identically 0.
    """
    xs, ys = [], []
    for x, d in zip(x_grid, d_curve, strict=True):
        if x > 0.0 and 1e-9 < d < 0.999:
            xs.append(math.log10(x))
            ys.append(math.log10(d))
    if len(xs) < 2:
        return 0.0
    n = len(xs)
    mx = math.fsum(xs) / n
    my = math.fsum(ys) / n
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return sxy / sxx if sxx > 0 else 0.0


def _x_grid():
    """Deterministic geometric competitor-abundance grid (copies/cell), 1 to 1e8."""
    return [10.0 ** (k / 6.0) for k in range(0, 49)]  # 1 .. 1e8, 6 points per decade


# ---------------------------------------------------------------------------------------------
# Experiment.
# ---------------------------------------------------------------------------------------------
def run(config: dict, target: float | None = None, scales=None, mode: str = "common") -> dict:
    systems, defs = load_systems(config)
    if target is None:
        target = float(defs["threshold_target"])
    scales = scales or DEFAULT_SCALES
    x_grid = _x_grid()

    # Ranking systems share ONE axis (added competing sites: Denzler miR-122, Bosson miR-92/25).
    # Support-axis systems (Mukherji target-production) are never ranked; kept as a support citation.
    ranking = [s for s in systems if s["threshold_axis"] == "added_competing_sites"]
    support = [s for s in systems if s["threshold_axis"] != "added_competing_sites"]

    per_system = {}
    for s in systems:
        op_curve = [derepression_fraction(s, x, 1.0, mode) for x in x_grid]
        base_curves = baseline_derepression(s, x_grid)
        nominal = derepression_threshold(s, target, 1.0, mode)
        per_system[s["name"]] = {
            "role": "ranking" if s in ranking else "support",
            "threshold_axis": s["threshold_axis"],
            "source": s["source"],
            "guide_total_M": s["M"],
            "measured_threshold_sites": s["measured_threshold"],
            "measured_threshold_source": s["measured_threshold_source"],
            "measured_background_sink": s["measured_background"],
            "measured_background_source": s["measured_background_source"],
            "measured_background_circular": s["measured_background_circular"],
            "operator_D_curve": op_curve,
            "operator_loglog_slope": _loglog_slope(x_grid, op_curve),
            "baseline_D_curves": base_curves,
            "baseline_loglog_slopes": {k: _loglog_slope(x_grid, v) for k, v in base_curves.items()},
            "nominal_threshold_sites": nominal,
            "predicted_over_measured_ratio": nominal / s["measured_threshold"] if s["measured_threshold"] else float("inf"),
        }

    # Predicted-over-measured ratio by mode, RANKING systems only. A system whose measured
    # background is circular with its own threshold is NOT given a per_system tightening ratio.
    mode_comparison = {}
    for s in ranking:
        thr_c = derepression_threshold(s, target, 1.0, "common")
        row = {"has_measured_background": s["measured_background"] is not None,
               "measured_background_circular": s["measured_background_circular"],
               "common": {"nominal_threshold_sites": thr_c,
                          "predicted_over_measured_ratio": thr_c / s["measured_threshold"]}}
        if s["measured_background"] is None:
            row["per_system"] = {"reported": False, "reason": "no independent measured background"}
        elif s["measured_background_circular"]:
            row["per_system"] = {"reported": False, "reason": "CIRCULAR: measured background equals measured_threshold_sites"}
        else:
            thr_p = derepression_threshold(s, target, 1.0, "per_system")
            row["per_system"] = {"reported": True, "nominal_threshold_sites": thr_p,
                                 "predicted_over_measured_ratio": thr_p / s["measured_threshold"]}
        mode_comparison[s["name"]] = row

    # Affinity-scale sweep, RANKING systems only. Robustness check, not a finding (see note).
    orderings = {}
    scale_thresholds = {}
    for ks in scales:
        thr = {s["name"]: derepression_threshold(s, target, ks, mode) for s in ranking}
        scale_thresholds[str(ks)] = thr
        orderings[str(ks)] = [name for name, _ in sorted(thr.items(), key=lambda kv: kv[1])]
    ordering_values = list(orderings.values())
    ordering_invariant = all(o == ordering_values[0] for o in ordering_values)

    support_citations = [{
        "name": s["name"], "threshold_axis": s["threshold_axis"],
        "measured_threshold_sites": s["measured_threshold"],
        "measured_threshold_source": s["measured_threshold_source"],
        "role": "SUPPORT ONLY: independent evidence that molecular-titration thresholds exist. This "
        "is a target-production threshold on a DIFFERENT axis from the added-competing-site "
        "thresholds; it is NOT placed in the ranking, Table B, or the scale sweep.",
    } for s in support]

    return {
        "experiment": "competitive-derepression comparative statics (conservation-law operator)",
        "house_rules": "bisection on monotone residual; deterministic CPU; anchors carry source + VERIFY",
        "background_mode": mode,
        "threshold_target": target,
        "x_grid_copies_per_cell": {"min": x_grid[0], "max": x_grid[-1], "n": len(x_grid),
                                   "note": "geometric, 6 points/decade, deterministic"},
        "ranking_systems": [s["name"] for s in ranking],
        "support_citations": support_citations,
        "systems": per_system,
        "mode_comparison_predicted_over_measured_ratio": mode_comparison,
        "affinity_scale_sweep": {
            "scales": scales,
            "thresholds_by_scale": scale_thresholds,
            "ordering_by_scale": orderings,
            "ordering_invariant_across_scales": ordering_invariant,
            "robustness_note_not_a_finding": "With the background held common the operator threshold "
            "is monotone in guide_total_M by construction, so the ranking cannot reorder; this sweep "
            "confirms a monotone function is monotone and is NOT framed as a finding.",
        },
        "capability_statement": "The baseline scorers are functions of (guide, reporter) only; they "
        "take no competitor-abundance argument, so their derepression D is identically 0 (an input-"
        "signature property, not tuning). The operator couples targets through the shared free guide, "
        "so D rises with competitor abundance. This is a capability difference, not an accuracy one.",
        "background_mode_note": "common (default) uses one nominal background sink for all systems so "
        "guide_total_M is the sole differentiator: it is ordering-only and calibration-free. per_system "
        "swaps in each system's MEASURED effective target-site abundance (measured_background_sites), "
        "which tightens the absolute predicted-over-measured threshold ratio at the cost of some "
        "calibration-freeness (it now depends on measured, VERIFY-marked, per-system inputs). common "
        "remains the reported default.",
    }


def _markdown(results: dict) -> str:
    lines = ["# Competitive-derepression comparative statics", "",
             f"background_mode = {results['background_mode']}  |  threshold_target = {results['threshold_target']}", "",
             "| system | guide_total_M | operator log-slope | baseline log-slopes | "
             "operator threshold (sites) | measured threshold (sites) |",
             "|---|---|---|---|---|---|"]
    for name, s in results["systems"].items():
        if s["role"] != "ranking":
            continue
        bslopes = ", ".join(f"{k}={v:g}" for k, v in s["baseline_loglog_slopes"].items())
        lines.append(f"| {name} | {s['guide_total_M']:g} | {s['operator_loglog_slope']:.3f} | "
                     f"{bslopes} | {s['nominal_threshold_sites']:.4g} | {s['measured_threshold_sites']:g} |")
    lines += ["", "## Predicted/measured threshold ratio by background mode (ranking systems only)", "",
              "| system | common | per_system |", "|---|---|---|"]
    for name, row in results["mode_comparison_predicted_over_measured_ratio"].items():
        ps = row["per_system"]
        ps_str = f"{ps['predicted_over_measured_ratio']:.3g}" if ps.get("reported") else "CIRCULAR (not reported)"
        lines.append(f"| {name} | {row['common']['predicted_over_measured_ratio']:.3g} | {ps_str} |")
    lines += ["", "## Support citations (not ranked, different axis)"]
    for c in results["support_citations"]:
        lines.append(f"- {c['name']} ({c['threshold_axis']}): measured threshold {c['measured_threshold_sites']:g}. "
                     f"{c['measured_threshold_source']}")
    sw = results["affinity_scale_sweep"]
    lines += ["", "Robustness footnote (not a finding): " + sw["robustness_note_not_a_finding"], "",
              results["capability_statement"], "", results["background_mode_note"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="ceRNA competitive-derepression comparative statics")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--scales", type=str, default=None, help="comma-separated affinity scales")
    ap.add_argument("--background-mode", choices=("common", "per_system"), default="common",
                    help="common (default, ordering-only, calibration-free) or per_system (measured backgrounds)")
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    scales = [float(x) for x in args.scales.split(",")] if args.scales else None
    results = run(config, target=args.target, scales=scales, mode=args.background_mode)

    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.background_mode}"
    (args.outdir / f"cerna_derepression{suffix}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.outdir / f"cerna_derepression{suffix}.md").write_text(_markdown(results), encoding="utf-8")

    print("Competitive-derepression comparative statics (conservation-law operator)")
    print(f"  background_mode = {results['background_mode']}   threshold_target = {results['threshold_target']}")
    print(f"  {'system':<12}{'guide_M':>12}{'op_slope':>10}{'op_thresh':>14}{'meas_thresh':>14}  baseline_slopes")
    for name, s in results["systems"].items():
        bslopes = ",".join(f"{v:g}" for v in s["baseline_loglog_slopes"].values())
        print(f"  {name:<12}{s['guide_total_M']:>12g}{s['operator_loglog_slope']:>10.3f}"
              f"{s['nominal_threshold_sites']:>14.4g}{s['measured_threshold_sites']:>14g}  [{bslopes}]")
    print(f"  ranking systems (added-competing-site axis): {results['ranking_systems']}")
    print("  predicted/measured threshold RATIO by background mode (ranking systems only):")
    print(f"    {'system':<12}{'common':>12}{'per_system':>16}")
    for name, row in results["mode_comparison_predicted_over_measured_ratio"].items():
        ps = row["per_system"]
        ps_str = f"{ps['predicted_over_measured_ratio']:.3g}" if ps.get("reported") else "CIRCULAR/skip"
        print(f"    {name:<12}{row['common']['predicted_over_measured_ratio']:>12.3g}{ps_str:>16}")
    for c in results["support_citations"]:
        print(f"  SUPPORT (not ranked): {c['name']} on axis '{c['threshold_axis']}', measured threshold "
              f"{c['measured_threshold_sites']:g} ({c['measured_threshold_source'][:40]}...)")
    sw = results["affinity_scale_sweep"]
    print(f"  robustness footnote (not a finding): ordering {' < '.join(sw['ordering_by_scale'][str(sw['scales'][0])])} "
          "is monotone in M by construction; the scale sweep confirms a monotone function is monotone.")
    print(f"  wrote {args.outdir / 'cerna_derepression.json'} and {args.outdir / 'cerna_derepression.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
