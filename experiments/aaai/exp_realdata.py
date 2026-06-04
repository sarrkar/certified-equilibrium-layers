#!/usr/bin/env python3
"""Project 2 (AAAI): certified off-target containment on the real-data pipeline.

Ties the certification machinery to the shared CASCADE data path: per-transcript
direct seed-binding repression x0 from the SeedMatchR seed logic
(cascade.features.seedmatch), propagated over a STRING GRN S, certified under the
inhomogeneous random-subgraph GRN noise model.

Deliverables:
1. Certified containment: the fraction of transcripts whose predicted off-target
   effect is certifiably contained within a tolerance tau, i.e. the certified
   upper bound |y_plugin_i| + radius stays below tau under the GRN noise model.
2. Certified-radius curves vs the STRING confidence threshold (0.4, 0.7, 0.9),
   confirming the Theorem 5 (symmetric-normalized, C5) story on real data.
3. Degradation: subsample STRING edges and inject false edges, showing graceful
   degradation and the C5 breakdown point (degree concentration rho >= 1).
4. Monte-Carlo cost: the number of samples N needed for a (1 - delta) confidence
   interval on the smoothed estimator at a target precision.

Provenance. The real GSE184929 / STRING sequence join is a tracked data gap, so
by default the graph is a synthetic STRING-like confidence matrix and x0 is
computed by the real SeedMatchR seed logic on synthetic 3'UTR sequences. Pass
--string-links to use a real STRING links file for S; the provenance is recorded
in the JSON either way.

Writes results/aaai/realdata.json and figures. Accepts --seed and logs it
(CLAUDE.md rule 3). Emits JSON and figures only, never LaTeX (rule 7).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cascade.features.seedmatch import seed_match_score, seed_site_words  # noqa: E402
from cascade.verify.containment import dense_propagation_operator  # noqa: E402
from certification import (  # noqa: E402
    EdgeBernoulliNoise,
    min_degree_diagnostic,
    normalized_certificate,
)
from certification.noise_model import normalize_adjacency  # noqa: E402

OUT = REPO / "results" / "aaai"
BASES = np.array(["A", "C", "G", "U"])
# A Ttr-like 21 nt guide (antisense, 5' to 3'); the seed (positions 2-8) drives
# the off-target seed matches.
GUIDE = "UGAGGUAGUAGGUUGUAUAGU"


# ---------------------------------------------------------------------------
# Data: STRING confidence matrix and SeedMatchR-derived x0.
# ---------------------------------------------------------------------------
def synthetic_confidence(n: int, edge_prob: float, lo: float, hi: float, rng) -> sp.csr_matrix:
    """Random symmetric STRING-like confidence matrix (each candidate edge ~ U(lo, hi))."""
    iu = np.triu_indices(n, 1)
    present = rng.random(iu[0].shape[0]) < edge_prob
    conf = rng.uniform(lo, hi, iu[0].shape[0]) * present
    rows = np.concatenate([iu[0][present], iu[1][present]])
    cols = np.concatenate([iu[1][present], iu[0][present]])
    data = np.concatenate([conf[present], conf[present]])
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def threshold_confidence(C: sp.csr_matrix, cutoff: float) -> sp.csr_matrix:
    """Keep only edges with confidence >= cutoff (the STRING combined-score cut)."""
    coo = C.tocoo()
    keep = coo.data >= cutoff
    return sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=C.shape).tocsr()


def seedmatch_x0(
    n: int, guide: str, rng, utr_len: int = 400, plant_frac: float = 0.15
) -> np.ndarray:
    """Per-transcript direct seed repression x0 from the SeedMatchR seed logic.

    Each transcript gets a synthetic 3'UTR; a fraction have a planted seed-match
    site (off-targets), the rest are random (some chance matches). x0_i is the
    SeedMatchR best canonical site grade in [0, 1].
    """
    site = seed_site_words(guide)["8mer"]
    x0 = np.zeros(n)
    for i in range(n):
        utr = "".join(rng.choice(BASES, utr_len))
        if rng.random() < plant_frac:
            pos = int(rng.integers(0, utr_len))
            utr = utr[:pos] + site + utr[pos:]
        x0[i] = seed_match_score(guide, utr)
    return x0


def build_string_confidence(args, rng) -> tuple[sp.csr_matrix, str]:
    """Return (confidence matrix at the lowest threshold, provenance string)."""
    if args.string_links is not None and Path(args.string_links).exists():
        from cascade.data import load_string_confidence

        _, c = load_string_confidence(args.string_links, threshold=int(args.min_threshold * 1000))
        return c.tocsr(), f"real STRING links ({args.string_links})"
    c = synthetic_confidence(args.n_nodes, args.edge_prob, args.min_threshold, 1.0, rng)
    return c, "synthetic STRING-like confidence matrix"


# ---------------------------------------------------------------------------
# Certified containment.
# ---------------------------------------------------------------------------
def plug_in_prediction(C: sp.csr_matrix, x0: np.ndarray, alpha: float) -> np.ndarray:
    """y_plugin = P_alpha(S_sym(A_bar)) x0 (the mean-graph off-target prediction)."""
    s_bar = normalize_adjacency(C, "sym")
    return dense_propagation_operator(s_bar, alpha) @ x0


def containment_fraction(y_plugin: np.ndarray, radius: float, tau: float) -> float:
    """Fraction of transcripts certifiably contained: |y_plugin_i| + radius <= tau."""
    if not math.isfinite(radius):
        return 0.0
    return float(np.mean(np.abs(y_plugin) + radius <= tau))


def part_containment(C: sp.csr_matrix, x0: np.ndarray, alpha: float, delta: float) -> dict:
    cert = normalized_certificate(x0, EdgeBernoulliNoise(C=C), alpha, delta)
    y_plugin = plug_in_prediction(C, x0, alpha)
    radius = cert.radius_sym
    max_eff = float(np.abs(y_plugin).max())
    hi = (radius + max_eff) * 1.25 if math.isfinite(radius) else max_eff * 2 + 1.0
    taus = np.linspace(0.0, hi, 40)
    certified = [containment_fraction(y_plugin, radius, t) for t in taus]
    point = [float(np.mean(np.abs(y_plugin) <= t)) for t in taus]  # uncertified point estimate
    tau90 = next((float(t) for t, f in zip(taus, certified, strict=False) if f >= 0.9), None)
    return {
        "radius_sym": radius if math.isfinite(radius) else None,
        "assumption_holds": cert.assumption_holds,
        "degree_rel_fluctuation": cert.degree_rel_fluctuation,
        "n_off_targets": int(np.count_nonzero(x0)),
        "max_predicted_effect": max_eff,
        "tau_grid": taus.tolist(),
        "certified_contained_fraction": certified,
        "point_contained_fraction": point,
        "tau_for_90pct_certified": tau90,
        "certified_fraction_at_2x_radius": containment_fraction(
            y_plugin, radius, 2.0 * radius if math.isfinite(radius) else math.inf
        ),
    }


# ---------------------------------------------------------------------------
# Radius vs STRING threshold (Theorem 5 story).
# ---------------------------------------------------------------------------
def part_threshold(C_full: sp.csr_matrix, x0: np.ndarray, alpha: float, delta: float) -> dict:
    per_threshold = []
    for theta in (0.4, 0.7, 0.9):
        c = threshold_confidence(C_full, theta)
        cert = normalized_certificate(x0, EdgeBernoulliNoise(C=c), alpha, delta)
        diag = min_degree_diagnostic(EdgeBernoulliNoise(C=c), delta)
        per_threshold.append(
            {
                "threshold": theta,
                "n_active": diag["n_active"],
                "min_expected_degree": diag["expected_min_degree"],
                "mean_expected_degree": diag["mean_expected_degree"],
                "degree_rel_fluctuation": cert.degree_rel_fluctuation,
                "assumption_holds": cert.assumption_holds,
                "radius_sym": cert.radius_sym if math.isfinite(cert.radius_sym) else None,
            }
        )
    return {"thresholds": [0.4, 0.7, 0.9], "per_threshold": per_threshold}


# ---------------------------------------------------------------------------
# Degradation: subsample edges and inject false edges.
# ---------------------------------------------------------------------------
def subsample_edges(C: sp.csr_matrix, drop_frac: float, rng) -> sp.csr_matrix:
    triu = sp.triu(C, k=1).tocoo()
    keep = rng.random(triu.data.shape[0]) >= drop_frac
    r, c, d = triu.row[keep], triu.col[keep], triu.data[keep]
    rows = np.concatenate([r, c])
    cols = np.concatenate([c, r])
    data = np.concatenate([d, d])
    return sp.coo_matrix((data, (rows, cols)), shape=C.shape).tocsr()


def inject_false_edges(
    C: sp.csr_matrix, false_frac: float, lo: float, hi: float, rng
) -> sp.csr_matrix:
    n = C.shape[0]
    triu = sp.triu(C, k=1).tocoo()
    existing = set(zip(triu.row.tolist(), triu.col.tolist(), strict=False))
    n_false = int(false_frac * len(existing))
    new_r, new_c, new_d = [], [], []
    tries = 0
    while len(new_r) < n_false and tries < 20 * (n_false + 1):
        i, j = sorted(rng.integers(0, n, size=2))
        tries += 1
        if i != j and (i, j) not in existing:
            existing.add((i, j))
            new_r.append(i)
            new_c.append(j)
            new_d.append(float(rng.uniform(lo, hi)))
    r = np.concatenate([triu.row, np.array(new_r, dtype=int)])
    c = np.concatenate([triu.col, np.array(new_c, dtype=int)])
    d = np.concatenate([triu.data, np.array(new_d)])
    rows = np.concatenate([r, c])
    cols = np.concatenate([c, r])
    data = np.concatenate([d, d])
    return sp.coo_matrix((data, (rows, cols)), shape=C.shape).tocsr()


def _degradation_point(
    c: sp.csr_matrix, x0: np.ndarray, alpha: float, delta: float, level: float, kind: str
) -> dict:
    cert = normalized_certificate(x0, EdgeBernoulliNoise(C=c), alpha, delta)
    return {
        "level": level,
        "kind": kind,
        "min_expected_degree": cert.expected_min_degree,
        "degree_rel_fluctuation": cert.degree_rel_fluctuation,
        "assumption_holds": cert.assumption_holds,
        "radius_sym": cert.radius_sym if math.isfinite(cert.radius_sym) else None,
    }


def part_degradation(
    C: sp.csr_matrix, x0: np.ndarray, alpha: float, delta: float, seed: int
) -> dict:
    subsample = []
    breakdown_drop = None
    for drop in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9):
        c = subsample_edges(C, drop, np.random.default_rng(seed + 100))
        pt = _degradation_point(c, x0, alpha, delta, drop, "subsample")
        subsample.append(pt)
        if breakdown_drop is None and not pt["assumption_holds"]:
            breakdown_drop = drop

    false_edges = []
    for ffrac in (0.0, 0.25, 0.5, 1.0, 2.0):
        c = inject_false_edges(C, ffrac, 0.4, 1.0, np.random.default_rng(seed + 200))
        false_edges.append(_degradation_point(c, x0, alpha, delta, ffrac, "false_edges"))

    return {
        "subsample": subsample,
        "false_edges": false_edges,
        "breakdown_subsample_fraction": breakdown_drop,
        "note": "subsampling sparsifies the GRN: degrees drop, rho rises, the C5 certificate breaks down; injecting false edges raises degrees so the certificate stays valid but the prediction is corrupted",
    }


# ---------------------------------------------------------------------------
# Monte-Carlo cost.
# ---------------------------------------------------------------------------
def part_mc_cost(x0: np.ndarray, n_nodes: int, eps_rel: float) -> dict:
    """Hoeffding sample complexity N(delta) for the smoothed-estimator CI.

    For y_i bounded by value_radius = ||x0||_2 (since ||P_alpha||=1), the
    per-coordinate Hoeffding half-width is range * sqrt(log(2d/delta)/(2N)) with
    range = 2 value_radius; solving for a target half-width eps gives
    N = ceil(range^2 log(2d/delta) / (2 eps^2)).
    """
    value_radius = float(np.linalg.norm(x0))
    rng_width = 2.0 * value_radius
    eps = eps_rel * value_radius
    rows = []
    for delta in (0.2, 0.1, 0.05, 0.01, 0.001):
        n_needed = math.ceil(rng_width**2 * math.log(2.0 * n_nodes / delta) / (2.0 * eps**2))
        rows.append({"delta": delta, "n_samples": n_needed})
    return {
        "value_radius": value_radius,
        "target_half_width_relative": eps_rel,
        "target_half_width": eps,
        "n_coordinates": n_nodes,
        "per_delta": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (logged)")
    parser.add_argument("--alpha", type=float, default=0.5, help="propagation teleport")
    parser.add_argument("--delta", type=float, default=0.1, help="confidence level")
    parser.add_argument("--n-nodes", type=int, default=250, help="transcripts / GRN nodes")
    parser.add_argument("--edge-prob", type=float, default=0.6, help="candidate edge probability")
    parser.add_argument("--min-threshold", type=float, default=0.4, help="lowest STRING cutoff")
    parser.add_argument(
        "--base-threshold", type=float, default=0.7, help="cutoff for parts 1 and 3"
    )
    parser.add_argument(
        "--plant-frac", type=float, default=0.15, help="fraction of seeded off-targets"
    )
    parser.add_argument("--eps-rel", type=float, default=0.1, help="MC target relative half-width")
    parser.add_argument("--string-links", type=str, default=None, help="real STRING links file")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(f"[realdata] seed={args.seed} alpha={args.alpha} delta={args.delta}")

    rng = np.random.default_rng(args.seed)
    c_full, graph_source = build_string_confidence(args, rng)
    n = c_full.shape[0]
    x0 = seedmatch_x0(n, GUIDE, rng, plant_frac=args.plant_frac)
    c_base = threshold_confidence(c_full, args.base_threshold)
    print(f"[realdata] graph: {graph_source}, n={n}, off-targets={int(np.count_nonzero(x0))}")

    containment = part_containment(c_base, x0, args.alpha, args.delta)
    threshold = part_threshold(c_full, x0, args.alpha, args.delta)
    degradation = part_degradation(c_base, x0, args.alpha, args.delta, args.seed)
    mc_cost = part_mc_cost(x0, n, args.eps_rel)

    summary = {
        "seed": args.seed,
        "alpha": args.alpha,
        "delta": args.delta,
        "provenance": {
            "graph_source": graph_source,
            "x0_source": "SeedMatchR seed-match scores on synthetic 3'UTR sequences",
            "note": "the real GSE184929 / STRING sequence join is a tracked data gap; x0 uses the real SeedMatchR seed logic",
            "guide": GUIDE,
        },
        "base_threshold": args.base_threshold,
        "containment": containment,
        "radius_vs_threshold": threshold,
        "degradation": degradation,
        "monte_carlo_cost": mc_cost,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "realdata.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[realdata] containment: radius={containment['radius_sym']} "
        f"tau90={containment['tau_for_90pct_certified']} "
        f"frac@2R={containment['certified_fraction_at_2x_radius']:.3f}"
    )
    for d in threshold["per_threshold"]:
        print(
            f"[realdata] threshold={d['threshold']}: min_deg={d['min_expected_degree']:.2f} "
            f"rho={d['degree_rel_fluctuation']:.3f} radius={d['radius_sym']} valid={d['assumption_holds']}"
        )
    for d in degradation["subsample"]:
        flag = "  <<< BREAKDOWN" if not d["assumption_holds"] else ""
        print(
            f"[realdata] subsample drop={d['level']:.2f}: min_deg={d['min_expected_degree']:.2f} "
            f"rho={d['degree_rel_fluctuation']:.3f} valid={d['assumption_holds']}{flag}"
        )
    print(
        f"[realdata] breakdown at subsample fraction: {degradation['breakdown_subsample_fraction']}"
    )
    print(
        f"[realdata] MC cost N(delta): {[(r['delta'], r['n_samples']) for r in mc_cost['per_delta']]}"
    )
    print(f"[realdata] wrote {out_path}")

    _plot(summary, args.out_dir)
    return 0


def _plot(summary: dict, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[realdata] matplotlib not installed; skipped figures")
        return

    cont = summary["containment"]
    thr = summary["radius_vs_threshold"]["per_threshold"]
    deg = summary["degradation"]
    mc = summary["monte_carlo_cost"]["per_delta"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # 1. Certified containment fraction vs tolerance.
    ax = axes[0, 0]
    ax.plot(cont["tau_grid"], cont["certified_contained_fraction"], "o-", ms=3, label="certified")
    ax.plot(cont["tau_grid"], cont["point_contained_fraction"], "s--", ms=3, label="point estimate")
    ax.set_xlabel("tolerance tau")
    ax.set_ylabel("fraction of transcripts contained")
    ax.set_title("Certified off-target containment")
    ax.legend(fontsize=8)

    # 2. Radius and rho vs STRING threshold.
    ax = axes[0, 1]
    ths = [d["threshold"] for d in thr]
    radii = [d["radius_sym"] if d["radius_sym"] is not None else np.nan for d in thr]
    ax.plot(ths, radii, "o-", color="C0", label="certified radius")
    ax.set_xlabel("STRING confidence threshold")
    ax.set_ylabel("certified L2 radius", color="C0")
    ax2 = ax.twinx()
    ax2.plot(ths, [d["degree_rel_fluctuation"] for d in thr], "s--", color="C3", label="rho")
    ax2.axhline(1.0, color="k", ls=":", lw=0.8)
    ax2.set_ylabel("degree fluctuation rho", color="C3")
    ax.set_title("Radius vs threshold (Theorem 5)")

    # 3. Degradation: rho vs subsample fraction and false-edge fraction.
    ax = axes[1, 0]
    sub = deg["subsample"]
    ax.plot(
        [d["level"] for d in sub],
        [d["degree_rel_fluctuation"] for d in sub],
        "o-",
        label="subsample edges",
    )
    fe = deg["false_edges"]
    ax.plot(
        [d["level"] for d in fe],
        [d["degree_rel_fluctuation"] for d in fe],
        "s-",
        label="inject false edges",
    )
    ax.axhline(1.0, color="k", ls="--", label="rho = 1 (breakdown)")
    if deg["breakdown_subsample_fraction"] is not None:
        ax.axvline(deg["breakdown_subsample_fraction"], color="gray", ls=":", label="breakdown")
    ax.set_xlabel("corruption level")
    ax.set_ylabel("degree fluctuation rho")
    ax.set_title("Graceful degradation and C5 breakdown")
    ax.legend(fontsize=8)

    # 4. Monte-Carlo cost N vs delta.
    ax = axes[1, 1]
    ax.semilogy([r["delta"] for r in mc], [r["n_samples"] for r in mc], "o-")
    ax.set_xlabel("delta")
    ax.set_ylabel("samples N needed")
    ax.set_title("Monte-Carlo cost for a (1 - delta) CI")

    fig.tight_layout()
    fig.savefig(out_dir / "realdata.png", dpi=120)
    plt.close(fig)
    print(f"[realdata] wrote {out_dir / 'realdata.png'}")


if __name__ == "__main__":
    sys.exit(main())
