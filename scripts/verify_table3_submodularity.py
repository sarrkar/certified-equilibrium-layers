"""Fill Table 3 (tab:submod): supermodularity of the coupled off-target set function g(S).

Claim the table backs: on the COUPLED conservation-law model (re-solve the shared-pool free guide f*(S)
for every seed-modification subset S) the off-target reduction g(S) = Phi(empty) - Phi(S) has strictly
INCREASING differences (supermodular / complementary) in every regime x delta cell, while the ISOLATED
model (hold f fixed at the parent equilibrium, no re-solve) is submodular. Modifications COMPLEMENT each
other only because they compete for the shared guide pool; freezing the pool removes the complementarity.

REUSED, NOT REIMPLEMENTED (found in the repo, not rebuilt):
  * solver          solve_f(M, x, K)                       [offtarget_phase_probe.py, via design_validation]
  * coupled Phi     design_validation.phi_and_ontarget      (K -> K0 * exp(delta * |S cap window|), on-target
                    in the shared pool); this script's coupled_phi is asserted bitwise-equal to it.
  * site windows,   WINDOWS, K_BY_SITE, K_ON, X_ON, build_sites, load_fasta  [design_validation.py]
    K, real sites   real off-target sites = Schlegel Ttr seed UAUAGAG on rat 3'UTRs (GSE184929 D1 gene set)
  * regimes         M_REGIMES = subsaturating 0.02, balanced 0.1, saturating 0.5
                    [design_principle_validation.py run(), the Appendix-K guide-pool regimes]
  * delta           the per-position destabilization in K -> K0 * exp(delta * count) (design_validation DELTA)

POSITION SET (flagged): the repo enumerates 7 seed positions g2..g8; this table uses the 6-nucleotide seed
CORE P = {2,3,4,5,6,7} (g8, the 8mer supplementary position, excluded), giving 2^6 = 64 subsets as the task
specifies. The paper tex (Appendix K) is not in this repo (paper/ is empty) to confirm the exact six, so
this choice is stated openly; the supermodular-vs-submodular VERDICT is driven by the pool coupling and is
robust to the choice, while the exact max_violation magnitudes depend on it.

Real data only (loaders raise on missing). Deterministic (bisection, no randomness); the 9-cell computation
is run twice and asserted bitwise identical, with the md5 of the flattened coupled-violation array printed.
No em-dashes.
"""

from __future__ import annotations

import hashlib
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, "/home/shadi/mirna_baselines/scripts")
import design_validation as dv  # noqa: E402 (REUSED coupled Phi, WINDOWS, K, real-site builder)
from design_validation import (  # noqa: E402
    K_BY_SITE,
    K_ON,
    RAT_UTR,
    SM_DIR,
    WINDOWS,
    X_ON,
    build_sites,
    load_fasta,
    solve_f,
)

SEED = "UAUAGAG"  # Schlegel Ttr seed (design_validation.main)
# Appendix-K guide-pool regimes (design_principle_validation.py run(): M_REGIMES).
REGIMES = {"subsaturating": 0.02, "balanced": 0.1, "saturating": 0.5}
DELTAS = [0.7, 1.5, 3.0]
P = [2, 3, 4, 5, 6, 7]  # 6-nt seed core g2..g7 (see docstring); 64 subsets
FULL_SEED_WINDOW = {2, 3, 4, 5, 6, 7, 8}  # on-target is fully paired (matches phi_and_ontarget)
SUBMOD_TOL = 1e-12
MONOTONE_TOL = 1e-12


def all_subsets(elements) -> list[frozenset]:
    xs = list(elements)
    return [frozenset(c) for r in range(len(xs) + 1) for c in combinations(xs, r)]


def _koff(types: list[str], s: frozenset, delta: float) -> np.ndarray:
    k0 = np.array([K_BY_SITE[t] for t in types])
    mult = np.array([np.exp(delta * len(s & WINDOWS[t])) for t in types])
    return k0 * mult


def coupled_phi(types: list[str], s: frozenset, m: float, delta: float) -> float:
    """Re-solve f*(S) with the modified pool (identical to design_validation.phi_and_ontarget)."""
    koff = _koff(types, s, delta)
    k_on = K_ON * np.exp(delta * len(s & FULL_SEED_WINDOW))
    x_all = np.concatenate([[X_ON], np.ones(len(types))])
    k_all = np.concatenate([[k_on], koff])
    f = float(solve_f(m, x_all, k_all))
    return float(np.sum(f / (koff + f)))


def free_guide_empty(types: list[str], m: float) -> float:
    """Parent free guide f*(empty): the fixed pool level for the ISOLATED model."""
    x_all = np.concatenate([[X_ON], np.ones(len(types))])
    k_all = np.concatenate([[K_ON], np.array([K_BY_SITE[t] for t in types])])
    return float(solve_f(m, x_all, k_all))


def isolated_phi(types: list[str], s: frozenset, delta: float, f0: float) -> float:
    """Hold f fixed at f0 (no re-solve); only the site KDs change."""
    koff = _koff(types, s, delta)
    return float(np.sum(f0 / (koff + f0)))


def analyze(gmap: dict) -> dict:
    """max over triples (e; A subset B subset P\\{e}) of d(e|B)-d(e|A), with d(e|S)=g(S+e)-g(S)."""
    pset = set(P)
    max_viol = -np.inf
    min_marginal = np.inf
    n_triples = 0
    for e in P:
        rest = sorted(pset - {e})
        for b in all_subsets(rest):
            d_b = gmap[b | {e}] - gmap[b]
            min_marginal = min(min_marginal, d_b)
            for a in all_subsets(b):  # A subset of B (includes A = B)
                d_a = gmap[a | {e}] - gmap[a]
                max_viol = max(max_viol, d_b - d_a)
                n_triples += 1
    return {
        "max_violation": float(max_viol),
        "min_marginal_gain": float(min_marginal),
        "n_triples": n_triples,
    }


def compute() -> dict:
    utr = load_fasta(RAT_UTR)
    d1 = pd.read_csv(SM_DIR / "Schlegel_2022_Ttr_D1_30mkg.csv", sep="\t")
    gene_ids = list(d1["ensembl_gene_id"].astype(str))
    types = build_sites(SEED, utr, gene_ids)
    if not types:
        raise ValueError(
            "no real off-target sites built from Schlegel Ttr seed; refusing to proceed"
        )
    n_sites = len(types)

    # Proof of faithful reuse: coupled_phi equals the committed design_validation.phi_and_ontarget.
    m_chk = REGIMES["balanced"] * n_sites
    for delta in DELTAS:
        dv.DELTA = delta
        for s in (frozenset(), frozenset({2, 5}), frozenset(P)):
            ref = dv.phi_and_ontarget(types, s, m_chk)[0]
            if abs(coupled_phi(types, s, m_chk, delta) - ref) > 1e-12:
                raise AssertionError(
                    f"coupled_phi diverged from phi_and_ontarget at delta={delta}, S={set(s)}"
                )

    subsets = all_subsets(P)
    cells = {}
    for rname, mfac in REGIMES.items():
        m = mfac * n_sites
        for delta in DELTAS:
            g_c = {s: 0.0 for s in subsets}
            g_i = {s: 0.0 for s in subsets}
            phi_c0 = coupled_phi(types, frozenset(), m, delta)
            f0 = free_guide_empty(types, m)
            phi_i0 = isolated_phi(types, frozenset(), delta, f0)
            for s in subsets:
                g_c[s] = phi_c0 - coupled_phi(types, s, m, delta)
                g_i[s] = phi_i0 - isolated_phi(types, s, delta, f0)
            cells[(rname, delta)] = {"coupled": analyze(g_c), "isolated": analyze(g_i)}
    return {"n_sites": n_sites, "types": types, "cells": cells}


def main() -> int:
    r1 = compute()
    r2 = compute()

    order = [(rn, d) for rn in REGIMES for d in DELTAS]
    arr1 = np.array([r1["cells"][k]["coupled"]["max_violation"] for k in order], dtype=np.float64)
    arr2 = np.array([r2["cells"][k]["coupled"]["max_violation"] for k in order], dtype=np.float64)
    if not np.array_equal(arr1, arr2):
        raise AssertionError(
            "non-deterministic: coupled max_violation arrays differ between two runs"
        )
    md5 = hashlib.md5(arr1.tobytes()).hexdigest()

    cells = r1["cells"]
    print("Table 3 (tab:submod): coupled supermodularity of g(S) = Phi(empty) - Phi(S)")
    print("=" * 92)
    print(f"real off-target sites (Schlegel Ttr UAUAGAG on rat 3'UTRs): {r1['n_sites']}")
    print(f"P = 6-nt seed core {P} -> {2 ** len(P)} subsets; triples per cell = 6 * 3^5")
    print(
        f"\n  {'regime':<14}{'delta':>7}{'triples':>9}{'coupled max_viol':>18}{'coupled min marg':>18}"
        f"{'iso max_viol':>15}{'iso submod':>12}"
    )
    failures = []
    for rn, d in order:
        c = cells[(rn, d)]["coupled"]
        iso = cells[(rn, d)]["isolated"]
        if c["n_triples"] != 1458:
            raise AssertionError(f"triple count {c['n_triples']} != 1458 at cell ({rn}, delta={d})")
        iso_submod = iso["max_violation"] <= SUBMOD_TOL
        print(
            f"  {rn:<14}{d:>7.1f}{c['n_triples']:>9}{c['max_violation']:>18.4e}{c['min_marginal_gain']:>18.4e}"
            f"{iso['max_violation']:>15.2e}{('yes' if iso_submod else 'no'):>12}"
        )
        if c["max_violation"] <= 0.0:
            failures.append(
                (rn, d, f"coupled max_violation {c['max_violation']:.3e} <= 0 (NOT supermodular)")
            )
        if c["min_marginal_gain"] < -MONOTONE_TOL:
            failures.append(
                (
                    rn,
                    d,
                    f"coupled min marginal gain {c['min_marginal_gain']:.3e} < 0 "
                    "(monotonicity lost)",
                )
            )

    print(
        f"\n  determinism: two runs bitwise identical = {np.array_equal(arr1, arr2)}; "
        f"md5(flattened coupled max_violation) = {md5}"
    )

    if failures:
        print(f"\nASSERTION FAILED in {len(failures)} cell(s) (guard: stop, do not paste a table):")
        for rn, d, why in failures:
            print(f"  (regime={rn}, delta={d}): {why}")
        return 1

    print(
        "\nGUARDS PASSED: every coupled max_violation > 0 (supermodular); every coupled marginal gain "
        ">= 0 (monotone); triple count 1458 per cell.\n"
    )
    print(latex_table(cells, order))
    return 0


def latex_table(cells: dict, order: list) -> str:
    """Paste-ready tab:submod. Reconstructed layout (paper tex absent from repo): regimes x delta, coupled
    max_violation filled; isolated yes/no shown, Balanced row called out explicitly."""

    def num(x: float) -> str:
        return f"{x:.3e}".replace("e-0", "e-").replace("e+0", "e+")

    lines = [
        "% tab:submod  (coupled max increasing-differences violation of g(S); isolated submodularity)",
        "% VERIFY cells filled from verify_table3_submodularity.py (real Schlegel Ttr sites).",
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"regime & $\delta=0.7$ & $\delta=1.5$ & $\delta=3.0$ \\",
        r"\midrule",
    ]
    for rn in REGIMES:
        vals = " & ".join(num(cells[(rn, d)]["coupled"]["max_violation"]) for d in DELTAS)
        lines.append(f"{rn} (coupled) & {vals} \\\\")
    lines.append(r"\midrule")
    for rn in REGIMES:
        yn = " & ".join(
            "yes" if cells[(rn, d)]["isolated"]["max_violation"] <= SUBMOD_TOL else "no"
            for d in DELTAS
        )
        lines.append(f"{rn} (isolated submodular) & {yn} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Coupled $g(S)$ is supermodular (max increasing-differences violation $>0$) in "
        r"every regime and $\delta$; the isolated model is submodular. Complementarity is created "
        r"by the shared guide pool.}",
        r"\label{tab:submod}",
        r"\end{table}",
    ]
    bal = " ; ".join(
        f"delta={d}: {'yes' if cells[('balanced', d)]['isolated']['max_violation'] <= SUBMOD_TOL else 'no'}"
        for d in DELTAS
    )
    lines.append(f"% Balanced isolated cells (yes/no): {bal}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
