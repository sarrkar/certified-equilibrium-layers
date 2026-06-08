#!/usr/bin/env python3
"""Theorem 4: certified off-target bound, validated on REAL off-target data.

For each off-target dataset it propagates the real seed effect e (ViennaRNA seed
occupancy, with its per-transcript error) over the real GRN S, forms the predicted
off-target E = P_alpha(S) e, and compares E to the measured DESeq2 log2 fold-change
(GSE184929 RNA-seq via the SeedMatchR export; Jackson 2006 and Birmingham 2006
microarray), scored with the SeedMatchR one-sided KS / ECDF metric.

It reports, per dataset:
- the deterministic certified bound (cascade.verify.offtarget_bound): the seed-error
  term ||P_alpha|| ||e - e_true|| plus the GRN-edge term via the second resolvent
  identity, ((1-alpha)/alpha) ||P_alpha(S)|| ||P_alpha(S')|| ||S - S'|| ||e||;
- whether the predicted-vs-measured deviation stays WITHIN the bound;
- the KS D-statistic of seed-matched vs background (measured and predicted);
- whether the bound is meaningful (not vacuously larger than the measured effects);
  if vacuous, the per-transcript row-wise refinement is reported as the tightening.

The cross-assay (RNA-seq vs microarray) behavior is reported, not hidden. The
probabilistic matrix-Bernstein companion (certification module) is included only
when REAL STRING confidences are supplied.

HARD DATA POLICY (OUTLINE.md): REAL DATA ONLY. Every input comes from a real loader
that RAISES when absent; aligning the measured transcripts to the GRN nodes and
computing e from real 3'UTR sequences is a tracked gap that RAISES
NotImplementedError. No number is produced from non-real inputs.

Output results/aaai/t4_bound.json and figures. Accepts --seed and logs it. No LaTeX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cascade.verify import offtarget_bound as ob  # noqa: E402

OUT = REPO / "results" / "aaai"
DATASETS = (
    ("gse184929", "rna_seq"),
    ("jackson2006", "microarray"),
    ("birmingham2006", "microarray"),
)
VACUOUS_FACTOR = 10.0


def assess_dataset(
    name: str,
    assay: str,
    e_pred: np.ndarray,
    e_measured: np.ndarray,
    S,
    alpha: float,
    e: np.ndarray,
    e_err_norm: float,
    s_err_norm: float,
    seed_match_mask: np.ndarray,
    p_prime_norm: float | None = None,
) -> dict:
    """Assemble the certified bound and compare it to the real measured deviation."""
    from cascade.data.offtarget import seedmatch_ks_test

    e_norm = float(np.linalg.norm(e))
    cert = ob.certified_bound(S, alpha, e_norm, e_err_norm, s_err_norm, p_prime_norm=p_prime_norm)
    check = ob.verify_bound(e_pred, e_measured, cert["bound"])
    vac = ob.is_vacuous(cert["bound"], e_measured, factor=VACUOUS_FACTOR)
    rw = ob.rowwise_bound(S, alpha, e_norm, e_err_norm, s_err_norm)

    ks_measured = seedmatch_ks_test(e_measured, seed_match_mask, alternative="greater")
    ks_predicted = seedmatch_ks_test(e_pred, seed_match_mask, alternative="greater")
    meaningful = bool(check["holds"] and not vac["vacuous"])
    return {
        "assay": assay,
        "n_transcripts": int(len(e_measured)),
        "alpha": alpha,
        "certified_bound": cert,
        "measured_deviation": check["deviation"],
        "bound_holds": check["holds"],
        "bound_over_measured": vac["bound_over_measured"],
        "vacuous": vac["vacuous"],
        "rowwise_refinement": {
            "max_bound": float(rw.max()),
            "median_bound": float(np.median(rw)),
            "global_bound": cert["bound"],
            "per_transcript_holds": bool(np.all(np.abs(e_pred - e_measured) <= rw + 1e-9)),
        },
        "ks_measured": {"D": ks_measured.statistic, "p": ks_measured.pvalue},
        "ks_predicted": {"D": ks_predicted.statistic, "p": ks_predicted.pvalue},
        "meaningful": meaningful,
    }


def load_real_dataset(name: str, string_links: str | None, alpha: float) -> dict:
    """Assemble the real (e, S, measured, seed-match) for one off-target dataset.

    Raises FileNotFoundError if the measured calls or the STRING file are absent,
    and NotImplementedError for the transcript-to-node alignment and the
    sequence-to-e (ViennaRNA seed occupancy) join, which is a tracked data gap. No
    value is returned from non-real inputs.
    """
    from cascade.data import (
        load_birmingham2006,
        load_jackson2003,
        load_jackson2006,
        load_seedseq,
        load_string_confidence,
    )

    microarray_loaders = {
        "jackson2003": load_jackson2003,
        "jackson2006": load_jackson2006,
        "birmingham2006": load_birmingham2006,
        "seedseq": load_seedseq,
    }
    if name in microarray_loaders:
        calls = microarray_loaders[name]()  # real loader; FileNotFoundError if absent
    elif name == "gse184929":
        from cascade.data import load_schlegel2022

        # GSE184929 RNA-seq via the SeedMatchR export (raises FileNotFoundError if absent).
        calls = load_schlegel2022("Schlegel_2022_Ttr_D1_30mkg")
    else:
        raise KeyError(f"unknown off-target dataset {name!r}")
    if string_links is None:
        raise FileNotFoundError(
            f"{name}: a real STRING confidence file is required (--string-links); none given. "
            "Fetch STRING v12 with scripts/download_data.sh. REAL DATA ONLY."
        )
    nodes, confidence = load_string_confidence(string_links, threshold=400)
    raise NotImplementedError(
        f"{name}: aligning the {len(calls)} measured off-target transcripts to the STRING nodes "
        f"({confidence.shape[0]}) and computing the per-transcript seed occupancy e (ViennaRNA, "
        "with its error) and the GRN-edge uncertainty ||S - S_true|| from real 3'UTR sequences is a "
        "tracked data gap. The certified bound runs the moment that real join is wired."
    )


def acceptance(per_dataset: dict) -> dict:
    if not per_dataset:
        return {
            "flag": "PENDING",
            "reason": "no real off-target dataset assembled (real e / S / measured join pending)",
        }
    holds = all(d["bound_holds"] for d in per_dataset.values())
    meaningful = all(d["meaningful"] for d in per_dataset.values())
    vacuous = [n for n, d in per_dataset.items() if d["vacuous"]]
    flag = "PASS" if (holds and meaningful) else "FAIL"
    return {
        "flag": flag,
        "bound_holds_all": bool(holds),
        "meaningful_all": bool(meaningful),
        "vacuous_datasets": vacuous,
        "tightening": (
            "row-wise (per-transcript) refinement reduces the bound on the flagged datasets; see "
            "rowwise_refinement"
            if vacuous
            else "global bound is already meaningful; no refinement needed"
        ),
        "note": "The certified bound must hold on the real data AND be meaningful (not vacuously "
        "larger than the measured effects). A vacuous global bound is flagged and the per-transcript "
        "row-wise refinement is reported as the needed tightening.",
    }


def cross_assay(per_dataset: dict) -> dict:
    rna = [d for d in per_dataset.values() if d["assay"] == "rna_seq"]
    micro = [d for d in per_dataset.values() if d["assay"] == "microarray"]

    def _mean_d(items):
        vals = [d["ks_measured"]["D"] for d in items if d["ks_measured"]["D"] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "rna_seq_mean_ks_D": _mean_d(rna),
        "microarray_mean_ks_D": _mean_d(micro),
        "note": "RNA-seq (GSE184929) and microarray (Jackson, Birmingham) are different assays; "
        "differences in the KS separation and the bound tightness are expected and reported.",
    }


def _plot(per_dataset: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[t4] matplotlib not installed; skipped figure")
        return
    if not per_dataset:
        return
    names = list(per_dataset)
    x = np.arange(len(names))
    dev = [per_dataset[n]["measured_deviation"] for n in names]
    bnd = [per_dataset[n]["certified_bound"]["bound"] for n in names]
    rw = [per_dataset[n]["rowwise_refinement"]["max_bound"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.25, dev, 0.25, label="measured deviation")
    ax.bar(x, bnd, 0.25, label="certified bound (global)")
    ax.bar(x + 0.25, rw, 0.25, label="rowwise max bound")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_ylabel("L2 magnitude")
    ax.set_title("T4 certified bound vs measured deviation (per dataset)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[t4] wrote {path}")


def assess_all(seed: int, string_links: str | None, alpha: float) -> tuple[dict, dict]:
    """Assemble + assess every off-target dataset for one seed (raises -> pending)."""
    per_dataset, pending = {}, {}
    for name, assay in DATASETS:
        try:
            load_real_dataset(name, string_links, alpha)  # raises until the real join is wired
            # When wired: assemble inputs and
            #   per_dataset[name] = assess_dataset(name, assay, ..., p_prime_norm=...)
            # with the resolvent norms computed at this seed.
        except (FileNotFoundError, NotImplementedError) as exc:
            pending[name] = {"assay": assay, "source": str(exc)}
    return per_dataset, pending


def aggregate_per_dataset(per_seed_pd: list[dict], seeds: list[int]) -> dict:
    """Conservatively merge per-seed assessments: worst-case bound, holds on every seed."""
    common = set(per_seed_pd[0]) if per_seed_pd else set()
    for pd in per_seed_pd[1:]:
        common &= set(pd)
    merged = {}
    for name in common:
        runs = [pd[name] for pd in per_seed_pd]
        merged[name] = {
            "assay": runs[0]["assay"],
            "bound_holds": all(r["bound_holds"] for r in runs),
            "meaningful": all(r["meaningful"] for r in runs),
            "vacuous": any(r["vacuous"] for r in runs),
            "certified_bound": {"bound": max(r["certified_bound"]["bound"] for r in runs)},
            "measured_deviation": max(r["measured_deviation"] for r in runs),
            "rowwise_refinement": runs[0]["rowwise_refinement"],
            "ks_measured": runs[0]["ks_measured"],
            "ks_predicted": runs[0]["ks_predicted"],
            "per_seed_bound": [r["certified_bound"]["bound"] for r in runs],
        }
    return merged


def aggregate_acceptance(merged: dict, per_seed_pd: list[dict], seeds: list[int]) -> dict:
    acc = acceptance(merged)
    acc["seeds"] = list(seeds)
    acc["n_seeds"] = len(seeds)
    acc["per_seed_flags"] = [acceptance(pd)["flag"] for pd in per_seed_pd]
    return acc


def _align_seed_input(grn, ds) -> tuple:
    """Map a rat off-target contrast onto the human TRRUST nodes by uppercase symbol.

    Returns (e, measured, seed_mask, aligned) over the n_nodes TRRUST vector: e is the
    direct seed-mediated repression input (negative seed-site count, the
    ViennaRNA-confirmed canonical seed of the Schlegel Ttr guide), measured is the
    DESeq2 log2FC, seed_mask marks seed-matched nodes, aligned marks measured nodes.
    Cross-species note: TRRUST is human and GSE184929 is rat, so the alignment is by
    orthologous gene SYMBOL (rat Ttr -> human TTR); unmapped genes are dropped.
    """
    idx = {sym.upper(): i for i, sym in enumerate(grn.nodes)}
    n = grn.n_nodes
    e = np.zeros(n)
    measured = np.zeros(n)
    seed_mask = np.zeros(n, dtype=bool)
    aligned = np.zeros(n, dtype=bool)
    f = ds.frame
    smc = (
        f["seed_match_count"].to_numpy()
        if "seed_match_count" in f.columns
        else np.full(len(f), np.nan)
    )
    for sym, lfc, sc in zip(f["gene_name"].astype(str), f["log2fc"], smc, strict=True):
        node = idx.get(sym.strip().upper())
        if node is None:
            continue
        aligned[node] = True
        measured[node] = float(lfc) if np.isfinite(lfc) else 0.0
        if np.isfinite(sc):
            e[node] = -float(sc)  # more canonical seed sites -> stronger direct repression
            seed_mask[node] = sc > 0
    return e, measured, seed_mask, aligned


def assess_trrust_join(
    grn, alpha: float, ds, name: str, e_rel_err: float, s_rel_err: float
) -> dict:
    """Propagate the seed input over the real TRRUST P_alpha and certify the deviation."""
    from cascade.data.offtarget import seedmatch_ks_test
    from cascade.verify.stability import operator_2norm

    S = grn.S
    e, measured, seed_mask, aligned = _align_seed_input(grn, ds)
    e_pred_full = ob.propagate(S, e, alpha)  # E = P_alpha(S) e over the FULL TRRUST
    a = aligned
    e_pred, e_meas, m = e_pred_full[a], measured[a], seed_mask[a]

    e_norm = float(np.linalg.norm(e))
    s2 = operator_2norm(S)
    p_norm = ob.p_alpha_norm(S, alpha)  # true ||P_alpha(S)||_2 (the T2 resolvent norm)
    e_err = e_rel_err * e_norm  # seed-thermodynamic input uncertainty budget
    s_err = s_rel_err * s2  # GRN-edge uncertainty budget ||S - S'||
    cert = ob.certified_bound(S, alpha, e_norm, e_err, s_err, p_norm=p_norm, p_prime_norm=p_norm)
    check = ob.verify_bound(e_pred, e_meas, cert["bound"])
    vac = ob.is_vacuous(cert["bound"], e_meas)
    rw = ob.rowwise_bound(S, alpha, e_norm, e_err, s_err)[a]  # per-transcript refinement

    has_split = bool(m.any() and (~m).any())
    ks_meas = seedmatch_ks_test(e_meas, m, alternative="greater") if has_split else None
    ks_pred = seedmatch_ks_test(e_pred, m, alternative="greater") if has_split else None
    return {
        "assay": "rna_seq",
        "n_aligned_to_trrust": int(a.sum()),
        "n_seed_matched_aligned": int(m.sum()),
        "p_alpha_2norm": p_norm,
        "operator_2norm_S": s2,
        "certified_bound": cert,
        "measured_deviation": check["deviation"],
        "bound_holds": check["holds"],
        "bound_over_measured": vac["bound_over_measured"],
        "vacuous": vac["vacuous"],
        "rowwise_refinement": {
            "max_bound": float(rw.max()) if rw.size else None,
            "median_bound": float(np.median(rw)) if rw.size else None,
            "global_bound": cert["bound"],
        },
        "ks_measured": {
            "D": ks_meas.statistic if ks_meas else None,
            "p": ks_meas.pvalue if ks_meas else None,
        },
        "ks_predicted": {
            "D": ks_pred.statistic if ks_pred else None,
            "p": ks_pred.pvalue if ks_pred else None,
        },
    }


def grn_propagation_join(
    alpha: float, seedmatchr_dir: Path, cache_root: Path, e_rel_err: float, s_rel_err: float
) -> dict:
    """T4 certified bound on the real TRRUST GRN, per Schlegel Ttr contrast."""
    from cascade.data import SEEDMATCHR_DATASETS, load_schlegel2022

    try:
        from cascade.data import load_trrust

        grn = load_trrust(species="human")
    except (FileNotFoundError, NotImplementedError) as exc:
        return {"status": "pending_real_data", "source": str(exc)}
    per_contrast, pending = {}, {}
    for name in SEEDMATCHR_DATASETS:
        try:
            ds = load_schlegel2022(name, raw_dir=seedmatchr_dir, cache_root=cache_root)
        except FileNotFoundError as exc:
            pending[name] = str(exc)
            continue
        if not ds.has_seed_annotation:
            pending[name] = "loaded but no seed-match annotation; run scripts/annotate_seedmatch.py"
            continue
        per_contrast[name] = assess_trrust_join(grn, alpha, ds, name, e_rel_err, s_rel_err)
    return {
        "status": "ok" if per_contrast else "pending_real_data",
        "grn": grn.source,
        "n_nodes": int(grn.n_nodes),
        "alpha": alpha,
        "uncertainty_budget": {"e_rel_err": e_rel_err, "s_rel_err": s_rel_err},
        "bound_formula": "||E - E_true|| <= ||P_a(S)|| ||e - e_true|| + ((1-a)/a) ||P_a(S)||^2 "
        "||S - S'|| ||e||  (T2 contraction constant + second-resolvent GRN-edge term)",
        "cross_species_note": "TRRUST is human, GSE184929 is rat; aligned by orthologous gene symbol. "
        "e is the seed-match-derived direct repression of the ViennaRNA-confirmed Ttr seed.",
        "per_contrast": per_contrast,
        "pending": pending,
    }


BUDGET_SWEEP = ((0.05, 0.05), (0.10, 0.10), (0.20, 0.20))  # (seed, edge) rel-err pairs


def budget_sweep(
    alpha: float, seedmatchr_dir: Path, cache_root: Path, budgets=BUDGET_SWEEP
) -> dict:
    """Sweep the (seed, edge) uncertainty budgets for the TRRUST certified bound.

    The seed propagation E = P_alpha(S) e and its measured deviation are
    budget-independent (computed once per contrast); only the certified bound, the
    bound/deviation ratio, and the binds / vacuous flags move with the budget.
    """
    from cascade.data import SEEDMATCHR_DATASETS, load_schlegel2022
    from cascade.verify.stability import operator_2norm

    try:
        from cascade.data import load_trrust

        grn = load_trrust(species="human")
    except (FileNotFoundError, NotImplementedError) as exc:
        return {"status": "pending_real_data", "source": str(exc)}

    S = grn.S
    s2 = operator_2norm(S)
    p_norm = ob.p_alpha_norm(S, alpha)  # ||P_alpha(S)||_2, budget-independent
    rows: list = []
    pending: dict = {}
    for name in SEEDMATCHR_DATASETS:
        try:
            ds = load_schlegel2022(name, raw_dir=seedmatchr_dir, cache_root=cache_root)
        except FileNotFoundError as exc:
            pending[name] = str(exc)
            continue
        if not ds.has_seed_annotation:
            pending[name] = "loaded but no seed-match annotation; run scripts/annotate_seedmatch.py"
            continue
        e, measured, _seed_mask, aligned = _align_seed_input(grn, ds)
        e_pred = ob.propagate(S, e, alpha)[aligned]
        e_meas = measured[aligned]
        deviation = float(np.linalg.norm(e_pred - e_meas))
        e_norm = float(np.linalg.norm(e))
        meas_scale = float(np.linalg.norm(e_meas))
        for e_rel, s_rel in budgets:
            cert = ob.certified_bound(
                S, alpha, e_norm, e_rel * e_norm, s_rel * s2, p_norm=p_norm, p_prime_norm=p_norm
            )
            bound = cert["bound"]
            vacuous = bool(meas_scale > 0 and bound > VACUOUS_FACTOR * meas_scale)
            binds = bool(deviation <= bound * (1.0 + 1e-9) and not vacuous)
            rows.append(
                {
                    "budget": {"e_rel_err": e_rel, "s_rel_err": s_rel},
                    "contrast": name,
                    "certified_bound": bound,
                    "measured_deviation": deviation,
                    "bound_over_deviation": (bound / deviation if deviation > 0 else None),
                    "binds": binds,
                    "vacuous": vacuous,
                }
            )
    return {
        "status": "ok" if rows else "pending_real_data",
        "grn": grn.source,
        "n_nodes": int(grn.n_nodes),
        "alpha": alpha,
        "p_alpha_2norm": p_norm,
        "operator_2norm_S": s2,
        "budgets": [{"e_rel_err": e, "s_rel_err": s} for e, s in budgets],
        "ratio_definition": "bound_over_deviation = certified_bound / measured_deviation; "
        "vacuous = bound > 10 * ||measured||",
        "rows": rows,
        "pending": pending,
    }


def measured_offtarget(seedmatchr_dir: Path, cache_root: Path) -> dict:
    """Real SeedMatchR one-sided KS on the MEASURED GSE184929 DESeq2 log2FC, per Ttr contrast.

    Independent of the certified-bound (GRN-propagation) gap: needs only the real
    measured log2FC and the native seed-match annotation (the confirmed Schlegel 2022
    Ttr guide, seed UAUAGAG, scanned against build-matched Rnor_6.0 rat 3'UTRs). D1
    (parent) and D4 (single g7 GNA) share the seed, so the seed-match set is identical;
    the KS contrast measures whether the GNA destabilization removes the seed off-target.
    """
    from cascade.data import SEEDMATCHR_DATASETS, load_schlegel2022

    per_contrast: dict = {}
    pending: dict = {}
    for name in SEEDMATCHR_DATASETS:
        try:
            ds = load_schlegel2022(name, raw_dir=seedmatchr_dir, cache_root=cache_root)
        except FileNotFoundError as exc:
            pending[name] = str(exc)
            continue
        if not ds.has_seed_annotation:
            pending[name] = "loaded but no seed-match annotation; run scripts/annotate_seedmatch.py"
            continue
        res = ds.ks(alternative="greater")
        per_contrast[name] = {
            "sirna_id": ds.sirna_id,
            "dose": ds.dose,
            "ks_D": res.statistic,
            "ks_p": res.pvalue,
            "n_seed_match": res.n_seed_match,
            "n_background": res.n_background,
        }
    return {
        "status": "ok" if per_contrast else "pending_real_data",
        "guide": "UUAUAGAGCAAGAACACUGUUUU",
        "seed": "UAUAGAG",
        "reference": "Ensembl Rnor_6.0 rat 3'UTRs (build-matched to GSE184929)",
        "metric": "SeedMatchR one-sided KS (D, p): seed-matched vs background measured DESeq2 log2FC",
        "per_contrast": per_contrast,
        "pending": pending,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="single RNG seed (logged)")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="multiple seeds; aggregated conservatively (bound holds / meaningful on every seed)",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--e-rel-err", type=float, default=0.1, help="seed-input uncertainty ||e-e_true||/||e||"
    )
    parser.add_argument(
        "--s-rel-err", type=float, default=0.1, help="GRN-edge uncertainty ||S-S'||/||S||_2"
    )
    parser.add_argument(
        "--string-links",
        type=str,
        default=None,
        help="real STRING file (required; no fabricated GRN)",
    )
    parser.add_argument(
        "--seedmatchr-dir",
        type=Path,
        default=REPO / "data" / "raw" / "offtarget" / "seedmatchr",
        help="dir with the seed-match-annotated GSE184929 DESeq2 CSVs",
    )
    parser.add_argument("--cache", type=Path, default=OUT / "cache")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    seeds = args.seeds if args.seeds else [args.seed]
    print(f"[t4] seeds={seeds} alpha={args.alpha}")

    per_seed = [assess_all(s, args.string_links, args.alpha) for s in seeds]
    per_seed_pd = [p[0] for p in per_seed]
    pending = per_seed[0][1]  # the pending set is seed-independent (loading raises identically)
    for name in pending:
        print(f"[t4] {name} PENDING")
    merged = aggregate_per_dataset(per_seed_pd, seeds)
    acc = aggregate_acceptance(merged, per_seed_pd, seeds)

    numbers = {
        "acceptance": acc,
        "seeds": seeds,
        "alpha": args.alpha,
        "bound_formula": "||E - E_true|| <= ||P_a(S)|| ||e - e_true|| + ((1-a)/a) ||P_a(S)|| "
        "||P_a(S')|| ||S - S'|| ||e||  (second resolvent identity for the GRN-edge term)",
        "per_dataset": merged,
        "pending": pending,
        "cross_assay": cross_assay(merged),
        "probabilistic_companion": {
            "status": "not_run",
            "note": "the matrix-Bernstein certificate (certification module) is included only with "
            "real STRING confidences; supply --string-links and the real measured join to enable it.",
        },
        # Real measured off-target signal, independent of the certified-bound GRN gap.
        "measured_offtarget": measured_offtarget(args.seedmatchr_dir, args.cache),
        # Real T4 certified bound: seed input propagated over TRRUST P_alpha.
        "grn_propagation_join": grn_propagation_join(
            args.alpha, args.seedmatchr_dir, args.cache, args.e_rel_err, args.s_rel_err
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "t4_bound.json"
    out_path.write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    if merged:
        _plot(merged, args.out_dir / "t4_bound.png")

    mo = numbers["measured_offtarget"]
    print(
        f"[t4] certified-bound ACCEPTANCE {acc['flag']} ({len(merged)} assembled, {len(pending)} pending)"
    )
    print(f"[t4] measured off-target KS ({mo['status']}):")
    for name, c in mo["per_contrast"].items():
        print(
            f"     {name}: KS D={c['ks_D']:.4f} p={c['ks_p']:.2e} (seed={c['n_seed_match']} bg={c['n_background']})"
        )
    gj = numbers["grn_propagation_join"]
    print(f"[t4] GRN-propagation join over TRRUST ({gj['status']}, alpha={args.alpha}):")
    for name, d in gj.get("per_contrast", {}).items():
        binds = bool(d["bound_holds"] and not d["vacuous"])
        print(
            f"     {name}: bound={d['certified_bound']['bound']:.3f} vs measured-dev="
            f"{d['measured_deviation']:.3f} | binds={binds} vacuous={d['vacuous']} "
            f"(||P_a||={d['p_alpha_2norm']:.2f} rowwise-max={d['rowwise_refinement']['max_bound']:.3f} "
            f"aligned={d['n_aligned_to_trrust']})"
        )
    print(f"[t4] wrote {out_path}")

    # Budget sweep (separate output; the single-budget result above is kept intact).
    sweep = budget_sweep(args.alpha, args.seedmatchr_dir, args.cache)
    sweep_path = args.out_dir / "t4_budget_sweep.json"
    sweep_path.write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    print(
        f"\n[t4] certified-bound budget sweep over TRRUST ({sweep['status']}, alpha={args.alpha}):"
    )
    if sweep.get("rows"):
        short = {
            "Schlegel_2022_Ttr_D1_30mkg": "D1_30mkg",
            "Schlegel_2022_Ttr_D4_30mkg": "D4_30mkg",
            "Schlegel_2022_Ttr_D1_10mkg": "D1_10mkg",
        }
        print(
            f"  {'budget':>9}  {'contrast':<9} {'bound':>9} {'meas-dev':>9} {'ratio':>7} {'binds':>6} {'vacuous':>8}"
        )
        for r in sweep["rows"]:
            bud = f"{int(r['budget']['e_rel_err'] * 100)}/{int(r['budget']['s_rel_err'] * 100)}%"
            ratio = r["bound_over_deviation"]
            print(
                f"  {bud:>9}  {short.get(r['contrast'], r['contrast']):<9} {r['certified_bound']:>9.3f} "
                f"{r['measured_deviation']:>9.3f} {('n/a' if ratio is None else f'{ratio:>7.2f}')} "
                f"{r['binds']!s:>6} {r['vacuous']!s:>8}"
            )
    print(f"[t4] wrote {sweep_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
