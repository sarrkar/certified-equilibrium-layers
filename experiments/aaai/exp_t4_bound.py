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

HARD DATA POLICY (CLAUDE.md): REAL DATA ONLY. Every input comes from a real loader
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (logged)")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--string-links",
        type=str,
        default=None,
        help="real STRING file (required; no fabricated GRN)",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(f"[t4] seed={args.seed} alpha={args.alpha}")

    per_dataset = {}
    pending = {}
    for name, assay in DATASETS:
        try:
            load_real_dataset(
                name, args.string_links, args.alpha
            )  # raises until the real join is wired
            # When wired, assemble inputs and call assess_dataset(name, assay, ...).
        except (FileNotFoundError, NotImplementedError) as exc:
            pending[name] = str(exc)
            print(f"[t4] {name} ({assay}) PENDING: {type(exc).__name__}")

    acc = acceptance(per_dataset)
    numbers = {
        "acceptance": acc,
        "seed": args.seed,
        "alpha": args.alpha,
        "bound_formula": "||E - E_true|| <= ||P_a(S)|| ||e - e_true|| + ((1-a)/a) ||P_a(S)|| "
        "||P_a(S')|| ||S - S'|| ||e||  (second resolvent identity for the GRN-edge term)",
        "per_dataset": per_dataset,
        "pending": pending,
        "cross_assay": cross_assay(per_dataset),
        "probabilistic_companion": {
            "status": "not_run",
            "note": "the matrix-Bernstein certificate (certification module) is included only with "
            "real STRING confidences; supply --string-links and the real measured join to enable it.",
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "t4_bound.json"
    out_path.write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    if per_dataset:
        _plot(per_dataset, args.out_dir / "t4_bound.png")
    print(
        f"[t4] ACCEPTANCE {acc['flag']}; assembled {len(per_dataset)} datasets, {len(pending)} pending"
    )
    print(f"[t4] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
