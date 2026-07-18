"""Conservation-law operator certified-direction panel over every parent-vs-modified contrast.

For each contrast in the loader config this computes the single-pool conservation-law operator's
predicted aggregate off-target OCCUPANCY on the seed-matched transcript set under the parent guide
and under the modified guide (whose seed-position modification RAISES the seed dissociation constant
K), records the predicted change, and compares its SIGN and RANK to the measured attenuation of the
SeedMatchR-style ECDF shift (Prompt 3) computed on the same contrast.

Operator (single pool, conservation of loaded guide M over the seed-matched set S):
  bound_t = x_t * theta_t,  theta_t = f / (K_t + f),  conservation: f + sum_{t in S} x_t*theta_t = M.
  f is the free guide, solved by BISECTION on the monotone residual g(f)=f+sum x_t f/(K_t+f)-M over
  [0, M] (house rule: never the Picard map). Aggregate off-target occupancy O = sum_t x_t*theta_t.
  x_t = relative transcript abundance (DESeq2 baseMean, normalized) times canonical site count.
  K_t = McGeary AGO-RBNS relative Kd by best canonical site type. The modification raises the seed K
  by a factor r>1 (weaker binding); the CERTIFIED comparative-statics sign dO/d(log K_seed) < 0 holds
  for every r>1 and every M>0 (single-pool residual Jacobian D = 1 + sum x*sigma'(f) >= 1, an
  M-matrix), so the predicted change is negative UNCONDITIONALLY. r is a stated modeling parameter,
  NOT fit to the measured attenuation (that would be circular); we report the certified sign plus an
  illustrative magnitude at a reference r, and verify sign-invariance over sweeps of r and M.

Measured attenuation (per contrast, from the Prompt 3 test): KS D of seed-matched vs background
log2FC for parent (D_p) and modified (D_m); attenuation = 1 - D_m/D_p (fraction of the ECDF shift
removed). Sign of the measured effect is "reduction" when D_m < D_p.

HONESTY: an independent per-target seed scorer re-run with the raised seed K ALSO predicts reduction
(raising K lowers occupancy monotonically), so this panel validates the certified DIRECTION on real
data; it does NOT establish superiority of the operator over independent-scoring baselines. The
capability separation lives in the competitive-derepression experiment, not here.

No synthetic data: contrasts whose processed inputs are absent are reported BLOCKED (loader raises).
Deterministic: bisection, fixed parameters, sorted iteration, no random. Bitwise-identical on rerun.
No em-dashes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from seedmod_loader import (  # noqa: E402  (loader lives alongside this script)
    load_config,
    load_contrast,
)

CONFIG = REPO / "scripts" / "seedmod_loader_config.json"
OUT_JSON = REPO / "results" / "operator_certified_panel.json"

# McGeary et al. 2019 AGO-RBNS relative Kd by canonical seed-site type (dissociation constant, lower
# is tighter). Source: McGeary 2019 (Science) AGO-RBNS site-type affinity ordering; the same table
# used in the certified comparative-statics experiments in this repo.
K_BY_SITE = {"8mer": 0.02, "7mer-m8": 0.06, "7mer-A1": 0.10, "6mer": 0.40}
K_SITE_SOURCE = "McGeary et al. 2019 (Science) AGO-RBNS site-type relative Kd ordering"

# Modeling parameters for the operator (stated, NOT fit to the measured attenuation).
M_REF = 0.10  # loaded guide pool as fraction of total seed-matched site abundance (reference)
R_REF = 2.0  # modification raises seed Kd by this factor (VERIFY: GNA g7 fold-change not
#                       measured here; the CERTIFIED claim is the sign, invariant to any r>1)
R_SOURCE = "VERIFY: seed Kd fold-increase from the seed modification is not measured here; the "
"certified operator claim is the SIGN of the occupancy change, invariant to any r>1 and any M>0"
M_SWEEP = [0.01, 0.1, 1.0, 10.0]
R_SWEEP = [1.5, 2.0, 5.0, 100.0]


def solve_free_guide(x: np.ndarray, k: np.ndarray, m: float) -> float:
    """Bisection on g(f)=f+sum_t x_t f/(k_t+f)-m over [0, m]. Monotone increasing, g(0)=-m<0,
    g(m)=m*(1+sum x/(k+m))>0, so a unique root exists (house rule: bisection, never Picard)."""
    lo, hi = 0.0, m
    for _ in range(200):
        f = 0.5 * (lo + hi)
        g = f + float(np.sum(x * f / (k + f))) - m
        if g > 0.0:
            hi = f
        else:
            lo = f
    return 0.5 * (lo + hi)


def aggregate_occupancy(x: np.ndarray, k: np.ndarray, m: float) -> float:
    f = solve_free_guide(x, k, m)
    return float(np.sum(x * f / (k + f)))


def operator_prediction(x: np.ndarray, k: np.ndarray) -> dict:
    """Aggregate occupancy under parent K vs modified (r*K); certified sign is negative."""
    o_parent = aggregate_occupancy(x, k, M_REF)
    o_mod = aggregate_occupancy(x, R_REF * k, M_REF)
    change = o_mod - o_parent
    frac = change / o_parent if o_parent > 0 else float("nan")
    # certified gradient sign: finite difference at the parent K (small r), expect < 0
    o_eps = aggregate_occupancy(x, 1.01 * k, M_REF)
    grad_sign = float(np.sign(o_eps - o_parent))
    # sign-invariance sweep over M and r
    all_negative = True
    for m in M_SWEEP:
        op = aggregate_occupancy(x, k, m)
        for r in R_SWEEP:
            if aggregate_occupancy(x, r * k, m) - op >= 0.0:
                all_negative = False
    return {
        "occupancy_parent": o_parent,
        "occupancy_modified": o_mod,
        "predicted_change": change,
        "predicted_change_fraction": frac,
        "certified_gradient_sign": grad_sign,  # expect -1.0 (dO/d seedK < 0)
        "sign_negative_over_all_M_and_r": bool(all_negative),
        "M_ref": M_REF,
        "r_ref": R_REF,
    }


def ks_left_shift(matched: np.ndarray, background: np.ndarray) -> dict:
    res = stats.ks_2samp(matched, background, alternative="greater")  # matched left-shifted
    return {
        "ks_D": float(res.statistic),
        "ks_p": float(res.pvalue),
        "n_matched": int(len(matched)),
        "n_background": int(len(background)),
    }


def measured_attenuation(df_long: pd.DataFrame) -> dict:
    """ECDF-shift attenuation from the SeedMatchR-style test on the find_sites annotation."""
    out = {}
    for arm in ("parent", "modified"):
        a = df_long[df_long["arm"] == arm]
        matched = a.loc[a["seed_matched"], "log2FoldChange"].to_numpy()
        background = a.loc[~a["seed_matched"], "log2FoldChange"].to_numpy()
        out[arm] = ks_left_shift(matched, background)
    d_p, d_m = out["parent"]["ks_D"], out["modified"]["ks_D"]
    out["attenuation_fraction"] = float(1.0 - d_m / d_p) if d_p > 0 else float("nan")
    out["measured_reduction"] = bool(d_m < d_p)
    return out


def build_operator_inputs(c: dict, df_long: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Relative abundance x_t and site-type Kd k_t for the seed-matched set (on-target removed)."""
    parent = df_long[df_long["arm"] == "parent"].copy()
    sm = parent[parent["seed_matched"] & parent["best_site_type"].isin(K_BY_SITE)].copy()
    # abundance from the parent DESeq2 baseMean, joined on transcript id
    praw = pd.read_csv(REPO / c["parent_arm"]["file"], sep=c.get("sep", "\t"))
    base = dict(
        zip(
            praw[c["gene_id_col"]].astype(str),
            pd.to_numeric(praw["baseMean"], errors="coerce"),
            strict=True,
        )
    )
    if "symbol" in praw.columns:  # drop the intended on-target gene if identifiable
        on = set(
            praw.loc[praw["symbol"].astype(str).str.lower().isin({"ttr"}), c["gene_id_col"]].astype(
                str
            )
        )
        sm = sm[~sm["transcript_id"].astype(str).isin(on)]
    sm = sm[sm["transcript_id"].astype(str).map(lambda g: pd.notna(base.get(g, np.nan)))]
    abund = sm["transcript_id"].astype(str).map(lambda g: float(base[g])).to_numpy()
    nsite = sm["n_canonical_sites"].to_numpy(dtype=float)
    x = abund * nsite
    x = x / x.sum()  # relative abundance; M is then guide-per-total-site (M_REF interpretable)
    k = sm["best_site_type"].map(K_BY_SITE).to_numpy(dtype=float)
    return x, k


def process(c: dict, canonical: set[str]) -> dict:
    row = {
        "contrast": c["name"],
        "series": c["series"],
        "organism": c["organism"],
        "guide": c.get("guide", ""),
        "chemistry": c["modification"]["chemistry"],
        "seed_position": c["modification"]["seed_position"],
        "dose": c["dose"]["value"],
    }
    try:
        df_long = load_contrast(c, REPO, canonical)
    except (FileNotFoundError, ValueError) as e:
        row["status"] = "BLOCKED"
        row["reason"] = str(e).splitlines()[0]
        return row
    meas = measured_attenuation(df_long)
    x, k = build_operator_inputs(c, df_long)
    if len(x) < 2:
        row["status"] = "BLOCKED"
        row["reason"] = "fewer than 2 annotated seed-matched transcripts with abundance"
        return row
    pred = operator_prediction(x, k)
    row["status"] = "COMPUTED"
    row["n_seed_matched_modeled"] = int(len(x))
    row["measured"] = meas
    row["predicted"] = pred
    # SIGN match: operator predicts reduction (change<0) AND the data show reduction (D_m<D_p)
    row["predicted_reduction"] = bool(pred["predicted_change"] < 0)
    row["sign_match"] = bool(
        row["predicted_reduction"] == meas["measured_reduction"] and meas["measured_reduction"]
    )
    return row


def main() -> int:
    cfg = load_config(CONFIG)
    canonical = set(cfg.get("canonical_site_types", list(K_BY_SITE)))
    rows = [process(c, canonical) for c in cfg["contrasts"]]

    computed = [r for r in rows if r["status"] == "COMPUTED"]
    blocked = [r for r in rows if r["status"] == "BLOCKED"]
    n_match = sum(1 for r in computed if r["sign_match"])
    frac = n_match / len(computed) if computed else float("nan")

    # rank comparison: predicted |change| vs measured attenuation across COMPUTED contrasts
    if len(computed) >= 2:
        pv = [abs(r["predicted"]["predicted_change"]) for r in computed]
        mv = [r["measured"]["attenuation_fraction"] for r in computed]
        rho = float(stats.spearmanr(pv, mv).statistic)
        rank_note = f"Spearman rho(|predicted change|, measured attenuation) = {rho:.3f} over {len(computed)} contrasts"
    else:
        rho = None
        rank_note = (
            f"RANK UNDEFINED: only {len(computed)} contrast has data on disk; a rank "
            "correlation needs >= 2. The other contrasts are BLOCKED awaiting preprocessing."
        )

    report = {
        "operator": "single-pool conservation-law equilibrium; bisection solver; certified sign dO/d(seedK)<0 (M-matrix D>=1)",
        "site_kd_source": K_SITE_SOURCE,
        "modification_fold_source": R_SOURCE,
        "honesty": (
            "An independent per-target seed scorer re-run with the raised seed K ALSO "
            "predicts reduction (raising K lowers occupancy monotonically). This panel "
            "validates the CERTIFIED DIRECTION on real data; it does NOT establish "
            "superiority of the operator over independent-scoring baselines."
        ),
        "rows": rows,
        "n_contrasts_total": len(rows),
        "n_computed": len(computed),
        "n_blocked": len(blocked),
        "n_sign_match": n_match,
        "sign_match_fraction_over_computed": frac,
        "rank": rank_note,
        "spearman_rho": rho,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True))

    # panel table
    print("Conservation-law operator certified-direction panel (all config contrasts)")
    print("=" * 108)
    hdr = (
        "series",
        "guide/chem",
        "seedpos",
        "dose",
        "meas.atten",
        "pred.dOcc",
        "pred.frac",
        "signmatch",
    )
    print(
        f"{hdr[0]:10s} {hdr[1]:22s} {hdr[2]:>7s} {hdr[3]:>8s} {hdr[4]:>11s} {hdr[5]:>11s} {hdr[6]:>10s} {hdr[7]:>10s}"
    )
    print("-" * 108)
    for r in rows:
        chem = f"{r['guide'][:12]}/{r['chemistry']}"[:22]
        if r["status"] == "BLOCKED":
            print(
                f"{r['series']:10s} {chem:22s} {r['seed_position']!s:>7s} {r['dose']!s:>8s} "
                f"{'BLOCKED':>11s} {'BLOCKED':>11s} {'-':>10s} {'-':>10s}"
            )
        else:
            m = r["measured"]["attenuation_fraction"]
            pc = r["predicted"]["predicted_change"]
            pf = r["predicted"]["predicted_change_fraction"]
            print(
                f"{r['series']:10s} {chem:22s} {r['seed_position']!s:>7s} {r['dose']!s:>8s} "
                f"{m:>11.4f} {pc:>+11.4e} {pf:>+10.4f} {r['sign_match']!s:>10s}"
            )
    print("-" * 108)
    for r in blocked:
        print(f"  BLOCKED {r['series']}: {r['reason']}")
    print()
    for r in computed:
        mp, mm = r["measured"]["parent"], r["measured"]["modified"]
        print(
            f"[{r['series']}] measured ECDF: parent KS_D={mp['ks_D']:.4f} p={mp['ks_p']:.2e} -> "
            f"modified KS_D={mm['ks_D']:.4f} p={mm['ks_p']:.2e}  (attenuation={r['measured']['attenuation_fraction']:.3f})"
        )
        p = r["predicted"]
        print(
            f"           operator: O_parent={p['occupancy_parent']:.4f} -> O_modified={p['occupancy_modified']:.4f} "
            f"(dO={p['predicted_change']:+.4e}, {100*p['predicted_change_fraction']:+.1f}%); "
            f"certified grad sign={p['certified_gradient_sign']:+.0f}; sign<0 over all M,r={p['sign_negative_over_all_M_and_r']}"
        )
    print()
    print(
        f"Sign matches in {n_match}/{len(computed)} computed contrasts "
        f"(fraction={frac:.3f}); {len(blocked)} BLOCKED awaiting preprocessing (loader raises)."
    )
    print(f"Rank: {rank_note}")
    print(
        "\nHONESTY: an independent seed scorer re-run with the raised seed K also predicts reduction"
    )
    print("(raising K lowers occupancy monotonically), so this validates the CERTIFIED DIRECTION,")
    print("not superiority over baselines.")
    print(f"\nwrote {OUT_JSON.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
