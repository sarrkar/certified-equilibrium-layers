#!/usr/bin/env python3
"""Theorem 3: KKT-projected mass-action / thermodynamic constraint, on REAL data.

Ties the predicted off-target knockdown k_g to the REAL ViennaRNA seed-duplex
free energy dG_g via a mass-action (Langmuir / Hill) occupancy law, enforced by a
differentiable KKT orthogonal-projection layer (cascade.models.kkt_projection)
applied at train and test.

- Mass-action line: logit(knockdown) = a + b*dG, with (a, b) calibrated from the
  real (dG, efficacy) training data (b is the empirical Boltzmann slope).
- Constrained model (KKT-hPINN, Chen et al. 2024): the predicted-knockdown logits
  are projected so their trend on [1, dG] equals (a, b) exactly, while the
  feature-driven component orthogonal to dG is preserved (so accuracy is kept).
- Exact-occupancy variant (KKT-Hardnet Newton, Iftakher et al. 2025): a Newton
  projection pinning sigmoid(logit) = sigmoid(a + b*dG) exactly (the pure
  occupancy curve); this discards the feature residual and is reported as the
  accuracy cost of an exact pin.

It verifies on real held-out data that the constraint residual is at machine
precision (max < 1e-8) and that there is no accuracy regression versus the
unconstrained model.

HARD DATA POLICY (CLAUDE.md): REAL DATA ONLY. The efficacy data comes from a real
loader that RAISES if absent, and the dG values come from ViennaRNA (RAISES if not
installed). The section is then recorded as pending with the exact source.

Output results/aaai/t3_constraint.json and results/aaai/t3_constraint.png.
Accepts --seed and logs it. No LaTeX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cascade.features import featurize_pairs  # noqa: E402
from cascade.features.seedmatch import revcomp  # noqa: E402

OUT = REPO / "results" / "aaai"
RESIDUAL_TOL = 1e-8
ACC_TOL = 0.02  # constrained Pearson must be within this of unconstrained


def _target_for(antisense: str, sense: str) -> str:
    sense = (sense or "").strip()
    return sense if sense else revcomp(antisense)


def _featurize(frame, mu=None, sd=None):
    anti = frame["antisense"].tolist()
    tgt = [_target_for(a, s) for a, s in zip(frame["antisense"], frame["sense"], strict=True)]
    x = featurize_pairs(list(zip(anti, tgt, strict=True))).matrix
    if mu is None:
        mu, sd = x.mean(0), x.std(0) + 1e-8
    return (x - mu) / sd, frame["efficacy"].to_numpy(dtype=float), mu, sd


def _seed_dgs(frame) -> np.ndarray:
    from cascade.features.thermo import seed_duplex_dg

    anti = frame["antisense"].tolist()
    tgt = [_target_for(a, s) for a, s in zip(frame["antisense"], frame["sense"], strict=True)]
    return np.array([seed_duplex_dg(a, t) for a, t in zip(anti, tgt, strict=True)], dtype=float)


def real_inputs(raw_dir: Path, cache_root: Path):
    """Real Huesken train/test split, features, efficacy, and ViennaRNA seed dG."""
    from cascade.data import load_huesken_efficacy
    from cascade.features.thermo import require_vienna

    require_vienna()  # raises ImportError (with the install command) if ViennaRNA is absent
    ds = load_huesken_efficacy(raw_dir, cache_root=cache_root)  # raises if absent
    train, test = ds.train().frame, ds.test().frame
    xtr, ytr, mu, sd = _featurize(train)
    xte, yte, _, _ = _featurize(test, mu, sd)
    dg_tr, dg_te = _seed_dgs(train), _seed_dgs(test)
    return xtr, ytr, dg_tr, xte, yte, dg_te


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    from scipy.stats import pearsonr, spearmanr

    d = pred - y
    mse, mae = float(np.mean(d**2)), float(np.mean(np.abs(d)))
    if len(y) < 2 or np.std(pred) == 0 or np.std(y) == 0:
        return {"pearson": None, "spearman": None, "mse": mse, "mae": mae}
    return {
        "pearson": float(pearsonr(pred, y)[0]),
        "spearman": float(spearmanr(pred, y)[0]),
        "mse": mse,
        "mae": mae,
    }


def run_t3(raw_dir: Path, cache_root: Path, seed: int, epochs: int, lr: float) -> dict:
    import torch
    from torch import nn

    from cascade.models.kkt_projection import (
        MassActionProjection,
        fit_mass_action_line,
        mass_action_residual,
        newton_hard_projection,
    )

    xtr, ytr, dg_tr, xte, yte, dg_te = real_inputs(raw_dir, cache_root)
    xtr_t = torch.tensor(xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    dgtr_t = torch.tensor(dg_tr, dtype=torch.float32)
    xte_t = torch.tensor(xte, dtype=torch.float32)
    dgte_d = torch.tensor(dg_te, dtype=torch.float64)

    a, b = fit_mass_action_line(dgtr_t, ytr_t)
    proj = MassActionProjection(a, b)

    class MLP(nn.Module):
        def __init__(self, d: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def train_model(constrained: bool):
        torch.manual_seed(seed)
        model = MLP(xtr.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            s = model(xtr_t)
            if constrained:
                s = proj(s, dgtr_t)
            loss = ((torch.sigmoid(s) - ytr_t) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        return model

    unconstrained = train_model(constrained=False)
    constrained = train_model(constrained=True)

    with torch.no_grad():
        # Unconstrained held-out predictions and their (nonzero) constraint residual.
        s_unc = unconstrained(xte_t).double()
        k_unc = torch.sigmoid(s_unc).numpy()
        residual_unc = mass_action_residual(s_unc, dgte_d, a, b)

        # Constrained held-out predictions: the deployed output IS the double KKT
        # projection, so its residual is at machine precision.
        s_con = proj(constrained(xte_t).double(), dgte_d)
        k_con = torch.sigmoid(s_con).numpy()
        residual_con = proj.residual(s_con, dgte_d)

        # Exact-occupancy (Newton) variant: pin sigmoid(s) = sigmoid(a + b dG).
        # Start from a non-saturated point so the sigmoid Jacobian is well conditioned.
        target_occ = torch.sigmoid(a + b * dgte_d)
        s_newton = newton_hard_projection(
            s_unc.clamp(-12.0, 12.0), lambda u: torch.sigmoid(u) - target_occ
        )
        k_newton = torch.sigmoid(s_newton).numpy()
        residual_newton = float((torch.sigmoid(s_newton) - target_occ).abs().max())

    return {
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "mass_action_line": {
            "a": a,
            "b": b,
            "note": "logit(knockdown) = a + b*dG; b is the Boltzmann slope (expected < 0)",
        },
        "constraint_residual": {
            "constrained_max": residual_con,
            "unconstrained_max": residual_unc,
            "exact_occupancy_newton_max": residual_newton,
            "tolerance": RESIDUAL_TOL,
        },
        "accuracy": {
            "unconstrained": _metrics(k_unc, yte),
            "constrained_kkt": _metrics(k_con, yte),
            "exact_occupancy_newton": _metrics(k_newton, yte),
        },
    }


def acceptance(result: dict) -> dict:
    res = result["constraint_residual"]["constrained_max"]
    residual_ok = res < RESIDUAL_TOL
    unc = result["accuracy"]["unconstrained"]["pearson"]
    con = result["accuracy"]["constrained_kkt"]["pearson"]
    if unc is None or con is None:
        no_regression, accuracy_cost = None, None
    else:
        no_regression = con >= unc - ACC_TOL
        accuracy_cost = con < unc - ACC_TOL
    flag = "PASS" if (residual_ok and no_regression is not False) else "FAIL"
    return {
        "flag": flag,
        "constraint_residual_below_tol": bool(residual_ok),
        "constrained_residual_max": res,
        "no_accuracy_regression": (None if no_regression is None else bool(no_regression)),
        "constraint_costs_accuracy": (None if accuracy_cost is None else bool(accuracy_cost)),
        "unconstrained_pearson": unc,
        "constrained_pearson": con,
        "exact_occupancy_pearson": result["accuracy"]["exact_occupancy_newton"]["pearson"],
        "note": "The KKT-hPINN projection enforces the mass-action trend exactly (residual at "
        "machine precision) while preserving the feature residual orthogonal to dG, so accuracy "
        "is kept. The exact-occupancy Newton pin discards the feature residual and is reported as "
        "the accuracy cost of an exact nonlinear pin.",
    }


def _plot(result: dict, acc: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[t3] matplotlib not installed; skipped figure")
        return

    res = result["constraint_residual"]
    acc_b = result["accuracy"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    labels = ["unconstrained", "constrained_kkt", "exact_newton"]
    residuals = [
        max(res["unconstrained_max"], 1e-18),
        max(res["constrained_max"], 1e-18),
        max(res["exact_occupancy_newton_max"], 1e-18),
    ]
    ax1.bar(range(3), residuals, color=["tab:red", "tab:green", "tab:orange"])
    ax1.axhline(RESIDUAL_TOL, color="k", ls="--", label=f"tol {RESIDUAL_TOL:g}")
    ax1.set_yscale("log")
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels, rotation=15, fontsize=8)
    ax1.set_ylabel("max constraint residual (log)")
    ax1.set_title("Mass-action constraint residual")
    ax1.legend(fontsize=8)

    pear = [
        acc_b["unconstrained"]["pearson"] or 0.0,
        acc_b["constrained_kkt"]["pearson"] or 0.0,
        acc_b["exact_occupancy_newton"]["pearson"] or 0.0,
    ]
    ax2.bar(range(3), pear, color=["tab:red", "tab:green", "tab:orange"])
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels, rotation=15, fontsize=8)
    ax2.set_ylabel("held-out Pearson")
    ax2.set_title(f"Accuracy (KKT keeps it, exact pin costs it); acceptance {acc['flag']}")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[t3] wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (logged)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--raw-dir", type=str, default=str(REPO / "data" / "raw" / "efficacy"))
    parser.add_argument("--cache", type=Path, default=OUT / "cache")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(f"[t3] seed={args.seed} epochs={args.epochs} raw_dir={args.raw_dir}")

    try:
        result = run_t3(Path(args.raw_dir), args.cache, args.seed, args.epochs, args.lr)
        acc = acceptance(result)
        status = "ok"
    except (FileNotFoundError, NotImplementedError) as exc:
        result = {"status": "pending_real_data", "source": str(exc)}
        acc = {"flag": "PENDING", "reason": "real efficacy data unavailable"}
        status = "pending_real_data"
        print(f"[t3] PENDING real data: {exc}")
    except ImportError as exc:
        result = {"status": "pending_dependency", "source": str(exc)}
        acc = {"flag": "PENDING", "reason": "ViennaRNA or torch unavailable"}
        status = "pending_dependency"
        print(f"[t3] PENDING dependency: {exc}")

    numbers = {
        "acceptance": acc,
        "seed": args.seed,
        "status": status,
        **(result if status == "ok" else {"detail": result}),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "t3_constraint.json"
    out_path.write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    if status == "ok":
        _plot(result, acc, args.out_dir / "t3_constraint.png")
        print(
            f"[t3] ACCEPTANCE {acc['flag']}: constrained residual={acc['constrained_residual_max']:.2e} "
            f"(< {RESIDUAL_TOL:g}); pearson unconstrained={acc['unconstrained_pearson']} vs "
            f"constrained={acc['constrained_pearson']} (costs accuracy={acc['constraint_costs_accuracy']})"
        )
    print(f"[t3] status={status} wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
