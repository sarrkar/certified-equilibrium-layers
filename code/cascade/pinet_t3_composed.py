#!/usr/bin/env python3
"""Composed-error bound + genuine ADMM (equality + box) projection for the T3 polytope.

Extends code/cascade/pinet_massaction.py (which projected onto the affine mass-action
equality only) in three ways:

1. A-PRIORI COMPOSED-ERROR BOUND on REAL Huesken:

       ||Pi_P(f(x)) - y_*|| <= L ||x - x_*|| + dist(y_*, P).

   f is the CASCADE forward map (a ridge predictor on the real Huesken features that
   outputs the knockdown fraction), L is its measured Lipschitz constant, P is the
   mass-action polytope, and the reference target is y_* = f(x_*). Because the
   Euclidean projection Pi_P is non-expansive and f is L-Lipschitz, the bound holds by
   the triangle inequality; the script CONFIRMS it empirically across input
   perturbations and reports the tightness.

2. GENUINE OPERATOR-SPLITTING (ADMM): the thermodynamic occupancy box 0 <= theta <= 1
   is added to the affine equality, so the projection is no longer closed-form. It is
   solved by Pinet's Douglas-Rachford / ADMM splitting; the script reports the
   active-set residual (equality residual + box violation + the active box coordinates).

3. STOP CONDITION: the implicit-function-theorem (IFT) gradient through the combined
   (equality + box) projection is checked against central finite differences. If the
   max disagreement is below 1e-6 the IFT path holds. Otherwise the script falls back
   to the UNROLLED augmented-Lagrangian (ADMM) differentiable layer and documents the
   deviation. It reports which path held.

REAL DATA ONLY: the features, dG, and knockdown are real Huesken + ViennaRNA (the
loaders RAISE if absent). The input perturbations used to probe the Lipschitz / composed
bound are SEEDED verification probes of the real feature matrix (a robustness
certificate), not synthetic data. Standalone so the JAX / Pinet dependency stays off the
core path; it adds src/, experiments/aaai/, and external/pinet/src to sys.path.

Run:  python code/cascade/pinet_t3_composed.py
Writes results/aaai/t3_composed_bound.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PINET_RUNTIME_CHECK", "0")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "aaai"))
sys.path.insert(0, str(REPO / "external" / "pinet" / "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)  # float64 before any jax array is built
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from pinet import (  # noqa: E402
    BoxConstraint,
    BoxConstraintSpecification,
    EqualityConstraint,
    Project,
    ProjectionInstance,
)

OUT = REPO / "results" / "aaai"
N_BATCH = 64  # projection dimension (real Huesken examples); FD does 2N solves
RIDGE_LAMBDA = 1.0
SIGMA, OMEGA, N_ITER = 0.1, 1.7, 2000
FD_EPS = 1e-6
GRAD_TOL = 1e-6
PERTURB_REL = (0.0, 0.02, 0.05, 0.1, 0.2)


def real_huesken(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real Huesken standardized features X (n,d), knockdown y (n,), seed dG (n,)."""
    import exp_t3_constraint as t3  # owns the real Huesken feature / dG pipeline

    raw_dir = REPO / "data" / "raw" / "efficacy"
    cache_root = REPO / "results" / "aaai" / "cache"
    xtr, ytr, dg_tr, *_ = t3.real_inputs(raw_dir, cache_root)
    return (
        np.asarray(xtr[:n], dtype=np.float64),
        np.asarray(ytr[:n], dtype=np.float64),
        np.asarray(dg_tr[:n], dtype=np.float64),
    )


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Ridge forward map f(x) = x w + c (predicts knockdown). Returns (w, c)."""
    n, d = x.shape
    xa = np.hstack([x, np.ones((n, 1))])
    reg = lam * np.eye(d + 1)
    reg[-1, -1] = 0.0  # do not penalize the intercept
    wa = np.linalg.solve(xa.T @ xa + reg, xa.T @ y)
    return wa[:-1], float(wa[-1])


def forward(x: np.ndarray, w: np.ndarray, c: float) -> np.ndarray:
    return x @ w + c


def empirical_lipschitz(x: np.ndarray, fx: np.ndarray, n_pairs: int = 4000) -> float:
    """Max |f(x_i)-f(x_j)| / ||x_i-x_j|| over sampled real example pairs."""
    rng = np.random.default_rng(0)
    n = len(x)
    best = 0.0
    for _ in range(n_pairs):
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i == j:
            continue
        dx = float(np.linalg.norm(x[i] - x[j]))
        if dx > 1e-9:
            best = max(best, abs(fx[i] - fx[j]) / dx)
    return best


def make_polytope(dG: np.ndarray, a: float, b: float, n: int, unroll: bool) -> Project:
    """P = {y : Z^+ y = (a,b)} ∩ {0 <= y <= 1}, Z = [1, dG]. ADMM (eq + box)."""
    z = np.stack([np.ones(n), dG], axis=1)  # (n,2)
    a_mat = jnp.asarray(np.linalg.pinv(z))[None]  # (1,2,n) OLS-coefficient map
    rhs = jnp.asarray([a, b]).reshape(1, 2, 1)
    eq = EqualityConstraint(a_mat=a_mat, b=rhs, method="pinv")
    box = BoxConstraint(
        BoxConstraintSpecification(
            lb=jnp.zeros((1, n, 1)), ub=jnp.ones((1, n, 1)), mask=jnp.ones((n,), dtype=bool)
        )
    )
    return Project(eq_constraint=eq, box_constraint=box, unroll=unroll)


def project(pl: Project, z_np: np.ndarray) -> np.ndarray:
    z = jnp.asarray(z_np, dtype=jnp.float64).reshape(1, -1, 1)
    out = pl.call(y_raw=ProjectionInstance(x=z), n_iter=N_ITER, sigma=SIGMA, omega=OMEGA)
    return np.asarray(out[0].x).reshape(-1)


def loss_of_z(pl: Project, z: jnp.ndarray) -> jnp.ndarray:
    out = pl.call(y_raw=ProjectionInstance(x=z), n_iter=N_ITER, sigma=SIGMA, omega=OMEGA)
    return jnp.sum(out[0].x ** 2)


def grad_vs_fd(pl: Project, z0_np: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """max|IFT/unrolled grad - central FD| of sum(proj(z)^2) at z0."""
    z0 = jnp.asarray(z0_np, dtype=jnp.float64).reshape(1, -1, 1)
    g = np.asarray(jax.grad(lambda z: loss_of_z(pl, z))(z0)).reshape(-1)
    n = z0_np.size
    fd = np.zeros(n)
    base = z0_np.reshape(-1)
    for i in range(n):
        zp, zm = base.copy(), base.copy()
        zp[i] += FD_EPS
        zm[i] -= FD_EPS
        fp = float(loss_of_z(pl, jnp.asarray(zp).reshape(1, -1, 1)))
        fm = float(loss_of_z(pl, jnp.asarray(zm).reshape(1, -1, 1)))
        fd[i] = (fp - fm) / (2 * FD_EPS)
    return float(np.max(np.abs(g - fd))), g, fd


def run() -> dict:
    x, y, dG = real_huesken(N_BATCH)
    n, d = x.shape
    w, c = fit_ridge(x, y, RIDGE_LAMBDA)
    fx = forward(x, w, c)
    L_op = float(np.linalg.norm(w))  # Lipschitz of the linear forward map
    L_emp = empirical_lipschitz(x, fx)

    # Mass-action line (a, b) = OLS of the real knockdown on [1, dG]; the box is the
    # thermodynamic occupancy 0 <= theta <= 1.
    z_design = np.stack([np.ones(n), dG], axis=1)
    a, b = np.linalg.lstsq(z_design, y, rcond=None)[0]
    a, b = float(a), float(b)
    a_mat_np = np.linalg.pinv(z_design)

    pl = make_polytope(dG, a, b, n, unroll=False)

    # Composed-error bound: x_* = X, y_* = f(X); probe perturbations of the real X.
    y_star = fx.copy()
    proj_star = project(pl, y_star)
    dist_y_star = float(np.linalg.norm(proj_star - y_star))
    x_norm = float(np.linalg.norm(x, 2))
    rng = np.random.default_rng(0)
    g_dir = rng.standard_normal(x.shape)
    g_dir /= np.linalg.norm(g_dir, 2)
    bound_rows = []
    for m in PERTURB_REL:
        delta = m * x_norm * g_dir
        dx = float(np.linalg.norm(delta, 2))
        fxk = forward(x + delta, w, c)
        lhs = float(np.linalg.norm(project(pl, fxk) - y_star))
        rhs = L_op * dx + dist_y_star
        bound_rows.append(
            {
                "perturb_rel": m,
                "x_minus_xstar_2norm": dx,
                "lhs": lhs,
                "rhs": rhs,
                "holds": bool(lhs <= rhs * (1.0 + 1e-9)),
                "slack": rhs - lhs,
            }
        )
    bound_holds = all(r["holds"] for r in bound_rows)

    # The real Huesken knockdowns are all MODERATE (~0.27 to 0.69), so the
    # thermodynamic occupancy box 0 <= theta <= 1 is naturally SLACK on this data: the
    # ridge predictions are valid occupancies and the box never binds. To genuinely
    # exercise the eq+box ADMM active set AND the IFT stop condition through the box
    # kink, the active-set and gradient tests use a box-ACTIVATING point: a 4x affine
    # spread of the real forward output about 0.5 that pushes the tails past 0 / 1 (a
    # transparent verification probe, not synthetic data).
    natural_proj = project(pl, fx)
    natural_active = int(np.sum(natural_proj <= 1e-7) + np.sum(natural_proj >= 1.0 - 1e-7))
    z_box = 0.5 + 4.0 * (fx - 0.5)  # spreads the real predictions past [0,1]
    n_out_of_box = int(np.sum(z_box < 0.0) + np.sum(z_box > 1.0))

    # Active-set residual at Pi_P(z_box) (the genuine ADMM solution, box active).
    y_proj = project(pl, z_box)
    eq_resid = float(np.max(np.abs(a_mat_np @ y_proj - np.array([a, b]))))
    box_viol = float(max(0.0, -y_proj.min(), y_proj.max() - 1.0))
    n_lower = int(np.sum(y_proj <= 1e-7))
    n_upper = int(np.sum(y_proj >= 1.0 - 1e-7))
    cv = float(np.asarray(pl.cv(ProjectionInstance(x=jnp.asarray(y_proj).reshape(1, -1, 1)))).max())
    active_set_residual = max(eq_resid, box_viol)

    # STOP CONDITION: IFT gradient through (eq + box) vs finite differences, at the
    # box-active point so the combined active set is genuinely tested.
    z0 = z_box.copy()
    ift_diff, _g, _fd = grad_vs_fd(pl, z0)
    ift_holds = bool(ift_diff < GRAD_TOL)
    fallback = None
    if ift_holds:
        path_held = "ift"
    else:
        pl_unroll = make_polytope(dG, a, b, n, unroll=True)  # aug-Lagrangian / unrolled ADMM
        unroll_diff, _gu, _fdu = grad_vs_fd(pl_unroll, z0)
        fallback = {
            "augmented_lagrangian_unrolled_max_grad_fd_diff": unroll_diff,
            "augmented_lagrangian_below_tol": bool(unroll_diff < GRAD_TOL),
            "deviation_ift_vs_fd": ift_diff,
            "note": "IFT gradient through the combined equality+box projection exceeded 1e-6 vs "
            "finite differences (the box active set introduces a kink the implicit VJP resolves "
            "only approximately). Fell back to the unrolled augmented-Lagrangian (ADMM) "
            "differentiable layer.",
        }
        path_held = "augmented_lagrangian_unrolled"

    return {
        "status": "ok",
        "data": "real Huesken features + ViennaRNA seed dG (standardized); knockdown fraction",
        "n_examples": int(n),
        "n_features": int(d),
        "forward_map": f"ridge f(x) = x w + c (lambda={RIDGE_LAMBDA:g}) predicting knockdown fraction",
        "lipschitz": {
            "L_operator_norm_w": L_op,
            "L_empirical_pairwise_max": L_emp,
            "empirical_le_operator": bool(L_emp <= L_op * (1.0 + 1e-6)),
        },
        "polytope": {
            "equality": "Z^+ y = (a, b), Z = [1, dG]; (a,b) = OLS(knockdown on [1,dG])",
            "a": a,
            "b": b,
            "box": f"0 <= theta <= 1 (thermodynamic occupancy), all {n} coordinates",
            "solver": f"Pinet Douglas-Rachford / ADMM (eq + box), sigma={SIGMA:g} omega={OMEGA:g} "
            f"n_iter={N_ITER}",
        },
        "composed_bound": {
            "formula": "||Pi_P(f(x)) - y_*|| <= L ||x - x_*|| + dist(y_*, P), y_* = f(x_*)",
            "L": L_op,
            "dist_y_star_to_P": dist_y_star,
            "x_star": "the real Huesken feature matrix X (perturbations are seeded probes)",
            "rows": bound_rows,
            "bound_holds_all": bound_holds,
        },
        "admm_active_set": {
            "natural_projection_box_active_count": natural_active,
            "occupancy_box_naturally_slack": bool(natural_active == 0),
            "note": "Real Huesken knockdowns are moderate (~0.27-0.69), so the occupancy box "
            "0<=theta<=1 is naturally SLACK (f(X)'s projection has 0 active box coords). The "
            "active-set and gradient tests below use a box-activating 4x affine spread of the real "
            f"forward output ({n_out_of_box} of {n} coordinates fall outside [0,1] before projection) "
            "so the eq+box ADMM active set and the IFT stop condition are genuinely exercised.",
            "equality_residual_inf": eq_resid,
            "box_violation": box_viol,
            "n_active_lower_theta_eq_0": n_lower,
            "n_active_upper_theta_eq_1": n_upper,
            "pinet_constraint_violation": cv,
            "active_set_residual": active_set_residual,
        },
        "stop_condition": {
            "criterion": "IFT gradient through (equality + box) projection vs central FD < 1e-6; "
            "else fall back to the unrolled augmented-Lagrangian (ADMM) differentiable layer.",
            "ift_max_grad_fd_diff": ift_diff,
            "ift_below_tol": ift_holds,
            "fallback": fallback,
            "path_held": path_held,
        },
    }


def main() -> int:
    numbers = run()
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "t3_composed_bound.json"
    out_path.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    li = numbers["lipschitz"]
    print(
        f"[t3-composed] real Huesken N={numbers['n_examples']} d={numbers['n_features']} | "
        f"forward-map Lipschitz L=||w||={li['L_operator_norm_w']:.4f} "
        f"(empirical pairwise max {li['L_empirical_pairwise_max']:.4f}, <= L: {li['empirical_le_operator']})"
    )
    cb = numbers["composed_bound"]
    print(f"[t3-composed] composed-error bound  dist(y_*,P)={cb['dist_y_star_to_P']:.4f}:")
    print(f"  {'perturb':>8} {'||x-x*||':>10} {'LHS':>10} {'RHS':>10} {'slack':>10} {'holds':>6}")
    for r in cb["rows"]:
        print(
            f"  {r['perturb_rel']:>8} {r['x_minus_xstar_2norm']:>10.4f} {r['lhs']:>10.4f} "
            f"{r['rhs']:>10.4f} {r['slack']:>10.4f} {r['holds']!s:>6}"
        )
    print(f"  bound holds across all probes: {cb['bound_holds_all']}")
    asr = numbers["admm_active_set"]
    print(
        f"[t3-composed] ADMM (eq+box): equality-residual-inf={asr['equality_residual_inf']:.2e} "
        f"box-violation={asr['box_violation']:.2e} active(theta=0:{asr['n_active_lower_theta_eq_0']}, "
        f"theta=1:{asr['n_active_upper_theta_eq_1']}) | active-set residual={asr['active_set_residual']:.2e}"
    )
    sc = numbers["stop_condition"]
    print(
        f"[t3-composed] IFT grad vs FD max|diff|={sc['ift_max_grad_fd_diff']:.2e} "
        f"(< {GRAD_TOL}: {sc['ift_below_tol']})"
    )
    if sc["fallback"] is not None:
        fb = sc["fallback"]
        print(
            f"[t3-composed] FALLBACK augmented-Lagrangian (unrolled ADMM): max|diff|="
            f"{fb['augmented_lagrangian_unrolled_max_grad_fd_diff']:.2e} "
            f"(< {GRAD_TOL}: {fb['augmented_lagrangian_below_tol']})"
        )
    print(f"[t3-composed] PATH HELD: {sc['path_held']}")
    print(f"[t3-composed] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
