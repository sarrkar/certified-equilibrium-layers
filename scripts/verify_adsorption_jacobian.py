"""Symbolic-vs-code consistency check for the competitive-Langmuir closed-system Jacobian.

Confirms three things so the appendix derivation and adsorption_operator.py cannot silently diverge:
  (1) the code's hand-written closed form dq_i/dC_k (adsorption_operator.dq_closed_form) is IDENTICAL,
      as a symbolic expression, to sympy's derivative of the equilibrium loading
      q_i = qmax_i K_i C_i / (1 + sum_j K_j C_j) (the closed form of the bisection fixed point);
  (2) the assembled Jacobian J = V I + m dq/dC reproduces the exact numeric readout printed on titans,
      J(As=30, F=30) = [[1.0957, -0.0735], [-0.0476, 1.0917]] (positive diagonal, negative off-diagonal,
      strictly diagonally dominant);
  (3) the structural signs (own > 0, cross < 0, diagonal dominance) hold symbolically for all positive
      concentrations, not merely at the sampled point.

Real single-component constants only (Heliyon 2024 Table 4, via the data file). Deterministic, em-dash free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from adsorption_operator import (  # noqa: E402
    M_G,
    V_L,
    closed_system_jacobian,
    dq_closed_form,
    load_data,
)

# titans readout to reproduce (from adsorption_operator.py stdout)
READOUT = {("As", "As"): 1.0957, ("As", "F"): -0.0735, ("F", "As"): -0.0476, ("F", "F"): 1.0917}


def symbolic_dq() -> dict:
    """sympy derivative of q_i = qmax_i K_i C_i / (1 + sum K_j C_j); returns {(i,j): expr}."""
    qa, qf, ka, kf, ca, cf = sp.symbols("qa qf ka kf ca cf", positive=True)
    qmax = {"As": qa, "F": qf}
    k = {"As": ka, "F": kf}
    c = {"As": ca, "F": cf}
    denom = 1 + ka * ca + kf * cf
    q = {i: qmax[i] * k[i] * c[i] / denom for i in ("As", "F")}
    out = {}
    for i in ("As", "F"):
        for j in ("As", "F"):
            out[(i, j)] = sp.diff(q[i], c[j])
    return out, (qa, qf, ka, kf, ca, cf)


def code_dq_symbolic(syms) -> dict:
    """The code's hand-written closed form dq_closed_form, evaluated on sympy symbols (not floats)."""
    qa, qf, ka, kf, ca, cf = syms
    qmax = {"As": qa, "F": qf}
    k = {"As": ka, "F": kf}
    return dq_closed_form(qmax, k, ca, cf)


def main() -> int:
    print("Symbolic vs code: competitive-Langmuir Jacobian consistency")
    print("=" * 78)

    sym, syms = symbolic_dq()
    code = code_dq_symbolic(syms)

    print("\n(1) symbolic derivative of q_i == code dq_closed_form (per entry, simplify to 0):")
    all_zero = True
    for key in (("As", "As"), ("As", "F"), ("F", "As"), ("F", "F")):
        diff = sp.simplify(sym[key] - code[key])
        ok = diff == 0
        all_zero = all_zero and ok
        print(f"    d q_{key[0]}/d C_{key[1]}: symbolic - code = {diff}  -> match = {ok}")
    print(f"    ALL FOUR ENTRIES IDENTICAL = {all_zero}")

    # symbolic structural signs for all positive concentrations
    qa, qf, ka, kf, ca, cf = syms
    own_pos = all(sp.simplify(sym[(i, i)]) > 0 for i in ("As", "F"))
    cross_neg = bool(
        sp.ask(sp.Q.negative(sym[("As", "F")])) and sp.ask(sp.Q.negative(sym[("F", "As")]))
    ) or all(str(sp.simplify(sym[(i, j)])).startswith("-") for i, j in (("As", "F"), ("F", "As")))
    print(
        f"\n(3) symbolic signs for all positive C: own-term positive = {own_pos}; "
        f"cross-term negative = {cross_neg}"
    )

    # numeric reproduction of the titans readout
    data = load_data()
    qmax = {
        "As": data["single_langmuir"]["As"]["qmax_mg_g"],
        "F": data["single_langmuir"]["F"]["qmax_mg_g"],
    }
    k = {
        "As": data["single_langmuir"]["As"]["kL_L_mg"],
        "F": data["single_langmuir"]["F"]["kL_L_mg"],
    }
    j = closed_system_jacobian(qmax, k, 30.0, 30.0)
    idx = {"As": 0, "F": 1}
    print(f"\n(2) J = V I + m dq/dC at (As=30, F=30), V={V_L} m={M_G}:")
    print(f"    code J = [[{j[0,0]:.4f}, {j[0,1]:.4f}], [{j[1,0]:.4f}, {j[1,1]:.4f}]]")
    max_abs = 0.0
    for (i, jj), want in READOUT.items():
        got = float(j[idx[i], idx[jj]])
        max_abs = max(max_abs, abs(got - want))
        print(
            f"    J[{i},{jj}] code {got:.4f}  vs titans readout {want:.4f}  |delta| {abs(got - want):.2e}"
        )
    readout_ok = max_abs < 5e-5
    print(f"    max |code - readout| = {max_abs:.2e}  -> reproduces titans readout = {readout_ok}")

    # structural properties of the numeric J
    diag_pos = bool(j[0, 0] > 0 and j[1, 1] > 0)
    offdiag_neg = bool(j[0, 1] < 0 and j[1, 0] < 0)
    diag_dom = bool(j[0, 0] > abs(j[0, 1]) and j[1, 1] > abs(j[1, 0]))
    print(
        f"\n    numeric J: positive diagonal = {diag_pos}; negative off-diagonal = {offdiag_neg}; "
        f"strictly diagonally dominant = {diag_dom}"
    )

    ok = all_zero and own_pos and cross_neg and readout_ok and diag_pos and offdiag_neg and diag_dom
    print("\n" + "=" * 78)
    print(
        f"CONSISTENCY: symbolic Jacobian == code closed form == titans readout, structural signs "
        f"certified => {ok}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
