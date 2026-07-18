"""Figure 4: competitive Langmuir adsorption - the certified comparative static on a non-biological
instance of the operator. Two panels: As(V) loading as F is swept, and F loading as As(V) is swept.
Measured antagonism (Heliyon 2024 Tables 7/8) vs operator prediction; the certified sign direction
(raise competitor -> lower target loading, dq/dC < 0) is annotated. Loadings are shown relative to the
no-competitor value so measured capacity and operator loading are on one comparable 0-1 scale.

Reads results/adsorption_operator.json (written by adsorption_operator.py) so the figure cannot diverge
from the computed numbers. Deterministic and bitwise-identical on rerun. No em-dashes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["SOURCE_DATE_EPOCH"] = "1577836800"  # fixed -> reproducible PDF/PNG timestamps

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
SERIES = json.loads((REPO / "results/adsorption_operator.json").read_text())["series"]
ACCENT = "#0072B2"
GREY = "0.45"
OUT_PDF = REPO / "results/figures/fig4_adsorption.pdf"
OUT_PNG = REPO / "results/figures/fig4_adsorption.png"


def panel(ax, s: dict, target: str, competitor: str, name_gap: bool = False) -> None:
    x = s["competitor_mg_L"]
    # shade the measured-vs-operator band: the magnitude gap the theorem does NOT claim (not concealed)
    ax.fill_between(
        x, s["operator_ratio"], s["measured_ratio"], color="0.55", alpha=0.13, lw=0, zorder=0
    )
    ax.plot(
        x,
        s["measured_ratio"],
        color=GREY,
        ls="--",
        lw=1.0,
        marker="o",
        ms=4,
        mfc="white",
        mec=GREY,
        label="measured (Heliyon Tab.7/8)",
    )
    ax.plot(
        x,
        s["operator_ratio"],
        color=ACCENT,
        ls="-",
        lw=1.2,
        marker="s",
        ms=4,
        mfc=ACCENT,
        mec=ACCENT,
        label="operator (extended Langmuir)",
    )
    ax.annotate(
        "",
        xy=(x[-1], s["operator_ratio"][-1] - 0.02),
        xytext=(x[-1], 0.98),
        arrowprops={"arrowstyle": "->", "color": "0.25", "lw": 0.8},
    )
    if name_gap:  # name the shaded band once, so the gap reads as a disclosed limit not an error
        xm = x[len(x) // 2]
        ymid = 0.5 * (s["operator_ratio"][len(x) // 2] + s["measured_ratio"][len(x) // 2])
        ax.annotate(
            "magnitude gap (not certified)",
            xy=(xm, ymid),
            xytext=(4.0, 0.99),
            fontsize=5.2,
            color="0.45",
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": "0.6", "lw": 0.5},
        )
    cert = (
        r"certified: $\partial q_{\mathrm{"
        + target
        + r"}}/\partial C_{\mathrm{"
        + competitor
        + r"}}<0$"
    )
    ax.text(
        0.97,
        0.17,
        cert,
        transform=ax.transAxes,
        fontsize=6.5,
        ha="right",
        va="center",
        color="0.25",
    )
    ax.text(
        0.97,
        0.075,
        "sign certified; magnitude over-predicted by\nsingle-constant extended Langmuir "
        "(see text)",
        transform=ax.transAxes,
        fontsize=4.8,
        ha="right",
        va="center",
        color="0.5",
    )
    ax.set_title(f"{target} loading vs {competitor} concentration", fontsize=7.5, loc="left")
    ax.set_xlabel(f"{competitor} concentration (mg/L)", fontsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(width=0.6, labelsize=6.5)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlim(-3, 33)


def main() -> int:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.55), sharey=True)
    panel(axes[0], SERIES["As vs F"], "As(V)", "F", name_gap=True)
    panel(axes[1], SERIES["F vs As"], "F", "As(V)")
    axes[0].set_ylabel("loading / no-competitor loading", fontsize=7)
    # single figure-level legend at the bottom keeps all panel corners clear of the disclosure note
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=5.8,
        frameon=False,
        loc="lower center",
        ncol=2,
        handlelength=1.8,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(pad=0.5, rect=(0.0, 0.07, 1.0, 1.0))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, metadata={"CreationDate": None})
    fig.savefig(OUT_PNG, dpi=300, metadata={"Software": None, "Creation Time": None})
    plt.close(fig)

    print("Figure 4 written:", OUT_PDF.relative_to(REPO), "and", OUT_PNG.name)
    for name, s in SERIES.items():
        print(
            f"  {name}: competitor {s['competitor_mg_L']}  measured_ratio {s['measured_ratio']}  "
            f"operator_ratio {[round(v, 3) for v in s['operator_ratio']]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
