# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "numpy>=1.24",
#   "scipy>=1.11",
# ]
# ///
"""Paper Figure 3: hysteresis-margin analysis.

Generates ``margin_analysis.{svg,eps}`` (panels a/b).

Figure in paper:  TheGreenEpochPaper/main.tex  \\label{fig:margin}
Outputs:
  publication/output/figures_margin/margin_analysis.svg
  ../TheGreenEpochPaper/assets/margin_analysis.{svg,eps}

Data: DeepSeek V3 only (``analyze_margins.INCLUDED_MODELS``), alpha == 1.0,
across the overhead-budget sweep folders ``publication/output/budget_*/results``
plus the published ``publication/output/results`` (B=200).

Run:  uv run scripts/fig_3_margin_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_margins as am  # noqa: E402
from fig_common import save_figure  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = BASE_DIR / "publication/output/figures_margin"

# ``analyze_margins`` resolves its budget folders relative to its own
# module-level BASE_DIR; point it at this checkout so the script works from
# any working directory.
am.BASE_DIR = BASE_DIR / "publication/output"
am.DEFAULT_RESULTS_DIR = am.BASE_DIR / "results"


def plot_margin_analysis(cost, attain) -> plt.Figure:
    """Two-panel figure: (a) savings attainable per margin bin,
    (b) distribution of the optimal margin over all runs."""
    prof = attain.groupby("margin_bin", observed=True)["relative"].agg(
        median="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75),
        n="size")
    prof = prof.reindex(am.BIN_LABELS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    colors = ["#059669" if i <= 2 else "#94a3b8" for i in range(len(am.BIN_LABELS))]

    # (a) attainable savings per margin bin
    x = np.arange(len(am.BIN_LABELS))
    ax1.bar(x, prof["median"], color=colors, edgecolor="black", linewidth=0.7, width=0.62,
            zorder=3)
    ax1.errorbar(x, prof["median"],
                 yerr=[prof["median"] - prof["q25"], prof["q75"] - prof["median"]],
                 fmt="none", ecolor="black", elinewidth=0.8, capsize=3, zorder=4)
    for xi, (med, n) in enumerate(zip(prof["median"], prof["n"])):
        ax1.text(xi, med + 4, f"{med:.0f}", ha="center", fontsize=10, zorder=5)
        ax1.text(xi, 4, f"n={int(n)}", ha="center", fontsize=9, color="white", zorder=5,
                 rotation=90, va="bottom")
    ax1.set_xticks(x)
    ax1.set_xticklabels(am.BIN_LABELS, fontsize=11, rotation=45, ha="right",
                        rotation_mode="anchor")
    ax1.set_xlim(-0.65, len(am.BIN_LABELS) - 0.35)
    ax1.set_ylim(0, 115)
    ax1.set_xlabel(r"hysteresis margin $\Delta_\theta$ (gCO$_2$eq/kWh)", fontsize=12)
    ax1.set_ylabel("attainable savings\n(% of run optimum)", fontsize=12)
    ax1.tick_params(axis="y", labelsize=11)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_title("(a)", loc="left", fontsize=12)

    # (b) ECDF of the optimal margin
    vals = np.sort(cost["margin_best"].to_numpy())
    y = np.arange(1, len(vals) + 1) / len(vals)
    ax2.step(np.concatenate([[0], vals]), np.concatenate([[0], y]), where="post",
             color="#2563eb", linewidth=1.8, zorder=3)
    for cap, ls, col in zip(am.SMALL_MARGINS, ("--", ":"), ("#059669", "#dc2626")):
        share = (vals <= cap).mean()
        ax2.axvline(cap, ls=ls, color=col, linewidth=1.2)
        ax2.hlines(share, 0, cap, colors="0.4", linestyles=":", linewidth=0.8, zorder=2)
        ax2.plot([cap], [share], "o", color="black", markersize=4, zorder=4)
        ax2.text(cap + 1.5, share - 0.05, f"{share * 100:.1f}%", fontsize=10,
                 ha="left", va="top", zorder=5)
    ax2.set_xlim(0, 45)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel(r"optimal margin $\Delta_\theta$ (gCO$_2$/kWh)", fontsize=12)
    ax2.set_ylabel("cumulative share of runs", fontsize=12)
    ax2.tick_params(labelsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_title("(b)", loc="left", fontsize=12)

    fig.tight_layout()
    return fig


def main() -> None:
    am.setup_style()
    budget_dirs = am.discover_budget_dirs()
    combined = am.load_all_points(budget_dirs)
    cost = am.constraint_cost(combined)
    attain = am.margin_bin_profile(combined, cost)
    fig = plot_margin_analysis(cost, attain)
    save_figure(fig, "margin_analysis", FIGURES_DIR)
    print("✓ fig 3 → margin_analysis.svg/.eps")


if __name__ == "__main__":
    main()
