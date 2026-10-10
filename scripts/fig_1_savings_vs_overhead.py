# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "numpy>=1.24",
# ]
# ///
"""Paper Figure 1: CO2 savings vs time overhead Pareto frontiers.

Generates ``savings_vs_overhead_all.{svg,eps}`` (one frontier per grid region).

Figure in paper:  TheGreenEpochPaper/main.tex  \\label{fig:pareto}
Outputs:
  publication/output/figures/savings_vs_overhead_all.svg
  ../TheGreenEpochPaper/assets/savings_vs_overhead_all.{svg,eps}

Data: DeepSeek V3 ``*_all_2025_*`` runs (year 2025, start date optimised),
alpha == 1.0 only. Each curve is the upper envelope (Pareto frontier) of the
feasible policies (θ_p, θ_r, t_s); stars mark the best composite score.

Run:  uv run scripts/fig_1_savings_vs_overhead.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualize_results import (  # noqa: E402
    COLORS_REGION,
    REGION_ORDER,
    compute_pareto_frontier,
    find_best_tradeoff,
    load_and_parse_results,
    setup_style,
)
from fig_common import save_figure  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "publication/output/results"
FIGURES_DIR = BASE_DIR / "publication/output/figures"

YEAR = "2025"


def plot_savings_vs_overhead(df) -> plt.Figure:
    df_all = df[
        (df["file_type"] == "all")
        & (df["year"].astype(str) == YEAR)
        & (df["alpha"] == 1.0)
    ]
    if df_all.empty:
        raise SystemExit("No _all_ rows for the requested year/alpha.")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    for region in REGION_ORDER:
        subset = df_all[df_all["region"] == region]
        if subset.empty:
            continue

        f_o, f_s = compute_pareto_frontier(
            subset["overhead_pct"].values, subset["co2_save_pct"].values
        )
        if len(f_o) < 2:
            continue

        ax.plot(
            f_o, f_s,
            linewidth=2.0,
            marker="o",
            markersize=4,
            color=COLORS_REGION.get(region, "gray"),
            label=region,
        )

        best = find_best_tradeoff(subset)
        ax.scatter(
            best["overhead_pct"],
            best["co2_save_pct"],
            marker="*",
            s=170,
            color="red",
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
        )

    ax.set_xlabel("Time overhead (%)", fontsize=16)
    ax.set_ylabel("CO₂ savings (%)", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=13, loc="lower right", frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    setup_style()
    df = load_and_parse_results(RESULTS_DIR)
    fig = plot_savings_vs_overhead(df)
    save_figure(fig, "savings_vs_overhead_all", FIGURES_DIR)
    print("✓ fig 1 → savings_vs_overhead_all.svg/.eps")


if __name__ == "__main__":
    main()
