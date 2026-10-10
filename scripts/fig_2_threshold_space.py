# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "numpy>=1.24",
# ]
# ///
"""Paper Figure 2: pause vs resume threshold space.

Generates ``threshold_space_DeepSeek_V3_DE_0101.{svg,eps}``.

Figure in paper:  TheGreenEpochPaper/main.tex  \\label{fig:threshold}
Outputs:
  publication/output/figures/threshold_space_DeepSeek_V3_DE_0101.svg
  ../TheGreenEpochPaper/assets/threshold_space_DeepSeek_V3_DE_0101.{svg,eps}

Data: DeepSeek V3, Germany, fixed start 2025-01-01, alpha == 1.0.
Colour encodes the composite score; the dashed diagonal marks
zero-hysteresis policies (theta_r = theta_p).

Run:  uv run scripts/fig_2_threshold_space.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualize_results import (  # noqa: E402
    find_best_tradeoff,
    load_and_parse_results,
    setup_style,
)
from fig_common import save_figure  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "publication/output/results"
FIGURES_DIR = BASE_DIR / "publication/output/figures"

MODEL = "DeepSeek V3"
REGION = "DE"
START = "01-01"
YEAR = "2025"
SAFE_START = START.replace("-", "")


def plot_threshold_space(df) -> plt.Figure:
    """Scatter of theta_p (x) vs theta_r (y), coloured by score."""
    group = df[
        (df["file_type"] == "fixed")
        & (df["model"] == MODEL)
        & (df["region"] == REGION)
        & (df["start_date"] == START)
        & (df["year"].astype(str) == YEAR)
        & (df["alpha"] == 1.0)
    ]
    if len(group) < 3:
        raise SystemExit(
            f"Not enough data for {MODEL} {REGION} {START} ({len(group)} rows)."
        )

    fig, ax = plt.subplots(figsize=(8, 7))

    scatter = ax.scatter(
        group["theta_p"],
        group["theta_r"],
        c=group["score"],
        cmap="RdYlGn",
        alpha=0.6,
        s=40,
        edgecolors="none",
    )

    lim_min = min(group["theta_r"].min(), group["theta_p"].min()) * 0.9
    lim_max = max(group["theta_p"].max(), group["theta_r"].max()) * 1.05
    ax.plot(
        [lim_min, lim_max], [lim_min, lim_max],
        color="gray", linestyle="--", linewidth=1, alpha=0.5,
        label="θ_p = θ_r (no hysteresis)",
    )
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    best = find_best_tradeoff(group)
    ax.scatter(
        best["theta_p"],
        best["theta_r"],
        marker="*",
        s=300,
        color="red",
        edgecolors="black",
        linewidths=1.2,
        zorder=10,
        label=f'Best: θ_p={best["theta_p"]:.0f}, θ_r={best["theta_r"]:.0f}',
    )
    ax.annotate(
        f'score={best["score"]:.4f}\nsavings={best["co2_save_pct"]:.2f}%',
        (best["theta_p"], best["theta_r"]),
        textcoords="offset points",
        xytext=(12, -15),
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
    )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Score")

    ax.set_xlabel("θ_p (Pause Threshold) [gCO₂eq/kWh]")
    ax.set_ylabel("θ_r (Resume Threshold) [gCO₂eq/kWh]")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def main() -> None:
    setup_style()
    df = load_and_parse_results(RESULTS_DIR)
    fig = plot_threshold_space(df)
    name = f"threshold_space_{MODEL.replace(' ', '_')}_{REGION}_{SAFE_START}"
    save_figure(fig, name, FIGURES_DIR)
    print(f"✓ fig 2 → {name}.svg/.eps")


if __name__ == "__main__":
    main()
