# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "numpy>=1.24",
# ]
# ///
"""Paper Figure 4: hysteresis margin under a checkpoint-cost sweep.

Generates ``checkpoint_margin.{svg,eps}`` (panels a/b) from the
(budget, cost) optima computed by ``analyze_checkpoint_margins.py``.

Figure in paper:  TheGreenEpochPaper/main.tex  \\label{fig:ckpt-margin}
Outputs:
  publication/output/checkpoint_sweep/figures/checkpoint_margin.{svg,eps}
  ../TheGreenEpochPaper/assets/checkpoint_margin.{svg,eps}

Data: Deepseek / DE / 2025, save = load cost,
``publication/output/checkpoint_sweep/margin_constraint_cost.csv``.

Run:  uv run scripts/fig_4_checkpoint_margin.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_checkpoint_margins as acm  # noqa: E402
from fig_common import save_figure  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[1]
SWEEP_DIR = BASE_DIR / "publication" / "output" / "checkpoint_sweep"
FIGURES_DIR = SWEEP_DIR / "figures"

MEASURED_COST_S = 148.8  # PyTorch Dist. checkpoint, 7B model


def _log_x(costs: np.ndarray) -> np.ndarray:
    """Cost positions on a log axis (cost 0 shown just left of 1 s)."""
    return np.where(costs == 0, 0.7, costs)


def _style_log_axis(ax: plt.Axes) -> None:
    ticks = [0.7, 1, 10, 100, 600]
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "1", "10", "100", "600"], fontsize=11)
    ax.minorticks_off()


def plot_checkpoint_margin(cost: pd.DataFrame) -> plt.Figure:
    """Two-panel figure: (a) optimal margin vs. checkpoint cost,
    (b) savings lost when capping the margin at 10 gCO2eq/kWh."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4))

    # (a) optimal margin vs checkpoint cost
    for b in acm.BUDGETS:
        sub = cost[cost["budget"] == b].sort_values("cost_s")
        ax1.plot(_log_x(sub["cost_s"].to_numpy()), sub["margin_best"], "o-",
                 color=acm.BUDGET_COLORS[b], lw=1.6, markersize=4.5,
                 label=f"B={b}%")
    for cap, ls, col in zip((acm.ABS_SMALL_10, acm.ABS_SMALL_20), ("--", ":"),
                            ("#059669", "#dc2626")):
        ax1.axhline(cap, ls=ls, color=col, lw=1.1, zorder=2,
                    label=rf"$\Delta_\theta={cap}$")
    ax1.axvline(MEASURED_COST_S, color="0.55", lw=0.9, ls="-.", zorder=1)
    ax1.text(MEASURED_COST_S * 0.92, 36, "measured\n148.8 s",
             fontsize=8.5, color="0.35", ha="right", va="top")
    _style_log_axis(ax1)
    ax1.set_ylim(0, 38)
    ax1.set_xlabel("checkpoint cost per save/load (s)", fontsize=12)
    ax1.set_ylabel(r"optimal margin $\Delta_\theta$ (gCO$_2$eq/kWh)", fontsize=12)
    ax1.tick_params(axis="y", labelsize=11)
    ax1.set_title("(a)", loc="left", fontsize=12)
    ax1.legend(fontsize=8, loc="upper left", handlelength=1.8,
               borderpad=0.4, labelspacing=0.35)
    ax1.grid(alpha=0.3, which="both")

    # (b) regret when capping the margin at <=10 gCO2eq/kWh
    for b in acm.BUDGETS:
        sub = cost[cost["budget"] == b].sort_values("cost_s")
        ax2.plot(_log_x(sub["cost_s"].to_numpy()), sub["regret10_pp"], "o-",
                 color=acm.BUDGET_COLORS[b], lw=1.6, markersize=4.5,
                 label=f"B={b}%")
    ax2.axhline(0, color="0.4", lw=0.8, zorder=2)
    ax2.axvline(MEASURED_COST_S, color="0.55", lw=0.9, ls="-.", zorder=1)
    worst = cost.loc[cost["regret10_pp"].idxmax()]
    ax2.annotate(f"max {worst['regret10_pp']:.2f} pp",
                 xy=(_log_x(np.array([worst["cost_s"]]))[0], worst["regret10_pp"]),
                 xytext=(-4, 8), textcoords="offset points", fontsize=9,
                 ha="right")
    _style_log_axis(ax2)
    ax2.set_ylim(-0.008, 0.145)
    ax2.set_xlabel("checkpoint cost per save/load (s)", fontsize=12)
    ax2.set_ylabel("savings lost by cap (pp)", fontsize=12)
    ax2.tick_params(axis="y", labelsize=11)
    ax2.set_title(r"(b) regret when capping $\Delta_\theta \leq 10$",
                  loc="left", fontsize=12)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    return fig


def main() -> None:
    acm.setup_style()
    path = SWEEP_DIR / "margin_constraint_cost.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path} — run `uv run scripts/analyze_checkpoint_margins.py` first")
    cost = pd.read_csv(path)
    cost["cost_s"] = cost["cost_s"].astype(float)
    fig = plot_checkpoint_margin(cost)
    save_figure(fig, "checkpoint_margin", FIGURES_DIR)
    print("✓ fig 4 → checkpoint_margin.svg/.eps")


if __name__ == "__main__":
    main()
