# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "numpy>=1.24",
# ]
# ///
"""Checkpoint-cost sweep figure.

Reads the per-budget checkpoint sweeps produced by
``publication/run_checkpoint_sweep.sh`` (Deepseek / DE / 2025,
save = load cost) and renders a 2x2 figure:

  (a) hysteresis band θ_p/θ_r vs. checkpoint cost (B=200%)
  (b) margin Δθ vs. checkpoint cost, one line per budget
  (c) CO₂ savings % vs. checkpoint cost, one line per budget
  (d) achieved overhead % vs. checkpoint cost, one line per budget

Outputs:
  publication/output/checkpoint_sweep/figures/checkpoint_cost_sweep.{svg,eps}

Run:  uv run scripts/plot_checkpoint_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fig_common import save_figure  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[1]
SWEEP_DIR = BASE_DIR / "publication" / "output" / "checkpoint_sweep"
FIGURES_DIR = SWEEP_DIR / "figures"

BUDGETS = [10, 25, 50, 100, 200]
HIGHLIGHT_BUDGET = 200
BUDGET_COLORS = {
    10: "#94a3b8",
    25: "#059669",
    50: "#d97706",
    100: "#7c3aed",
    200: "#2563eb",
}


def load_sweeps() -> dict[int, pd.DataFrame]:
    sweeps: dict[int, pd.DataFrame] = {}
    for b in BUDGETS:
        path = SWEEP_DIR / f"budget_{b}" / "checkpoint_sweep.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing sweep result: {path}")
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df.sort_values("cost_s").reset_index(drop=True)
        sweeps[b] = df
    return sweeps


def _log_x(df: pd.DataFrame) -> np.ndarray:
    """Cost positions on a log axis (cost 0 shown just left of 1 s)."""
    return np.where(df["cost_s"].to_numpy() == 0, 0.7, df["cost_s"].to_numpy())


def _cost_ticks(costs: list[float]) -> list[float]:
    return [c if c > 0 else 0.7 for c in costs]


def _style_log_axis(ax: plt.Axes, costs: list[float]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(_cost_ticks(costs))
    ax.set_xticklabels(["0" if c == 0 else f"{c:g}" for c in costs], fontsize=10)
    ax.minorticks_off()


def plot_checkpoint_sweep(sweeps: dict[int, pd.DataFrame]) -> plt.Figure:
    costs = [float(c) for c in sweeps[BUDGETS[0]]["cost_s"]]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11.5, 7.6))

    # (a) hysteresis band for the publication budget
    hi = sweeps[HIGHLIGHT_BUDGET]
    x = _log_x(hi)
    ax1.fill_between(x, hi["theta_resume_lower"], hi["theta_pause_upper"],
                     color="#2563eb", alpha=0.18, zorder=2,
                     label=r"margin $\Delta_\theta$")
    ax1.plot(x, hi["theta_pause_upper"], "o-", color="#2563eb", lw=1.8,
             markersize=5, zorder=3, label=r"$\theta_p$ (upper bound)")
    ax1.plot(x, hi["theta_resume_lower"], "s--", color="#dc2626", lw=1.8,
             markersize=5, zorder=3, label=r"$\theta_r$ (lower bound)")
    _style_log_axis(ax1, costs)
    ax1.set_xlabel("checkpoint cost per save/load (s)", fontsize=12)
    ax1.set_ylabel(r"$\theta$ (gCO$_2$eq/kWh)", fontsize=12)
    ax1.set_title(f"(a) hysteresis band, B={HIGHLIGHT_BUDGET}%", loc="left", fontsize=12)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(alpha=0.3, which="both")

    # (b) margin vs cost, per budget
    for b, df in sweeps.items():
        ax2.plot(_log_x(df), df["margin"], "o-", color=BUDGET_COLORS[b], lw=1.8,
                 markersize=5, label=f"B={b}%")
    _style_log_axis(ax2, costs)
    ax2.set_xlabel("checkpoint cost per save/load (s)", fontsize=12)
    ax2.set_ylabel(r"margin $\Delta_\theta$ (gCO$_2$eq/kWh)", fontsize=12)
    ax2.set_title("(b) optimal hysteresis margin", loc="left", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which="both")

    # (c) CO2 savings vs cost, per budget
    for b, df in sweeps.items():
        x = _log_x(df)
        ax3.plot(x, df["co2_savings_pct"], "o-", color=BUDGET_COLORS[b],
                 lw=1.8, markersize=5, label=f"B={b}%")
        base = df["co2_savings_pct"].iloc[0]
        end = df["co2_savings_pct"].iloc[-1]
        ax3.annotate(f"−{base - end:.2f} pp", xy=(x[-1], end),
                     xytext=(-6, 8 if b in (10, 200) else -14),
                     textcoords="offset points", fontsize=8.5, color=BUDGET_COLORS[b],
                     ha="right")
    _style_log_axis(ax3, costs)
    ax3.set_xlabel("checkpoint cost per save/load (s)", fontsize=12)
    ax3.set_ylabel("CO₂ savings (%)", fontsize=12)
    ax3.set_title("(c) CO₂ savings of best policy", loc="left", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3, which="both")

    # (d) achieved overhead vs cost (should hug the budget line)
    for b, df in sweeps.items():
        ax4.plot(_log_x(df), df["overhead_pct"], "o-", color=BUDGET_COLORS[b],
                 lw=1.8, markersize=5, label=f"B={b}%")
        ax4.axhline(b, ls=":", color=BUDGET_COLORS[b], lw=1.0, alpha=0.7)
    _style_log_axis(ax4, costs)
    ax4.set_xlabel("checkpoint cost per save/load (s)", fontsize=12)
    ax4.set_ylabel("overhead (%)", fontsize=12)
    ax4.set_title("(d) achieved overhead (dotted: budget limit)", loc="left", fontsize=12)
    ax4.legend(fontsize=9, ncol=2)
    ax4.grid(alpha=0.3, which="both")

    fig.tight_layout()
    return fig


def write_summary(sweeps: dict[int, pd.DataFrame]) -> Path:
    """One combined CSV across budgets for further analysis."""
    rows = []
    for b, df in sweeps.items():
        out = df.copy()
        out.insert(0, "budget_pct", b)
        out.insert(1, "model", "Deepseek")
        out.insert(2, "region", "DE")
        out.insert(3, "years", "2025")
        rows.append(out)
    combined = pd.concat(rows, ignore_index=True)
    path = SWEEP_DIR / "checkpoint_sweep_all_budgets.csv"
    combined.to_csv(path, index=False)
    return path


def main() -> None:
    sweeps = load_sweeps()
    fig = plot_checkpoint_sweep(sweeps)
    save_figure(fig, "checkpoint_cost_sweep", FIGURES_DIR)
    summary = write_summary(sweeps)
    print(f"✓ checkpoint_cost_sweep.svg/.eps → {FIGURES_DIR}")
    print(f"✓ combined CSV → {summary}")


if __name__ == "__main__":
    main()
