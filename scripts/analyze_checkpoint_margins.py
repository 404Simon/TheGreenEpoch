# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "numpy>=1.24",
# ]
# ///
"""Margin-Analyse für den Checkpoint-Cost-Sweep.

Analog zu analyze_margins.py, aber auf den Sweep-Ergebnissen
(publication/output/checkpoint_sweep/budget_*/checkpoint_points.csv,
Deepseek / DE / 2025, save = load).

Pro (Budget B, Checkpoint-Cost c):
  - Beste Konfiguration = höchste Score (within budget, completed, CO₂>0;
    Tie-Break auf CO₂-Savings, da Score auf 4 Decimals gerundet)
  - Beste Konfiguration unter Margin-Cap ≤10 / ≤20 / =0
  - Regret = verlorene CO₂-Savings (Prozentpunkte und relativ) durch den Cap

Fragen:
  - Wie häufig sind die *optimalen* Results kleine Margins (≤10, ≤20) oder
    Margin genau 0?
  - Wie viel % CO₂-Savings verliert man, wenn man diese Margins erzwingt?

Erzeugt:
  publication/output/checkpoint_sweep/margin_optima_rates.csv
  publication/output/checkpoint_sweep/margin_constraint_cost.csv
  publication/output/checkpoint_sweep/figures/margin_analysis_ckptcost.{svg,eps}

Hinweis: Die Diagnose-Figur wird bewusst *nicht* über ``fig_common.save_figure``
geschrieben -- das würde sie auch nach ``../TheGreenEpochPaper/assets``
kopieren, wo ``margin_analysis`` bereits die Paper-Figur 3 ist
(``fig_3_margin_analysis.py``). Die Paper-Figur zum Checkpoint-Cost-Sweep
erzeugt ``fig_4_checkpoint_margin.py`` (``checkpoint_margin``).

Usage:
  uv run scripts/analyze_checkpoint_margins.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
SWEEP_DIR = BASE_DIR / "publication" / "output" / "checkpoint_sweep"
FIGURES_DIR = SWEEP_DIR / "figures"

BUDGETS = [10, 25, 50, 100, 200]
BUDGET_COLORS = {10: "#059669", 25: "#0891b2", 50: "#2563eb", 100: "#d97706", 200: "#dc2626"}

# Definition "kleine Margin" (wie analyze_margins.py)
ZERO_EPS = 1e-9
ABS_SMALL_10 = 10
ABS_SMALL_20 = 20


def setup_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.35)
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1, "font.family": "sans-serif",
    })


def load_points() -> pd.DataFrame:
    frames = []
    for b in BUDGETS:
        path = SWEEP_DIR / f"budget_{b}" / "checkpoint_points.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing points CSV: {path} (rerun with --points)")
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["budget"] = b
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def feasible_mask(df: pd.DataFrame) -> pd.Series:
    return (df["within_budget"] == True) & (df["completed"] == True) & (df["co2_savings_pct"] > 0)  # noqa: E712


def best_under(grp: pd.DataFrame, cap: float | None) -> pd.Series | None:
    """Highest-score feasible policy with margin <= cap (cap=None: unconstrained).

    Mirrors ``runOptimization`` tie-breaking exactly: per iteration the *last*
    max-score point wins (``reduce`` with strict >), and across iterations the
    *first* iteration's best is kept on equal scores.  File order is iteration
    order, so: last max-score point of the earliest iteration that reaches the
    global max score.
    """
    if cap is not None:
        grp = grp[grp["margin"] <= cap]
    if grp.empty:
        return None
    mx = grp["score"].max()
    top = grp[grp["score"] == mx]
    first_iter = top["iteration"].min()
    # last max-score point inside that iteration (reduce keeps `b` on ties)
    cand = top[top["iteration"] == first_iter]
    return cand.iloc[-1]


def constraint_cost(points: pd.DataFrame) -> pd.DataFrame:
    """Per (budget, cost_s): optimum vs. best policy under a margin cap."""
    feas = points[feasible_mask(points)].copy()
    feas["margin"] = feas["theta_pause"] - feas["theta_resume"]

    rows = []
    for (budget, cost_s), grp in feas.groupby(["budget", "cost_s"]):
        best = best_under(grp, None)
        if best is None:
            continue
        b10 = best_under(grp, ABS_SMALL_10)
        b20 = best_under(grp, ABS_SMALL_20)
        b0 = best_under(grp, ZERO_EPS)

        def _capped(b):
            return b["co2_savings_pct"] if b is not None else np.nan

        def _regret(b):
            return best["co2_savings_pct"] - _capped(b)

        rows.append({
            "budget": budget,
            "cost_s": cost_s,
            "n_feasible": len(grp),
            "margin_best": best["margin"],
            "savings_best": best["co2_savings_pct"],
            "score_best": best["score"],
            "pauses_best": best["num_pauses"],
            "overhead_best": best["overhead_pct"],
            "small10": best["margin"] <= ABS_SMALL_10,
            "small20": best["margin"] <= ABS_SMALL_20,
            "zero": best["margin"] <= ZERO_EPS,
            "savings_capped0": _capped(b0),
            "savings_capped10": _capped(b10),
            "savings_capped20": _capped(b20),
            "regret0_pp": _regret(b0),
            "regret10_pp": _regret(b10),
            "regret20_pp": _regret(b20),
            "regret10_rel_pct": _regret(b10) / best["co2_savings_pct"] * 100 if b10 is not None else np.nan,
            "regret20_rel_pct": _regret(b20) / best["co2_savings_pct"] * 100 if b20 is not None else np.nan,
            "pause_delta10": (b10["num_pauses"] if b10 is not None else np.nan) - best["num_pauses"],
        })
    return pd.DataFrame(rows).sort_values(["budget", "cost_s"]).reset_index(drop=True)


def optima_rates(cost: pd.DataFrame) -> pd.DataFrame:
    """Share of optimal results with margin =0 / <=10 / <=20, per budget & overall."""
    def _rates(df: pd.DataFrame) -> dict:
        return {
            "n": len(df),
            "zero_rate_pct": df["zero"].mean() * 100,
            "small10_rate_pct": df["small10"].mean() * 100,
            "small20_rate_pct": df["small20"].mean() * 100,
            "median_margin": df["margin_best"].median(),
            "mean_regret10_pp": df["regret10_pp"].mean(),
            "max_regret10_pp": df["regret10_pp"].max(),
            "mean_regret20_pp": df["regret20_pp"].mean(),
            "max_regret20_pp": df["regret20_pp"].max(),
        }

    rows = [{"budget": "all", **_rates(cost)}]
    for b, grp in cost.groupby("budget"):
        rows.append({"budget": int(b), **_rates(grp)})
    return pd.DataFrame(rows)


def print_stats(cost: pd.DataFrame, rates: pd.DataFrame) -> None:
    n = len(cost)
    print("\n" + "=" * 78)
    print("MARGIN-ANALYSE: CHECKPOINT-COST-SWEEP (Deepseek / DE / 2025)")
    print("=" * 78)
    print(f"(budget, cost)-Paare: {n}; Budgets: {BUDGETS}; "
          f"feasible Policies: {int(cost['n_feasible'].sum())}")
    m = cost["margin_best"]
    print(f"optimale Margin: median {m.median():.2f}, "
          f"IQR {m.quantile(.25):.2f}–{m.quantile(.75):.2f}, "
          f"max {m.max():.2f} gCO₂/kWh")

    print("\nAnteile unter den optimalen Results:")
    all_row = rates[rates["budget"] == "all"].iloc[0]
    print(f"  gesamt (n={int(all_row['n'])}): "
          f"Margin = 0: {all_row['zero_rate_pct']:.1f}% · "
          f"≤10: {all_row['small10_rate_pct']:.1f}% · "
          f"≤20: {all_row['small20_rate_pct']:.1f}%")
    for _, r in rates[rates["budget"] != "all"].iterrows():
        print(f"  B={int(r['budget'])}% (n={int(r['n'])}): "
              f"= 0: {r['zero_rate_pct']:.0f}% · "
              f"≤10: {r['small10_rate_pct']:.0f}% · "
              f"≤20: {r['small20_rate_pct']:.0f}% · "
              f"med. Margin {r['median_margin']:.2f}")

    print("\nRegret beim Cap der Margin (Prozentpunkte CO₂-Savings):")
    for cap, col in [(0, "regret0_pp"), (10, "regret10_pp"), (20, "regret20_pp")]:
        s = cost[col]
        print(f"  Cap ≤{cap}: mean {s.mean():.3f}, median {s.median():.3f}, "
              f"max {s.max():.3f} pp · "
              f"= 0 in {int((s <= 1e-9).sum())}/{n} Fällen "
              f"(Cap erreichbar ohne Verlust)")

    r10 = cost["regret10_rel_pct"]
    r20 = cost["regret20_rel_pct"]
    print("\nRelativer Verlust (% der besten CO₂-Savings):")
    print(f"  Cap ≤10: mean {r10.mean():.3f}%, median {r10.median():.3f}%, max {r10.max():.3f}%")
    print(f"  Cap ≤20: mean {r20.mean():.3f}%, median {r20.median():.3f}%, max {r20.max():.3f}%")

    worst = cost.nlargest(5, "regret10_pp")[["budget", "cost_s", "margin_best", "savings_best", "savings_capped10", "regret10_pp"]]
    print("\n5 teuerste Cappings bei Cap ≤10:")
    print(worst.round(3).to_string(index=False))


def plot_margin_analysis(cost: pd.DataFrame) -> plt.Figure:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11.5, 7.6))

    # (a) rate of small/zero margins among the 40 optima, per budget
    rates = (cost.groupby("budget")
             .agg(zero=("zero", "mean"), small10=("small10", "mean"),
                  small20=("small20", "mean"))
             .reindex(BUDGETS) * 100)
    x = np.arange(len(BUDGETS))
    w = 0.27
    for off, col, label, color in [(-w, "zero", r"$\Delta_\theta = 0$", "#dc2626"),
                                   (0, "small10", r"$\Delta_\theta \leq 10$", "#059669"),
                                   (w, "small20", r"$\Delta_\theta \leq 20$", "#2563eb")]:
        vals = rates[col].to_numpy()
        ax1.bar(x + off, vals, w, label=label, color=color, edgecolor="black",
                linewidth=0.6, zorder=3)
        for xi, v in zip(x + off, vals):
            if v > 0:
                ax1.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, zorder=4)
    overall = cost[["zero", "small10", "small20"]].mean() * 100
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"B={b}%" for b in BUDGETS])
    ax1.set_ylabel("share of optima (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title(f"(a) small margins among optima (all: "
                  f"=0 {overall['zero']:.0f}%, ≤10 {overall['small10']:.0f}%, "
                  f"≤20 {overall['small20']:.0f}%)", loc="left", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # (b) optimal margin per (budget, cost)
    for b in BUDGETS:
        sub = cost[cost["budget"] == b]
        ax2.plot(sub["cost_s"], sub["margin_best"], "o-", color=BUDGET_COLORS[b],
                 lw=1.8, markersize=5, label=f"B={b}%")
    ax2.axhline(10, ls="--", color="#059669", lw=1.2)
    ax2.axhline(20, ls=":", color="#d97706", lw=1.2)
    ax2.text(620, 10.5, "10", color="#059669", fontsize=9, va="bottom", ha="right")
    ax2.text(620, 20.5, "20", color="#d97706", fontsize=9, va="bottom", ha="right")
    ax2.set_xlabel("checkpoint cost per save/load (s)")
    ax2.set_ylabel(r"optimal margin $\Delta_\theta$ (gCO$_2$eq/kWh)")
    ax2.set_title("(b) optimal margin vs. checkpoint cost", loc="left", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # (c) regret in pp when capping at <=10, per budget over cost
    for b in BUDGETS:
        sub = cost[cost["budget"] == b]
        ax3.plot(sub["cost_s"], sub["regret10_pp"], "o-", color=BUDGET_COLORS[b],
                 lw=1.8, markersize=5, label=f"B={b}%")
    ax3.axhline(0, color="0.4", lw=0.8)
    ax3.set_xlabel("checkpoint cost per save/load (s)")
    ax3.set_ylabel("savings lost (pp)")
    ax3.set_title(r"(c) regret when capping $\Delta_\theta \leq 10$", loc="left", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # (d) regret in pp when capping at <=20 + relative inset summary
    for b in BUDGETS:
        sub = cost[cost["budget"] == b]
        ax4.plot(sub["cost_s"], sub["regret20_pp"], "o-", color=BUDGET_COLORS[b],
                 lw=1.8, markersize=5, label=f"B={b}%")
    ax4.axhline(0, color="0.4", lw=0.8)
    ax4.set_xlabel("checkpoint cost per save/load (s)")
    ax4.set_ylabel("savings lost (pp)")
    ax4.set_title(r"(d) regret when capping $\Delta_\theta \leq 20$", loc="left", fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def main() -> None:
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    points = load_points()
    print(f"Loaded {len(points)} points across {len(BUDGETS)} budgets, "
          f"costs: {sorted(points['cost_s'].unique())}")

    cost = constraint_cost(points)
    rates = optima_rates(cost)

    cost.to_csv(SWEEP_DIR / "margin_constraint_cost.csv", index=False)
    rates.to_csv(SWEEP_DIR / "margin_optima_rates.csv", index=False)
    print(f"✓ CSV → {SWEEP_DIR / 'margin_constraint_cost.csv'}")
    print(f"✓ CSV → {SWEEP_DIR / 'margin_optima_rates.csv'}")

    print_stats(cost, rates)

    fig = plot_margin_analysis(cost)
    # bewusst nur lokal (siehe Docstring) -- nicht nach TheGreenEpochPaper/assets
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "margin_analysis_ckptcost.svg")
    fig.savefig(FIGURES_DIR / "margin_analysis_ckptcost.eps")
    plt.close(fig)
    print(f"\n✓ figure → {FIGURES_DIR}/margin_analysis_ckptcost.svg")


if __name__ == "__main__":
    main()
