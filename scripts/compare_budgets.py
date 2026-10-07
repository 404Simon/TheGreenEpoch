# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
# ]
# ///
"""Compare CO₂ savings and time overhead across overhead budgets.

Reads the optimization result CSVs produced by the experiment scripts:

  publication/output/results/                  (budget 200, published run)
  publication/output/budget_<B>/results/       (budget <B>, run_budget_sweep.sh)

and writes:

  publication/output/budget_comparison.csv     (one row per run)
  publication/output/figures_budget/*.svg      (comparison figures)

Usage:
    uv run scripts/compare_budgets.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path("publication/output")
DEFAULT_RESULTS_DIR = BASE_DIR / "results"  # published B=200% run
DEFAULT_BUDGET = 200
COMPARISON_CSV = BASE_DIR / "budget_comparison.csv"
FIGURES_DIR = BASE_DIR / "figures_budget"

MODEL_LABELS = {"DS": "DeepSeek V3", "KM": "Kimi K2"}
MODEL_ORDER = ["DeepSeek V3", "Kimi K2"]
REGION_ORDER = ["SE", "DE", "IT", "US", "CN"]

# SE runs use θ_max=100, all other regions θ_max=800 (see run_experiments.sh)
THRESHOLDS = {"SE": 100, "other": 800}

BUDGET_COLORS = {
    10: "#059669",
    25: "#0891b2",
    50: "#2563eb",
    100: "#d97706",
    200: "#dc2626",
}


# ── Data loading ───────────────────────────────────────────────


def parse_filename(filename: str) -> dict:
    """Extract metadata from a result CSV filename.

    Expected patterns:
      Fixed start:    {MODEL}_{REGION}_{MM}_{DD}_{YYYY}_{MAXTHR}_{MAXIT}.csv
      All starts:     {MODEL}_{REGION}_all_{YYYY}_{MAXTHR}_{MAXIT}.csv
      Multi-year:     {MODEL}_{REGION}_all_{YYYY-YYYY}_{MAXTHR}_{MAXIT}_alpha{N}.csv
      Alpha variants: {MODEL}_{REGION}_all_{YYYY}_{MAXTHR}_{MAXIT}_alpha{NN}.csv
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    model_code = parts[0]
    region = parts[1]

    if parts[2] == "all":
        start_date = "all"
        year = parts[3]
        max_threshold = int(parts[4])
        max_iter = int(parts[5].replace("it", ""))
    else:
        start_date = f"{parts[2]}-{parts[3]}"
        year = parts[4]
        max_threshold = int(parts[5])
        max_iter = int(parts[6].replace("it", ""))

    # _alpha1 → 1.0, _alpha05 → 0.5, _alpha08 → 0.8
    alpha = 1.0
    for part in parts:
        if part.startswith("alpha"):
            token = part[len("alpha"):]
            if token == "1":
                alpha = 1.0
            elif token.isdigit() and len(token) == 2:
                alpha = int(token) / 100
            else:
                alpha = int(token) / 10

    return {
        "model_code": model_code,
        "model": MODEL_LABELS.get(model_code, model_code),
        "region": region,
        "start_date": start_date,
        "year": year,
        "max_threshold": max_threshold,
        "max_iterations": max_iter,
        "alpha": alpha,
    }


def parse_pct(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace("\u2014", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan})
        .astype(float)
    )


def discover_budget_dirs() -> dict[int, Path]:
    """Map budget → results directory (explicit budget_* dirs win over default)."""
    mapping: dict[int, Path] = {}
    for path in sorted(BASE_DIR.glob("budget_*/results")):
        match = re.fullmatch(r"budget_(\d+)", path.parent.name)
        if match:
            mapping[int(match.group(1))] = path
    if DEFAULT_BUDGET not in mapping and DEFAULT_RESULTS_DIR.is_dir():
        mapping[DEFAULT_BUDGET] = DEFAULT_RESULTS_DIR
    return dict(sorted(mapping.items()))


def load_budget(budget: int, results_dir: Path) -> pd.DataFrame:
    """Load all run-level best points for one budget."""
    csv_files = sorted(results_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    rows: list[dict] = []
    for csv_path in csv_files:
        meta = parse_filename(csv_path.name)

        # Threshold consistency check (SE → 100, others → 800)
        expected = THRESHOLDS["SE"] if meta["region"] == "SE" else THRESHOLDS["other"]
        if meta["max_threshold"] != expected:
            continue

        df = pd.read_csv(csv_path)
        df = df.rename(
            columns={
                "Iter": "iter",
                "θ_p": "theta_p",
                "θ_r": "theta_r",
                "Start": "start",
                "Overhead %": "overhead_raw",
                "CO₂ Save %": "co2_save_raw",
                "Score": "score",
                "Pauses": "pauses",
                "Budget": "budget_ok",
                "Stop": "stop",
            }
        )
        df["overhead_pct"] = parse_pct(df["overhead_raw"])
        df["co2_save_pct"] = parse_pct(df["co2_save_raw"])

        within = df[df["budget_ok"] == "\u2713 Yes"]
        # Savings are only meaningful if training finished: runs stopped
        # early by the budget guard (stop == "budget_exceeded") report
        # partial emissions against a full-run baseline and would appear
        # wildly inflated (cf. computeIsOk in src/domain/result.ts).
        complete = within[within["stop"] == "completed"]
        valid = complete[complete["co2_save_pct"] > 0]

        row = {
            "budget": budget,
            **meta,
            "file": csv_path.name,
            "n_points": len(df),
            "n_within_budget": len(within),
            "n_incomplete": len(within) - len(complete),
            "n_valid": len(valid),
        }

        if len(valid):
            best = valid.loc[valid["score"].idxmax()]
            row.update(
                {
                    "best_score": best["score"],
                    "best_savings_pct": best["co2_save_pct"],
                    "best_overhead_pct": best["overhead_pct"],
                    "best_theta_p": best["theta_p"],
                    "best_theta_r": best["theta_r"],
                    "best_start": str(best["start"]).strip(),
                    "best_pauses": best["pauses"],
                    "best_stop": best["stop"],
                }
            )
        else:
            row.update(
                {
                    "best_score": np.nan,
                    "best_savings_pct": np.nan,
                    "best_overhead_pct": np.nan,
                    "best_theta_p": np.nan,
                    "best_theta_r": np.nan,
                    "best_start": None,
                    "best_pauses": np.nan,
                    "best_stop": None,
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def load_all() -> pd.DataFrame:
    budget_dirs = discover_budget_dirs()
    if not budget_dirs:
        raise FileNotFoundError(f"No results directories found under {BASE_DIR}")
    frames = []
    for budget, path in budget_dirs.items():
        print(f"  budget {budget:>3}%  ← {path}")
        frames.append(load_budget(budget, path))
    return pd.concat(frames, ignore_index=True)


# ── Reporting ──────────────────────────────────────────────────


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print("BUDGET COMPARISON — best run per scenario (α = 1)")
    print("=" * 88)

    header = (
        f"{'Budget':>7} {'Runs':>5} {'NoValid':>8} {'Savings %':>18} "
        f"{'Median':>8} {'Overhead %':>11} {'Max':>7} {'Valid pts':>10}"
    )
    print(header)
    print("-" * len(header))

    for budget, grp in df.groupby("budget"):
        valid = grp["best_savings_pct"].dropna()
        share = grp["n_within_budget"].sum() / max(grp["n_points"].sum(), 1)
        mean = f"{valid.mean():.2f} ± {valid.std():.2f}" if len(valid) else "—"
        median = f"{valid.median():.2f}" if len(valid) else "—"
        oh = f"{grp.loc[valid.index, 'best_overhead_pct'].mean():.1f}" if len(valid) else "—"
        oh_max = f"{grp.loc[valid.index, 'best_overhead_pct'].max():.1f}" if len(valid) else "—"
        print(
            f"{budget:>6}% {len(grp):>5} {int(grp['best_savings_pct'].isna().sum()):>8} "
            f"{mean:>18} {median:>8} {oh:>11} {oh_max:>7} {share:>9.1%}"
        )

    print("\nPer model (mean best savings % / mean best overhead %):")
    piv_s = df.pivot_table(
        index="budget", columns="model", values="best_savings_pct", aggfunc="mean"
    )
    piv_o = df.pivot_table(
        index="budget", columns="model", values="best_overhead_pct", aggfunc="mean"
    )
    for budget in piv_s.index:
        cells = [
            f"{m}: {piv_s.loc[budget, m]:.2f}% / {piv_o.loc[budget, m]:.1f}%"
            if m in piv_s.columns and not pd.isna(piv_s.loc[budget, m])
            else f"{m}: —"
            for m in MODEL_ORDER
        ]
        print(f"  {budget:>3}%  " + "   ".join(cells))

    n_no_valid = int(df["best_savings_pct"].isna().sum())
    if n_no_valid:
        print(
            f"\n  ⚠ {n_no_valid} run(s) had no completed within-budget point with savings > 0:"
        )
        for _, r in df[df["best_savings_pct"].isna()].iterrows():
            print(
                f"    - budget {r['budget']}%: {r['file']} "
                f"({r['n_within_budget'] - r['n_incomplete']}/{r['n_points']} completed within budget)"
            )
    print("=" * 88)


# ── Figures ────────────────────────────────────────────────────


def setup_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.5)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "font.family": "sans-serif",
        }
    )


def _budget_scale(ax) -> None:
    ax.set_xscale("log")
    budgets = sorted(BUDGET_COLORS)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])


def plot_savings_vs_budget(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    rng = np.random.default_rng(0)
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        stats = sub.groupby("budget")["best_savings_pct"].agg(["mean", "min", "max"])
        ax.plot(
            stats.index, stats["mean"], marker="o", lw=2, label=model, zorder=3
        )
        ax.fill_between(
            stats.index, stats["min"], stats["max"], alpha=0.15, zorder=1
        )
        jitter = np.exp(rng.normal(0, 0.03, len(sub)))
        ax.scatter(
            sub["budget"] * jitter,
            sub["best_savings_pct"],
            s=28,
            alpha=0.55,
            zorder=2,
            color=ax.lines[-1].get_color(),
        )
    _budget_scale(ax)
    ax.set_xlabel("Overhead budget B (%)")
    ax.set_ylabel("Best CO₂ savings (%)")
    ax.set_title("CO₂ savings vs. overhead budget")
    ax.legend(frameon=True)
    fig.savefig(FIGURES_DIR / "savings_vs_budget.svg")
    plt.close(fig)


def plot_overhead_vs_budget(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    rng = np.random.default_rng(1)
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        stats = sub.groupby("budget")["best_overhead_pct"].agg(["mean", "min", "max"])
        ax.plot(stats.index, stats["mean"], marker="o", lw=2, label=model, zorder=3)
        ax.fill_between(stats.index, stats["min"], stats["max"], alpha=0.15, zorder=1)
        jitter = np.exp(rng.normal(0, 0.03, len(sub)))
        ax.scatter(
            sub["budget"] * jitter,
            sub["best_overhead_pct"],
            s=28,
            alpha=0.55,
            zorder=2,
            color=ax.lines[-1].get_color(),
        )
    budgets = sorted(BUDGET_COLORS)
    ax.plot(budgets, budgets, ls="--", c="0.4", lw=1.5, label="budget cap", zorder=1)
    _budget_scale(ax)
    ax.set_xlabel("Overhead budget B (%)")
    ax.set_ylabel("Time overhead of best run (%)")
    ax.set_title("Time overhead vs. overhead budget")
    ax.legend(frameon=True)
    fig.savefig(FIGURES_DIR / "overhead_vs_budget.svg")
    plt.close(fig)


def plot_savings_vs_overhead_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for budget in sorted(df["budget"].unique()):
        sub = df[df["budget"] == budget].dropna(subset=["best_savings_pct"])
        color = BUDGET_COLORS.get(budget, "0.5")
        ax.scatter(
            sub["best_overhead_pct"],
            sub["best_savings_pct"],
            s=60,
            alpha=0.8,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=f"B = {budget}%",
            zorder=3,
        )
        ax.axvline(budget, ls=":", lw=1, color=color, alpha=0.5, zorder=1)
    ax.set_xlabel("Time overhead (%)")
    ax.set_ylabel("Best CO₂ savings (%)")
    ax.set_title("Savings vs. overhead, colored by budget")
    ax.legend(frameon=True, fontsize=12)
    fig.savefig(FIGURES_DIR / "savings_vs_overhead_scatter.svg")
    plt.close(fig)


def plot_valid_share(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    share = (df.groupby("budget")["n_within_budget"].sum() /
             df.groupby("budget")["n_points"].sum())
    colors = [BUDGET_COLORS.get(b, "0.5") for b in share.index]
    ax.bar([str(int(b)) for b in share.index], share.values * 100, color=colors)
    for i, v in enumerate(share.values * 100):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=12)
    ax.set_xlabel("Overhead budget B (%)")
    ax.set_ylabel("Sweep points within budget (%)")
    ax.set_title("Share of feasible sweep points")
    ax.set_ylim(0, 105)
    fig.savefig(FIGURES_DIR / "valid_points_share.svg")
    plt.close(fig)


def make_figures(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_savings_vs_budget(df)
    plot_overhead_vs_budget(df)
    plot_savings_vs_overhead_scatter(df)
    plot_valid_share(df)
    print(f"\nFigures written to {FIGURES_DIR}/")


def main() -> None:
    setup_style()

    print("Loading results...")
    df = load_all()

    # Main comparison uses the default optimizer settings (α = 1);
    # α-sensitivity files stay in the CSV but are excluded here.
    df_main = df[df["alpha"] == 1.0].copy()

    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMPARISON_CSV, index=False)
    print(f"\nRun-level table written to {COMPARISON_CSV} ({len(df)} rows)")

    excluded = len(df) - len(df_main)
    if excluded:
        print(f"  ({excluded} α-variant run(s) excluded from summary/figures)")

    print_summary(df_main)
    make_figures(df_main)


if __name__ == "__main__":
    main()
