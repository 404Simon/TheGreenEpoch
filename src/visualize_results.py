"""Generate publication-quality figures from optimization result CSVs.

Reads CSV files from output/results/ and generates PNG figures to output/figures/
for the paper draft. Covers three analysis goals:

  #63  Pareto trade-off frontiers per model/region/start-time
  #64  Absolute and relative carbon reduction comparison across models
  #67  Hysteresis-margin analysis

Usage:
    python src/visualize_results.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path("output/results")
FIGURES_DIR = Path("output/figures")

MODEL_LABELS = {"DS": "DeepSeek V3", "KM": "Kimi K2"}
REGION_ORDER = ["SE", "DE", "IT", "US", "CN"]
START_LABELS = {"01-01": "Jan 1", "07-01": "Jul 1"}
MONTH_FROM_MMDD = {
    "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
    "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MODEL_ORDER = ["DeepSeek V3", "Kimi K2"]

COLORS_MODEL = {"DeepSeek V3": "#2563eb", "Kimi K2": "#dc2626"}
COLORS_REGION = {
    "SE": "#059669",
    "DE": "#2563eb",
    "IT": "#d97706",
    "US": "#7c3aed",
    "CN": "#dc2626",
}


# ── Data Loading ─────────────────────────────────────────────────────


def parse_filename(filename: str) -> dict:
    """Extract metadata from a result CSV filename.

    Expected patterns:
      Fixed start:    {MODEL}_{REGION}_{MM}_{DD}_{YYYY}_{MAXTHR}_{MAXIT}.csv
      All starts:     {MODEL}_{REGION}_all_{YYYY}_{MAXTHR}_{MAXIT}.csv
      Multi-year:     {MODEL}_{REGION}_all_{YYYY-YYYY}_{MAXTHR}_{MAXIT}_alpha{N}.csv
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    model_code = parts[0]
    region = parts[1]

    if parts[2] == "all":
        start_date = "all"
        if "-" in parts[3] and len(parts[3]) == 9:
            # Multi-year: all_2022-2025_800_10it_alpha1
            year = parts[3]
            max_threshold = int(parts[4])
            max_iter = int(parts[5].replace("it", ""))
        else:
            # Single year: all_2025_800_10it
            year = parts[3]
            max_threshold = int(parts[4])
            max_iter = int(parts[5].replace("it", ""))
    else:
        start_month = parts[2]
        start_day = parts[3]
        start_date = f"{start_month}-{start_day}"
        year = parts[4]
        max_threshold = int(parts[5])
        max_iter = int(parts[6].replace("it", ""))

    # Extract alpha variant if present (e.g. _alpha08 → 0.8)
    alpha = 1.0
    for part in parts:
        if part.startswith("alpha"):
            alpha = float(part.replace("alpha", "")) / 10

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


def load_and_parse_results(results_dir: Path) -> pd.DataFrame:
    """Load all result CSVs into a single DataFrame with parsed metadata."""
    frames: list[pd.DataFrame] = []
    csv_files = sorted(results_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    for csv_path in csv_files:
        meta = parse_filename(csv_path.name)
        df = pd.read_csv(csv_path)
        df = df.rename(columns={
            "Iter": "iter",
            "θ_p": "theta_p",
            "θ_r": "theta_r",
            "Start": "start",
            "Overhead %": "overhead_raw",
            "CO₂ Save %": "co2_save_raw",
            "Score": "score",
            "Pauses": "pauses",
            "Budget": "budget",
            "Stop": "stop",
        })
        for key, val in meta.items():
            df[key] = val

        # For _all_ files, use per-row start as start_date
        if meta["start_date"] == "all":
            df["start_date"] = df["start"].astype(str).str.strip()
            df["file_type"] = "all"
        else:
            df["file_type"] = "fixed"

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # For SE, use threshold-100 files; for other regions, use threshold-800
    se_mask = (combined["region"] == "SE") & (combined["max_threshold"] == 100)
    other_mask = (combined["region"] != "SE") & (combined["max_threshold"] == 800)
    combined = combined[se_mask | other_mask].reset_index(drop=True)

    combined["overhead_pct"] = (
        combined["overhead_raw"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("+", "", regex=False)
        .astype(float)
    )
    combined["co2_save_pct"] = (
        combined["co2_save_raw"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("+", "", regex=False)
        .astype(float)
    )
    combined["theta_p"] = pd.to_numeric(combined["theta_p"], errors="coerce")
    combined["theta_r"] = pd.to_numeric(combined["theta_r"], errors="coerce")
    combined["score"] = pd.to_numeric(combined["score"], errors="coerce")
    combined["pauses"] = pd.to_numeric(combined["pauses"], errors="coerce")
    combined["hysteresis_margin"] = combined["theta_p"] - combined["theta_r"]

    return combined


# ── Utility Functions ────────────────────────────────────────────────


def setup_style():
    """Set publication-quality matplotlib style."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "font.family": "sans-serif",
    })


def compute_pareto_frontier(
    overhead: np.ndarray, savings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Pareto frontier (upper envelope) for (overhead, savings).

    Returns sorted (overhead_frontier, savings_frontier) arrays.
    """
    order = np.argsort(overhead)
    o_sorted = overhead[order]
    s_sorted = savings[order]

    frontier_o: list[float] = []
    frontier_s: list[float] = []
    max_s = -np.inf

    for o, s in zip(o_sorted, s_sorted):
        if s > max_s:
            frontier_o.append(o)
            frontier_s.append(s)
            max_s = s

    return np.array(frontier_o), np.array(frontier_s)


def find_best_tradeoff(group: pd.DataFrame) -> pd.Series:
    """Return the row with the highest score in a group."""
    return group.loc[group["score"].idxmax()]


# ── Plot #63: Pareto Frontiers ───────────────────────────────────────


def plot_pareto_per_scenario(df: pd.DataFrame, output_dir: Path):
    """Generate individual Pareto frontier plots per (model, region, start)."""
    grouped = df.groupby(["model", "region", "start_date"])

    for (model, region, start), group in grouped:
        if len(group) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        overhead = group["overhead_pct"].values
        savings = group["co2_save_pct"].values
        thresholds = group["theta_p"].values

        scatter = ax.scatter(
            overhead,
            savings,
            c=thresholds,
            cmap="viridis",
            alpha=0.5,
            s=20,
            edgecolors="none",
            label="All configurations",
        )

        f_o, f_s = compute_pareto_frontier(overhead, savings)
        if len(f_o) > 1:
            ax.plot(
                f_o, f_s,
                color="black",
                linewidth=2,
                linestyle="-",
                marker="o",
                markersize=4,
                label="Pareto frontier",
                zorder=5,
            )

        best = find_best_tradeoff(group)
        ax.scatter(
            best["overhead_pct"],
            best["co2_save_pct"],
            marker="*",
            s=250,
            color="red",
            edgecolors="black",
            linewidths=1,
            zorder=10,
            label=f'Best score ({best["score"]:.3f})',
        )
        ax.annotate(
            f'θ_p={best["theta_p"]:.0f}\nθ_r={best["theta_r"]:.0f}',
            (best["overhead_pct"], best["co2_save_pct"]),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("Pause Threshold (θ_p) [gCO₂eq/kWh]")

        ax.set_xlabel("Time Overhead (%)")
        ax.set_ylabel("CO₂ Savings (%)")
        start_label = START_LABELS.get(start, start)
        ax.set_title(f"Trade-off Frontier: {model} in {region} (Start: {start_label})")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        safe_start = start.replace("-", "")
        filename = f"pareto_{model.replace(' ', '_')}_{region}_{safe_start}.png"
        fig.savefig(output_dir / filename)
        plt.close(fig)
        print(f"  ✓ {filename}")


def plot_pareto_overview(df: pd.DataFrame, output_dir: Path):
    """Generate a 2×5 overview grid of Pareto frontiers (models × regions)."""
    fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharex=False, sharey=False)

    for col_idx, region in enumerate(REGION_ORDER):
        for row_idx, model in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            subset = df[(df["model"] == model) & (df["region"] == region)]

            if subset.empty:
                ax.set_visible(False)
                continue

            for start_date, start_group in subset.groupby("start_date"):
                if len(start_group) < 2:
                    continue
                o = start_group["overhead_pct"].values
                s = start_group["co2_save_pct"].values
                label = START_LABELS.get(start_date, start_date)

                ax.scatter(o, s, alpha=0.3, s=10, color=COLORS_REGION.get(region, "gray"))

                f_o, f_s = compute_pareto_frontier(o, s)
                if len(f_o) > 1:
                    linestyle = "--" if start_date == "07-01" else "-"
                    ax.plot(
                        f_o, f_s,
                        linewidth=2,
                        linestyle=linestyle,
                        marker="o",
                        markersize=3,
                        label=f"Frontier ({label})",
                    )

            best = find_best_tradeoff(subset)
            ax.scatter(
                best["overhead_pct"],
                best["co2_save_pct"],
                marker="*",
                s=150,
                color="red",
                edgecolors="black",
                linewidths=0.8,
                zorder=10,
            )

            if row_idx == 0:
                ax.set_title(region, fontsize=14, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{model}\nCO₂ Savings (%)", fontsize=11)
            if row_idx == 1:
                ax.set_xlabel("Time Overhead (%)", fontsize=11)

            ax.legend(fontsize=7, loc="lower right")
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Pareto Trade-off Frontiers: CO₂ Savings vs Time Overhead",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    filename = "pareto_overview.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def plot_pareto_combined_start(df: pd.DataFrame, output_dir: Path):
    """For each model, one figure with all regions overlaid, both start dates."""
    for model in MODEL_ORDER:
        fig, ax = plt.subplots(figsize=(10, 7))
        model_df = df[df["model"] == model]

        for region in REGION_ORDER:
            region_df = model_df[model_df["region"] == region]
            if region_df.empty:
                continue

            for start_date, start_group in region_df.groupby("start_date"):
                if len(start_group) < 2:
                    continue
                o = start_group["overhead_pct"].values
                s = start_group["co2_save_pct"].values

                f_o, f_s = compute_pareto_frontier(o, s)
                if len(f_o) < 2:
                    continue

                label = f"{region} ({START_LABELS.get(start_date, start_date)})"
                linestyle = "--" if start_date == "07-01" else "-"
                ax.plot(
                    f_o, f_s,
                    linewidth=2,
                    linestyle=linestyle,
                    marker="o",
                    markersize=4,
                    color=COLORS_REGION.get(region, "gray"),
                    label=label,
                )

        ax.set_xlabel("Time Overhead (%)")
        ax.set_ylabel("CO₂ Savings (%)")
        ax.set_title(f"Pareto Frontiers: {model} — All Regions & Start Dates")
        ax.legend(fontsize=9, loc="lower right", ncol=2)
        ax.grid(True, alpha=0.3)

        safe_model = model.replace(" ", "_")
        filename = f"pareto_combined_{safe_model}.png"
        fig.savefig(output_dir / filename)
        plt.close(fig)
        print(f"  ✓ {filename}")


# ── Plot #64: Carbon Reduction Comparison ────────────────────────────


def plot_carbon_reduction_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare absolute and relative carbon reduction potential across models.

    For each (model, region, start_date), extract the best-scoring config.
    """
    best_rows = []
    for (model, region, start), group in df.groupby(["model", "region", "start_date"]):
        if group.empty:
            continue
        best = find_best_tradeoff(group)
        best_rows.append(best)

    best_df = pd.DataFrame(best_rows)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Sub-plot 1: Relative savings (CO₂ Save %)
    ax1 = axes[0]
    pivot_rel = best_df.pivot_table(
        index="region",
        columns="model",
        values="co2_save_pct",
        aggfunc="max",
    )
    pivot_rel = pivot_rel.reindex(REGION_ORDER)
    pivot_rel.plot(
        kind="bar",
        ax=ax1,
        color=[COLORS_MODEL.get(m, "gray") for m in pivot_rel.columns],
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_ylabel("Best CO₂ Savings (%)")
    ax1.set_xlabel("Region")
    ax1.set_title("Relative Carbon Reduction")
    ax1.legend(title="Model")
    ax1.tick_params(axis="x", rotation=0)
    ax1.grid(True, alpha=0.3, axis="y")

    # Sub-plot 2: Score
    ax2 = axes[1]
    pivot_score = best_df.pivot_table(
        index="region",
        columns="model",
        values="score",
        aggfunc="max",
    )
    pivot_score = pivot_score.reindex(REGION_ORDER)
    pivot_score.plot(
        kind="bar",
        ax=ax2,
        color=[COLORS_MODEL.get(m, "gray") for m in pivot_score.columns],
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_ylabel("Best Score")
    ax2.set_xlabel("Region")
    ax2.set_title("Composite Score (Savings / Overhead)")
    ax2.legend(title="Model")
    ax2.tick_params(axis="x", rotation=0)
    ax2.grid(True, alpha=0.3, axis="y")

    # Sub-plot 3: Overhead at best score
    ax3 = axes[2]
    pivot_oh = best_df.pivot_table(
        index="region",
        columns="model",
        values="overhead_pct",
        aggfunc="min",
    )
    pivot_oh = pivot_oh.reindex(REGION_ORDER)
    pivot_oh.plot(
        kind="bar",
        ax=ax3,
        color=[COLORS_MODEL.get(m, "gray") for m in pivot_oh.columns],
        edgecolor="black",
        linewidth=0.5,
    )
    ax3.set_ylabel("Time Overhead at Best Score (%)")
    ax3.set_xlabel("Region")
    ax3.set_title("Overhead at Best Trade-off")
    ax3.legend(title="Model")
    ax3.tick_params(axis="x", rotation=0)
    ax3.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Carbon Reduction Potential: DeepSeek V3 vs Kimi K2",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    filename = "carbon_reduction_comparison.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def plot_carbon_reduction_by_start(df: pd.DataFrame, output_dir: Path):
    """Per start-date version of the carbon reduction comparison."""
    for start_date in df["start_date"].unique():
        start_df = df[df["start_date"] == start_date]
        best_rows = []
        for (model, region), group in start_df.groupby(["model", "region"]):
            if group.empty:
                continue
            best = find_best_tradeoff(group)
            best_rows.append(best)

        if not best_rows:
            continue

        best_df = pd.DataFrame(best_rows)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Relative savings
        ax1 = axes[0]
        pivot_rel = best_df.pivot_table(
            index="region",
            columns="model",
            values="co2_save_pct",
            aggfunc="max",
        )
        pivot_rel = pivot_rel.reindex(REGION_ORDER)
        pivot_rel.plot(
            kind="bar",
            ax=ax1,
            color=[COLORS_MODEL.get(m, "gray") for m in pivot_rel.columns],
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.set_ylabel("Best CO₂ Savings (%)")
        ax1.set_xlabel("Region")
        ax1.set_title(f"Relative Savings (Start: {START_LABELS.get(start_date, start_date)})")
        ax1.legend(title="Model")
        ax1.tick_params(axis="x", rotation=0)
        ax1.grid(True, alpha=0.3, axis="y")

        # Score
        ax2 = axes[1]
        pivot_score = best_df.pivot_table(
            index="region",
            columns="model",
            values="score",
            aggfunc="max",
        )
        pivot_score = pivot_score.reindex(REGION_ORDER)
        pivot_score.plot(
            kind="bar",
            ax=ax2,
            color=[COLORS_MODEL.get(m, "gray") for m in pivot_score.columns],
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.set_ylabel("Best Score")
        ax2.set_xlabel("Region")
        ax2.set_title(f"Composite Score (Start: {START_LABELS.get(start_date, start_date)})")
        ax2.legend(title="Model")
        ax2.tick_params(axis="x", rotation=0)
        ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        safe_start = start_date.replace("-", "")
        filename = f"carbon_reduction_{safe_start}.png"
        fig.savefig(output_dir / filename)
        plt.close(fig)
        print(f"  ✓ {filename}")


# ── Plot #67: Threshold Space & Hysteresis Analysis ───────────────────


def plot_threshold_space_overview(df: pd.DataFrame, output_dir: Path):
    """2×5 grid: θ_p (x) vs θ_r (y), colored by score.

    Both start dates overlaid with different markers.
    """
    fig, axes = plt.subplots(
        len(MODEL_ORDER),
        len(REGION_ORDER),
        figsize=(24, 10),
        sharex=False,
        sharey=False,
    )

    markers = {"01-01": "o", "07-01": "^"}

    for col_idx, region in enumerate(REGION_ORDER):
        for row_idx, model in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            subset = df[(df["model"] == model) & (df["region"] == region)]

            if subset.empty:
                ax.set_visible(False)
                continue

            vmin = subset["score"].min()
            vmax = subset["score"].max()

            for start_date, start_group in subset.groupby("start_date"):
                if start_group.empty:
                    continue
                marker = markers.get(start_date, "o")
                label = START_LABELS.get(start_date, start_date)
                scatter = ax.scatter(
                    start_group["theta_p"],
                    start_group["theta_r"],
                    c=start_group["score"],
                    cmap="RdYlGn",
                    vmin=vmin,
                    vmax=vmax,
                    marker=marker,
                    alpha=0.5,
                    s=18,
                    edgecolors="none",
                    label=label,
                )

            lim_min = min(subset["theta_r"].min(), subset["theta_p"].min()) * 0.9
            lim_max = max(subset["theta_p"].max(), subset["theta_r"].max()) * 1.05
            ax.plot(
                [lim_min, lim_max], [lim_min, lim_max],
                color="gray", linestyle="--", linewidth=1, alpha=0.5,
                label="θ_p = θ_r (no hysteresis)",
            )
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)

            best = find_best_tradeoff(subset)
            ax.scatter(
                best["theta_p"],
                best["theta_r"],
                marker="*",
                s=200,
                color="red",
                edgecolors="black",
                linewidths=1,
                zorder=10,
            )

            if row_idx == 0:
                ax.set_title(region, fontsize=14, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{model}\nθ_r (Resume Threshold)", fontsize=11)
            if row_idx == len(MODEL_ORDER) - 1:
                ax.set_xlabel("θ_p (Pause Threshold)", fontsize=11)

            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.3)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    sm = plt.cm.ScalarMappable(
        cmap="RdYlGn",
        norm=plt.Normalize(vmin=df["score"].min(), vmax=df["score"].max()),
    )
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Score")

    fig.suptitle(
        "Threshold Space: Pause vs Resume Threshold (colored by Score)",
        fontsize=16,
        fontweight="bold",
        y=1.0,
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.3, right=0.9)
    filename = "threshold_space_overview.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def plot_threshold_space_per_scenario(df: pd.DataFrame, output_dir: Path):
    """Per (model, region, start): θ_p vs θ_r scatter colored by score."""
    grouped = df.groupby(["model", "region", "start_date"])

    for (model, region, start), group in grouped:
        if len(group) < 3:
            continue

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
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("Score")

        ax.set_xlabel("θ_p (Pause Threshold) [gCO₂eq/kWh]")
        ax.set_ylabel("θ_r (Resume Threshold) [gCO₂eq/kWh]")
        start_label = START_LABELS.get(start, start)
        ax.set_title(
            f"Threshold Space: {model} in {region} (Start: {start_label})"
        )
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        safe_start = start.replace("-", "")
        filename = (
            f"threshold_space_{model.replace(' ', '_')}_{region}_{safe_start}.png"
        )
        fig.savefig(output_dir / filename)
        plt.close(fig)
        print(f"  ✓ {filename}")


def plot_margin_vs_best(df: pd.DataFrame, output_dir: Path):
    """For each hysteresis margin, plot score / savings / overhead of the best-scoring run.

    2×5 grid (models × regions). Uses _all_ data (optimized over start dates).
    For each hysteresis margin, finds the run with the highest score and plots
    that run's score, savings, and overhead.
    """
    df_all = df[df["file_type"] == "all"]
    if df_all.empty:
        print("  No _all_ data found, skipping margin vs best overview.")
        return

    fig, axes = plt.subplots(
        len(MODEL_ORDER),
        len(REGION_ORDER),
        figsize=(24, 10),
        sharex=False,
        sharey=False,
    )

    for col_idx, region in enumerate(REGION_ORDER):
        for row_idx, model in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            ax2 = ax.twinx()
            subset = df_all[(df_all["model"] == model) & (df_all["region"] == region)]

            if subset.empty:
                ax.set_visible(False)
                ax2.set_visible(False)
                continue

            # For each hysteresis margin, find the run with highest score
            best_per_margin = subset.loc[
                subset.groupby("hysteresis_margin")["score"].idxmax()
            ].sort_values("hysteresis_margin")

            if len(best_per_margin) < 2:
                continue

            ax.plot(
                best_per_margin["hysteresis_margin"],
                best_per_margin["score"],
                color="#2563eb",
                linewidth=2,
                marker="o",
                markersize=4,
                label="Score",
            )
            ax.plot(
                best_per_margin["hysteresis_margin"],
                best_per_margin["co2_save_pct"] / 100,
                color="#059669",
                linewidth=2,
                marker="s",
                markersize=4,
                label="Savings/100",
            )
            ax2.plot(
                best_per_margin["hysteresis_margin"],
                best_per_margin["overhead_pct"],
                color="#dc2626",
                linewidth=2,
                marker="^",
                markersize=4,
                label="Overhead",
            )

            if row_idx == 0:
                ax.set_title(region, fontsize=14, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{model}\nScore / Savings/100", fontsize=11, color="#2563eb")
            if col_idx == len(REGION_ORDER) - 1:
                ax2.set_ylabel("Overhead (%)", fontsize=11, color="#dc2626")
            if row_idx == len(MODEL_ORDER) - 1:
                ax.set_xlabel("Hysteresis Margin (θ_p − θ_r)", fontsize=11)

            ax.tick_params(axis="y", labelcolor="#2563eb")
            ax2.tick_params(axis="y", labelcolor="#dc2626")
            ax.grid(True, alpha=0.3)

            if col_idx == len(REGION_ORDER) - 1:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(
                    lines1 + lines2, labels1 + labels2,
                    fontsize=7,
                    loc="upper left",
                    ncol=1,
                )

    fig.suptitle(
        "Hysteresis Margin vs Best Score / Savings / Overhead (All Start Dates)",
        fontsize=16,
        fontweight="bold",
        y=1.0,
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.4)
    filename = "margin_vs_best_overview.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def plot_margin_vs_best_per_model(df: pd.DataFrame, output_dir: Path):
    """Per (model, region): margin on x, score/savings/overhead of best-scoring run on y.

    Uses _all_ data (optimized over start dates). For each hysteresis margin,
    finds the run with the highest score and plots that run's metrics.
    """
    df_all = df[df["file_type"] == "all"]
    if df_all.empty:
        print("  No _all_ data found, skipping margin vs best per model.")
        return

    for model in MODEL_ORDER:
        fig, axes = plt.subplots(1, len(REGION_ORDER), figsize=(24, 5), sharey=False)
        model_df = df_all[df_all["model"] == model]

        for col_idx, region in enumerate(REGION_ORDER):
            ax = axes[col_idx]
            ax2 = ax.twinx()
            region_df = model_df[model_df["region"] == region]

            if region_df.empty:
                ax.set_visible(False)
                continue

            # For each hysteresis margin, find the run with highest score
            best_per_margin = region_df.loc[
                region_df.groupby("hysteresis_margin")["score"].idxmax()
            ].sort_values("hysteresis_margin")

            if len(best_per_margin) < 2:
                continue

            ax.plot(
                best_per_margin["hysteresis_margin"],
                best_per_margin["score"],
                color="#2563eb",
                linewidth=2,
                marker="o",
                markersize=4,
                label="Score",
            )
            ax.plot(
                best_per_margin["hysteresis_margin"],
                best_per_margin["co2_save_pct"] / 100,
                color="#059669",
                linewidth=2,
                marker="s",
                markersize=4,
                label="Savings/100",
            )
            ax2.plot(
                best_per_margin["hysteresis_margin"],
                best_per_margin["overhead_pct"],
                color="#dc2626",
                linewidth=2,
                marker="^",
                markersize=4,
                label="Overhead",
            )

            ax.set_title(region, fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel("Score / Savings/100", fontsize=11, color="#2563eb")
            if col_idx == len(REGION_ORDER) - 1:
                ax2.set_ylabel("Overhead (%)", fontsize=11, color="#dc2626")
            ax.set_xlabel("Hysteresis Margin (θ_p − θ_r)", fontsize=10)

            ax.tick_params(axis="y", labelcolor="#2563eb")
            ax2.tick_params(axis="y", labelcolor="#dc2626")
            ax.grid(True, alpha=0.3)
            if col_idx == len(REGION_ORDER) - 1:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

        fig.suptitle(
            f"Margin vs Best Metrics: {model} (All Start Dates)",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        fig.subplots_adjust(wspace=0.4)
        safe_model = model.replace(" ", "_")
        filename = f"margin_vs_best_{safe_model}.png"
        fig.savefig(output_dir / filename)
        plt.close(fig)
        print(f"  ✓ {filename}")


# ── Summary Table ────────────────────────────────────────────────────


def print_summary_table(df: pd.DataFrame):
    """Print a summary table of best configurations per (model, region, start)."""
    print("\n" + "=" * 90)
    print("BEST CONFIGURATION PER (Model, Region, Start Date)")
    print("=" * 90)
    print(
        f"{'Model':<14} {'Region':<6} {'Start':<8} {'θ_p':>5} {'θ_r':>5} "
        f"{'Δhyst':>6} {'Overhead%':>10} {'Save%':>8} {'Score':>7} {'Pauses':>7}"
    )
    print("-" * 90)

    for (model, region, start), group in df.groupby(["model", "region", "start_date"]):
        if group.empty:
            continue
        best = find_best_tradeoff(group)
        print(
            f"{model:<14} {region:<6} {START_LABELS.get(start, start):<8} "
            f"{best['theta_p']:>5.0f} {best['theta_r']:>5.0f} "
            f"{best['hysteresis_margin']:>6.0f} "
            f"{best['overhead_pct']:>9.1f}% "
            f"{best['co2_save_pct']:>7.2f}% "
            f"{best['score']:>7.4f} "
            f"{best['pauses']:>7.0f}"
        )

    print("=" * 90)


# ── Helpers for Start-Date Analysis ──────────────────────────────────


def _month_from_date(date_str: str) -> str:
    """Extract month label from MM-DD string."""
    mm = str(date_str).split("-")[0].zfill(2)
    return MONTH_FROM_MMDD.get(mm, date_str)


# ── Plot #68: Start-Date Optimization ───────────────────────────────


def print_best_startdate_table(df: pd.DataFrame, output_dir: Path):
    """For each (model, region), find the best-scoring start date.

    Prints formatted table and saves LaTeX table.
    """
    df_all = df[df["file_type"] == "all"]
    if df_all.empty:
        print("  No _all_ data found, skipping best-startdate table.")
        return

    print("\n" + "=" * 95)
    print("BEST STARTING DATE PER (Model, Region) — _all_ files")
    print("=" * 95)
    print(
        f"{'Model':<14} {'Region':<6} {'Best Start':<12} {'θ_p':>5} {'θ_r':>5} "
        f"{'Overhead%':>10} {'Save%':>8} {'Score':>7} {'Pauses':>7}"
    )
    print("-" * 95)

    rows = []
    for (model, region), group in df_all.groupby(["model", "region"]):
        if group.empty:
            continue
        best = find_best_tradeoff(group)
        start_label = best["start_date"]
        month_label = _month_from_date(start_label)
        rows.append({
            "model": model,
            "region": region,
            "best_start": start_label,
            "month": month_label,
            "theta_p": best["theta_p"],
            "theta_r": best["theta_r"],
            "overhead_pct": best["overhead_pct"],
            "co2_save_pct": best["co2_save_pct"],
            "score": best["score"],
            "pauses": best["pauses"],
        })
        print(
            f"{model:<14} {region:<6} {month_label + ' ' + start_label:<12} "
            f"{best['theta_p']:>5.0f} {best['theta_r']:>5.0f} "
            f"{best['overhead_pct']:>9.1f}% "
            f"{best['co2_save_pct']:>7.2f}% "
            f"{best['score']:>7.4f} "
            f"{best['pauses']:>7.0f}"
        )

    print("=" * 95)

    # Also print table per start date (aggregated across models/regions)
    print("\n" + "=" * 95)
    print("BEST CONFIGURATION PER START DATE — _all_ files")
    print("=" * 95)
    print(
        f"{'Model':<14} {'Region':<6} {'Start':<8} {'θ_p':>5} {'θ_r':>5} "
        f"{'Overhead%':>10} {'Save%':>8} {'Score':>7}"
    )
    print("-" * 95)

    for (model, region, start), group in df_all.groupby(["model", "region", "start_date"]):
        if group.empty:
            continue
        best = find_best_tradeoff(group)
        month_label = _month_from_date(start)
        print(
            f"{model:<14} {region:<6} {month_label:<8} "
            f"{best['theta_p']:>5.0f} {best['theta_r']:>5.0f} "
            f"{best['overhead_pct']:>9.1f}% "
            f"{best['co2_save_pct']:>7.2f}% "
            f"{best['score']:>7.4f}"
        )

    print("=" * 95)

    # Save LaTeX table
    best_df = pd.DataFrame(rows)
    if not best_df.empty:
        latex_path = output_dir / "best_startdate_table.tex"
        with open(latex_path, "w") as f:
            f.write("% Best starting date per (Model, Region)\n")
            f.write("\\begin{tabular}{lllrrrrr}\n")
            f.write("\\toprule\n")
            f.write("Model & Region & Best Start & $\\theta_p$ & $\\theta_r$ "
                    "& Overhead\\% & Save\\% & Score \\\\\n")
            f.write("\\midrule\n")
            for _, r in best_df.iterrows():
                f.write(
                    f"{r['model']} & {r['region']} & {r['month']} "
                    f"& {r['theta_p']:.0f} & {r['theta_r']:.0f} "
                    f"& {r['overhead_pct']:.1f} & {r['co2_save_pct']:.2f} "
                    f"& {r['score']:.4f} \\\\\n"
                )
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
        print(f"  ✓ LaTeX table saved to {latex_path}")


def plot_best_startdate_histogram(df: pd.DataFrame, output_dir: Path):
    """Histogram of best-scoring start dates with monthly bins.

    10 entries total (2 models × 5 regions). For each (model, region),
    extract the single best-scoring run across all (start_date, θ_p, θ_r)
    and plot which month it falls in. Shows whether training in a
    particular month is generally better, independent of country.
    """
    df_all = df[df["file_type"] == "all"]
    if df_all.empty:
        print("  No _all_ data found, skipping startdate histogram.")
        return

    # Collect the single best-scoring row per (model, region)
    best_rows = []
    for (model, region), group in df_all.groupby(["model", "region"]):
        if group.empty:
            continue
        best = find_best_tradeoff(group)
        best_rows.append({
            "model": model,
            "region": region,
            "month": _month_from_date(best["start_date"]),
            "score": best["score"],
        })

    if not best_rows:
        return

    best_df = pd.DataFrame(best_rows)

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    x = np.arange(len(MONTH_ORDER))

    for i, model in enumerate(MODEL_ORDER):
        model_df = best_df[best_df["model"] == model]
        counts = (
            model_df["month"]
            .value_counts()
            .reindex(MONTH_ORDER, fill_value=0)
        )
        offset = (i - 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            counts.values,
            bar_width,
            label=model,
            color=COLORS_MODEL.get(model, "gray"),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
        # Annotate region names on bars
        for month_idx, month in enumerate(MONTH_ORDER):
            region_hits = model_df[model_df["month"] == month]["region"].tolist()
            if region_hits:
                ax.annotate(
                    ",".join(region_hits),
                    (x[month_idx] + offset, counts.values[month_idx]),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=6,
                    rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_ORDER, rotation=45, ha="right")
    ax.set_ylabel("Count (best run per model & region)")
    ax.set_xlabel("Month")
    ax.set_title(
        "Best Start Date Distribution Across Models & Regions",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 6)

    fig.tight_layout()
    filename = "best_startdate_histogram.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def plot_best_abs_startdate_histogram(output_dir: Path):
    """Histogram of best absolute-savings start dates with monthly bins.

    Reads opt_*.json files. For each file, finds the completed run with
    the highest absolute savings (baseline - total emissions), extracts
    its start date, and plots which month it falls in.
    """
    import json as _json

    opt_dir = Path("output")
    opt_files = sorted(opt_dir.glob("opt_*.json"))
    if not opt_files:
        print("  No opt_*.json files found, skipping abs savings histogram.")
        return

    MODEL_MAP = {"Deepseek": "DeepSeek V3", "Kimi": "Kimi K2"}

    best_rows = []
    for fpath in opt_files:
        with open(fpath) as f:
            data = _json.load(f)

        completed = [
            p for p in data["points"]
            if p.get("stopReason") == "completed" and p.get("completed") is True
        ]
        if not completed:
            continue

        # Find run with highest absolute savings
        best = max(
            completed,
            key=lambda p: p["baselineEmissionsKgco2"] - p["totalEmissionsKgco2"],
        )

        model = MODEL_MAP.get(data.get("model", ""), data.get("model", "?"))
        region = data.get("region", "?")
        start_date = best.get("startTime", "?")
        abs_save = (best["baselineEmissionsKgco2"] - best["totalEmissionsKgco2"]) / 1000

        best_rows.append({
            "model": model,
            "region": region,
            "month": _month_from_date(start_date),
            "abs_save_tco2": abs_save,
        })

    if not best_rows:
        print("  No completed runs found in opt files, skipping histogram.")
        return

    best_df = pd.DataFrame(best_rows)

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    x = np.arange(len(MONTH_ORDER))

    for i, model in enumerate(MODEL_ORDER):
        model_df = best_df[best_df["model"] == model]
        counts = (
            model_df["month"]
            .value_counts()
            .reindex(MONTH_ORDER, fill_value=0)
        )
        offset = (i - 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            counts.values,
            bar_width,
            label=model,
            color=COLORS_MODEL.get(model, "gray"),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
        # Annotate region names on bars
        for month_idx, month in enumerate(MONTH_ORDER):
            region_hits = model_df[model_df["month"] == month]["region"].tolist()
            if region_hits:
                ax.annotate(
                    ",".join(region_hits),
                    (x[month_idx] + offset, counts.values[month_idx]),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=6,
                    rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_ORDER, rotation=45, ha="right")
    ax.set_ylabel("Count (best abs. savings run per model & region)")
    ax.set_xlabel("Month")
    ax.set_title(
        "Best Absolute Savings Start Date Distribution",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 6)

    fig.tight_layout()
    filename = "best_abs_startdate_histogram.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def plot_savings_vs_overhead_all(df: pd.DataFrame, output_dir: Path):
    """Pareto frontiers: CO₂ Savings vs Time Overhead for _all_ files.

    One figure per region, both models overlaid. Frontier style matches
    plot_pareto_combined_start (linewidth=2, marker="o", markersize=4,
    colored by model). Best-scoring run highlighted with red star.
    """
    df_all = df[df["file_type"] == "all"]
    if df_all.empty:
        print("  No _all_ data found, skipping savings vs overhead plot.")
        return

    for region in REGION_ORDER:
        region_df = df_all[df_all["region"] == region]
        if region_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))

        for model in MODEL_ORDER:
            model_df = region_df[region_df["model"] == model]
            if model_df.empty:
                continue

            o = model_df["overhead_pct"].values
            s = model_df["co2_save_pct"].values

            f_o, f_s = compute_pareto_frontier(o, s)
            if len(f_o) < 2:
                continue

            ax.plot(
                f_o, f_s,
                linewidth=2,
                linestyle="-",
                marker="o",
                markersize=4,
                color=COLORS_MODEL.get(model, "gray"),
                label=f"{model} frontier",
            )

            # Best-scoring run
            best = find_best_tradeoff(model_df)
            ax.scatter(
                best["overhead_pct"],
                best["co2_save_pct"],
                marker="*",
                s=200,
                color="red",
                edgecolors="black",
                linewidths=1,
                zorder=10,
                label=f'{model} best ({best["score"]:.3f})',
            )

        ax.set_xlabel("Time Overhead (%)")
        ax.set_ylabel("CO₂ Savings (%)")
        ax.set_title(f"Pareto Frontiers: CO₂ Savings vs Overhead — {region}")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

        safe_region = region
        filename = f"savings_vs_overhead_{safe_region}.png"
        fig.savefig(output_dir / filename)
        plt.close(fig)
        print(f"  ✓ {filename}")


def plot_score_vs_overhead_all(df: pd.DataFrame, output_dir: Path):
    """Pareto frontiers: Score vs Time Overhead for _all_ files.

    One figure per region, both models overlaid. Frontier style matches
    plot_pareto_combined_start (linewidth=2, marker="o", markersize=4,
    colored by model). Best-scoring run highlighted with red star.
    """
    df_all = df[df["file_type"] == "all"]
    if df_all.empty:
        print("  No _all_ data found, skipping score vs overhead plot.")
        return

    for region in REGION_ORDER:
        region_df = df_all[df_all["region"] == region]
        if region_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))

        for model in MODEL_ORDER:
            model_df = region_df[region_df["model"] == model]
            if model_df.empty:
                continue

            o = model_df["overhead_pct"].values
            sc = model_df["score"].values

            # Pareto frontier: maximize score for given overhead
            order = np.argsort(o)
            o_sorted = o[order]
            sc_sorted = sc[order]
            frontier_o: list[float] = []
            frontier_sc: list[float] = []
            max_sc = -np.inf
            for oi, sci in zip(o_sorted, sc_sorted):
                if sci > max_sc:
                    frontier_o.append(oi)
                    frontier_sc.append(sci)
                    max_sc = sci
            f_o = np.array(frontier_o)
            f_sc = np.array(frontier_sc)

            if len(f_o) < 2:
                continue

            ax.plot(
                f_o, f_sc,
                linewidth=2,
                linestyle="-",
                marker="o",
                markersize=4,
                color=COLORS_MODEL.get(model, "gray"),
                label=f"{model} frontier",
            )

            # Best-scoring run
            best = find_best_tradeoff(model_df)
            ax.scatter(
                best["overhead_pct"],
                best["score"],
                marker="*",
                s=200,
                color="red",
                edgecolors="black",
                linewidths=1,
                zorder=10,
                label=f'{model} best ({best["score"]:.3f})',
            )

        ax.set_xlabel("Time Overhead (%)")
        ax.set_ylabel("Score")
        ax.set_title(f"Pareto Frontiers: Score vs Overhead — {region}")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

        safe_region = region
        filename = f"score_vs_overhead_{safe_region}.png"
        fig.savefig(output_dir / filename)
        plt.close(fig)
        print(f"  ✓ {filename}")


def plot_savings_vs_overhead_combined(df: pd.DataFrame, output_dir: Path):
    """Combined Pareto frontiers: CO₂ Savings vs Overhead, all regions.

    Single figure with both models × all regions. DS uses solid lines,
    KM uses dashed lines. Colored by region (COLORS_REGION).
    Style matches plot_pareto_combined_start.
    """
    df_all = df[df["file_type"] == "all"]
    if df_all.empty:
        print("  No _all_ data found, skipping combined savings vs overhead.")
        return

    linestyles = {"DeepSeek V3": "-", "Kimi K2": "--"}

    fig, ax = plt.subplots(figsize=(12, 8))

    for model in MODEL_ORDER:
        for region in REGION_ORDER:
            subset = df_all[(df_all["model"] == model) & (df_all["region"] == region)]
            if subset.empty:
                continue

            o = subset["overhead_pct"].values
            s = subset["co2_save_pct"].values

            f_o, f_s = compute_pareto_frontier(o, s)
            if len(f_o) < 2:
                continue

            ls = linestyles.get(model, "-")
            label = f"{region} ({model})"
            ax.plot(
                f_o, f_s,
                linewidth=2,
                linestyle=ls,
                marker="o",
                markersize=4,
                color=COLORS_REGION.get(region, "gray"),
                label=label,
            )

            best = find_best_tradeoff(subset)
            ax.scatter(
                best["overhead_pct"],
                best["co2_save_pct"],
                marker="*",
                s=150,
                color="red",
                edgecolors="black",
                linewidths=0.8,
                zorder=10,
            )

    ax.set_xlabel("Time Overhead (%)")
    ax.set_ylabel("CO₂ Savings (%)")
    ax.set_title("Pareto Frontiers: CO₂ Savings vs Overhead — All Regions & Models")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)

    filename = "savings_vs_overhead_all.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def plot_score_vs_overhead_combined(df: pd.DataFrame, output_dir: Path):
    """Combined Pareto frontiers: Score vs Overhead, all regions.

    Single figure with both models × all regions. DS uses solid lines,
    KM uses dashed lines. Colored by region (COLORS_REGION).
    Style matches plot_pareto_combined_start.
    """
    df_all = df[df["file_type"] == "all"]
    if df_all.empty:
        print("  No _all_ data found, skipping combined score vs overhead.")
        return

    linestyles = {"DeepSeek V3": "-", "Kimi K2": "--"}

    fig, ax = plt.subplots(figsize=(12, 8))

    for model in MODEL_ORDER:
        for region in REGION_ORDER:
            subset = df_all[(df_all["model"] == model) & (df_all["region"] == region)]
            if subset.empty:
                continue

            o = subset["overhead_pct"].values
            sc = subset["score"].values

            order = np.argsort(o)
            o_sorted = o[order]
            sc_sorted = sc[order]
            frontier_o: list[float] = []
            frontier_sc: list[float] = []
            max_sc = -np.inf
            for oi, sci in zip(o_sorted, sc_sorted):
                if sci > max_sc:
                    frontier_o.append(oi)
                    frontier_sc.append(sci)
                    max_sc = sci
            f_o = np.array(frontier_o)
            f_sc = np.array(frontier_sc)

            if len(f_o) < 2:
                continue

            ls = linestyles.get(model, "-")
            label = f"{region} ({model})"
            ax.plot(
                f_o, f_sc,
                linewidth=2,
                linestyle=ls,
                marker="o",
                markersize=4,
                color=COLORS_REGION.get(region, "gray"),
                label=label,
            )

            best = find_best_tradeoff(subset)
            ax.scatter(
                best["overhead_pct"],
                best["score"],
                marker="*",
                s=150,
                color="red",
                edgecolors="black",
                linewidths=0.8,
                zorder=10,
            )

    ax.set_xlabel("Time Overhead (%)")
    ax.set_ylabel("Score")
    ax.set_title("Pareto Frontiers: Score vs Overhead — All Regions & Models")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)

    filename = "score_vs_overhead_all.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def plot_score_vs_iteration(df: pd.DataFrame, output_dir: Path):
    """Plot best score vs optimization iteration.

    5 per-region figures + 1 combined. For each (model, region, iteration),
    take the max score across all (θ_p, θ_r, start_date) configurations.
    Uses all data (fixed + _all_).
    """
    linestyles = {"DeepSeek V3": "-", "Kimi K2": "--"}

    # Per-region figures
    for region in REGION_ORDER:
        region_df = df[df["region"] == region]
        if region_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))

        for model in MODEL_ORDER:
            model_df = region_df[region_df["model"] == model]
            if model_df.empty:
                continue

            agg = (
                model_df.groupby("iter")
                .agg(best_score=("score", "max"))
                .reset_index()
                .sort_values("iter")
            )

            ax.plot(
                agg["iter"],
                agg["best_score"],
                linewidth=2,
                marker="o",
                markersize=6,
                color=COLORS_MODEL.get(model, "gray"),
                label=model,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Score")
        ax.set_title(f"Best Score vs Iteration — {region}")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(10))

        filename = f"score_vs_iteration_{region}.png"
        fig.savefig(output_dir / filename)
        plt.close(fig)
        print(f"  ✓ {filename}")

    # Combined figure
    fig, ax = plt.subplots(figsize=(14, 9))

    for model in MODEL_ORDER:
        for region in REGION_ORDER:
            subset = df[(df["model"] == model) & (df["region"] == region)]
            if subset.empty:
                continue

            agg = (
                subset.groupby("iter")
                .agg(best_score=("score", "max"))
                .reset_index()
                .sort_values("iter")
            )

            ls = linestyles.get(model, "-")
            ax.plot(
                agg["iter"],
                agg["best_score"],
                linewidth=2,
                linestyle=ls,
                marker="o",
                markersize=5,
                color=COLORS_REGION.get(region, "gray"),
                label=f"{region} ({model})",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Score")
    ax.set_title("Best Score vs Iteration — All Regions & Models")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(10))

    filename = "score_vs_iteration_all.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def plot_avg_score_vs_iteration(df: pd.DataFrame, output_dir: Path):
    """Plot average score vs optimization iteration.

    Same layout as plot_score_vs_iteration, but uses mean instead of max.
    For each (model, region, iteration), compute the average score across
    all (θ_p, θ_r, start_date) configurations.
    Uses all data (fixed + _all_).
    """
    linestyles = {"DeepSeek V3": "-", "Kimi K2": "--"}

    # Per-region figures
    for region in REGION_ORDER:
        region_df = df[df["region"] == region]
        if region_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))

        for model in MODEL_ORDER:
            model_df = region_df[region_df["model"] == model]
            if model_df.empty:
                continue

            agg = (
                model_df.groupby("iter")
                .agg(avg_score=("score", "mean"))
                .reset_index()
                .sort_values("iter")
            )

            ax.plot(
                agg["iter"],
                agg["avg_score"],
                linewidth=2,
                marker="o",
                markersize=6,
                color=COLORS_MODEL.get(model, "gray"),
                label=model,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Average Score")
        ax.set_title(f"Average Score vs Iteration — {region}")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(10))

        filename = f"avg_score_vs_iteration_{region}.png"
        fig.savefig(output_dir / filename)
        plt.close(fig)
        print(f"  ✓ {filename}")

    # Combined figure
    fig, ax = plt.subplots(figsize=(14, 9))

    for model in MODEL_ORDER:
        for region in REGION_ORDER:
            subset = df[(df["model"] == model) & (df["region"] == region)]
            if subset.empty:
                continue

            agg = (
                subset.groupby("iter")
                .agg(avg_score=("score", "mean"))
                .reset_index()
                .sort_values("iter")
            )

            ls = linestyles.get(model, "-")
            ax.plot(
                agg["iter"],
                agg["avg_score"],
                linewidth=2,
                linestyle=ls,
                marker="o",
                markersize=5,
                color=COLORS_REGION.get(region, "gray"),
                label=f"{region} ({model})",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Score")
    ax.set_title("Average Score vs Iteration — All Regions & Models")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(10))

    filename = "avg_score_vs_iteration_all.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def plot_alpha_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare alpha variants for DS_DE (alpha=1.0, 0.8, 0.5).

    Three figures:
      A. Grouped bar chart: Score, Savings%, Overhead% of best run per alpha
      B. Grouped bar chart: θ_p, θ_r of best run per alpha
      C. Scatter with Pareto frontiers per alpha
    """
    # Filter to DS_DE only, _all_ files
    subset = df[
        (df["region"] == "DE")
        & (df["model_code"] == "DS")
        & (df["file_type"] == "all")
    ]
    if subset.empty:
        print("  No DS_DE _all_ data found, skipping alpha comparison.")
        return

    alpha_values = sorted(subset["alpha"].unique())
    if len(alpha_values) < 2:
        print("  Only one alpha value found, skipping alpha comparison.")
        return

    alpha_colors = {1.0: "#2563eb", 0.8: "#059669", 0.5: "#dc2626"}
    alpha_labels = {1.0: "α = 1.0", 0.8: "α = 0.8", 0.5: "α = 0.5"}

    # Collect best run per alpha
    best_rows = []
    for alpha in alpha_values:
        alpha_df = subset[subset["alpha"] == alpha]
        if alpha_df.empty:
            continue
        best = find_best_tradeoff(alpha_df)
        best_rows.append(best)

    if not best_rows:
        return

    best_df = pd.DataFrame(best_rows)

    # ── Figure A: Metrics comparison ──
    fig, ax = plt.subplots(figsize=(10, 7))
    metrics = ["score", "co2_save_pct", "overhead_pct"]
    metric_labels = ["Score", "CO₂ Savings (%)", "Time Overhead (%)"]
    x = np.arange(len(metrics))
    bar_width = 0.25

    for i, alpha in enumerate(alpha_values):
        row = best_df[best_df["alpha"] == alpha]
        if row.empty:
            continue
        vals = [row[m].values[0] for m in metrics]
        offset = (i - 1) * bar_width
        bars = ax.bar(
            x + offset,
            vals,
            bar_width,
            label=alpha_labels.get(alpha, f"α = {alpha}"),
            color=alpha_colors.get(alpha, "gray"),
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.annotate(
                f"{val:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Value")
    ax.set_title("Alpha Comparison: Best-Run Metrics (DS_DE)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    filename = "alpha_comparison_metrics.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")

    # ── Figure B: Threshold comparison ──
    fig, ax = plt.subplots(figsize=(8, 7))
    thresh_labels = ["θ_p (Pause)", "θ_r (Resume)"]
    x = np.arange(len(thresh_labels))
    bar_width = 0.25

    for i, alpha in enumerate(alpha_values):
        row = best_df[best_df["alpha"] == alpha]
        if row.empty:
            continue
        vals = [row["theta_p"].values[0], row["theta_r"].values[0]]
        offset = (i - 1) * bar_width
        bars = ax.bar(
            x + offset,
            vals,
            bar_width,
            label=alpha_labels.get(alpha, f"α = {alpha}"),
            color=alpha_colors.get(alpha, "gray"),
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.annotate(
                f"{val:.0f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(thresh_labels)
    ax.set_ylabel("Threshold [gCO₂eq/kWh]")
    ax.set_title("Alpha Comparison: Optimal Thresholds (DS_DE)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    filename = "alpha_comparison_thresholds.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")

    # ── Figure C: Pareto frontiers ──
    fig, ax = plt.subplots(figsize=(10, 7))

    for alpha in alpha_values:
        alpha_df = subset[subset["alpha"] == alpha]
        if alpha_df.empty:
            continue

        o = alpha_df["overhead_pct"].values
        s = alpha_df["co2_save_pct"].values

        f_o, f_s = compute_pareto_frontier(o, s)
        if len(f_o) < 2:
            continue

        color = alpha_colors.get(alpha, "gray")
        label = alpha_labels.get(alpha, f"α = {alpha}")
        ax.plot(
            f_o, f_s,
            linewidth=2,
            marker="o",
            markersize=4,
            color=color,
            label=label,
        )

        best = find_best_tradeoff(alpha_df)
        ax.scatter(
            best["overhead_pct"],
            best["co2_save_pct"],
            marker="*",
            s=200,
            color="red",
            edgecolors="black",
            linewidths=1,
            zorder=10,
        )

    ax.set_xlabel("Time Overhead (%)")
    ax.set_ylabel("CO₂ Savings (%)")
    ax.set_title("Alpha Comparison: Pareto Frontiers (DS_DE)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filename = "alpha_comparison_pareto.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def plot_multiyear_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare multi-year (2022-2025) vs single-year (2025) results.

    Figures:
      A. Best score per (model, region): single-year vs multi-year
      B. Savings% and Overhead% per (model, region)
      C. Average best score across regions per model
      D. Average iteration score vs iteration (per-region + combined)
    """
    df_all = df[df["file_type"] == "all"].copy()
    if df_all.empty:
        print("  No _all_ data found, skipping multi-year comparison.")
        return

    # Classify as single-year or multi-year
    df_all["is_multiyear"] = df_all["year"].astype(str).str.contains("-")
    df_single = df_all[~df_all["is_multiyear"]]
    df_multi = df_all[df_all["is_multiyear"]]

    if df_multi.empty:
        print("  No multi-year data found, skipping multi-year comparison.")
        return

    year_label_single = "2025"
    year_labels_multi = {y: y for y in df_multi["year"].unique()}

    # ── Figure A: Best score comparison ──
    rows_a = []
    for (model, region), group in df_all.groupby(["model", "region"]):
        for is_my, sub in group.groupby("is_multiyear"):
            if sub.empty:
                continue
            best = find_best_tradeoff(sub)
            rows_a.append({
                "model": model,
                "region": region,
                "is_multiyear": is_my,
                "score": best["score"],
            })

    if not rows_a:
        return

    best_df = pd.DataFrame(rows_a)

    fig, ax = plt.subplots(figsize=(14, 7))
    x_labels = []
    x_positions = []
    bar_width = 0.35
    x = 0

    for model in MODEL_ORDER:
        for region in REGION_ORDER:
            sub = best_df[(best_df["model"] == model) & (best_df["region"] == region)]
            if sub.empty:
                continue
            x_labels.append(f"{region}\n({model.split()[0]})")
            x_positions.append(x)

            single = sub[sub["is_multiyear"] == False]
            multi = sub[sub["is_multiyear"] == True]

            if not single.empty:
                ax.bar(
                    x - bar_width / 2, single["score"].values[0], bar_width,
                    color=COLORS_MODEL.get(model, "gray"), alpha=0.5,
                    edgecolor="black", linewidth=0.5,
                    label=year_label_single if x == 0 else "",
                )
            if not multi.empty:
                ax.bar(
                    x + bar_width / 2, multi["score"].values[0], bar_width,
                    color=COLORS_MODEL.get(model, "gray"), alpha=1.0,
                    edgecolor="black", linewidth=0.5,
                    label="Multi-year" if x == 0 else "",
                )
            x += 1

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Best Score")
    ax.set_title("Best Score: Single-Year vs Multi-Year")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    filename = "multiyear_comparison_best.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")

    # ── Figure B: Savings and Overhead comparison ──
    rows_b = []
    for (model, region), group in df_all.groupby(["model", "region"]):
        for is_my, sub in group.groupby("is_multiyear"):
            if sub.empty:
                continue
            best = find_best_tradeoff(sub)
            rows_b.append({
                "model": model,
                "region": region,
                "is_multiyear": is_my,
                "co2_save_pct": best["co2_save_pct"],
                "overhead_pct": best["overhead_pct"],
            })

    best_b = pd.DataFrame(rows_b)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax_idx, (metric, label) in enumerate([
        ("co2_save_pct", "CO₂ Savings (%)"),
        ("overhead_pct", "Time Overhead (%)"),
    ]):
        ax = axes[ax_idx]
        x = 0
        x_labels = []
        x_positions = []

        for model in MODEL_ORDER:
            for region in REGION_ORDER:
                sub = best_b[(best_b["model"] == model) & (best_b["region"] == region)]
                if sub.empty:
                    continue
                x_labels.append(f"{region}\n({model.split()[0]})")
                x_positions.append(x)

                single = sub[sub["is_multiyear"] == False]
                multi = sub[sub["is_multiyear"] == True]

                if not single.empty:
                    ax.bar(
                        x - bar_width / 2, single[metric].values[0], bar_width,
                        color=COLORS_MODEL.get(model, "gray"), alpha=0.5,
                        edgecolor="black", linewidth=0.5,
                        label=year_label_single if x == 0 else "",
                    )
                if not multi.empty:
                    ax.bar(
                        x + bar_width / 2, multi[metric].values[0], bar_width,
                        color=COLORS_MODEL.get(model, "gray"), alpha=1.0,
                        edgecolor="black", linewidth=0.5,
                        label="Multi-year" if x == 0 else "",
                    )
                x += 1

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(f"{label}: Single-Year vs Multi-Year")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    filename = "multiyear_comparison_metrics.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")

    # ── Figure C: Average best score across regions ──
    fig, ax = plt.subplots(figsize=(8, 7))
    x = np.arange(len(MODEL_ORDER))
    bar_width_c = 0.35

    for i, is_my in enumerate([False, True]):
        label = "Multi-year" if is_my else year_label_single
        vals = []
        for model in MODEL_ORDER:
            sub = best_df[(best_df["model"] == model) & (best_df["is_multiyear"] == is_my)]
            vals.append(sub["score"].mean() if not sub.empty else 0)

        offset = (i - 0.5) * bar_width_c
        bars = ax.bar(
            x + offset, vals, bar_width_c,
            color=[COLORS_MODEL.get(m, "gray") for m in MODEL_ORDER],
            alpha=1.0 if is_my else 0.5,
            edgecolor="black", linewidth=0.5,
            label=label,
        )
        for bar, val in zip(bars, vals):
            ax.annotate(
                f"{val:.4f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("Average Best Score")
    ax.set_title("Average Best Score Across Regions")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    filename = "multiyear_avg_score.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


BUDGET_PCT = 200


def _compute_score(savings_pct, overhead_pct, alpha, budget=BUDGET_PCT):
    """Compute score using the project's scoring function."""
    savings_norm = savings_pct / 100
    overhead_norm = overhead_pct / budget
    return (alpha * savings_norm + 1 - (1 - alpha) * overhead_norm) / 2


def plot_score_heatmaps(output_dir: Path):
    """2D heatmaps showing score distribution across savings (y) and overhead (x).

    One heatmap per alpha (1.0, 0.8, 0.5, 0.0). Green = high score, red = low.
    """
    alphas = [1.0, 0.8, 0.5, 0.0]

    savings_range = np.linspace(0, 50, 200)
    overhead_range = np.linspace(0, 200, 200)
    savings_grid, overhead_grid = np.meshgrid(savings_range, overhead_range)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for ax, alpha in zip(axes.flat, alphas):
        scores = _compute_score(savings_grid, overhead_grid, alpha)

        im = ax.imshow(
            scores.T,
            extent=[0, 200, 0, 50],
            origin="lower",
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
        )

        # Contour lines at key thresholds
        contour_levels = [0.3, 0.4, 0.5, 0.6, 0.7]
        cs = ax.contour(
            overhead_range, savings_range,
            scores.T,
            levels=contour_levels,
            colors="black",
            linewidths=0.8,
            linestyles="--",
        )
        ax.clabel(cs, fmt="%.1f", fontsize=8)

        ax.set_xlabel("Time Overhead (%)")
        ax.set_ylabel("CO₂ Savings (%)")
        ax.set_title(f"α = {alpha}", fontsize=14, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Score")

    fig.suptitle(
        "Score Heatmap: Savings vs Overhead (budget = 200%)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    filename = "score_heatmaps.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def plot_score_example_comparison(output_dir: Path):
    """Bar chart comparing example runs across different alpha values.

    Examples chosen so that different alphas produce different winners.
    """
    examples = [
        {"name": "A: Aggressive", "savings": 43, "overhead": 190},
        {"name": "B: Conservative", "savings": 3, "overhead": 5},
        {"name": "C: Efficient", "savings": 30, "overhead": 80},
        {"name": "D: Cautious", "savings": 15, "overhead": 30},
        {"name": "E: Extreme", "savings": 50, "overhead": 200},
    ]

    alphas = [1.0, 0.8, 0.5, 0.0]
    alpha_colors = {1.0: "#2563eb", 0.8: "#059669", 0.5: "#d97706", 0.0: "#dc2626"}

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(examples))
    bar_width = 0.2

    for i, alpha in enumerate(alphas):
        scores = [
            _compute_score(ex["savings"], ex["overhead"], alpha)
            for ex in examples
        ]
        offset = (i - 1.5) * bar_width
        bars = ax.bar(
            x + offset, scores, bar_width,
            label=f"α = {alpha}",
            color=alpha_colors[alpha],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, score in zip(bars, scores):
            ax.annotate(
                f"{score:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7,
            )

    # Add savings/overhead annotations below x-axis
    xlabels = []
    for ex in examples:
        xlabels.append(f"{ex['name']}\n({ex['savings']}% save, {ex['overhead']}% OH)")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Score Comparison: Example Runs Across Alpha Values", fontsize=14, fontweight="bold")
    ax.legend(title="Alpha")
    ax.grid(True, alpha=0.3, axis="y")

    # Highlight winner per example
    for j, ex in enumerate(examples):
        best_alpha = max(alphas, key=lambda a: _compute_score(ex["savings"], ex["overhead"], a))
        best_score = _compute_score(ex["savings"], ex["overhead"], best_alpha)
        ax.annotate(
            f"★ α={best_alpha}",
            (j, best_score + 0.02),
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=alpha_colors[best_alpha],
        )

    fig.tight_layout()
    filename = "score_example_comparison.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def plot_absolute_relative_savings(output_dir: Path):
    """Compare absolute and relative savings from optimization JSON files.

    Reads opt_*.json files, finds best completed run per file, and plots
    absolute (tCO₂) vs relative (%) savings side by side.
    """
    import json as _json

    opt_dir = Path("output")
    opt_files = sorted(opt_dir.glob("opt_*.json"))
    if not opt_files:
        print("  No opt_*.json files found, skipping absolute/relative savings plot.")
        return

    labels = []
    rel_savings = []
    abs_savings = []

    for fpath in opt_files:
        with open(fpath) as f:
            data = _json.load(f)

        completed = [
            p for p in data["points"]
            if p.get("stopReason") == "completed" and p.get("completed") is True
        ]
        if not completed:
            continue

        best = max(completed, key=lambda p: p["score"])
        model_code = "DS" if "Deepseek" in data.get("model", "") else "KM"
        region = data.get("region", "?")
        labels.append(f"{region}\n({model_code})")
        rel_savings.append(best["co2SavingsPct"])
        abs_savings.append(
            (best["baselineEmissionsKgco2"] - best["totalEmissionsKgco2"]) / 1000
        )

    if not labels:
        print("  No completed runs found, skipping absolute/relative savings plot.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))

    x = np.arange(len(labels))
    bar_width = 0.35

    # Relative savings (left axis)
    bars1 = ax1.bar(
        x - bar_width / 2, rel_savings, bar_width,
        color="#2563eb", alpha=0.8, edgecolor="black", linewidth=0.5,
        label="Relative Savings (%)",
    )
    ax1.set_ylabel("Relative Savings (%)", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")

    # Absolute savings (right axis)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + bar_width / 2, abs_savings, bar_width,
        color="#059669", alpha=0.8, edgecolor="black", linewidth=0.5,
        label="Absolute Savings (tCO₂)",
    )
    ax2.set_ylabel("Absolute Savings (tCO₂)", color="#059669")
    ax2.tick_params(axis="y", labelcolor="#059669")

    # Annotations
    for bar, val in zip(bars1, rel_savings):
        if val > 0:
            ax1.annotate(
                f"{val:.1f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=8, color="#2563eb",
            )
    for bar, val in zip(bars2, abs_savings):
        if val > 0:
            ax2.annotate(
                f"{val:.0f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=8, color="#059669",
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_title(
        "Absolute vs Relative CO₂ Savings (Best Completed Run per Scenario)",
        fontsize=14, fontweight="bold",
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    filename = "absolute_relative_savings.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def plot_best_run_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare savings and overhead of the best run of each model per country.

    Grouped bar chart with two subplots: one for savings, one for overhead.
    """
    df_all = df[df["file_type"] == "all"]
    if df_all.empty:
        print("  No _all_ data found, skipping best-run comparison.")
        return

    best_rows = []
    for (model, region), group in df_all.groupby(["model", "region"]):
        if group.empty:
            continue
        best = find_best_tradeoff(group)
        best_rows.append(best)

    if not best_rows:
        return

    best_df = pd.DataFrame(best_rows)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Sub-plot 1: Savings
    ax1 = axes[0]
    pivot_save = best_df.pivot_table(
        index="region",
        columns="model",
        values="co2_save_pct",
        aggfunc="max",
    )
    pivot_save = pivot_save.reindex(REGION_ORDER)
    pivot_save.plot(
        kind="bar",
        ax=ax1,
        color=[COLORS_MODEL.get(m, "gray") for m in pivot_save.columns],
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_ylabel("CO₂ Savings (%)")
    ax1.set_xlabel("Region")
    ax1.set_title("Best CO₂ Savings per Model & Region")
    ax1.legend(title="Model")
    ax1.tick_params(axis="x", rotation=0)
    ax1.grid(True, alpha=0.3, axis="y")

    # Sub-plot 2: Overhead
    ax2 = axes[1]
    pivot_oh = best_df.pivot_table(
        index="region",
        columns="model",
        values="overhead_pct",
        aggfunc="min",
    )
    pivot_oh = pivot_oh.reindex(REGION_ORDER)
    pivot_oh.plot(
        kind="bar",
        ax=ax2,
        color=[COLORS_MODEL.get(m, "gray") for m in pivot_oh.columns],
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_ylabel("Time Overhead (%)")
    ax2.set_xlabel("Region")
    ax2.set_title("Overhead at Best Score per Model & Region")
    ax2.legend(title="Model")
    ax2.tick_params(axis="x", rotation=0)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Best Run Comparison: DeepSeek V3 vs Kimi K2",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    filename = "best_run_comparison.png"
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    setup_style()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_and_parse_results(RESULTS_DIR)
    print(f"  Loaded {len(df)} rows from {df['model'].nunique()} models, "
          f"{df['region'].nunique()} regions, {df['start_date'].nunique()} start dates\n")

    # Split into fixed-start and all-start data
    df_fixed = df[df["file_type"] == "fixed"]
    df_all = df[df["file_type"] == "all"]

    print(f"  Fixed-start rows: {len(df_fixed)}, All-start rows: {len(df_all)}\n")

    print("Generating Pareto frontier plots (#63)...")
    plot_pareto_per_scenario(df_fixed, FIGURES_DIR)
    plot_pareto_overview(df_fixed, FIGURES_DIR)
    plot_pareto_combined_start(df_fixed, FIGURES_DIR)

    print("\nGenerating carbon reduction comparison plots (#64)...")
    plot_carbon_reduction_comparison(df_fixed, FIGURES_DIR)
    plot_carbon_reduction_by_start(df_fixed, FIGURES_DIR)

    print("\nGenerating hysteresis comparison plots (#67)...")
    plot_threshold_space_overview(df_fixed, FIGURES_DIR)
    plot_threshold_space_per_scenario(df_fixed, FIGURES_DIR)
    plot_margin_vs_best(df, FIGURES_DIR)
    plot_margin_vs_best_per_model(df, FIGURES_DIR)

    print_summary_table(df_fixed)

    print("\nGenerating start-date optimization plots (#68)...")
    print_best_startdate_table(df, FIGURES_DIR)
    plot_best_startdate_histogram(df, FIGURES_DIR)
    plot_best_abs_startdate_histogram(FIGURES_DIR)
    plot_score_vs_iteration(df, FIGURES_DIR)
    plot_avg_score_vs_iteration(df, FIGURES_DIR)
    plot_savings_vs_overhead_all(df, FIGURES_DIR)
    plot_score_vs_overhead_all(df, FIGURES_DIR)
    plot_savings_vs_overhead_combined(df, FIGURES_DIR)
    plot_score_vs_overhead_combined(df, FIGURES_DIR)
    plot_alpha_comparison(df, FIGURES_DIR)
    plot_multiyear_comparison(df, FIGURES_DIR)
    plot_score_heatmaps(FIGURES_DIR)
    plot_score_example_comparison(FIGURES_DIR)
    plot_absolute_relative_savings(FIGURES_DIR)
    plot_best_run_comparison(df, FIGURES_DIR)

    print(f"\n✓ All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
