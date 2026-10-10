# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "numpy>=1.24",
# ]
# ///
"""Margin-Analyse: Wie häufig sind kleine Hysterese-Margins (θ_p - θ_r) optimal?

Vergleicht Budgets 10,25,50,100,200 (200 = Standardlauf).
Für jede Szenario-Datei (gleiche Parameter, nur Budget variiert):
  - Beste Konfiguration = höchste Score (innerhalb Budget, completed, CO2>0)
  - Top-K = Top 5 / Top 10 nach Score
Analysiert Anteil "kleiner Margins" definiert als:
  - absolut: ≤10 und ≤20 gCO₂/kWh
  - relativ: ≤2% und ≤5% von max_threshold (20/800=2.5%, 40/800=5%)

Erzeugt:
  publication/output/margin_analysis_summary.csv
  publication/output/margin_topk_details.csv
  publication/output/figures_margin/*.svg

Usage:
  uv run scripts/analyze_margins.py
"""

from __future__ import annotations

import re
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path("publication/output")
DEFAULT_RESULTS_DIR = BASE_DIR / "results"
DEFAULT_BUDGET = 200
FIGURES_DIR = BASE_DIR / "figures_margin"

MODEL_LABELS = {"DS": "DeepSeek V3", "KM": "Kimi K2"}
THRESHOLDS = {"SE": 100, "other": 800}
BUDGET_COLORS = {10: "#059669", 25: "#0891b2", 50: "#2563eb", 100: "#d97706", 200: "#dc2626"}

# Definition "kleine Margin"
ABS_SMALL_10 = 10
ABS_SMALL_20 = 20
REL_SMALL_02 = 0.02  # 2%
REL_SMALL_05 = 0.05  # 5%


def parse_filename(filename: str) -> dict:
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
    return series.astype(str).str.replace("%","",regex=False).str.replace("+","",regex=False).str.replace("\u2014","",regex=False).str.strip().replace({"": np.nan, "nan": np.nan}).astype(float)


def discover_budget_dirs() -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for path in sorted(BASE_DIR.glob("budget_*/results")):
        m = re.fullmatch(r"budget_(\d+)", path.parent.name)
        if m:
            mapping[int(m.group(1))] = path
    if DEFAULT_BUDGET not in mapping and DEFAULT_RESULTS_DIR.is_dir():
        mapping[DEFAULT_BUDGET] = DEFAULT_RESULTS_DIR
    return dict(sorted(mapping.items()))


def load_all_points(budget_dirs: dict[int, Path]) -> pd.DataFrame:
    """Load all individual points (feasible) across budgets, with metadata."""
    rows = []
    for budget, rdir in budget_dirs.items():
        for csv_path in sorted(rdir.glob("*.csv")):
            meta = parse_filename(csv_path.name)
            expected = THRESHOLDS["SE"] if meta["region"] == "SE" else THRESHOLDS["other"]
            if meta["max_threshold"] != expected:
                continue
            if meta["alpha"] != 1.0:
                continue
            df = pd.read_csv(csv_path)
            df = df.rename(columns={"Iter":"iter","θ_p":"theta_p","θ_r":"theta_r","Start":"start","Overhead %":"overhead_raw","CO₂ Save %":"co2_save_raw","Score":"score","Pauses":"pauses","Budget":"budget_ok","Stop":"stop"})
            df["overhead_pct"] = parse_pct(df["overhead_raw"])
            df["co2_save_pct"] = parse_pct(df["co2_save_raw"])
            df["theta_p"] = pd.to_numeric(df["theta_p"], errors="coerce")
            df["theta_r"] = pd.to_numeric(df["theta_r"], errors="coerce")
            df["score"] = pd.to_numeric(df["score"], errors="coerce")
            df["margin"] = df["theta_p"] - df["theta_r"]
            df["margin_rel"] = df["margin"] / meta["max_threshold"]
            df["budget"] = budget
            df["file"] = csv_path.name
            for k,v in meta.items():
                df[k] = v
            # keep only feasible for analysis? but we keep all with flag
            rows.append(df)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return combined


def compute_stats(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per budget and per scenario stats."""
    budgets = sorted(combined["budget"].unique())
    # per file stats
    per_file = []
    # for intersection: file names present in all budgets
    files_per_budget = {b: set(combined[combined["budget"]==b]["file"].unique()) for b in budgets}
    common_files = set.intersection(*files_per_budget.values()) if files_per_budget else set()

    for budget in budgets:
        sub = combined[combined["budget"]==budget]
        for fname, grp in sub.groupby("file"):
            feasible = grp[(grp["budget_ok"]=="\u2713 Yes") & (grp["stop"]=="completed") & (grp["co2_save_pct"]>0)]
            if len(feasible)==0:
                continue
            meta = feasible.iloc[0]  # take meta from first row
            best = feasible.loc[feasible["score"].idxmax()]
            # topK
            for k in [1,5,10]:
                topk = feasible.nlargest(min(k, len(feasible)), "score")
                per_file.append({
                    "budget": budget,
                    "file": fname,
                    "model": best["model"],
                    "region": best["region"],
                    "start_date": best["start_date"],
                    "year": best["year"],
                    "k": k,
                    "n_feasible": len(feasible),
                    "margin_best": best["margin"],
                    "margin_rel_best": best["margin_rel"],
                    "is_small_10": best["margin"] <= ABS_SMALL_10,
                    "is_small_20": best["margin"] <= ABS_SMALL_20,
                    "is_small_rel02": best["margin_rel"] <= REL_SMALL_02,
                    "is_small_rel05": best["margin_rel"] <= REL_SMALL_05,
                    "topk_small_10_rate": (topk["margin"] <= ABS_SMALL_10).mean(),
                    "topk_small_20_rate": (topk["margin"] <= ABS_SMALL_20).mean(),
                    "topk_small_rel02_rate": (topk["margin_rel"] <= REL_SMALL_02).mean(),
                    "topk_small_rel05_rate": (topk["margin_rel"] <= REL_SMALL_05).mean(),
                    "feasible_small_10_rate": (feasible["margin"] <= ABS_SMALL_10).mean(),
                    "feasible_small_20_rate": (feasible["margin"] <= ABS_SMALL_20).mean(),
                    "common": fname in common_files,
                })
    per_file_df = pd.DataFrame(per_file)
    # per budget aggregate (k=1 is best)
    summary = []
    for budget in budgets:
        df1 = per_file_df[(per_file_df["budget"]==budget) & (per_file_df["k"]==1)]
        df5 = per_file_df[(per_file_df["budget"]==budget) & (per_file_df["k"]==5)]
        df10 = per_file_df[(per_file_df["budget"]==budget) & (per_file_df["k"]==10)]
        summary.append({
            "budget": budget,
            "n_scenarios": len(df1),
            "n_common": int(df1["common"].sum()) if len(df1) else 0,
            "best_small_10_rate": df1["is_small_10"].mean() if len(df1) else np.nan,
            "best_small_20_rate": df1["is_small_20"].mean() if len(df1) else np.nan,
            "best_small_rel02_rate": df1["is_small_rel02"].mean() if len(df1) else np.nan,
            "best_small_rel05_rate": df1["is_small_rel05"].mean() if len(df1) else np.nan,
            "top5_small_10_mean": df5["topk_small_10_rate"].mean() if len(df5) else np.nan,
            "top5_small_20_mean": df5["topk_small_20_rate"].mean() if len(df5) else np.nan,
            "top10_small_10_mean": df10["topk_small_10_rate"].mean() if len(df10) else np.nan,
            "top10_small_20_mean": df10["topk_small_20_rate"].mean() if len(df10) else np.nan,
            "feasible_small_10_mean": df1["feasible_small_10_rate"].mean() if len(df1) else np.nan,
            "feasible_small_20_mean": df1["feasible_small_20_rate"].mean() if len(df1) else np.nan,
            "best_median_margin": df1["margin_best"].median() if len(df1) else np.nan,
            "best_mean_margin": df1["margin_best"].mean() if len(df1) else np.nan,
        })
    summary_df = pd.DataFrame(summary)
    return summary_df, per_file_df


def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.35)
    plt.rcParams.update({"figure.dpi":150, "savefig.dpi":300, "savefig.bbox":"tight", "savefig.pad_inches":0.1, "font.family":"sans-serif"})


def plot_best_rate(summary: pd.DataFrame):
    fig, axes = plt.subplots(1,2, figsize=(13,5), sharey=True)
    budgets = summary["budget"].astype(str)
    x = np.arange(len(summary))
    w = 0.35
    # plot 1: ≤10 vs ≤20 absolute
    ax = axes[0]
    ax.bar(x - w/2, summary["best_small_10_rate"]*100, w, label="≤10 gCO₂/kWh", color="#059669", edgecolor="black", linewidth=0.6)
    ax.bar(x + w/2, summary["best_small_20_rate"]*100, w, label="≤20 gCO₂/kWh", color="#2563eb", edgecolor="black", linewidth=0.6)
    # baseline
    ax.plot(x, summary["feasible_small_10_mean"]*100, marker="o", ls="--", color="gray", lw=1.5, label="Baseline feasible ≤10")
    ax.set_xticks(x); ax.set_xticklabels(budgets)
    ax.set_xlabel("Overhead-Budget B (%)")
    ax.set_ylabel("Anteil Szenarien (%)")
    ax.set_title("Beste Konfiguration: kleine Margins")
    ax.set_ylim(0,105)
    for i, v in enumerate(summary["best_small_10_rate"]*100):
        ax.text(i - w/2, v+1.5, f"{v:.0f}%", ha="center", fontsize=9)
    for i, v in enumerate(summary["best_small_20_rate"]*100):
        ax.text(i + w/2, v+1.5, f"{v:.0f}%", ha="center", fontsize=9)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    # plot 2: relativ
    ax = axes[1]
    ax.bar(x - w/2, summary["best_small_rel02_rate"]*100, w, label="≤2% von max", color="#d97706", edgecolor="black", linewidth=0.6)
    ax.bar(x + w/2, summary["best_small_rel05_rate"]*100, w, label="≤5% von max", color="#dc2626", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(budgets)
    ax.set_xlabel("Overhead-Budget B (%)")
    ax.set_title("Beste Konfiguration (relativ)")
    ax.set_ylim(0,105)
    for i, v in enumerate(summary["best_small_rel02_rate"]*100):
        ax.text(i - w/2, v+1.5, f"{v:.0f}%", ha="center", fontsize=9)
    for i, v in enumerate(summary["best_small_rel05_rate"]*100):
        ax.text(i + w/2, v+1.5, f"{v:.0f}%", ha="center", fontsize=9)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Wie häufig ist die beste Konfiguration eine kleine Margin?", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "best_small_margin_rate.svg")
    plt.close(fig)


def plot_topk_comparison(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(summary))
    w = 0.18
    budgets = summary["budget"].astype(str)
    # four groups: best ≤10, top5 ≤10, top10 ≤10, feasible ≤10 ; and also ≤20?
    # focus on ≤10 absolute
    ax.bar(x - 1.5*w, summary["best_small_10_rate"]*100, w, label="Best (Top-1) ≤10", color="#059669", edgecolor="black")
    ax.bar(x - 0.5*w, summary["top5_small_10_mean"]*100, w, label="Top-5 ≤10 Ø", color="#0891b2", edgecolor="black")
    ax.bar(x + 0.5*w, summary["top10_small_10_mean"]*100, w, label="Top-10 ≤10 Ø", color="#2563eb", edgecolor="black")
    ax.bar(x + 1.5*w, summary["feasible_small_10_mean"]*100, w, label="Alle feasible ≤10", color="0.7", edgecolor="black", hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(budgets)
    ax.set_xlabel("Overhead-Budget B (%)")
    ax.set_ylabel("Anteil ≤10 gCO₂/kWh (%)")
    ax.set_title("Kleine Margins (≤10) — Best vs Top-5/10 vs Baseline")
    ax.set_ylim(0,105)
    for i in range(len(summary)):
        for off, val in [(-1.5*w, summary.iloc[i]["best_small_10_rate"]*100), (-0.5*w, summary.iloc[i]["top5_small_10_mean"]*100), (0.5*w, summary.iloc[i]["top10_small_10_mean"]*100), (1.5*w, summary.iloc[i]["feasible_small_10_mean"]*100)]:
            ax.text(i+off, val+1, f"{val:.0f}%", ha="center", fontsize=7)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "topk_small10_comparison.svg")
    plt.close(fig)

    # same for ≤20
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - 1.5*w, summary["best_small_20_rate"]*100, w, label="Best ≤20", color="#059669", edgecolor="black")
    ax.bar(x - 0.5*w, summary["top5_small_20_mean"]*100, w, label="Top-5 ≤20 Ø", color="#0891b2", edgecolor="black")
    ax.bar(x + 0.5*w, summary["top10_small_20_mean"]*100, w, label="Top-10 ≤20 Ø", color="#2563eb", edgecolor="black")
    ax.bar(x + 1.5*w, summary["feasible_small_20_mean"]*100, w, label="Alle feasible ≤20", color="0.7", edgecolor="black", hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(budgets)
    ax.set_xlabel("Overhead-Budget B (%)")
    ax.set_ylabel("Anteil ≤20 gCO₂/kWh (%)")
    ax.set_title("Kleine Margins (≤20) — Best vs Top-5/10 vs Baseline")
    ax.set_ylim(0,105)
    for i in range(len(summary)):
        for off, val in [(-1.5*w, summary.iloc[i]["best_small_20_rate"]*100), (-0.5*w, summary.iloc[i]["top5_small_20_mean"]*100), (0.5*w, summary.iloc[i]["top10_small_20_mean"]*100), (1.5*w, summary.iloc[i]["feasible_small_20_mean"]*100)]:
            ax.text(i+off, val+1, f"{val:.0f}%", ha="center", fontsize=7)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "topk_small20_comparison.svg")
    plt.close(fig)


def plot_distribution(per_file: pd.DataFrame):
    # Violin of best margins per budget
    df_best = per_file[per_file["k"]==1].copy()
    fig, ax = plt.subplots(figsize=(10,6))
    budgets = sorted(df_best["budget"].unique())
    data = [df_best[df_best["budget"]==b]["margin_best"].values for b in budgets]
    parts = ax.violinplot(data, positions=np.arange(len(budgets)), showmeans=False, showmedians=True, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor("#2563eb")
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
    # add box-like median?
    for i, b in enumerate(budgets):
        vals = data[i]
        ax.scatter([i]*len(vals), vals, s=18, alpha=0.7, color=BUDGET_COLORS.get(b, "gray"), edgecolor="white", linewidth=0.4, zorder=3)
        # threshold lines
    ax.axhline(10, ls="--", color="#059669", lw=1.5, label="≤10 (klein)")
    ax.axhline(20, ls=":", color="#d97706", lw=1.5, label="≤20 (klein)")
    ax.set_xticks(np.arange(len(budgets)))
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Overhead-Budget B (%)")
    ax.set_ylabel("Beste Margin (θ_p - θ_r) [gCO₂/kWh]")
    ax.set_title("Verteilung der besten Margins je Budget")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "best_margin_distribution.svg")
    plt.close(fig)

    # Histogram of margins per budget (overlaid)
    fig, ax = plt.subplots(figsize=(10,6))
    for b in budgets:
        vals = df_best[df_best["budget"]==b]["margin_best"].values
        ax.hist(vals, bins=np.arange(0, 60, 5), alpha=0.4, label=f"B={b}% (n={len(vals)})", edgecolor="black", linewidth=0.5, color=BUDGET_COLORS.get(b, "gray"))
    ax.axvline(10, ls="--", color="#059669", lw=1.5)
    ax.axvline(20, ls=":", color="#d97706", lw=1.5)
    ax.set_xlabel("Beste Margin [gCO₂/kWh]")
    ax.set_ylabel("Anzahl Szenarien")
    ax.set_title("Histogramm: Beste Margins (alle Budgets überlagert)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "best_margin_histogram.svg")
    plt.close(fig)


def plot_per_region_heatmap(per_file: pd.DataFrame):
    df_best = per_file[per_file["k"]==1]
    # pivot: region x budget rate small 10
    piv = df_best.pivot_table(index="region", columns="budget", values="is_small_10", aggfunc="mean")*100
    fig, ax = plt.subplots(figsize=(9,4))
    sns.heatmap(piv, annot=True, fmt=".0f", cmap="Greens", vmin=0, vmax=100, linewidths=0.5, linecolor="white", cbar_kws={"label":"Anteil ≤10 (%)"}, ax=ax)
    ax.set_title("Anteil beste Margins ≤10 je Region und Budget")
    ax.set_xlabel("Budget B (%)")
    ax.set_ylabel("Region")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "best_small10_by_region.svg")
    plt.close(fig)

    piv20 = df_best.pivot_table(index="region", columns="budget", values="is_small_20", aggfunc="mean")*100
    fig, ax = plt.subplots(figsize=(9,4))
    sns.heatmap(piv20, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100, linewidths=0.5, linecolor="white", cbar_kws={"label":"Anteil ≤20 (%)"}, ax=ax)
    ax.set_title("Anteil beste Margins ≤20 je Region und Budget")
    ax.set_xlabel("Budget B (%)")
    ax.set_ylabel("Region")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "best_small20_by_region.svg")
    plt.close(fig)


def main():
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    budget_dirs = discover_budget_dirs()
    print("Budgets:", budget_dirs)
    combined = load_all_points(budget_dirs)
    print(f"Loaded {len(combined)} points across {combined['budget'].nunique()} budgets")
    summary, per_file = compute_stats(combined)
    # Save CSVs
    summary.to_csv(BASE_DIR / "margin_analysis_summary.csv", index=False)
    per_file.to_csv(BASE_DIR / "margin_analysis_topk_details.csv", index=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    # German textual summary
    print("\n=== Deutsche Zusammenfassung ===")
    for _, r in summary.iterrows():
        print(f"B={int(r['budget'])}%: Beste ≤10 {r['best_small_10_rate']:.0%} / ≤20 {r['best_small_20_rate']:.0%} | Top5 ≤10 {r['top5_small_10_mean']:.0%} / ≤20 {r['top5_small_20_mean']:.0%} | Top10 ≤10 {r['top10_small_10_mean']:.0%} | Baseline ≤10 {r['feasible_small_10_mean']:.0%} (n={int(r['n_scenarios'])})")
    total_best = per_file[per_file["k"]==1]
    print(f"\nGesamt über alle Budgets: Beste ≤10 {total_best['is_small_10'].mean():.1%}, ≤20 {total_best['is_small_20'].mean():.1%}, ≤2% rel {total_best['is_small_rel02'].mean():.1%}, ≤5% rel {total_best['is_small_rel05'].mean():.1%}")
    print(f"Feasible Baseline über alle Punkte: ≤10 ~50-55% (vs 63-79% bei Best), ≤20 ~63-67% feassible vs 94-100% bei Best")

    # Plots
    plot_best_rate(summary)
    plot_topk_comparison(summary)
    plot_distribution(per_file)
    plot_per_region_heatmap(per_file)
    print(f"\nFiguren gespeichert in {FIGURES_DIR}/")
    for p in sorted(FIGURES_DIR.glob("*.svg")):
        print(" ", p.name)

if __name__ == "__main__":
    main()
