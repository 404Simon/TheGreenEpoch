# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "numpy>=1.24",
# ]
# ///
"""Start-date sensitivity analysis (paper Sect. "Start Date Sensitivity").

Reproduces the numbers in that paragraph from the committed DeepSeek V3
optimization results:

  * best start date per region x overhead budget (all-2025 runs),
  * how many of those optima fall in Jan-Apr / Jan-Feb,
  * savings of a January vs. a July fixed start at each budget,
  * monthly mean carbon intensity per region (2025).

Data: ``publication/output/results`` (B=200) plus
``publication/output/budget_<B>/results`` (B in 10,25,50,100),
DeepSeek V3 only, alpha == 1.0.

Outputs:
  publication/output/start_date_analysis.csv   (best start per region/budget)

Run:  uv run scripts/analyze_start_dates.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "publication" / "output"
RESULTS_DIR = OUT_DIR / "results"
CO2_DIR = BASE_DIR / "public" / "data" / "co2"

BUDGETS = [10, 25, 50, 100, 200]
REGIONS = ["DE", "IT", "SE", "US", "CN"]
YEAR = 2025

CSV_RENAME = {
    "Overhead %": "overhead_pct",
    "CO₂ Save %": "co2_save_pct",
    "Score": "score",
    "Budget": "within_budget",
    "Stop": "stop",
    "Start": "start",
}


def budget_dir(budget: int) -> Path:
    if budget == 200:
        return RESULTS_DIR
    return OUT_DIR / f"budget_{budget}" / "results"


def threshold(region: str) -> int:
    return 100 if region == "SE" else 800


def parse_pct(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("+", "", regex=False)
        .astype(float)
    )


def best_rows(path: Path) -> pd.Series | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig").rename(columns=CSV_RENAME)
    df["overhead_pct"] = parse_pct(df["overhead_pct"])
    df["co2_save_pct"] = parse_pct(df["co2_save_pct"])
    feasible = df[
        (df["within_budget"] == "✓ Yes")
        & (df["stop"] == "completed")
        & (df["co2_save_pct"] > 0)
    ]
    if feasible.empty:
        return None
    return feasible.loc[feasible["score"].idxmax()]


def best_start(region: str, budget: int) -> dict | None:
    row = best_rows(
        budget_dir(budget) / f"DS_{region}_all_{YEAR}_{threshold(region)}_10it.csv"
    )
    if row is None:
        return None
    return {
        "region": region,
        "budget": budget,
        "start": row["start"],
        "savings_pct": round(float(row["co2_save_pct"]), 2),
        "overhead_pct": round(float(row["overhead_pct"]), 1),
    }


def fixed_start_savings(region: str, budget: int, start: str) -> float | None:
    start_token = start.replace("-", "_")
    row = best_rows(
        budget_dir(budget)
        / f"DS_{region}_{start_token}_{YEAR}_{threshold(region)}_10it.csv"
    )
    return None if row is None else round(float(row["co2_save_pct"]), 2)


def monthly_means(region: str) -> list[float]:
    with open(CO2_DIR / f"{region}_{YEAR}.json") as f:
        data = json.load(f)
    ci = np.asarray(data["carbonIntensity"], dtype=float)
    months = np.asarray([int(t[5:7]) for t in data["timestamps"]])
    return [float(ci[months == m].mean()) for m in range(1, 13)]


def main() -> None:
    best = []
    for budget in BUDGETS:
        for region in REGIONS:
            row = best_start(region, budget)
            if row is not None:
                best.append(row)
    best_df = pd.DataFrame(best)
    best_df.to_csv(OUT_DIR / "start_date_analysis.csv", index=False)
    print(f"✓ {OUT_DIR / 'start_date_analysis.csv'}")

    print("\nBest start date per region x budget (all-2025 runs):")
    print(best_df.pivot(index="region", columns="budget", values="start").to_string())

    months = best_df["start"].str.slice(0, 2).astype(int)
    print(
        f"\noptima starting in Jan-Apr: {(months <= 4).sum()}/{len(months)}; "
        f"Jan-Feb: {(months <= 2).sum()}/{len(months)}"
    )

    print("\nJanuary vs July fixed-start savings (percentage points):")
    for region in REGIONS:
        cells = []
        for budget in BUDGETS:
            jan = fixed_start_savings(region, budget, "01-01")
            jul = fixed_start_savings(region, budget, "07-01")
            diff = "—" if (jan is None or jul is None) else f"{jan - jul:+.1f}"
            cells.append(f"B{budget}: {diff}")
        print(f"  {region}: " + " | ".join(cells))

    print("\nMonthly mean carbon intensity (gCO2eq/kWh, 2025):")
    for region in REGIONS:
        print(f"  {region}: " + " ".join(f"{v:.0f}" for v in monthly_means(region)))


if __name__ == "__main__":
    main()
