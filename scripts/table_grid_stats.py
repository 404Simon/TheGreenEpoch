# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.24",
# ]
# ///
"""Paper Table 1: grid carbon-intensity characteristics (2025 only).

Generates the LaTeX tabular body for the paper's grid-statistics table
(``\\label{tab:grid}``) from the same 2025 Electricity Maps series the
optimizer uses (``public/data/co2/{ZONE}_2025.json``).

Regions are ordered by coefficient of variation (descending), matching the
caption. mean/std are in gCO2eq/kWh; std is the sample standard deviation
(ddof=1).

Outputs:
  publication/output/tables/grid_stats.tex
  ../TheGreenEpochPaper/assets/tables/grid_stats.tex

Run:  uv run scripts/table_grid_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "public" / "data" / "co2"
OUT_DIR = REPO_ROOT / "publication" / "output" / "tables"
PAPER_TABLES = REPO_ROOT.parent / "TheGreenEpochPaper" / "assets" / "tables"

YEAR = 2025

# zone code -> display name (paper order is by CV, computed below)
REGION_NAMES = {
    "DE": "Germany",
    "IT": "Italy",
    "SE": "Sweden",
    "US": "United States",
    "CN": "China",
}


def region_stats(zone: str) -> dict:
    with open(DATA_DIR / f"{zone}_{YEAR}.json") as f:
        data = json.load(f)
    ci = np.asarray(data["carbonIntensity"], dtype=float)
    ci = ci[np.isfinite(ci)]
    mean = float(ci.mean())
    std = float(ci.std(ddof=1))
    return {"zone": zone, "mean": mean, "std": std, "cv": std / mean}


def render_tabular(stats: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{@{}l S[table-format=3.0] S[table-format=3.2]",
        r"  S[table-format=1.2]@{}}",
        r"  \toprule",
        r"  Region & {Mean} & {Std.\ dev.} & {CV} \\",
        r"  & {(gCO$_2$eq/kWh)} & {(gCO$_2$eq/kWh)} & \\",
        r"  \midrule",
    ]
    for s in stats:
        label = f"{s['zone']} ({REGION_NAMES[s['zone']]})"
        lines.append(
            f"  {label} & {s['mean']:.0f} & {s['std']:.2f} & {s['cv']:.2f} \\\\"
        )
    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    stats = [region_stats(z) for z in REGION_NAMES]
    stats.sort(key=lambda s: s["cv"], reverse=True)
    tabular = render_tabular(stats)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "grid_stats.tex").write_text(tabular, encoding="utf-8")
    print(f"✓ {OUT_DIR / 'grid_stats.tex'}")

    if PAPER_TABLES.parent.is_dir():
        PAPER_TABLES.mkdir(parents=True, exist_ok=True)
        (PAPER_TABLES / "grid_stats.tex").write_text(tabular, encoding="utf-8")
        print(f"✓ {PAPER_TABLES / 'grid_stats.tex'}")

    print(f"\n{YEAR} grid carbon intensity (ordered by CV):")
    for s in stats:
        print(f"  {s['zone']}: mean {s['mean']:.1f}  std {s['std']:.2f}  CV {s['cv']:.2f}")


if __name__ == "__main__":
    main()
