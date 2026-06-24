"""Pre-process CSV data files into compact JSON for the web app."""

import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "public" / "data"


def load_constants():
    with (DATA_DIR / "datacenter.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        return {row["variable"]: float(row["report_value"]) for row in reader}


def load_profiles():
    profiles = {}
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        if csv_path.name in ("datacenter.csv", "scenarios.csv"):
            continue
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            params = {row["variable"]: row["report_value"] for row in reader}
        name = csv_path.stem.replace("_", " ").title()
        name = (
            __import__("re")
            .sub(r"[_-]?[vk]\d+$", "", name, flags=__import__("re").IGNORECASE)
            .strip()
        )
        profiles[name] = {
            "name": name,
            "modelParams": _parse_int(params.get("model_params", "0")),
            "datasetTokens": _parse_int(params.get("dataset_tokens", "0")),
            "gpuCount": _parse_int(params.get("gpu_count", "0")),
        }
    return profiles


def load_scenarios():
    scenarios = []
    with (DATA_DIR / "scenarios.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            scenarios.append(
                {
                    "description": row["description"].strip(),
                    "model": row["model"].strip(),
                    "thresholds": _parse_csv_floats(row["threshold"]),
                    "hysteresis": _parse_csv_floats(row["hysteresis"]),
                    "region": row["region"].strip(),
                    "startTimes": [
                        s.strip()
                        for s in row["start_times_set"].split(",")
                        if s.strip()
                    ],
                    "historicalYears": _parse_csv_ints(row["historical"]),
                    "overheadBudgetPct": float(row["overhead_budget_pct"]),
                }
            )
    return scenarios


def load_co2_data(zone: str, year: int):
    path = DATA_DIR / "co2_intensity" / zone / f"carbon_intensity_{year}.csv"
    if not path.exists():
        return None
    timestamps = []
    values = []
    is_estimated = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(row["timestamp"])
            values.append(float(row["carbonIntensity"]))
            is_estimated.append(row["isEstimated"].strip().lower() == "true")
    return {
        "zone": zone,
        "year": year,
        "timestamps": timestamps,
        "values": values,
        "isEstimated": is_estimated,
    }


def build_timeline(zone: str, years: list[int]):
    """Average multiple years of CO2 data by aligning timestamps."""
    all_data = {}
    for y in years:
        data = load_co2_data(zone, y)
        if data is None:
            continue
        all_data[y] = data

    if not all_data:
        return None

    canonical = min(all_data.keys())

    # Group by (year, shifted_timestamp) - first average within-year duplicates
    # (e.g. leap-year Feb 29 → Feb 28 collides with original Feb 28).
    by_year_ts: dict[tuple[int, str], list[float]] = defaultdict(list)
    for year, data in all_data.items():
        shift = canonical - year
        for ts, val in zip(data["timestamps"], data["values"]):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            try:
                shifted = dt.replace(year=dt.year + shift)
            except ValueError:
                shifted = dt.replace(year=dt.year + shift, day=28)
            key = shifted.strftime("%Y-%m-%dT%H:%M:%SZ")
            by_year_ts[(year, key)].append(val)

    # Average within-year duplicates, then group by shifted timestamp across years
    groups: dict[str, list[float]] = defaultdict(list)
    for (year, key), vals in by_year_ts.items():
        groups[key].append(sum(vals) / len(vals))

    # Only keep timestamps present in ALL years (one value per year)
    n_years = len(all_data)
    timestamps = []
    values = []
    for ts in sorted(groups.keys()):
        vals = groups[ts]
        if len(vals) == n_years:
            timestamps.append(ts)
            values.append(sum(vals) / n_years)

    return {
        "zone": zone,
        "years": years,
        "timestamps": timestamps,
        "carbonIntensity": values,
    }


def _parse_int(raw: str) -> int:
    raw = raw.strip()
    multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    for suffix, mult in multipliers.items():
        if raw.endswith(suffix):
            return int(float(raw[:-1]) * mult)
    return int(float(raw))


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(p.strip()) for p in raw.split(",") if p.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(p.strip()) for p in raw.split(",") if p.strip()]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Constants
    constants = load_constants()
    (OUT_DIR / "constants.json").write_text(json.dumps(constants, indent=2))
    print(f"constants.json ({len(constants)} keys)")

    # Profiles
    profiles = load_profiles()
    (OUT_DIR / "profiles.json").write_text(json.dumps(profiles, indent=2))
    print(f"profiles.json ({len(profiles)} profiles)")

    # Scenarios
    scenarios = load_scenarios()
    (OUT_DIR / "scenarios.json").write_text(json.dumps(scenarios, indent=2))
    print(f"scenarios.json ({len(scenarios)} scenarios)")

    # CO2 data per zone
    zones = ["DE", "SE", "CN", "IT", "US"]
    years = [2022, 2023, 2024, 2025]
    co2_dir = OUT_DIR / "co2"
    co2_dir.mkdir(exist_ok=True)
    for zone in zones:
        tl = build_timeline(zone, years)
        if tl:
            (co2_dir / f"{zone}.json").write_text(json.dumps(tl))
            print(f"co2/{zone}.json ({len(tl['timestamps'])} points)")
        else:
            print(f"co2/{zone}.json - NO DATA")

    print("\nDone. Files in", OUT_DIR)


if __name__ == "__main__":
    main()
