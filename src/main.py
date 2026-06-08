"""Pilot run: CO2-aware LLM training simulation.

Runs simulation scenarios with available CO2 intensity data and
prints a summary table + CSV output.

Usage:
    python src/main.py                           # fast: 2 representative scenarios
    python src/main.py --all                     # all available scenarios (slow)
    python src/main.py --limit N                 # first N scenarios only
    python src/main.py --gui                     # interactive simulation GUI
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

from simulation import ScenarioParameters, SimulationRunner, load_scenarios

# ── data availability ───────────────────────────────────────────────
AVAILABLE_ZONES = {"DE", "SE", "FR", "IT", "ES"}
AVAILABLE_YEARS = {2024, 2025}
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


# ── helpers ─────────────────────────────────────────────────────────


def filter_scenarios(
    scenarios: list[ScenarioParameters],
    limit: int | None = None,
) -> list[ScenarioParameters]:
    """Keep scenarios with available zone data and years."""
    filtered: list[ScenarioParameters] = []
    for sc in scenarios:
        if sc.region not in AVAILABLE_ZONES:
            logging.warning(
                "Skipping '%s': zone '%s' has no CO₂ data", sc.description, sc.region
            )
            continue
        years = sorted(set(sc.historical_years) & AVAILABLE_YEARS)
        if not years:
            years = [sorted(AVAILABLE_YEARS)[0]]
            logging.warning(
                "No requested years %s for '%s', using %s",
                sc.historical_years,
                sc.description,
                years[0],
            )
        filtered.append(
            ScenarioParameters(
                description=sc.description,
                model=sc.model,
                thresholds=sc.thresholds,
                hysteresis=sc.hysteresis,
                region=sc.region,
                start_times=sc.start_times[:1],  # single start time for pilot speed
                historical_years=years,
                overhead_budget_pct=sc.overhead_budget_pct,
            )
        )
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def save_results(results, path: Path) -> None:
    fields = [
        "scenario_description",
        "model",
        "region",
        "historical_years",
        "start_time",
        "threshold",
        "hysteresis_margin",
        "total_wall_time_h",
        "training_time_h",
        "paused_time_h",
        "checkpoint_overhead_h",
        "total_energy_kwh",
        "total_emissions_kgco2",
        "tokens_processed",
        "tokens_total",
        "completed",
        "num_pauses",
        "actual_overhead_pct",
        "within_overhead_budget",
        "stop_reason",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            d = vars(r).copy()
            d["historical_years"] = str(r.historical_years)
            d["start_time"] = r.start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            w.writerow(d)


# ── main ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="TheGreenEpoch pilot simulation run")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available scenarios (default: fast subset)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit to first N scenarios"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch interactive simulation GUI",
    )
    args = parser.parse_args()

    if args.gui:
        from gui import run_gui

        run_gui(DATA_DIR)
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 72)
    print("  TheGreenEpoch — Pilot Simulation Run")
    print("=" * 72)

    runner = SimulationRunner.from_data_dir(DATA_DIR)
    raw = load_scenarios(DATA_DIR)
    print(f"\n  Loaded {len(raw)} scenarios from scenarios.csv")

    limit = args.limit if args.limit else (None if args.all else 2)
    scenarios = filter_scenarios(raw, limit=limit)
    print(f"  Filtered to {len(scenarios)} scenario(s) with available data")

    if not scenarios:
        print("\n  Nothing to run. Exiting.")
        return

    expanded = sum(len(s.thresholds) * len(s.start_times) for s in scenarios)
    print(f"  → {expanded} individual simulation(s)")

    t0 = time.time()
    results = runner.run_scenarios(scenarios, record_time_series=False)
    elapsed = time.time() - t0

    # Summary
    print(f"\n  {'=' * 68}")
    print(f"  RESULTS  ({len(results)} runs in {elapsed:.1f}s)")
    print(f"  {'=' * 68}")
    ok_count = sum(1 for r in results if r.ok)
    ko_count = len(results) - ok_count
    total_em = sum(r.total_emissions_kgco2 for r in results)
    total_en = sum(r.total_energy_kwh for r in results)
    total_pz = sum(r.num_pauses for r in results)
    print(
        f"  ✓ {ok_count:>3}  ✗ {ko_count:<3}  "
        f"emissions={total_em:>10,.0f} kg CO₂  "
        f"energy={total_en:>10,.0f} kWh  "
        f"pauses={total_pz}"
    )

    # Detail table
    print(f"\n  {'─' * 68}")
    print(
        f"  {'REGION':<8} {'THR':>4} {'RES':>4} {'PAUSES':>7} "
        f"{'OVERH':>6} {'EMIS(kg)':>9} {'WALL(h)':>8}  DETAIL"
    )
    print(f"  {'─' * 68}")
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(
            f"  {r.region:<8} {r.threshold:>4.0f} {r.hysteresis_margin:>4.0f} "
            f"{r.num_pauses:>7} {r.actual_overhead_pct:>5.1f}% "
            f"{r.total_emissions_kgco2:>9,.0f} {r.total_wall_time_h:>8.1f}  "
            f"{status}  {r.scenario_description[:38]:38s}"
        )
        for issue in r.issues[:2]:
            print(f"  {'':<70} ⚠ {issue}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "pilot_results.csv"
    save_results(results, csv_path)
    print(f"\n  Results saved → {csv_path}")
    print(f"  {'=' * 68}")


if __name__ == "__main__":
    main()
