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
import json
import logging
import time
from pathlib import Path

from simulation import SimulationRunner, filter_scenarios, load_scenarios

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


def save_results(results, path: Path) -> None:
    fields = [
        "timeseries_id",
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
        "baseline_emissions_kgco2",
        "baseline_time_h",
        "co2_savings_pct",
        "score",
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
        for i, r in enumerate(results):
            ts_id = str(i)
            d = vars(r).copy()
            d["timeseries_id"] = ts_id
            d["historical_years"] = str(r.historical_years)
            d["start_time"] = r.start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            d["co2_savings_pct"] = r.co2_savings_pct
            d["score"] = r.score
            w.writerow(d)


def save_timeseries(results, ts_dir: Path, max_points: int = 3200) -> None:
    ts_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(results):
        n = len(r.timestamps)
        if n > max_points:
            step = n / max_points
            idx = [int(i * step) for i in range(max_points)]
        else:
            idx = list(range(n))
        data = {
            "scenario_description": r.scenario_description,
            "model": r.model,
            "region": r.region,
            "historical_years": r.historical_years,
            "start_time": r.start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "theta_pause": r.threshold,
            "theta_resume": r.hysteresis_margin,
            "total_wall_time_h": r.total_wall_time_h,
            "training_time_h": r.training_time_h,
            "paused_time_h": r.paused_time_h,
            "num_pauses": r.num_pauses,
            "actual_overhead_pct": r.actual_overhead_pct,
            "co2_savings_pct": r.co2_savings_pct,
            "score": r.score,
            "completed": r.completed,
            "stop_reason": r.stop_reason,
            "timestamps": [r.timestamps[i].strftime("%Y-%m-%dT%H:%M:%SZ") for i in idx],
            "carbon_intensity": [r.carbon_intensity_series[i] for i in idx],
            "state": [r.state_series[i] for i in idx],
            "emissions_kg": [r.emissions_series[i] / 1000.0 for i in idx],
            "tokens_remaining": [r.tokens_remaining_series[i] for i in idx],
        }
        (ts_dir / f"{i}.json").write_text(json.dumps(data, separators=(",", ":")))


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
        format="%(asctime)s [%(levelname)-4s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 72)
    print("  TheGreenEpoch Pilot Simulation Run")
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
    results = runner.run_scenarios(scenarios, record_time_series=True)
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
    print(f"\n  {'─' * 82}")
    print(
        f"  {'REGION':<8} {'THR':>4} {'RES':>4} {'PAUSES':>7} "
        f"{'OVERH':>6} {'CO2%':>7} {'SCORE':>7} "
        f"{'WALL(h)':>8}  DETAIL"
    )
    print(f"  {'─' * 82}")
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(
            f"  {r.region:<8} {r.threshold:>4.0f} {r.hysteresis_margin:>4.0f} "
            f"{r.num_pauses:>7} {r.actual_overhead_pct:>5.1f}% "
            f"{r.co2_savings_pct:>6.1f}% {r.score:>7.1f} "
            f"{r.total_wall_time_h:>8.1f}  "
            f"{status}  {r.scenario_description[:38]:38s}"
        )
        for issue in r.issues[:2]:
            print(f"  {'':<84} ⚠ {issue}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "results.csv"
    save_results(results, csv_path)
    print(f"\n  Results saved → {csv_path}")

    ts_dir = OUTPUT_DIR / "timeseries"
    save_timeseries(results, ts_dir)
    print(f"  Time series saved → {ts_dir}/")
    print(f"  {'=' * 68}")


if __name__ == "__main__":
    main()
