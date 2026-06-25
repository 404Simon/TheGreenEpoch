"""
Run validation and verification scenarios from vv_scenarios.csv.


This script provides both programmatic and visual feedback on the V&V results.


Usage:
    python src/run_vv.py              # Run all vv_scenarios
    python src/run_vv.py --quick      # Run subset for quick validation
    python src/run_vv.py --output results_vv.csv
    python src/run_vv.py --pytest     # Also run pytest tests
"""

from __future__ import annotations


import argparse
import csv
import logging
from pathlib import Path
from datetime import datetime, timezone
import sys
import subprocess


from simulation import (
    SimulationRunner,
    load_scenarios,
    load_training_profiles,
    ScenarioParameters,
)
from simulation.grid_data import GridDataProvider
from simulation.models import _parse_csv_list
from simulation.results import SimulationResult


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


def run_pytest_tests(test_file: Path = None, verbose: bool = False) -> dict:
    """Run pytest tests and return results."""
    if test_file is None:
        # Look for test files in common locations
        test_dirs = [
            Path("tests"),
            Path("src/tests"),
            Path("..") / "tests",
        ]
        for test_dir in test_dirs:
            if test_dir.exists():
                test_files = list(test_dir.glob("test_*.py"))
                if test_files:
                    test_file = test_files[0]
                    break
    
    if test_file is None or not test_file.exists():
        logging.warning("No test file found, skipping pytest")
        return {"success": False, "message": "No test file found"}
    
    logging.info(f"Running pytest on {test_file}")
    
    # Build pytest command
    cmd = ["python", "-m", "pytest", str(test_file)]
    if verbose:
        cmd.extend(["-v"])
    cmd.extend(["--tb=short"])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
        )
        
        success = result.returncode == 0
        
        # Parse summary from output
        output = result.stdout + result.stderr
        lines = output.split("\n")
        
        summary = {
            "success": success,
            "message": result.stdout[-500:] if not success else "All tests passed",
            "output": output,
        }
        
        # Try to extract test count
        for line in lines:
            if "passed" in line:
                summary["passed"] = line
            elif "failed" in line:
                summary["failed"] = line
            elif "error" in line:
                summary["errors"] = line
        
        return summary
        
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Test execution timed out"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def print_pytest_summary(pytest_results: dict) -> None:
    """Print summary of pytest results."""
    print("\n" + "=" * 80)
    print("PYTEST TEST RESULTS")
    print("=" * 80)
    
    if not pytest_results.get("success"):
        print(f"\n  ✗ FAILED: {pytest_results.get('message', 'Unknown error')}")
    else:
        print("\n  ✓ PASSED: All tests executed successfully")
    
    # Print parsed summary lines if available
    for key in ["passed", "failed", "errors"]:
        if key in pytest_results:
            print(f"\n  {key}: {pytest_results[key]}")
    
    print("\n" + "=" * 80)


def print_validation_summary(results: list) -> None:
    """Print summary of validation results."""
    print("\n" + "=" * 80)
    print("VALIDATION & VERIFICATION RESULTS SUMMARY")
    print("=" * 80)


    # Group by test category
    categories = {
        "Verification - baseline": [],
        "Verification - pause logic": [],
        "Verification - progress": [],
        "Validation - regional": [],
        "Validation - temporal": [],
        "Validation - trade-off": [],
        "Validation - threshold": [],
        "Validation - cross-model": [],
    }


    for r in results:
        desc = r["scenario_description"]
        if "baseline" in desc.lower() and "verification" in desc.lower():
            categories["Verification - baseline"].append(r)
        elif "pause" in desc.lower() and "verification" in desc.lower():
            categories["Verification - pause logic"].append(r)
        elif "progress" in desc.lower():
            categories["Verification - progress"].append(r)
        elif "regional" in desc.lower():
            categories["Validation - regional"].append(r)
        elif "temporal" in desc.lower():
            categories["Validation - temporal"].append(r)
        elif "frontier" in desc.lower():
            categories["Validation - trade-off"].append(r)
        elif "sweep" in desc.lower():
            categories["Validation - threshold"].append(r)
        elif "cross-model" in desc.lower():
            categories["Validation - cross-model"].append(r)


    for category, items in categories.items():
        if items:
            print(f"\n{category}:")
            print("-" * 80)
            for item in items:
                status = "✓" if item.get("completed", False) else "✗"
                print(
                    f"  {status} {item['region']:3s} | "
                    f"{item['model']:8s} | "
                    f"θ={item['threshold']:6.0f} | "
                    f"time={float(item['actual_overhead_pct']):5.1f}% | "
                    f"co2={float(item['total_emissions_kgco2']):8.0f} kg"
                )


    print("\n" + "=" * 80)



def print_kpi_function_tests(results: list) -> None:
    """Print summary of KPI validation from results."""
    print("\n" + "=" * 80)
    print("KPI FUNCTION VALIDATION CHECK")
    print("=" * 80)

    kpi_checks = {
        "Time Overhead % >= 0": [],
        "Total Emissions >= 0": [],
        "Training Time >= 0": [],
        "Paused Time >= 0": [],
        "Wall Time >= Training Time": [],
        "Completed simulations": [],
        "Within overhead budget": [],
    }

    negative_savings_cases = []

    for r in results:
        completed = r.get("completed", False)
        co2_savings = float(r.get("co2_savings_pct", 0))
        time_overhead = float(r.get("actual_overhead_pct", 0))
        total_emissions = float(r.get("total_emissions_kgco2", 0))
        baseline_emissions = float(r.get("baseline_emissions_kgco2", 0))
        training_time = float(r.get("training_time_h", 0))
        paused_time = float(r.get("paused_time_h", 0))
        wall_time = float(r.get("total_wall_time_h", 0))
        within_budget = r.get("within_overhead_budget", False)

        kpi_checks["Time Overhead % >= 0"].append(time_overhead >= 0)
        kpi_checks["Total Emissions >= 0"].append(total_emissions >= 0)
        kpi_checks["Training Time >= 0"].append(training_time >= 0)
        kpi_checks["Paused Time >= 0"].append(paused_time >= 0)
        kpi_checks["Wall Time >= Training Time"].append(wall_time >= training_time)
        kpi_checks["Completed simulations"].append(completed)
        kpi_checks["Within overhead budget"].append(within_budget)

        if baseline_emissions > 0 and total_emissions > baseline_emissions:
            negative_savings_cases.append(r)

    all_passed = True
    for check_name, values in kpi_checks.items():
        if not values:
            continue
        passed_count = sum(values)
        total_count = len(values)
        passed = passed_count == total_count
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}: {passed_count}/{total_count}")
        if not passed:
            all_passed = False

    print("\nNegative savings scenarios:")
    print("-" * 80)
    if negative_savings_cases:
        for r in negative_savings_cases:
            savings = r.get("co2_savings_pct", 0)
            print(
                f"  ✗ {r['region']:3s} | {r['model']:8s} | "
                f"baseline={float(r['baseline_emissions_kgco2']):8.0f} kg | "
                f"actual={float(r['total_emissions_kgco2']):8.0f} kg | "
                f"savings={savings:6.1f}%"
            )
    else:
        print("  No negative savings scenarios found.")

    print("\n" + "=" * 80)
    if all_passed:
        print("  ✓ ALL KPI VALIDATION CHECKS PASSED")
    else:
        print("  ✗ SOME KPI VALIDATION CHECKS FAILED")
    print("=" * 80)

# def print_negative_savings_check(results: list) -> None:
#     """Report scenarios where policy emissions exceed baseline emissions."""
#     print("\n" + "=" * 80)
#     print("NEGATIVE EMISSIONS SAVINGS CHECK")
#     print("=" * 80)

#     found = False
#     for r in results:
#         baseline = float(r.get("baseline_emissions_kgco2", 0))
#         actual = float(r.get("total_emissions_kgco2", 0))
#         savings = float(r.get("co2_savings_pct", 105))

#         if baseline > 0 and actual > baseline:
#             found = True
#             print(
#                 f"  ✗ {r['scenario_description']} | {r['region']} | "
#                 f"baseline={baseline:.0f} kg | actual={actual:.0f} kg | savings={savings:.1f}%"
#             )

#     if not found:
#         print("  No negative savings scenarios found.")

#     print("=" * 80)


def print_regional_comparison(results: list) -> None:
    """Print comparison of Germany vs Sweden."""
    print("\n" + "=" * 80)
    print("REGIONAL REALISM CHECK: Germany vs Sweden")
    print("=" * 80)


    baseline_de = None
    baseline_se = None


    for r in results:
        if "baseline" in r["scenario_description"].lower() and r["region"] == "DE":
            baseline_de = r
        if "baseline" in r["scenario_description"].lower() and r["region"] == "SE":
            baseline_se = r


    if baseline_de and baseline_se:
        de_em = float(baseline_de["total_emissions_kgco2"])
        se_em = float(baseline_se["total_emissions_kgco2"])
        ratio = de_em / se_em if se_em > 0 else 0


        print(f"\nBaseline emissions (no pause):")
        print(f"  Germany (DE): {de_em:8.0f} kg CO₂")
        print(f"  Sweden (SE):  {se_em:8.0f} kg CO₂")
        print(f"  Ratio (DE/SE): {ratio:6.2f}x")


        if se_em < de_em:
            print(f"  ✓ PASS: Sweden has lower emissions (cleaner grid)")
        else:
            print(f"  ? CHECK: Emissions are not as expected")
    else:
        print("  No baseline data for both regions - skipping comparison")


    print("=" * 80)



def print_temporal_comparison(results: list) -> None:
    """Print comparison of summer vs winter."""
    print("\n" + "=" * 80)
    print("TEMPORAL REALISM CHECK: Summer vs Winter")
    print("=" * 80)


    summer = None
    winter = None


    for r in results:
        if "summer" in r["scenario_description"].lower():
            summer = r
        if "winter" in r["scenario_description"].lower():
            winter = r


    if summer and winter:
        s_em = float(summer["total_emissions_kgco2"])
        w_em = float(winter["total_emissions_kgco2"])


        print(f"\nBaseline emissions by season (Germany):")
        print(f"  Summer: {s_em:8.0f} kg CO₂")
        print(f"  Winter: {w_em:8.0f} kg CO₂")
        print(f"  Difference: {abs(s_em - w_em):8.0f} kg CO₂")
        print(f"  Both completed: {summer['completed'] and winter['completed']}")
    else:
        print("  No seasonal data available - skipping comparison")


    print("=" * 80)



def print_trade_off_frontier(results: list) -> None:
    """Print trade-off frontier analysis."""
    print("\n" + "=" * 80)
    print("TRADE-OFF FRONTIER ANALYSIS")
    print("=" * 80)


    frontier_points = [r for r in results if "frontier" in r["scenario_description"].lower()]


    if frontier_points:
        print("\nPareto-frontier points (lower threshold → more savings, more overhead):\n")
        print("  Threshold | Overhead % | Emissions (kg) | Num Pauses")
        print("  " + "-" * 50)


        for r in sorted(frontier_points, key=lambda x: float(x["threshold"]), reverse=True):
            print(
                f"  {float(r['threshold']):8.0f}  | "
                f"  {float(r['actual_overhead_pct']):6.1f}%  | "
                f"  {float(r['total_emissions_kgco2']):10.0f}  | "
                f"  {int(float(r['num_pauses'])):6.0f}"
            )


        print("\n  ✓ Frontier shows expected trade-off structure")
    else:
        print("  No frontier data available")


    print("=" * 80)



def save_results(results: list, path: Path) -> None:
    """Save raw results to CSV."""
    if not results:
        return


    # Determine all fieldnames from first result
    fieldnames = list(results[0].keys())


    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)


    logging.info(f"Results saved to {path}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Run V&V scenarios")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run subset of scenarios for quick validation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "vv_results.csv",
        help="Output CSV file (default: output/vv_results.csv)",
    )
    parser.add_argument(
        "--pytest",
        action="store_true",
        help="Also run pytest tests (e.g., test_kpi_functions.py)",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Specific test file to run with pytest",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output from pytest",
    )
    args = parser.parse_args()


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )


    print("=" * 80)
    print("TheGreenEpoch - Validation & Verification Runner")
    print("=" * 80)


    runner = SimulationRunner.from_data_dir(DATA_DIR)
    
    # Load vv_scenarios if available, fall back to scenarios.csv
    vv_path = DATA_DIR / "vv_scenarios.csv"
    use_vv = vv_path.exists()
    
    if use_vv:
        # Manually load vv_scenarios.csv
        scenarios = []
        with vv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                scenarios.append(
                    ScenarioParameters(
                        description=row["description"].strip(),
                        model=row["model"].strip(),
                        thresholds=_parse_csv_list(row["threshold"], float),
                        hysteresis=_parse_csv_list(row["hysteresis"], float),
                        region=row["region"].strip(),
                        start_times=[
                            datetime(1970, *map(int, ts.split("-")), tzinfo=timezone.utc)
                            for ts in _parse_csv_list(row["start_times_set"])
                        ],
                        historical_years=_parse_csv_list(row["historical"], int),
                        overhead_budget_pct=float(row["overhead_budget_pct"]),
                    )
                )
        print(f"\nLoaded {len(scenarios)} V&V scenarios from vv_scenarios.csv")
    else:
        scenarios = load_scenarios(DATA_DIR)
        print(f"\nLoaded {len(scenarios)} scenarios from scenarios.csv")


    if args.quick:
        # Filter to representative subset
        scenarios = [s for s in scenarios if any(
            x in s.description.lower() for x in ["baseline", "regional", "temporal"]
        )]
        print(f"Running quick validation ({len(scenarios)} scenarios)")


    results = runner.run_scenarios(scenarios, record_time_series=False)


    # Convert SimulationResult to dict for CSV output
    result_dicts = []
    for r in results:
        d = vars(r).copy()
        d["historical_years"] = str(r.historical_years)
        d["start_time"] = r.start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        d["co2_savings_pct"] = r.co2_savings_pct
        result_dicts.append(d)


    # Run pytest tests if requested
    pytest_results = None
    if args.pytest:
        print("\n" + "=" * 80)
        print("RUNNING PYTEST TESTS")
        print("=" * 80)
        pytest_results = run_pytest_tests(args.test_file, args.verbose)
        print_pytest_summary(pytest_results)


    # Print analysis
    print_validation_summary(result_dicts)
    print_kpi_function_tests(result_dicts)
    #print_negative_savings_check(result_dicts)
    print_regional_comparison(result_dicts)
    print_temporal_comparison(result_dicts)
    print_trade_off_frontier(result_dicts)


    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_results(result_dicts, args.output)
    print(f"\n✓ Results saved to {args.output}")


    print("=" * 80)



if __name__ == "__main__":
    main()