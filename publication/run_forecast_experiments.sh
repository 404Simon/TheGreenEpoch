#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────
# Forecast-Error Sensitivity Study
#
# Runs the full pipeline end-to-end:
#   1. forecast-calibrate          → publication/output/forecast/calibration_{region}.{json,csv} + README.md
#   2. forecast-sweep --mode fixed → fixed_{region}.{json,csv}, fixed_all.csv, fixed_summary.json
#                                    (600 simulations, deterministic)
#   3. forecast-sweep --mode reopt → reopt_{region}.{json,csv}, reopt_all.csv, reopt_summary.json
#                                    (54 optimizations, deterministic)
#   4. plot-forecast               → publication/ICREC_Rome/assets/forecast_*.{svg,eps} (4 figures)
#
# Study constants (kept in sync with the CLI defaults / SPEC):
#   fixed sweep seeds: DE = 10, IT/SE = 5   (reopt: 3 per region)
#   fixed policies:    DE 272/268 @ 02-01, IT 246/231 @ 01-14, SE 19/18 @ 04-22
#   budget 200%, alpha = 1
#
# All output (stdout + stderr) is appended to publication/experiments_run.log
# behind a timestamped separator while still streaming to the terminal.
# Idempotent: every run overwrites the same deterministic artifacts.
# ───────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/publication/output/forecast"
LOG="$ROOT/publication/experiments_run.log"
PNPM="pnpm"

mkdir -p "$OUT_DIR"

started="$(date -u +%FT%TZ)"
SECONDS=0

section() {
  printf '\n── %s ──\n' "$1"
}

{
  printf '\n==== %s ==== run_forecast_experiments.sh (calibrate → fixed → reopt → plots)\n' "$started"

  cd "$ROOT"

  section "Step 1/4: forecast-calibrate (persistence/AR baselines, DE/IT/SE)"
  "$PNPM" cli forecast-calibrate

  section "Step 2/4: fixed-policy sweep (600 simulations, DE=10 seeds, IT/SE=5)"
  "$PNPM" cli forecast-sweep --mode fixed \
    -o "$OUT_DIR/fixed" \
    --csv "$OUT_DIR/fixed_all.csv"

  section "Step 3/4: re-optimization sweep (54 optimizations, 3 seeds/region)"
  "$PNPM" cli forecast-sweep --mode reopt \
    -o "$OUT_DIR/reopt" \
    --csv "$OUT_DIR/reopt_all.csv"

  section "Step 4/4: plot-forecast (SVG + EPS figures)"
  "$PNPM" cli plot-forecast

  printf '\n==== done in %s ===== %d s elapsed ====\n' "$(date -u +%FT%TZ)" "$SECONDS"
} 2>&1 | tee -a "$LOG"
