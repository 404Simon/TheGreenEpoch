#!/usr/bin/env bash
# e-Energy 2027 — deterministic end-to-end experiment runner.
# Phase E finalizes this script (seeds, commands, runtimes, commit hash).
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "[1/5] checkpoint-realism sweep (B.0)"
bash publication/output/checkpoint/run_checkpoint_sweep.sh

echo "[2/5] adaptive controller + sensitivity (B.1)"
bash publication/output/forecast/run_adaptive_sweep.sh
bash publication/output/forecast/run_adaptive_sensitivity.sh

echo "[3/5] multi-year + budget (B.3, B.4)"
bash publication/output/forecast/run_multiyear_budget.sh

echo "[4/5] grace-horizon validation (B.5)"
node publication/output/forecast/grace_horizon_analysis.mjs

echo "[5/5] tests"
pnpm test

echo "All experiments completed."
