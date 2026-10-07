#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────
# Run the full publication experiment matrix for several overhead
# budgets and keep each budget in its own output folder:
#
#   publication/output/budget_<B>/{opt_*.json, results/*.csv, run.log}
#
# Budget 200 is skipped by default because publication/output/results
# already contains the published B=200% run (pass it explicitly to
# regenerate).
#
# Usage:
#   bash publication/run_budget_sweep.sh              # 10 25 50 100
#   bash publication/run_budget_sweep.sh 100 200      # explicit list
# ───────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_OUT="$ROOT/publication/output"

if [[ $# -gt 0 ]]; then
  BUDGETS=("$@")
else
  BUDGETS=(10 25 50 100)
fi

for B in "${BUDGETS[@]}"; do
  OUT="$BASE_OUT/budget_$B"
  mkdir -p "$OUT"
  echo ""
  echo "═══ Budget sweep: B=${B}% → $OUT ═══"
  BUDGET="$B" OUT_DIR="$OUT" bash "$ROOT/publication/run_experiments.sh" 2>&1 | tee "$OUT/run.log"
done

echo ""
echo "═══ SWEEP DONE ═══"
for B in "${BUDGETS[@]}"; do
  n=$(ls "$BASE_OUT/budget_$B/results"/*.csv 2>/dev/null | wc -l | tr -d ' ')
  echo "  budget_$B: $n CSV files"
done
