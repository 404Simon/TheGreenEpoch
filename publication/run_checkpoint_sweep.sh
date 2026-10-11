#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────
# Checkpoint-cost sweep (Deepseek only) across several overhead
# budgets. Each budget gets its own folder:
#
#   publication/output/checkpoint_sweep/budget_<B>/{checkpoint_sweep.json, checkpoint_sweep.csv, run.log}
#
# Usage:
#   bash publication/run_checkpoint_sweep.sh              # 10 25 50 100 200
#   bash publication/run_checkpoint_sweep.sh 100 200      # explicit list
#   COSTS=0,148.8 bash publication/run_checkpoint_sweep.sh
# ───────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BASE="$ROOT/publication/output/checkpoint_sweep"
PNPM="pnpm"

if [[ $# -gt 0 ]]; then
  BUDGETS=("$@")
else
  BUDGETS=(10 25 50 100 200)
fi

# Checkpoint costs in seconds (save = load), incl. 0 baseline and
# 148.8 s ≈ the original published single-measurement case.
COSTS="${COSTS:-0,1,10,30,60,148.8,300,600}"

# Optimizer settings (publication defaults, see run_experiments.sh)
MODEL="Deepseek"
REGION="${REGION:-DE}"
YEARS="${YEARS:-2025}"
TP_MAX=800
RES=10
DATE_RES=7
MAX_ITER=10
ALPHA=1

mkdir -p "$OUT_BASE"

for B in "${BUDGETS[@]}"; do
  OUT="$OUT_BASE/budget_$B"
  mkdir -p "$OUT"
  JSON="$OUT/checkpoint_sweep.json"
  CSV="$OUT/checkpoint_sweep.csv"
  POINTS="$OUT/checkpoint_points.csv"
  LOG="$OUT/run.log"

  echo ""
  echo "═══ Checkpoint sweep: B=${B}% → $OUT ═══"

  "$PNPM" cli checkpoint-sweep \
    -m "$MODEL" -r "$REGION" -y "$YEARS" \
    --costs "$COSTS" \
    --tp-max "$TP_MAX" --budget "$B" \
    --resolution "$RES" --date-res "$DATE_RES" \
    --max-iter "$MAX_ITER" --alpha "$ALPHA" \
    -o "$JSON" --csv "$CSV" --points "$POINTS" 2>&1 | tee "$LOG"
done

echo ""
echo "═══ CHECKPOINT SWEEP DONE ═══"
for B in "${BUDGETS[@]}"; do
  f="$OUT_BASE/budget_$B/checkpoint_sweep.csv"
  if [[ -f "$f" ]]; then
    echo "  budget_$B: $(grep -c '' "$f") Zeilen (inkl. Header)"
  else
    echo "  budget_$B: MISSING"
  fi
done
