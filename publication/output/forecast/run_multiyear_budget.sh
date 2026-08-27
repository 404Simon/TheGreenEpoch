#!/usr/bin/env bash
# Deterministic multi-year robustness + overhead-budget sweep (Phase B.3 + B.4).
#
#   Step 1 (B.3.1): re-optimize DE/IT/SE on each year 2022..2025 (DeepSeek,
#                   budget 200%, alpha 1, resolution 10, iterations 6, fixed
#                   per-region start, tpMax 800/800/100, ckpt = constants 148.8s)
#   Step 2 (B.3.2): mirror the committed `forecast-sweep --mode fixed` config
#                   (default families/levels/horizons/seeds) on test years 2025
#                   (reproduces fixed_summary.json), 2024 and 2023; the decision
#                   models use the 2025 calibration bundle as-is (documented).
#   Step 3 (B.4.1): re-run the 2025 headline optimization for B in {30,50,100,200}%
#                   at ckpt-pause 148.8s (Story A verdict checkpoint) and, as the
#                   900s robustness block, at ckpt-pause 900s.
#
# Aggregates (node) into:
#   multiyear_{DE,IT,SE}.{json,csv}, multiyear_summary.json,
#   multiyear_fixed_summary.json, budget_summary.{json,csv}
# No RNG in the optimizer/decision paths -> running twice yields byte-identical
# artifacts (verify with cmp). Set DETERMINISM_CHECK=1 to auto-verify.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../../.." && pwd)"
RAW="$DIR/_raw"

run_all() {
  mkdir -p "$RAW/multiyear" "$RAW/budget" "$RAW/fixed_2025" "$RAW/fixed_2024" "$RAW/fixed_2023"

  echo "== Phase B.3.1: multi-year re-optimization (12 runs) =="
  for spec in "DE 02-01 800" "IT 01-14 800" "SE 04-22 100"; do
    set -- $spec
    REGION=$1; START=$2; TP_MAX=$3
    for YEAR in 2022 2023 2024 2025; do
      echo "  [$REGION] year=$YEAR (start=$START, tpMax=$TP_MAX)"
      (cd "$ROOT" && pnpm cli optimize -m Deepseek -r "$REGION" -y "$YEAR" \
        --start "$START" --tp-max "$TP_MAX" \
        --budget 200 --resolution 10 --max-iter 6 \
        -o "$RAW/multiyear/${REGION}_${YEAR}.json") > /dev/null
    done
  done

  echo "== Phase B.3.2: fixed-policy sweep on test years 2025/2024/2023 =="
  for YEAR in 2025 2024 2023; do
    echo "  fixed sweep year=$YEAR"
    (cd "$ROOT" && pnpm cli forecast-sweep --mode fixed \
      -o "$RAW/fixed_$YEAR/fixed" -y "$YEAR" --csv "$RAW/fixed_$YEAR/fixed_all.csv") > /dev/null
  done

  echo "== Phase B.4.1: overhead-budget sweep at ckpt 148.8s and 900s (24 runs) =="
  for spec in "DE 02-01 800" "IT 01-14 800" "SE 04-22 100"; do
    set -- $spec
    REGION=$1; START=$2; TP_MAX=$3
    for BUDGET in 30 50 100 200; do
      for CKPT in 148.8 900; do
        echo "  [$REGION] budget=${BUDGET}% ckpt-pause=${CKPT}s"
        (cd "$ROOT" && pnpm cli optimize -m Deepseek -r "$REGION" -y 2025 \
          --start "$START" --tp-max "$TP_MAX" \
          --budget "$BUDGET" --resolution 10 --max-iter 6 \
          --ckpt-pause "$CKPT" --ckpt-resume 0 \
          -o "$RAW/budget/${REGION}_${BUDGET}_${CKPT}.json") > /dev/null
      done
    done
  done

  echo "== Aggregating artifacts =="
  node "$DIR/_aggregate_multiyear_budget.mjs" "$RAW" "$DIR"
}

run_all

if [ "${DETERMINISM_CHECK:-0}" = "1" ]; then
  echo "== Determinism check: snapshot -> re-run -> cmp =="
  SNAP="/tmp/multiyear_budget_determinism_snap"
  rm -rf "$SNAP" && mkdir -p "$SNAP"
  for f in multiyear_DE.json multiyear_IT.json multiyear_SE.json \
           multiyear_DE.csv multiyear_IT.csv multiyear_SE.csv \
           multiyear_summary.json multiyear_fixed_summary.json \
           budget_summary.json budget_summary.csv; do
    cp "$DIR/$f" "$SNAP/"
  done
  rm -rf "$RAW"
  run_all
  ok=1
  for f in multiyear_DE.json multiyear_IT.json multiyear_SE.json \
           multiyear_DE.csv multiyear_IT.csv multiyear_SE.csv \
           multiyear_summary.json multiyear_fixed_summary.json \
           budget_summary.json budget_summary.csv; do
    if cmp -s "$SNAP/$f" "$DIR/$f"; then
      echo "  IDENTICAL: $f"
    else
      echo "  DIFFER: $f"
      ok=0
    fi
  done
  [ "$ok" = "1" ] && echo "== Determinism check PASSED ==" || { echo "== Determinism check FAILED =="; exit 1; }
fi

echo "== Done. Artifacts in $DIR =="
