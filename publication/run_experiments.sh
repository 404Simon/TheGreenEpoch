#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────
# Run all publication experiments with B=200% and consistent
# optimizer settings. Overwrites publication/output/{opt_*.json, results/*.csv}
# ───────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/publication/output"
CSV_DIR="$OUT_DIR/results"
PNPM="pnpm"
mkdir -p "$CSV_DIR"

# Common flags
BUDGET=200
RES=10
DATE_RES=7
MAX_ITER=10
ALPHA=1
THR_SE=100
THR_OTHER=800

REGIONS=( DE IT SE US CN )

cli() {
  local model=$1 region=$2 years=$3 tpmax=$4
  local json=$5 csv=$6
  shift 6

  local cmd=(
    "$PNPM" cli optimize
    -m "$model" -r "$region" -y "$years"
    --tp-max "$tpmax" --budget "$BUDGET"
    --resolution "$RES" --date-res "$DATE_RES"
    --max-iter "$MAX_ITER" --alpha "$ALPHA"
  )

  echo "  → $model / $region (θₚ-max=$tpmax, budget=$BUDGET)"

  [[ -n "$json" ]] && cmd+=(-o "$json")
  [[ -n "$csv"  ]] && cmd+=(--csv "$csv")
  cmd+=("$@")

  "${cmd[@]}"
}

# Model list: Deepseek first (fast), then Kimi (slow)
run_model() {
  local model_key=$1 code=$2
  local tpmax res_flag

  echo ""
  echo "═══ ${model_key}: ALL START DATES 2025 ═══"
  for region in "${REGIONS[@]}"; do
    tpmax=$THR_OTHER; [[ "$region" == "SE" ]] && tpmax=$THR_SE
    local json="$OUT_DIR/opt_${code}_${region}.json"
    local csv="$CSV_DIR/${code}_${region}_all_2025_${tpmax}_10it.csv"
    cli "$model_key" "$region" "2025" "$tpmax" "$json" "$csv"
  done

  echo ""
  echo "═══ ${model_key}: FIXED START 01-01 2025 ═══"
  for region in "${REGIONS[@]}"; do
    tpmax=$THR_OTHER; [[ "$region" == "SE" ]] && tpmax=$THR_SE
    local csv="$CSV_DIR/${code}_${region}_01_01_2025_${tpmax}_10it.csv"
    cli "$model_key" "$region" "2025" "$tpmax" "" "$csv" --start 01-01
  done

  echo ""
  echo "═══ ${model_key}: FIXED START 07-01 2025 ═══"
  for region in "${REGIONS[@]}"; do
    tpmax=$THR_OTHER; [[ "$region" == "SE" ]] && tpmax=$THR_SE
    local csv="$CSV_DIR/${code}_${region}_07_01_2025_${tpmax}_10it.csv"
    cli "$model_key" "$region" "2025" "$tpmax" "" "$csv" --start 07-01
  done

  echo ""
  echo "═══ ${model_key}: MULTI-YEAR 2022-2025 ═══"
  for region in "${REGIONS[@]}"; do
    tpmax=$THR_OTHER; [[ "$region" == "SE" ]] && tpmax=$THR_SE
    local csv="$CSV_DIR/${code}_${region}_all_2022-2025_${tpmax}_10it_alpha1.csv"
    cli "$model_key" "$region" "2022,2023,2024,2025" "$tpmax" "" "$csv"
  done

  # α-sensitivity: DE only
  if [[ "$code" == "DS" ]]; then
    echo ""
    echo "═══ ${model_key}: ALPHA SENSITIVITY (DE) ═══"
    for av in 0.5 0.8; do
      local csv="$CSV_DIR/${code}_DE_all_2025_${THR_OTHER}_10it_alpha${av//./}.csv"
      echo "  → $model_key / DE (α=$av)"
      "$PNPM" cli optimize \
        -m "$model_key" -r DE -y 2025 \
        --tp-max "$THR_OTHER" --budget "$BUDGET" \
        --resolution "$RES" --date-res "$DATE_RES" \
        --max-iter "$MAX_ITER" --alpha "$av" \
        --csv "$csv" 2>&1 | tail -3
    done
  fi
}

# ── Run ──────────────────────────────────────────────────────

run_model "Deepseek" "DS"
run_model "Kimi" "KM"

echo ""
echo "═══ DONE ═══"
echo "  JSON: $(ls "$OUT_DIR"/opt_*.json 2>/dev/null | wc -l) files"
echo "  CSV:  $(ls "$CSV_DIR"/*.csv 2>/dev/null | wc -l) files"
