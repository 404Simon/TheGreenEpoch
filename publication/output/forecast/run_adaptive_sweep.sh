#!/usr/bin/env bash
# Deterministic stale-aware adaptive-controller sweep (Phase B.1).
# Runs `forecast-sweep --mode adaptive` for DE/IT/SE (test year 2025, train
# years 2022-2024, c-grid 0..2, horizons 1..72) and writes
#   publication/output/forecast/adaptive_{DE,IT,SE}.json
#   publication/output/forecast/adaptive_{DE,IT,SE}.csv
#   publication/output/forecast/adaptive_summary.json
# The path is deterministic (arma/identity/delay decision models and the
# optimizer have no RNG): running twice yields byte-identical artifacts.
# Set ADAPTIVE_DETERMINISM_CHECK=1 to snapshot + re-run + cmp automatically.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../../.." && pwd)"
OUT="$DIR/adaptive"
SNAP="/tmp/adaptive_determinism_snap"
mkdir -p "$SNAP"

echo "== Running stale-aware adaptive sweep (Phase B.1) =="
(cd "$ROOT" && pnpm cli forecast-sweep --mode adaptive -o "$OUT")

if [ "${ADAPTIVE_DETERMINISM_CHECK:-0}" = "1" ]; then
  echo "== Determinism check: snapshot -> re-run -> cmp =="
  cp "$OUT"_DE.json "$OUT"_IT.json "$OUT"_SE.json "$OUT"_DE.csv "$OUT"_IT.csv "$OUT"_SE.csv "$OUT"_summary.json "$SNAP"/
  (cd "$ROOT" && pnpm cli forecast-sweep --mode adaptive -o "$OUT" --quiet)
  ok=1
  for f in _DE.json _IT.json _SE.json _DE.csv _IT.csv _SE.csv _summary.json; do
    if cmp -s "$SNAP/adaptive$f" "$OUT$f"; then
      echo "  IDENTICAL: adaptive$f"
    else
      echo "  DIFFER: adaptive$f"
      ok=0
    fi
  done
  [ "$ok" = "1" ] && echo "== Determinism check PASSED ==" || { echo "== Determinism check FAILED =="; exit 1; }
fi

echo "== Done. Artifacts in $DIR =="
