#!/usr/bin/env bash
# Deterministic recovery-vs-c sensitivity sweep for the stale-aware adaptive
# controller (Phase B.1, adversarial review M1/M2/m1). Computed on the test
# year 2025 with the extended c grid (0..8) and horizons 1..72, using the SAME
# decision model (arma(h)), reopt nominal anchors, 200% budget and completion
# guard as `--mode adaptive`. Writes
#   publication/output/forecast/adaptive_sensitivity_{DE,IT,SE}.json
#   publication/output/forecast/adaptive_sensitivity_summary.json
# The path is deterministic (arma/identity/delay decision models and the
# optimizer have no RNG): running twice yields byte-identical artifacts.
# Set ADAPTIVE_SENSITIVITY_DETERMINISM_CHECK=1 to snapshot + re-run + cmp.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../../.." && pwd)"
OUT="$DIR/adaptive_sensitivity"
SNAP="/tmp/adaptive_sensitivity_determinism_snap"
mkdir -p "$SNAP"

echo "== Running adaptive recovery-vs-c sensitivity sweep (Phase B.1) =="
(cd "$ROOT" && pnpm cli forecast-sweep --mode adaptive-sensitivity -o "$OUT")

if [ "${ADAPTIVE_SENSITIVITY_DETERMINISM_CHECK:-0}" = "1" ]; then
  echo "== Determinism check: snapshot -> re-run -> cmp =="
  cp "$OUT"_DE.json "$OUT"_IT.json "$OUT"_SE.json "$OUT"_summary.json "$SNAP"/
  (cd "$ROOT" && pnpm cli forecast-sweep --mode adaptive-sensitivity -o "$OUT" --quiet)
  ok=1
  for f in _DE.json _IT.json _SE.json _summary.json; do
    if cmp -s "$SNAP/adaptive_sensitivity$f" "$OUT$f"; then
      echo "  IDENTICAL: adaptive_sensitivity$f"
    else
      echo "  DIFFER: adaptive_sensitivity$f"
      ok=0
    fi
  done
  [ "$ok" = "1" ] && echo "== Determinism check PASSED ==" || { echo "== Determinism check FAILED =="; exit 1; }
fi

echo "== Done. Artifacts in $DIR =="
