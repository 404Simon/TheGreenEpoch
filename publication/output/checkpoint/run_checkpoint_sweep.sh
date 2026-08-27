#!/usr/bin/env bash
# Deterministic checkpoint-realism sweep (Phase B.0.2).
# Runs the headline optimization for DE/IT/SE at ckpt-pause in {148.8,150,900,2700}s
# (resume 0) using the reopt optimizer settings (resolution 10, iterations 6,
# budget 200%, alpha 1, fixed per-region start, tpMax 800/800/100) and aggregates
# publication/output/checkpoint/{checkpoint_{DE,IT,SE}.json,checkpoint_all.csv,checkpoint_summary.json}.
# No RNG in this path -> running twice yields byte-identical artifacts (verify with cmp).
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../../.." && pwd)"
RAW="$DIR/_raw"
mkdir -p "$RAW"

CKPT_VALUES=(148.8 150 900 2700)
# region -> "start tpMax"
REGIONS_DE="02-01 800"
REGIONS_IT="01-14 800"
REGIONS_SE="04-22 100"

echo "== Running checkpoint sweep (12 optimizations) =="
for spec in "DE $REGIONS_DE" "IT $REGIONS_IT" "SE $REGIONS_SE"; do
  set -- $spec
  REGION=$1; START=$2; TP_MAX=$3
  for CKPT in "${CKPT_VALUES[@]}"; do
    echo "  [$REGION] ckpt-pause=${CKPT}s (start=$START, tpMax=$TP_MAX)"
    (cd "$ROOT" && pnpm cli optimize \
      -m Deepseek -r "$REGION" -y 2025 \
      --start "$START" --tp-max "$TP_MAX" \
      --budget 200 --resolution 10 --max-iter 6 \
      --ckpt-pause "$CKPT" --ckpt-resume 0 \
      -o "$RAW/${REGION}_${CKPT}.json") > /dev/null
  done
done

echo "== Aggregating artifacts =="
CHECKPOINT_DIR="$DIR" node --input-type=module <<'NODE_EOF'
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const dir = process.env.CHECKPOINT_DIR;
const rawDir = resolve(dir, "_raw");
const ckptValues = [148.8, 150, 900, 2700];
const regions = {
  DE: { start: "02-01", tpMax: 800 },
  IT: { start: "01-14", tpMax: 800 },
  SE: { start: "04-22", tpMax: 100 },
};
const reopt = JSON.parse(readFileSync(resolve(dir, "../forecast/reopt_summary.json"), "utf-8"));
const reoptByRegion = Object.fromEntries(reopt.map((r) => [r.region, r.baseline]));

function fmt(v) {
  return v == null ? "NA" : v.toPrecision(6);
}

function bestCompleted(points) {
  let best = null;
  for (const p of points) {
    if (p.withinBudget && p.completed && p.co2SavingsPct > 0 && (best === null || p.score > best.score)) best = p;
  }
  return best;
}

const summary = [];
const csvRows = [];
for (const region of Object.keys(regions)) {
  const cfg = regions[region];
  const runs = [];
  for (const ckpt of ckptValues) {
    const raw = JSON.parse(readFileSync(resolve(rawDir, `${region}_${ckpt}.json`), "utf-8"));
    const best = raw.best;
    const cb = bestCompleted(raw.points);
    const cell = best
      ? {
          ckpt_pause: ckpt,
          ckpt_resume: 0,
          theta_p: best.thetaPause,
          theta_r: best.thetaResume,
          margin: best.thetaPause - best.thetaResume,
          savings: best.co2SavingsPct,
          overhead: best.actualOverheadPct,
          score: best.score,
          num_pauses: best.numPauses,
          completed: best.completed,
          found: true,
        }
      : {
          ckpt_pause: ckpt,
          ckpt_resume: 0,
          theta_p: null,
          theta_r: null,
          margin: null,
          savings: null,
          overhead: null,
          score: null,
          num_pauses: null,
          completed: null,
          found: false,
        };
    runs.push({ ...cell, bestCompleted: cb ? { theta_p: cb.thetaPause, theta_r: cb.thetaResume, margin: cb.thetaPause - cb.thetaResume, savings: cb.co2SavingsPct, overhead: cb.actualOverheadPct, score: cb.score, num_pauses: cb.numPauses, completed: true } : null });
    csvRows.push([
      region,
      String(ckpt),
      "0",
      fmt(cell.theta_p),
      fmt(cell.theta_r),
      fmt(cell.margin),
      fmt(cell.savings),
      fmt(cell.overhead),
      fmt(cell.score),
      cell.num_pauses == null ? "NA" : String(cell.num_pauses),
      String(cell.completed),
      cb ? fmt(cb.co2SavingsPct) : "NA",
      cb ? fmt(cb.actualOverheadPct) : "NA",
    ].join(","));
  }
  const base = reoptByRegion[region];
  const regionObj = {
    region,
    model: "Deepseek",
    year: 2025,
    start: cfg.start,
    optimizer: { resolution: 10, iterations: 6, budget: 200, alpha: 1, tpMax: cfg.tpMax },
    ckptValues,
    runs,
  };
  writeFileSync(resolve(dir, `checkpoint_${region}.json`), JSON.stringify(regionObj, null, 2) + "\n", "utf-8");
  summary.push({
    region,
    model: "Deepseek",
    year: 2025,
    start: cfg.start,
    tp_max: cfg.tpMax,
    baseline: { theta_p: base.thetaP, theta_r: base.thetaR, margin: base.margin, savings: base.savings, overhead: base.overhead, score: base.score },
    runs,
  });
}
writeFileSync(
  resolve(dir, "checkpoint_all.csv"),
  "region,ckpt_pause,ckpt_resume,theta_p,theta_r,margin,savings,overhead,score,num_pauses,completed,completed_best_savings,completed_best_overhead\n" + csvRows.join("\n") + "\n",
  "utf-8",
);
writeFileSync(resolve(dir, "checkpoint_summary.json"), JSON.stringify(summary, null, 2) + "\n", "utf-8");
console.log("  wrote checkpoint_{DE,IT,SE}.json, checkpoint_all.csv, checkpoint_summary.json");
NODE_EOF

echo "== Done. Artifacts in $DIR =="
