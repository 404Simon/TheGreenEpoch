import { writeFileSync } from "node:fs";
import type { Constants, TrainingProfile, FullProfile, CO2Timeline, SweepPoint } from "../domain/types";
import { runOptimization } from "../domain/optimize";
import type { AdaptiveOptions } from "../domain/optimize";
import { loadJSON, loadCO2Timeline } from "./optimize";
import { round2 } from "../domain/utils";

export const DEFAULT_COSTS_S = [1, 10, 30, 60, 150, 300, 600];

export interface CheckpointSweepRow {
  costS: number;
  thetaPause: number | null;
  thetaResume: number | null;
  margin: number | null;
  startTime: string | null;
  actualOverheadPct: number | null;
  co2SavingsPct: number | null;
  score: number | null;
  numPauses: number | null;
  withinBudget: boolean | null;
  stopReason: string | null;
  validPoints: number;
  totalPoints: number;
}

export interface CheckpointSweepResult {
  model: string;
  region: string;
  historicalYears: number[];
  costsS: number[];
  options: AdaptiveOptions;
  rows: CheckpointSweepRow[];
}

function toRow(costS: number, points: ReturnType<typeof runOptimization>["points"], best: ReturnType<typeof runOptimization>["best"]): CheckpointSweepRow {
  const valid = points.filter((p) => p.withinBudget && p.co2SavingsPct > 0);
  return {
    costS,
    thetaPause: best ? best.thetaPause : null,
    thetaResume: best ? best.thetaResume : null,
    margin: best ? best.thetaPause - best.thetaResume : null,
    startTime: best ? best.startTime : null,
    actualOverheadPct: best ? best.actualOverheadPct : null,
    co2SavingsPct: best ? best.co2SavingsPct : null,
    score: best ? best.score : null,
    numPauses: best ? best.numPauses : null,
    withinBudget: best ? best.withinBudget : null,
    stopReason: best ? best.stopReason : null,
    validPoints: valid.length,
    totalPoints: points.length,
  };
}

export function sweepCheckpoints(
  profile: TrainingProfile,
  constants: Constants,
  timeline: CO2Timeline,
  historicalYears: number[],
  costsS: number[],
  options: AdaptiveOptions,
  onProgress?: (costS: number, iteration: number, row: CheckpointSweepRow) => void,
): CheckpointSweepRow[] {
  const rows: CheckpointSweepRow[] = [];

  for (const costS of costsS) {
    const fullProfile: FullProfile = {
      ...profile,
      gpuPowerTrain: constants.gpu_power_train,
      gpuPowerPause: constants.gpu_power_pause,
      pue: constants.pue,
      checkpointPauseTime: costS,
      checkpointResumeTime: costS,
    };

    const acc: SweepPoint[] = [];
    let currentBest: SweepPoint | null = null;
    const { points, best } = runOptimization(fullProfile, timeline, historicalYears, options, (iter, iterPoints, iterBest) => {
      acc.push(...iterPoints);
      currentBest = iterBest;
      onProgress?.(costS, iter, toRow(costS, acc, currentBest));
    });

    rows.push(toRow(costS, points, best));
  }

  return rows;
}

// ── CLI handler ──────────────────────────────────────────────────

function fmt(n: number | null, decimals = 1, suffix = ""): string {
  if (n === null || isNaN(n)) return "\u2014";
  return n.toFixed(decimals) + suffix;
}

function parseCosts(raw: string | undefined): number[] {
  if (!raw) return [...DEFAULT_COSTS_S];
  const costs = raw
    .split(",")
    .map((s) => parseFloat(s.trim()))
    .filter((n) => !isNaN(n) && n >= 0);
  if (costs.length === 0) {
    console.error(`  Invalid --costs value: "${raw}" (expected comma-separated seconds)`);
    process.exit(1);
  }
  return costs;
}

export async function checkpointSweepCli(raw: {
  model: string; region: string; years: string;
  costs?: string;
  tpMax?: string; budget?: string; resolution?: string;
  dateRes?: string; maxIter?: string; alpha?: string;
  start?: string; output?: string; csv?: string;
}): Promise<void> {
  const historicalYears = raw.years.split(",").map(Number);
  const costsS = parseCosts(raw.costs);

  const constants = loadJSON<Constants>("constants.json");
  const profiles = loadJSON<Record<string, TrainingProfile>>("profiles.json");

  const profile = profiles[raw.model];
  if (!profile) {
    console.error(`  Unknown model: ${raw.model}. Available: ${Object.keys(profiles).join(", ")}`);
    process.exit(1);
  }

  const timeline = loadCO2Timeline(raw.region, historicalYears);

  const options: AdaptiveOptions = {
    thetaPauseMax: raw.tpMax ? parseFloat(raw.tpMax) : 500,
    overheadBudgetPct: raw.budget ? parseFloat(raw.budget) : 200,
    resolution: raw.resolution ? parseInt(raw.resolution) : 10,
    startDateResolution: raw.dateRes ? parseInt(raw.dateRes) : 7,
    maxIterations: raw.maxIter ? parseInt(raw.maxIter) : 6,
    minStep: 3,
    shrinkFactor: 0.45,
    alpha: raw.alpha ? parseFloat(raw.alpha) : 1,
    ...(raw.start ? { fixedStartTime: raw.start } : {}),
  };

  console.log(`\n  TheGreenEpoch Checkpoint-Sweep`);
  console.log(`  Model: ${raw.model}, Region: ${raw.region}, Years: ${raw.years}`);
  console.log(`  Costs: ${costsS.join(", ")} s (save = load)`);
  console.log(`  Grid: ${options.resolution}\u00D7${options.startDateResolution}, ${options.maxIterations} iter(s)`);
  console.log(`  Budget: ${options.overheadBudgetPct}%, \u03B1=${options.alpha}\n`);

  const rows = sweepCheckpoints(profile, constants, timeline, historicalYears, costsS, options, (costS, iter, row) => {
    const msg = row.thetaPause !== null
      ? `best \u03B8\u209A=${row.thetaPause}, \u03B8\u209B=${row.thetaResume}, score=${row.score?.toFixed(4)}`
      : "no valid point";
    console.log(`  cost=${costS}s · iter ${iter + 1}: ${msg}`);
  });

  const upper = rows.map((r) => r.thetaPause).filter((v): v is number => v !== null);
  const lower = rows.map((r) => r.thetaResume).filter((v): v is number => v !== null);

  console.log(`\n  ${"\u2500".repeat(80)}`);
  console.log(`  ${"COST".padStart(8)} ${"\u03B8_p UPPER".padStart(11)} ${"\u03B8_r LOWER".padStart(11)} ${"MARGIN".padStart(8)} ${"START".padStart(6)} ${"OVERH".padStart(7)} ${"CO\u2082\u2193".padStart(8)} ${"SCORE".padStart(8)} ${"PAUSES".padStart(7)}`);
  console.log(`  ${"\u2500".repeat(80)}`);

  for (const r of rows) {
    const cost = `${r.costS}s`.padStart(8);
    const up = fmt(r.thetaPause, 2).padStart(11);
    const down = fmt(r.thetaResume, 2).padStart(11);
    const margin = fmt(r.margin, 2).padStart(8);
    const start = (r.startTime ?? "\u2014").padStart(6);
    const overh = fmt(r.actualOverheadPct, 1, "%").padStart(7);
    const save = fmt(r.co2SavingsPct, 2, "%").padStart(8);
    const score = fmt(r.score, 4).padStart(8);
    const pauses = fmt(r.numPauses, 0).padStart(7);
    console.log(`  ${cost} ${up} ${down} ${margin} ${start} ${overh} ${save} ${score} ${pauses}`);
  }

  console.log(`  ${"\u2500".repeat(80)}`);
  if (upper.length > 0 && lower.length > 0) {
    const minUp = Math.min(...upper), maxUp = Math.max(...upper);
    const minDown = Math.min(...lower), maxDown = Math.max(...lower);
    const spanUp = round2(maxUp - minUp);
    const spanDown = round2(maxDown - minDown);
    console.log(`  \u03B8_p (upper bound): ${minUp.toFixed(2)}\u2013${maxUp.toFixed(2)}  span=${spanUp}`);
    console.log(`  \u03B8_r (lower bound): ${minDown.toFixed(2)}\u2013${maxDown.toFixed(2)}  span=${spanDown}`);
    console.log(spanUp === 0 && spanDown === 0
      ? `  \u2192 Checkpoint cost has NO effect on the hysteresis band.`
      : `  \u2192 Checkpoint cost influences the hysteresis band.`);
  } else {
    console.log(`  No valid hysteresis found for any cost.`);
  }

  const result: CheckpointSweepResult = {
    model: raw.model,
    region: raw.region,
    historicalYears,
    costsS,
    options,
    rows,
  };

  if (raw.output) {
    writeFileSync(raw.output, JSON.stringify(result, null, 2), "utf-8");
    console.log(`\n  JSON: ${raw.output} (${rows.length} rows)`);
  }

  if (raw.csv) {
    const header = [
      "cost_s", "theta_pause_upper", "theta_resume_lower", "margin", "start_time",
      "overhead_pct", "co2_savings_pct", "score", "num_pauses", "within_budget",
      "stop_reason", "valid_points", "total_points",
    ];
    const body = rows.map((r) => [
      r.costS,
      r.thetaPause ?? "",
      r.thetaResume ?? "",
      r.margin ?? "",
      r.startTime ?? "",
      r.actualOverheadPct ?? "",
      r.co2SavingsPct ?? "",
      r.score ?? "",
      r.numPauses ?? "",
      r.withinBudget ?? "",
      r.stopReason ?? "",
      r.validPoints,
      r.totalPoints,
    ].join(","));
    writeFileSync(raw.csv, "\uFEFF" + header.join(",") + "\n" + body.join("\n") + "\n", "utf-8");
    console.log(`  CSV:  ${raw.csv} (${rows.length} rows)`);
  }

  console.log(`  Done.\n`);
}
