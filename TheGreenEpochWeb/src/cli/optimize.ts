import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Constants, TrainingProfile, CO2Timeline, YearCO2, FullProfile } from "../domain/types";
import { runOptimization } from "../domain/optimize";
import type { AdaptiveOptions } from "../domain/optimize";
import { averageYears } from "../data/co2-loader";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(__dirname, "../../public/data");

function loadJSON<T>(path: string): T {
  return JSON.parse(readFileSync(resolve(DATA_DIR, path), "utf-8")) as T;
}

function loadCO2Timeline(zone: string, years: number[]): CO2Timeline {
  const allData = years.map((y) => {
    return loadJSON<YearCO2>(`co2/${zone}_${y}.json`);
  });
  return averageYears(allData);
}

export async function optimizeCli(raw: {
  model: string; region: string; years: string;
  tpMax?: string; budget?: string; resolution?: string;
  dateRes?: string; maxIter?: string; alpha?: string;
  start?: string; output?: string;
}): Promise<void> {
  const output = raw.output ?? null;
  const historicalYears = raw.years.split(",").map(Number);
  const cliOpts = {
    thetaPauseMax: raw.tpMax ? parseFloat(raw.tpMax) : undefined,
    overheadBudgetPct: raw.budget ? parseFloat(raw.budget) : undefined,
    resolution: raw.resolution ? parseInt(raw.resolution) : undefined,
    startDateResolution: raw.dateRes ? parseInt(raw.dateRes) : undefined,
    maxIterations: raw.maxIter ? parseInt(raw.maxIter) : undefined,
    alpha: raw.alpha ? parseFloat(raw.alpha) : undefined,
  };

  const constants = loadJSON<Constants>("constants.json");
  const profiles = loadJSON<Record<string, TrainingProfile>>("profiles.json");

  const profile = profiles[raw.model];
  if (!profile) {
    console.error(`  Unknown model: ${raw.model}. Available: ${Object.keys(profiles).join(", ")}`);
    process.exit(1);
  }

  const fullProfile: FullProfile = {
    ...profile,
    gpuPowerTrain: constants.gpu_power_train,
    gpuPowerPause: constants.gpu_power_pause,
    pue: constants.pue,
    checkpointPauseTime: constants.checkpoint_pause_time,
    checkpointResumeTime: constants.checkpoint_resume_time,
  };

  const timeline = loadCO2Timeline(raw.region, historicalYears);

  const options: AdaptiveOptions = {
    thetaPauseMax: cliOpts.thetaPauseMax ?? 500,
    overheadBudgetPct: cliOpts.overheadBudgetPct ?? 200,
    resolution: cliOpts.resolution ?? 10,
    startDateResolution: cliOpts.startDateResolution ?? 7,
    maxIterations: cliOpts.maxIterations ?? 6,
    minStep: 3,
    shrinkFactor: 0.45,
    alpha: cliOpts.alpha ?? 1,
    ...(raw.start ? { fixedStartTime: raw.start } : {}),
  };

  console.log(`\n  TheGreenEpoch Optimize`);
  console.log(`  Model: ${raw.model}, Region: ${raw.region}, Years: ${raw.years}`);
  console.log(`  Grid: ${options.resolution}×${options.startDateResolution}, ${options.maxIterations} iter(s)`);
  console.log(`  Budget: ${options.overheadBudgetPct}%, α=${options.alpha}\n`);

  const { points, best } = runOptimization(fullProfile, timeline, historicalYears, options, (iter, iterPts, iterBest) => {
    const msg = iterBest
      ? `iter ${iter + 1}: ${iterPts.length} pts, best score=${iterBest.score.toFixed(4)} (θₚ=${iterBest.thetaPause}, θᵣ=${iterBest.thetaResume})`
      : `iter ${iter + 1}: ${iterPts.length} pts, no valid point`;
    console.log(`  ${msg}`);
  });

  const withinBudget = points.filter((p) => p.withinBudget);
  const valid = points.filter((p) => p.withinBudget && p.co2SavingsPct > 0);

  console.log(`\n  ─────────────────────────────────────────────`);
  console.log(`  Total points: ${points.length}`);
  console.log(`  Within budget: ${withinBudget.length}`);
  console.log(`  Valid (budget + savings>0): ${valid.length}`);

  if (best) {
    console.log(`  Best: θₚ=${best.thetaPause}, θᵣ=${best.thetaResume}`);
    console.log(`        start=${best.startTime}, CO₂ savings=${best.co2SavingsPct.toFixed(2)}%`);
    console.log(`        overhead=${best.actualOverheadPct.toFixed(1)}%, score=${best.score.toFixed(4)}`);
  } else {
    console.log(`  No valid best point found`);
  }

  const result = { model: raw.model, region: raw.region, historicalYears, options, points, best };

  if (output) {
    writeFileSync(output, JSON.stringify(result, null, 2), "utf-8");
    console.log(`\n  Output: ${output} (${points.length} points)`);
  }

  console.log(`  Done.\n`);
}
