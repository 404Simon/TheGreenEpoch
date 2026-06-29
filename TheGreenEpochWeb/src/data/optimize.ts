import type { FullProfile, SimConfig, CO2Timeline, Scenario } from "../types";
import { simulateStepwise, buildResult } from "./simulation";

export interface SweepPoint {
  thetaPause: number;
  thetaResume: number;
  actualOverheadPct: number;
  co2SavingsPct: number;
  score: number;
  numPauses: number;
  totalEmissionsKgco2: number;
  baselineEmissionsKgco2: number;
  withinBudget: boolean;
  stopReason: string;
  completed: boolean;
  iteration: number;
}

export interface SweepOptions {
  thetaPauseMin: number;
  thetaPauseMax: number;
  thetaPauseStep: number;
  hysteresisMode: "ratio" | "offset";
  hysteresisValue: number;
  overheadBudgetPct: number;
}

export interface AdaptiveOptions {
  thetaPauseMax: number;
  overheadBudgetPct: number;
  resolution: number;
  maxIterations: number;
  minStep: number;
  shrinkFactor: number;
}

export interface AdaptiveResult {
  points: SweepPoint[];
  best: SweepPoint | null;
  iterations: number;
}

export function runSweep(
  profile: FullProfile,
  timeline: CO2Timeline,
  scenario: Scenario,
  options: SweepOptions,
  startTimeIdx = 0,
): SweepPoint[] {
  const startTime = scenario.startTimes[startTimeIdx] || "01-01";

  const shared = {
    scenarioDescription: scenario.description,
    region: scenario.region,
    historicalYears: scenario.historicalYears,
    startTime,
    overheadBudgetPct: options.overheadBudgetPct,
  };

  const baselineLast = runBaseline(profile, shared, timeline);

  const points: SweepPoint[] = [];
  for (let tp = options.thetaPauseMin; tp <= options.thetaPauseMax; tp = round(tp + options.thetaPauseStep)) {
    const tr = options.hysteresisMode === "ratio"
      ? round(tp * options.hysteresisValue)
      : Math.max(0, round(tp - options.hysteresisValue));

    const pt = evaluate(profile, { ...shared, thetaPause: tp, thetaResume: tr }, timeline, baselineLast, 0);
    if (pt) points.push(pt);
  }

  return points;
}

export function adaptiveSweep(
  profile: FullProfile,
  timeline: CO2Timeline,
  scenario: Scenario,
  options: AdaptiveOptions,
  startTimeIdx = 0,
  onIteration?: (iteration: number, points: SweepPoint[], best: SweepPoint | null) => void,
): AdaptiveResult {
  const startTime = scenario.startTimes[startTimeIdx] || "01-01";

  const shared = {
    scenarioDescription: scenario.description,
    region: scenario.region,
    historicalYears: scenario.historicalYears,
    startTime,
    overheadBudgetPct: options.overheadBudgetPct,
  };

  const baselineLast = runBaseline(profile, shared, timeline);

  const allPoints: SweepPoint[] = [];
  let iterPoints: SweepPoint[] = [];
  let best: SweepPoint | null = null;

  let tpMin = 10;
  let tpMax = options.thetaPauseMax;
  let trMin = 0;
  let trMax = options.thetaPauseMax;

  for (let iter = 0; iter < options.maxIterations; iter++) {
    const stepTp = Math.max((tpMax - tpMin) / (options.resolution - 1), 1);
    const stepTr = Math.max((trMax - trMin) / (options.resolution - 1), 1);

    if (stepTp < options.minStep && stepTr < options.minStep) break;

    iterPoints = [];

    for (let tpi = 0; tpi < options.resolution; tpi++) {
      const tp = round(tpMin + tpi * stepTp);
      if (tp > options.thetaPauseMax) break;

      const trCount = Math.max(2, Math.round(((tp - trMin) / (trMax - trMin)) * options.resolution));
      for (let tri = 0; tri < trCount; tri++) {
        const tr = round(trMin + tri * (tp - trMin) / Math.max(trCount - 1, 1));
        if (tr > tp) break;

        const pt = evaluate(profile, { ...shared, thetaPause: tp, thetaResume: tr }, timeline, baselineLast, iter);
        if (pt) iterPoints.push(pt);
      }
    }

    allPoints.push(...iterPoints);

    const iterBest = findBest(iterPoints, options.overheadBudgetPct);
    if (iterBest && (!best || iterBest.score > best.score)) {
      best = iterBest;
    }

    if (onIteration) onIteration(iter, iterPoints, best);

    if (iter === options.maxIterations - 1) break;

    if (!best) {
      tpMin = Math.max(10, tpMin * 0.8);
      tpMax = Math.min(options.thetaPauseMax, tpMax * 0.9);
      trMin = 0;
      trMax = tpMax;
      continue;
    }

    const span = tpMax - tpMin;
    const newMin = Math.max(10, best.thetaPause - span * options.shrinkFactor / 2);
    const newMax = Math.min(options.thetaPauseMax, best.thetaPause + span * options.shrinkFactor / 2);

    tpMin = round(newMin);
    tpMax = round(newMax);
    trMin = round(Math.max(0, best.thetaResume - span * options.shrinkFactor / 2));
    trMax = round(Math.min(tpMax, best.thetaResume + span * options.shrinkFactor / 2));
    trMin = Math.max(0, trMin);
    trMax = Math.max(trMin + options.minStep, trMax);
  }

  return { points: allPoints, best, iterations: Math.min(options.maxIterations, allPoints.length > 0 ? allPoints[allPoints.length - 1].iteration + 1 : 0) };
}

function runBaseline(profile: FullProfile, shared: { scenarioDescription: string; region: string; historicalYears: number[]; startTime: string; overheadBudgetPct: number }, timeline: CO2Timeline) {
  const baselineProgress: import("../types").SimProgress[] = [];
  for (const p of simulateStepwise(profile, { ...shared, thetaPause: Infinity, thetaResume: 0 }, timeline)) {
    baselineProgress.push(p);
  }
  return baselineProgress[baselineProgress.length - 1];
}

function evaluate(
  profile: FullProfile,
  config: SimConfig,
  timeline: CO2Timeline,
  baselineLast: import("../types").SimProgress,
  iteration: number,
): SweepPoint | null {
  let lastProgress: import("../types").SimProgress | null = null;
  for (const p of simulateStepwise(profile, config, timeline)) {
    lastProgress = p;
  }
  if (!lastProgress) return null;

  const meta = buildResult(profile, config, lastProgress, baselineLast);
  const actualOverheadPct = meta.actualOverheadPct;
  const co2SavingsPct = meta.baselineEmissionsKgco2 > 0
    ? (meta.baselineEmissionsKgco2 - meta.totalEmissionsKgco2) / meta.baselineEmissionsKgco2 * 100
    : 0;

  return {
    thetaPause: config.thetaPause === Infinity ? 9999 : config.thetaPause,
    thetaResume: config.thetaResume,
    actualOverheadPct,
    co2SavingsPct,
    score: co2SavingsPct / Math.max(actualOverheadPct, 0.001),
    numPauses: meta.numPauses,
    totalEmissionsKgco2: meta.totalEmissionsKgco2,
    baselineEmissionsKgco2: meta.baselineEmissionsKgco2,
    withinBudget: meta.withinOverheadBudget,
    stopReason: meta.stopReason,
    completed: meta.completed,
    iteration,
  };
}

function findBest(points: SweepPoint[], budget: number): SweepPoint | null {
  const valid = points.filter(r => r.withinBudget && r.co2SavingsPct > 0);
  if (valid.length === 0) return null;
  return valid.reduce((a, b) => (a.score > b.score ? a : b));
}

function round(n: number): number {
  return Math.round(n * 100) / 100;
}
