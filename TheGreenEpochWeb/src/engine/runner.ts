import type { FullProfile, SimConfig, CO2Timeline, SimProgress, SimResult, Policy, Scenario, SweepPoint } from "../domain/types";
import { simulateStepwise } from "../domain/simulation";
import { buildSimResult } from "../domain/result";
import { neverPausePolicy, hysteresisPolicy } from "../domain/policy";
import { tokensPerSecond } from "../domain/physics";
import { generateGrid, refineBounds, expandBounds, findBest, initialBounds } from "../domain/optimize";

export interface RunOptions {
  onProgress?: (p: SimProgress) => void;
}

export function runSimulation(
  profile: FullProfile,
  policy: Policy,
  timeline: CO2Timeline,
  simConfig: SimConfig & { scenarioDescription: string; model: string; region: string },
  thetaPause: number,
  thetaResume: number,
  options?: RunOptions,
): SimResult {
  const baselinePolicy = neverPausePolicy();

  const baselineProgress: SimProgress[] = [];
  for (const p of simulateStepwise(profile, baselinePolicy, timeline, simConfig)) {
    baselineProgress.push(p);
  }
  const baselineLast = baselineProgress[baselineProgress.length - 1];

  const tsSeries: string[] = [];
  const co2Series: number[] = [];
  const stateSeries: string[] = [];
  const emissionsSeries: number[] = [];
  const tokensRemainingSeries: number[] = [];

  let lastProgress: SimProgress | null = null;
  for (const p of simulateStepwise(profile, policy, timeline, simConfig)) {
    lastProgress = p;
    if (options?.onProgress) options.onProgress(p);
    tsSeries.push(p.timestamp);
    co2Series.push(p.carbonIntensity);
    stateSeries.push(p.state);
    emissionsSeries.push(p.totalEmissionsG / 1000);
    tokensRemainingSeries.push(p.tokensRemaining);
  }

  return buildSimResult(
    profile, simConfig, lastProgress!, baselineLast,
    thetaPause, thetaResume,
    {
      id: `result-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      scenarioDescription: simConfig.scenarioDescription,
      model: simConfig.model,
      region: simConfig.region,
      timestamps: tsSeries,
      carbonIntensitySeries: co2Series,
      stateSeries,
      emissionsSeries,
      tokensRemainingSeries,
    },
  );
}

export function evaluatePoint(
  profile: FullProfile,
  timeline: CO2Timeline,
  simConfig: SimConfig,
  thetaPause: number,
  thetaResume: number,
  baselineLast: SimProgress,
  iteration: number,
): SweepPoint | null {
  const policy = hysteresisPolicy(thetaPause, thetaResume);

  let lastProgress: SimProgress | null = null;
  for (const p of simulateStepwise(profile, policy, timeline, simConfig)) {
    lastProgress = p;
  }
  if (!lastProgress) return null;

  const tps = tokensPerSecond(profile.gpuCount) || 1;
  const overheadS = lastProgress.pausedS + lastProgress.checkpointS;
  const actualOverheadPct = 100 * overheadS / (lastProgress.tokensTotal / tps || 1);
  const totalEm = lastProgress.totalEmissionsG / 1000;
  const baselineEm = baselineLast.totalEmissionsG / 1000;
  const co2SavingsPct = baselineEm > 0 ? (baselineEm - totalEm) / baselineEm * 100 : 0;

  return {
    thetaPause,
    thetaResume,
    actualOverheadPct,
    co2SavingsPct,
    score: co2SavingsPct / Math.max(actualOverheadPct, 0.001),
    numPauses: lastProgress.numPauses,
    totalEmissionsKgco2: totalEm,
    baselineEmissionsKgco2: baselineEm,
    withinBudget: overheadS / (lastProgress.tokensTotal / tps || 1) <= simConfig.overheadBudgetPct / 100,
    stopReason: lastProgress.stopReason,
    completed: lastProgress.tokensRemaining <= 0,
    iteration,
  };
}

export function runBaseline(profile: FullProfile, timeline: CO2Timeline, simConfig: SimConfig): SimProgress {
  const bp: SimProgress[] = [];
  for (const p of simulateStepwise(profile, neverPausePolicy(), timeline, simConfig)) {
    bp.push(p);
  }
  return bp[bp.length - 1];
}

export interface AdaptiveOptions {
  thetaPauseMax: number;
  overheadBudgetPct: number;
  resolution: number;
  maxIterations: number;
  minStep: number;
  shrinkFactor: number;
}

export function runOptimization(
  profile: FullProfile,
  timeline: CO2Timeline,
  scenario: Scenario,
  options: AdaptiveOptions,
  onIteration?: (iteration: number, points: SweepPoint[], best: SweepPoint | null) => void,
): SweepPoint[] {
  const startTime = scenario.startTimes[0] || "01-01";
  const simConfig: SimConfig = {
    startTime,
    historicalYears: scenario.historicalYears,
    overheadBudgetPct: options.overheadBudgetPct,
  };

  const baselineLast = runBaseline(profile, timeline, simConfig);
  const allPoints: SweepPoint[] = [];
  let best: SweepPoint | null = null;
  let bounds = initialBounds(options.thetaPauseMax);

  for (let iter = 0; iter < options.maxIterations; iter++) {
    const grid = generateGrid(bounds, options.resolution);
    if (grid.length === 0) break;

    const iterPoints: SweepPoint[] = [];
    for (const pt of grid) {
      const result = evaluatePoint(profile, timeline, simConfig, pt.thetaPause, pt.thetaResume, baselineLast, iter);
      if (result) iterPoints.push(result);
    }

    allPoints.push(...iterPoints);

    const iterBest = findBest(iterPoints, options.overheadBudgetPct);
    if (iterBest && (!best || iterBest.score > best.score)) {
      best = iterBest;
    }

    if (onIteration) onIteration(iter, iterPoints, best);

    if (iter >= options.maxIterations - 1) break;

    if (!best) {
      bounds = expandBounds(bounds);
    } else {
      bounds = refineBounds(best, bounds, options.shrinkFactor, options.minStep);
    }
  }

  return allPoints;
}
