import type { FullProfile, CO2Timeline, Scenario, SimConfig, SimProgress, SweepPoint } from "../domain/types";
import { simulateStepwise } from "../domain/simulation";
import { neverPausePolicy, hysteresisPolicy } from "../domain/policy";
import { tokensPerSecond } from "../domain/physics";
import { generateGrid, refineBounds, expandBounds, findBest, initialBounds } from "../domain/optimize";

interface StartMessage {
  type: "start";
  profile: FullProfile;
  timeline: CO2Timeline;
  scenario: Scenario;
  options: {
    thetaPauseMax: number;
    overheadBudgetPct: number;
    resolution: number;
    maxIterations: number;
    minStep: number;
    shrinkFactor: number;
  };
  startTimeIdx: number;
}

self.onmessage = (e: MessageEvent<StartMessage>) => {
  if (e.data.type !== "start") return;

  const { profile, timeline, scenario, options, startTimeIdx } = e.data;
  const startTime = scenario.startTimes[startTimeIdx] || "01-01";

  const simConfig: SimConfig = {
    startTime,
    historicalYears: scenario.historicalYears,
    overheadBudgetPct: options.overheadBudgetPct,
  };

  const baselineProgress: SimProgress[] = [];
  for (const p of simulateStepwise(profile, neverPausePolicy(), timeline, simConfig)) {
    baselineProgress.push(p);
  }
  const baselineLast = baselineProgress[baselineProgress.length - 1];
  const tps = tokensPerSecond(profile.gpuCount) || 1;

  const allPoints: SweepPoint[] = [];
  let best: SweepPoint | null = null;
  let bounds = initialBounds(options.thetaPauseMax);

  for (let iter = 0; iter < options.maxIterations; iter++) {
    const grid = generateGrid(bounds, options.resolution);
    if (grid.length === 0) break;

    const iterPoints: SweepPoint[] = [];
    for (const pt of grid) {
      const policy = hysteresisPolicy(pt.thetaPause, pt.thetaResume);
      let last: SimProgress | null = null;
      for (const p of simulateStepwise(profile, policy, timeline, simConfig)) {
        last = p;
      }
      if (!last) continue;

      const overheadS = last.pausedS + last.checkpointS;
      const actualOverheadPct = 100 * overheadS / (last.tokensTotal / tps || 1);
      const totalEm = last.totalEmissionsG / 1000;
      const baselineEm = baselineLast.totalEmissionsG / 1000;
      const co2SavingsPct = baselineEm > 0 ? (baselineEm - totalEm) / baselineEm * 100 : 0;

      iterPoints.push({
        thetaPause: pt.thetaPause,
        thetaResume: pt.thetaResume,
        actualOverheadPct,
        co2SavingsPct,
        score: co2SavingsPct / Math.max(actualOverheadPct, 0.001),
        numPauses: last.numPauses,
        totalEmissionsKgco2: totalEm,
        baselineEmissionsKgco2: baselineEm,
        withinBudget: overheadS / (last.tokensTotal / tps || 1) <= options.overheadBudgetPct / 100,
        stopReason: last.stopReason,
        completed: last.tokensRemaining <= 0,
        iteration: iter,
      });
    }

    allPoints.push(...iterPoints);

    const iterBest = findBest(iterPoints, options.overheadBudgetPct);
    if (iterBest && (!best || iterBest.score > best.score)) {
      best = iterBest;
    }

    (self as any).postMessage({ type: "iteration", iteration: iter, points: iterPoints, best });

    if (iter >= options.maxIterations - 1) break;

    if (!best) {
      bounds = expandBounds(bounds);
    } else {
      bounds = refineBounds(best, bounds, options.shrinkFactor, options.minStep);
    }
  }

  (self as any).postMessage({ type: "done", points: allPoints, best });
};
