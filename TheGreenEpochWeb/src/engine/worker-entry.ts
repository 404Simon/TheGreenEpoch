import type { FullProfile, CO2Timeline, Scenario, SimConfig, SimProgress, SweepPoint } from "../domain/types";
import { simulateStepwise } from "../domain/simulation";
import { neverPausePolicy, hysteresisPolicy } from "../domain/policy";
import { tokensPerSecond } from "../domain/physics";
import { generateGrid, generateDateSamples, refineBounds, expandBounds, findBest, initialBounds, dayToDate, dateToDay } from "../domain/optimize";

interface StartMessage {
  type: "start";
  profile: FullProfile;
  timeline: CO2Timeline;
  scenario: Scenario;
  options: {
    thetaPauseMax: number;
    overheadBudgetPct: number;
    resolution: number;
    startDateResolution: number;
    maxIterations: number;
    minStep: number;
    shrinkFactor: number;
    alpha: number;
  };
}

self.onmessage = (e: MessageEvent<StartMessage>) => {
  if (e.data.type !== "start") return;

  const { profile, timeline, scenario, options } = e.data;

  const baseSimConfig: SimConfig = {
    startTime: "01-01",
    historicalYears: scenario.historicalYears,
    overheadBudgetPct: options.overheadBudgetPct,
  };

  const tps = tokensPerSecond(profile.gpuCount) || 1;
  const allPoints: SweepPoint[] = [];
  let best: SweepPoint | null = null;
  let bounds = initialBounds(options.thetaPauseMax);

  for (let iter = 0; iter < options.maxIterations; iter++) {
    const dateSamples = generateDateSamples(bounds, options.startDateResolution);
    const grid = generateGrid(bounds, options.resolution);
    if (grid.length === 0 || dateSamples.length === 0) break;

    const iterPoints: SweepPoint[] = [];

    for (const day of dateSamples) {
      const startTime = dayToDate(day);
      const dateSimConfig: SimConfig = { ...baseSimConfig, startTime };

      const baselineProgress: SimProgress[] = [];
      for (const p of simulateStepwise(profile, neverPausePolicy(), timeline, dateSimConfig)) {
        baselineProgress.push(p);
      }
      const baselineLast = baselineProgress[baselineProgress.length - 1];
      const baselineEm = baselineLast.totalEmissionsG / 1000;

      for (const pt of grid) {
        const policy = hysteresisPolicy(pt.thetaPause, pt.thetaResume);
        let last: SimProgress | null = null;
        for (const p of simulateStepwise(profile, policy, timeline, dateSimConfig)) {
          last = p;
        }
        if (!last) continue;

        const overheadS = last.pausedS + last.checkpointS;
        const actualOverheadPct = 100 * overheadS / (last.tokensTotal / tps || 1);
        const totalEm = last.totalEmissionsG / 1000;
        const co2SavingsPct = baselineEm > 0 ? (baselineEm - totalEm) / baselineEm * 100 : 0;

        const savingsNorm = co2SavingsPct / 100;
        const overheadNorm = actualOverheadPct / Math.max(options.overheadBudgetPct, 0.001);
        iterPoints.push({
          thetaPause: pt.thetaPause,
          thetaResume: pt.thetaResume,
          startTime,
          actualOverheadPct,
          co2SavingsPct,
          score: options.alpha * savingsNorm - (1 - options.alpha) * overheadNorm,
          numPauses: last.numPauses,
          totalEmissionsKgco2: totalEm,
          baselineEmissionsKgco2: baselineEm,
          withinBudget: overheadS / (last.tokensTotal / tps || 1) <= options.overheadBudgetPct / 100,
          stopReason: last.stopReason,
          completed: last.tokensRemaining <= 0,
          iteration: iter,
        });
      }
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
      bounds = refineBounds(
        { thetaPause: best.thetaPause, thetaResume: best.thetaResume, startDay: dateToDay(best.startTime) },
        bounds,
        options.shrinkFactor,
        options.minStep,
      );
    }
  }

  (self as any).postMessage({ type: "done", points: allPoints, best });
};
