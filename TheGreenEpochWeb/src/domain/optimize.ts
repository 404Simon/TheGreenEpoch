import type { SweepPoint, FullProfile, CO2Timeline, SimConfig } from "./types";
import { simulateStepwise } from "./simulation";
import { neverPausePolicy, hysteresisPolicy } from "./policy";
import { tokensPerSecond } from "./physics";
import { computeOverheadPct, computeSavingsPct, computeScore } from "./result";

export interface AdaptiveOptions {
  thetaPauseMax: number;
  overheadBudgetPct: number;
  resolution: number;
  startDateResolution: number;
  maxIterations: number;
  minStep: number;
  shrinkFactor: number;
  alpha: number;
  fixedStartTime?: string;
}

export function runOptimization(
  profile: FullProfile,
  timeline: CO2Timeline,
  historicalYears: number[],
  options: AdaptiveOptions,
  onIteration?: (iteration: number, points: SweepPoint[], best: SweepPoint | null) => void,
): { points: SweepPoint[]; best: SweepPoint | null } {
  const baseSimConfig: SimConfig = {
    startTime: "01-01",
    historicalYears,
    overheadBudgetPct: options.overheadBudgetPct,
  };

  const tps = tokensPerSecond(profile.gpuCount) || 1;
  const allPoints: SweepPoint[] = [];
  let best: SweepPoint | null = null;
  let bounds = initialBounds(options.thetaPauseMax);

  for (let iter = 0; iter < options.maxIterations; iter++) {
    const dateSamples = options.fixedStartTime
      ? [dateToDay(options.fixedStartTime)]
      : generateDateSamples(bounds, options.startDateResolution);
    const grid = generateGrid(bounds, options.resolution);
    if (grid.length === 0 || dateSamples.length === 0) break;

    const iterPoints: SweepPoint[] = [];

    for (const day of dateSamples) {
      const startTime = dayToDate(day);
      const dateSimConfig: SimConfig = { ...baseSimConfig, startTime };

      let baselineLast: import("./types").SimProgress | null = null;
      for (const p of simulateStepwise(profile, neverPausePolicy(), timeline, dateSimConfig)) {
        baselineLast = p;
      }
      if (!baselineLast) continue;
      const baselineEm = baselineLast.totalEmissionsG / 1000;

      for (const pt of grid) {
        const policy = hysteresisPolicy(pt.thetaPause, pt.thetaResume);
        let last = null;
        for (const p of simulateStepwise(profile, policy, timeline, dateSimConfig)) {
          last = p;
        }
        if (!last) continue;

        const actualOverheadPct = computeOverheadPct(last.pausedS, last.checkpointS, last.tokensTotal, tps);
        const totalEm = last.totalEmissionsG / 1000;
        const co2SavingsPct = computeSavingsPct(totalEm, baselineEm);

        const score = computeScore(co2SavingsPct, actualOverheadPct, options.overheadBudgetPct, options.alpha);
        iterPoints.push({
          thetaPause: pt.thetaPause,
          thetaResume: pt.thetaResume,
          startTime,
          actualOverheadPct,
          co2SavingsPct,
          score,
          numPauses: last.numPauses,
          totalEmissionsKgco2: totalEm,
          baselineEmissionsKgco2: baselineEm,
          withinBudget: actualOverheadPct <= options.overheadBudgetPct,
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

    onIteration?.(iter, iterPoints, best);

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

  return { points: allPoints, best };
}

export interface Bounds {
  tpMin: number;
  tpMax: number;
  trMin: number;
  trMax: number;
  dayMin: number;
  dayMax: number;
}

const DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

export function dayToDate(day: number): string {
  let d = day;
  for (let m = 0; m < 12; m++) {
    if (d < DAYS_IN_MONTH[m]) {
      const mm = String(m + 1).padStart(2, "0");
      const dd = String(d + 1).padStart(2, "0");
      return `${mm}-${dd}`;
    }
    d -= DAYS_IN_MONTH[m];
  }
  return "12-31";
}

export function dateToDay(dateStr: string): number {
  const [month, day] = dateStr.split("-").map(Number);
  let d = 0;
  for (let m = 0; m < month - 1; m++) {
    d += DAYS_IN_MONTH[m];
  }
  return d + day - 1;
}

export function generateDateSamples(bounds: Bounds, resolution: number): number[] {
  const { dayMin, dayMax } = bounds;
  if (dayMin >= dayMax) return [Math.round(dayMin)];
  const step = Math.max(1, Math.round((dayMax - dayMin) / (resolution - 1)));
  const days: number[] = [];
  for (let d = dayMin; d <= dayMax; d += step) {
    days.push(Math.round(d));
  }
  if (days[days.length - 1] < dayMax) {
    days.push(Math.round(dayMax));
  }
  return days;
}

export function initialBounds(thetaPauseMax: number): Bounds {
  return {
    tpMin: 10, tpMax: thetaPauseMax,
    trMin: 0, trMax: thetaPauseMax,
    dayMin: 0, dayMax: 364,
  };
}

export function generateGrid(bounds: Bounds, resolution: number): Array<{ thetaPause: number; thetaResume: number }> {
  const { tpMin, tpMax, trMin, trMax } = bounds;
  const stepTp = Math.max((tpMax - tpMin) / (resolution - 1), 1);
  const points: Array<{ thetaPause: number; thetaResume: number }> = [];

  for (let tpi = 0; tpi < resolution; tpi++) {
    const tp = round(tpMin + tpi * stepTp);
    if (tp > tpMax) break;

    const trCount = Math.max(2, Math.round(((tp - trMin) / (trMax - trMin)) * resolution));
    for (let tri = 0; tri < trCount; tri++) {
      const tr = round(trMin + tri * (tp - trMin) / Math.max(trCount - 1, 1));
      if (tr > tp) break;
      points.push({ thetaPause: tp, thetaResume: tr });
    }
  }

  return points;
}

export function refineBounds(
  best: { thetaPause: number; thetaResume: number; startDay: number },
  currentBounds: Bounds,
  shrinkFactor: number,
  minStep: number,
): Bounds {
  const span = currentBounds.tpMax - currentBounds.tpMin;
  const newMin = Math.max(10, best.thetaPause - span * shrinkFactor / 2);
  const newMax = Math.min(currentBounds.tpMax, best.thetaPause + span * shrinkFactor / 2);

  const tpMin = round(newMin);
  const tpMax = round(newMax);
  let trMin = round(Math.max(0, best.thetaResume - span * shrinkFactor / 2));
  let trMax = round(Math.min(tpMax, best.thetaResume + span * shrinkFactor / 2));
  trMin = Math.max(0, trMin);
  trMax = Math.max(trMin + minStep, trMax);

  const daySpan = currentBounds.dayMax - currentBounds.dayMin;
  let dayMin = Math.round(Math.max(0, best.startDay - daySpan * shrinkFactor / 2));
  let dayMax = Math.round(Math.min(364, best.startDay + daySpan * shrinkFactor / 2));
  dayMax = Math.max(dayMin + 1, dayMax);

  return { tpMin, tpMax, trMin, trMax, dayMin, dayMax };
}

export function expandBounds(bounds: Bounds): Bounds {
  return {
    tpMin: Math.max(10, round(bounds.tpMin * 0.8)),
    tpMax: Math.min(bounds.tpMax, round(bounds.tpMax * 0.9)),
    trMin: 0,
    trMax: bounds.tpMax,
    dayMin: 0,
    dayMax: 364,
  };
}

export function findBest(points: SweepPoint[], budget: number): SweepPoint | null {
  const valid = points.filter(r => r.withinBudget && r.co2SavingsPct > 0);
  if (valid.length === 0) return null;
  return valid.reduce((a, b) => (a.score > b.score ? a : b));
}

function round(n: number): number {
  return Math.round(n * 100) / 100;
}
