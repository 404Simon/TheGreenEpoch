import type { SweepPoint } from "./types";

export interface Bounds {
  tpMin: number;
  tpMax: number;
  trMin: number;
  trMax: number;
}

export function initialBounds(thetaPauseMax: number): Bounds {
  return { tpMin: 10, tpMax: thetaPauseMax, trMin: 0, trMax: thetaPauseMax };
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
  best: { thetaPause: number; thetaResume: number },
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

  return { tpMin, tpMax, trMin, trMax };
}

export function expandBounds(bounds: Bounds): Bounds {
  return {
    tpMin: Math.max(10, round(bounds.tpMin * 0.8)),
    tpMax: Math.min(bounds.tpMax, round(bounds.tpMax * 0.9)),
    trMin: 0,
    trMax: bounds.tpMax,
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
