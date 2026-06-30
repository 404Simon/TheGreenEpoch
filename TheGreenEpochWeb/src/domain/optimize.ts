import type { SweepPoint } from "./types";

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
