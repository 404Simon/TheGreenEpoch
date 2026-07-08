import { describe, it, expect } from "vitest";
import {
  generateGrid,
  generateDateSamples,
  refineBounds,
  expandBounds,
  findBest,
  initialBounds,
  dateToDay,
  dayToDate,
} from "./optimize";
import type { Bounds, SweepPoint } from "./types";

function bounds(overrides?: Partial<Bounds>): Bounds {
  return {
    tpMin: 10,
    tpMax: 500,
    trMin: 0,
    trMax: 500,
    dayMin: 0,
    dayMax: 364,
    ...overrides,
  };
}

function sweepPoint(overrides?: Partial<SweepPoint>): SweepPoint {
  return {
    thetaPause: 200,
    thetaResume: 150,
    startTime: "01-01",
    actualOverheadPct: 10,
    co2SavingsPct: 15,
    score: 0.8,
    numPauses: 10,
    totalEmissionsKgco2: 5000,
    baselineEmissionsKgco2: 6000,
    withinBudget: true,
    stopReason: "completed",
    completed: true,
    iteration: 0,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// generateGrid
// ---------------------------------------------------------------------------

describe("generateGrid", () => {
  it("returns points within bounds", () => {
    const pts = generateGrid(bounds(), 5);
    for (const p of pts) {
      expect(p.thetaPause).toBeGreaterThanOrEqual(10);
      expect(p.thetaPause).toBeLessThanOrEqual(500);
      expect(p.thetaResume).toBeGreaterThanOrEqual(0);
      expect(p.thetaResume).toBeLessThanOrEqual(p.thetaPause);
    }
  });

  it("generates expected number of theta_pause values", () => {
    const pts = generateGrid(bounds(), 5);
    const tpValues = new Set(pts.map((p) => p.thetaPause));
    expect(tpValues.size).toBe(5);
  });

  it("produces non-empty grid with minimum resolution", () => {
    const pts = generateGrid(bounds(), 2);
    expect(pts.length).toBeGreaterThan(0);
  });

  it("always includes tpMax as the last thetaPause", () => {
    const pts = generateGrid(bounds(), 10);
    const tpValues = [...new Set(pts.map((p) => p.thetaPause))];
    expect(tpValues[tpValues.length - 1]).toBe(500);
  });

  it("thetaResume never exceeds thetaPause (hysteresis constraint)", () => {
    const pts = generateGrid(bounds(), 10);
    for (const p of pts) {
      expect(p.thetaResume).toBeLessThanOrEqual(p.thetaPause);
    }
  });

  it("handles narrow bounds (tpMin close to tpMax)", () => {
    const pts = generateGrid({ tpMin: 100, tpMax: 110, trMin: 0, trMax: 110, dayMin: 0, dayMax: 364 }, 5);
    expect(pts.length).toBeGreaterThan(0);
    for (const p of pts) {
      expect(p.thetaPause).toBeGreaterThanOrEqual(100);
      expect(p.thetaPause).toBeLessThanOrEqual(110);
    }
  });
});

// ---------------------------------------------------------------------------
// generateDateSamples
// ---------------------------------------------------------------------------

describe("generateDateSamples", () => {
  it("generates correct number of samples", () => {
    const days = generateDateSamples(bounds(), 7);
    expect(days.length).toBe(7);
  });

  it("returns single point when min >= max", () => {
    const days = generateDateSamples({ ...bounds(), dayMin: 100, dayMax: 100 }, 5);
    expect(days).toEqual([100]);
  });

  it("covers the full range", () => {
    const days = generateDateSamples(bounds({ dayMin: 0, dayMax: 364 }), 5);
    expect(days[0]).toBe(0);
    expect(days[days.length - 1]).toBe(364);
  });
});

// ---------------------------------------------------------------------------
// expandBounds
// ---------------------------------------------------------------------------

describe("expandBounds", () => {
  it("expands tpMin and tpMax outward", () => {
    const b = bounds({ tpMin: 100, tpMax: 300 });
    const expanded = expandBounds(b, 500);
    expect(expanded.tpMin).toBeLessThan(b.tpMin);
    expect(expanded.tpMax).toBeGreaterThan(b.tpMax);
  });

  it("never expands tpMin below 10", () => {
    const b = bounds({ tpMin: 10, tpMax: 50 });
    const expanded = expandBounds(b, 500);
    expect(expanded.tpMin).toBe(10);
  });

  it("caps tpMax at maxThetaPause", () => {
    const b = bounds({ tpMin: 100, tpMax: 200 });
    const expanded = expandBounds(b, 250);
    expect(expanded.tpMax).toBeLessThanOrEqual(250);
  });

  it("resets day range to full year", () => {
    const b = bounds({ dayMin: 100, dayMax: 200 });
    const expanded = expandBounds(b, 500);
    expect(expanded.dayMin).toBe(0);
    expect(expanded.dayMax).toBe(364);
  });

  it("resets trMin to 0", () => {
    const b = bounds({ trMin: 50, trMax: 400 });
    const expanded = expandBounds(b, 500);
    expect(expanded.trMin).toBe(0);
  });

  it("produces tpMax >= tpMin", () => {
    const b = bounds({ tpMin: 490, tpMax: 500 });
    const expanded = expandBounds(b, 500);
    expect(expanded.tpMax).toBeGreaterThanOrEqual(expanded.tpMin);
  });
});

// ---------------------------------------------------------------------------
// refineBounds
// ---------------------------------------------------------------------------

describe("refineBounds", () => {
  it("centers bounds around best point", () => {
    const b = bounds({ tpMin: 0, tpMax: 500 });
    const refined = refineBounds({ thetaPause: 250, thetaResume: 200, startDay: 180 }, b, 0.5, 3);
    expect(refined.tpMin).toBeGreaterThan(0);
    expect(refined.tpMax).toBeLessThan(500);
    expect(refined.trMin).toBeGreaterThanOrEqual(0);
    expect(refined.trMax).toBeLessThanOrEqual(refined.tpMax);
  });

  it("ensures tpMin >= 10", () => {
    const b = bounds({ tpMin: 0, tpMax: 50 });
    const refined = refineBounds({ thetaPause: 15, thetaResume: 10, startDay: 0 }, b, 0.5, 3);
    expect(refined.tpMin).toBeGreaterThanOrEqual(10);
  });

  it("ensures trMax >= trMin + minStep", () => {
    const b = bounds({ tpMin: 0, tpMax: 500, trMin: 0, trMax: 500 });
    const refined = refineBounds({ thetaPause: 250, thetaResume: 250, startDay: 180 }, b, 0.5, 3);
    expect(refined.trMax - refined.trMin).toBeGreaterThanOrEqual(3);
  });

  it("ensures dayMax > dayMin", () => {
    const b = bounds({ dayMin: 0, dayMax: 364 });
    const refined = refineBounds({ thetaPause: 250, thetaResume: 200, startDay: 200 }, b, 0.1, 3);
    expect(refined.dayMax).toBeGreaterThan(refined.dayMin);
  });

  it("tpMin <= tpMax", () => {
    const b = bounds({ tpMin: 100, tpMax: 500 });
    const refined = refineBounds({ thetaPause: 300, thetaResume: 200, startDay: 180 }, b, 0.5, 3);
    expect(refined.tpMin).toBeLessThanOrEqual(refined.tpMax);
  });
});

// ---------------------------------------------------------------------------
// findBest
// ---------------------------------------------------------------------------

describe("findBest", () => {
  it("returns null when no points within budget", () => {
    const pts = [
      sweepPoint({ withinBudget: false, co2SavingsPct: 10, score: 0.5 }),
      sweepPoint({ withinBudget: false, co2SavingsPct: 20, score: 0.8 }),
    ];
    expect(findBest(pts, 200)).toBeNull();
  });

  it("returns null when no points have positive savings", () => {
    const pts = [
      sweepPoint({ withinBudget: true, co2SavingsPct: 0, score: 0.5 }),
      sweepPoint({ withinBudget: true, co2SavingsPct: -5, score: 0.3 }),
    ];
    expect(findBest(pts, 200)).toBeNull();
  });

  it("returns the point with highest score among budget+savings valid", () => {
    const pts = [
      sweepPoint({ withinBudget: true, co2SavingsPct: 10, score: 0.5 }),
      sweepPoint({ withinBudget: true, co2SavingsPct: 20, score: 0.9 }),
      sweepPoint({ withinBudget: true, co2SavingsPct: 15, score: 0.7 }),
    ];
    const best = findBest(pts, 200);
    expect(best).not.toBeNull();
    expect(best!.score).toBe(0.9);
    expect(best!.co2SavingsPct).toBe(20);
  });

  it("ignores out-of-budget points even if they have high score", () => {
    const pts = [
      sweepPoint({ withinBudget: false, co2SavingsPct: 50, score: 0.99 }),
      sweepPoint({ withinBudget: true, co2SavingsPct: 5, score: 0.4 }),
    ];
    const best = findBest(pts, 200);
    expect(best).not.toBeNull();
    expect(best!.score).toBe(0.4);
  });

  it("returns null for empty array", () => {
    expect(findBest([], 200)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// dateToDay / dayToDate round-trip
// ---------------------------------------------------------------------------

describe("dateToDay / dayToDate", () => {
  it("round-trips: 01-01", () => {
    expect(dayToDate(dateToDay("01-01"))).toBe("01-01");
  });

  it("round-trips: 06-15", () => {
    expect(dayToDate(dateToDay("06-15"))).toBe("06-15");
  });

  it("round-trips: 12-31", () => {
    expect(dayToDate(dateToDay("12-31"))).toBe("12-31");
  });

  it("dateToDay 01-01 is 0", () => {
    expect(dateToDay("01-01")).toBe(0);
  });

  it("dateToDay 12-31 is 364", () => {
    expect(dateToDay("12-31")).toBe(364);
  });
});

// ---------------------------------------------------------------------------
// initialBounds
// ---------------------------------------------------------------------------

describe("initialBounds", () => {
  it("uses the given thetaPauseMax as tpMax and trMax", () => {
    const b = initialBounds(500);
    expect(b.tpMin).toBe(10);
    expect(b.tpMax).toBe(500);
    expect(b.trMin).toBe(0);
    expect(b.trMax).toBe(500);
  });

  it("covers full year", () => {
    const b = initialBounds(500);
    expect(b.dayMin).toBe(0);
    expect(b.dayMax).toBe(364);
  });
});

// ---------------------------------------------------------------------------
// Interaction: expandBounds followed by generateGrid
// ---------------------------------------------------------------------------

describe("expandBounds + generateGrid integration", () => {
  it("grid produced from expanded bounds is valid", () => {
    const b = bounds({ tpMin: 100, tpMax: 200 });
    const expanded = expandBounds(b, 500);
    const grid = generateGrid(expanded, 5);
    expect(grid.length).toBeGreaterThan(0);
    for (const p of grid) {
      expect(p.thetaPause).toBeGreaterThanOrEqual(expanded.tpMin);
      expect(p.thetaPause).toBeLessThanOrEqual(expanded.tpMax);
      expect(p.thetaResume).toBeGreaterThanOrEqual(0);
      expect(p.thetaResume).toBeLessThanOrEqual(p.thetaPause);
    }
  });
});
