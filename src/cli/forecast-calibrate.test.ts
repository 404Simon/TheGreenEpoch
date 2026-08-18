import { describe, it, expect } from "vitest";
import { gaussian, mulberry32 } from "../domain/forecast";
import { computeAutocorr, evaluateHorizon, fitArPooled, innovationStd, runCalibration } from "./forecast-calibrate";

function ar1Series(phi: number, mu: number, n: number, seed: number): number[] {
  const rand = mulberry32(seed);
  const series: number[] = [mu];
  for (let t = 1; t < n; t++) {
    series.push(mu + phi * (series[t - 1] - mu) + gaussian(rand));
  }
  return series;
}

describe("fitArPooled", () => {
  it("recovers a known AR(1) coefficient across pooled years", () => {
    const years = [ar1Series(0.9, 500, 2000, 1), ar1Series(0.9, 500, 2000, 2), ar1Series(0.9, 500, 2000, 3)];
    const fit = fitArPooled(years, 1);
    expect(Math.abs(fit.ar[0] - 0.9)).toBeLessThan(0.05);
    expect(Math.abs(fit.intercept - 50)).toBeLessThan(5);
  });

  it("recovers the innovation scale of the pooled fit", () => {
    const years = [ar1Series(0.9, 500, 2000, 11), ar1Series(0.9, 500, 2000, 12)];
    const coeffs = fitArPooled(years, 1);
    const sig = innovationStd(years, 1, coeffs);
    expect(Math.abs(sig - 1)).toBeLessThan(0.15);
  });

  it("throws on constant (degenerate) input", () => {
    expect(() => fitArPooled([[5, 5, 5, 5, 5], [5, 5, 5, 5, 5]], 1)).toThrow();
  });
});

describe("computeAutocorr", () => {
  it("is ~0 at a 90° phase shift of a sine", () => {
    const series = Array.from({ length: 810 }, (_, t) => Math.sin((2 * Math.PI * t) / 360));
    const r = computeAutocorr(series, 90);
    expect(Math.abs(r)).toBeLessThan(0.05);
  });

  it("is ~1 at lag 1 for a constant-shifted smooth series", () => {
    const series = Array.from({ length: 810 }, (_, t) => 500 + Math.sin((2 * Math.PI * t) / 360));
    const r = computeAutocorr(series, 1);
    expect(Math.abs(r)).toBeGreaterThan(0.99);
  });
});

describe("evaluateHorizon", () => {
  it("persistence RMSE grows with horizon on a persistent series", () => {
    const rand = mulberry32(7);
    const series: number[] = [0];
    for (let t = 1; t < 2000; t++) series.push(series[t - 1] + 0.001 * gaussian(rand));
    const coeffs = { intercept: 0, ar: [0.999] };
    const h1 = evaluateHorizon(coeffs, series, 1);
    const h72 = evaluateHorizon(coeffs, series, 72);
    expect(h1.persistence.rmse).toBeLessThan(h72.persistence.rmse);
  });

  it("returns finite metrics on a constant-zero series (MAPE guard)", () => {
    const series = new Array<number>(200).fill(0);
    const coeffs = { intercept: 0, ar: [0.5] };
    const ev = evaluateHorizon(coeffs, series, 3);
    for (const m of [ev.ar, ev.persistence]) {
      expect(Number.isFinite(m.rmse)).toBe(true);
      expect(Number.isFinite(m.mae)).toBe(true);
      expect(Number.isFinite(m.mape)).toBe(true);
      expect(m.rmse).toBe(0);
      expect(m.mae).toBe(0);
      expect(m.mape).toBe(0);
    }
  });
});

describe("runCalibration", () => {
  it("returns the full bundle schema on synthetic series", () => {
    const trainSeries = [ar1Series(0.9, 500, 2000, 21), ar1Series(0.9, 500, 2000, 22), ar1Series(0.9, 500, 2000, 23)];
    const testSeries = ar1Series(0.9, 500, 1000, 24);
    const bundle = runCalibration("XX", [2022, 2023, 2024], 2025, [1], [1, 3], trainSeries, testSeries);

    expect(bundle.region).toBe("XX");
    expect(bundle.trainYears).toEqual([2022, 2023, 2024]);
    expect(bundle.testYear).toBe(2025);
    expect(typeof bundle.trainMean).toBe("number");
    expect(typeof bundle.trainStd).toBe("number");
    expect(typeof bundle.trainCv).toBe("number");
    expect(typeof bundle.lag1AutoCorr).toBe("number");
    expect(typeof bundle.lag2AutoCorr).toBe("number");
    expect(bundle.sigmaStar).toBe(bundle.orders["1"].innovationStd);
    expect(bundle.orders["1"].coeffs.ar).toHaveLength(1);
    expect(typeof bundle.orders["1"].innovationStd).toBe("number");
    expect(bundle.evaluation).toHaveLength(4);
    for (const row of bundle.evaluation) {
      expect(typeof row.horizon).toBe("number");
      expect(typeof row.order).toBe("number");
      expect(["ar", "persistence"]).toContain(row.model);
      expect(typeof row.rmse).toBe("number");
      expect(typeof row.mae).toBe("number");
      expect(typeof row.mape).toBe("number");
      expect(Number.isFinite(row.mape)).toBe(true);
    }
    expect(bundle.evaluation.map((r) => r.model).sort()).toEqual(["ar", "ar", "persistence", "persistence"]);
  });
});
