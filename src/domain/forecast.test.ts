import { describe, it, expect } from "vitest";
import {
  mulberry32,
  gaussian,
  sampleNormal,
  sampleLognormal,
  fitAr,
  forecastInnovationStd,
  applyForecast,
} from "./forecast";
import type { CO2Timeline, ForecastModel } from "./types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTimeline(values: number[]): CO2Timeline {
  return {
    zone: "DE",
    years: [2022],
    timestamps: values.map((_, i) => new Date(Date.UTC(2022, 0, 1) + i * 300_000).toISOString()),
    carbonIntensity: values,
  };
}

const realized = makeTimeline([100, 200, 150, 300, 250, 400, 350, 500, 450, 600, 550, 200, 180, 320, 410]);

// ---------------------------------------------------------------------------
// mulberry32
// ---------------------------------------------------------------------------

describe("mulberry32", () => {
  it("is deterministic for the same seed", () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    const seqA = Array.from({ length: 10 }, () => a());
    const seqB = Array.from({ length: 10 }, () => b());
    expect(seqA).toEqual(seqB);
  });

  it("differs across seeds", () => {
    const a = mulberry32(1);
    const b = mulberry32(2);
    const seqA = Array.from({ length: 10 }, () => a());
    const seqB = Array.from({ length: 10 }, () => b());
    expect(seqA).not.toEqual(seqB);
  });

  it("returns floats in [0, 1)", () => {
    const rand = mulberry32(7);
    for (let i = 0; i < 1000; i++) {
      const v = rand();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
});

// ---------------------------------------------------------------------------
// gaussian / sampleNormal / sampleLognormal
// ---------------------------------------------------------------------------

describe("gaussian / sampleNormal / sampleLognormal", () => {
  it("mean ~0 and std ~1 over 10000 draws", () => {
    const rand = mulberry32(123);
    const draws = Array.from({ length: 10000 }, () => gaussian(rand));
    const mean = draws.reduce((a, b) => a + b, 0) / draws.length;
    const variance = draws.reduce((a, b) => a + (b - mean) ** 2, 0) / draws.length;
    expect(Math.abs(mean)).toBeLessThan(0.05);
    expect(Math.sqrt(variance)).toBeGreaterThan(0.9);
    expect(Math.sqrt(variance)).toBeLessThan(1.1);
  });

  it("sampleNormal shifts mean by mu", () => {
    const rand = mulberry32(5);
    const draws = Array.from({ length: 10000 }, () => sampleNormal(rand, 100, 10));
    const mean = draws.reduce((a, b) => a + b, 0) / draws.length;
    expect(Math.abs(mean - 100)).toBeLessThan(0.5);
  });

  it("sampleLognormal returns positive values", () => {
    const rand = mulberry32(6);
    const draws = Array.from({ length: 1000 }, () => sampleLognormal(rand, 0, 0.5));
    for (const v of draws) expect(v).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// fitAr / forecastInnovationStd
// ---------------------------------------------------------------------------

describe("fitAr", () => {
  it("recovers AR(1) coefficients from a synthetic series", () => {
    const phi = 0.9;
    const mu = 50;
    const rand = mulberry32(99);
    const series: number[] = [mu];
    for (let t = 1; t < 2000; t++) {
      const prev = series[t - 1];
      series.push(mu + phi * (prev - mu) + gaussian(rand));
    }
    const fit = fitAr(series, 1);
    expect(Math.abs(fit.ar[0] - phi)).toBeLessThan(0.05);
    expect(Math.abs(fit.intercept - mu * (1 - phi))).toBeLessThan(5);
  });

  it("recovers innovation scale via forecastInnovationStd", () => {
    const rand = mulberry32(31);
    const series: number[] = [0];
    for (let t = 1; t < 2000; t++) {
      const prev = series[t - 1];
      series.push(0.5 * prev + gaussian(rand));
    }
    const sigma = forecastInnovationStd(series);
    expect(Math.abs(sigma - 1)).toBeLessThan(0.15);
  });

  it("throws on series too short for the order", () => {
    expect(() => fitAr([1, 2], 2)).toThrow();
  });

  it("throws on constant series (singular matrix)", () => {
    expect(() => fitAr([5, 5, 5, 5, 5], 1)).toThrow();
  });

  it("throws on non-finite values", () => {
    expect(() => fitAr([1, NaN, 2, 3], 1)).toThrow();
  });

  it("throws on invalid order", () => {
    expect(() => fitAr([1, 2, 3], 0)).toThrow();
  });
});

// ---------------------------------------------------------------------------
// applyForecast
// ---------------------------------------------------------------------------

describe("applyForecast", () => {
  it("identity returns bit-identical realized values", () => {
    const out = applyForecast(realized, { type: "identity" }, 1);
    expect(out.zone).toBe(realized.zone);
    expect(out.years).toEqual(realized.years);
    expect(out.timestamps).toEqual(realized.timestamps);
    expect(out.carbonIntensity).toEqual(realized.carbonIntensity);
  });

  it("is deterministic for the same model and seed", () => {
    const a = applyForecast(realized, { type: "additive", sigma: 20 }, 7);
    const b = applyForecast(realized, { type: "additive", sigma: 20 }, 7);
    expect(a.carbonIntensity).toEqual(b.carbonIntensity);
  });

  it("differs across seeds", () => {
    const a = applyForecast(realized, { type: "additive", sigma: 20 }, 1);
    const b = applyForecast(realized, { type: "additive", sigma: 20 }, 2);
    expect(a.carbonIntensity).not.toEqual(b.carbonIntensity);
  });

  it("uses the same z sequence regardless of sigma", () => {
    const base = mulberry32(42);
    const z = Array.from({ length: realized.carbonIntensity.length }, () => gaussian(base));
    const low = applyForecast(realized, { type: "additive", sigma: 1 }, 42);
    const high = applyForecast(realized, { type: "additive", sigma: 100 }, 42);
    for (let t = 0; t < z.length; t++) {
      expect(low.carbonIntensity[t]).toBeCloseTo(Math.max(0, realized.carbonIntensity[t] + z[t]), 10);
      expect(high.carbonIntensity[t]).toBeCloseTo(Math.max(0, realized.carbonIntensity[t] + 100 * z[t]), 10);
    }
  });

  it("clamps additive forecasts at zero", () => {
    const out = applyForecast(realized, { type: "additive", sigma: 1e6 }, 3);
    for (const v of out.carbonIntensity) expect(v).toBeGreaterThanOrEqual(0);
    expect(Math.min(...out.carbonIntensity)).toBe(0);
  });

  it("clamps multiplicative forecasts at zero", () => {
    const out = applyForecast(realized, { type: "multiplicative", sigma: 1e6 }, 3);
    for (const v of out.carbonIntensity) expect(v).toBeGreaterThanOrEqual(0);
  });

  it("delay head-fallback copies realized values", () => {
    const out = applyForecast(realized, { type: "delay", steps: 3 }, 1);
    for (let t = 0; t < 3; t++) {
      expect(out.carbonIntensity[t]).toBe(realized.carbonIntensity[t]);
    }
    for (let t = 3; t < realized.carbonIntensity.length; t++) {
      expect(out.carbonIntensity[t]).toBe(realized.carbonIntensity[t - 3]);
    }
  });

  it("arma head-fallback copies realized values", () => {
    const model: ForecastModel = { type: "arma", order: 2, horizon: 3, coeffs: { intercept: 0, ar: [0.5, 0.25] } };
    const out = applyForecast(realized, model, 1);
    const lag = 3 + 2 - 1;
    for (let t = 0; t < lag; t++) {
      expect(out.carbonIntensity[t]).toBe(realized.carbonIntensity[t]);
    }
  });

  it("arma without coeffs fits AR on the realized series", () => {
    const model: ForecastModel = { type: "arma", order: 1, horizon: 1 };
    const out = applyForecast(realized, model, 1);
    expect(out.carbonIntensity.length).toBe(realized.carbonIntensity.length);
    const fit = fitAr(realized.carbonIntensity, 1);
    for (let t = 1; t < realized.carbonIntensity.length; t++) {
      expect(out.carbonIntensity[t]).toBeCloseTo(fit.intercept + fit.ar[0] * realized.carbonIntensity[t - 1], 10);
    }
  });
});
