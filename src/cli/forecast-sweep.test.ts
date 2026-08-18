import { describe, it, expect } from "vitest";
import {
  expandConfigs,
  meanStd,
  computeSummary,
  computeGraceLevel,
  degradationFrac,
  runSweepRegion,
  seedMeanBest,
  drift,
  marginRuleSurvives,
  regionalRuleSurvives,
  assertIdentityRegression,
} from "./forecast-sweep";
import type { FixedRow, FixedSummary, ReoptSeedRow, ReoptConfigResult, ReoptBest } from "./forecast-sweep";
import type { CO2Timeline, FullProfile } from "../domain/types";

const COEFFS = { intercept: 0.5, ar: [0.9] };

describe("expandConfigs", () => {
  const configs = expandConfigs({
    families: ["additive", "multiplicative", "delay", "arma", "persistence"],
    levels: [0, 0.25, 0.5, 1, 2, 4],
    horizons: [1, 3, 6, 12, 24, 72],
    sigmaStar: 10,
    sigmaRel: 0.01,
    coeffs: COEFFS,
  });

  it("expands the full 30-config matrix (6 per family)", () => {
    expect(configs).toHaveLength(30);
    const byFamily = (f: string): number => configs.filter((c) => c.family === f).length;
    for (const family of ["additive", "multiplicative", "delay", "arma", "persistence"]) {
      expect(byFamily(family)).toBe(6);
    }
  });

  it("uses absolute sigma = level*sigmaStar for additive", () => {
    const l1 = configs.find((c) => c.family === "additive" && c.param_value === 1);
    const l4 = configs.find((c) => c.family === "additive" && c.param_value === 4);
    expect(l1?.sigma).toBeCloseTo(10, 10);
    expect(l4?.sigma).toBeCloseTo(40, 10);
    expect(l1?.model).toEqual({ type: "additive", sigma: 10 });
  });

  it("uses relative sigma = level*sigmaRel for multiplicative", () => {
    const l1 = configs.find((c) => c.family === "multiplicative" && c.param_value === 1);
    const l4 = configs.find((c) => c.family === "multiplicative" && c.param_value === 4);
    expect(l1?.sigma).toBeCloseTo(0.01, 12);
    expect(l4?.sigma).toBeCloseTo(0.04, 12);
    expect(l1?.model).toEqual({ type: "multiplicative", sigma: 0.01 });
  });

  it("delay uses steps param and null sigma", () => {
    const c = configs.find((c) => c.family === "delay" && c.param_value === 3);
    expect(c?.param).toBe("steps");
    expect(c?.sigma).toBeNull();
    expect(c?.model).toEqual({ type: "delay", steps: 3 });
  });

  it("arma threads calibrated AR(1) coeffs and horizon", () => {
    const c = configs.find((c) => c.family === "arma" && c.param_value === 6);
    expect(c?.param).toBe("horizon");
    expect(c?.sigma).toBeNull();
    expect(c?.model).toEqual({ type: "arma", order: 1, horizon: 6, coeffs: COEFFS });
  });

  it("persistence reuses delay model but labels horizon", () => {
    const c = configs.find((c) => c.family === "persistence" && c.param_value === 72);
    expect(c?.param).toBe("horizon");
    expect(c?.sigma).toBeNull();
    expect(c?.model).toEqual({ type: "delay", steps: 72 });
  });

  it("respects restricted families", () => {
    const sub = expandConfigs({
      families: ["additive"],
      levels: [0, 1],
      horizons: [1, 3],
      sigmaStar: 10,
      sigmaRel: 0.01,
      coeffs: COEFFS,
    });
    expect(sub).toHaveLength(2);
    for (const c of sub) expect(c.family).toBe("additive");
  });
});

describe("meanStd / computeSummary", () => {
  it("computes sample mean/std over seeds", () => {
    const { mean, std } = meanStd([96, 100, 104]);
    expect(mean).toBeCloseTo(100, 12);
    expect(std).toBeCloseTo(4, 12);
  });

  it("returns std 0 for a single value", () => {
    const { mean, std } = meanStd([50]);
    expect(mean).toBe(50);
    expect(std).toBe(0);
  });

  it("aggregates savings, degradation, completed_rate, within_budget_rate", () => {
    const rows: FixedRow[] = [96, 100, 104].map((savings, i) => ({
      region: "DE",
      family: "additive",
      param: "level",
      param_value: 1,
      seed: i + 1,
      sigma: 3.66,
      theta_p: 272,
      theta_r: 268,
      start: "02-01",
      savings,
      overhead: 150,
      score: 0.5,
      num_pauses: 10,
      completed: i < 2,
      within_budget: i > 0,
      savings_perfect: 100,
      degradation_frac: (100 - savings) / 100,
    }));
    const s = computeSummary(100, rows);
    expect(s.s0).toBe(100);
    expect(s.savings_mean).toBeCloseTo(100, 12);
    expect(s.savings_std).toBeCloseTo(4, 12);
    expect(s.delta_s_pp).toBeCloseTo(0, 12);
    expect(s.delta_s_frac).toBeCloseTo(0, 12);
    expect(s.degradation_frac_mean).toBeCloseTo(0, 12);
    expect(s.degradation_frac_std).toBeCloseTo(0.04, 12);
    expect(s.completed_rate).toBeCloseTo(2 / 3, 12);
    expect(s.within_budget_rate).toBeCloseTo(2 / 3, 12);
    expect(s.n_seeds).toBe(3);
  });

  it("guards delta_s_frac when s0 is 0", () => {
    const rows: FixedRow[] = [
      {
        region: "DE", family: "additive", param: "level", param_value: 1, seed: 1,
        sigma: null, theta_p: 272, theta_r: 268, start: "02-01",
        savings: 0, overhead: 0, score: 0.5, num_pauses: 0,
        completed: true, within_budget: true, savings_perfect: 0, degradation_frac: 0,
      },
    ];
    const s = computeSummary(0, rows);
    expect(s.delta_s_frac).toBe(0);
    expect(Number.isFinite(s.delta_s_frac)).toBe(true);
  });
});

describe("degradationFrac", () => {
  it("measures relative savings loss", () => {
    expect(degradationFrac(100, 90)).toBeCloseTo(0.1, 12);
    expect(degradationFrac(100, 100)).toBe(0);
    expect(degradationFrac(100, 110)).toBeCloseTo(-0.1, 12);
  });

  it("guards zero savings_perfect to 0 (no NaN)", () => {
    expect(degradationFrac(0, 0)).toBe(0);
    expect(Number.isNaN(degradationFrac(0, 5))).toBe(false);
  });
});

describe("computeGraceLevel", () => {
  function summaries(degMeans: number[], paramValues: number[]): FixedSummary[] {
    return degMeans.map((d, i) => ({
      family: "additive",
      param: "level",
      param_value: paramValues[i],
      sigma: null,
      s0: 100,
      savings_mean: 0,
      savings_std: 0,
      delta_s_pp: 0,
      delta_s_frac: 0,
      degradation_frac_mean: d,
      degradation_frac_std: 0,
      overhead_mean: 0,
      num_pauses_mean: 0,
      completed_rate: 1,
      within_budget_rate: 1,
      n_seeds: 1,
      graceLevel: 0,
      graceAtMax: false,
    }));
  }

  it("picks the largest value with mean degradation <= 10%", () => {
    const g = computeGraceLevel(summaries([0.05, 0.09, 0.12, 0.2], [0.25, 0.5, 1, 2]));
    expect(g.graceLevel).toBe(0.5);
    expect(g.graceAtMax).toBe(false);
  });

  it("graceAtMax when all tested values are within 10%", () => {
    const g = computeGraceLevel(summaries([0, 0.03, 0.05, 0.08], [0.25, 0.5, 1, 2]));
    expect(g.graceLevel).toBe(2);
    expect(g.graceAtMax).toBe(true);
  });

  it("graceLevel 0 when even the smallest positive value exceeds 10%", () => {
    const g = computeGraceLevel(summaries([0.2, 0.3, 0.5], [0.25, 0.5, 1]));
    expect(g.graceLevel).toBe(0);
    expect(g.graceAtMax).toBe(false);
  });

  it("ignores the trivial param_value 0 entry", () => {
    const g = computeGraceLevel(summaries([0, 0.05, 0.2], [0, 0.25, 1]));
    expect(g.graceLevel).toBe(0.25);
    expect(g.graceAtMax).toBe(false);
  });
});

describe("runSweepRegion (synthetic)", () => {
  const profile: FullProfile = {
    name: "T",
    modelParams: 1e9,
    datasetTokens: 42_000_000,
    gpuCount: 1,
    gpuPowerTrain: 700,
    gpuPowerPause: 60,
    pue: 1,
    checkpointPauseTime: 0,
    checkpointResumeTime: 0,
  };

  function blockTimeline(): CO2Timeline {
    const values: number[] = [];
    for (let i = 0; i < 100; i++) values.push(400);
    for (let i = 0; i < 100; i++) values.push(50);
    for (let i = 0; i < 88; i++) values.push(400);
    const timestamps = values.map((_, i) => new Date(Date.UTC(2025, 0, 1) + i * 300_000).toISOString());
    return { zone: "DE", years: [2025], timestamps, carbonIntensity: values };
  }

  it("level-0 additive/multiplicative rows have degradation_frac === 0 exactly", () => {
    const configs = expandConfigs({
      families: ["additive", "multiplicative"],
      levels: [0, 1],
      horizons: [1],
      sigmaStar: 150,
      sigmaRel: 0.5,
      coeffs: COEFFS,
    });
    const { control, rows } = runSweepRegion({
      region: "DE",
      profile,
      realized: blockTimeline(),
      historicalYears: [2025],
      thetaP: 300,
      thetaR: 100,
      start: "01-01",
      budget: 200,
      seeds: [1, 2, 3],
      configs,
    });

    expect(control.savings).toBeGreaterThan(0);
    const level0 = rows.filter((r) => r.param_value === 0);
    expect(level0.length).toBe(2 * 3);
    for (const row of level0) {
      expect(row.degradation_frac).toBe(0);
      expect(row.savings).toBe(control.savings);
      expect(row.savings_perfect).toBe(control.savings);
    }
    const level1 = rows.filter((r) => r.param_value === 1);
    expect(level1.length).toBe(2 * 3);
    expect(new Set(level1.map((r) => r.savings)).size).toBeGreaterThan(1);
    for (const row of level1) {
      expect(Number.isFinite(row.degradation_frac)).toBe(true);
    }
  });

  it("delay distorts decisions and degrades savings", () => {
    const configs = expandConfigs({
      families: ["persistence"],
      levels: [0],
      horizons: [72],
      sigmaStar: 10,
      sigmaRel: 0.01,
      coeffs: COEFFS,
    });
    const { rows, summary } = runSweepRegion({
      region: "DE",
      profile,
      realized: blockTimeline(),
      historicalYears: [2025],
      thetaP: 300,
      thetaR: 100,
      start: "01-01",
      budget: 200,
      seeds: [1, 2],
      configs,
    });

    expect(rows).toHaveLength(2);
    for (const row of rows) {
      expect(row.degradation_frac).toBeGreaterThan(0);
    }
    expect(summary).toHaveLength(1);
    expect(summary[0].family).toBe("persistence");
    expect(summary[0].param).toBe("horizon");
    expect(summary[0].completed_rate).toBeGreaterThanOrEqual(0);
    expect(summary[0].within_budget_rate).toBeGreaterThanOrEqual(0);
  });

  it("populates graceLevel/graceAtMax on the per-family summaries", () => {
    const configs = expandConfigs({
      families: ["additive", "delay"],
      levels: [0, 0.25, 0.5, 1, 2, 4],
      horizons: [1, 3, 6, 12, 24, 72],
      sigmaStar: 10,
      sigmaRel: 0.01,
      coeffs: COEFFS,
    });
    const { summary } = runSweepRegion({
      region: "DE",
      profile,
      realized: blockTimeline(),
      historicalYears: [2025],
      thetaP: 300,
      thetaR: 100,
      start: "01-01",
      budget: 200,
      seeds: [1],
      configs,
    });

    expect(summary).toHaveLength(12);
    const additive = summary.filter((s) => s.family === "additive");
    const delay = summary.filter((s) => s.family === "delay");
    for (const s of additive) {
      expect(typeof s.graceLevel).toBe("number");
      expect(typeof s.graceAtMax).toBe("boolean");
      expect(s.graceLevel).toBe(additive[0].graceLevel);
    }
    for (const s of delay) {
      expect(s.graceLevel).toBe(delay[0].graceLevel);
      expect(s.graceAtMax).toBe(delay[0].graceAtMax);
    }
  });
});

function mkRow(seed: number, thetaP: number, thetaR: number, savings: number, overhead: number): ReoptSeedRow {
  return {
    seed,
    thetaP,
    thetaR,
    margin: thetaP - thetaR,
    savings,
    overhead,
    score: savings / 100,
    found: true,
  };
}

function mkConfig(rows: ReoptSeedRow[], family = "additive", paramValue = 1): ReoptConfigResult {
  return {
    family,
    param: "level",
    param_value: paramValue,
    sigma: 3.66,
    perSeed: rows,
    best: seedMeanBest(rows),
    foundRate: rows.filter((r) => r.found).length / rows.length,
  };
}

describe("reopt drift helpers", () => {
  const baseline: ReoptBest = { thetaP: 272, thetaR: 268, margin: 4, savings: 43.35, overhead: 174.3, score: 0.7 };

  it("seedMeanBest averages only found seeds and ignores found:false rows", () => {
    const rows: ReoptSeedRow[] = [
      mkRow(1, 280, 260, 40, 190),
      mkRow(2, 290, 255, 42, 180),
      { seed: 3, thetaP: null, thetaR: null, margin: null, savings: null, overhead: null, score: null, found: false },
    ];
    const best = seedMeanBest(rows);
    expect(best).not.toBeNull();
    expect(best?.thetaP).toBeCloseTo(285, 10);
    expect(best?.thetaR).toBeCloseTo(257.5, 10);
    expect(best?.margin).toBeCloseTo(27.5, 10);
    expect(best?.savings).toBeCloseTo(41, 10);
  });

  it("seedMeanBest returns null when every seed is found:false", () => {
    const rows: ReoptSeedRow[] = [1, 2, 3].map((seed) => ({
      seed,
      thetaP: null,
      thetaR: null,
      margin: null,
      savings: null,
      overhead: null,
      score: null,
      found: false,
    }));
    expect(seedMeanBest(rows)).toBeNull();
  });

  it("drift subtracts the baseline field-by-field", () => {
    const best: ReoptBest = { thetaP: 284.3, thetaR: 255.9, margin: 28.4, savings: 43.3, overhead: 175, score: 0.7 };
    const d = drift(best, baseline);
    expect(d.thetaP_drift).toBeCloseTo(12.3, 10);
    expect(d.thetaR_drift).toBeCloseTo(-12.1, 10);
    expect(d.margin_drift).toBeCloseTo(24.4, 10);
  });

  it("marginRuleSurvives: seed-mean margin <= 16 survives, fraction counts seeds only", () => {
    const config = mkConfig([mkRow(1, 272, 262, 40, 190), mkRow(2, 274, 260, 40, 190), mkRow(3, 272, 256, 40, 190)]);
    expect(config.best?.margin).toBeCloseTo(40 / 3, 10);
    expect(marginRuleSurvives(config)).toEqual({ survives: true, seedFraction: 1 });
  });

  it("marginRuleSurvives: mixed seeds -> survives by mean but seedFraction reflects per-seed rule", () => {
    const config = mkConfig([mkRow(1, 272, 262, 40, 190), mkRow(2, 270, 265, 40, 190), mkRow(3, 292, 262, 40, 190)]);
    expect(config.best?.margin).toBeCloseTo(15, 10);
    expect(marginRuleSurvives(config)).toEqual({ survives: true, seedFraction: 2 / 3 });
  });

  it("marginRuleSurvives: fails when seed-mean margin exceeds 16", () => {
    const config = mkConfig([mkRow(1, 290, 262, 40, 190), mkRow(2, 292, 260, 40, 190)]);
    const r = marginRuleSurvives(config);
    expect(r.survives).toBe(false);
    expect(r.seedFraction).toBe(0);
  });

  it("marginRuleSurvives: no valid point -> survives false and fraction 0", () => {
    const config = mkConfig([{ seed: 1, thetaP: null, thetaR: null, margin: null, savings: null, overhead: null, score: null, found: false }]);
    expect(marginRuleSurvives(config)).toEqual({ survives: false, seedFraction: 0 });
  });

  it("regionalRuleSurvives: theta_p within +-50% of baseline survives", () => {
    expect(regionalRuleSurvives(mkConfig([mkRow(1, 250, 200, 40, 190)]), baseline)).toBe(true);
    expect(regionalRuleSurvives(mkConfig([mkRow(1, 450, 300, 40, 190)]), baseline)).toBe(false);
    expect(regionalRuleSurvives(mkConfig([mkRow(1, 136, 100, 40, 190)]), baseline)).toBe(true);
  });

  it("regionalRuleSurvives: no valid point -> false", () => {
    const config = mkConfig([{ seed: 1, thetaP: null, thetaR: null, margin: null, savings: null, overhead: null, score: null, found: false }]);
    expect(regionalRuleSurvives(config, baseline)).toBe(false);
  });
});

describe("identity regression (reopt)", () => {
  const profile: FullProfile = {
    name: "T",
    modelParams: 1e9,
    datasetTokens: 1_000_000_000,
    gpuCount: 1,
    gpuPowerTrain: 700,
    gpuPowerPause: 60,
    pue: 1,
    checkpointPauseTime: 0,
    checkpointResumeTime: 0,
  };

  it("runOptimization with an identity decisionTimeline equals the no-decisionTimeline run", () => {
    const values: number[] = [];
    for (let i = 0; i < 100; i++) values.push(400);
    for (let i = 0; i < 100; i++) values.push(5);
    for (let i = 0; i < 88; i++) values.push(400);
    const timestamps = values.map((_, i) => new Date(Date.UTC(2025, 0, 1) + i * 300_000).toISOString());
    const realized: CO2Timeline = { zone: "DE", years: [2025], timestamps, carbonIntensity: values };

    const options = {
      thetaPauseMax: 500,
      overheadBudgetPct: 200,
      resolution: 2,
      startDateResolution: 1,
      maxIterations: 1,
      minStep: 3,
      shrinkFactor: 0.45,
      alpha: 1,
      fixedStartTime: "01-01",
    };
    expect(() => assertIdentityRegression(profile, realized, [2025], options)).not.toThrow();
  });
});
