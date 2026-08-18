import { describe, expect, it } from "vitest";
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { CalibrationBundle, CalibrationRow } from "../domain/types";
import type { FixedRow, FixedSummary, ReoptRegionResult } from "./forecast-sweep";
import {
  aggregateBand,
  buildDegradation,
  buildDrift,
  buildF1Spec,
  buildF2Spec,
  buildF3Spec,
  buildF4Spec,
  buildRmse,
  configLabel,
  modelLabel,
  normalizeEps,
} from "./plot-forecast";

function seedRow(over: Partial<FixedRow>): FixedRow {
  return {
    region: "DE",
    family: "additive",
    param: "level",
    param_value: 0,
    seed: 1,
    sigma: 0,
    theta_p: 272,
    theta_r: 268,
    start: "02-01",
    savings: 40,
    overhead: 170,
    score: 0.7,
    num_pauses: 100,
    completed: true,
    within_budget: true,
    savings_perfect: 40,
    degradation_frac: 0,
    ...over,
  };
}

function summaryRow(over: Partial<FixedSummary>): FixedSummary {
  return {
    family: "additive",
    param: "level",
    param_value: 0,
    sigma: 0,
    s0: 43,
    savings_mean: 43,
    savings_std: 0,
    delta_s_pp: 0,
    delta_s_frac: 0,
    degradation_frac_mean: 0,
    degradation_frac_std: 0,
    overhead_mean: 174,
    num_pauses_mean: 102,
    completed_rate: 1,
    within_budget_rate: 1,
    n_seeds: 2,
    graceLevel: 4,
    graceAtMax: true,
    ...over,
  };
}

describe("aggregateBand", () => {
  it("emits one row per metric with ±1 std band for band families only", () => {
    const rows = [
      seedRow({ family: "additive", param_value: 1, savings: 40, overhead: 170, score: 0.7 }),
      seedRow({ family: "additive", param_value: 1, savings: 44, overhead: 178, score: 0.71 }),
      seedRow({ family: "delay", param_value: 1 }),
    ];
    const out = aggregateBand(rows);
    const additive = out.filter((r) => r.family === "additive");
    expect(additive).toHaveLength(3);
    const savings = additive.find((r) => r.metric === "savings")!;
    expect(savings.value).toBeCloseTo(42);
    expect(savings.lo).toBeCloseTo(42 - 2.8284, 3);
    expect(savings.hi).toBeCloseTo(42 + 2.8284, 3);
    expect(out.every((r) => r.family !== "delay")).toBe(true);
  });

  it("sorts by level then family", () => {
    const rows = [
      seedRow({ family: "multiplicative", param_value: 0 }),
      seedRow({ family: "additive", param_value: 2 }),
      seedRow({ family: "additive", param_value: 0 }),
    ];
    const out = aggregateBand(rows);
    const keys = out.map((r) => `${r.level}-${r.family}`);
    expect(keys[0]).toBe("0-additive");
    expect(keys[3]).toBe("0-multiplicative");
    expect(keys[keys.length - 1]).toBe("2-additive");
  });
});

describe("buildDegradation", () => {
  it("drops the level-0 baseline row and keeps both band families", () => {
    const summaries = [
      summaryRow({ family: "additive", param_value: 0, degradation_frac_mean: 0 }),
      summaryRow({ family: "additive", param_value: 2, degradation_frac_mean: 0.05 }),
      summaryRow({ family: "multiplicative", param_value: 2, degradation_frac_mean: 0.03 }),
      summaryRow({ family: "delay", param_value: 24, degradation_frac_mean: 0.3 }),
    ];
    const out = buildDegradation(summaries, "DE");
    expect(out).toHaveLength(2);
    expect(out.every((r) => r.level > 0)).toBe(true);
    expect(out.every((r) => r.region === "DE")).toBe(true);
  });
});

describe("configLabel / modelLabel", () => {
  it("labels additive/multiplicative with ×σ* and delay with horizon", () => {
    expect(configLabel("additive", 1)).toBe("additive 1×σ*");
    expect(configLabel("multiplicative", 0.5)).toBe("multiplicative 0.5×σ*");
    expect(configLabel("delay", 6)).toBe("delay 6");
  });

  it("maps calibration rows to persistence / AR(1) / AR(7)", () => {
    const persistence: CalibrationRow = { horizon: 1, order: 1, model: "persistence", rmse: 4, mae: 2, mape: 0.01 };
    const ar1: CalibrationRow = { horizon: 1, order: 1, model: "ar", rmse: 4, mae: 2, mape: 0.01 };
    const ar7: CalibrationRow = { horizon: 1, order: 7, model: "ar", rmse: 2, mae: 1, mape: 0.01 };
    expect(modelLabel(persistence)).toBe("persistence");
    expect(modelLabel(ar1)).toBe("AR(1)");
    expect(modelLabel(ar7)).toBe("AR(7)");
  });
});

describe("buildDrift", () => {
  const result: ReoptRegionResult = {
    region: "DE",
    model: "Deepseek",
    year: 2025,
    budget: 200,
    start: "02-01",
    seeds: [1, 2, 3],
    optimizer: { resolution: 10, iterations: 6, tpMax: 800 },
    baseline: { thetaP: 272.37, thetaR: 267.73, margin: 4.64, savings: 43.35, overhead: 174.3, score: 0.716 },
    configs: [
      {
        family: "additive",
        param: "level",
        param_value: 1,
        sigma: 1,
        perSeed: [],
        best: { thetaP: 278.5, thetaR: 261.8, margin: 16.7, savings: 41, overhead: 178, score: 0.7 },
        foundRate: 1,
      },
      { family: "delay", param: "steps", param_value: 6, sigma: null, perSeed: [], best: null, foundRate: 0 },
    ],
  };

  it("builds a segment + point per found config and skips null best", () => {
    const d = buildDrift(result);
    expect(d.segments).toHaveLength(1);
    expect(d.points).toHaveLength(1);
    expect(d.segments[0].label).toBe("additive 1×σ*");
    expect(d.segments[0]).toMatchObject({ x: 272.37, y: 267.73, x2: 278.5, y2: 261.8 });
    expect(d.baseline.thetaP).toBe(272.37);
    expect(d.domain[0]).toBeLessThan(d.domain[1]);
  });
});

describe("buildRmse", () => {
  const bundle: CalibrationBundle = {
    region: "DE",
    trainYears: [2022, 2023, 2024],
    testYear: 2025,
    trainMean: 100,
    trainStd: 20,
    trainCv: 0.2,
    lag1AutoCorr: 0.999,
    lag2AutoCorr: 0.99,
    sigmaStar: 3.6,
    orders: { "1": { coeffs: { intercept: 0, ar: [0.9] }, innovationStd: 3.6 } },
    evaluation: [
      { horizon: 1, order: 1, model: "ar", rmse: 4.2, mae: 1.7, mape: 0.006 },
      { horizon: 1, order: 1, model: "persistence", rmse: 4.21, mae: 1.7, mape: 0.006 },
      { horizon: 1, order: 7, model: "ar", rmse: 2.3, mae: 1, mape: 0.003 },
      { horizon: 1, order: 7, model: "persistence", rmse: 4.21, mae: 1.7, mape: 0.006 },
      { horizon: 72, order: 1, model: "ar", rmse: 113, mae: 60, mape: 0.5 },
    ],
  };

  it("builds persistence + AR(1) + AR(7) series and dedupes persistence across orders", () => {
    const out = buildRmse(bundle);
    expect(out.map((r) => r.model).sort()).toEqual(["AR(1)", "AR(1)", "AR(7)", "persistence"]);
    expect(out).toHaveLength(4);
    const persistence = out.filter((r) => r.model === "persistence");
    expect(persistence).toHaveLength(1);
    expect(persistence[0].rmse).toBeCloseTo(4.21);
  });
});

describe("spec builders", () => {
  it("f1 embeds band rows and facets by region and metric", () => {
    const rows = [
      { region: "DE", family: "additive", level: 1, metric: "savings" as const, value: 42, lo: 40, hi: 44 },
      { region: "DE", family: "multiplicative", level: 1, metric: "savings" as const, value: 43, lo: 41, hi: 45 },
    ];
    const spec = buildF1Spec(rows);
    expect(spec.facet).toBeDefined();
    const data = spec.data as { values: unknown[] };
    expect(data.values).toHaveLength(2);
  });

  it("f2 uses a log x-scale and a 10% grace rule", () => {
    const spec = buildF2Spec([{ region: "DE", family: "additive", level: 0.25, degradation: 0.001 }]);
    const layers = (spec.spec as { layer: Record<string, unknown>[] }).layer;
    const rules = layers.filter((l) => (l.mark as { type: string }).type === "rule");
    expect(rules).toHaveLength(1);
    expect(layers.some((l) => (l.mark as { type: string }).type === "text")).toBe(true);
  });

  it("f3 hconcats one panel per region with baseline point", () => {
    const spec = buildF3Spec([
      {
        region: "DE",
        baseline: { label: "baseline", family: "baseline", thetaP: 272, thetaR: 268 },
        segments: [],
        points: [],
        domain: [200, 300],
      },
    ]);
    const children = spec.hconcat as unknown[];
    expect(children).toHaveLength(1);
  });

  it("f4 facets by region and carries one line per model", () => {
    const spec = buildF4Spec([{ region: "DE", horizon: 1, model: "persistence", rmse: 4.2 }]);
    const data = spec.data as { values: unknown[] };
    expect(data.values).toHaveLength(1);
    expect(spec.facet).toBeDefined();
  });
});

describe("normalizeEps", () => {
  it("strips the nondeterministic CreationDate header line", () => {
    const dir = mkdtempSync(join(tmpdir(), "tge-eps-"));
    const file = join(dir, "fig.eps");
    const eps = "%!PS-Adobe-3.0 EPSF-3.0\n%%Creator: cairo\n%%CreationDate: Tue Aug 18 13:21:02 2026\n...body...\n";
    writeFileSync(file, eps, "utf-8");
    normalizeEps(file);
    expect(readFileSync(file, "utf-8")).toBe("%!PS-Adobe-3.0 EPSF-3.0\n%%Creator: cairo\n...body...\n");
    rmSync(dir, { recursive: true, force: true });
  });

  it("leaves EPS without a CreationDate line untouched", () => {
    const dir = mkdtempSync(join(tmpdir(), "tge-eps-"));
    const file = join(dir, "fig.eps");
    const eps = "%!PS-Adobe-3.0 EPSF-3.0\n%%Creator: cairo\n...body...\n";
    writeFileSync(file, eps, "utf-8");
    normalizeEps(file);
    expect(readFileSync(file, "utf-8")).toBe(eps);
    rmSync(dir, { recursive: true, force: true });
  });
});
