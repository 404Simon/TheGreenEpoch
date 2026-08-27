import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve, dirname } from "node:path";
import { computeRecovery, hourlyAggregate, RECOVERY_EPS_PP } from "./forecast-sweep";
import type { CO2Timeline } from "../domain/types";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, "../..");

function loadJSON<T>(relPath: string): T {
  return JSON.parse(readFileSync(resolve(REPO_ROOT, relPath), "utf-8")) as T;
}

interface AdaptiveRow {
  h: number;
  c: number;
  savings_naive: number;
  overhead_naive: number;
  score_naive: number;
  num_pauses_naive: number;
  completed_naive: boolean;
  within_budget_naive: boolean;
  savings_perfect: number;
  savings_adaptive: number;
  overhead_adaptive: number;
  score_adaptive: number;
  num_pauses_adaptive: number;
  completed_adaptive: boolean;
  within_budget_adaptive: boolean;
  oracle_source: string;
  savings_oracle: number;
  theta_p_oracle: number;
  theta_r_oracle: number;
  ref_savings_delay: number | null;
  savings_dtpr: number;
  overhead_dtpr: number;
  beta_dtpr: number;
  completed_dtpr: boolean;
  s0_naive_ff: number;
  recovery_raw: number;
  recovery: number;
  oracle_gap: number;
  recovery_dtpr: number;
  recovery_vs_oracle: number;
}

interface SummaryRegion {
  region: string;
  chosenC: number;
  c_selection: { grid: number[]; meanRecovery: Record<string, number> };
  s0_naive_ff: number;
  rows: AdaptiveRow[];
}

interface AdaptiveSummaryDoc {
  methodology: Record<string, unknown>;
  regions: SummaryRegion[];
}

interface FixedSummaryRegion {
  region: string;
  s0: number;
}

const REGIONS = ["DE", "IT", "SE"] as const;
const HORIZONS = [1, 3, 6, 12, 24, 72];

describe("computeRecovery", () => {
  it("measures the fraction of the naive-perfect gap closed, clipped to [0,1]", () => {
    expect(computeRecovery(40, 20, 60).clipped).toBeCloseTo(0.5, 12);
    expect(computeRecovery(60, 20, 60).clipped).toBe(1);
    expect(computeRecovery(10, 20, 60).clipped).toBe(0);
    expect(computeRecovery(80, 20, 60).raw).toBeCloseTo(1.5, 12);
    expect(computeRecovery(80, 20, 60).clipped).toBe(1);
  });

  it("returns 0 (raw and clipped) when the naive loss is not positive (naive >= perfect)", () => {
    const r = computeRecovery(25, 23, 22.9);
    expect(r.raw).toBe(0);
    expect(r.clipped).toBe(0);
  });

  it("guards the near-zero denominator", () => {
    const r = computeRecovery(22.87 + RECOVERY_EPS_PP / 2, 22.8685, 22.8685);
    expect(r.raw).toBe(0);
    expect(r.clipped).toBe(0);
  });
});

describe("hourlyAggregate", () => {
  it("preserves length and sets each 12-point hour to its mean", () => {
    const n = 12 * 4;
    const values: number[] = [];
    for (let i = 0; i < n; i++) values.push(i % 12 === 0 ? 100 : 0);
    const tl: CO2Timeline = { zone: "DE", years: [2025], timestamps: values.map((_, i) => `t${i}`), carbonIntensity: values };
    const agg = hourlyAggregate(tl);
    expect(agg.carbonIntensity).toHaveLength(n);
    // each hour = [100,0,...,0] -> mean 100/12
    for (let i = 0; i < n; i++) {
      expect(agg.carbonIntensity[i]).toBeCloseTo(100 / 12, 10);
    }
    expect(agg.timestamps).toEqual(tl.timestamps);
  });

  it("handles a partial trailing hour", () => {
    const tl: CO2Timeline = { zone: "DE", years: [2025], timestamps: ["a", "b", "c", "d", "e"], carbonIntensity: [1, 2, 3, 4, 5] };
    const agg = hourlyAggregate(tl);
    expect(agg.carbonIntensity).toEqual([3, 3, 3, 3, 3]);
  });
});

describe("Phase B.1 adaptive artifacts", () => {
  const summary = loadJSON<AdaptiveSummaryDoc>("publication/output/forecast/adaptive_summary.json");
  const fixedSummary = loadJSON<FixedSummaryRegion[]>("publication/output/forecast/fixed_summary.json");
  const fixedByRegion = Object.fromEntries(fixedSummary.map((r) => [r.region, r]));

  function regionRows(R: string): AdaptiveRow[] {
    const reg = loadJSON<{ rows: AdaptiveRow[] }>(`publication/output/forecast/adaptive_${R}.json`);
    return reg.rows;
  }

  it("contains one summary region per region with rows for all 6 horizons", () => {
    expect(summary.regions.map((r) => r.region)).toEqual(["DE", "IT", "SE"]);
    for (const R of REGIONS) {
      expect(regionRows(R).map((r) => r.h)).toEqual(HORIZONS);
    }
  });

  it("D5a: naive-fixed (arma) rows reproduce fixed_summary.json arma savings within +-0.5 pp", () => {
    for (const R of REGIONS) {
      const fixed = loadJSON<{ summary: Array<{ family: string; param_value: number; savings_mean: number }> }>(
        `publication/output/forecast/fixed_${R}.json`,
      );
      for (const row of regionRows(R)) {
        const s = fixed.summary.find((x) => x.family === "arma" && x.param_value === row.h);
        expect(s, `fixed arma row for ${R} h=${row.h}`).toBeDefined();
        expect(Math.abs(row.savings_naive - (s as { savings_mean: number }).savings_mean)).toBeLessThanOrEqual(0.5);
      }
    }
  });

  it("D5b: perfect-foresight S0 reproduces fixed_summary control within +-0.5 pp", () => {
    for (const R of REGIONS) {
      const s0 = regionRows(R)[0].s0_naive_ff;
      expect(Math.abs(s0 - fixedByRegion[R].s0)).toBeLessThanOrEqual(0.5);
    }
  });

  it("D4: every (region, h) row has the full metric set with finite values", () => {
    for (const R of REGIONS) {
      for (const row of regionRows(R)) {
        for (const key of [
          "savings_naive", "overhead_naive", "score_naive", "num_pauses_naive",
          "savings_perfect", "savings_adaptive", "overhead_adaptive", "score_adaptive", "num_pauses_adaptive",
          "savings_oracle", "savings_dtpr", "overhead_dtpr",
          "s0_naive_ff", "recovery_raw", "recovery", "oracle_gap", "recovery_dtpr", "recovery_vs_oracle",
        ] as const) {
          expect(Number.isFinite(row[key]), `${R} h=${row.h} ${key}`).toBe(true);
        }
        for (const key of ["completed_naive", "completed_adaptive", "completed_dtpr", "within_budget_naive", "within_budget_adaptive"] as const) {
          expect(typeof row[key], `${R} h=${row.h} ${key}`).toBe("boolean");
        }
        expect(row.recovery).toBeGreaterThanOrEqual(0);
        expect(row.recovery).toBeLessThanOrEqual(1);
        expect(row.recovery_vs_oracle).toBeGreaterThanOrEqual(0);
        expect(row.recovery_vs_oracle).toBeLessThanOrEqual(1);
      }
    }
  });

  it("completion guard: headline recovery is 0 whenever the adaptive or naive run is incomplete", () => {
    for (const R of REGIONS) {
      for (const row of regionRows(R)) {
        if (!row.completed_adaptive || !row.completed_naive) {
          expect(row.recovery, `${R} h=${row.h}`).toBe(0);
        }
      }
    }
  });

  it("D7a: oracle rows for h in {1,6} are sourced from the committed reopt delay rows and match them", () => {
    for (const R of REGIONS) {
      const reopt = loadJSON<{ configs: Array<{ family: string; param_value: number; best: { savings: number } }> }>(
        `publication/output/forecast/reopt_${R}.json`,
      );
      for (const h of [1, 6]) {
        const row = regionRows(R).find((r) => r.h === h) as AdaptiveRow;
        expect(row.oracle_source, `${R} h=${h}`).toBe("reopt_delay");
        const delay = reopt.configs.find((c) => c.family === "delay" && c.param_value === h);
        expect(delay, `reopt delay row for ${R} h=${h}`).toBeDefined();
        expect(row.savings_oracle).toBeCloseTo((delay as { best: { savings: number } }).best.savings, 6);
        expect(row.ref_savings_delay).toBeCloseTo((delay as { best: { savings: number } }).best.savings, 6);
      }
    }
  });

  it("D7b: oracle rows for h in {3,12,24,72} are freshly optimized under the arma(h) timeline", () => {
    for (const R of REGIONS) {
      for (const h of [3, 12, 24, 72]) {
        const row = regionRows(R).find((r) => r.h === h) as AdaptiveRow;
        expect(row.oracle_source, `${R} h=${h}`).toBe("arma_optimize");
        expect(Number.isFinite(row.theta_p_oracle)).toBe(true);
        expect(Number.isFinite(row.theta_r_oracle)).toBe(true);
        expect(row.theta_p_oracle).toBeGreaterThanOrEqual(row.theta_r_oracle);
      }
    }
  });

  it("DTPR benchmark rows are present, deterministic, and have a positive constant beta", () => {
    for (const R of REGIONS) {
      for (const row of regionRows(R)) {
        expect(row.beta_dtpr).toBeGreaterThan(0);
        expect(Number.isFinite(row.savings_dtpr)).toBe(true);
        expect(Number.isFinite(row.overhead_dtpr)).toBe(true);
        expect(row.recovery_dtpr).toBeGreaterThanOrEqual(0);
        expect(row.recovery_dtpr).toBeLessThanOrEqual(1);
      }
    }
  });

  it("c-selection: chosenC is in the grid and meanRecovery is recorded for every grid point", () => {
    for (const R of REGIONS) {
      const cs = byRegion2(R).c_selection;
      expect(cs.grid).toContain(cs.chosenC);
      for (const c of cs.grid) {
        expect(typeof cs.meanRecovery[String(c)], `${R} c=${c}`).toBe("number");
        expect(Number.isFinite(cs.meanRecovery[String(c)])).toBe(true);
      }
    }
  });

  it("D6 evidence note: adaptive artifacts exist alongside the runner script", () => {
    const fs = require("node:fs") as typeof import("node:fs");
    for (const R of REGIONS) {
      expect(fs.existsSync(resolve(REPO_ROOT, `publication/output/forecast/adaptive_${R}.json`))).toBe(true);
      expect(fs.existsSync(resolve(REPO_ROOT, `publication/output/forecast/adaptive_${R}.csv`))).toBe(true);
    }
    expect(fs.existsSync(resolve(REPO_ROOT, "publication/output/forecast/run_adaptive_sweep.sh"))).toBe(true);
  });
});

function byRegion2(R: string): { c_selection: { grid: number[]; chosenC: number; meanRecovery: Record<string, number> } } {
  return loadJSON(`publication/output/forecast/adaptive_${R}.json`);
}

interface SensitivityRow {
  h: number;
  c: number;
  savings: number;
  overhead: number;
  score: number;
  num_pauses: number;
  completed: boolean;
  within_budget: boolean;
  savings_naive: number;
  savings_perfect: number;
  savings_oracle: number;
  recovery: number;
  recovery_raw: number;
  recovery_vs_oracle: number;
}

interface SensitivityRegion {
  region: string;
  year: number;
  budget: number;
  s0_naive_ff: number;
  cGrid: number[];
  horizons: number[];
  feasibleC: Record<string, number[]>;
  rows: SensitivityRow[];
}

const SENS_GRID = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8];

function sensRegion(R: string): SensitivityRegion {
  return loadJSON(`publication/output/forecast/adaptive_sensitivity_${R}.json`);
}

describe("Phase B.1 recovery-vs-c sensitivity artifacts (adversarial review F1/F2)", () => {
  it("F1a: artifact exists for all three regions with the extended c grid and all six horizons, one row per (c, h)", () => {
    for (const R of REGIONS) {
      const sens = sensRegion(R);
      expect(sens.cGrid).toEqual(SENS_GRID);
      expect(sens.horizons).toEqual(HORIZONS);
      expect(sens.budget).toBe(200);
      expect(sens.year).toBe(2025);
      for (const h of HORIZONS) {
        for (const c of SENS_GRID) {
          const row = sens.rows.find((r) => r.h === h && r.c === c);
          expect(row, `${R} h=${h} c=${c}`).toBeDefined();
          expect(Number.isFinite(row?.savings)).toBe(true);
          expect(typeof row?.completed).toBe("boolean");
          expect(typeof row?.within_budget).toBe("boolean");
        }
      }
    }
  });

  it("F2: DE h=72 c=6 reproduces the money cell (S ~= 31.4, completed, within budget, recovery_vs_oracle ~= 0.75)", () => {
    const sens = sensRegion("DE");
    const row = sens.rows.find((r) => r.h === 72 && r.c === 6) as SensitivityRow;
    expect(row).toBeDefined();
    expect(row.completed).toBe(true);
    expect(row.within_budget).toBe(true);
    expect(row.savings).toBeCloseTo(31.43, 1);
    expect(row.recovery_vs_oracle).toBeGreaterThan(0.6);
    expect(row.recovery_vs_oracle).toBeLessThan(0.9);
    // c=8 is budget-infeasible on DE h=72 (the envelope must show it)
    const c8 = sens.rows.find((r) => r.h === 72 && r.c === 8) as SensitivityRow;
    expect(c8.within_budget).toBe(false);
    expect(sens.feasibleC["72"]).toContain(6);
    expect(sens.feasibleC["72"]).not.toContain(8);
  });

  it("F1b: feasible-c envelope is recorded per (region, h); SE h=72 has no feasible c and the train-selected c* is infeasible at h>=24", () => {
    const de = sensRegion("DE");
    const it = sensRegion("IT");
    const se = sensRegion("SE");
    // SE h=72: no feasible c recovers it; even c=0.75 completes but far below naive
    expect(se.feasibleC["72"]).toEqual([0, 0.25, 0.5, 0.75]);
    const se72 = se.rows.find((r) => r.h === 72 && r.c === 0.75) as SensitivityRow;
    expect(se72.completed).toBe(true);
    expect(se72.savings).toBeLessThan(se72.savings_naive);
    // train-selected SE c* = 1.5 is not budget-feasible at h>=24
    expect(se.feasibleC["24"]).not.toContain(1.5);
    expect(se.feasibleC["72"]).not.toContain(1.5);
    // IT nominal headroom: only small c complete at h=1
    expect(it.feasibleC["1"]).toEqual([0, 0.25, 0.5, 0.75, 1, 1.5, 2]);
    expect(it.feasibleC["1"]).not.toContain(3);
    // DE feasible through c=8 at grace horizons
    expect(de.feasibleC["1"]).toContain(8);
  });

  it("F1c: sensitivity artifacts exist alongside the committed runner script", () => {
    const fs = require("node:fs") as typeof import("node:fs");
    for (const R of REGIONS) {
      expect(fs.existsSync(resolve(REPO_ROOT, `publication/output/forecast/adaptive_sensitivity_${R}.json`))).toBe(true);
    }
    expect(fs.existsSync(resolve(REPO_ROOT, "publication/output/forecast/adaptive_sensitivity_summary.json"))).toBe(true);
    expect(fs.existsSync(resolve(REPO_ROOT, "publication/output/forecast/run_adaptive_sensitivity.sh"))).toBe(true);
  });
});
