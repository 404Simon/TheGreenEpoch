import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve, dirname } from "node:path";
import { marginRuleSurvives, MARGIN_RULE_CEILING } from "./forecast-sweep";
import type { ReoptConfigResult, ReoptSeedRow } from "./forecast-sweep";
import type { Constants, CO2Timeline, FullProfile, TrainingProfile, YearCO2, SimProgress } from "../domain/types";
import { runOptimization } from "../domain/optimize";
import type { AdaptiveOptions } from "../domain/optimize";
import { simulateStepwise } from "../domain/simulation";
import { hysteresisPolicy, neverPausePolicy } from "../domain/policy";
import { computeOverheadPct, computeSavingsPct } from "../domain/result";
import { tokensPerSecond } from "../domain/physics";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, "../..");

const CKPT_VALUES = [148.8, 150, 900, 2700];

interface CkptRow {
  ckpt_pause: number;
  theta_p: number | null;
  theta_r: number | null;
  margin: number | null;
  savings: number | null;
  overhead: number | null;
  score: number | null;
  num_pauses: number | null;
  completed: boolean | null;
  found: boolean;
  bestCompleted: {
    theta_p: number;
    theta_r: number;
    margin: number;
    savings: number;
    overhead: number;
    score: number;
    num_pauses: number;
  } | null;
}

interface RegionSummary {
  region: string;
  baseline: { theta_p: number; theta_r: number; margin: number; savings: number; overhead: number; score: number };
  runs: CkptRow[];
}

function loadJSON<T>(relPath: string): T {
  return JSON.parse(readFileSync(resolve(REPO_ROOT, relPath), "utf-8")) as T;
}

function loadSummary(): RegionSummary[] {
  return loadJSON<RegionSummary[]>("publication/output/checkpoint/checkpoint_summary.json");
}

function loadYear(zone: string, year: number): CO2Timeline {
  const y = loadJSON<YearCO2>(`public/data/co2/${zone}_${year}.json`);
  return { zone, years: [year], timestamps: y.timestamps, carbonIntensity: y.carbonIntensity };
}

function loadFullProfile(ckptPause?: number, ckptResume?: number): FullProfile {
  const constants = loadJSON<Constants>("public/data/constants.json");
  const profiles = loadJSON<Record<string, TrainingProfile>>("public/data/profiles.json");
  const profile = profiles.Deepseek;
  return {
    ...profile,
    gpuPowerTrain: constants.gpu_power_train,
    gpuPowerPause: constants.gpu_power_pause,
    pue: constants.pue,
    checkpointPauseTime: ckptPause ?? constants.checkpoint_pause_time,
    checkpointResumeTime: ckptResume ?? constants.checkpoint_resume_time,
  };
}

// Reopt-optimal fixed policy per region (theta_p, theta_r, start) — matches reopt_summary.json.
const REGION_POLICY: Record<string, { thetaP: number; thetaR: number; start: string }> = {
  DE: { thetaP: 272.37, thetaR: 267.73, start: "02-01" },
  IT: { thetaP: 246.7, thetaR: 230.45, start: "01-14" },
  SE: { thetaP: 18.18, thetaR: 17.51, start: "04-22" },
};

function drain(profile: FullProfile, policy: ReturnType<typeof hysteresisPolicy>, timeline: CO2Timeline, start: string): SimProgress {
  let last: SimProgress | null = null;
  for (const p of simulateStepwise(profile, policy, timeline, { startTime: start, historicalYears: [2025], overheadBudgetPct: 200 })) {
    last = p;
  }
  if (!last) throw new Error("simulateStepwise produced no progress");
  return last;
}

describe("Phase B.0 checkpoint-realism sweep artifacts", () => {
  const summary = loadSummary();
  const byRegion = Object.fromEntries(summary.map((r) => [r.region, r]));

  it("records all 4 ckpt rows with all required fields per region", () => {
    expect(summary).toHaveLength(3);
    for (const region of ["DE", "IT", "SE"]) {
      const r = byRegion[region];
      expect(r.runs.map((x) => x.ckpt_pause)).toEqual(CKPT_VALUES);
      for (const run of r.runs) {
        expect(typeof run.theta_p).toBe("number");
        expect(typeof run.theta_r).toBe("number");
        expect(typeof run.margin).toBe("number");
        expect(typeof run.savings).toBe("number");
        expect(typeof run.overhead).toBe("number");
        expect(typeof run.score).toBe("number");
        expect(typeof run.num_pauses).toBe("number");
        expect(typeof run.completed).toBe("boolean");
        expect(run.found).toBe(true);
        expect(run.margin).toBeCloseTo((run.theta_p as number) - (run.theta_r as number), 6);
      }
    }
  });
});

describe("B.0.3(i) monotonicity", () => {
  const summary = loadSummary();
  const byRegion = Object.fromEntries(summary.map((r) => [r.region, r]));

  it("completed-best savings in the committed artifact is monotone non-increasing in --ckpt-pause", () => {
    for (const region of ["DE", "IT", "SE"]) {
      const savings = byRegion[region].runs.map((r) => r.bestCompleted?.savings as number);
      for (let i = 1; i < savings.length; i++) {
        expect(savings[i], `${region} savings at ckpt ${CKPT_VALUES[i]} should not exceed ckpt ${CKPT_VALUES[i - 1]}`).toBeLessThanOrEqual(savings[i - 1]);
      }
    }
  });

  it("fixed-policy savings/overhead are monotone (non-increasing / non-decreasing) as --ckpt-pause grows", () => {
    for (const region of ["DE", "IT", "SE"]) {
      const policy = REGION_POLICY[region];
      const timeline = loadYear(region, 2025);
      const baseline = drain(loadFullProfile(), neverPausePolicy(), timeline, policy.start);
      const baselineEmKg = baseline.totalEmissionsG / 1000;
      const tps = tokensPerSecond(2048) || 1;
      const savingsSeries: number[] = [];
      const overheadSeries: number[] = [];
      for (const ckpt of CKPT_VALUES) {
        const last = drain(loadFullProfile(ckpt, 0), hysteresisPolicy(policy.thetaP, policy.thetaR), timeline, policy.start);
        savingsSeries.push(computeSavingsPct(last.totalEmissionsG / 1000, baselineEmKg));
        overheadSeries.push(computeOverheadPct(last.pausedS, last.checkpointS, last.tokensTotal, tps));
      }
      for (let i = 1; i < CKPT_VALUES.length; i++) {
        expect(savingsSeries[i], `${region} fixed-policy savings at ${CKPT_VALUES[i]}s`).toBeLessThanOrEqual(savingsSeries[i - 1] + 1e-9);
        expect(overheadSeries[i], `${region} fixed-policy overhead at ${CKPT_VALUES[i]}s`).toBeGreaterThanOrEqual(overheadSeries[i - 1] - 1e-9);
      }
    }
  });
});

describe("B.0.3(ii) identity at ckpt=148.8", () => {
  const summary = loadSummary();
  const byRegion = Object.fromEntries(summary.map((r) => [r.region, r]));

  it("reported best at ckpt 148.8 reproduces the region baseline within tolerance", () => {
    const tol = { theta: 2, savings: 0.5, overhead: 1 };
    for (const region of ["DE", "IT", "SE"]) {
      const r = byRegion[region];
      const row = r.runs.find((x) => x.ckpt_pause === 148.8);
      expect(row).toBeDefined();
      expect(row?.theta_p).toBeCloseTo(r.baseline.theta_p, 0); // within ±0.5 -> ±2 tolerates
      expect(Math.abs((row?.theta_p as number) - r.baseline.theta_p)).toBeLessThanOrEqual(tol.theta);
      expect(Math.abs((row?.theta_r as number) - r.baseline.theta_r)).toBeLessThanOrEqual(tol.theta);
      expect(Math.abs((row?.savings as number) - r.baseline.savings)).toBeLessThanOrEqual(tol.savings);
      expect(Math.abs((row?.overhead as number) - r.baseline.overhead)).toBeLessThanOrEqual(tol.overhead);
      expect(row?.completed).toBe(true);
    }
  });

  it("runOptimization with an explicit 148.8/0 profile equals the constants profile (small resolution)", () => {
    const timeline = loadYear("DE", 2025);
    const options: AdaptiveOptions = {
      thetaPauseMax: 800,
      overheadBudgetPct: 200,
      resolution: 3,
      startDateResolution: 1,
      maxIterations: 2,
      minStep: 3,
      shrinkFactor: 0.45,
      alpha: 1,
      fixedStartTime: "02-01",
    };
    const constantsBest = runOptimization(loadFullProfile(), timeline, [2025], options).best;
    const explicitBest = runOptimization(loadFullProfile(148.8, 0), timeline, [2025], options).best;
    expect(constantsBest).not.toBeNull();
    expect(explicitBest).not.toBeNull();
    expect(explicitBest?.thetaPause).toBe(constantsBest?.thetaPause);
    expect(explicitBest?.thetaResume).toBe(constantsBest?.thetaResume);
    expect(explicitBest?.co2SavingsPct).toBe(constantsBest?.co2SavingsPct);
    expect(explicitBest?.actualOverheadPct).toBe(constantsBest?.actualOverheadPct);
    expect(explicitBest?.score).toBe(constantsBest?.score);
  });
});

describe("B.0.3(iii) marginRuleSurvives under each ckpt (ceiling 16 g/kWh)", () => {
  const summary = loadSummary();
  const byRegion = Object.fromEntries(summary.map((r) => [r.region, r]));

  it("ceiling constant is 16 g/kWh", () => {
    expect(MARGIN_RULE_CEILING).toBe(16);
  });

  it("marginRuleSurvives applies the 16 g/kWh ceiling to every (region, ckpt) margin", () => {
    for (const region of ["DE", "IT", "SE"]) {
      for (const run of byRegion[region].runs) {
        const seedRow: ReoptSeedRow = {
          seed: 1,
          thetaP: run.theta_p,
          thetaR: run.theta_r,
          margin: run.margin,
          savings: run.savings,
          overhead: run.overhead,
          score: run.score,
          found: true,
        };
        const config: ReoptConfigResult = {
          family: "additive",
          param: "level",
          param_value: 0,
          sigma: null,
          perSeed: [seedRow],
          best: seedMeanBestOf([seedRow]),
          foundRate: 1,
        };
        const rule = marginRuleSurvives(config);
        const marginOk = (run.margin as number) <= MARGIN_RULE_CEILING;
        expect(rule.seedFraction).toBe(marginOk ? 1 : 0);
        expect(rule.survives, `${region} ckpt=${run.ckpt_pause}s margin=${run.margin}`).toBe(marginOk);
      }
    }
  });

  it("marginRuleSurvives boundary behavior around the 16 g/kWh ceiling", () => {
    const mk = (margin: number): ReoptConfigResult => {
      const seedRow: ReoptSeedRow = { seed: 1, thetaP: 280, thetaR: 280 - margin, margin, savings: 40, overhead: 180, score: 0.2, found: true };
      return { family: "additive", param: "level", param_value: 0, sigma: null, perSeed: [seedRow], best: seedMeanBestOf([seedRow]), foundRate: 1 };
    };
    expect(marginRuleSurvives(mk(15.9)).survives).toBe(true);
    expect(marginRuleSurvives(mk(16)).survives).toBe(true);
    expect(marginRuleSurvives(mk(16.1)).survives).toBe(false);
  });
});

function seedMeanBestOf(rows: ReoptSeedRow[]): { thetaP: number; thetaR: number; margin: number; savings: number; overhead: number; score: number } {
  const r = rows[0];
  return { thetaP: r.thetaP as number, thetaR: r.thetaR as number, margin: r.margin as number, savings: r.savings as number, overhead: r.overhead as number, score: r.score as number };
}
