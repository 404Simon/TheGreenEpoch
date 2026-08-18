import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type {
  ArCoeffs,
  CalibrationBundle,
  Constants,
  CO2Timeline,
  ForecastModel,
  FullProfile,
  SimConfig,
  SimProgress,
  TrainingProfile,
  YearCO2,
} from "../domain/types";
import { simulateStepwise } from "../domain/simulation";
import { hysteresisPolicy, neverPausePolicy } from "../domain/policy";
import { computeOverheadPct, computeSavingsPct, computeScore } from "../domain/result";
import { applyForecast } from "../domain/forecast";
import { runOptimization } from "../domain/optimize";
import type { AdaptiveOptions } from "../domain/optimize";
import type { SweepPoint } from "../domain/types";
import { tokensPerSecond } from "../domain/physics";
import { averageYears } from "../data/co2-loader";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(__dirname, "../../public/data");

export const GRACE_THRESHOLD = 0.10;

const DEFAULT_POLICIES: Record<string, { thetaP: number; thetaR: number; start: string }> = {
  DE: { thetaP: 272, thetaR: 268, start: "02-01" },
  IT: { thetaP: 246, thetaR: 231, start: "01-14" },
  SE: { thetaP: 19, thetaR: 18, start: "04-22" },
};

const VALID_FAMILIES: string[] = ["additive", "multiplicative", "delay", "arma", "persistence"];

export interface ForecastConfig {
  family: string;
  param: string;
  param_value: number;
  sigma: number | null;
  model: ForecastModel;
}

export interface FixedRow {
  region: string;
  family: string;
  param: string;
  param_value: number;
  seed: number;
  sigma: number | null;
  theta_p: number;
  theta_r: number;
  start: string;
  savings: number;
  overhead: number;
  score: number;
  num_pauses: number;
  completed: boolean;
  within_budget: boolean;
  savings_perfect: number;
  degradation_frac: number;
}

export interface FixedSummary {
  family: string;
  param: string;
  param_value: number;
  sigma: number | null;
  s0: number;
  savings_mean: number;
  savings_std: number;
  delta_s_pp: number;
  delta_s_frac: number;
  degradation_frac_mean: number;
  degradation_frac_std: number;
  overhead_mean: number;
  num_pauses_mean: number;
  completed_rate: number;
  within_budget_rate: number;
  n_seeds: number;
  graceLevel: number;
  graceAtMax: boolean;
}

export interface ControlMetrics {
  savings: number;
  overhead: number;
  score: number;
  num_pauses: number;
  completed: boolean;
  within_budget: boolean;
}

export interface SweepRegionParams {
  region: string;
  profile: FullProfile;
  realized: CO2Timeline;
  historicalYears: number[];
  thetaP: number;
  thetaR: number;
  start: string;
  budget: number;
  seeds: number[];
  configs: ForecastConfig[];
}

export function meanStd(values: number[]): { mean: number; std: number } {
  const n = values.length;
  if (n === 0) return { mean: NaN, std: NaN };
  const mean = values.reduce((a, b) => a + b, 0) / n;
  if (n < 2) return { mean, std: 0 };
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1);
  return { mean, std: Math.sqrt(variance) };
}

export function degradationFrac(savingsPerfect: number, savings: number): number {
  if (savingsPerfect === 0) return 0;
  const v = (savingsPerfect - savings) / savingsPerfect;
  return Number.isNaN(v) ? 0 : v;
}

export function expandConfigs(opts: {
  families: string[];
  levels: number[];
  horizons: number[];
  sigmaStar: number;
  sigmaRel: number;
  coeffs: ArCoeffs;
}): ForecastConfig[] {
  const configs: ForecastConfig[] = [];
  for (const family of opts.families) {
    if (family === "additive") {
      for (const level of opts.levels) {
        const sigma = level * opts.sigmaStar;
        configs.push({ family, param: "level", param_value: level, sigma, model: { type: "additive", sigma } });
      }
    } else if (family === "multiplicative") {
      for (const level of opts.levels) {
        const sigma = level * opts.sigmaRel;
        configs.push({ family, param: "level", param_value: level, sigma, model: { type: "multiplicative", sigma } });
      }
    } else if (family === "delay") {
      for (const steps of opts.horizons) {
        configs.push({ family, param: "steps", param_value: steps, sigma: null, model: { type: "delay", steps } });
      }
    } else if (family === "arma") {
      for (const horizon of opts.horizons) {
        configs.push({
          family,
          param: "horizon",
          param_value: horizon,
          sigma: null,
          model: { type: "arma", order: 1, horizon, coeffs: opts.coeffs },
        });
      }
    } else if (family === "persistence") {
      for (const steps of opts.horizons) {
        configs.push({ family, param: "horizon", param_value: steps, sigma: null, model: { type: "delay", steps } });
      }
    }
  }
  return configs;
}

export function computeSummary(s0: number, rows: FixedRow[]): FixedSummary {
  if (rows.length === 0) throw new Error("computeSummary: at least one row required");
  const savings = rows.map((r) => r.savings);
  const { mean: savingsMean, std: savingsStd } = meanStd(savings);
  const degradation = rows.map((r) => r.degradation_frac);
  const { mean: degradationFracMean, std: degradationFracStd } = meanStd(degradation);
  const overhead = rows.map((r) => r.overhead);
  const { mean: overheadMean } = meanStd(overhead);
  const pauses = rows.map((r) => r.num_pauses);
  const { mean: numPausesMean } = meanStd(pauses);
  const deltaSPp = s0 - savingsMean;
  return {
    family: rows[0].family,
    param: rows[0].param,
    param_value: rows[0].param_value,
    sigma: rows[0].sigma,
    s0,
    savings_mean: savingsMean,
    savings_std: savingsStd,
    delta_s_pp: deltaSPp,
    delta_s_frac: s0 > 0 ? deltaSPp / s0 : 0,
    degradation_frac_mean: degradationFracMean,
    degradation_frac_std: degradationFracStd,
    overhead_mean: overheadMean,
    num_pauses_mean: numPausesMean,
    completed_rate: rows.filter((r) => r.completed).length / rows.length,
    within_budget_rate: rows.filter((r) => r.within_budget).length / rows.length,
    n_seeds: rows.length,
    graceLevel: 0,
    graceAtMax: false,
  };
}

export function computeGraceLevel(summaries: FixedSummary[]): { graceLevel: number; graceAtMax: boolean } {
  const positive = summaries.filter((s) => s.param_value > 0);
  if (positive.length === 0) return { graceLevel: 0, graceAtMax: false };
  const sorted = [...positive].sort((a, b) => a.param_value - b.param_value);
  const maxValue = sorted[sorted.length - 1].param_value;
  let graceLevel = 0;
  for (const s of sorted) {
    if (s.degradation_frac_mean <= GRACE_THRESHOLD) graceLevel = s.param_value;
  }
  return { graceLevel, graceAtMax: graceLevel === maxValue };
}

function drain(profile: FullProfile, policy: ReturnType<typeof hysteresisPolicy>, timeline: CO2Timeline, simConfig: SimConfig): SimProgress {
  let last: SimProgress | null = null;
  for (const p of simulateStepwise(profile, policy, timeline, simConfig)) {
    last = p;
  }
  if (!last) throw new Error("simulateStepwise produced no progress");
  return last;
}

function metricsFromProgress(profile: FullProfile, last: SimProgress, baselineEmissionsKg: number, budget: number): ControlMetrics {
  const tps = tokensPerSecond(profile.gpuCount) || 1;
  const emissionsKg = last.totalEmissionsG / 1000;
  const savings = computeSavingsPct(emissionsKg, baselineEmissionsKg);
  const overhead = computeOverheadPct(last.pausedS, last.checkpointS, last.tokensTotal, tps);
  const score = computeScore(savings, overhead, budget, 1);
  return {
    savings,
    overhead,
    score,
    num_pauses: last.numPauses,
    completed: last.tokensRemaining <= 0,
    within_budget: overhead <= budget,
  };
}

export function runSweepRegion(params: SweepRegionParams): {
  control: ControlMetrics;
  rows: FixedRow[];
  summary: FixedSummary[];
} {
  const { region, profile, realized, historicalYears, thetaP, thetaR, start, budget, seeds, configs } = params;
  const baseSimConfig: SimConfig = { startTime: start, historicalYears, overheadBudgetPct: budget };

  const baselineLast = drain(profile, neverPausePolicy(), realized, baseSimConfig);
  const baselineEmissionsKg = baselineLast.totalEmissionsG / 1000;
  const controlLast = drain(profile, hysteresisPolicy(thetaP, thetaR), realized, baseSimConfig);
  const control = metricsFromProgress(profile, controlLast, baselineEmissionsKg, budget);

  const rows: FixedRow[] = [];
  const groups = new Map<string, FixedRow[]>();
  for (const cfg of configs) {
    for (const seed of seeds) {
      const decision = applyForecast(realized, cfg.model, seed);
      const last = drain(profile, hysteresisPolicy(thetaP, thetaR), realized, {
        ...baseSimConfig,
        decisionTimeline: decision,
      });
      const m = metricsFromProgress(profile, last, baselineEmissionsKg, budget);
      const row: FixedRow = {
        region,
        family: cfg.family,
        param: cfg.param,
        param_value: cfg.param_value,
        seed,
        sigma: cfg.sigma,
        theta_p: thetaP,
        theta_r: thetaR,
        start,
        savings: m.savings,
        overhead: m.overhead,
        score: m.score,
        num_pauses: m.num_pauses,
        completed: m.completed,
        within_budget: m.within_budget,
        savings_perfect: control.savings,
        degradation_frac: degradationFrac(control.savings, m.savings),
      };
      rows.push(row);
      const key = `${cfg.family}|${cfg.param}|${cfg.param_value}`;
      const arr = groups.get(key);
      if (arr) {
        arr.push(row);
      } else {
        groups.set(key, [row]);
      }
    }
  }

  const summary: FixedSummary[] = [];
  for (const cfg of configs) {
    const groupRows = groups.get(`${cfg.family}|${cfg.param}|${cfg.param_value}`);
    if (!groupRows) continue;
    summary.push(computeSummary(control.savings, groupRows));
  }

  const families = [...new Set(summary.map((s) => s.family))];
  for (const family of families) {
    const g = computeGraceLevel(summary.filter((s) => s.family === family));
    for (const s of summary) {
      if (s.family === family) {
        s.graceLevel = g.graceLevel;
        s.graceAtMax = g.graceAtMax;
      }
    }
  }

  return { control, rows, summary };
}

function loadJSON<T>(path: string): T {
  return JSON.parse(readFileSync(resolve(DATA_DIR, path), "utf-8")) as T;
}

function loadCO2Timeline(zone: string, year: number): CO2Timeline {
  return averageYears([loadJSON<YearCO2>(`co2/${zone}_${year}.json`)]);
}

function loadCalibration(region: string, calibrationDir: string): CalibrationBundle {
  return JSON.parse(readFileSync(resolve(calibrationDir, `calibration_${region}.json`), "utf-8")) as CalibrationBundle;
}

function makeSeeds(region: string, seedCount: number, seedCountOther: number): number[] {
  const count = region === "DE" ? seedCount : seedCountOther;
  return Array.from({ length: count }, (_, i) => i + 1);
}

function resolvePolicy(
  region: string,
  thetaPOverride: number | null,
  thetaROverride: number | null,
  startOverride: string | null,
): { thetaP: number; thetaR: number; start: string } {
  const base = DEFAULT_POLICIES[region];
  if (!base) throw new Error(`No default policy for region ${region}`);
  return {
    thetaP: thetaPOverride ?? base.thetaP,
    thetaR: thetaROverride ?? base.thetaR,
    start: startOverride ?? base.start,
  };
}

const FIXED_CSV_HEADER =
  "region,family,param,param_value,seed,sigma,theta_p,theta_r,start,savings,overhead,score,num_pauses,completed,within_budget,savings_perfect,degradation_frac";

function toCsvRow(row: FixedRow): string[] {
  return [
    row.region,
    row.family,
    row.param,
    row.param_value.toPrecision(6),
    String(row.seed),
    row.sigma == null ? "NA" : row.sigma.toPrecision(6),
    row.theta_p.toPrecision(6),
    row.theta_r.toPrecision(6),
    row.start,
    row.savings.toPrecision(6),
    row.overhead.toPrecision(6),
    row.score.toPrecision(6),
    row.num_pauses.toPrecision(6),
    String(row.completed),
    String(row.within_budget),
    row.savings_perfect.toPrecision(6),
    row.degradation_frac.toPrecision(6),
  ];
}

function writeCsv(path: string, rows: FixedRow[]): void {
  const body = rows.map((r) => toCsvRow(r).join(",")).join("\n");
  writeFileSync(path, FIXED_CSV_HEADER + (rows.length > 0 ? "\n" + body + "\n" : "\n"), "utf-8");
}

function graceEntry(summaries: FixedSummary[]): Record<string, number | boolean> {
  const g = computeGraceLevel(summaries);
  const param = summaries.length > 0 ? summaries[0].param : "level";
  return { [param]: g.graceLevel, atMax: g.graceAtMax };
}

function degradationEntry(summary: FixedSummary | undefined): { delta_s_frac: number; delta_s_pp: number } | null {
  return summary ? { delta_s_frac: summary.delta_s_frac, delta_s_pp: summary.delta_s_pp } : null;
}

export const MARGIN_RULE_CEILING = 16;
export const REGIONAL_RULE_TOLERANCE = 0.5;

export interface ReoptSeedRow {
  seed: number;
  thetaP: number | null;
  thetaR: number | null;
  margin: number | null;
  savings: number | null;
  overhead: number | null;
  score: number | null;
  found: boolean;
}

export interface ReoptBest {
  thetaP: number;
  thetaR: number;
  margin: number;
  savings: number;
  overhead: number;
  score: number;
}

export interface ReoptConfigResult {
  family: string;
  param: string;
  param_value: number;
  sigma: number | null;
  perSeed: ReoptSeedRow[];
  best: ReoptBest | null;
  foundRate: number;
}

export interface ReoptOptimizerInfo {
  resolution: number;
  iterations: number;
  tpMax: number;
}

export interface ReoptRegionResult {
  region: string;
  model: string;
  year: number;
  budget: number;
  start: string;
  seeds: number[];
  optimizer: ReoptOptimizerInfo;
  configs: ReoptConfigResult[];
  baseline: ReoptBest;
}

export interface ReoptDriftEntry {
  family: string;
  param_value: number;
  thetaP_drift: number | null;
  thetaR_drift: number | null;
  margin_drift: number | null;
}

export interface ReoptMarginRuleEntry {
  family: string;
  param_value: number;
  survives: boolean;
  seedFraction: number;
}

export interface ReoptRegionalRuleEntry {
  family: string;
  param_value: number;
  survives: boolean;
}

export interface ReoptSummaryRegion {
  region: string;
  baseline: ReoptBest;
  drift: ReoptDriftEntry[];
  marginRuleSurvives: ReoptMarginRuleEntry[];
  regionalRule: ReoptRegionalRuleEntry[];
}

export function seedMeanBest(rows: ReoptSeedRow[]): ReoptBest | null {
  const found = rows.filter((r) => r.found);
  if (found.length === 0) return null;
  const n = found.length;
  const mean = (sel: (r: ReoptSeedRow) => number): number => found.reduce((a, r) => a + sel(r), 0) / n;
  return {
    thetaP: mean((r) => r.thetaP as number),
    thetaR: mean((r) => r.thetaR as number),
    margin: mean((r) => r.margin as number),
    savings: mean((r) => r.savings as number),
    overhead: mean((r) => r.overhead as number),
    score: mean((r) => r.score as number),
  };
}

export function drift(best: ReoptBest, baseline: ReoptBest): { thetaP_drift: number; thetaR_drift: number; margin_drift: number } {
  return {
    thetaP_drift: best.thetaP - baseline.thetaP,
    thetaR_drift: best.thetaR - baseline.thetaR,
    margin_drift: best.margin - baseline.margin,
  };
}

export function marginRuleSurvives(config: ReoptConfigResult): { survives: boolean; seedFraction: number } {
  const found = config.perSeed.filter((r) => r.found);
  if (found.length === 0) return { survives: false, seedFraction: 0 };
  const ok = found.filter((r) => (r.margin as number) <= MARGIN_RULE_CEILING).length;
  return {
    survives: config.best != null && config.best.margin <= MARGIN_RULE_CEILING,
    seedFraction: ok / found.length,
  };
}

export function regionalRuleSurvives(config: ReoptConfigResult, baseline: ReoptBest): boolean {
  if (config.best == null) return false;
  const lo = baseline.thetaP * (1 - REGIONAL_RULE_TOLERANCE);
  const hi = baseline.thetaP * (1 + REGIONAL_RULE_TOLERANCE);
  return config.best.thetaP >= lo && config.best.thetaP <= hi;
}

export function buildReoptSummary(region: string, result: ReoptRegionResult): ReoptSummaryRegion {
  const driftEntries: ReoptDriftEntry[] = result.configs.map((c) => {
    const base = { family: c.family, param_value: c.param_value };
    if (!c.best) return { ...base, thetaP_drift: null, thetaR_drift: null, margin_drift: null };
    return { ...base, ...drift(c.best, result.baseline) };
  });
  const marginEntries: ReoptMarginRuleEntry[] = result.configs.map((c) => ({
    family: c.family,
    param_value: c.param_value,
    ...marginRuleSurvives(c),
  }));
  const regionalEntries: ReoptRegionalRuleEntry[] = result.configs.map((c) => ({
    family: c.family,
    param_value: c.param_value,
    survives: regionalRuleSurvives(c, result.baseline),
  }));
  return { region, baseline: result.baseline, drift: driftEntries, marginRuleSurvives: marginEntries, regionalRule: regionalEntries };
}

function bestFromPoint(p: SweepPoint): ReoptBest {
  return {
    thetaP: p.thetaPause,
    thetaR: p.thetaResume,
    margin: p.thetaPause - p.thetaResume,
    savings: p.co2SavingsPct,
    overhead: p.actualOverheadPct,
    score: p.score,
  };
}

function rowFromBest(seed: number, best: SweepPoint | null): ReoptSeedRow {
  if (!best) return { seed, thetaP: null, thetaR: null, margin: null, savings: null, overhead: null, score: null, found: false };
  return {
    seed,
    thetaP: best.thetaPause,
    thetaR: best.thetaResume,
    margin: best.thetaPause - best.thetaResume,
    savings: best.co2SavingsPct,
    overhead: best.actualOverheadPct,
    score: best.score,
    found: true,
  };
}

function sameBest(a: SweepPoint | null, b: SweepPoint | null): boolean {
  if (a === null || b === null) return a === b;
  return (
    a.thetaPause === b.thetaPause &&
    a.thetaResume === b.thetaResume &&
    a.startTime === b.startTime &&
    a.actualOverheadPct === b.actualOverheadPct &&
    a.co2SavingsPct === b.co2SavingsPct &&
    a.score === b.score &&
    a.numPauses === b.numPauses &&
    a.totalEmissionsKgco2 === b.totalEmissionsKgco2 &&
    a.baselineEmissionsKgco2 === b.baselineEmissionsKgco2 &&
    a.withinBudget === b.withinBudget &&
    a.stopReason === b.stopReason &&
    a.completed === b.completed &&
    a.iteration === b.iteration
  );
}

export function assertIdentityRegression(
  profile: FullProfile,
  realized: CO2Timeline,
  historicalYears: number[],
  options: AdaptiveOptions,
): void {
  const identity = applyForecast(realized, { type: "identity" }, 1);
  const freeBest = runOptimization(profile, realized, historicalYears, options).best;
  const identityBest = runOptimization(profile, realized, historicalYears, options, undefined, identity).best;
  if (!sameBest(freeBest, identityBest)) {
    throw new Error(
      "Identity regression: runOptimization with an identity decisionTimeline must match the no-decisionTimeline run",
    );
  }
}

export interface ReoptRegionRunParams {
  region: string;
  model: string;
  year: number;
  profile: FullProfile;
  realized: CO2Timeline;
  sigmaStar: number;
  start: string;
  budget: number;
  alpha: number;
  seeds: number[];
  additiveLevels: number[];
  delaySteps: number[];
  resolution: number;
  iterations: number;
  tpMax: number;
}

export function runReoptRegion(params: ReoptRegionRunParams): ReoptRegionResult {
  const {
    region,
    model,
    year,
    profile,
    realized,
    sigmaStar,
    start,
    budget,
    alpha,
    seeds,
    additiveLevels,
    delaySteps,
    resolution,
    iterations,
    tpMax,
  } = params;
  const options: AdaptiveOptions = {
    thetaPauseMax: tpMax,
    overheadBudgetPct: budget,
    resolution,
    startDateResolution: 1,
    maxIterations: iterations,
    minStep: 3,
    shrinkFactor: 0.45,
    alpha,
    fixedStartTime: start,
  };
  const historicalYears = [year];

  const freeBest = runOptimization(profile, realized, historicalYears, options).best;
  if (!freeBest) {
    throw new Error(`runReoptRegion: no valid best for the identity (no decisionTimeline) baseline of ${region}`);
  }

  const configs: ReoptConfigResult[] = [];

  for (const level of additiveLevels) {
    const sigma = level * sigmaStar;
    configs.push(runReoptConfig({
      profile,
      realized,
      historicalYears,
      options,
      seeds,
      freeBest,
      family: "additive",
      param: "level",
      param_value: level,
      sigma,
      model: { type: "additive", sigma },
      assertIdentity: level === 0,
    }));
  }

  for (const steps of delaySteps) {
    configs.push(runReoptConfig({
      profile,
      realized,
      historicalYears,
      options,
      seeds,
      freeBest,
      family: "delay",
      param: "steps",
      param_value: steps,
      sigma: null,
      model: { type: "delay", steps },
      assertIdentity: false,
    }));
  }

  return {
    region,
    model,
    year,
    budget,
    start,
    seeds: [...seeds],
    optimizer: { resolution, iterations, tpMax },
    configs,
    baseline: configs[0].best ?? bestFromPoint(freeBest),
  };
}

interface ReoptConfigParams {
  profile: FullProfile;
  realized: CO2Timeline;
  historicalYears: number[];
  options: AdaptiveOptions;
  seeds: number[];
  freeBest: SweepPoint;
  family: string;
  param: string;
  param_value: number;
  sigma: number | null;
  model: ForecastModel;
  assertIdentity: boolean;
}

function runReoptConfig(params: ReoptConfigParams): ReoptConfigResult {
  const { profile, realized, historicalYears, options, seeds, freeBest, family, param, param_value, sigma, model, assertIdentity } = params;
  const perSeed: ReoptSeedRow[] = [];
  for (const seed of seeds) {
    const decision = applyForecast(realized, model, seed);
    const { best } = runOptimization(profile, realized, historicalYears, options, undefined, decision);
    if (assertIdentity && !sameBest(best, freeBest)) {
      throw new Error(
        `Identity regression failed for ${family} ${param}=${param_value} seed ${seed}: decisionTimeline best differs from the no-decisionTimeline best`,
      );
    }
    perSeed.push(rowFromBest(seed, best));
  }
  return {
    family,
    param,
    param_value,
    sigma,
    perSeed,
    best: seedMeanBest(perSeed),
    foundRate: perSeed.filter((r) => r.found).length / perSeed.length,
  };
}

const REOPT_CSV_HEADER = "region,family,param_value,seed,theta_p,theta_r,margin,savings,overhead,score,found";

interface ReoptCsvRow {
  region: string;
  family: string;
  param_value: number;
  seed: number;
  theta_p: number | null;
  theta_r: number | null;
  margin: number | null;
  savings: number | null;
  overhead: number | null;
  score: number | null;
  found: boolean;
}

function reoptCell(v: number | null): string {
  return v == null ? "NA" : v.toPrecision(6);
}

function writeReoptCsv(path: string, rows: ReoptCsvRow[]): void {
  const body = rows.map((r) =>
    [
      r.region,
      r.family,
      r.param_value.toPrecision(6),
      String(r.seed),
      reoptCell(r.theta_p),
      reoptCell(r.theta_r),
      reoptCell(r.margin),
      reoptCell(r.savings),
      reoptCell(r.overhead),
      reoptCell(r.score),
      String(r.found),
    ].join(","),
  );
  writeFileSync(path, REOPT_CSV_HEADER + (body.length > 0 ? "\n" + body.join("\n") + "\n" : "\n"), "utf-8");
}

async function reoptSweepCli(raw: {
  model?: string;
  regions?: string;
  year?: string;
  seedCount?: string;
  additiveLevels?: string;
  delaySteps?: string;
  resolution?: string;
  iterations?: string;
  budget?: string;
  alpha?: string;
  calibrationDir?: string;
  output?: string;
  csv?: string;
  quiet?: boolean;
}): Promise<void> {
  const model = raw.model ?? "Deepseek";
  const regions = (raw.regions ?? "DE,IT,SE")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  if (regions.length === 0) {
    console.error("  No regions given");
    process.exit(1);
  }
  const year = Number(raw.year ?? "2025");
  const additiveLevels = (raw.additiveLevels ?? "0,0.5,1,2").split(",").map((s) => Number(s.trim()));
  const delaySteps = (raw.delaySteps ?? "1,6").split(",").map((s) => Number(s.trim()));
  const seedCount = Number(raw.seedCount ?? "3");
  const resolution = Number(raw.resolution ?? "10");
  const iterations = Number(raw.iterations ?? "6");
  const budget = Number(raw.budget ?? "200");
  const alpha = Number(raw.alpha ?? "1");
  const calibrationDir = raw.calibrationDir ?? "publication/output/forecast";
  const outputPrefix = raw.output ?? "publication/output/forecast/reopt";
  const csvPath = raw.csv ?? null;
  const quiet = raw.quiet ?? false;
  const seeds = Array.from({ length: seedCount }, (_, i) => i + 1);

  const constants = loadJSON<Constants>("constants.json");
  const profiles = loadJSON<Record<string, TrainingProfile>>("profiles.json");
  const profile = profiles[model];
  if (!profile) {
    console.error(`  Unknown model: ${model}. Available: ${Object.keys(profiles).join(", ")}`);
    process.exit(1);
  }
  const fullProfile: FullProfile = {
    ...profile,
    gpuPowerTrain: constants.gpu_power_train,
    gpuPowerPause: constants.gpu_power_pause,
    pue: constants.pue,
    checkpointPauseTime: constants.checkpoint_pause_time,
    checkpointResumeTime: constants.checkpoint_resume_time,
  };

  mkdirSync(dirname(resolve(outputPrefix)), { recursive: true });
  if (csvPath) mkdirSync(dirname(resolve(csvPath)), { recursive: true });

  if (!quiet) {
    console.log(`\n  TheGreenEpoch Forecast Sweep \u2500 re-optimization (design-rule drift)`);
    console.log(`  Model: ${model}, Regions: ${regions.join(",")}, Year: ${year}`);
    console.log(`  Budget: ${budget}%, \u03B1=${alpha}, resolution=${resolution}, iterations=${iterations}`);
    console.log(`  Seeds: ${seeds.join(",")} per region; additive levels ${additiveLevels.join(",")} x \u03C3*, delay steps ${delaySteps.join(",")}`);
  }

  const summaryRecords: ReoptSummaryRegion[] = [];
  const allRows: ReoptCsvRow[] = [];

  for (const region of regions) {
    const realized = loadCO2Timeline(region, year);
    const calibration = loadCalibration(region, calibrationDir);
    const start = DEFAULT_POLICIES[region].start;
    if (!start) {
      console.error(`  No default start date for region ${region}`);
      process.exit(1);
    }
    const tpMax = region === "SE" ? 100 : 800;

    const result = runReoptRegion({
      region,
      model,
      year,
      profile: fullProfile,
      realized,
      sigmaStar: calibration.sigmaStar,
      start,
      budget,
      alpha,
      seeds,
      additiveLevels,
      delaySteps,
      resolution,
      iterations,
      tpMax,
    });

    writeFileSync(`${outputPrefix}_${region}.json`, JSON.stringify(result, null, 2) + "\n", "utf-8");

    const rows: ReoptCsvRow[] = [];
    for (const cfg of result.configs) {
      for (const s of cfg.perSeed) {
        rows.push({
          region,
          family: cfg.family,
          param_value: cfg.param_value,
          seed: s.seed,
          theta_p: s.thetaP,
          theta_r: s.thetaR,
          margin: s.margin,
          savings: s.savings,
          overhead: s.overhead,
          score: s.score,
          found: s.found,
        });
      }
    }
    writeReoptCsv(`${outputPrefix}_${region}.csv`, rows);
    allRows.push(...rows);

    summaryRecords.push(buildReoptSummary(region, result));

    if (!quiet) {
      const baseline = result.baseline;
      console.log(`  [${region}] start=${start} tpMax=${tpMax} \u03C3*=${calibration.sigmaStar.toFixed(3)}`);
      console.log(
        `    baseline \u03B8\u209A=${baseline.thetaP.toFixed(1)} \u03B8\u209B=${baseline.thetaR.toFixed(1)} margin=${baseline.margin.toFixed(1)} savings=${baseline.savings.toFixed(2)}% overhead=${baseline.overhead.toFixed(1)}%`,
      );
      for (const cfg of result.configs) {
        const label = cfg.family === "additive" ? `add ${cfg.param_value} x \u03C3*` : `delay ${cfg.param_value}`;
        if (cfg.best) {
          console.log(
            `    ${label.padEnd(12)} \u03B8\u209A=${cfg.best.thetaP.toFixed(1)} \u03B8\u209B=${cfg.best.thetaR.toFixed(1)} margin=${cfg.best.margin.toFixed(1)} savings=${cfg.best.savings.toFixed(2)}% found=${cfg.foundRate.toFixed(2)}`,
          );
        } else {
          console.log(`    ${label.padEnd(12)} no valid point (foundRate=${cfg.foundRate.toFixed(2)})`);
        }
      }
      console.log(`  JSON: ${outputPrefix}_${region}.json`);
      console.log(`  CSV:  ${outputPrefix}_${region}.csv`);
    }
  }

  if (regions.length > 1) {
    writeFileSync(`${outputPrefix}_summary.json`, JSON.stringify(summaryRecords, null, 2) + "\n", "utf-8");
  }
  if (csvPath) writeReoptCsv(csvPath, allRows);

  if (!quiet) {
    if (regions.length > 1) console.log(`  Summary: ${outputPrefix}_summary.json`);
    if (csvPath) console.log(`  CSV (all regions): ${csvPath}`);
    console.log("  Done.\n");
  }
}

export async function forecastSweepCli(raw: {
  mode: string;
  model?: string;
  regions?: string;
  year?: string;
  errorTypes?: string;
  levels?: string;
  horizons?: string;
  seedCount?: string;
  seedCountOther?: string;
  additiveLevels?: string;
  delaySteps?: string;
  resolution?: string;
  iterations?: string;
  budget?: string;
  alpha?: string;
  calibrationDir?: string;
  output?: string;
  csv?: string;
  quiet?: boolean;
  thetaP?: string;
  thetaR?: string;
  start?: string;
}): Promise<void> {
  if (raw.mode === "reopt") {
    await reoptSweepCli(raw);
    return;
  }
  if (raw.mode !== "fixed") {
    console.error(`  Unknown mode "${raw.mode}" (expected "fixed" or "reopt")`);
    process.exit(1);
  }

  const model = raw.model ?? "Deepseek";
  const regions = (raw.regions ?? "DE,IT,SE")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  if (regions.length === 0) {
    console.error("  No regions given");
    process.exit(1);
  }
  const year = Number(raw.year ?? "2025");
  const families = (raw.errorTypes ?? "additive,multiplicative,delay,arma,persistence")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  const levels = (raw.levels ?? "0,0.25,0.5,1,2,4").split(",").map((s) => Number(s.trim()));
  const horizons = (raw.horizons ?? "1,3,6,12,24,72").split(",").map((s) => Number(s.trim()));
  const seedCount = Number(raw.seedCount ?? "10");
  const seedCountOther = Number(raw.seedCountOther ?? "5");
  const calibrationDir = raw.calibrationDir ?? "publication/output/forecast";
  const outputPrefix = raw.output ?? "publication/output/forecast/fixed";
  const csvPath = raw.csv ?? null;
  const quiet = raw.quiet ?? false;
  const budget = 200;

  for (const f of families) {
    if (!VALID_FAMILIES.includes(f)) {
      console.error(`  Unknown error type "${f}". Available: ${VALID_FAMILIES.join(", ")}`);
      process.exit(1);
    }
  }

  const thetaPOverride = raw.thetaP != null ? Number(raw.thetaP) : null;
  const thetaROverride = raw.thetaR != null ? Number(raw.thetaR) : null;
  const startOverride = raw.start ?? null;
  if (regions.length !== 1 && (thetaPOverride != null || thetaROverride != null || startOverride)) {
    console.error("  --theta-p/--theta-r/--start are only meaningful when -r has exactly one region");
    process.exit(1);
  }

  const constants = loadJSON<Constants>("constants.json");
  const profiles = loadJSON<Record<string, TrainingProfile>>("profiles.json");
  const profile = profiles[model];
  if (!profile) {
    console.error(`  Unknown model: ${model}. Available: ${Object.keys(profiles).join(", ")}`);
    process.exit(1);
  }
  const fullProfile: FullProfile = {
    ...profile,
    gpuPowerTrain: constants.gpu_power_train,
    gpuPowerPause: constants.gpu_power_pause,
    pue: constants.pue,
    checkpointPauseTime: constants.checkpoint_pause_time,
    checkpointResumeTime: constants.checkpoint_resume_time,
  };

  mkdirSync(dirname(resolve(outputPrefix)), { recursive: true });
  if (csvPath) mkdirSync(dirname(resolve(csvPath)), { recursive: true });

  if (!quiet) {
    console.log(`\n  TheGreenEpoch Forecast Sweep \u2500 fixed policy`);
    console.log(`  Model: ${model}, Regions: ${regions.join(",")}, Year: ${year}`);
    console.log(`  Budget: ${budget}%, \u03B1=1, families: ${families.join(",")}`);
    console.log(`  Seeds: DE=1..${seedCount}, others=1..${seedCountOther}`);
  }

  const summaryRecords: Array<Record<string, unknown>> = [];
  const allRows: FixedRow[] = [];

  for (const region of regions) {
    const policy = resolvePolicy(region, thetaPOverride, thetaROverride, startOverride);
    const realized = loadCO2Timeline(region, year);
    const calibration = loadCalibration(region, calibrationDir);
    const sigmaStar = calibration.sigmaStar;
    const trainMean = calibration.trainMean;
    const sigmaRel = sigmaStar / trainMean;
    const coeffs = calibration.orders["1"]?.coeffs;
    if (!coeffs) {
      console.error(`  Calibration for ${region} is missing orders["1"].coeffs`);
      process.exit(1);
    }
    const seeds = makeSeeds(region, seedCount, seedCountOther);
    const configs = expandConfigs({ families, levels, horizons, sigmaStar, sigmaRel, coeffs });

    const { control, rows, summary } = runSweepRegion({
      region,
      profile: fullProfile,
      realized,
      historicalYears: [year],
      thetaP: policy.thetaP,
      thetaR: policy.thetaR,
      start: policy.start,
      budget,
      seeds,
      configs,
    });

    allRows.push(...rows);

    const regionJson = {
      model,
      region,
      year,
      policy: { thetaP: policy.thetaP, thetaR: policy.thetaR, start: policy.start },
      budget,
      seeds,
      calibration: { sigmaStar, sigmaRel, trainMean },
      control,
      rows,
      summary,
    };
    writeFileSync(`${outputPrefix}_${region}.json`, JSON.stringify(regionJson, null, 2) + "\n", "utf-8");
    writeCsv(`${outputPrefix}_${region}.csv`, rows);

    const summaryByFamily = new Map<string, FixedSummary[]>();
    for (const s of summary) {
      const arr = summaryByFamily.get(s.family);
      if (arr) {
        arr.push(s);
      } else {
        summaryByFamily.set(s.family, [s]);
      }
    }

    summaryRecords.push({
      region,
      sigmaStar,
      sigmaRel,
      s0: control.savings,
      graceLevels: {
        additive: graceEntry(summaryByFamily.get("additive") ?? []),
        multiplicative: graceEntry(summaryByFamily.get("multiplicative") ?? []),
        delay: graceEntry(summaryByFamily.get("delay") ?? []),
        persistence: graceEntry(summaryByFamily.get("persistence") ?? []),
        arma: graceEntry(summaryByFamily.get("arma") ?? []),
      },
      degradationAtSigmaStar: {
        additive: degradationEntry(summary.find((s) => s.family === "additive" && s.param_value === 1)),
        multiplicative: degradationEntry(summary.find((s) => s.family === "multiplicative" && s.param_value === 1)),
      },
      degradationAtH72: {
        persistence: degradationEntry(summary.find((s) => s.family === "persistence" && s.param_value === 72)),
        arma: degradationEntry(summary.find((s) => s.family === "arma" && s.param_value === 72)),
      },
    });

    if (!quiet) {
      console.log(`  [${region}] \u03B8\u209A=${policy.thetaP} \u03B8\u209B=${policy.thetaR} start=${policy.start} configs=${configs.length} seeds=${seeds.length}`);
      console.log(
        `    control S\u2080=${control.savings.toFixed(2)}% overhead=${control.overhead.toFixed(2)}% score=${control.score.toFixed(4)} pauses=${control.num_pauses} completed=${control.completed} within_budget=${control.within_budget}`,
      );
      for (const family of families) {
        const sums = summary.filter((s) => s.family === family);
        if (sums.length === 0) continue;
        const g = computeGraceLevel(sums);
        const vals = sums.map((s) => `${s.param_value}:${s.degradation_frac_mean.toFixed(4)}`).join(" ");
        console.log(
          `    ${family.padEnd(14)} deg(${vals}) grace=${g.graceLevel}${g.graceAtMax ? " (at max)" : ""}`,
        );
      }
      console.log(`  JSON: ${outputPrefix}_${region}.json`);
      console.log(`  CSV:  ${outputPrefix}_${region}.csv`);
    }
  }

  writeFileSync(`${outputPrefix}_summary.json`, JSON.stringify(summaryRecords, null, 2) + "\n", "utf-8");
  if (csvPath) writeCsv(csvPath, allRows);

  if (!quiet) {
    console.log(`  Summary: ${outputPrefix}_summary.json`);
    if (csvPath) console.log(`  CSV (all regions): ${csvPath}`);
    console.log("  Done.\n");
  }
}
