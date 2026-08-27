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
import { adaptiveMargin, widenedThresholds } from "../domain/adaptive-margin";
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
  ckptPause?: string;
  ckptResume?: string;
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
    checkpointPauseTime: raw.ckptPause != null ? parseFloat(raw.ckptPause) : constants.checkpoint_pause_time,
    checkpointResumeTime: raw.ckptResume != null ? parseFloat(raw.ckptResume) : constants.checkpoint_resume_time,
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
  ckptPause?: string;
  ckptResume?: string;
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
  if (raw.mode === "adaptive") {
    await adaptiveSweepCli(raw);
    return;
  }
  if (raw.mode === "adaptive-sensitivity") {
    await adaptiveSensitivityCli(raw);
    return;
  }
  if (raw.mode !== "fixed") {
    console.error(`  Unknown mode "${raw.mode}" (expected "fixed", "reopt", "adaptive" or "adaptive-sensitivity")`);
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
    checkpointPauseTime: raw.ckptPause != null ? parseFloat(raw.ckptPause) : constants.checkpoint_pause_time,
    checkpointResumeTime: raw.ckptResume != null ? parseFloat(raw.ckptResume) : constants.checkpoint_resume_time,
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

// ============================================================================
// Phase B.1 — stale-aware adaptive controller (--mode adaptive)
// ============================================================================

export const RECOVERY_EPS_PP = 1e-3;

/**
 * Fraction of the naive policy's staleness loss recovered by the adaptive
 * controller. S0_FF is the perfect-foresight savings of the naive (published)
 * policy. Convention: when the naive loss is not strictly positive
 * (S0_FF - S_naive <= RECOVERY_EPS_PP pp, including naive >= perfect), there is
 * nothing to recover and recovery is 0. `raw` is unclipped, `clipped` is
 * clamped to [0, 1].
 */
export function computeRecovery(
  sAdaptive: number,
  sNaive: number,
  sPerfect: number,
): { raw: number; clipped: number } {
  const denom = sPerfect - sNaive;
  if (denom <= RECOVERY_EPS_PP) return { raw: 0, clipped: 0 };
  const raw = (sAdaptive - sNaive) / denom;
  return { raw, clipped: Math.min(1, Math.max(0, raw)) };
}

/**
 * Aggregate a 5-min decision timeline to hourly means (12 points per hour),
 * expanded back to the original 5-min resolution so it can be used as a
 * decisionTimeline of the same length.
 */
export function hourlyAggregate(decision: CO2Timeline): CO2Timeline {
  const src = decision.carbonIntensity;
  const n = src.length;
  const out = new Array<number>(n);
  for (let i = 0; i < n; i += 12) {
    const hi = Math.min(i + 12, n);
    let sum = 0;
    for (let j = i; j < hi; j++) sum += src[j];
    const mean = sum / (hi - i);
    for (let j = i; j < hi; j++) out[j] = mean;
  }
  return { zone: decision.zone, years: [...decision.years], timestamps: [...decision.timestamps], carbonIntensity: out };
}

export interface AdaptivePolicyRow {
  theta_p: number;
  theta_r: number;
  savings: number;
  overhead: number;
  score: number;
  num_pauses: number;
  completed: boolean;
  within_budget: boolean;
}

export function evaluatePolicy(
  profile: FullProfile,
  realized: CO2Timeline,
  start: string,
  budget: number,
  thetaP: number,
  thetaR: number,
  decision: CO2Timeline,
  baselineEmissionsKg: number,
): AdaptivePolicyRow {
  const last = drain(profile, hysteresisPolicy(thetaP, thetaR), realized, {
    startTime: start,
    historicalYears: realized.years,
    overheadBudgetPct: budget,
    decisionTimeline: decision,
  });
  const m = metricsFromProgress(profile, last, baselineEmissionsKg, budget);
  return {
    theta_p: thetaP,
    theta_r: thetaR,
    savings: m.savings,
    overhead: m.overhead,
    score: m.score,
    num_pauses: m.num_pauses,
    completed: m.completed,
    within_budget: m.within_budget,
  };
}

export interface AdaptiveOraclePoint {
  source: "reopt_delay" | "arma_optimize";
  thetaP: number;
  thetaR: number;
  margin: number;
  savings: number;
  overhead: number;
  score: number;
  num_pauses: number | null;
  completed: boolean | null;
  within_budget: boolean | null;
}

/** Static oracle: reoptimized static thresholds under the arma(h) decision timeline. */
export function oracleStaticThresholds(params: {
  profile: FullProfile;
  realized: CO2Timeline;
  start: string;
  budget: number;
  tpMax: number;
  resolution: number;
  iterations: number;
  coeffs: ArCoeffs;
  h: number;
  seed: number;
}): AdaptiveOraclePoint | null {
  const { profile, realized, start, budget, tpMax, resolution, iterations, coeffs, h, seed } = params;
  const decision = applyForecast(realized, { type: "arma", order: 1, horizon: h, coeffs }, seed);
  const options: AdaptiveOptions = {
    thetaPauseMax: tpMax,
    overheadBudgetPct: budget,
    resolution,
    startDateResolution: 1,
    maxIterations: iterations,
    minStep: 3,
    shrinkFactor: 0.45,
    alpha: 1,
    fixedStartTime: start,
  };
  const { best } = runOptimization(profile, realized, realized.years, options, undefined, decision);
  if (!best) return null;
  return {
    source: "arma_optimize",
    thetaP: best.thetaPause,
    thetaR: best.thetaResume,
    margin: best.thetaPause - best.thetaResume,
    savings: best.co2SavingsPct,
    overhead: best.actualOverheadPct,
    score: best.score,
    num_pauses: best.numPauses,
    completed: best.completed,
    within_budget: best.withinBudget,
  };
}

/**
 * Per-region c* selection on the train years. c* = argmax over the grid of the
 * mean (over train years x horizons) clipped recovery, ties broken toward the
 * smallest c. Recovery is evaluated against each (year, h)'s own naive-perfect
 * gap, so the metric is self-normalizing across years.
 */
export function selectAdaptiveC(params: {
  region: string;
  profile: FullProfile;
  budget: number;
  calibration: CalibrationBundle;
  nominal: { thetaP: number; thetaR: number };
  naivePolicy: { thetaP: number; thetaR: number; start: string };
  trainYears: number[];
  horizons: number[];
  cGrid: number[];
  seed: number;
}): {
  grid: number[];
  chosenC: number;
  meanRecovery: Record<number, number>;
  perYear: Record<number, { meanRecovery: Record<number, number>; perH: Record<number, Record<number, number>> }>;
} {
  const { region, profile, budget, calibration, nominal, naivePolicy, trainYears, horizons, cGrid, seed } = params;
  const phi = calibration.orders["1"]?.coeffs.ar[0];
  const sigmaStar = calibration.sigmaStar;
  const coeffs = calibration.orders["1"]?.coeffs;
  if (!Number.isFinite(phi) || !coeffs) {
    throw new Error(`selectAdaptiveC: calibration for ${region} is missing orders["1"].coeffs`);
  }

  const perYear: Record<number, { meanRecovery: Record<number, number>; perH: Record<number, Record<number, number>> }> = {};

  for (const year of trainYears) {
    const realized = loadCO2Timeline(region, year);
    const start = naivePolicy.start;
    const baselineLast = drain(profile, neverPausePolicy(), realized, {
      startTime: start,
      historicalYears: realized.years,
      overheadBudgetPct: budget,
    });
    const baselineEm = baselineLast.totalEmissionsG / 1000;

    const decisions = new Map<number, CO2Timeline>();
    for (const h of horizons) {
      decisions.set(h, applyForecast(realized, { type: "arma", order: 1, horizon: h, coeffs }, seed));
    }

    const perfect = evaluatePolicy(
      profile, realized, start, budget, naivePolicy.thetaP, naivePolicy.thetaR,
      applyForecast(realized, { type: "identity" }, seed), baselineEm,
    ).savings;

    const naiveByH: Record<number, number> = {};
    const naiveCompletedByH: Record<number, boolean> = {};
    for (const h of horizons) {
      const naive = evaluatePolicy(
        profile, realized, start, budget, naivePolicy.thetaP, naivePolicy.thetaR,
        decisions.get(h) as CO2Timeline, baselineEm,
      );
      naiveByH[h] = naive.savings;
      naiveCompletedByH[h] = naive.completed;
    }

    const perH: Record<number, Record<number, number>> = {};
    for (const c of cGrid) {
      perH[c] = {};
      for (const h of horizons) {
        const margin = adaptiveMargin({ sigmaStar, phi, c, h });
        const w = widenedThresholds(nominal, margin);
        const adaptive = evaluatePolicy(
          profile, realized, start, budget, w.thetaP, w.thetaR,
          decisions.get(h) as CO2Timeline, baselineEm,
        );
        const rec = computeRecovery(adaptive.savings, naiveByH[h], perfect);
        // Completion guard (phase B.0 caveat): an incomplete (budget-exhausted)
        // run's savings is inflated vs a completed baseline and must not be
        // rewarded in c* selection. Both the adaptive run and the naive run
        // must complete for the recovery to count.
        perH[c][h] = adaptive.completed && naiveCompletedByH[h] ? rec.clipped : 0;
      }
    }

    const meanRecovery: Record<number, number> = {};
    for (const c of cGrid) {
      meanRecovery[c] = horizons.reduce((a, h) => a + perH[c][h], 0) / horizons.length;
    }
    perYear[year] = { meanRecovery, perH };
  }

  const meanRecovery: Record<number, number> = {};
  for (const c of cGrid) {
    meanRecovery[c] = trainYears.reduce((a, y) => a + perYear[y].meanRecovery[c], 0) / trainYears.length;
  }

  let chosenC = cGrid[0];
  for (const c of cGrid) {
    if (meanRecovery[c] > meanRecovery[chosenC]) chosenC = c;
  }
  return { grid: cGrid, chosenC, meanRecovery, perYear };
}

export interface AdaptiveRow {
  region: string;
  h: number;
  c: number;
  seed: number;
  theta_p_naive: number;
  theta_r_naive: number;
  savings_naive: number;
  overhead_naive: number;
  score_naive: number;
  num_pauses_naive: number;
  completed_naive: boolean;
  within_budget_naive: boolean;
  savings_perfect: number;
  overhead_perfect: number;
  score_perfect: number;
  num_pauses_perfect: number;
  completed_perfect: boolean;
  within_budget_perfect: boolean;
  theta_p_adaptive: number;
  theta_r_adaptive: number;
  margin_adaptive: number;
  savings_adaptive: number;
  overhead_adaptive: number;
  score_adaptive: number;
  num_pauses_adaptive: number;
  completed_adaptive: boolean;
  within_budget_adaptive: boolean;
  oracle_source: string;
  theta_p_oracle: number;
  theta_r_oracle: number;
  margin_oracle: number;
  savings_oracle: number;
  overhead_oracle: number;
  score_oracle: number;
  num_pauses_oracle: number | null;
  completed_oracle: boolean | null;
  within_budget_oracle: boolean | null;
  ref_savings_delay: number | null;
  ref_theta_p_delay: number | null;
  ref_theta_r_delay: number | null;
  theta_p_dtpr: number;
  theta_r_dtpr: number;
  margin_dtpr: number;
  beta_dtpr: number;
  savings_dtpr: number;
  overhead_dtpr: number;
  score_dtpr: number;
  num_pauses_dtpr: number;
  completed_dtpr: boolean;
  within_budget_dtpr: boolean;
  s0_naive_ff: number;
  s0_adaptive_ff: number;
  recovery_raw: number;
  recovery: number;
  oracle_gap: number;
  recovery_dtpr: number;
  recovery_vs_oracle: number;
}

export interface AdaptiveRegionResult {
  region: string;
  model: string;
  year: number;
  budget: number;
  start: string;
  seed: number;
  calibration: { sigmaStar: number; phi: number; trainYears: number[]; testYear: number };
  nominal: { thetaP: number; thetaR: number; margin: number; source: string };
  naivePolicy: { thetaP: number; thetaR: number; start: string };
  s0_naive_ff: number;
  s0_adaptive_ff: number;
  c_selection: {
    grid: number[];
    chosenC: number;
    meanRecovery: Record<number, number>;
    perYear: Record<number, { meanRecovery: Record<number, number>; perH: Record<number, Record<number, number>> }>;
  };
  dtpr: {
    beta: number;
    center: number;
    thetaP: number;
    thetaR: number;
    margin: number;
    derivation: string;
  };
  rows: AdaptiveRow[];
}

export function runAdaptiveRegion(params: {
  region: string;
  model: string;
  year: number;
  profile: FullProfile;
  calibration: CalibrationBundle;
  nominal: { thetaP: number; thetaR: number };
  naivePolicy: { thetaP: number; thetaR: number; start: string };
  trainYears: number[];
  horizons: number[];
  cGrid: number[];
  budget: number;
  delayOracle: Map<number, { thetaP: number; thetaR: number; savings: number; overhead: number; score: number }>;
  optimizer: { resolution: number; iterations: number; tpMax: number };
  seed: number;
}): AdaptiveRegionResult {
  const {
    region, model, year, profile, calibration, nominal, naivePolicy, trainYears,
    horizons, cGrid, budget, delayOracle, optimizer, seed,
  } = params;
  const phi = calibration.orders["1"]?.coeffs.ar[0];
  const coeffs = calibration.orders["1"]?.coeffs;
  const sigmaStar = calibration.sigmaStar;
  if (!Number.isFinite(phi) || !coeffs) {
    throw new Error(`runAdaptiveRegion: calibration for ${region} is missing orders["1"].coeffs`);
  }
  const start = naivePolicy.start;

  const realized = loadCO2Timeline(region, year);
  const baselineLast = drain(profile, neverPausePolicy(), realized, {
    startTime: start,
    historicalYears: realized.years,
    overheadBudgetPct: budget,
  });
  const baselineEm = baselineLast.totalEmissionsG / 1000;

  const perfect = evaluatePolicy(
    profile, realized, start, budget, naivePolicy.thetaP, naivePolicy.thetaR,
    applyForecast(realized, { type: "identity" }, seed), baselineEm,
  );
  const adaptivePerfect = evaluatePolicy(
    profile, realized, start, budget, nominal.thetaP, nominal.thetaR,
    applyForecast(realized, { type: "identity" }, seed), baselineEm,
  );

  const cSelection = selectAdaptiveC({
    region, profile, budget, calibration, nominal, naivePolicy, trainYears, horizons, cGrid, seed,
  });
  const chosenC = cSelection.chosenC;

  // DTPR-style benchmark: constant double thresholds around the nominal midpoint.
  const tauCkptH = (profile.checkpointPauseTime + profile.checkpointResumeTime) / 3600;
  const dtprBeta = tauCkptH * nominal.thetaP;
  const dtprCenter = (nominal.thetaP + nominal.thetaR) / 2;
  const dtprThetaP = dtprCenter + dtprBeta;
  const dtprThetaR = dtprCenter - dtprBeta;

  const rows: AdaptiveRow[] = [];
  for (const h of horizons) {
    const decision = applyForecast(realized, { type: "arma", order: 1, horizon: h, coeffs }, seed);

    const naive = evaluatePolicy(profile, realized, start, budget, naivePolicy.thetaP, naivePolicy.thetaR, decision, baselineEm);

    const margin = adaptiveMargin({ sigmaStar, phi, c: chosenC, h });
    const w = widenedThresholds(nominal, margin);
    const adaptive = evaluatePolicy(profile, realized, start, budget, w.thetaP, w.thetaR, decision, baselineEm);

    const dtprDecision = hourlyAggregate(applyForecast(realized, { type: "arma", order: 1, horizon: h, coeffs }, seed));
    const dtpr = evaluatePolicy(profile, realized, start, budget, dtprThetaP, dtprThetaR, dtprDecision, baselineEm);

    let oracle: AdaptiveOraclePoint;
    let ref: { savings: number; thetaP: number; thetaR: number } | null = null;
    const delayRef = delayOracle.get(h);
    if (delayRef) {
      oracle = {
        source: "reopt_delay",
        thetaP: delayRef.thetaP,
        thetaR: delayRef.thetaR,
        margin: delayRef.thetaP - delayRef.thetaR,
        savings: delayRef.savings,
        overhead: delayRef.overhead,
        score: delayRef.score,
        num_pauses: null,
        completed: null,
        within_budget: null,
      };
      ref = { savings: delayRef.savings, thetaP: delayRef.thetaP, thetaR: delayRef.thetaR };
    } else {
      oracle = oracleStaticThresholds({
        profile, realized, start, budget,
        tpMax: optimizer.tpMax, resolution: optimizer.resolution, iterations: optimizer.iterations,
        coeffs, h, seed,
      }) ?? {
        source: "arma_optimize" as const,
        thetaP: NaN, thetaR: NaN, margin: NaN, savings: NaN, overhead: NaN, score: NaN,
        num_pauses: null, completed: null, within_budget: null,
      };
    }

    const rec = computeRecovery(adaptive.savings, naive.savings, perfect.savings);
    const recDtpr = computeRecovery(dtpr.savings, naive.savings, perfect.savings);
    // Completion guard (phase B.0 caveat): savings of an incomplete
    // (budget-exhausted) run is inflated vs a completed baseline. Headline
    // recovery is 0 for incomplete runs; the raw formula is kept as recovery_raw.
    const recoveryCompleted = adaptive.completed && naive.completed;
    const recoveryDtprCompleted = dtpr.completed && naive.completed;
    // Fraction of the *recoverable* gap (naive -> static oracle) that the
    // adaptive controller closes. 0 when naive >= oracle or runs incomplete.
    let recoveryVsOracle = 0;
    if (recoveryCompleted && Number.isFinite(oracle.savings)) {
      const denom = oracle.savings - naive.savings;
      if (denom > RECOVERY_EPS_PP) {
        recoveryVsOracle = Math.min(1, Math.max(0, (adaptive.savings - naive.savings) / denom));
      }
    }

    rows.push({
      region, h, c: chosenC, seed,
      theta_p_naive: naive.theta_p, theta_r_naive: naive.theta_r,
      savings_naive: naive.savings, overhead_naive: naive.overhead, score_naive: naive.score,
      num_pauses_naive: naive.num_pauses, completed_naive: naive.completed, within_budget_naive: naive.within_budget,
      savings_perfect: perfect.savings, overhead_perfect: perfect.overhead, score_perfect: perfect.score,
      num_pauses_perfect: perfect.num_pauses, completed_perfect: perfect.completed, within_budget_perfect: perfect.within_budget,
      theta_p_adaptive: adaptive.theta_p, theta_r_adaptive: adaptive.theta_r, margin_adaptive: adaptive.theta_p - adaptive.theta_r,
      savings_adaptive: adaptive.savings, overhead_adaptive: adaptive.overhead, score_adaptive: adaptive.score,
      num_pauses_adaptive: adaptive.num_pauses, completed_adaptive: adaptive.completed, within_budget_adaptive: adaptive.within_budget,
      oracle_source: oracle.source,
      theta_p_oracle: oracle.thetaP, theta_r_oracle: oracle.thetaR, margin_oracle: oracle.thetaP - oracle.thetaR,
      savings_oracle: oracle.savings, overhead_oracle: oracle.overhead, score_oracle: oracle.score,
      num_pauses_oracle: oracle.num_pauses, completed_oracle: oracle.completed, within_budget_oracle: oracle.within_budget,
      ref_savings_delay: ref?.savings ?? null, ref_theta_p_delay: ref?.thetaP ?? null, ref_theta_r_delay: ref?.thetaR ?? null,
      theta_p_dtpr: dtprThetaP, theta_r_dtpr: dtprThetaR, margin_dtpr: dtprThetaP - dtprThetaR, beta_dtpr: dtprBeta,
      savings_dtpr: dtpr.savings, overhead_dtpr: dtpr.overhead, score_dtpr: dtpr.score,
      num_pauses_dtpr: dtpr.num_pauses, completed_dtpr: dtpr.completed, within_budget_dtpr: dtpr.within_budget,
      s0_naive_ff: perfect.savings,
      s0_adaptive_ff: adaptivePerfect.savings,
      recovery_raw: rec.raw,
      recovery: recoveryCompleted ? rec.clipped : 0,
      oracle_gap: oracle.savings - adaptive.savings,
      recovery_dtpr: recoveryDtprCompleted ? recDtpr.clipped : 0,
      recovery_vs_oracle: recoveryVsOracle,
    });
  }

  return {
    region, model, year, budget, start, seed,
    calibration: { sigmaStar, phi, trainYears: [...trainYears], testYear: year },
    nominal: { thetaP: nominal.thetaP, thetaR: nominal.thetaR, margin: nominal.thetaP - nominal.thetaR, source: "reopt_summary.json baseline" },
    naivePolicy: { thetaP: naivePolicy.thetaP, thetaR: naivePolicy.thetaR, start },
    s0_naive_ff: perfect.savings,
    s0_adaptive_ff: adaptivePerfect.savings,
    c_selection: {
      grid: cSelection.grid,
      chosenC,
      meanRecovery: cSelection.meanRecovery,
      perYear: cSelection.perYear,
    },
    dtpr: {
      beta: dtprBeta,
      center: dtprCenter,
      thetaP: dtprThetaP,
      thetaR: dtprThetaR,
      margin: dtprThetaP - dtprThetaR,
      derivation: `DTPR-style double threshold: constant separation 2*beta around the reopt nominal midpoint; beta = (checkpointPauseTime + checkpointResumeTime)/3600 * nominal.thetaP = the CO2-equivalent (in g/kWh of the nominal operating scale) of one checkpoint/restore cycle amortized over one hour of shifted training energy.`,
    },
    rows,
  };
}

const ADAPTIVE_CSV_HEADER =
  "region,h,c,seed," +
  "theta_p_naive,theta_r_naive,savings_naive,overhead_naive,score_naive,num_pauses_naive,completed_naive,within_budget_naive," +
  "savings_perfect,overhead_perfect,score_perfect,num_pauses_perfect,completed_perfect,within_budget_perfect," +
  "theta_p_adaptive,theta_r_adaptive,margin_adaptive,savings_adaptive,overhead_adaptive,score_adaptive,num_pauses_adaptive,completed_adaptive,within_budget_adaptive," +
  "oracle_source,theta_p_oracle,theta_r_oracle,margin_oracle,savings_oracle,overhead_oracle,score_oracle,num_pauses_oracle,completed_oracle,within_budget_oracle," +
  "ref_savings_delay,ref_theta_p_delay,ref_theta_r_delay," +
  "theta_p_dtpr,theta_r_dtpr,margin_dtpr,beta_dtpr,savings_dtpr,overhead_dtpr,score_dtpr,num_pauses_dtpr,completed_dtpr,within_budget_dtpr," +
  "s0_naive_ff,s0_adaptive_ff,recovery_raw,recovery,oracle_gap,recovery_dtpr,recovery_vs_oracle";

function adaptiveCell(v: number | boolean | null): string {
  if (v === null) return "NA";
  if (typeof v === "boolean") return String(v);
  return v.toPrecision(6);
}

function adaptiveRowToCsv(r: AdaptiveRow): string[] {
  return [
    r.region, String(r.h), String(r.c), String(r.seed),
    adaptiveCell(r.theta_p_naive), adaptiveCell(r.theta_r_naive),
    adaptiveCell(r.savings_naive), adaptiveCell(r.overhead_naive), adaptiveCell(r.score_naive), adaptiveCell(r.num_pauses_naive), adaptiveCell(r.completed_naive), adaptiveCell(r.within_budget_naive),
    adaptiveCell(r.savings_perfect), adaptiveCell(r.overhead_perfect), adaptiveCell(r.score_perfect), adaptiveCell(r.num_pauses_perfect), adaptiveCell(r.completed_perfect), adaptiveCell(r.within_budget_perfect),
    adaptiveCell(r.theta_p_adaptive), adaptiveCell(r.theta_r_adaptive), adaptiveCell(r.margin_adaptive),
    adaptiveCell(r.savings_adaptive), adaptiveCell(r.overhead_adaptive), adaptiveCell(r.score_adaptive), adaptiveCell(r.num_pauses_adaptive), adaptiveCell(r.completed_adaptive), adaptiveCell(r.within_budget_adaptive),
    r.oracle_source, adaptiveCell(r.theta_p_oracle), adaptiveCell(r.theta_r_oracle), adaptiveCell(r.margin_oracle),
    adaptiveCell(r.savings_oracle), adaptiveCell(r.overhead_oracle), adaptiveCell(r.score_oracle), adaptiveCell(r.num_pauses_oracle), adaptiveCell(r.completed_oracle), adaptiveCell(r.within_budget_oracle),
    adaptiveCell(r.ref_savings_delay), adaptiveCell(r.ref_theta_p_delay), adaptiveCell(r.ref_theta_r_delay),
    adaptiveCell(r.theta_p_dtpr), adaptiveCell(r.theta_r_dtpr), adaptiveCell(r.margin_dtpr), adaptiveCell(r.beta_dtpr),
    adaptiveCell(r.savings_dtpr), adaptiveCell(r.overhead_dtpr), adaptiveCell(r.score_dtpr), adaptiveCell(r.num_pauses_dtpr), adaptiveCell(r.completed_dtpr), adaptiveCell(r.within_budget_dtpr),
    adaptiveCell(r.s0_naive_ff), adaptiveCell(r.s0_adaptive_ff),
    adaptiveCell(r.recovery_raw), adaptiveCell(r.recovery), adaptiveCell(r.oracle_gap), adaptiveCell(r.recovery_dtpr), adaptiveCell(r.recovery_vs_oracle),
  ];
}

function writeAdaptiveCsv(path: string, rows: AdaptiveRow[]): void {
  const body = rows.map((r) => adaptiveRowToCsv(r).join(",")).join("\n");
  writeFileSync(path, ADAPTIVE_CSV_HEADER + (rows.length > 0 ? "\n" + body + "\n" : "\n"), "utf-8");
}

function adaptiveSummaryRegionEntry(result: AdaptiveRegionResult): Record<string, unknown> {
  return {
    region: result.region,
    chosenC: result.c_selection.chosenC,
    c_selection: {
      grid: result.c_selection.grid,
      meanRecovery: result.c_selection.meanRecovery,
      perYear: Object.fromEntries(
        Object.entries(result.c_selection.perYear).map(([year, v]) => [year, { meanRecovery: v.meanRecovery }]),
      ),
    },
    s0_naive_ff: result.s0_naive_ff,
    s0_adaptive_ff: result.s0_adaptive_ff,
    nominal: result.nominal,
    naivePolicy: result.naivePolicy,
    dtpr: { beta: result.dtpr.beta, center: result.dtpr.center, thetaP: result.dtpr.thetaP, thetaR: result.dtpr.thetaR, margin: result.dtpr.margin, derivation: result.dtpr.derivation },
    rows: result.rows.map((r) => ({
      h: r.h,
      savings_naive: r.savings_naive,
      overhead_naive: r.overhead_naive,
      savings_perfect: r.savings_perfect,
      savings_adaptive: r.savings_adaptive,
      overhead_adaptive: r.overhead_adaptive,
      completed_adaptive: r.completed_adaptive,
      recovery: r.recovery,
      recovery_raw: r.recovery_raw,
      oracle_gap: r.oracle_gap,
      savings_oracle: r.savings_oracle,
      oracle_theta_p: r.theta_p_oracle,
      oracle_theta_r: r.theta_r_oracle,
      savings_dtpr: r.savings_dtpr,
      overhead_dtpr: r.overhead_dtpr,
      recovery_dtpr: r.recovery_dtpr,
      recovery_vs_oracle: r.recovery_vs_oracle,
    })),
  };
}

async function adaptiveSweepCli(raw: {
  model?: string;
  regions?: string;
  year?: string;
  trainYears?: string;
  horizons?: string;
  cGrid?: string;
  budget?: string;
  resolution?: string;
  iterations?: string;
  ckptPause?: string;
  ckptResume?: string;
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
  const trainYears = (raw.trainYears ?? "2022,2023,2024").split(",").map((s) => Number(s.trim()));
  const horizons = (raw.horizons ?? "1,3,6,12,24,72").split(",").map((s) => Number(s.trim()));
  const cGrid = (raw.cGrid ?? "0,0.25,0.5,0.75,1,1.5,2").split(",").map((s) => Number(s.trim()));
  const budget = Number(raw.budget ?? "200");
  const resolution = Number(raw.resolution ?? "10");
  const iterations = Number(raw.iterations ?? "6");
  const calibrationDir = raw.calibrationDir ?? "publication/output/forecast";
  const outputPrefix = raw.output ?? "publication/output/forecast/adaptive";
  const csvPath = raw.csv ?? null;
  const quiet = raw.quiet ?? false;
  const seed = 1;

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
    checkpointPauseTime: raw.ckptPause != null ? parseFloat(raw.ckptPause) : constants.checkpoint_pause_time,
    checkpointResumeTime: raw.ckptResume != null ? parseFloat(raw.ckptResume) : constants.checkpoint_resume_time,
  };

  mkdirSync(dirname(resolve(outputPrefix)), { recursive: true });
  if (csvPath) mkdirSync(dirname(resolve(csvPath)), { recursive: true });

  const reoptSummary = JSON.parse(readFileSync(resolve(calibrationDir, "reopt_summary.json"), "utf-8")) as ReoptSummaryRegion[];
  const nominalByRegion = Object.fromEntries(reoptSummary.map((r) => [r.region, r.baseline]));

  if (!quiet) {
    console.log(`\n  TheGreenEpoch Forecast Sweep \u2500 adaptive controller (Phase B.1)`);
    console.log(`  Model: ${model}, Regions: ${regions.join(",")}, Test year: ${year}, Train years: ${trainYears.join(",")}`);
    console.log(`  Budget: ${budget}%, optimizer ${resolution}x${iterations}, horizons: ${horizons.join(",")}`);
    console.log(`  c-grid: ${cGrid.join(",")}; deterministic (no RNG), seed kept for API parity`);
  }

  const summary: Array<Record<string, unknown>> = [];
  const allRows: AdaptiveRow[] = [];

  for (const region of regions) {
    const calibration = loadCalibration(region, calibrationDir);
    const nominalBase = nominalByRegion[region];
    if (!nominalBase) {
      console.error(`  No reopt baseline for region ${region} in reopt_summary.json`);
      process.exit(1);
    }
    const nominal = { thetaP: nominalBase.thetaP, thetaR: nominalBase.thetaR };
    const naivePolicy = DEFAULT_POLICIES[region];
    if (!naivePolicy) {
      console.error(`  No default policy for region ${region}`);
      process.exit(1);
    }
    const tpMax = region === "SE" ? 100 : 800;

    const reoptRegion = JSON.parse(readFileSync(resolve(calibrationDir, `reopt_${region}.json`), "utf-8")) as ReoptRegionResult;
    const delayOracle = new Map<number, { thetaP: number; thetaR: number; savings: number; overhead: number; score: number }>();
    for (const cfg of reoptRegion.configs) {
      if (cfg.family === "delay" && cfg.best) {
        delayOracle.set(cfg.param_value, {
          thetaP: cfg.best.thetaP,
          thetaR: cfg.best.thetaR,
          savings: cfg.best.savings,
          overhead: cfg.best.overhead,
          score: cfg.best.score,
        });
      }
    }

    const result = runAdaptiveRegion({
      region, model, year, profile: fullProfile, calibration, nominal, naivePolicy,
      trainYears, horizons, cGrid, budget,
      delayOracle,
      optimizer: { resolution, iterations, tpMax },
      seed,
    });

    writeFileSync(`${outputPrefix}_${region}.json`, JSON.stringify(result, null, 2) + "\n", "utf-8");
    writeAdaptiveCsv(`${outputPrefix}_${region}.csv`, result.rows);
    allRows.push(...result.rows);
    summary.push(adaptiveSummaryRegionEntry(result));

    if (!quiet) {
      const cs = result.c_selection;
      console.log(`  [${region}] \u03C3*=${calibration.sigmaStar.toFixed(3)} \u03C6=${calibration.orders["1"].coeffs.ar[0].toFixed(6)} c*=${cs.chosenC} S\u2080_FF=${result.s0_naive_ff.toFixed(2)}% (adaptive ceiling ${result.s0_adaptive_ff.toFixed(2)}%)`);
      console.log(`    meanRecovery by c: ${cs.grid.map((c) => `${c}:${cs.meanRecovery[c].toFixed(4)}`).join(" ")}`);
      console.log("    h   naive   adapt  recov  gap    oracle dtpr");
      for (const r of result.rows) {
        console.log(
          `    ${String(r.h).padEnd(3)} ${r.savings_naive.toFixed(2).padStart(6)} ${r.savings_adaptive.toFixed(2).padStart(7)} ${r.recovery.toFixed(3).padStart(5)} ${r.oracle_gap.toFixed(3).padStart(6)} ${r.savings_oracle.toFixed(2).padStart(7)} ${r.savings_dtpr.toFixed(2).padStart(5)}`,
        );
      }
      console.log(`  JSON: ${outputPrefix}_${region}.json`);
      console.log(`  CSV:  ${outputPrefix}_${region}.csv`);
    }
  }

  const dtprExample = summary[0]?.dtpr as Record<string, unknown> | undefined;
  const summaryDoc = {
    spec: "Phase B.1 stale-aware adaptive controller",
    generatedDate: new Date().toISOString().slice(0, 10),
    deterministic: true,
    methodology: {
      decisionModel:
        "decide-on-forecast / pay-on-realized. Decision timeline = AR(1) h-step-ahead forecast (applyForecast {type:arma, order:1, horizon:h, coeffs: calibration.orders['1'].coeffs}) = a decision made on data h*5min old. Emissions evaluated on the realized timeline. Deterministic (arma/identity have no RNG); the seed field is kept for API parity and is a no-op.",
      adaptiveRule:
        "margin(h) = c*sigmaStar*sqrt((1-phi^(2h))/(1-phi^2)); thresholds = widenedThresholds(reoptNominal, margin) i.e. theta_p = nominal.thetaP + margin/2, theta_r = nominal.thetaR - margin/2.",
      adaptiveRuleLimitation:
        "The margin rule is linear in c (margin(h) = c * sigmaStar * scale(h)), so it can express any margin; the observed modest recovery at h>=6 is a property of the CHOSEN c*, not of the rule's expressiveness. At c=1 the model interval under-widens vs the empirical staleness error (DE h=72: model interval ~= 30.7 g/kWh vs AR(1) h-step RMSE ~= 113 g/kWh; IT ~= 77.6; SE ~= 7.8). The c-grid ceiling (<= 2) — within which the train-year mean-recovery signal is near-zero and noisy (<= 0.056 everywhere; see c_selection) — together with the train-year c-selection prevented selecting the c ~= 6-7 that approaches the static oracle on the 2025 test year (DE h=72: c=6 -> S ~= 31.4, completed, recovery_vs_oracle ~= 0.75; c=8 -> 36.3 but budget-infeasible; see adaptive_sensitivity_*.json). The oracle's center at h=72 also drifts (~283 g/kWh vs the nominal midpoint ~= 270), which the symmetric widening rule cannot express. The residual h=72 loss unreachable by any static-threshold policy is 32/48/30 % of the loss for DE/IT/SE.",
      cSelection:
        "per-region c* = argmax over c in grid of the mean clipped recovery across train years {trainYears} x horizons {1,3,6,12,24,72}; ties -> smallest c. Recovery measured per (year,h) against that (year,h) naive-perfect gap; self-normalizing across years.",
      recovery:
        "recovery = clip((S_adaptive - S_naive)/(S0_FF - S_naive), 0, 1) with S0_FF = perfect-foresight savings of the naive (published) policy; recovery = 0 when S0_FF - S_naive <= 1e-3 pp (no positive loss to recover, incl. naive >= perfect). recovery_raw is the unclipped value. Completion guard (phase B.0 caveat): an incomplete (budget-exhausted) run's savings is inflated vs a completed baseline, so headline recovery is 0 whenever the adaptive or naive run does not complete. The guard changes only the budget-exhausted rows; every completed-run recovery is identical with or without the guard.",
      recoveryVsOracle:
        "recovery_vs_oracle = clip((S_adaptive - S_naive)/(S_oracle - S_naive), 0, 1): the fraction of the recoverable (naive -> static oracle) gap that the adaptive controller closes; 0 when naive >= oracle or runs incomplete. This is the HEADLINE controller metric (recovery is dominated by the structural, oracle-unreachable share of the loss): DE h=1 recovery_vs_oracle 0.95, h=3 0.80.",
      completionWarning:
        "Phase B.0 caveat confirmed in B.1: aggressive margin widening (large c) and DTPR-style constant margins push overhead to the 200% budget cap and the run stops incomplete with inflated savings. All headline recovery numbers therefore require the completion guard. This makes the completion constraint a first-class design rule for the adaptive controller.",
      baselines: {
        naiveFixed: "published thresholds (DEFAULT_POLICIES) under the arma(h) decision timeline; reproduces fixed_summary.json arma rows.",
        perfectForesight: "published thresholds under the identity decision; equals fixed_summary.json control S0.",
        staticOracle:
          "per-h reoptimized static thresholds under the arma(h) decision timeline (runOptimization, reopt settings, one seed). h in {1,6} sourced from committed reopt_{region}.json delay rows (delay(h) == arma(h) exactly to 6 decimals for near-unit-root phi).",
      },
      dtprBenchmark:
        "DTPR-style double threshold with constant separation 2*beta around the reopt nominal midpoint; decision on the hourly-aggregated arma(h) forecast signal; evaluated on the same arma(h) traces. beta = (checkpointPauseTime + checkpointResumeTime)/3600 * nominal.thetaP (CO2-equivalent of one checkpoint/restore cycle amortized over one hour of shifted training energy).",
      dtprNote:
        "For IT the DTPR run is budget-exhausted (incomplete) at every h (overhead 200.0%); its reported savings (33.4-34.2%, above the perfect-foresight S0 of 32.67%) are an incompleteness artifact of the phase-B.0 caveat and must not be read as genuine savings.",
      reproducibilityNote:
        "The CSV columns are formatted with toPrecision(6); any claim-to-artifact trace must use the JSON artifacts, which carry full numeric precision. The path is deterministic: re-running produces byte-identical JSON (no RNG in the arma/identity/delay decision models or the optimizer).",
    },
    dtprDerivation: dtprExample ? {
      betaFormula: "beta = (tau_pause + tau_resume)/3600 * thetaP_nominal",
      unit: "g/kWh of the nominal operating scale (thetaP_nominal)",
      rationale:
        "A pause/resume cycle costs P*(tau_pause+tau_resume) of energy at the checkpoint's CI. Equating the CO2 cost of that cycle to the CO2 saved by shifting one hour of training from the resume level to the pause level gives (theta_p - theta_r) = 2*beta with beta = tau_hours*CI_ref, CI_ref = nominal thetaP. This is the margin at which switching cost equals pause savings (per hour of shifted training), matching DTPR's switching-cost parameter beta.",
    } : undefined,
    regions: summary,
  };
  writeFileSync(`${outputPrefix}_summary.json`, JSON.stringify(summaryDoc, null, 2) + "\n", "utf-8");
  if (csvPath) writeAdaptiveCsv(csvPath, allRows);

  if (!quiet) {
    if (regions.length > 1) console.log(`  Summary: ${outputPrefix}_summary.json`);
    if (csvPath) console.log(`  CSV (all regions): ${csvPath}`);
    console.log("  Done.\n");
  }
}

// ============================================================================
// Phase B.1 — recovery-vs-c sensitivity (adversarial review M1/M2/m1)
// Deterministic per-(region, c, h) sweep on the test year with the extended c
// grid, using the SAME decision model (arma(h)), reopt nominal anchors, budget
// (200%) and completion-guard conventions as --mode adaptive. Every cell is
// kept (including budget-infeasible ones) so the feasible-c envelope is
// directly readable; `recovery` and `recovery_vs_oracle` apply the completion
// guard exactly as in the adaptive mode.
// ============================================================================

export interface SensitivityOracle {
  source: string;
  thetaP: number;
  thetaR: number;
  margin: number;
  savings: number;
  overhead: number;
  score: number;
}

export interface AdaptiveSensitivityRow {
  region: string;
  h: number;
  c: number;
  theta_p: number;
  theta_r: number;
  margin: number;
  savings: number;
  overhead: number;
  score: number;
  num_pauses: number;
  completed: boolean;
  within_budget: boolean;
  savings_naive: number;
  completed_naive: boolean;
  savings_perfect: number;
  savings_oracle: number;
  recovery_raw: number;
  recovery: number;
  recovery_vs_oracle: number;
}

export interface AdaptiveSensitivityRegion {
  region: string;
  model: string;
  year: number;
  budget: number;
  start: string;
  seed: number;
  calibration: { sigmaStar: number; phi: number };
  nominal: { thetaP: number; thetaR: number; margin: number; source: string };
  naivePolicy: { thetaP: number; thetaR: number; start: string };
  s0_naive_ff: number;
  s0_adaptive_ff: number;
  cGrid: number[];
  horizons: number[];
  oracles: Record<number, SensitivityOracle>;
  feasibleC: Record<number, number[]>;
  rows: AdaptiveSensitivityRow[];
}

export function runAdaptiveSensitivityRegion(params: {
  region: string;
  model: string;
  year: number;
  profile: FullProfile;
  calibration: CalibrationBundle;
  nominal: { thetaP: number; thetaR: number };
  naivePolicy: { thetaP: number; thetaR: number; start: string };
  horizons: number[];
  cGrid: number[];
  budget: number;
  delayOracle: Map<number, { thetaP: number; thetaR: number; savings: number; overhead: number; score: number }>;
  optimizer: { resolution: number; iterations: number; tpMax: number };
  seed: number;
}): AdaptiveSensitivityRegion {
  const {
    region, model, year, profile, calibration, nominal, naivePolicy,
    horizons, cGrid, budget, delayOracle, optimizer, seed,
  } = params;
  const phi = calibration.orders["1"]?.coeffs.ar[0];
  const coeffs = calibration.orders["1"]?.coeffs;
  const sigmaStar = calibration.sigmaStar;
  if (!Number.isFinite(phi) || !coeffs) {
    throw new Error(`runAdaptiveSensitivityRegion: calibration for ${region} is missing orders["1"].coeffs`);
  }
  const start = naivePolicy.start;

  const realized = loadCO2Timeline(region, year);
  const baselineLast = drain(profile, neverPausePolicy(), realized, {
    startTime: start,
    historicalYears: realized.years,
    overheadBudgetPct: budget,
  });
  const baselineEm = baselineLast.totalEmissionsG / 1000;

  const perfect = evaluatePolicy(
    profile, realized, start, budget, naivePolicy.thetaP, naivePolicy.thetaR,
    applyForecast(realized, { type: "identity" }, seed), baselineEm,
  );
  const adaptivePerfect = evaluatePolicy(
    profile, realized, start, budget, nominal.thetaP, nominal.thetaR,
    applyForecast(realized, { type: "identity" }, seed), baselineEm,
  );

  const oracles: Record<number, SensitivityOracle> = {};
  const rows: AdaptiveSensitivityRow[] = [];
  const feasibleC: Record<number, number[]> = {};

  for (const h of horizons) {
    const decision = applyForecast(realized, { type: "arma", order: 1, horizon: h, coeffs }, seed);
    const naive = evaluatePolicy(profile, realized, start, budget, naivePolicy.thetaP, naivePolicy.thetaR, decision, baselineEm);

    let oracle: AdaptiveOraclePoint;
    const delayRef = delayOracle.get(h);
    if (delayRef) {
      oracle = {
        source: "reopt_delay",
        thetaP: delayRef.thetaP,
        thetaR: delayRef.thetaR,
        margin: delayRef.thetaP - delayRef.thetaR,
        savings: delayRef.savings,
        overhead: delayRef.overhead,
        score: delayRef.score,
        num_pauses: null,
        completed: null,
        within_budget: null,
      };
    } else {
      oracle = oracleStaticThresholds({
        profile, realized, start, budget,
        tpMax: optimizer.tpMax, resolution: optimizer.resolution, iterations: optimizer.iterations,
        coeffs, h, seed,
      }) ?? {
        source: "arma_optimize" as const,
        thetaP: NaN, thetaR: NaN, margin: NaN, savings: NaN, overhead: NaN, score: NaN,
        num_pauses: null, completed: null, within_budget: null,
      };
    }
    oracles[h] = {
      source: oracle.source,
      thetaP: oracle.thetaP,
      thetaR: oracle.thetaR,
      margin: oracle.thetaP - oracle.thetaR,
      savings: oracle.savings,
      overhead: oracle.overhead,
      score: oracle.score,
    };

    const feasibleHere: number[] = [];
    for (const c of cGrid) {
      const margin = adaptiveMargin({ sigmaStar, phi, c, h });
      const w = widenedThresholds(nominal, margin);
      const adaptive = evaluatePolicy(profile, realized, start, budget, w.thetaP, w.thetaR, decision, baselineEm);
      const rec = computeRecovery(adaptive.savings, naive.savings, perfect.savings);
      const recovery = adaptive.completed && naive.completed ? rec.clipped : 0;
      let recoveryVsOracle = 0;
      if (adaptive.completed && naive.completed && Number.isFinite(oracle.savings)) {
        const denom = oracle.savings - naive.savings;
        if (denom > RECOVERY_EPS_PP) {
          recoveryVsOracle = Math.min(1, Math.max(0, (adaptive.savings - naive.savings) / denom));
        }
      }
      if (adaptive.completed && adaptive.within_budget) feasibleHere.push(c);
      rows.push({
        region, h, c,
        theta_p: adaptive.theta_p, theta_r: adaptive.theta_r, margin: adaptive.theta_p - adaptive.theta_r,
        savings: adaptive.savings, overhead: adaptive.overhead, score: adaptive.score,
        num_pauses: adaptive.num_pauses, completed: adaptive.completed, within_budget: adaptive.within_budget,
        savings_naive: naive.savings, completed_naive: naive.completed,
        savings_perfect: perfect.savings, savings_oracle: oracle.savings,
        recovery_raw: rec.raw, recovery, recovery_vs_oracle: recoveryVsOracle,
      });
    }
    feasibleC[h] = feasibleHere;
  }

  return {
    region, model, year, budget, start, seed,
    calibration: { sigmaStar, phi },
    nominal: { thetaP: nominal.thetaP, thetaR: nominal.thetaR, margin: nominal.thetaP - nominal.thetaR, source: "reopt_summary.json baseline" },
    naivePolicy: { thetaP: naivePolicy.thetaP, thetaR: naivePolicy.thetaR, start },
    s0_naive_ff: perfect.savings,
    s0_adaptive_ff: adaptivePerfect.savings,
    cGrid: [...cGrid],
    horizons: [...horizons],
    oracles,
    feasibleC,
    rows,
  };
}

async function adaptiveSensitivityCli(raw: {
  model?: string;
  regions?: string;
  year?: string;
  horizons?: string;
  cSensitivity?: string;
  budget?: string;
  resolution?: string;
  iterations?: string;
  ckptPause?: string;
  ckptResume?: string;
  calibrationDir?: string;
  output?: string;
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
  const horizons = (raw.horizons ?? "1,3,6,12,24,72").split(",").map((s) => Number(s.trim()));
  const cGrid = (raw.cSensitivity ?? "0,0.25,0.5,0.75,1,1.5,2,3,4,6,8").split(",").map((s) => Number(s.trim()));
  const budget = Number(raw.budget ?? "200");
  const resolution = Number(raw.resolution ?? "10");
  const iterations = Number(raw.iterations ?? "6");
  const calibrationDir = raw.calibrationDir ?? "publication/output/forecast";
  const outputPrefix = raw.output ?? "publication/output/forecast/adaptive_sensitivity";
  const quiet = raw.quiet ?? false;
  const seed = 1;

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
    checkpointPauseTime: raw.ckptPause != null ? parseFloat(raw.ckptPause) : constants.checkpoint_pause_time,
    checkpointResumeTime: raw.ckptResume != null ? parseFloat(raw.ckptResume) : constants.checkpoint_resume_time,
  };

  mkdirSync(dirname(resolve(outputPrefix)), { recursive: true });

  const reoptSummary = JSON.parse(readFileSync(resolve(calibrationDir, "reopt_summary.json"), "utf-8")) as ReoptSummaryRegion[];
  const nominalByRegion = Object.fromEntries(reoptSummary.map((r) => [r.region, r.baseline]));

  if (!quiet) {
    console.log(`\n  TheGreenEpoch Forecast Sweep \u2500 adaptive recovery-vs-c sensitivity (Phase B.1)`);
    console.log(`  Model: ${model}, Regions: ${regions.join(",")}, Test year: ${year}`);
    console.log(`  Budget: ${budget}%, optimizer ${resolution}x${iterations}, horizons: ${horizons.join(",")}`);
    console.log(`  c-grid: ${cGrid.join(",")}; deterministic (no RNG), seed kept for API parity`);
  }

  const summaryRegions: Array<Record<string, unknown>> = [];

  for (const region of regions) {
    const calibration = loadCalibration(region, calibrationDir);
    const nominalBase = nominalByRegion[region];
    if (!nominalBase) {
      console.error(`  No reopt baseline for region ${region} in reopt_summary.json`);
      process.exit(1);
    }
    const nominal = { thetaP: nominalBase.thetaP, thetaR: nominalBase.thetaR };
    const naivePolicy = DEFAULT_POLICIES[region];
    if (!naivePolicy) {
      console.error(`  No default policy for region ${region}`);
      process.exit(1);
    }
    const tpMax = region === "SE" ? 100 : 800;

    const reoptRegion = JSON.parse(readFileSync(resolve(calibrationDir, `reopt_${region}.json`), "utf-8")) as ReoptRegionResult;
    const delayOracle = new Map<number, { thetaP: number; thetaR: number; savings: number; overhead: number; score: number }>();
    for (const cfg of reoptRegion.configs) {
      if (cfg.family === "delay" && cfg.best) {
        delayOracle.set(cfg.param_value, {
          thetaP: cfg.best.thetaP,
          thetaR: cfg.best.thetaR,
          savings: cfg.best.savings,
          overhead: cfg.best.overhead,
          score: cfg.best.score,
        });
      }
    }

    const result = runAdaptiveSensitivityRegion({
      region, model, year, profile: fullProfile, calibration, nominal, naivePolicy,
      horizons, cGrid, budget, delayOracle,
      optimizer: { resolution, iterations, tpMax },
      seed,
    });

    writeFileSync(`${outputPrefix}_${region}.json`, JSON.stringify(result, null, 2) + "\n", "utf-8");

    summaryRegions.push({
      region: result.region,
      year: result.year,
      budget: result.budget,
      s0_naive_ff: result.s0_naive_ff,
      s0_adaptive_ff: result.s0_adaptive_ff,
      nominal: result.nominal,
      naivePolicy: result.naivePolicy,
      cGrid: result.cGrid,
      horizons: result.horizons,
      oracles: result.oracles,
      feasibleC: result.feasibleC,
      rows: result.rows.map((r) => ({
        h: r.h, c: r.c,
        savings: r.savings, overhead: r.overhead, score: r.score,
        num_pauses: r.num_pauses, completed: r.completed, within_budget: r.within_budget,
        savings_naive: r.savings_naive, savings_perfect: r.savings_perfect, savings_oracle: r.savings_oracle,
        recovery: r.recovery, recovery_raw: r.recovery_raw, recovery_vs_oracle: r.recovery_vs_oracle,
      })),
    });

    if (!quiet) {
      console.log(`  [${region}] \u03C3*=${calibration.sigmaStar.toFixed(3)} \u03C6=${calibration.orders["1"].coeffs.ar[0].toFixed(6)}`);
      console.log(`    feasible c by h: ${result.horizons.map((h) => `${h}:[${result.feasibleC[h].join(",")}]`).join("  ")}`);
      const money = result.rows.find((r) => r.h === 72 && r.c === 6);
      if (money) {
        console.log(
          `    h=72 c=6 money cell (${region}): S=${money.savings.toFixed(2)} completed=${money.completed} within_budget=${money.within_budget} recVsOracle=${money.recovery_vs_oracle.toFixed(3)}`,
        );
      }
      console.log(`  JSON: ${outputPrefix}_${region}.json`);
    }
  }

  const summaryDoc = {
    spec: "Phase B.1 stale-aware adaptive controller — recovery-vs-c sensitivity (test year)",
    generatedDate: new Date().toISOString().slice(0, 10),
    deterministic: true,
    methodology: {
      decisionModel:
        "Same decision model as --mode adaptive: decide-on-forecast / pay-on-realized, decision timeline = AR(1) h-step-ahead forecast (applyForecast {type:arma, order:1, horizon:h, coeffs}); emissions evaluated on the realized 2025 timeline.",
      adaptiveRule:
        "margin(h) = c*sigmaStar*sqrt((1-phi^(2h))/(1-phi^2)); thresholds widened symmetrically around the reopt nominal midpoint. c is swept over the extended grid on the test year (no c-selection).",
      staticOracleCeiling:
        "per-h reoptimized static thresholds under the arma(h) timeline (runOptimization, reopt settings, one seed). h in {1,6} sourced from committed reopt_{region}.json delay rows (delay(h) == arma(h) exactly to 6 decimals for near-unit-root phi); h in {3,12,24,72} freshly optimized. This is the ceiling: no static-threshold policy can exceed it.",
      recovery:
        "recovery = clip((S_adaptive - S_naive)/(S0_FF - S_naive), 0, 1); recovery_raw is the unclipped value. Completion guard (identical to --mode adaptive): recovery = 0 whenever the adaptive or naive run does not complete (an incomplete, budget-exhausted run's savings is inflated).",
      recoveryVsOracle:
        "recovery_vs_oracle = clip((S_adaptive - S_naive)/(S_oracle - S_naive), 0, 1): the fraction of the recoverable (naive -> static oracle) gap closed. Headline controller metric. 0 when naive >= oracle or runs incomplete.",
      feasibleC:
        "feasibleC[h] = c values whose run completes AND stays within the 200% budget. All (region, c, h) cells are retained in rows, including budget-infeasible ones, so the feasible envelope is directly readable.",
    },
    cGrid,
    horizons,
    regions: summaryRegions,
  };
  writeFileSync(`${outputPrefix}_summary.json`, JSON.stringify(summaryDoc, null, 2) + "\n", "utf-8");

  if (!quiet) {
    if (regions.length > 1) console.log(`  Summary: ${outputPrefix}_summary.json`);
    console.log("  Done.\n");
  }
}
