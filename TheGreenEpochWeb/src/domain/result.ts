import type { FullProfile, SimConfig, SimProgress, SimResult } from "./types";
import { tokensPerSecond } from "./physics";

export function computeOverheadPct(
  pausedS: number,
  checkpointS: number,
  tokensTotal: number,
  tps: number,
): number {
  const idealS = tokensTotal / (tps || 1);
  return 100 * (pausedS + checkpointS) / (idealS || 1);
}

export function computeSavingsPct(
  totalEmissionsKgco2: number,
  baselineEmissionsKgco2: number,
): number {
  return baselineEmissionsKgco2 > 0
    ? (baselineEmissionsKgco2 - totalEmissionsKgco2) / baselineEmissionsKgco2 * 100
    : 0;
}

export function computeScore(savingsPct: number, overheadPct: number): number {
  return savingsPct / Math.max(overheadPct, 0.001);
}

export function computeIsOk(
  tokensRemaining: number,
  actualOverheadPct: number,
  budgetPct: number,
  issues: string[],
): boolean {
  return tokensRemaining <= 0
    && actualOverheadPct <= budgetPct
    && issues.length === 0;
}

export function buildSimResult(
  profile: FullProfile,
  simConfig: SimConfig,
  lastProgress: SimProgress,
  baselineProgress: SimProgress,
  thetaPause: number,
  thetaResume: number,
  overrides: {
    id: string;
    scenarioDescription: string;
    model: string;
    region: string;
    timestamps: string[];
    carbonIntensitySeries: number[];
    stateSeries: string[];
    emissionsSeries: number[];
    tokensRemainingSeries: number[];
  },
): SimResult {
  const tps = tokensPerSecond(profile.gpuCount) || 1;
  const overheadS = lastProgress.pausedS + lastProgress.checkpointS;
  const actualOverheadPct = computeOverheadPct(lastProgress.pausedS, lastProgress.checkpointS, lastProgress.tokensTotal, tps);
  const totalEm = lastProgress.totalEmissionsG / 1000;
  const baselineEm = baselineProgress.totalEmissionsG / 1000;
  const co2SavingsPct = computeSavingsPct(totalEm, baselineEm);
  const score = computeScore(co2SavingsPct, actualOverheadPct);
  const tokensProcessed = lastProgress.tokensTotal - lastProgress.tokensRemaining;
  const pausedTimeH = lastProgress.pausedS / 3600;
  const checkpointOverheadH = lastProgress.checkpointS / 3600;

  return {
    id: overrides.id,
    scenarioDescription: overrides.scenarioDescription,
    model: overrides.model,
    region: overrides.region,
    historicalYears: simConfig.historicalYears,
    startTime: simConfig.startTime,
    threshold: thetaPause === Infinity ? 9999 : thetaPause,
    hysteresisMargin: thetaResume,
    totalWallTimeH: lastProgress.totalWallS / 3600,
    trainingTimeH: lastProgress.trainingS / 3600,
    pausedTimeH,
    checkpointOverheadH,
    totalEnergyKwh: lastProgress.totalEnergyWh / 1000,
    trainingEnergyKwh: lastProgress.trainingEnergyWh / 1000,
    pausedEnergyKwh: lastProgress.pausedEnergyWh / 1000,
    checkpointEnergyKwh: lastProgress.checkpointEnergyWh / 1000,
    totalEmissionsKgco2: totalEm,
    tokensProcessed,
    tokensTotal: lastProgress.tokensTotal,
    completed: lastProgress.tokensRemaining <= 0,
    numPauses: lastProgress.numPauses,
    overheadBudgetPct: simConfig.overheadBudgetPct,
    actualOverheadPct: round(actualOverheadPct),
    withinOverheadBudget: overheadS / (lastProgress.tokensTotal / tps || 1) <= simConfig.overheadBudgetPct / 100,
    timestamps: overrides.timestamps,
    carbonIntensitySeries: overrides.carbonIntensitySeries,
    stateSeries: overrides.stateSeries,
    emissionsSeries: overrides.emissionsSeries,
    tokensRemainingSeries: overrides.tokensRemainingSeries,
    issues: lastProgress.issues,
    stopReason: lastProgress.stopReason,
    baselineEmissionsKgco2: baselineEm,
    baselineTimeH: baselineProgress.totalWallS / 3600,
    co2SavingsPct: round(co2SavingsPct),
    score: round(score),
    idleTimeH: computeIdleTimeH(pausedTimeH, checkpointOverheadH),
    completionPct: computeCompletionPct(tokensProcessed, lastProgress.tokensTotal),
    ok: computeIsOk(lastProgress.tokensRemaining, actualOverheadPct, simConfig.overheadBudgetPct, lastProgress.issues),
  };
}

function computeIdleTimeH(pausedH: number, checkpointH: number): number {
  return pausedH + checkpointH;
}

function computeCompletionPct(tokensProcessed: number, tokensTotal: number): number {
  return tokensTotal > 0 ? 100 * tokensProcessed / tokensTotal : 0;
}

function round(n: number): number {
  return Math.round(n * 100) / 100;
}
