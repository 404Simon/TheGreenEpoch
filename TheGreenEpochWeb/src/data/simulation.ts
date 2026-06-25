import type { FullProfile, SimConfig, SimProgress, CO2Timeline } from "../types";
import { SimState } from "../types";

const GPU_HOURS_PER_TRILLION_TOKENS = 180_000;

export function tokensPerSecond(gpuCount: number): number {
  const perGpu = 1e12 / (GPU_HOURS_PER_TRILLION_TOKENS * 3600);
  return perGpu * gpuCount;
}

export function energyWh(powerW: number, durationS: number): number {
  return powerW * durationS / 3600;
}

export function emissionsG(energyWh_: number, co2IntensityGPerKwh: number): number {
  return energyWh_ / 1000 * co2IntensityGPerKwh;
}

export function findStartIndex(timestamps: string[], startTime: string): number {
  const baseYear = new Date(timestamps[0]).getUTCFullYear();
  const [month, day] = startTime.split("-").map(Number);
  let targetStr: string;
  try {
    const d = new Date(Date.UTC(baseYear, month - 1, day));
    targetStr = d.toISOString();
  } catch {
    const d = new Date(Date.UTC(baseYear, month - 1, 28));
    targetStr = d.toISOString();
  }
  let lo = 0, hi = timestamps.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (timestamps[mid] < targetStr) lo = mid + 1;
    else hi = mid;
  }
  return Math.min(lo, timestamps.length - 1);
}

export function* simulateStepwise(
  profile: FullProfile,
  config: SimConfig,
  timeline: CO2Timeline,
): Generator<SimProgress> {
  const { timestamps, carbonIntensity: carbon } = timeline;
  if (timestamps.length < 2) throw new Error("Insufficient grid data");

  const tps = tokensPerSecond(profile.gpuCount);
  const trainPowerW = profile.gpuCount * profile.gpuPowerTrain * profile.pue;
  const pausePowerW = profile.gpuCount * profile.gpuPowerPause * profile.pue;
  const ckptPowerW = trainPowerW;

  const meanCo2 = carbon.reduce((a, b) => a + b, 0) / carbon.length;

  const diffMs = new Date(timestamps[1]).getTime() - new Date(timestamps[0]).getTime();
  const stepS = diffMs / 1000;
  if (!stepS || stepS <= 0) throw new Error("Invalid step size");

  const startIdx = findStartIndex(timestamps, config.startTime);
  const baseYear = new Date(timestamps[0]).getUTCFullYear();
  const [sm, sd] = config.startTime.split("-").map(Number);
  const startDate = new Date(Date.UTC(baseYear, sm - 1, sd));

  let idx = startIdx;
  const nPoints = timestamps.length;

  const policy = {
    thetaPause: config.thetaPause,
    thetaResume: config.thetaResume,
    evaluate(co2: number, isPaused: boolean): "pause" | "resume" | "continue" {
      if (isPaused) return co2 < this.thetaResume ? "resume" : "continue";
      return co2 > this.thetaPause ? "pause" : "continue";
    },
  };

  let state: SimState = SimState.RUNNING;
  let transitionTimerS = 0;
  let targetAfterTransition: SimState | null = null;

  const tokensTotal = profile.datasetTokens;
  let tokensRemaining = tokensTotal;
  const idealTrainingS = tokensTotal / (tps || 1);

  let totalWallS = 0;
  let trainingS = 0;
  let pausedS = 0;
  let checkpointS = 0;
  let totalEnergyWh_ = 0;
  let trainingEnergyWh_ = 0;
  let pausedEnergyWh_ = 0;
  let checkpointEnergyWh_ = 0;
  let totalEmissionsG_ = 0;
  let numPauses = 0;

  const maxIterations = 10_000_000;
  let iterations = 0;
  const issues: string[] = [];
  let nanFallbacks = 0;

  function getCo2(i: number): number {
    const v = carbon[i];
    if (!isFinite(v)) {
      nanFallbacks++;
      return meanCo2;
    }
    return v;
  }

  let initCo2 = getCo2(startIdx);
  if (initCo2 > config.thetaPause) {
    const ckptS = profile.checkpointPauseTime;
    if (ckptS > 0) {
      transitionTimerS = ckptS;
      targetAfterTransition = SimState.PAUSED;
    } else {
      state = SimState.PAUSED;
    }
    numPauses++;
  }

  let budgetExceeded = false;

  function maybeYield(done?: boolean, stopReason?: string): SimProgress {
    return {
      timestamp: timestamps[idx] || startDate.toISOString(),
      carbonIntensity: getCo2(idx),
      state,
      tokensRemaining,
      tokensTotal,
      totalWallS: round2(totalWallS),
      trainingS: round2(trainingS),
      pausedS: round2(pausedS),
      checkpointS: round2(checkpointS),
      totalEnergyWh: round2(totalEnergyWh_),
      trainingEnergyWh: round2(trainingEnergyWh_),
      pausedEnergyWh: round2(pausedEnergyWh_),
      checkpointEnergyWh: round2(checkpointEnergyWh_),
      totalEmissionsG: round2(totalEmissionsG_),
      numPauses,
      done: done || false,
      stopReason: stopReason || "",
      issues: [...issues],
      nanFallbacks,
    };
  }

  while (tokensRemaining > 0 && !budgetExceeded) {
    iterations++;
    if (iterations > maxIterations) {
      issues.push(`Iteration limit (${maxIterations}) reached`);
      break;
    }

    const co2 = getCo2(idx);
    let dtS = stepS;
    const curTs = timestamps[idx];

    if (transitionTimerS > 0) {
      const spentS = Math.min(dtS, transitionTimerS);
      transitionTimerS -= spentS;
      const eWh = energyWh(ckptPowerW, spentS);
      const emG = emissionsG(eWh, co2);
      checkpointS += spentS;
      checkpointEnergyWh_ += eWh;
      totalEnergyWh_ += eWh;
      totalEmissionsG_ += emG;
      totalWallS += spentS;
      dtS -= spentS;
      if (transitionTimerS <= 0 && targetAfterTransition !== null) {
        state = targetAfterTransition;
        targetAfterTransition = null;
      }
      if (dtS <= 0) {
        idx = (idx + 1) % nPoints;
        yield maybeYield();
        continue;
      }
    }

    const action = policy.evaluate(co2, state === SimState.PAUSED);

    if (action === "pause" && state === SimState.RUNNING) {
      const ckptS = profile.checkpointPauseTime;
      if ((pausedS + checkpointS + ckptS) / idealTrainingS > config.overheadBudgetPct / 100) {
        issues.push(`Overhead would exceed budget - blocking pause`);
        budgetExceeded = true;
        break;
      }
      numPauses++;
      if (ckptS <= 0) {
        state = SimState.PAUSED;
      } else {
        transitionTimerS = ckptS;
        targetAfterTransition = SimState.PAUSED;
        yield maybeYield();
        continue;
      }
    }

    if (action === "resume" && state === SimState.PAUSED) {
      const ckptS = profile.checkpointResumeTime;
      if ((pausedS + checkpointS + ckptS) / idealTrainingS > config.overheadBudgetPct / 100) {
        issues.push(`Overhead would exceed budget - blocking resume`);
        budgetExceeded = true;
        break;
      }
      if (ckptS <= 0) {
        state = SimState.RUNNING;
      } else {
        transitionTimerS = ckptS;
        targetAfterTransition = SimState.RUNNING;
        yield maybeYield();
        continue;
      }
    }

    if (state === SimState.RUNNING) {
      const maxT = Math.floor(tps * dtS);
      const tokensStep = Math.min(maxT, tokensRemaining);
      const effectiveS = tokensStep >= maxT ? dtS : tokensStep / tps;
      tokensRemaining -= tokensStep;
      const eWh = energyWh(trainPowerW, effectiveS);
      const emG = emissionsG(eWh, co2);
      trainingS += effectiveS;
      trainingEnergyWh_ += eWh;
      totalEnergyWh_ += eWh;
      totalEmissionsG_ += emG;
      totalWallS += effectiveS;
      const idleS = dtS - effectiveS;
      if (idleS > 0) {
        const iWh = energyWh(pausePowerW, idleS);
        pausedS += idleS;
        pausedEnergyWh_ += iWh;
        totalEnergyWh_ += iWh;
        totalEmissionsG_ += emissionsG(iWh, co2);
        totalWallS += idleS;
      }
    } else {
      const eWh = energyWh(pausePowerW, dtS);
      const emG = emissionsG(eWh, co2);
      pausedS += dtS;
      pausedEnergyWh_ += eWh;
      totalEnergyWh_ += eWh;
      totalEmissionsG_ += emG;
      totalWallS += dtS;
    }

    if ((pausedS + checkpointS) / idealTrainingS > config.overheadBudgetPct / 100) {
      issues.push(`Overhead exceeded budget`);
      break;
    }

    idx = (idx + 1) % nPoints;
    yield maybeYield();
  }

  const stopReason = tokensRemaining <= 0
    ? "completed"
    : budgetExceeded || (pausedS + checkpointS) / idealTrainingS > config.overheadBudgetPct / 100
      ? "budget_exceeded"
      : "iteration_limit";

  yield maybeYield(true, stopReason);
}

export function buildResult(
  profile: FullProfile,
  config: SimConfig,
  lastProgress: SimProgress,
  baselineProgress: SimProgress,
): {
  totalWallTimeH: number;
  trainingTimeH: number;
  pausedTimeH: number;
  checkpointOverheadH: number;
  totalEnergyKwh: number;
  trainingEnergyKwh: number;
  pausedEnergyKwh: number;
  checkpointEnergyKwh: number;
  totalEmissionsKgco2: number;
  tokensProcessed: number;
  tokensTotal: number;
  completed: boolean;
  numPauses: number;
  actualOverheadPct: number;
  withinOverheadBudget: boolean;
  issues: string[];
  stopReason: string;
  baselineEmissionsKgco2: number;
  baselineTimeH: number;
} {
  const overheadS = lastProgress.pausedS + lastProgress.checkpointS;
  const tps = tokensPerSecond(profile.gpuCount);
  const idealS = lastProgress.tokensTotal / (tps || 1);
  const actualOverheadPct = 100 * overheadS / (idealS || 1);

  return {
    totalWallTimeH: lastProgress.totalWallS / 3600,
    trainingTimeH: lastProgress.trainingS / 3600,
    pausedTimeH: lastProgress.pausedS / 3600,
    checkpointOverheadH: lastProgress.checkpointS / 3600,
    totalEnergyKwh: lastProgress.totalEnergyWh / 1000,
    trainingEnergyKwh: lastProgress.trainingEnergyWh / 1000,
    pausedEnergyKwh: lastProgress.pausedEnergyWh / 1000,
    checkpointEnergyKwh: lastProgress.checkpointEnergyWh / 1000,
    totalEmissionsKgco2: lastProgress.totalEmissionsG / 1000,
    tokensProcessed: lastProgress.tokensTotal - lastProgress.tokensRemaining,
    tokensTotal: lastProgress.tokensTotal,
    completed: lastProgress.done && lastProgress.tokensRemaining <= 0,
    numPauses: lastProgress.numPauses,
    actualOverheadPct: actualOverheadPct,
    withinOverheadBudget: overheadS / idealS <= config.overheadBudgetPct / 100,
    issues: lastProgress.issues,
    stopReason: lastProgress.stopReason,
    baselineEmissionsKgco2: baselineProgress.totalEmissionsG / 1000,
    baselineTimeH: baselineProgress.totalWallS / 3600,
  };
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}
