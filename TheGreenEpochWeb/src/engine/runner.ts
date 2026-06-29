import type { FullProfile, SimConfig, CO2Timeline, SimProgress, SimResult, Policy } from "../domain/types";
import { simulateStepwise } from "../domain/simulation";
import { buildSimResult } from "../domain/result";
import { neverPausePolicy } from "../domain/policy";

export interface RunOptions {
  onProgress?: (p: SimProgress) => void;
}

export function runSimulation(
  profile: FullProfile,
  policy: Policy,
  timeline: CO2Timeline,
  simConfig: SimConfig & { scenarioDescription: string; model: string; region: string },
  thetaPause: number,
  thetaResume: number,
  options?: RunOptions,
): SimResult {
  const baselinePolicy = neverPausePolicy();

  const baselineProgress: SimProgress[] = [];
  for (const p of simulateStepwise(profile, baselinePolicy, timeline, simConfig)) {
    baselineProgress.push(p);
  }
  const baselineLast = baselineProgress[baselineProgress.length - 1];

  const tsSeries: string[] = [];
  const co2Series: number[] = [];
  const stateSeries: string[] = [];
  const emissionsSeries: number[] = [];
  const tokensRemainingSeries: number[] = [];

  let lastProgress: SimProgress | null = null;
  for (const p of simulateStepwise(profile, policy, timeline, simConfig)) {
    lastProgress = p;
    if (options?.onProgress) options.onProgress(p);
    tsSeries.push(p.timestamp);
    co2Series.push(p.carbonIntensity);
    stateSeries.push(p.state);
    emissionsSeries.push(p.totalEmissionsG / 1000);
    tokensRemainingSeries.push(p.tokensRemaining);
  }

  return buildSimResult(
    profile, simConfig, lastProgress!, baselineLast,
    thetaPause, thetaResume,
    {
      id: `result-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      scenarioDescription: simConfig.scenarioDescription,
      model: simConfig.model,
      region: simConfig.region,
      timestamps: tsSeries,
      carbonIntensitySeries: co2Series,
      stateSeries,
      emissionsSeries,
      tokensRemainingSeries,
    },
  );
}
