import type { FullProfile, SimConfig, CO2Timeline, SimProgress, SimResult, Policy } from "../domain/types";
import { simulateStepwise } from "../domain/simulation";
import { buildSimResult } from "../domain/result";
import { neverPausePolicy } from "../domain/policy";

export interface RunOptions {
  onProgress?: (p: SimProgress) => void;
  alpha?: number;
  scenarioDescription?: string;
  model?: string;
  region?: string;
}

export function runBaseline(
  profile: FullProfile,
  timeline: CO2Timeline,
  simConfig: SimConfig,
): SimProgress {
  let last: SimProgress | null = null;
  for (const p of simulateStepwise(profile, neverPausePolicy(), timeline, simConfig)) {
    last = p;
  }
  return last!;
}

export function runSimulation(
  profile: FullProfile,
  policy: Policy,
  timeline: CO2Timeline,
  simConfig: SimConfig,
  thetaPause: number,
  thetaResume: number,
  options: RunOptions = {},
): SimResult {
  const baselineLast = runBaseline(profile, timeline, simConfig);

  const tsSeries: string[] = [];
  const co2Series: number[] = [];
  const stateSeries: string[] = [];
  const emissionsSeries: number[] = [];
  const tokensRemainingSeries: number[] = [];

  let lastProgress: SimProgress | null = null;
  for (const p of simulateStepwise(profile, policy, timeline, simConfig)) {
    lastProgress = p;
    options.onProgress?.(p);
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
      scenarioDescription: options.scenarioDescription ?? "",
      model: options.model ?? profile.name,
      region: options.region ?? "",
      timestamps: tsSeries,
      carbonIntensitySeries: co2Series,
      stateSeries,
      emissionsSeries,
      tokensRemainingSeries,
    },
    options.alpha ?? 1,
  );
}
