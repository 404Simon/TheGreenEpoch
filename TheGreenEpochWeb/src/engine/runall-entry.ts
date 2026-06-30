import type { Constants, TrainingProfile, CO2Timeline, Scenario, FullProfile, SimConfig, SimResult } from "../domain/types";
import { simulateStepwise } from "../domain/simulation";
import { neverPausePolicy, hysteresisPolicy } from "../domain/policy";
import { buildSimResult } from "../domain/result";

interface StartMessage {
  type: "start";
  constants: Constants;
  profiles: Record<string, TrainingProfile>;
  co2Cache: Record<string, CO2Timeline>;
  scenarios: Scenario[];
}

function fullProfile(profile: TrainingProfile, c: Constants): FullProfile {
  return {
    ...profile,
    gpuPowerTrain: c.gpu_power_train,
    gpuPowerPause: c.gpu_power_pause,
    pue: c.pue,
    checkpointPauseTime: c.checkpoint_pause_time,
    checkpointResumeTime: c.checkpoint_resume_time,
  };
}

function simConfig(scenario: Scenario, startTime: string, overheadBudgetPct: number): SimConfig {
  return { startTime, historicalYears: scenario.historicalYears, overheadBudgetPct };
}

function runOnce(
  profile: FullProfile,
  timeline: CO2Timeline,
  config: SimConfig,
  thetaPause: number,
  thetaResume: number,
  scenario: Scenario,
): SimResult {
  const baseline: import("../domain/types").SimProgress[] = [];
  for (const p of simulateStepwise(profile, neverPausePolicy(), timeline, config)) {
    baseline.push(p);
  }
  const baselineLast = baseline[baseline.length - 1];

  const timestamps: string[] = [];
  const carbonIntensitySeries: number[] = [];
  const stateSeries: string[] = [];
  const emissionsSeries: number[] = [];
  const tokensRemainingSeries: number[] = [];

  const sim: import("../domain/types").SimProgress[] = [];
  for (const p of simulateStepwise(profile, hysteresisPolicy(thetaPause, thetaResume), timeline, config)) {
    sim.push(p);
    timestamps.push(p.timestamp);
    carbonIntensitySeries.push(p.carbonIntensity);
    stateSeries.push(p.state);
    emissionsSeries.push(p.totalEmissionsG / 1000);
    tokensRemainingSeries.push(p.tokensRemaining);
  }
  const last = sim[sim.length - 1];

  return buildSimResult(profile, config, last, baselineLast, thetaPause, thetaResume, {
    id: `result-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    scenarioDescription: scenario.description,
    model: profile.name,
    region: scenario.region,
    timestamps,
    carbonIntensitySeries,
    stateSeries,
    emissionsSeries,
    tokensRemainingSeries,
  });
}

self.onmessage = (e: MessageEvent<StartMessage>) => {
  if (e.data.type !== "start") return;
  const { constants, profiles, co2Cache, scenarios } = e.data;

  const totalRuns = scenarios.reduce((s, sc) => s + sc.thresholds.length * sc.startTimes.length, 0);
  let done = 0;

  for (const scenario of scenarios) {
    const profile = profiles[scenario.model];
    if (!profile) {
      done += scenario.thresholds.length * scenario.startTimes.length;
      (self as any).postMessage({ type: "progress", done, total: totalRuns });
      continue;
    }
    const fp = fullProfile(profile, constants);
    const timeline = co2Cache[scenario.region];
    if (!timeline) {
      done += scenario.thresholds.length * scenario.startTimes.length;
      (self as any).postMessage({ type: "progress", done, total: totalRuns });
      continue;
    }

    for (let ti = 0; ti < scenario.thresholds.length; ti++) {
      for (let si = 0; si < scenario.startTimes.length; si++) {
        const startTime = scenario.startTimes[si];
        const thetaPause = scenario.thresholds[ti];
        const thetaResume = scenario.hysteresis[ti];
        const config = simConfig(scenario, startTime, scenario.overheadBudgetPct);

        const result = runOnce(fp, timeline, config, thetaPause, thetaResume, scenario);
        done++;

        (self as any).postMessage({ type: "result", result, done, total: totalRuns });
      }
    }
  }

  (self as any).postMessage({ type: "done" });
};
