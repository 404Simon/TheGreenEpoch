import type { Constants, TrainingProfile, CO2Timeline, Scenario, FullProfile, SimConfig } from "../domain/types";
import { runSimulation } from "./simulate";
import { hysteresisPolicy } from "../domain/policy";

interface StartMessage {
  type: "start";
  constants: Constants;
  profiles: Record<string, TrainingProfile>;
  co2Cache: Record<string, CO2Timeline>;
  scenarios: Scenario[];
  alpha: number;
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

function simConfig(scenario: Scenario, startTime: string): SimConfig {
  return { startTime, historicalYears: scenario.historicalYears, overheadBudgetPct: scenario.overheadBudgetPct };
}

self.onmessage = (e: MessageEvent<StartMessage>) => {
  if (e.data.type !== "start") return;
  const { constants, profiles, co2Cache, scenarios, alpha } = e.data;

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
        const config = simConfig(scenario, startTime);

        const result = runSimulation(fp, hysteresisPolicy(thetaPause, thetaResume), timeline, config, thetaPause, thetaResume, {
          scenarioDescription: scenario.description,
          model: fp.name,
          region: scenario.region,
          alpha,
        });
        done++;

        (self as any).postMessage({ type: "result", result, done, total: totalRuns });
      }
    }
  }

  (self as any).postMessage({ type: "done" });
};
