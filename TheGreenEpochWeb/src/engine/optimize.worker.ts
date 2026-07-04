import type { FullProfile, CO2Timeline, Scenario, SweepPoint } from "../domain/types";
import { runOptimization } from "../domain/optimize";
import type { AdaptiveOptions } from "../domain/optimize";

interface StartMessage {
  type: "start";
  profile: FullProfile;
  timeline: CO2Timeline;
  scenario: Scenario;
  options: AdaptiveOptions;
}

self.onmessage = (e: MessageEvent<StartMessage>) => {
  if (e.data.type !== "start") return;

  const { profile, timeline, scenario, options } = e.data;

  const result = runOptimization(profile, timeline, scenario, options, (iteration, points, best) => {
    (self as any).postMessage({ type: "iteration", iteration, points, best });
  });

  (self as any).postMessage({ type: "done", points: result.points, best: result.best });
};
