import type { FullProfile, CO2Timeline, SweepPoint } from "../domain/types";
import { runOptimization } from "../domain/optimize";
import type { AdaptiveOptions } from "../domain/optimize";

interface StartMessage {
  type: "start";
  profile: FullProfile;
  timeline: CO2Timeline;
  historicalYears: number[];
  options: AdaptiveOptions;
}

self.onmessage = (e: MessageEvent<StartMessage>) => {
  if (e.data.type !== "start") return;

  const { profile, timeline, historicalYears, options } = e.data;

  const result = runOptimization(profile, timeline, historicalYears, options, (iteration, points, best) => {
    (self as any).postMessage({ type: "iteration", iteration, points, best });
  });

  (self as any).postMessage({ type: "done", points: result.points, best: result.best });
};
