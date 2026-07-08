import type { FullProfile, CO2Timeline, SweepPoint } from "../domain/types";
import type { AdaptiveOptions } from "../domain/optimize";
import { stripProxies } from "../domain/utils";

export function runOptimizationInWorker(
  profile: FullProfile,
  timeline: CO2Timeline,
  historicalYears: number[],
  options: AdaptiveOptions,
  onIteration?: (iteration: number, points: SweepPoint[], best: SweepPoint | null) => void,
): Promise<{ points: SweepPoint[]; best: SweepPoint | null }> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./optimize.worker", import.meta.url), { type: "module" });

    worker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "iteration") {
        onIteration?.(msg.iteration, msg.points, msg.best);
      } else if (msg.type === "done") {
        worker.terminate();
        resolve({ points: msg.points, best: msg.best });
      }
    };

    worker.onerror = (err) => {
      worker.terminate();
      reject(err);
    };

    worker.postMessage({
      type: "start",
      profile: stripProxies(profile),
      timeline: stripProxies(timeline),
      historicalYears,
      options,
    });
  });
}
