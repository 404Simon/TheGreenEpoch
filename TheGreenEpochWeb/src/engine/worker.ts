import type { FullProfile, CO2Timeline, Scenario, SweepPoint } from "../domain/types";

export interface AdaptiveOptions {
  thetaPauseMax: number;
  overheadBudgetPct: number;
  resolution: number;
  maxIterations: number;
  minStep: number;
  shrinkFactor: number;
  alpha: number;
}

/**
 * Deep-clone data before sending to a Worker.
 *
 * Solid's `createStore` wraps objects in Proxy, and the Structured Clone
 * algorithm used by `postMessage` cannot clone Proxy objects.  A
 * JSON round-trip strips all proxies and produces plain, cloneable data.
 *
 * The CO2 timeline contains ~105k entries — this runs in <5ms.
 */
function stripProxies<T>(data: T): T {
  return JSON.parse(JSON.stringify(data));
}

export function runOptimizationInWorker(
  profile: FullProfile,
  timeline: CO2Timeline,
  scenario: Scenario,
  options: AdaptiveOptions,
  startTimeIdx: number,
  onIteration?: (iteration: number, points: SweepPoint[], best: SweepPoint | null) => void,
): Promise<{ points: SweepPoint[]; best: SweepPoint | null }> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./worker-entry", import.meta.url), { type: "module" });

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
      scenario: stripProxies(scenario),
      options,
      startTimeIdx,
    });
  });
}
