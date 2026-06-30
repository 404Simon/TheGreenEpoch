import type { Constants, TrainingProfile, CO2Timeline, Scenario, SimResult } from "../domain/types";

function stripProxies<T>(data: T): T {
  return JSON.parse(JSON.stringify(data));
}

export function runAllInWorker(
  constants: Constants,
  profiles: Record<string, TrainingProfile>,
  co2Cache: Record<string, CO2Timeline>,
  scenarios: Scenario[],
  onResult: (result: SimResult, done: number, total: number) => void,
  alpha: number = 1,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./runall-entry", import.meta.url), { type: "module" });

    worker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "result") {
        onResult(msg.result, msg.done, msg.total);
      } else if (msg.type === "done") {
        worker.terminate();
        resolve();
      } else if (msg.type === "progress") {
        onResult(null as any, msg.done, msg.total);
      }
    };

    worker.onerror = (err) => {
      worker.terminate();
      reject(err);
    };

    worker.postMessage({
      type: "start",
      constants: stripProxies(constants),
      profiles: stripProxies(profiles),
      co2Cache: stripProxies(co2Cache),
      scenarios: stripProxies(scenarios),
      alpha,
    });
  });
}
