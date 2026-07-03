import type { CO2Timeline } from "../domain/types";

function stripProxies<T>(data: T): T {
  return JSON.parse(JSON.stringify(data));
}

let nextId = 0;
const pending = new Map<string, { resolve: (t: CO2Timeline) => void; reject: (e: Error) => void }>();
let worker: Worker | null = null;

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./co2.worker", import.meta.url), { type: "module" });
    worker.onmessage = (e) => {
      const msg = e.data;
      const p = pending.get(msg.requestId);
      if (!p) return;
      pending.delete(msg.requestId);
      if (msg.type === "loaded") {
        p.resolve(msg.timeline);
      } else {
        p.reject(new Error(msg.error));
      }
    };
    worker.onerror = () => {
      for (const [, p] of pending) p.reject(new Error("CO2 worker error"));
      pending.clear();
    };
  }
  return worker;
}

export function loadCO2Timeline(baseUrl: string, zone: string, years: number[]): Promise<CO2Timeline> {
  const requestId = `co2-${nextId++}`;
  return new Promise((resolve, reject) => {
    pending.set(requestId, { resolve, reject });
    getWorker().postMessage({ type: "load", requestId, baseUrl, zone, years: stripProxies(years) });
  });
}
