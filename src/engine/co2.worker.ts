import { averageYears } from "../data/co2-loader";
import type { YearCO2 } from "../domain/types";

interface LoadRequest {
  type: "load";
  requestId: string;
  baseUrl: string;
  zone: string;
  years: number[];
}

self.onmessage = async (e: MessageEvent<LoadRequest>) => {
  if (e.data.type !== "load") return;
  const { requestId, baseUrl, zone, years } = e.data;

  try {
    const entries = await Promise.all(
      years.map(async (y) => {
        const r = await fetch(`${baseUrl}/co2/${zone}_${y}.json`);
        return r.json() as Promise<YearCO2>;
      }),
    );
    const timeline = averageYears(entries);
    (self as any).postMessage({ type: "loaded", requestId, timeline });
  } catch (err) {
    (self as any).postMessage({ type: "error", requestId, error: String(err) });
  }
};
