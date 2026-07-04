import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(__dirname, "../../public/data/co2");

const API_URL = "https://api.electricitymaps.com/v3/carbon-intensity/past-range";

interface ApiEntry {
  datetime?: string;
  updatedAt?: string;
  carbonIntensity?: number;
  isEstimated?: boolean;
}

interface FetchOpts {
  zones: string[];
  years: number[];
  token: string;
  granularity: string;
  disableEstimations: boolean;
  maxRequestsPerMinute: number;
}

interface ApiResponse {
  history?: ApiEntry[];
  data?: ApiEntry[];
  datetime?: string;
  carbonIntensity?: number;
  isEstimated?: boolean;
}

function isoZ(d: Date): string {
  return d.toISOString();
}

function fmtDay(d: Date): string {
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

async function fetchDay(
  opts: FetchOpts,
  zone: string,
  start: Date,
  end: Date,
): Promise<ApiResponse> {
  const params = new URLSearchParams({
    zone,
    start: isoZ(start),
    end: isoZ(end),
    temporalGranularity: opts.granularity,
    disableEstimations: String(opts.disableEstimations),
  });

  const response = await fetch(`${API_URL}?${params}`, {
    headers: { "auth-token": opts.token },
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${zone} ${isoZ(start)}`);
  }

  return response.json() as Promise<ApiResponse>;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

interface YearAccum {
  timestamps: string[];
  carbonIntensity: number[];
  isEstimated: boolean[];
  dayCount: number;
}

async function fetchZoneYear(opts: FetchOpts, zone: string, year: number): Promise<YearAccum> {
  const startDate = new Date(Date.UTC(year, 0, 1));
  const endDate = new Date(Date.UTC(year, 11, 31));

  const rateLimitDelay = opts.maxRequestsPerMinute > 0
    ? Math.ceil(60_000 / opts.maxRequestsPerMinute)
    : 0;

  const accum: YearAccum = { timestamps: [], carbonIntensity: [], isEstimated: [], dayCount: 0 };
  const current = new Date(startDate);

  while (current <= endDate) {
    const nextDay = new Date(current);
    nextDay.setUTCDate(nextDay.getUTCDate() + 1);

    let retries = 0;
    // eslint-disable-next-line no-constant-condition
    while (true) {
      try {
        const payload = await fetchDay(opts, zone, current, nextDay);
        const entries = payload.history ?? payload.data ?? [];

        if (entries.length === 0 && payload.datetime) {
          accum.timestamps.push(payload.datetime);
          accum.carbonIntensity.push(payload.carbonIntensity ?? 0);
          accum.isEstimated.push(payload.isEstimated ?? false);
        } else {
          for (const e of entries) {
            accum.timestamps.push(e.datetime ?? "");
            accum.carbonIntensity.push(e.carbonIntensity ?? 0);
            accum.isEstimated.push(e.isEstimated ?? false);
          }
        }
        accum.dayCount++;
        break;
      } catch (err) {
        retries++;
        if (retries > 5) {
          console.error(`  ✗ ${zone} ${year} day ${fmtDay(current)} — failed after ${retries} retries`);
          throw err;
        }
        console.warn(`  ⚠ retry ${retries}/5 — ${zone} ${year} day ${fmtDay(current)}`);
        await sleep(Math.min(1000 * retries, 30_000));
      }
    }

    if (rateLimitDelay > 0) {
      await sleep(rateLimitDelay);
    }

    current.setUTCDate(current.getUTCDate() + 1);
  }

  return accum;
}

export async function fetchCarbon(raw: { zones: string; years: string; token: string; granularity: string; disableEstimations: boolean; maxRpm: string }): Promise<void> {
  const opts = {
    zones: raw.zones.split(",").filter(Boolean),
    years: raw.years.split(",").map(Number).filter((y) => !isNaN(y)),
    token: raw.token,
    granularity: raw.granularity,
    disableEstimations: raw.disableEstimations,
    maxRequestsPerMinute: parseInt(raw.maxRpm, 10),
  };

  mkdirSync(OUT_DIR, { recursive: true });
  console.log(`Output: ${OUT_DIR}`);
  console.log(`Zones:  ${opts.zones.join(", ")}`);
  console.log(`Years:  ${opts.years.join(", ")}`);
  console.log();

  for (const zone of opts.zones) {
    for (const year of opts.years) {
      const filePath = resolve(OUT_DIR, `${zone}_${year}.json`);

      if (existsSync(filePath)) {
        console.log(`  · ${zone} ${year} — exists, skipping`);
        continue;
      }

      console.log(`  · ${zone} ${year} — fetching...`);
      try {
        const accum = await fetchZoneYear(opts, zone, year);
        writeFileSync(filePath, JSON.stringify({ zone, year, ...accum }, null, 2));
        console.log(`    ✓ ${accum.dayCount} days, ${accum.timestamps.length} data points`);
      } catch (err) {
        console.error(`    ✗ ${zone} ${year} — ${err}`);
      }
    }
  }

  console.log(`\nDone. Files in ${OUT_DIR}`);
}
