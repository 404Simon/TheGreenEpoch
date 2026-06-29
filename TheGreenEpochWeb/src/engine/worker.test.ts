import { describe, it, expect } from "vitest";
import type { FullProfile, CO2Timeline, Scenario, SweepPoint } from "../domain/types";

/**
 * The Structured Clone algorithm is what postMessage uses internally.
 * If the data can't be structured-cloned, the worker will throw
 * DataCloneError.  This test verifies that all data types we send
 * to the worker are cloneable at the type level.
 */

function structuredCloneSafe<T>(data: T): T {
  return structuredClone(data);
}

// ---------------------------------------------------------------------------
// Sample data factories — these mirror what the app sends to the worker
// ---------------------------------------------------------------------------

function sampleProfile(): FullProfile {
  return {
    name: "Deepseek",
    modelParams: 671e9,
    datasetTokens: 14.8e12,
    gpuCount: 2048,
    gpuPowerTrain: 700,
    gpuPowerPause: 60,
    pue: 1.27,
    checkpointPauseTime: 148.8,
    checkpointResumeTime: 0,
  };
}

function sampleTimeline(): CO2Timeline {
  // Realistic size: 5-min intervals for 1 year ≈ 105_120 entries
  const n = 105_120;
  const base = new Date("2022-01-01T00:00:00Z");
  const timestamps: string[] = [];
  const carbonIntensity: number[] = [];
  for (let i = 0; i < n; i++) {
    timestamps.push(new Date(base.getTime() + i * 300_000).toISOString());
    carbonIntensity.push(100 + Math.random() * 400);
  }
  return { zone: "DE", years: [2022], timestamps, carbonIntensity };
}

function sampleScenario(): Scenario {
  return {
    id: "builtin-0",
    description: "DeepSeek-V3 Germany default",
    model: "Deepseek",
    thresholds: [200, 300],
    hysteresis: [150, 250],
    region: "DE",
    startTimes: ["01-01", "06-01"],
    historicalYears: [2022, 2023, 2024, 2025],
    overheadBudgetPct: 200,
  };
}

function sampleSweepPoint(overrides?: Partial<SweepPoint>): SweepPoint {
  return {
    thetaPause: 200,
    thetaResume: 150,
    actualOverheadPct: 12.5,
    co2SavingsPct: 8.3,
    score: 0.664,
    numPauses: 42,
    totalEmissionsKgco2: 5000,
    baselineEmissionsKgco2: 5500,
    withinBudget: true,
    stopReason: "completed",
    completed: true,
    iteration: 0,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Cloneability tests
// ---------------------------------------------------------------------------

describe("worker message data is structured-clone safe", () => {
  it("clones FullProfile", () => {
    const p = sampleProfile();
    const cloned = structuredCloneSafe(p);
    expect(cloned.name).toBe("Deepseek");
    expect(cloned.gpuCount).toBe(2048);
  });

  it("clones CO2Timeline with large arrays", () => {
    const tl = sampleTimeline();
    expect(tl.timestamps.length).toBe(105_120);
    expect(tl.carbonIntensity.length).toBe(105_120);
    const cloned = structuredCloneSafe(tl);
    expect(cloned.timestamps.length).toBe(105_120);
    expect(cloned.carbonIntensity[0]).toBe(tl.carbonIntensity[0]);
    expect(cloned.carbonIntensity[105_119]).toBe(tl.carbonIntensity[105_119]);
  });

  it("clones Scenario", () => {
    const sc = sampleScenario();
    const cloned = structuredCloneSafe(sc);
    expect(cloned.thresholds).toEqual([200, 300]);
    expect(cloned.startTimes).toEqual(["01-01", "06-01"]);
  });

  it("clones the full worker message payload", () => {
    const profile = sampleProfile();
    const timeline = sampleTimeline();
    const scenario = sampleScenario();
    const options = {
      thetaPauseMax: 500,
      overheadBudgetPct: 200,
      resolution: 10,
      maxIterations: 6,
      minStep: 3,
      shrinkFactor: 0.45,
    };
    const payload = { type: "start" as const, profile, timeline, scenario, options, startTimeIdx: 0 };
    const cloned = structuredCloneSafe(payload);
    expect(cloned.type).toBe("start");
    expect(cloned.profile.name).toBe("Deepseek");
    expect(cloned.timeline.timestamps.length).toBe(105_120);
    expect(cloned.scenario.thresholds).toEqual([200, 300]);
  });

  it("clones SweepPoint arrays (iteration messages)", () => {
    const points = [
      sampleSweepPoint({ thetaPause: 100, score: 0.5, iteration: 0 }),
      sampleSweepPoint({ thetaPause: 200, score: 0.8, iteration: 1 }),
      sampleSweepPoint({ thetaPause: 300, score: 0.3, iteration: 2, withinBudget: false }),
    ];
    const msg = { type: "iteration" as const, iteration: 2, points, best: points[1] };
    const cloned = structuredCloneSafe(msg);
    expect(cloned.points).toHaveLength(3);
    expect(cloned.best!.thetaPause).toBe(200);
  });

  it("clones 'done' message with null best", () => {
    const msg = { type: "done" as const, points: [] as SweepPoint[], best: null };
    const cloned = structuredCloneSafe(msg);
    expect(cloned.best).toBeNull();
  });

  it("clones SweepPoint with Infinity-like large thetaPause", () => {
    // Simulates thetaPause = Infinity converted to 9999
    const pt = sampleSweepPoint({ thetaPause: 9999 });
    const cloned = structuredCloneSafe(pt);
    expect(cloned.thetaPause).toBe(9999);
  });

  it("clones scenario with empty threshold arrays", () => {
    const sc = sampleScenario();
    sc.thresholds = [];
    sc.hysteresis = [];
    const cloned = structuredCloneSafe(sc);
    expect(cloned.thresholds).toEqual([]);
    expect(cloned.hysteresis).toEqual([]);
  });

  it("clones timeline with minimal data (2 points)", () => {
    const tl: CO2Timeline = {
      zone: "DE",
      years: [2022],
      timestamps: ["2022-01-01T00:00:00Z", "2022-01-01T00:05:00Z"],
      carbonIntensity: [100, 200],
    };
    const cloned = structuredCloneSafe(tl);
    expect(cloned.timestamps).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// Solid-store proxy simulation — the REAL cause of DataCloneError
// ---------------------------------------------------------------------------

describe("Solid store proxies (runtime DataCloneError reproduction)", () => {
  /**
   * Solid's createStore wraps objects in Proxy.  When those proxied
   * objects are passed to postMessage, the Structured Clone algorithm
   * fails because it can't clone Proxy objects.
   */

  function solidProxy<T extends object>(target: T): T {
    return new Proxy(target, {
      get(t, key) {
        const v = (t as any)[key];
        // Solid deeply wraps nested objects/arrays in proxies too
        if (v !== null && typeof v === "object" && !Array.isArray(v)) {
          return solidProxy(v);
        }
        return v;
      },
    });
  }

  it("fails to clone a Solid-proxied CO2Timeline (!!!)", () => {
    const raw: CO2Timeline = {
      zone: "DE",
      years: [2022],
      timestamps: ["2022-01-01T00:00:00Z", "2022-01-01T00:05:00Z"],
      carbonIntensity: [100, 200],
    };
    const proxied = solidProxy(raw);
    expect(() => structuredClone(proxied)).toThrow();
  });

  it("passes after JSON round-trip (the fix)", () => {
    const raw: CO2Timeline = {
      zone: "DE",
      years: [2022],
      timestamps: ["2022-01-01T00:00:00Z", "2022-01-01T00:05:00Z"],
      carbonIntensity: [100, 200],
    };
    const proxied = solidProxy(raw);
    const safe = JSON.parse(JSON.stringify(proxied)) as CO2Timeline;
    const cloned = structuredClone(safe);
    expect(cloned.timestamps).toHaveLength(2);
  });

  it("deep-clones FullProfile from Solid store", () => {
    const raw: FullProfile = sampleProfile();
    // Simulate the store returning a proxy
    const proxied = solidProxy(raw);
    const safe = JSON.parse(JSON.stringify(proxied)) as FullProfile;
    const cloned = structuredClone(safe);
    expect(cloned.gpuCount).toBe(2048);
  });

  it("deep-clones Scenario from Solid store", () => {
    const raw: Scenario = sampleScenario();
    const proxied = solidProxy(raw);
    const safe = JSON.parse(JSON.stringify(proxied)) as Scenario;
    const cloned = structuredClone(safe);
    expect(cloned.thresholds).toEqual([200, 300]);
  });

  it("deep-clones full worker payload from Solid stores", () => {
    const profile = solidProxy(sampleProfile());
    const timeline = solidProxy(sampleTimeline());
    const scenario = solidProxy(sampleScenario());
    const options = { thetaPauseMax: 500, overheadBudgetPct: 200, resolution: 10, maxIterations: 6, minStep: 3, shrinkFactor: 0.45 };

    const payload = {
      type: "start" as const,
      profile: JSON.parse(JSON.stringify(profile)),
      timeline: JSON.parse(JSON.stringify(timeline)),
      scenario: JSON.parse(JSON.stringify(scenario)),
      options,
      startTimeIdx: 0,
    };
    const cloned = structuredClone(payload);
    expect(cloned.scenario.thresholds).toEqual([200, 300]);
    expect(cloned.timeline.timestamps.length).toBe(105_120);
  });
});
