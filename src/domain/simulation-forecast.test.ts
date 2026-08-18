import { describe, it, expect } from "vitest";
import { simulateStepwise } from "./simulation";
import { applyForecast } from "./forecast";
import { runOptimization } from "./optimize";
import type { AdaptiveOptions } from "./optimize";
import { hysteresisPolicy } from "./policy";
import type { FullProfile, CO2Timeline, SimConfig } from "./types";
import { SimState } from "./types";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const testProfile: FullProfile = {
  name: "TestModel",
  modelParams: 1e9,
  datasetTokens: 100_000_000,
  gpuCount: 1,
  gpuPowerTrain: 700,
  gpuPowerPause: 60,
  pue: 1.0,
  checkpointPauseTime: 0,
  checkpointResumeTime: 0,
};

const testTimeline: CO2Timeline = {
  zone: "DE",
  years: [2022],
  timestamps: [
    "2022-01-01T00:00:00Z",
    "2022-01-01T00:05:00Z",
    "2022-01-01T00:10:00Z",
    "2022-01-01T00:15:00Z",
    "2022-01-01T00:20:00Z",
    "2022-01-01T00:25:00Z",
  ],
  carbonIntensity: [100, 200, 300, 400, 500, 600],
};

function makeConfig(overrides?: Partial<SimConfig>): SimConfig {
  return {
    startTime: "01-01",
    historicalYears: [2022],
    overheadBudgetPct: 200,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Identity decisionTimeline is bit-identical to no decisionTimeline
// ---------------------------------------------------------------------------

describe("simulateStepwise with identity decisionTimeline", () => {
  it("produces bit-identical SimProgress sequences vs no decisionTimeline", () => {
    const policy = hysteresisPolicy(200, 100);
    const identity = applyForecast(testTimeline, { type: "identity" }, 1);
    const without = [...simulateStepwise(testProfile, policy, testTimeline, makeConfig())];
    const withDecision = [
      ...simulateStepwise(testProfile, policy, testTimeline, makeConfig({ decisionTimeline: identity })),
    ];
    expect(without.length).toBe(withDecision.length);
    for (let i = 0; i < without.length; i++) {
      expect(withDecision[i]).toEqual(without[i]);
    }
  });
});

// ---------------------------------------------------------------------------
// Decision/accounting split
// ---------------------------------------------------------------------------

describe("decision/accounting split", () => {
  const realized: CO2Timeline = {
    zone: "DE",
    years: [2022],
    timestamps: testTimeline.timestamps,
    carbonIntensity: [100, 100, 100, 100, 100, 100],
  };
  const decision: CO2Timeline = {
    zone: "DE",
    years: [2022],
    timestamps: testTimeline.timestamps,
    carbonIntensity: [600, 600, 30, 30, 30, 30],
  };
  const policy = hysteresisPolicy(300, 50);

  it("pauses based on the decision value, not the realized value", () => {
    const noDecision = [...simulateStepwise(testProfile, policy, realized, makeConfig())];
    const withDecision = [...simulateStepwise(testProfile, policy, realized, makeConfig({ decisionTimeline: decision }))];

    const lastNo = noDecision[noDecision.length - 1];
    const lastYes = withDecision[withDecision.length - 1];

    expect(lastNo.numPauses).toBe(0);
    expect(lastYes.numPauses).toBeGreaterThan(0);
    expect(withDecision.some((p) => p.state === SimState.PAUSED)).toBe(true);
  });

  it("accounts emissions at the realized intensity", () => {
    const withDecision = [...simulateStepwise(testProfile, policy, realized, makeConfig({ decisionTimeline: decision }))];
    const last = withDecision[withDecision.length - 1];
    const ratio = last.totalEmissionsG / last.totalEnergyWh;
    expect(ratio).toBeCloseTo(0.1, 3);
    expect(last.carbonIntensity).toBe(100);
  });
});

// ---------------------------------------------------------------------------
// decisionTimeline validation
// ---------------------------------------------------------------------------

describe("decisionTimeline validation", () => {
  it("throws on carbonIntensity length mismatch", () => {
    const bad: CO2Timeline = { ...testTimeline, carbonIntensity: [1, 2, 3] };
    expect(() =>
      [...simulateStepwise(testProfile, hysteresisPolicy(200, 100), testTimeline, makeConfig({ decisionTimeline: bad }))],
    ).toThrow(/decisionTimeline/);
  });

  it("throws on timestamps length mismatch", () => {
    const bad: CO2Timeline = { ...testTimeline, timestamps: testTimeline.timestamps.slice(0, 3) };
    expect(() =>
      [...simulateStepwise(testProfile, hysteresisPolicy(200, 100), testTimeline, makeConfig({ decisionTimeline: bad }))],
    ).toThrow(/decisionTimeline/);
  });
});

// ---------------------------------------------------------------------------
// runOptimization with identity forecast
// ---------------------------------------------------------------------------

describe("runOptimization with identity forecast", () => {
  it("produces identical points with and without decisionTimeline", () => {
    const n = 200;
    const synthetic: CO2Timeline = {
      zone: "DE",
      years: [2022],
      timestamps: Array.from({ length: n }, (_, i) => new Date(Date.UTC(2022, 0, 1) + i * 300_000).toISOString()),
      carbonIntensity: Array.from({ length: n }, (_, i) => 150 + 100 * Math.sin(i / 20)),
    };
    const options: AdaptiveOptions = {
      thetaPauseMax: 300,
      overheadBudgetPct: 200,
      resolution: 3,
      startDateResolution: 2,
      maxIterations: 1,
      minStep: 3,
      shrinkFactor: 0.45,
      alpha: 1,
      fixedStartTime: "01-01",
    };
    const identity = applyForecast(synthetic, { type: "identity" }, 1);
    const without = runOptimization(testProfile, synthetic, [2022], options);
    const withDecision = runOptimization(testProfile, synthetic, [2022], options, undefined, identity);
    expect(withDecision.points).toEqual(without.points);
    expect(withDecision.best).toEqual(without.best);
  });
});
