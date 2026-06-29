import { describe, it, expect } from "vitest";
import { tokensPerSecond, energyWh, emissionsG } from "../domain/physics";
import { findStartIndex, simulateStepwise } from "../domain/simulation";
import { buildSimResult } from "../domain/result";
import { hysteresisPolicy, neverPausePolicy } from "../domain/policy";
import type { FullProfile, CO2Timeline, SimConfig } from "../domain/types";
import { SimState } from "../domain/types";

const HOUR = 3600;

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
// Physics
// ---------------------------------------------------------------------------

describe("physics", () => {
  it("tokensPerSecond returns positive rate", () => {
    const rate = tokensPerSecond(1);
    expect(rate).toBeGreaterThan(0);
    expect(tokensPerSecond(2)).toBe(2 * tokensPerSecond(1));
  });

  it("tokensPerSecond returns zero for zero GPUs", () => {
    expect(tokensPerSecond(0)).toBe(0);
  });

  it("energyWh computes correctly", () => {
    expect(energyWh(1000, HOUR)).toBeCloseTo(1000, 0);
    expect(energyWh(700, 360)).toBeCloseTo(70, 0);
    expect(energyWh(0, 100)).toBe(0);
  });

  it("emissionsG computes correctly", () => {
    expect(emissionsG(1000, 500)).toBeCloseTo(500, 0);
    expect(emissionsG(0, 500)).toBe(0);
    expect(emissionsG(1000, 0)).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// findStartIndex
// ---------------------------------------------------------------------------

describe("findStartIndex", () => {
  const tss = [
    "2022-01-01T00:00:00Z",
    "2022-01-01T00:05:00Z",
    "2022-01-01T00:10:00Z",
    "2022-01-01T00:15:00Z",
  ];

  it("finds index 0 for 01-01", () => {
    expect(findStartIndex(tss, "01-01")).toBe(0);
  });

  it("returns last index when start time is beyond all timestamps", () => {
    const idx = findStartIndex(tss, "12-31");
    expect(idx).toBe(tss.length - 1);
  });
});

// ---------------------------------------------------------------------------
// simulateStepwise – basic correctness
// ---------------------------------------------------------------------------

describe("simulateStepwise", () => {
  it("yields multiple progress items until training completes", () => {
    const policy = neverPausePolicy();
    const results = [...simulateStepwise(testProfile, policy, testTimeline, makeConfig())];
    expect(results.length).toBeGreaterThan(1);
    const last = results[results.length - 1];
    expect(last.done).toBe(true);
    expect(last.tokensRemaining).toBe(0);
  });

  it("never triggers a policy pause when using never-pause", () => {
    const policy = neverPausePolicy();
    const results = [...simulateStepwise(testProfile, policy, testTimeline, makeConfig())];
    const last = results[results.length - 1];
    expect(last.numPauses).toBe(0);
    expect(last.state).toBe(SimState.RUNNING);
  });

  it("pauses when CO2 exceeds threshold", () => {
    const policy = hysteresisPolicy(150, 50);
    const results = [...simulateStepwise(testProfile, policy, testTimeline, makeConfig())];
    const last = results[results.length - 1];
    expect(last.numPauses).toBeGreaterThan(0);
  });

  it("resumes when CO2 drops below theta_resume", () => {
    const tl: CO2Timeline = {
      zone: "DE",
      years: [2022],
      timestamps: [
        "2022-01-01T00:00:00Z",
        "2022-01-01T00:05:00Z",
        "2022-01-01T00:10:00Z",
        "2022-01-01T00:15:00Z",
        "2022-01-01T00:20:00Z",
      ],
      carbonIntensity: [400, 400, 400, 30, 30],
    };
    const policy = hysteresisPolicy(300, 50);
    const results = [...simulateStepwise(testProfile, policy, tl, makeConfig())];
    const states = results.map((r) => r.state);
    expect(states).toContain(SimState.PAUSED);
    expect(states).toContain(SimState.RUNNING);
    let sawPaused = false;
    let sawRunningAfterPaused = false;
    for (const s of states) {
      if (s === SimState.PAUSED) sawPaused = true;
      if (sawPaused && s === SimState.RUNNING) sawRunningAfterPaused = true;
    }
    expect(sawRunningAfterPaused).toBe(true);
  });

  it("stops when overhead budget is exceeded", () => {
    const policy = hysteresisPolicy(50, 10);
    const results = [...simulateStepwise(testProfile, policy, testTimeline, makeConfig({ overheadBudgetPct: 5 }))];
    const last = results[results.length - 1];
    expect(last.stopReason).toBe("budget_exceeded");
    expect(last.tokensRemaining).toBeGreaterThan(0);
  });

  it("completes training with never-pause policy", () => {
    const policy = neverPausePolicy();
    const results = [...simulateStepwise(testProfile, policy, testTimeline, makeConfig())];
    const last = results[results.length - 1];
    expect(last.done).toBe(true);
    expect(last.tokensRemaining).toBe(0);
    expect(last.stopReason).toBe("completed");
  });
});

// ---------------------------------------------------------------------------
// buildSimResult
// ---------------------------------------------------------------------------

describe("buildSimResult", () => {
  it("computes zero savings for baseline-vs-baseline", () => {
    const policy = neverPausePolicy();
    const config = makeConfig();
    const prog = [...simulateStepwise(testProfile, policy, testTimeline, config)];
    const last = prog[prog.length - 1];

    const result = buildSimResult(testProfile, config, last, last, 1e9, 0, {
      id: "test",
      scenarioDescription: "test",
      model: "TestModel",
      region: "DE",
      timestamps: [],
      carbonIntensitySeries: [],
      stateSeries: [],
      emissionsSeries: [],
      tokensRemainingSeries: [],
    });
    expect(result.baselineEmissionsKgco2).toBeCloseTo(result.totalEmissionsKgco2, 3);
    expect(result.withinOverheadBudget).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

describe("determinism", () => {
  it("produces identical results on repeated runs", () => {
    const policy = hysteresisPolicy(200, 100);
    const config = makeConfig();
    const run1 = [...simulateStepwise(testProfile, policy, testTimeline, config)];
    const run2 = [...simulateStepwise(testProfile, policy, testTimeline, config)];
    expect(run1.length).toBe(run2.length);
    for (let i = 0; i < run1.length; i++) {
      expect(run1[i].totalEmissionsG).toBe(run2[i].totalEmissionsG);
      expect(run1[i].numPauses).toBe(run2[i].numPauses);
      expect(run1[i].state).toBe(run2[i].state);
    }
  }, 60_000);
});

// ---------------------------------------------------------------------------
// Real-data regression test
// ---------------------------------------------------------------------------

describe("regression vs Python", () => {
  const deTimeline: CO2Timeline = {
    zone: "DE",
    years: [2022],
    timestamps: ["2022-01-01T00:00:00Z", "2022-01-01T00:05:00Z"],
    carbonIntensity: [215, 215],
  };

  const deepseek: FullProfile = {
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

  it("baseline never pauses", () => {
    const policy = neverPausePolicy();
    const results = [...simulateStepwise(deepseek, policy, deTimeline, makeConfig())];
    const last = results[results.length - 1];
    expect(last.numPauses).toBe(0);
  });

  it("pauses when CO2 exceeds deepseek threshold", () => {
    const policy = hysteresisPolicy(200, 100);
    const results = [...simulateStepwise(deepseek, policy, deTimeline, makeConfig())];
    const last = results[results.length - 1];
    expect(last.numPauses).toBeGreaterThan(0);
  });
});
