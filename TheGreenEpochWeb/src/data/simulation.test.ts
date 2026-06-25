import { describe, it, expect } from "vitest";
import { tokensPerSecond, energyWh, emissionsG, findStartIndex, simulateStepwise, buildResult } from "./simulation";
import type { FullProfile, SimConfig, CO2Timeline } from "../types";
import { SimState } from "../types";

const HOUR = 3600;

const testProfile: FullProfile = {
  name: "TestModel",
  modelParams: 1e9,
  datasetTokens: 100_000_000, // 100M tokens -> ~216 steps at 1 GPU
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
    // "12-31" would be way past Jan 1 -> should return last index
    const idx = findStartIndex(tss, "12-31");
    expect(idx).toBe(tss.length - 1);
  });
});

// ---------------------------------------------------------------------------
// simulateStepwise – basic correctness
// ---------------------------------------------------------------------------

describe("simulateStepwise", () => {
  it("yields multiple progress items until training completes", () => {
    const config: SimConfig = {
      scenarioDescription: "test",
      region: "DE",
      historicalYears: [2022],
      startTime: "01-01",
      thetaPause: 9999, // effectively infinity
      thetaResume: 0,
      overheadBudgetPct: 200,
    };
    const results = [...simulateStepwise(testProfile, config, testTimeline)];
    expect(results.length).toBeGreaterThan(1);
    const last = results[results.length - 1];
    expect(last.done).toBe(true);
    expect(last.tokensRemaining).toBe(0);
  });

  it("never triggers a policy pause when theta_pause is huge (baseline)", () => {
    const config: SimConfig = {
      scenarioDescription: "baseline",
      region: "DE",
      historicalYears: [2022],
      startTime: "01-01",
      thetaPause: 1e9, // very high threshold
      thetaResume: 0,
      overheadBudgetPct: 200,
    };
    const results = [...simulateStepwise(testProfile, config, testTimeline)];
    const last = results[results.length - 1];
    expect(last.numPauses).toBe(0);
    expect(last.state).toBe(SimState.RUNNING);
  });

  it("pauses when CO2 exceeds threshold", () => {
    const config: SimConfig = {
      scenarioDescription: "pause-test",
      region: "DE",
      historicalYears: [2022],
      startTime: "01-01",
      thetaPause: 150, // exceeds first CO2=100, but second CO2=200
      thetaResume: 50,
      overheadBudgetPct: 200,
    };
    const results = [...simulateStepwise(testProfile, config, testTimeline)];
    const last = results[results.length - 1];
    expect(last.numPauses).toBeGreaterThan(0);
  });

  it("resumes when CO2 drops below theta_resume", () => {
    // Timeline: starts high, then drops low
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
    const config: SimConfig = {
      scenarioDescription: "resume-test",
      region: "DE",
      historicalYears: [2022],
      startTime: "01-01",
      thetaPause: 300, // CO2=400 > 300 -> pause
      thetaResume: 50, // CO2=30 < 50 -> resume
      overheadBudgetPct: 200,
    };
    const results = [...simulateStepwise(testProfile, config, tl)];
    const states = results.map((r) => r.state);
    expect(states).toContain(SimState.PAUSED);
    expect(states).toContain(SimState.RUNNING);
    // At some point it should resume: find a RUNNING after a PAUSED
    let sawPaused = false;
    let sawRunningAfterPaused = false;
    for (const s of states) {
      if (s === SimState.PAUSED) sawPaused = true;
      if (sawPaused && s === SimState.RUNNING) sawRunningAfterPaused = true;
    }
    expect(sawRunningAfterPaused).toBe(true);
  });

  it("stops when overhead budget is exceeded", () => {
    const config: SimConfig = {
      scenarioDescription: "budget-test",
      region: "DE",
      historicalYears: [2022],
      startTime: "01-01",
      thetaPause: 50,    // very low -> almost always pauses
      thetaResume: 10,
      overheadBudgetPct: 5, // very tight budget
    };
    const results = [...simulateStepwise(testProfile, config, testTimeline)];
    const last = results[results.length - 1];
    expect(last.stopReason).toBe("budget_exceeded");
    expect(last.tokensRemaining).toBeGreaterThan(0);
  });

  it("completes training when theta_pause is above all CO2 values", () => {
    const config: SimConfig = {
      scenarioDescription: "complete-test",
      region: "DE",
      historicalYears: [2022],
      startTime: "01-01",
      thetaPause: 9999,
      thetaResume: 0,
      overheadBudgetPct: 200,
    };
    const results = [...simulateStepwise(testProfile, config, testTimeline)];
    const last = results[results.length - 1];
    expect(last.done).toBe(true);
    expect(last.tokensRemaining).toBe(0);
    expect(last.stopReason).toBe("completed");
  });
});

// ---------------------------------------------------------------------------
// buildResult
// ---------------------------------------------------------------------------

describe("buildResult", () => {
  it("computes zero savings for baseline-vs-baseline", () => {
    const config: SimConfig = {
      scenarioDescription: "test",
      region: "DE",
      historicalYears: [2022],
      startTime: "01-01",
      thetaPause: 1e9,
      thetaResume: 0,
      overheadBudgetPct: 200,
    };
    const prog = [...simulateStepwise(testProfile, config, testTimeline)];
    const last = prog[prog.length - 1];

    const result = buildResult(testProfile, config, last, last);
    expect(result.baselineEmissionsKgco2).toBeCloseTo(result.totalEmissionsKgco2, 3);
    expect(result.withinOverheadBudget).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Determinism - same inputs ⇒ same outputs
// ---------------------------------------------------------------------------

describe("determinism", () => {
  it("produces identical results on repeated runs", () => {
    const config: SimConfig = {
      scenarioDescription: "det-test",
      region: "DE",
      historicalYears: [2022],
      startTime: "01-01",
      thetaPause: 200,
      thetaResume: 100,
      overheadBudgetPct: 200,
    };
    const run1 = [...simulateStepwise(testProfile, config, testTimeline)];
    const run2 = [...simulateStepwise(testProfile, config, testTimeline)];
    expect(run1.length).toBe(run2.length);
    for (let i = 0; i < run1.length; i++) {
      expect(run1[i].totalEmissionsG).toBe(run2[i].totalEmissionsG);
      expect(run1[i].numPauses).toBe(run2[i].numPauses);
      expect(run1[i].state).toBe(run2[i].state);
    }
  }, 60_000);
});

// ---------------------------------------------------------------------------
// Real-data regression test (matches Python output)
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
    const cfg: SimConfig = {
      scenarioDescription: "regression",
      region: "DE",
      historicalYears: [2022],
      startTime: "01-01",
      thetaPause: Infinity,
      thetaResume: 0,
      overheadBudgetPct: 200,
    };
    const results = [...simulateStepwise(deepseek, cfg, deTimeline)];
    const last = results[results.length - 1];
    expect(last.numPauses).toBe(0);
  });

  it("pauses when CO2 exceeds deepseek threshold", () => {
    // DE carbonIntensity=215, theta_pause=200 -> should pause
    const cfg: SimConfig = {
      scenarioDescription: "regression",
      region: "DE",
      historicalYears: [2022],
      startTime: "01-01",
      thetaPause: 200,
      thetaResume: 100,
      overheadBudgetPct: 200,
    };
    const results = [...simulateStepwise(deepseek, cfg, deTimeline)];
    const last = results[results.length - 1];
    expect(last.numPauses).toBeGreaterThan(0);
  });
});
