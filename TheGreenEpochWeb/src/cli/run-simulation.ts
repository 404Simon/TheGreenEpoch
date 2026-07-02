import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Constants, TrainingProfile, Scenario, FullProfile, SimConfig, CO2Timeline, YearCO2, SimProgress } from "../domain/types";
import { simulateStepwise } from "../domain/simulation";
import { neverPausePolicy, hysteresisPolicy } from "../domain/policy";
import { buildSimResult } from "../domain/result";
import { averageYears } from "../data/co2-loader";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(__dirname, "../../public/data");

function loadJSON<T>(path: string): T {
  return JSON.parse(readFileSync(resolve(DATA_DIR, path), "utf-8")) as T;
}

function loadCO2Timeline(zone: string, years: number[]): CO2Timeline {
  const allData = years.map((y) => {
    return loadJSON<YearCO2>(`co2/${zone}_${y}.json`);
  });
  return averageYears(allData);
}

function loadData() {
  const constants = loadJSON<Constants>("constants.json");
  const profiles = loadJSON<Record<string, TrainingProfile>>("profiles.json");
  const scenarios = loadJSON<Scenario[]>("scenarios.json");
  return { constants, profiles, scenarios };
}

function wrapProfile(p: TrainingProfile, c: Constants): FullProfile {
  return {
    ...p,
    gpuPowerTrain: c.gpu_power_train,
    gpuPowerPause: c.gpu_power_pause,
    pue: c.pue,
    checkpointPauseTime: c.checkpoint_pause_time,
    checkpointResumeTime: c.checkpoint_resume_time,
  };
}

interface CliResult {
  scenario: string;
  model: string;
  region: string;
  historicalYears: string;
  startTime: string;
  thetaPause: number;
  thetaResume: number;
  overheadBudgetPct: number;
  totalWallTimeH: number;
  trainingTimeH: number;
  pausedTimeH: number;
  checkpointOverheadH: number;
  totalEnergyKwh: number;
  trainingEnergyKwh: number;
  pausedEnergyKwh: number;
  checkpointEnergyKwh: number;
  totalEmissionsKgco2: number;
  tokensProcessed: number;
  tokensTotal: number;
  completed: boolean;
  numPauses: number;
  actualOverheadPct: number;
  withinOverheadBudget: boolean;
  baselineEmissionsKgco2: number;
  baselineTimeH: number;
  co2SavingsPct: number;
  score: number;
  idleTimeH: number;
  completionPct: number;
  ok: boolean;
  stopReason: string;
  issues: string;
}

function r6(n: number) { return Math.round(n * 1_000_000) / 1_000_000; }

export async function runSimulation(): Promise<void> {
  const args = process.argv.slice(2);
  const idx = args.indexOf("run");
  const sub = idx !== -1 ? args.slice(idx + 1) : args;
  const limitIdx = sub.indexOf("--limit");
  const limit = limitIdx !== -1 && sub[limitIdx + 1] ? parseInt(sub[limitIdx + 1]) : null;
  const csvIdx = sub.indexOf("--csv");
  const csvPath = csvIdx !== -1 && sub[csvIdx + 1] ? sub[csvIdx + 1] : null;
  const skipLive = sub.includes("--no-live");

  const { constants, profiles, scenarios } = loadData();
  const toRun = limit ? scenarios.slice(0, limit) : scenarios;

  console.log(`\n TheGreenEpoch CLI - ${toRun.length} scenario(s)\n`);

  const results: CliResult[] = [];
  let ok = 0, fail = 0;

  for (const sc of toRun) {
    const profile = profiles[sc.model];
    if (!profile) {
      console.warn(`  ⚠ Unknown model: ${sc.model}`);
      continue;
    }
    const full = wrapProfile(profile, constants);
    const tl = loadCO2Timeline(sc.region, sc.historicalYears);

    for (const startTime of sc.startTimes) {
      for (let ti = 0; ti < sc.thresholds.length; ti++) {
        const simConfig: SimConfig = {
          startTime,
          historicalYears: sc.historicalYears,
          overheadBudgetPct: sc.overheadBudgetPct,
        };
        const thetaPause = sc.thresholds[ti];
        const thetaResume = sc.hysteresis[ti];
        const policy = hysteresisPolicy(thetaPause, thetaResume);
        const baselinePolicy = neverPausePolicy();

        const progress: SimProgress[] = [];
        for (const p of simulateStepwise(full, baselinePolicy, tl, simConfig)) progress.push(p);
        const lastBaseline = progress[progress.length - 1];

        const simProgress: SimProgress[] = [];
        for (const p of simulateStepwise(full, policy, tl, simConfig)) simProgress.push(p);
        const last = simProgress[simProgress.length - 1];

        const result = buildSimResult(full, simConfig, last, lastBaseline, thetaPause, thetaResume, {
          id: `cli-${Date.now()}`,
          scenarioDescription: sc.description,
          model: profile.name,
          region: sc.region,
          timestamps: [],
          carbonIntensitySeries: [],
          stateSeries: [],
          emissionsSeries: [],
          tokensRemainingSeries: [],
        });

        if (result.ok) ok++; else fail++;

        results.push({
          scenario: sc.description,
          model: profile.name,
          region: sc.region,
          historicalYears: sc.historicalYears.join(";"),
          startTime,
          thetaPause,
          thetaResume,
          overheadBudgetPct: sc.overheadBudgetPct,
          totalWallTimeH: r6(result.totalWallTimeH),
          trainingTimeH: r6(result.trainingTimeH),
          pausedTimeH: r6(result.pausedTimeH),
          checkpointOverheadH: r6(result.checkpointOverheadH),
          totalEnergyKwh: r6(result.totalEnergyKwh),
          trainingEnergyKwh: r6(result.trainingEnergyKwh),
          pausedEnergyKwh: r6(result.pausedEnergyKwh),
          checkpointEnergyKwh: r6(result.checkpointEnergyKwh),
          totalEmissionsKgco2: r6(result.totalEmissionsKgco2),
          tokensProcessed: result.tokensProcessed,
          tokensTotal: result.tokensTotal,
          completed: result.completed,
          numPauses: result.numPauses,
          actualOverheadPct: r6(result.actualOverheadPct),
          withinOverheadBudget: result.withinOverheadBudget,
          baselineEmissionsKgco2: r6(result.baselineEmissionsKgco2),
          baselineTimeH: r6(result.baselineTimeH),
          co2SavingsPct: r6(result.co2SavingsPct),
          score: r6(result.score),
          idleTimeH: r6(result.idleTimeH),
          completionPct: r6(result.completionPct),
          ok: result.ok,
          stopReason: result.stopReason,
          issues: result.issues.join("; "),
        });

        if (!skipLive) {
          process.stdout.write(".");
        }
      }
    }
  }

  console.log(`\n\n  ─────────────────────────────────────────────`);
  console.log(`  Results: ${results.length} runs · ✓ ${ok} · ✗ ${fail}`);
  console.log(`  ─────────────────────────────────────────────`);
  console.log(`  ${"REGION".padEnd(8)} ${"THR".padStart(4)} ${"CO₂↓".padStart(7)} ${"SCORE".padStart(7)} ${"OVERH".padStart(6)} ${"PAUSES".padStart(7)}`);
  console.log(`  ${"──".repeat(20)}`);

  for (const r of results) {
    const status = r.ok ? "✓" : "✗";
    console.log(`  ${r.region.padEnd(8)} ${String(r.thetaPause).padStart(4)} ${(r.co2SavingsPct + "%").padStart(7)} ${String(r.score).padStart(7)} ${(r.actualOverheadPct + "%").padStart(6)} ${String(r.numPauses).padStart(7)}  ${status} ${r.scenario.slice(0, 36).padEnd(36)}`);
  }

  if (csvPath) {
    const cols: { key: keyof CliResult; header: string }[] = [
      { key: "scenario", header: "scenario" },
      { key: "model", header: "model" },
      { key: "region", header: "region" },
      { key: "historicalYears", header: "historical_years" },
      { key: "startTime", header: "start_time" },
      { key: "thetaPause", header: "theta_pause" },
      { key: "thetaResume", header: "theta_resume" },
      { key: "overheadBudgetPct", header: "overhead_budget_pct" },
      { key: "totalWallTimeH", header: "total_wall_time_h" },
      { key: "trainingTimeH", header: "training_time_h" },
      { key: "pausedTimeH", header: "paused_time_h" },
      { key: "checkpointOverheadH", header: "checkpoint_overhead_h" },
      { key: "totalEnergyKwh", header: "total_energy_kwh" },
      { key: "trainingEnergyKwh", header: "training_energy_kwh" },
      { key: "pausedEnergyKwh", header: "paused_energy_kwh" },
      { key: "checkpointEnergyKwh", header: "checkpoint_energy_kwh" },
      { key: "totalEmissionsKgco2", header: "total_emissions_kgco2" },
      { key: "tokensProcessed", header: "tokens_processed" },
      { key: "tokensTotal", header: "tokens_total" },
      { key: "completed", header: "completed" },
      { key: "numPauses", header: "num_pauses" },
      { key: "actualOverheadPct", header: "actual_overhead_pct" },
      { key: "withinOverheadBudget", header: "within_overhead_budget" },
      { key: "baselineEmissionsKgco2", header: "baseline_emissions_kgco2" },
      { key: "baselineTimeH", header: "baseline_time_h" },
      { key: "co2SavingsPct", header: "co2_savings_pct" },
      { key: "score", header: "score" },
      { key: "idleTimeH", header: "idle_time_h" },
      { key: "completionPct", header: "completion_pct" },
      { key: "ok", header: "ok" },
      { key: "stopReason", header: "stop_reason" },
      { key: "issues", header: "issues" },
    ];

    function esc(v: unknown): string {
      const s = String(v);
      return s.includes(",") || s.includes('"') || s.includes("\n") ? `"${s.replace(/"/g, '""')}"` : s;
    }

    const lines = results.map((r) => cols.map((c) => esc(r[c.key])).join(","));
    lines.unshift(cols.map((c) => c.header).join(","));
    writeFileSync(csvPath, lines.join("\n"), "utf-8");
    console.log(`\n  📄 CSV: ${csvPath} (${results.length} rows)\n`);
  }

  console.log(`\n  Done. ✓ ${ok} · ✗ ${fail}\n`);
}
