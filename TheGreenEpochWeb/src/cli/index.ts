/**
 * CLI entry point - runs the simulation engine from Node.js.
 *
 * Usage:
 *   npx tsx src/cli/index.ts
 *   npx tsx src/cli/index.ts --limit 2
 *
 * Shares the exact same simulation engine (simulation.ts) with the web app.
 * Only the data-loading layer differs (fs instead of fetch).
 */

import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Constants, TrainingProfile, Scenario, CO2Timeline, FullProfile } from "../types";
import { simulateStepwise, buildResult, tokensPerSecond } from "../data/simulation.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(__dirname, "../../public/data");

function loadJSON<T>(path: string): T {
  return JSON.parse(readFileSync(resolve(DATA_DIR, path), "utf-8")) as T;
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

async function main() {
  const args = process.argv.slice(2);
  const limitIdx = args.indexOf("--limit");
  const limit = limitIdx !== -1 && args[limitIdx + 1] ? parseInt(args[limitIdx + 1]) : null;
  const csvIdx = args.indexOf("--csv");
  const csvPath = csvIdx !== -1 && args[csvIdx + 1] ? args[csvIdx + 1] : null;
  const skipLive = args.includes("--no-live");

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
    const tl = loadJSON<CO2Timeline>(`co2/${sc.region}.json`);

    for (const startTime of sc.startTimes) {
      for (let ti = 0; ti < sc.thresholds.length; ti++) {
        const config = {
          scenarioDescription: sc.description,
          region: sc.region,
          historicalYears: sc.historicalYears,
          startTime,
          thetaPause: sc.thresholds[ti],
          thetaResume: sc.hysteresis[ti],
          overheadBudgetPct: sc.overheadBudgetPct,
        };

        const baselineConfig = { ...config, thetaPause: Infinity, thetaResume: 0 };

        const progress: Parameters<typeof buildResult>[2][] = [];
        for (const p of simulateStepwise(full, config, tl)) progress.push(p);
        const last = progress[progress.length - 1];

        const baselineProgress: Parameters<typeof buildResult>[2][] = [];
        for (const p of simulateStepwise(full, baselineConfig, tl)) baselineProgress.push(p);
        const lastBaseline = baselineProgress[baselineProgress.length - 1];

        const meta = buildResult(full, config, last, lastBaseline);

        const idealS = last.tokensTotal / (tokensPerSecond(profile.gpuCount) || 1);
        const actualOverheadPct = 100 * (last.pausedS + last.checkpointS) / (idealS || 1);
        const co2SavingsPct = meta.baselineEmissionsKgco2 > 0
          ? (meta.baselineEmissionsKgco2 - meta.totalEmissionsKgco2) / meta.baselineEmissionsKgco2 * 100
          : 0;
        const score = co2SavingsPct / Math.max(actualOverheadPct, 0.001);

        const isOk = last.done && meta.withinOverheadBudget && last.issues.length === 0;
        if (isOk) ok++; else fail++;

        function r6(n: number) { return Math.round(n * 1_000_000) / 1_000_000; }
        results.push({
          scenario: sc.description,
          model: profile.name,
          region: sc.region,
          historicalYears: sc.historicalYears.join(";"),
          startTime,
          thetaPause: config.thetaPause,
          thetaResume: config.thetaResume,
          overheadBudgetPct: config.overheadBudgetPct,
          totalWallTimeH: r6(last.totalWallS / 3600),
          trainingTimeH: r6(last.trainingS / 3600),
          pausedTimeH: r6(last.pausedS / 3600),
          checkpointOverheadH: r6(last.checkpointS / 3600),
          totalEnergyKwh: r6(last.totalEnergyWh / 1000),
          trainingEnergyKwh: r6(last.trainingEnergyWh / 1000),
          pausedEnergyKwh: r6(last.pausedEnergyWh / 1000),
          checkpointEnergyKwh: r6(last.checkpointEnergyWh / 1000),
          totalEmissionsKgco2: r6(last.totalEmissionsG / 1000),
          tokensProcessed: last.tokensTotal - last.tokensRemaining,
          tokensTotal: last.tokensTotal,
          completed: last.done && last.tokensRemaining <= 0,
          numPauses: last.numPauses,
          actualOverheadPct: r6(actualOverheadPct),
          withinOverheadBudget: meta.withinOverheadBudget,
          baselineEmissionsKgco2: r6(meta.baselineEmissionsKgco2),
          baselineTimeH: r6(meta.baselineTimeH),
          co2SavingsPct: r6(co2SavingsPct),
          score: r6(score),
          idleTimeH: r6((last.pausedS + last.checkpointS) / 3600),
          completionPct: last.tokensTotal > 0 ? r6(100 * (last.tokensTotal - last.tokensRemaining) / last.tokensTotal) : 0,
          ok: isOk,
          stopReason: last.stopReason,
          issues: last.issues.join("; "),
        });

        if (!skipLive) {
          process.stdout.write(".");
        }
      }
    }
  }

  // Summary
  console.log(`\n\n  ─────────────────────────────────────────────`);
  console.log(`  Results: ${results.length} runs · ✓ ${ok} · ✗ ${fail}`);
  console.log(`  ─────────────────────────────────────────────`);
  console.log(`  ${"REGION".padEnd(8)} ${"THR".padStart(4)} ${"CO₂↓".padStart(7)} ${"SCORE".padStart(7)} ${"OVERH".padStart(6)} ${"PAUSES".padStart(7)}`);
  console.log(`  ${"──".repeat(20)}`);

  for (const r of results.slice(0, 20)) {
    const status = r.ok ? "✓" : "✗";
    console.log(`  ${r.region.padEnd(8)} ${String(r.thetaPause).padStart(4)} ${(r.co2SavingsPct + "%").padStart(7)} ${String(r.score).padStart(7)} ${(r.actualOverheadPct + "%").padStart(6)} ${String(r.numPauses).padStart(7)}  ${status} ${r.scenario.slice(0, 36).padEnd(36)}`);
  }

  if (results.length > 20) {
    console.log(`  … and ${results.length - 20} more`);
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

main().catch((err) => {
  console.error("CLI error:", err);
  process.exit(1);
});
