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

import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Constants, TrainingProfile, Scenario, CO2Timeline, FullProfile } from "../types";
import { simulateStepwise, buildResult, tokensPerSecond } from "../data/simulation";

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
  region: string;
  threshold: number;
  hysteresis: number;
  startTime: string;
  co2SavingsPct: number;
  score: number;
  overheadPct: number;
  pauses: number;
  wallTimeH: number;
  emissionsKg: number;
  ok: boolean;
}

async function main() {
  const args = process.argv.slice(2);
  const limitIdx = args.indexOf("--limit");
  const limit = limitIdx !== -1 && args[limitIdx + 1] ? parseInt(args[limitIdx + 1]) : null;
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

    for (let ti = 0; ti < sc.thresholds.length; ti++) {
      for (const startTime of sc.startTimes) {
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

        results.push({
          scenario: sc.description,
          region: sc.region,
          threshold: config.thetaPause,
          hysteresis: config.thetaResume,
          startTime,
          co2SavingsPct: Math.round(co2SavingsPct * 10) / 10,
          score: Math.round(score * 10) / 10,
          overheadPct: Math.round(actualOverheadPct * 10) / 10,
          pauses: last.numPauses,
          wallTimeH: Math.round(last.totalWallS / 36) / 100,
          emissionsKg: Math.round(last.totalEmissionsG / 1000),
          ok: isOk,
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
    console.log(`  ${r.region.padEnd(8)} ${String(r.threshold).padStart(4)} ${(r.co2SavingsPct + "%").padStart(7)} ${String(r.score).padStart(7)} ${(r.overheadPct + "%").padStart(6)} ${String(r.pauses).padStart(7)}  ${status} ${r.scenario.slice(0, 36).padEnd(36)}`);
  }

  if (results.length > 20) {
    console.log(`  … and ${results.length - 20} more`);
  }

  console.log(`\n  Done. ✓ ${ok} · ✗ ${fail}\n`);
}

main().catch((err) => {
  console.error("CLI error:", err);
  process.exit(1);
});
