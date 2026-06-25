import { createMemo, For, Show } from "solid-js";
import { useParams, A } from "@solidjs/router";
import { useApp } from "../data/store";
import { CO2Chart } from "../components/CO2Chart";
import { StatsPanel } from "../components/StatsPanel";

export function ResultsPage() {
  const params = useParams();
  const app = useApp();

  const result = createMemo(() => {
    return app.state.results.find((r) => r.id === params.id);
  });

  const r = result();

  if (!r) {
    return (
      <div class="text-center py-16 text-gray-500">
        <p class="text-lg">Result not found</p>
        <A href="/" class="text-emerald-400 hover:text-emerald-300 mt-2 inline-block">← Back to scenarios</A>
      </div>
    );
  }

  const inputsRows = [
    { label: "Model", value: r.model },
    { label: "Region", value: r.region },
    { label: "Years", value: r.historicalYears.join(", ") },
    { label: "Start Time", value: r.startTime },
    { label: "θ_pause", value: `${r.threshold} gCO₂/kWh` },
    { label: "θ_resume", value: `${r.hysteresisMargin} gCO₂/kWh` },
    { label: "Overhead Budget", value: `${r.overheadBudgetPct}%` },
  ];

  const resultsRows = [
    { label: "Wall Time", value: `${r.totalWallTimeH.toFixed(1)}`, unit: "h" },
    { label: "Training Time", value: `${r.trainingTimeH.toFixed(1)}`, unit: "h" },
    { label: "Paused Time", value: `${r.pausedTimeH.toFixed(1)}`, unit: "h", highlight: r.pausedTimeH > 0 },
    { label: "Checkpoint Overhead", value: `${r.checkpointOverheadH.toFixed(1)}`, unit: "h" },
    { label: "Total Energy", value: `${r.totalEnergyKwh.toFixed(0)}`, unit: "kWh" },
    { label: "Total Emissions", value: `${r.totalEmissionsKgco2.toFixed(0)}`, unit: "kg CO₂" },
    { label: "Num Pauses", value: `${r.numPauses}` },
    { label: "Overhead", value: `${r.actualOverheadPct.toFixed(1)}%` },
  ];

  const kpiRows = [
    { label: "CO₂ Savings", value: `${r.co2SavingsPct.toFixed(1)}%`, highlight: r.co2SavingsPct > 0 },
    { label: "Score", value: `${r.score.toFixed(1)}`, highlight: r.score > 1 },
    { label: "Status", value: r.completed ? "Completed" : r.stopReason, highlight: !r.completed },
    { label: "Within Budget", value: r.withinOverheadBudget ? "Yes" : "No", highlight: !r.withinOverheadBudget },
    { label: "Baseline Emissions", value: `${r.baselineEmissionsKgco2.toFixed(0)}`, unit: "kg CO₂" },
    { label: "Baseline Time", value: `${r.baselineTimeH.toFixed(1)}`, unit: "h" },
  ];

  return (
    <div>
      <div class="flex items-center justify-between mb-4">
        <div>
          <h1 class="text-xl font-bold text-white">{r.scenarioDescription}</h1>
          <A href="/" class="text-sm text-emerald-400 hover:text-emerald-300">← Back to Scenarios</A>
        </div>
      </div>

      <div class="grid gap-4 lg:grid-cols-2 mb-4">
        <div class="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
          <h2 class="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-3">Inputs</h2>
          <div class="space-y-1.5">
            <For each={inputsRows}>{(row) => (
              <div class="flex justify-between text-sm">
                <span class="text-gray-400">{row.label}</span>
                <span class="text-gray-200">{row.value}</span>
              </div>
            )}</For>
          </div>
        </div>

        <div class="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
          <h2 class="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-3">KPIs</h2>
          <StatsPanel rows={kpiRows} />
        </div>
      </div>

      <div class="rounded-xl border border-gray-800 bg-gray-900/60 p-4 mb-4">
        <h2 class="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-3">CO₂ Intensity During Simulation</h2>
        <div class="h-72">
          <CO2Chart
            labels={r.timestamps}
            co2Data={r.carbonIntensitySeries}
            thetaPause={r.threshold}
            thetaResume={r.hysteresisMargin}
          />
        </div>
      </div>

      <div class="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
        <h2 class="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-3">Detailed Results</h2>
        <StatsPanel rows={resultsRows} class="mb-4" />

        <Show when={r.issues.length > 0}>
          <div class="mt-3 space-y-1">
            <h3 class="text-xs text-amber-400 uppercase tracking-wide mb-1">Issues</h3>
            <For each={r.issues}>{(issue) => (
              <div class="text-xs text-amber-300">⚠ {issue}</div>
            )}</For>
          </div>
        </Show>
      </div>
    </div>
  );
}
