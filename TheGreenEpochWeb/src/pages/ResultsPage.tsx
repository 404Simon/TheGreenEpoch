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
      <div class="text-center py-20 text-fg-muted">
        <div class="text-4xl mb-4 opacity-20">⚡</div>
        <p class="text-lg">Result not found</p>
        <p class="text-sm text-fg-muted mt-1">This simulation result may have been cleared or does not exist.</p>
        <A href="/" class="mt-4 inline-flex items-center gap-1.5 text-sm text-accent hover:text-accent/80 transition-colors">
          &larr; Back to scenarios
        </A>
      </div>
    );
  }

  const inputsRows = [
    { label: "Model", value: r.model },
    { label: "Region", value: r.region },
    { label: "Years", value: r.historicalYears.join(", ") },
    { label: "Start time", value: r.startTime },
    { label: "θ_pause", value: `${r.threshold} gCO₂/kWh` },
    { label: "θ_resume", value: `${r.hysteresisMargin} gCO₂/kWh` },
    { label: "Overhead budget", value: `${r.overheadBudgetPct}%` },
  ];

  const resultsRows = [
    { label: "Wall time", value: `${r.totalWallTimeH.toFixed(1)}`, unit: "h" },
    { label: "Training time", value: `${r.trainingTimeH.toFixed(1)}`, unit: "h" },
    { label: "Paused time", value: `${r.pausedTimeH.toFixed(1)}`, unit: "h", highlight: r.pausedTimeH > 0 },
    { label: "Checkpoint overhead", value: `${r.checkpointOverheadH.toFixed(1)}`, unit: "h" },
    { label: "Total energy", value: `${r.totalEnergyKwh.toFixed(0)}`, unit: "kWh" },
    { label: "Total emissions", value: `${r.totalEmissionsKgco2.toFixed(0)}`, unit: "kg CO₂" },
    { label: "Num pauses", value: `${r.numPauses}` },
    { label: "Overhead", value: `${r.actualOverheadPct.toFixed(1)}%` },
  ];

  const kpiRows = [
    { label: "CO₂ savings", value: `${r.co2SavingsPct.toFixed(1)}%`, highlight: r.co2SavingsPct > 0 },
    { label: "Score", value: `${r.score.toFixed(1)}`, highlight: r.score > 1 },
    { label: "Status", value: r.completed ? "Completed" : r.stopReason, highlight: !r.completed },
    { label: "Within budget", value: r.withinOverheadBudget ? "Yes" : "No", highlight: !r.withinOverheadBudget },
    { label: "Baseline emissions", value: `${r.baselineEmissionsKgco2.toFixed(0)}`, unit: "kg CO₂" },
    { label: "Baseline time", value: `${r.baselineTimeH.toFixed(1)}`, unit: "h" },
  ];

  return (
    <div>
      <div class="flex items-center justify-between mb-6">
        <div>
          <h1 class="text-xl font-semibold tracking-tight text-fg-primary">{r.scenarioDescription}</h1>
          <A href="/" class="text-sm text-accent hover:text-accent/80 inline-flex items-center gap-1 mt-0.5 transition-colors">
            &larr; Back to scenarios
          </A>
        </div>
      </div>

      <div class="grid gap-5 lg:grid-cols-2 mb-5">
        <div class="rounded-xl bg-surface-2 border border-border-default/60 p-5">
          <h2 class="text-xs font-semibold text-fg-subtle tracking-wide mb-3">Inputs</h2>
          <div class="space-y-1.5">
            <For each={inputsRows}>{(row) => (
              <div class="flex justify-between text-sm">
                <span class="text-fg-muted">{row.label}</span>
                <span class="text-fg-body tabular-nums">{row.value}</span>
              </div>
            )}</For>
          </div>
        </div>

        <div class="rounded-xl bg-surface-2 border border-border-default/60 p-5">
          <h2 class="text-xs font-semibold text-fg-subtle tracking-wide mb-3">KPIs</h2>
          <StatsPanel rows={kpiRows} />
        </div>
      </div>

      <div class="rounded-xl bg-surface-2 border border-border-default/60 p-4 mb-5">
        <h2 class="text-xs font-semibold text-fg-subtle tracking-wide mb-3">CO₂ intensity during simulation</h2>
        <div class="h-72">
          <CO2Chart
            labels={r.timestamps}
            co2Data={r.carbonIntensitySeries}
            thetaPause={r.threshold}
            thetaResume={r.hysteresisMargin}
          />
        </div>
      </div>

      <div class="rounded-xl bg-surface-2 border border-border-default/60 p-5">
        <h2 class="text-xs font-semibold text-fg-subtle tracking-wide mb-3">Detailed results</h2>
        <StatsPanel rows={resultsRows} class="mb-4" />

        <Show when={r.issues.length > 0}>
          <div class="mt-4 pt-3 border-t border-border-default/50 space-y-1">
            <h3 class="text-xs text-alert-amber/80 tracking-wide mb-1">Issues</h3>
            <For each={r.issues}>{(issue) => (
              <div class="text-xs text-alert-amber/70">{issue}</div>
            )}</For>
          </div>
        </Show>
      </div>
    </div>
  );
}
