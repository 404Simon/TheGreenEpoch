import { createSignal, createMemo, Show, For, onCleanup } from "solid-js";
import { useParams, useSearchParams, A } from "@solidjs/router";
import { useApp } from "../data/store";
import { CO2Chart } from "../components/CO2Chart";
import { StatsPanel } from "../components/StatsPanel";
import { simulateStepwise } from "../domain/simulation";
import { buildSimResult } from "../domain/result";
import { hysteresisPolicy } from "../domain/policy";
import { runBaseline } from "../engine";
import type { SimResult, Scenario, FullProfile, SimConfig, CO2Timeline, SimProgress } from "../domain/types";

export function LiveSimPage() {
  const params = useParams();
  const [searchParams] = useSearchParams();
  const app = useApp();

  const scenario = createMemo(() => {
    const all = app.allScenarios();
    return all.find((s) => s.id === params.id);
  });

  const thresholdIdx = () => parseInt(searchParams.threshold || "0");
  const startIdx = () => parseInt(searchParams.start || "0");

  const [running, setRunning] = createSignal(false);
  const [finished, setFinished] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [progress, setProgress] = createSignal<SimProgress | null>(null);
  const [speed, setSpeed] = createSignal(100);
  const [labels, setLabels] = createSignal<string[]>([]);
  const [co2Points, setCo2Points] = createSignal<number[]>([]);
  const [stateSeries, setStateSeries] = createSignal<string[]>([]);
  const [emissionsSeries, setEmissionsSeries] = createSignal<number[]>([]);
  const [tokensRemainingSeries, setTokensRemainingSeries] = createSignal<number[]>([]);
  const [scrollMode, setScrollMode] = createSignal(true);
  const [windowSize, setWindowSize] = createSignal(300);
  const [simResult, setSimResult] = createSignal<SimResult | null>(null);
  const [alpha, setAlpha] = createSignal(1);

  let cancelFlag = false;

  const runSim = async () => {
    try {
      const sc = scenario();
      if (!sc) { setError("No scenario selected"); setRunning(false); return; }
      cancelFlag = false;
      setRunning(true);
      setFinished(false);
      setError(null);
      setSimResult(null);
      setLabels([]);
      setCo2Points([]);
      setEmissionsSeries([]);
      setTokensRemainingSeries([]);

      const pr = app.state.profiles;
      if (!pr) { setError("Data not loaded"); setRunning(false); return; }
      const p = pr[sc.model];
      if (!p) { setError("Unknown model: " + sc.model); setRunning(false); return; }

      const c = app.state.constants;
      if (!c) { setError("Constants not loaded"); setRunning(false); return; }
      const full: FullProfile = {
        ...p, gpuPowerTrain: c.gpu_power_train, gpuPowerPause: c.gpu_power_pause,
        pue: c.pue, checkpointPauseTime: c.checkpoint_pause_time, checkpointResumeTime: c.checkpoint_resume_time,
      };

      const tl = await app.getTimeline(sc.region, sc.historicalYears);
      const ts = tl.timestamps;
      if (!ts || ts.length < 2) { setError("No grid data for " + sc.region); setRunning(false); return; }

      const simConfig: SimConfig = {
        startTime: sc.startTimes[startIdx()],
        historicalYears: sc.historicalYears,
        overheadBudgetPct: sc.overheadBudgetPct,
      };

      const thetaPause = sc.thresholds[thresholdIdx()];
      const thetaResume = sc.hysteresis[thresholdIdx()];

      const gen = simulateStepwise(
        full, hysteresisPolicy(thetaPause, thetaResume), tl, simConfig,
      );

      const chartIntervalS = 3600;
      const maxChartPoints = 100_000;
      const labels: string[] = [];
      const co2PointsArr: number[] = [];
      const stateArr: string[] = [];
      const emissionsArr: number[] = [];
      const tokensRemainingArr: number[] = [];
      let nextChartWallS = 0;
      let lastProgress: SimProgress | null = null;

      const tick = () => {
        if (cancelFlag) return;
        const deadline = performance.now() + 10;
        const maxSteps = Math.min(Math.max(1, speed()), 1000);
        let steps = 0;

        while (steps < maxSteps) {
          const { value, done } = gen.next();
          if (done || !value) break;
          lastProgress = value;
          steps++;

          while (labels.length < maxChartPoints && lastProgress.totalWallS >= nextChartWallS) {
            labels.push(lastProgress.timestamp);
            co2PointsArr.push(lastProgress.carbonIntensity);
            stateArr.push(lastProgress.state);
            emissionsArr.push(lastProgress.totalEmissionsG / 1000);
            tokensRemainingArr.push(lastProgress.tokensRemaining);
            nextChartWallS += chartIntervalS;
          }

          if (lastProgress.done) break;
          if (performance.now() > deadline) break;
        }

        if (lastProgress && !lastProgress.done) {
          setProgress(lastProgress);
          setLabels(labels.slice());
          setCo2Points(co2PointsArr.slice());
          setStateSeries(stateArr.slice());
          setEmissionsSeries(emissionsArr.slice());
          setTokensRemainingSeries(tokensRemainingArr.slice());
          setTimeout(tick, 0);
        } else if (lastProgress) {
          setProgress(lastProgress);
          setLabels(labels.slice());
          setCo2Points(co2PointsArr.slice());
          setStateSeries(stateArr.slice());
          setEmissionsSeries(emissionsArr.slice());
          setTokensRemainingSeries(tokensRemainingArr.slice());
          setRunning(false);
          setFinished(true);
          const result = saveResult(lastProgress, labels, co2PointsArr, stateArr, emissionsArr, tokensRemainingArr, full, sc, thresholdIdx(), startIdx(), tl, app, alpha());
          if (result) setSimResult(result);
        }
      };

      setTimeout(tick, 50);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setRunning(false);
    }
  };

  onCleanup(() => {
    cancelFlag = true;
  });

  const sc = scenario();
  if (!sc) {
    return <div class="text-fg-muted text-center py-12">Scenario not found</div>;
  }

  const threshold = sc.thresholds[thresholdIdx()];
  const hysteresis = sc.hysteresis[thresholdIdx()];

  const stats = () => {
    const p = progress();
    if (!p) return [];
    return [
      { label: "Wall Time", value: fmtTime(p.totalWallS) },
      { label: "Training", value: fmtTime(p.trainingS) },
      { label: "Paused", value: fmtTime(p.pausedS), highlight: p.state === "paused" },
      { label: "Checkpoints", value: fmtTime(p.checkpointS) },
      { label: "Emissions", value: `${(p.totalEmissionsG / 1000).toFixed(1)}`, unit: "kg CO₂" },
      { label: "Energy", value: `${(p.totalEnergyWh / 1000).toFixed(1)}`, unit: "kWh" },
      { label: "Pauses", value: `${p.numPauses}` },
      { label: "Progress", value: `${((p.tokensTotal - p.tokensRemaining) / p.tokensTotal * 100).toFixed(1)}%` },
      { label: "State", value: p.state, highlight: true },
    ];
  };

  return (
    <div class="overflow-x-hidden">
      <div class="flex flex-col sm:flex-row items-start justify-between gap-3 mb-4">
        <div class="min-w-0">
          <h1 class="text-lg sm:text-xl font-semibold tracking-tight text-fg-primary">{sc.description}</h1>
          <p class="text-sm text-fg-muted mt-0.5 break-words">
            {sc.region} &middot; &#952;pause={threshold} &middot; &#952;resume={hysteresis} &middot; start {sc.startTimes[startIdx()]}
          </p>
        </div>
        <div class="flex items-center gap-2 shrink-0 flex-wrap">
          <select
            value={speed()}
            onChange={(e) => setSpeed(Number(e.currentTarget.value))}
            class="px-2.5 py-1.5 rounded-lg bg-surface-2 border border-border-default text-fg-primary text-sm focus:border-accent transition-colors"
          >
            <option value={10}>10&times;</option>
            <option value={100}>100&times;</option>
            <option value={500}>500&times;</option>
            <option value={1000}>1K&times;</option>
          </select>

          <label class="flex items-center gap-1.5 text-xs text-fg-muted cursor-pointer select-none">
            <input type="checkbox" checked={scrollMode()} onChange={(e) => setScrollMode(e.currentTarget.checked)} class="accent-accent" />
            Scroll
          </label>

          <Show when={scrollMode()}>
            <input type="range" min={50} max={2000} value={windowSize()} onInput={(e) => setWindowSize(Number(e.currentTarget.value))}
              class="w-20 h-1 accent-accent cursor-pointer" title={`Window: ${windowSize()} pts`} />
            <span class="text-xs text-fg-muted w-8 tabular-nums">{windowSize()}</span>
          </Show>

          <div class="flex items-center gap-1.5">
            <label class="text-xs text-fg-muted whitespace-nowrap">{"\u03B1"} (CO₂ weight):</label>
            <input
              type="number" value={alpha()}
              onInput={e => setAlpha(+e.currentTarget.value || 0)}
              step="0.1" min="0" max="1"
              class="w-16 bg-surface-2 border border-border-default/50 rounded px-2 py-1.5 text-sm text-fg-body tabular-nums focus:outline-none focus:border-accent"
            />
          </div>

          <button
            onClick={runSim}
            disabled={running()}
            class="px-4 py-2 rounded-lg bg-accent text-fg-primary text-sm font-medium hover:bg-accent/90 active:scale-[0.97] disabled:opacity-40 disabled:active:scale-100 transition-all"
          >
            {running() ? "Running..." : finished() ? "Re-run" : "Play"}
          </button>
        </div>
      </div>

      <Show when={error()}>
        <div class="mb-4 rounded-lg border border-alert-red/20 bg-alert-red-bg px-4 py-2.5 text-sm text-alert-red">{error()}</div>
      </Show>

      <div class="grid gap-5 lg:grid-cols-3">
        <div class="lg:col-span-2">
          <div class="rounded-xl bg-surface-2 border border-border-default/60 p-3">
            <div class="h-72">
              <CO2Chart
                labels={labels()}
                co2Data={co2Points()}
                stateSeries={stateSeries()}
                thetaPause={threshold}
                thetaResume={hysteresis}
                windowSize={scrollMode() ? windowSize() : 0}
              />
            </div>
          </div>
        </div>

        <div>
          <div class="rounded-xl bg-surface-2 border border-border-default/60 p-5">
            <h2 class="text-xs font-semibold text-fg-subtle tracking-wide mb-3">Live stats</h2>
            <div class="grid grid-cols-1 gap-1.5">
              <For each={stats()}>
                {(row) => (
                  <div class={`flex justify-between rounded-lg px-3 py-2 text-sm ${
                    row.highlight
                      ? "bg-accent-subtle border border-accent/20"
                      : "bg-surface/40"
                  }`}>
                    <span class="text-fg-muted">{row.label}</span>
                    <span class="text-fg-primary font-medium tabular-nums">
                      {row.value}{row.unit && <span class="text-fg-muted font-normal ml-1">{row.unit}</span>}
                    </span>
                  </div>
                )}
              </For>
            </div>
            <Show when={(() => { const p = progress(); return p && Array.isArray(p.issues) && p.issues.length > 0; })()}>
              <div class="mt-4 space-y-1 pt-3 border-t border-border-default/50">
                <For each={progress()!.issues}>{(issue) => (
                  <div class="text-xs text-alert-amber/80">{issue}</div>
                )}</For>
              </div>
            </Show>
          </div>
        </div>
      </div>

      <Show when={finished() && simResult()}>
        {(r) => {
          const result = r();
          const statusColor = result.ok ? "bg-accent" : result.completed ? "bg-alert-amber" : "bg-alert-red";
          const statusText = result.ok ? "OK" : result.completed ? "Completed" : result.stopReason;

          const kpiRows = [
            { label: "CO₂ Savings", value: `${result.co2SavingsPct.toFixed(1)}%`, highlight: result.co2SavingsPct > 0 },
            { label: "Score", value: `${result.score.toFixed(1)}`, highlight: result.score > 1 },
            { label: "Status", value: statusText, highlight: !result.ok },
            { label: "Within Budget", value: result.withinOverheadBudget ? "Yes" : "No", highlight: !result.withinOverheadBudget },
            { label: "Completion", value: `${result.completionPct.toFixed(1)}%` },
            { label: "Baseline Emissions", value: `${result.baselineEmissionsKgco2.toFixed(1)}`, unit: "kg CO₂" },
            { label: "Baseline Time", value: `${result.baselineTimeH.toFixed(1)}`, unit: "h" },
          ];

          const timeRows = [
            { label: "Wall Time", value: `${result.totalWallTimeH.toFixed(2)}`, unit: "h" },
            { label: "Training Time", value: `${result.trainingTimeH.toFixed(2)}`, unit: "h" },
            { label: "Paused Time", value: `${result.pausedTimeH.toFixed(2)}`, unit: "h", highlight: result.pausedTimeH > 0 },
            { label: "Checkpoint Overhead", value: `${result.checkpointOverheadH.toFixed(2)}`, unit: "h" },
            { label: "Idle Time", value: `${result.idleTimeH.toFixed(2)}`, unit: "h", highlight: result.idleTimeH > 0 },
          ];

          const energyRows = [
            { label: "Total Energy", value: `${result.totalEnergyKwh.toFixed(1)}`, unit: "kWh" },
            { label: "Training Energy", value: `${result.trainingEnergyKwh.toFixed(1)}`, unit: "kWh" },
            { label: "Paused Energy", value: `${result.pausedEnergyKwh.toFixed(1)}`, unit: "kWh", highlight: result.pausedEnergyKwh > 0 },
            { label: "Checkpoint Energy", value: `${result.checkpointEnergyKwh.toFixed(1)}`, unit: "kWh" },
            { label: "Total Emissions", value: `${result.totalEmissionsKgco2.toFixed(1)}`, unit: "kg CO₂" },
          ];

          const progressRows = [
            { label: "Tokens Processed", value: `${result.tokensProcessed.toLocaleString()}` },
            { label: "Tokens Total", value: `${result.tokensTotal.toLocaleString()}` },
            { label: "Completion", value: `${result.completionPct.toFixed(1)}%` },
            { label: "Num Pauses", value: `${result.numPauses}` },
            { label: "Overhead", value: `${result.actualOverheadPct.toFixed(1)}%` },
            { label: "Overhead Budget", value: `${result.overheadBudgetPct}%` },
          ];

          const inputsRows = [
            { label: "Model", value: result.model },
            { label: "Region", value: result.region },
            { label: "Years", value: result.historicalYears.join(", ") },
            { label: "Start Time", value: result.startTime },
            { label: "θ_pause", value: `${result.threshold} gCO₂/kWh` },
            { label: "θ_resume", value: `${result.hysteresisMargin} gCO₂/kWh` },
            { label: "Overhead Budget", value: `${result.overheadBudgetPct}%` },
          ];

          return (
            <div class="mt-8 space-y-5">
              <div class="flex items-center justify-between">
                <h2 class="text-lg font-semibold tracking-tight text-fg-primary">Simulation results</h2>
                <span class={`px-2 py-0.5 rounded text-xs font-semibold text-white ${statusColor}`}>
                  {statusText}
                </span>
              </div>

              <div class="grid gap-5 lg:grid-cols-2">
                <div class="rounded-xl bg-surface-2 border border-border-default/60 p-5">
                  <h3 class="text-xs font-semibold text-fg-subtle tracking-wide mb-3">KPIs</h3>
                  <StatsPanel rows={kpiRows} />
                </div>
                <div class="rounded-xl bg-surface-2 border border-border-default/60 p-5">
                  <h3 class="text-xs font-semibold text-fg-subtle tracking-wide mb-3">Inputs</h3>
                  <div class="space-y-1.5">
                    <For each={inputsRows}>{(row) => (
                      <div class="flex justify-between text-sm">
                        <span class="text-fg-muted">{row.label}</span>
                        <span class="text-fg-body tabular-nums">{row.value}</span>
                      </div>
                    )}</For>
                  </div>
                </div>
              </div>

              <div class="grid gap-5 lg:grid-cols-3">
                <div class="rounded-xl bg-surface-2 border border-border-default/60 p-5">
                  <h3 class="text-xs font-semibold text-fg-subtle tracking-wide mb-3">Time</h3>
                  <StatsPanel rows={timeRows} />
                </div>
                <div class="rounded-xl bg-surface-2 border border-border-default/60 p-5">
                  <h3 class="text-xs font-semibold text-fg-subtle tracking-wide mb-3">Energy &amp; emissions</h3>
                  <StatsPanel rows={energyRows} />
                </div>
                <div class="rounded-xl bg-surface-2 border border-border-default/60 p-5">
                  <h3 class="text-xs font-semibold text-fg-subtle tracking-wide mb-3">Progress</h3>
                  <StatsPanel rows={progressRows} />
                </div>
              </div>

              <Show when={result.issues.length > 0}>
                <div class="rounded-xl border border-alert-red/20 bg-alert-red-bg p-5">
                  <h3 class="text-xs font-semibold text-alert-red tracking-wide mb-2">Issues</h3>
                  <div class="space-y-1">
                    <For each={result.issues}>{(issue) => (
                      <div class="text-xs text-alert-red/80">{issue}</div>
                    )}</For>
                    <Show when={result.stopReason}>
                      <div class="text-xs text-alert-red/60 mt-1">Stop reason: {result.stopReason}</div>
                    </Show>
                  </div>
                </div>
              </Show>
            </div>
          );
        }}
      </Show>
    </div>
  );
}

function saveResult(
  lastP: SimProgress, allLabels: string[], allCo2: number[], allStates: string[], allEmissions: number[], allTokensRemaining: number[],
  full: FullProfile, sc: Scenario, ti: number, si: number, tl: CO2Timeline,
  app: ReturnType<typeof useApp>,
  alpha: number = 1,
): SimResult | null {
  try {
    const simConfig: SimConfig = {
      startTime: sc.startTimes[si],
      historicalYears: sc.historicalYears,
      overheadBudgetPct: sc.overheadBudgetPct,
    };
    const thetaPause = sc.thresholds[ti];
    const thetaResume = sc.hysteresis[ti];
    const baseLast = runBaseline(full, tl, simConfig);

    const result = buildSimResult(full, simConfig, lastP, baseLast, thetaPause, thetaResume, {
      id: `result-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      scenarioDescription: sc.description,
      model: full.name,
      region: sc.region,
      timestamps: allLabels,
      carbonIntensitySeries: allCo2,
      stateSeries: allStates,
      emissionsSeries: allEmissions,
      tokensRemainingSeries: allTokensRemaining,
    }, alpha);
    app.addResult(result);
    return result;
  } catch (e) {
    console.error("saveResult:", e);
    return null;
  }
}

// SimStepper class removed in favor of simulateStepwise generator
// with chunked execution. See runSim() above for the replacement.

function fmtTime(s: number): string {
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${(s / 60).toFixed(1)}m`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m`;
}
