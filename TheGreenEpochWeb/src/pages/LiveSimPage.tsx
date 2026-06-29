import { createSignal, createMemo, Show, For, onCleanup } from "solid-js";
import { useParams, useSearchParams, A } from "@solidjs/router";
import { useApp } from "../data/store";
import { CO2Chart } from "../components/CO2Chart";
import { StatsPanel } from "../components/StatsPanel";
import { tokensPerSecond, energyWh, emissionsG, simulateStepwise } from "../data/simulation";
import { loadCO2Timeline } from "../data/loadData";
import type { SimResult, FullProfile, SimConfig, CO2Timeline } from "../types";

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
  const [progress, setProgress] = createSignal<any>(null);
  const [speed, setSpeed] = createSignal(100);
  const [labels, setLabels] = createSignal<string[]>([]);
  const [co2Points, setCo2Points] = createSignal<number[]>([]);
  const [stateSeries, setStateSeries] = createSignal<string[]>([]);
  const [emissionsSeries, setEmissionsSeries] = createSignal<number[]>([]);
  const [tokensRemainingSeries, setTokensRemainingSeries] = createSignal<number[]>([]);
  const [scrollMode, setScrollMode] = createSignal(true);
  const [windowSize, setWindowSize] = createSignal(300);
  const [simResult, setSimResult] = createSignal<SimResult | null>(null);

  let cancelFlag = false;
  let stepper: SimStepper | null = null;

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

      let tl = app.state.co2Cache[sc.region];
      if (!tl) tl = await loadCO2Timeline(sc.region);
      const ts = tl?.timestamps;
      if (!ts || ts.length < 2) { setError("No grid data for " + sc.region); setRunning(false); return; }

      stepper = new SimStepper(full, sc, thresholdIdx(), startIdx(), tl, () => speed());

      stepper.run(
        (p: any, chartLabels: string[], chartCo2: number[], chartStates: string[], chartEmissions: number[], chartTokensRemaining: number[]) => {
          if (cancelFlag) return;
          setProgress(p);
          if (chartLabels.length > 0) {
            setLabels(chartLabels.slice());
            setCo2Points(chartCo2.slice());
            setStateSeries(chartStates.slice());
            setEmissionsSeries(chartEmissions.slice());
            setTokensRemainingSeries(chartTokensRemaining.slice());
          }
        },
        (lastP: any, allLabels: string[], allCo2: number[], allStates: string[], allEmissions: number[], allTokensRemaining: number[]) => {
          stepper = null;
          setProgress(lastP);
          setLabels(allLabels.slice());
          setCo2Points(allCo2.slice());
          setStateSeries(allStates.slice());
          setEmissionsSeries(allEmissions.slice());
          setTokensRemainingSeries(allTokensRemaining.slice());
          setRunning(false);
          setFinished(true);
          const result = saveResult(lastP, allLabels, allCo2, allEmissions, allTokensRemaining, full, sc, thresholdIdx(), startIdx(), tl, app);
          if (result) setSimResult(result);
        },
        () => cancelFlag,
      );
    } catch (err: any) {
      setError(err?.message || String(err));
      setRunning(false);
    }
  };

  onCleanup(() => {
    cancelFlag = true;
    stepper = null;
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
    <div>
      <div class="flex items-start justify-between gap-4 mb-6">
        <div>
          <h1 class="text-xl font-semibold tracking-tight text-fg-primary">{sc.description}</h1>
          <p class="text-sm text-fg-muted mt-0.5">
            {sc.region} &middot; &#952;pause={threshold} &middot; &#952;resume={hysteresis} &middot; start {sc.startTimes[startIdx()]}
          </p>
        </div>
        <div class="flex items-center gap-2.5 shrink-0">
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
  lastP: any, allLabels: string[], allCo2: number[], allEmissions: number[], allTokensRemaining: number[],
  full: FullProfile, sc: any, ti: number, si: number, tl: CO2Timeline,
  app: ReturnType<typeof useApp>,
): SimResult | null {
  try {
    const cfg: SimConfig = {
      scenarioDescription: sc.description, region: sc.region,
      historicalYears: sc.historicalYears, startTime: sc.startTimes[si],
      thetaPause: sc.thresholds[ti], thetaResume: sc.hysteresis[ti],
      overheadBudgetPct: sc.overheadBudgetPct,
    };
    const baseCfg: SimConfig = { ...cfg, thetaPause: Infinity, thetaResume: 0 };
    let baseLast: any = null;
    for (const p of simulateStepwise(full, baseCfg, tl)) {
      baseLast = p;
    }

    const tps = tokensPerSecond(full.gpuCount) || 1;
    const idealS = lastP.tokensTotal / tps;
    const overheadS = lastP.pausedS + lastP.checkpointS;
    const actualOverheadPct = 100 * overheadS / (idealS || 1);
    const baselineEm = baseLast ? baseLast.totalEmissionsG / 1000 : 0;
    const totalEm = lastP.totalEmissionsG / 1000;
    const co2SavingsPct = baselineEm > 0 ? (baselineEm - totalEm) / baselineEm * 100 : 0;
    const score = co2SavingsPct / Math.max(actualOverheadPct, 0.001);
    const tokensProcessed = lastP.tokensTotal - lastP.tokensRemaining;
    const totalWallTimeH = lastP.totalWallS / 3600;
    const pausedTimeH = lastP.pausedS / 3600;
    const checkpointOverheadH = lastP.checkpointS / 3600;
    const idleTimeH = pausedTimeH + checkpointOverheadH;

    const result: SimResult = {
      id: `result-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      scenarioDescription: cfg.scenarioDescription, model: full.name,
      region: cfg.region, historicalYears: cfg.historicalYears,
      startTime: cfg.startTime, threshold: cfg.thetaPause,
      hysteresisMargin: cfg.thetaResume,
      totalWallTimeH,
      trainingTimeH: lastP.trainingS / 3600,
      pausedTimeH,
      checkpointOverheadH,
      totalEnergyKwh: lastP.totalEnergyWh / 1000,
      trainingEnergyKwh: lastP.trainingEnergyWh / 1000,
      pausedEnergyKwh: lastP.pausedEnergyWh / 1000,
      checkpointEnergyKwh: lastP.checkpointEnergyWh / 1000,
      totalEmissionsKgco2: totalEm,
      tokensProcessed,
      tokensTotal: lastP.tokensTotal,
      completed: lastP.tokensRemaining <= 0,
      numPauses: lastP.numPauses,
      overheadBudgetPct: cfg.overheadBudgetPct,
      actualOverheadPct: Math.round(actualOverheadPct * 100) / 100,
      withinOverheadBudget: overheadS / idealS <= cfg.overheadBudgetPct / 100,
      timestamps: allLabels, carbonIntensitySeries: allCo2, stateSeries: [],
      emissionsSeries: allEmissions,
      tokensRemainingSeries: allTokensRemaining,
      issues: lastP.issues, stopReason: lastP.stopReason,
      baselineEmissionsKgco2: baselineEm,
      baselineTimeH: baseLast ? baseLast.totalWallS / 3600 : 0,
      co2SavingsPct: Math.round(co2SavingsPct * 100) / 100,
      score: Math.round(score * 100) / 100,
      idleTimeH,
      completionPct: lastP.tokensTotal > 0 ? 100.0 * tokensProcessed / lastP.tokensTotal : 0,
      ok: lastP.tokensRemaining <= 0 && overheadS / idealS <= cfg.overheadBudgetPct / 100 && !(lastP.issues && lastP.issues.length > 0),
    };
    app.addResult(result);
    return result;
  } catch (e) {
    console.error("saveResult:", e);
    return null;
  }
}

// ---------------------------------------------------------------------------
// SimStepper - pure simulation engine with time-based chart sampling
// ---------------------------------------------------------------------------

class SimStepper {
  private idx = 0;
  private nPoints = 0;
  private carbon: number[] = [];
  private timestamps: string[] = [];
  private stepS = 0;
  private meanCo2 = 0;

  private tps = 0;
  private trainPowerW = 0;
  private pausePowerW = 0;
  private ckptPowerW = 0;

  private state: "running" | "paused" = "running";
  private transitionTimerS = 0;
  private targetAfterTransition: "running" | "paused" | null = null;

  private tokensTotal = 0;
  private tokensRemaining = 0;
  private idealTrainingS = 0;

  private totalWallS = 0;
  private trainingS = 0;
  private pausedS = 0;
  private checkpointS = 0;
  private totalEnergyWh = 0;
  private trainingEnergyWh = 0;
  private pausedEnergyWh = 0;
  private checkpointEnergyWh = 0;
  private totalEmissionsG = 0;
  private numPauses = 0;
  issues: string[] = [];
  private nanFallbacks = 0;

  private thetaPause = 0;
  private thetaResume = 0;
  private overheadBudgetPct = 0;

  // Chart data - sampled at fixed simulated-time intervals
  private chartLabels: string[] = [];
  private chartCo2: number[] = [];
  private chartStates: string[] = [];
  private chartEmissions: number[] = [];
  private chartTokensRemaining: number[] = [];
  private chartIntervalS = 3600;
  private nextChartWallS = 0;
  private maxChartPoints = 100_000;

  private getSpeed: () => number = () => 100;

  constructor(
    private profile: FullProfile,
    scenario: any,
    thresholdIdx: number,
    startIdx: number,
    tl: CO2Timeline,
    getSpeed: () => number,
  ) {
    this.getSpeed = getSpeed;
    if (!tl) throw new Error("SimStepper: tl is null");
    const tsArr = tl.timestamps;
    const co2Arr = tl.carbonIntensity;
    if (!tsArr || !co2Arr) throw new Error("SimStepper: missing timestamps or carbonIntensity");
    if (tsArr.length < 2 || co2Arr.length < 2) throw new Error("SimStepper: timeline too short");
    this.carbon = co2Arr;
    this.timestamps = tsArr;
    this.nPoints = tsArr.length;

    const diffMs = new Date(tsArr[1]).getTime() - new Date(tsArr[0]).getTime();
    if (!diffMs || diffMs <= 0) throw new Error("SimStepper: invalid step size");
    this.stepS = diffMs / 1000;

    let sum = 0;
    for (let i = 0; i < co2Arr.length; i++) sum += co2Arr[i];
    this.meanCo2 = sum / co2Arr.length;

    this.tps = tokensPerSecond(profile.gpuCount);
    this.trainPowerW = profile.gpuCount * profile.gpuPowerTrain * profile.pue;
    this.pausePowerW = profile.gpuCount * profile.gpuPowerPause * profile.pue;
    this.ckptPowerW = this.trainPowerW;

    this.thetaPause = scenario.thresholds[thresholdIdx];
    this.thetaResume = scenario.hysteresis[thresholdIdx];
    this.overheadBudgetPct = scenario.overheadBudgetPct;

    this.tokensTotal = profile.datasetTokens;
    this.tokensRemaining = this.tokensTotal;
    this.idealTrainingS = this.tokensTotal / (this.tps || 1);

    const startTs = scenario.startTimes[startIdx];
    const baseYear = new Date(tsArr[0]).getUTCFullYear();
    const [sm, sd] = startTs.split("-").map(Number);
    const targetStr = new Date(Date.UTC(baseYear, sm - 1, sd)).toISOString();
    let lo = 0, hi = this.nPoints;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (tsArr[mid] < targetStr) lo = mid + 1; else hi = mid; }
    this.idx = Math.min(lo, this.nPoints - 1);

    // Sample the initial point
    this.sampleChart();

    const initCo2 = this.getCo2(this.idx);
    if (initCo2 > this.thetaPause) {
      if (profile.checkpointPauseTime > 0) {
        this.transitionTimerS = profile.checkpointPauseTime;
        this.targetAfterTransition = "paused";
      } else {
        this.state = "paused";
      }
      this.numPauses++;
    }
  }

  getDebug() {
    return { nextChartWallS: this.nextChartWallS, totalWallS: this.totalWallS, chartLen: this.chartLabels.length, tokensRemaining: this.tokensRemaining };
  }

  private sampleChart() {
    if (this.chartLabels.length >= this.maxChartPoints) return;
    this.chartLabels.push(this.timestamps[this.idx]);
    this.chartCo2.push(this.getCo2(this.idx));
    this.chartStates.push(this.state);
    this.chartEmissions.push(this.totalEmissionsG / 1000);
    this.chartTokensRemaining.push(this.tokensRemaining);
  }

  private getCo2(i: number): number {
    const v = this.carbon[i];
    if (!isFinite(v)) { this.nanFallbacks++; return this.meanCo2; }
    return v;
  }

  run(
    onProgress: (p: any, chartLabels: string[], chartCo2: number[], chartStates: string[], chartEmissions: number[], chartTokensRemaining: number[]) => void,
    onDone: (last: any, labels: string[], co2: number[], states: string[], emissions: number[], tokensRemaining: number[]) => void,
    isCancelled: () => boolean,
  ) {
    const tick = () => {
      try {
        if (isCancelled()) return;
        const deadline = performance.now() + 10;
        let stepsThisChunk = 0;
        const maxPerChunk = Math.min(Math.max(1, typeof this.getSpeed === "function" ? this.getSpeed() : 100), 1000);

        while (this.tokensRemaining > 0 && !isCancelled()) {
          stepsThisChunk++;
          const co2 = this.getCo2(this.idx);
          let dtS = this.stepS;

          // Checkpoint transition
          if (this.transitionTimerS > 0) {
            const spentS = Math.min(dtS, this.transitionTimerS);
            this.transitionTimerS -= spentS;
            const eWh = energyWh(this.ckptPowerW, spentS);
            const emG = emissionsG(eWh, co2);
            this.checkpointS += spentS;
            this.checkpointEnergyWh += eWh;
            this.totalEnergyWh += eWh;
            this.totalEmissionsG += emG;
            this.totalWallS += spentS;
            dtS -= spentS;
            if (this.transitionTimerS <= 0 && this.targetAfterTransition) {
              this.state = this.targetAfterTransition;
              this.targetAfterTransition = null;
            }
            if (dtS <= 0) {
              this.advance();
              this.maybeYield(onProgress);
              if (stepsThisChunk >= maxPerChunk || performance.now() > deadline) {
                setTimeout(tick, 0); return;
              }
              continue;
            }
          }

          // Policy
          const shouldPause = !this.isPaused() && co2 > this.thetaPause;
          const shouldResume = this.isPaused() && co2 < this.thetaResume;

          if (shouldPause) {
            const ckpt = this.profile.checkpointPauseTime;
            if ((this.pausedS + this.checkpointS + ckpt) / this.idealTrainingS > this.overheadBudgetPct / 100) {
              this.issues.push("Overhead would exceed budget - blocking pause");
              break;
            }
            this.numPauses++;
            if (ckpt <= 0) this.state = "paused";
            else { this.transitionTimerS = ckpt; this.targetAfterTransition = "paused"; }
            this.advance();
            this.maybeYield(onProgress);
            if (stepsThisChunk >= maxPerChunk || performance.now() > deadline) {
              setTimeout(tick, 0); return;
            }
            continue;
          }

          if (shouldResume) {
            const ckpt = this.profile.checkpointResumeTime;
            if ((this.pausedS + this.checkpointS + ckpt) / this.idealTrainingS > this.overheadBudgetPct / 100) {
              this.issues.push("Overhead would exceed budget - blocking resume");
              break;
            }
            if (ckpt <= 0) this.state = "running";
            else { this.transitionTimerS = ckpt; this.targetAfterTransition = "running"; }
            this.advance();
            this.maybeYield(onProgress);
            if (stepsThisChunk >= maxPerChunk || performance.now() > deadline) {
              setTimeout(tick, 0); return;
            }
            continue;
          }

          // Spend dtS in current state
          if (this.state === "running") {
            const maxT = Math.floor(this.tps * dtS);
            const tokensStep = Math.min(maxT, this.tokensRemaining);
            const effectiveS = tokensStep >= maxT ? dtS : tokensStep / this.tps;
            this.tokensRemaining -= tokensStep;
            const eWh = energyWh(this.trainPowerW, effectiveS);
            const emG = emissionsG(eWh, co2);
            this.trainingS += effectiveS;
            this.trainingEnergyWh += eWh;
            this.totalEnergyWh += eWh;
            this.totalEmissionsG += emG;
            this.totalWallS += effectiveS;
            const idleS = dtS - effectiveS;
            if (idleS > 0) {
              const iWh = energyWh(this.pausePowerW, idleS);
              this.pausedS += idleS;
              this.pausedEnergyWh += iWh;
              this.totalEnergyWh += iWh;
              this.totalEmissionsG += emissionsG(iWh, co2);
              this.totalWallS += idleS;
            }
          } else {
            const eWh = energyWh(this.pausePowerW, dtS);
            const emG = emissionsG(eWh, co2);
            this.pausedS += dtS;
            this.pausedEnergyWh += eWh;
            this.totalEnergyWh += eWh;
            this.totalEmissionsG += emG;
            this.totalWallS += dtS;
          }

          if ((this.pausedS + this.checkpointS) / this.idealTrainingS > this.overheadBudgetPct / 100) {
            this.issues.push("Overhead exceeded budget");
            break;
          }

          this.advance();

          if (stepsThisChunk >= maxPerChunk || performance.now() > deadline) {
            this.maybeYield(onProgress);
            setTimeout(tick, 0);
            return;
          }
        }

        const stopReason = this.tokensRemaining <= 0 ? "completed"
          : (this.pausedS + this.checkpointS) / this.idealTrainingS > this.overheadBudgetPct / 100 ? "budget_exceeded"
            : "iteration_limit";

        onDone(this.snap(stopReason), this.chartLabels, this.chartCo2, this.chartStates, this.chartEmissions, this.chartTokensRemaining);
      } catch (err: any) {
        console.error("SimStepper error:", err);
        onDone({ ...this.snap("error"), issues: ["Simulation error: " + (err?.message || String(err))] }, this.chartLabels, this.chartCo2, this.chartStates, this.chartEmissions, this.chartTokensRemaining);
      }
    };

    setTimeout(tick, 50);
  }

  private isPaused() { return this.state === "paused"; }

  private advance() {
    this.idx = (this.idx + 1) % this.nPoints;
    // Sample chart data at fixed simulated-time intervals
    while (this.chartLabels.length < this.maxChartPoints && this.totalWallS >= this.nextChartWallS) {
      this.sampleChart();
      this.nextChartWallS += this.chartIntervalS;
    }
  }

  private snap(stopReason: string) {
    return {
      timestamp: this.timestamps[this.idx], carbonIntensity: this.getCo2(this.idx),
      state: this.state, tokensRemaining: this.tokensRemaining, tokensTotal: this.tokensTotal,
      totalWallS: Math.round(this.totalWallS * 100) / 100,
      trainingS: Math.round(this.trainingS * 100) / 100,
      pausedS: Math.round(this.pausedS * 100) / 100,
      checkpointS: Math.round(this.checkpointS * 100) / 100,
      totalEnergyWh: Math.round(this.totalEnergyWh * 100) / 100,
      trainingEnergyWh: Math.round(this.trainingEnergyWh * 100) / 100,
      pausedEnergyWh: Math.round(this.pausedEnergyWh * 100) / 100,
      checkpointEnergyWh: Math.round(this.checkpointEnergyWh * 100) / 100,
      totalEmissionsG: Math.round(this.totalEmissionsG * 100) / 100,
      numPauses: this.numPauses, done: true, stopReason,
      issues: [...this.issues], nanFallbacks: this.nanFallbacks,
    };
  }

  private maybeYield(onProgress: (p: any, chartLabels: string[], chartCo2: number[], chartStates: string[], chartEmissions: number[], chartTokensRemaining: number[]) => void) {
    onProgress({
      timestamp: this.timestamps[this.idx], carbonIntensity: this.getCo2(this.idx),
      state: this.state, tokensRemaining: this.tokensRemaining, tokensTotal: this.tokensTotal,
      totalWallS: Math.round(this.totalWallS * 100) / 100,
      trainingS: Math.round(this.trainingS * 100) / 100,
      pausedS: Math.round(this.pausedS * 100) / 100,
      checkpointS: Math.round(this.checkpointS * 100) / 100,
      totalEnergyWh: Math.round(this.totalEnergyWh * 100) / 100,
      trainingEnergyWh: Math.round(this.trainingEnergyWh * 100) / 100,
      pausedEnergyWh: Math.round(this.pausedEnergyWh * 100) / 100,
      checkpointEnergyWh: Math.round(this.checkpointEnergyWh * 100) / 100,
      totalEmissionsG: Math.round(this.totalEmissionsG * 100) / 100,
      numPauses: this.numPauses, issues: [...this.issues], done: false,
    }, this.chartLabels, this.chartCo2, this.chartStates, this.chartEmissions, this.chartTokensRemaining);
  }
}

function fmtTime(s: number): string {
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${(s / 60).toFixed(1)}m`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m`;
}
