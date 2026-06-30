import { createSignal, createMemo, createEffect, For, Show, onCleanup } from "solid-js";
import { useApp } from "../data/store";
import { runOptimizationInWorker } from "../engine/worker";
import type { SweepPoint, FullProfile } from "../domain/types";
import {
  Chart, ScatterController, PointElement,
  LinearScale, Tooltip,
} from "chart.js";

Chart.register(ScatterController, PointElement, LinearScale, Tooltip);

const ITER_COLORS = [
  "#00c9a7", "#36a2eb", "#a29bfe", "#00cec9",
  "#6c5ce7", "#0984e3", "#00b894", "#74b9ff",
];

interface FieldDef {
  key: string;
  label: string;
  fmt: "text" | "num" | "pct" | "kpct" | "score" | "bool";
  dec?: number;
}

const FIELDS: FieldDef[] = [
  { key: "iteration", label: "Iter", fmt: "num", dec: 0 },
  { key: "thetaPause", label: "\u03B8_p", fmt: "num", dec: 0 },
  { key: "thetaResume", label: "\u03B8_r", fmt: "num", dec: 0 },
  { key: "actualOverheadPct", label: "Overhead %", fmt: "pct", dec: 1 },
  { key: "co2SavingsPct", label: "CO\u2082 Save %", fmt: "kpct", dec: 2 },
  { key: "score", label: "Score", fmt: "score", dec: 4 },
  { key: "numPauses", label: "Pauses", fmt: "num", dec: 0 },
  { key: "withinBudget", label: "Budget", fmt: "bool" },
  { key: "stopReason", label: "Stop", fmt: "text" },
];

function fmtNum(v: unknown, d = 0) {
  if (v == null || (typeof v === "number" && isNaN(v))) return "\u2014";
  return Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
}

function fmtPct(v: unknown, d = 1) {
  if (v == null || (typeof v === "number" && isNaN(v))) return "\u2014";
  const n = Number(v);
  return (n >= 0 ? "+" : "") + n.toFixed(d) + "%";
}

function valClass(v: unknown) {
  if (v == null || (typeof v === "number" && isNaN(v))) return "text-fg-muted";
  const n = Number(v);
  if (n > 0.01) return "text-accent";
  if (n < -0.01) return "text-alert-red";
  return "text-fg-muted";
}

function fmtCell(v: unknown, fmt: string, dec?: number) {
  if (v == null || v === "") return "\u2014";
  if (fmt !== "text" && fmt !== "bool" && (typeof v !== "number" || isNaN(v))) return "\u2014";
  switch (fmt) {
    case "num": return fmtNum(v, dec ?? 0);
    case "pct": return fmtPct(v, dec ?? 1);
    case "kpct": return fmtPct(v, 2);
    case "score": return Number(v).toFixed(dec ?? 4);
    case "bool": return v ? "\u2713 Yes" : "\u2717 No";
    default: return String(v);
  }
}

function uniqVals(data: SweepPoint[], key: string, fmt: string, dec?: number) {
  const map = new Map<string, unknown>();
  data.forEach(r => {
    const v = (r as any)[key];
    const f = fmtCell(v, fmt, dec);
    if (!map.has(f)) map.set(f, v);
  });
  return [...map.entries()]
    .sort((a, b) => {
      const va = a[1] as number;
      const vb = b[1] as number;
      if (va == null) return 1; if (vb == null) return -1;
      if (typeof va === "number" && !isNaN(va) && typeof vb === "number" && !isNaN(vb)) return va - vb;
      return String(va).localeCompare(String(vb));
    })
    .map(e => e[0]);
}

export function OptimizePage() {
  const app = useApp();

  const [scenarioId, setScenarioId] = createSignal("");
  const [tpMax, setTpMax] = createSignal(500);
  const [resolution, setResolution] = createSignal(10);
  const [budget, setBudget] = createSignal(200);
  const [alpha, setAlpha] = createSignal(1);
  const [running, setRunning] = createSignal(false);
  const [iterMsg, setIterMsg] = createSignal("");
  const [points, setPoints] = createSignal<SweepPoint[]>([]);
  const [error, setError] = createSignal<string | null>(null);
  const [filters, setFilters] = createSignal<Record<string, string>>({ stopReason: "completed" });
  const [sortKey, setSortKey] = createSignal("co2SavingsPct");
  const [sortAsc, setSortAsc] = createSignal(false);

  let scatterCanvas: HTMLCanvasElement | undefined;
  let scatterChart: Chart | undefined;

  const scenarios = createMemo(() => app.allScenarios());

  const selectedScenario = createMemo(() =>
    scenarios().find(s => s.id === scenarioId()),
  );

  const optimal = createMemo(() => {
    const p = points();
    const filtered = p.filter(r => r.withinBudget && r.co2SavingsPct > 0);
    if (filtered.length === 0) return null;
    return filtered.reduce((a, b) => (a.score > b.score ? a : b));
  });

  const iterCount = createMemo(() => {
    const p = points();
    if (p.length === 0) return 0;
    return Math.max(...p.map(r => r.iteration)) + 1;
  });

  const handleSort = (key: string) => {
    if (sortKey() === key) {
      setSortAsc(!sortAsc());
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const handleFilter = (key: string, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const clearFilters = () => setFilters({});

  const filteredData = createMemo(() => {
    const data = points();
    const f = filters();
    const keys = Object.keys(f);
    if (keys.length === 0 || keys.every(k => !f[k])) return data;
    return data.filter(row =>
      keys.every(key => {
        const val = f[key];
        if (!val) return true;
        const field = FIELDS.find(fi => fi.key === key);
        const cell = fmtCell((row as any)[key], field?.fmt || "text", field?.dec);
        return cell === val;
      })
    );
  });

  const sortedData = createMemo(() => {
    const data = [...filteredData()];
    const sk = sortKey();
    const sa = sortAsc();
    data.sort((a, b) => {
      const va = (a as any)[sk] as number;
      const vb = (b as any)[sk] as number;
      if (va == null) return 1; if (vb == null) return -1;
      return sa ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
    });
    return data;
  });

  const handleRun = async () => {
    const scenario = selectedScenario();
    if (!scenario) return;
    setRunning(true);
    setError(null);
    setPoints([]);
    setIterMsg("");

    try {
      const profile = app.state.profiles![scenario.model];
      if (!profile) throw new Error(`Unknown model: ${scenario.model}`);
      const constants = app.state.constants!;

      const fullProfile: FullProfile = {
        ...profile,
        gpuPowerTrain: constants.gpu_power_train,
        gpuPowerPause: constants.gpu_power_pause,
        pue: constants.pue,
        checkpointPauseTime: constants.checkpoint_pause_time,
        checkpointResumeTime: constants.checkpoint_resume_time,
      };

      let timeline = app.state.co2Cache[scenario.region];
      if (!timeline) {
        const { loadCO2Timeline } = await import("../data/loadData");
        timeline = await loadCO2Timeline(scenario.region);
      }

      await runOptimizationInWorker(
        fullProfile, timeline, scenario,
        {
          thetaPauseMax: tpMax(),
          overheadBudgetPct: budget(),
          resolution: resolution(),
          maxIterations: 6,
          minStep: 3,
          shrinkFactor: 0.45,
          alpha: alpha(),
        },
        0,
        (iter, iterPts, best) => {
          setPoints(prev => {
            const existing = new Set(prev.map(p => `${p.thetaPause},${p.thetaResume}`));
            const merged = [...prev];
            for (const pt of iterPts) {
              const key = `${pt.thetaPause},${pt.thetaResume}`;
              if (!existing.has(key)) {
                existing.add(key);
                merged.push(pt);
              }
            }
            return merged;
          });
          setIterMsg(`Iteration ${iter + 1}: ${iterPts.length} points, best score = ${best ? best.score.toFixed(4) : "none"}`);
        },
      );
    } catch (e) {
      setError(String(e));
    }
    setRunning(false);
    setIterMsg("");
  };

  createEffect(() => {
    const data = filteredData();
    if (data.length === 0) {
      if (scatterChart) { scatterChart.destroy(); scatterChart = undefined; }
      return;
    }
    if (!scatterCanvas) return;
    if (scatterChart) { scatterChart.destroy(); scatterChart = undefined; }

    const opt = optimal();
    const maxIter = Math.max(...data.map(r => r.iteration), 0);

    scatterChart = new Chart(scatterCanvas, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Sweep point",
            data: data.map(r => ({ x: r.actualOverheadPct, y: r.co2SavingsPct })),
            backgroundColor: data.map(r =>
              r.withinBudget
                ? (r === opt ? "#ffd700" : ITER_COLORS[r.iteration % ITER_COLORS.length])
                : "#ff5e7a55",
            ),
            pointRadius: data.map(r => (r === opt ? 10 : r.iteration === maxIter ? 6 : 4)),
            pointHoverRadius: data.map(r => (r === opt ? 12 : 8)),
            borderColor: data.map(r => (r === opt ? "#ffd700" : "transparent")),
            borderWidth: data.map(r => (r === opt ? 2 : 0)),
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const r = data[ctx.dataIndex];
                if (!r) return "";
                return [
                  `Iter ${r.iteration + 1}: \u03B8_p=${r.thetaPause}, \u03B8_r=${r.thetaResume}`,
                  `  Savings: ${r.co2SavingsPct.toFixed(2)}%`,
                  `  Overhead: ${r.actualOverheadPct.toFixed(1)}%`,
                  `  Score: ${r.score.toFixed(4)}`,
                  `  Budget: ${r.withinBudget ? "\u2713" : "\u2717"}`,
                ].join("\n");
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "Overhead %", color: "rgba(156,163,175,0.6)" },
            ticks: { color: "rgba(156,163,175,0.6)", font: { size: 10 } },
            grid: { color: "rgba(75,85,99,0.15)" },
          },
          y: {
            title: { display: true, text: "CO\u2082 Savings %", color: "rgba(156,163,175,0.6)" },
            ticks: { color: "rgba(156,163,175,0.6)", font: { size: 10 } },
            grid: { color: "rgba(75,85,99,0.15)" },
          },
        },
      },
    });
  });

  onCleanup(() => { scatterChart?.destroy(); });

  return (
    <div>
      <div class="flex items-center justify-between mb-6">
        <h1 class="text-2xl font-semibold tracking-tight text-fg-primary">Adaptive Optimization</h1>
      </div>

      <div class="rounded-xl bg-surface-2 border p-6 mb-6 transition-[border-color,box-shadow] duration-700" classList={{ "animate-breathe": running(), "border-border-default/60": !running() }}>
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <div>
            <label class="block text-xs font-medium text-fg-muted mb-1">Scenario</label>
            <select
              value={scenarioId()}
              onChange={e => setScenarioId(e.currentTarget.value)}
              class="w-full bg-surface-3 border border-border-default/50 rounded px-3 py-2 text-sm text-fg-body focus:outline-none focus:border-accent"
            >
              <option value="">Select...</option>
              <For each={scenarios()}>
                {s => <option value={s.id}>{s.description} ({s.model}, {s.region})</option>}
              </For>
            </select>
          </div>

          <div>
            <label class="block text-xs font-medium text-fg-muted mb-1">Max {"\u03B8"}_p</label>
            <input
              type="number" value={tpMax()}
              onInput={e => setTpMax(+e.currentTarget.value || 100)}
              class="w-full bg-surface-3 border border-border-default/50 rounded px-3 py-2 text-sm text-fg-body focus:outline-none focus:border-accent"
              placeholder="Max"
            />
          </div>

          <div>
            <label class="block text-xs font-medium text-fg-muted mb-1">Resolution (per axis)</label>
            <input
              type="number" value={resolution()}
              onInput={e => setResolution(Math.max(3, +e.currentTarget.value || 3))}
              min="3" max="25"
              class="w-full bg-surface-3 border border-border-default/50 rounded px-3 py-2 text-sm text-fg-body focus:outline-none focus:border-accent"
            />
          </div>

          <div>
            <label class="block text-xs font-medium text-fg-muted mb-1">Overhead budget %</label>
            <input
              type="number" value={budget()}
              onInput={e => setBudget(+e.currentTarget.value || 0)}
              class="w-full bg-surface-3 border border-border-default/50 rounded px-3 py-2 text-sm text-fg-body focus:outline-none focus:border-accent"
            />
          </div>

          <div>
            <label class="block text-xs font-medium text-fg-muted mb-1">{"\u03B1"} (CO₂ weight)</label>
            <input
              type="number" value={alpha()}
              onInput={e => setAlpha(+e.currentTarget.value || 0)}
              step="0.1" min="0" max="1"
              class="w-full bg-surface-3 border border-border-default/50 rounded px-3 py-2 text-sm text-fg-body focus:outline-none focus:border-accent"
            />
          </div>
        </div>

        <div class="flex items-center gap-3">
          <button
            onClick={handleRun}
            disabled={running() || !selectedScenario()}
            class="px-5 py-2 rounded-lg bg-accent text-fg-primary text-sm font-medium hover:bg-accent/90 active:scale-[0.97] disabled:opacity-40 disabled:active:scale-100 transition-all"
          >
            {running() ? "Running..." : "Run optimization"}
          </button>
          <Show when={running()}>
            <div class="flex items-center gap-2 text-xs text-fg-subtle">
              <span class="relative flex h-2 w-2">
                <span class="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-75" />
                <span class="relative inline-flex h-2 w-2 rounded-full bg-accent" />
              </span>
              <Show when={iterMsg()} fallback={<span>Initializing...</span>}>
                <span>{iterMsg()}</span>
              </Show>
            </div>
          </Show>
        </div>
      </div>

      <Show when={error()}>
        <div class="rounded-xl border border-alert-red/20 bg-alert-red-bg p-4 mb-6 text-alert-red text-sm">
          {error()}
        </div>
      </Show>

      <Show when={points().length > 0}>
        <Show when={filteredData().length > 0}>
          {s => {
            const fd = filteredData();
            const n = fd.length;
            const avgSave = fd.reduce((s, r) => s + r.co2SavingsPct, 0) / n;
            const within = fd.filter(r => r.withinBudget).length;
            const bestSave = Math.max(...fd.filter(r => r.withinBudget).map(r => r.co2SavingsPct), 0);
            const optPt = optimal();
            const isOptInView = optPt && fd.includes(optPt);
            return (
              <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
                <div class="rounded-xl bg-surface-2 border border-border-default/60 p-4">
                  <div class="text-xs text-fg-muted mb-1">Points</div>
                  <div class="text-xl font-semibold text-fg-primary">{n}</div>
                </div>
                <div class="rounded-xl bg-surface-2 border border-border-default/60 p-4">
                  <div class="text-xs text-fg-muted mb-1">Iterations</div>
                  <div class="text-xl font-semibold text-fg-primary">{iterCount()}</div>
                </div>
                <div class="rounded-xl bg-surface-2 border border-border-default/60 p-4">
                  <div class="text-xs text-fg-muted mb-1">Within budget</div>
                  <div class="text-xl font-semibold text-accent">{within}</div>
                </div>
                <div class="rounded-xl bg-surface-2 border border-border-default/60 p-4">
                  <div class="text-xs text-fg-muted mb-1">Avg CO{"\u2082"} savings</div>
                  <div class="text-xl font-semibold" classList={{ "text-accent": true }}>{fmtPct(avgSave, 2)}</div>
                </div>
                <Show when={isOptInView}>
                  <div class="rounded-xl bg-surface-2 border border-border-default/60 p-4">
                    <div class="text-xs text-fg-muted mb-1">Optimal {"\u03B8"}_p</div>
                    <div class="text-xl font-semibold text-accent">{optPt!.thetaPause}</div>
                  </div>
                  <div class="rounded-xl bg-surface-2 border border-border-default/60 p-4">
                    <div class="text-xs text-fg-muted mb-1">Optimal {"\u03B8"}_r</div>
                    <div class="text-xl font-semibold text-accent">{optPt!.thetaResume}</div>
                  </div>
                </Show>
              </div>
            );
          }}
        </Show>

        <div class="rounded-xl bg-surface-2 border border-border-default/60 p-4 mb-6">
          <h3 class="text-xs font-semibold text-fg-subtle tracking-wide mb-3">Search space: Overhead vs CO{"\u2082"} Savings</h3>
          <div class="relative" style="height: 350px">
            <canvas ref={scatterCanvas!} class="w-full h-full" />
          </div>
          <div class="flex flex-wrap gap-3 mt-3 text-xs text-fg-muted">
            <For each={Array.from({ length: iterCount() })}>
              {(_v, i) => (
                <span class="flex items-center gap-1">
                  <span
                    class="inline-block w-3 h-3 rounded-full"
                    style={{ background: ITER_COLORS[i() % ITER_COLORS.length] }}
                  />
                  Iter {i() + 1}
                </span>
              )}
            </For>
            <span class="flex items-center gap-1">
              <span class="inline-block w-3 h-3 rounded-full bg-[#ff5e7a55]" />
              Over budget
            </span>
            <span class="flex items-center gap-1">
              <span class="inline-block w-3 h-3 rounded-full bg-[#ffd700]" />
              Optimal
            </span>
          </div>
        </div>

        <div class="rounded-xl bg-surface-2 border border-border-default/60 overflow-hidden">
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="border-b border-border-default/60">
                  <For each={FIELDS}>
                    {(field) => {
                      const sorted = sortKey() === field.key;
                      const dir = sorted ? (sortAsc() ? " \u2191" : " \u2193") : "";
                      return (
                        <th
                          onClick={() => handleSort(field.key)}
                          class={`px-3 py-2.5 text-left font-medium cursor-pointer select-none whitespace-nowrap hover:text-fg-body transition-colors ${sorted ? "text-accent" : "text-fg-muted"}`}
                        >
                          {field.label}{dir}
                        </th>
                      );
                    }}
                  </For>
                  <th class="px-3 py-2.5 text-left font-medium text-fg-muted whitespace-nowrap">
                    {Object.values(filters()).some(v => v) ? (
                      <button onClick={clearFilters} class="text-xs text-fg-muted hover:text-fg-body bg-surface-3 border border-border-default/50 rounded px-2 py-1 transition-colors">
                        Clear
                      </button>
                    ) : (
                      "Details"
                    )}
                  </th>
                </tr>
                <tr class="border-b border-border-default/30">
                  <For each={FIELDS}>
                    {(field) => {
                      const uniq = uniqVals(points(), field.key, field.fmt, field.dec);
                      return (
                        <th class="px-3 py-1.5">
                          <select
                            value={filters()[field.key] || ""}
                            onChange={e => handleFilter(field.key, e.currentTarget.value)}
                            class="w-full text-xs bg-surface-3 border border-border-default/50 rounded px-1.5 py-1 text-fg-body focus:outline-none focus:border-accent"
                          >
                            <option value="">All</option>
                            <For each={uniq}>
                              {v => <option value={v}>{v}</option>}
                            </For>
                          </select>
                        </th>
                      );
                    }}
                  </For>
                  <th />
                </tr>
              </thead>
              <tbody>
                <For each={sortedData()}>
                  {(r) => {
                    const isOpt = r === optimal();
                    const cls = isOpt
                      ? "bg-yellow-500/10 border-b border-border-default/20"
                      : "border-b border-border-default/20 hover:bg-white/[0.02]";
                    return (
                      <tr class={cls}>
                        <For each={FIELDS}>
                          {(field) => {
                            const v = (r as any)[field.key];
                            const base = "px-3 py-2.5";
                            let content: string;
                            let extra = " tabular-nums";

                            if (field.fmt === "num") {
                              content = fmtNum(v, field.dec ?? 0);
                            } else if (field.fmt === "pct") {
                              content = fmtPct(v, field.dec ?? 1);
                              extra += " " + valClass(v);
                            } else if (field.fmt === "kpct") {
                              content = fmtPct(v, 2);
                              extra += " " + valClass(v);
                            } else if (field.fmt === "score") {
                              content = v != null && !isNaN(Number(v)) ? Number(v).toFixed(field.dec ?? 4) : "\u2014";
                              extra += " " + valClass(v);
                            } else if (field.fmt === "bool") {
                              content = v ? "\u2713" : "\u2717";
                              extra = ` px-3 py-2.5 text-center${v ? " text-accent" : " text-alert-red"}`;
                            } else {
                              content = String(v ?? "");
                              extra = " px-3 py-2.5";
                            }

                            return <td class={base + extra}>{content}</td>;
                          }}
                        </For>
                        <td class="px-3 py-2.5" />
                      </tr>
                    );
                  }}
                </For>
              </tbody>
            </table>
          </div>
          <div class="flex items-center justify-between px-3 py-2 text-xs text-fg-muted border-t border-border-default/30">
            <span>{sortedData().length} / {points().length} point{points().length !== 1 ? "s" : ""}</span>
          </div>
        </div>
      </Show>
    </div>
  );
}
