import { createSignal, createMemo, createEffect, For, Show, onCleanup } from "solid-js";
import { A } from "@solidjs/router";
import { useApp } from "../data/store";
import {
  Chart, ScatterController, PointElement,
  LinearScale, Tooltip,
} from "chart.js";
import type { SimResult } from "../types";

Chart.register(ScatterController, PointElement, LinearScale, Tooltip);

interface FieldDef {
  key: string;
  label: string;
  fmt: "text" | "num" | "pct" | "kpct" | "score" | "bool";
  dec?: number;
}

const FIELDS: FieldDef[] = [
  { key: "scenarioDescription", label: "Scenario", fmt: "text" },
  { key: "model", label: "Model", fmt: "text" },
  { key: "region", label: "Region", fmt: "text" },
  { key: "startTime", label: "Start", fmt: "text" },
  { key: "threshold", label: "\u03B8_p", fmt: "num", dec: 0 },
  { key: "hysteresisMargin", label: "\u03B8_r", fmt: "num", dec: 0 },
  { key: "numPauses", label: "Pauses", fmt: "num", dec: 0 },
  { key: "actualOverheadPct", label: "Overhead %", fmt: "pct", dec: 1 },
  { key: "co2SavingsPct", label: "CO\u2082 Save %", fmt: "kpct", dec: 2 },
  { key: "score", label: "Score", fmt: "score", dec: 4 },
  { key: "totalEmissionsKgco2", label: "Emissions (kg)", fmt: "num", dec: 0 },
  { key: "totalWallTimeH", label: "Wall (h)", fmt: "num", dec: 1 },
  { key: "trainingTimeH", label: "Train (h)", fmt: "num", dec: 1 },
  { key: "pausedTimeH", label: "Paused (h)", fmt: "num", dec: 1 },
  { key: "totalEnergyKwh", label: "Energy (kWh)", fmt: "num", dec: 0 },
  { key: "completed", label: "Done", fmt: "bool" },
  { key: "withinOverheadBudget", label: "Budget", fmt: "bool" },
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

function colorForSave(v: number, budgetExceeded?: boolean) {
  if (budgetExceeded) return "#ff5e7a";
  if (v > 5) return "#00c9a7";
  if (v > 0) return "#7ddbc6";
  if (v > -5) return "#ff9f43";
  return "#ff5e7a";
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

function uniqVals(data: SimResult[], key: string, fmt: string, dec?: number) {
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

export function RunsPage() {
  const app = useApp();

  const [running, setRunning] = createSignal(false);
  const [done, setDone] = createSignal(0);
  const [total, setTotal] = createSignal(0);
  const [error, setError] = createSignal<string | null>(null);
  const [filters, setFilters] = createSignal<Record<string, string>>({ stopReason: "completed" });
  const [sortKey, setSortKey] = createSignal("co2SavingsPct");
  const [sortAsc, setSortAsc] = createSignal(false);

  const results = () => app.state.batchResults;

  let scatterCanvas: HTMLCanvasElement | undefined;
  let scatterChart: Chart | undefined;

  const handleRunAll = async () => {
    setRunning(true);
    setError(null);
    setDone(0);
    const scenarios = app.allScenarios();
    const totalRuns = scenarios.reduce((s, sc) => s + sc.thresholds.length * sc.startTimes.length, 0);
    setTotal(totalRuns);
    try {
      await app.runAllScenarios((d) => setDone(d));
    } catch (e) {
      setError(String(e));
    }
    setRunning(false);
  };

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
    const data = results();
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

  const stats = createMemo(() => {
    const data = filteredData();
    const n = data.length;
    if (n === 0) return null;
    const avgSave = data.reduce((s, r) => s + r.co2SavingsPct, 0) / n;
    const avgScore = data.reduce((s, r) => s + r.score, 0) / n;
    const worst = data.reduce((a, b) => (a.co2SavingsPct < b.co2SavingsPct ? a : b), data[0]);
    const totalPauses = data.reduce((s, r) => s + r.numPauses, 0);
    const totalEm = data.reduce((s, r) => s + r.totalEmissionsKgco2, 0);
    const totalBaseEm = data.reduce((s, r) => s + r.baselineEmissionsKgco2, 0);
    const totalSave = totalBaseEm > 0 ? (totalBaseEm - totalEm) / totalBaseEm * 100 : 0;
    return { n, avgSave, avgScore, worst, totalPauses, totalEm, totalBaseEm, totalSave };
  });

  createEffect(() => {
    if (running()) { if (scatterChart) { scatterChart.destroy(); scatterChart = undefined; } return; }
    const data = filteredData();
    if (!scatterCanvas) return;
    if (scatterChart) { scatterChart.destroy(); scatterChart = undefined; }
    if (data.length === 0) return;
    scatterChart = new Chart(scatterCanvas, {
      type: "scatter",
      data: {
        datasets: [{
          label: "Run",
          data: data.map(r => ({ x: r.actualOverheadPct, y: r.co2SavingsPct })),
          backgroundColor: data.map(r => colorForSave(r.co2SavingsPct, r.stopReason === "budget_exceeded")),
          pointRadius: 6,
          pointHoverRadius: 9,
        }],
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
                  `${r.scenarioDescription} (\u03B8=${r.threshold}, ${r.region})`,
                  `  Savings: ${r.co2SavingsPct.toFixed(2)}%`,
                  `  Overhead: ${r.actualOverheadPct.toFixed(1)}%`,
                  `  Score: ${r.score.toFixed(4)}`,
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
        <h1 class="text-2xl font-semibold tracking-tight text-fg-primary">Run all scenarios</h1>
        <button
          onClick={handleRunAll}
          disabled={running()}
          class="px-5 py-2 rounded-lg bg-accent text-fg-primary text-sm font-medium hover:bg-accent/90 active:scale-[0.97] disabled:opacity-40 disabled:active:scale-100 transition-all"
        >
          {running() ? "Running..." : results().length > 0 ? "Re-run all" : "Run all"}
        </button>
      </div>

      <Show when={running()}>
        <div class="rounded-xl bg-surface-2 border border-border-default/60 p-6 mb-6">
          <div class="text-center">
            <div class="text-base text-fg-primary font-medium mb-3">
              Running simulations... {done()} / {total()}
            </div>
            <div class="w-full bg-surface-3 rounded-full h-2 overflow-hidden">
              <div
                class="h-full bg-accent rounded-full transition-all duration-500 ease-out"
                style={{ width: `${total() > 0 ? (done() / total()) * 100 : 0}%` }}
              />
            </div>
          </div>
        </div>
      </Show>

      <Show when={error()}>
        <div class="rounded-xl border border-alert-red/20 bg-alert-red-bg p-4 mb-6 text-alert-red text-sm">
          {error()}
        </div>
      </Show>

      <Show when={results().length > 0 && !running()}>
        <Show when={stats()}>
          {s => (
            <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
              <div class="rounded-xl bg-surface-2 border border-border-default/60 p-4">
                <div class="text-xs text-fg-muted mb-1">Total Runs</div>
                <div class="text-xl font-semibold text-fg-primary">{s().n}</div>
              </div>
              <div class="rounded-xl bg-surface-2 border border-border-default/60 p-4">
                <div class="text-xs text-fg-muted mb-1">{"Avg CO\u2082 Savings"}</div>
                <div class={`text-xl font-semibold ${valClass(s().avgSave)}`}>{fmtPct(s().avgSave, 2)}</div>
              </div>
              <div class="rounded-xl bg-surface-2 border border-border-default/60 p-4">
                <div class="text-xs text-fg-muted mb-1">Avg Score</div>
                <div class={`text-xl font-semibold ${valClass(s().avgScore)}`}>{s().avgScore.toFixed(4)}</div>
              </div>
              <div class="rounded-xl bg-surface-2 border border-border-default/60 p-4">
                <div class="text-xs text-fg-muted mb-1">Total Saved</div>
                <div class={`text-xl font-semibold ${valClass(s().totalSave)}`}>{fmtPct(s().totalSave, 2)}</div>
                <div class="text-xs text-fg-muted">{fmtNum(s().totalEm, 0)} vs {fmtNum(s().totalBaseEm, 0)} kg</div>
              </div>
              <div class="rounded-xl bg-surface-2 border border-border-default/60 p-4">
                <div class="text-xs text-fg-muted mb-1">Total Pauses</div>
                <div class="text-xl font-semibold text-fg-primary">{fmtNum(s().totalPauses, 0)}</div>
              </div>
              <div class="rounded-xl bg-surface-2 border border-border-default/60 p-4">
                <div class="text-xs text-fg-muted mb-1">Worst Savings</div>
                <div class={`text-xl font-semibold ${valClass(s().worst.co2SavingsPct)}`}>{fmtPct(s().worst.co2SavingsPct, 2)}</div>
                <div class="text-xs text-fg-muted truncate" title={s().worst.scenarioDescription}>{s().worst.scenarioDescription}</div>
              </div>
            </div>
          )}
        </Show>

        <Show when={filteredData().length > 0}>
          <div class="rounded-xl bg-surface-2 border border-border-default/60 p-4 mb-6">
            <h3 class="text-xs font-semibold text-fg-subtle tracking-wide mb-3">{"CO\u2082 Savings vs Overhead"}</h3>
            <div class="relative" style="height: 350px">
              <canvas ref={scatterCanvas!} class="w-full h-full" />
            </div>
          </div>
        </Show>

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
                      const uniq = uniqVals(results(), field.key, field.fmt, field.dec);
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
                  {(r) => (
                    <tr class="border-b border-border-default/20 hover:bg-white/[0.02] transition-colors">
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
                      <td class="px-3 py-2.5">
                        <A
                          href={`/results/${r.id}`}
                          class="text-accent hover:text-accent/80 text-xs transition-colors"
                        >
                          Details
                        </A>
                      </td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </div>
          <div class="flex items-center justify-between px-3 py-2 text-xs text-fg-muted border-t border-border-default/30">
            <span>{sortedData().length} / {results().length} run{results().length !== 1 ? "s" : ""}</span>
          </div>
        </div>
      </Show>

      <Show when={results().length === 0 && !running()}>
        <div class="text-center py-20">
          <div class="text-4xl mb-4 opacity-20">{/* lightning */}&#x26A1;</div>
          <p class="text-fg-muted">No results yet</p>
          <p class="text-xs text-fg-muted mt-1">Click "Run all" to start batch simulation.</p>
        </div>
      </Show>
    </div>
  );
}
