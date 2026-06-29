import { createSignal, For, Show } from "solid-js";
import { A } from "@solidjs/router";
import { useApp } from "../data/store";

export function RunAllPage() {
  const app = useApp();

  const [running, setRunning] = createSignal(false);
  const [done, setDone] = createSignal(0);
  const [total, setTotal] = createSignal(0);
  const [error, setError] = createSignal<string | null>(null);

  const handleRunAll = async () => {
    setRunning(true);
    setError(null);
    setDone(0);

    const scenarios = app.allScenarios();
    const totalRuns = scenarios.reduce((s, sc) => s + sc.thresholds.length * sc.startTimes.length, 0);
    setTotal(totalRuns);

    try {
      await app.runAllScenarios((d, t) => {
        setDone(d);
      });
    } catch (e) {
      setError(String(e));
    }

    setRunning(false);
  };

  const results = () => app.state.results;

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
        <div class="rounded-xl bg-surface-2 border border-border-default/60 overflow-hidden">
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="border-b border-border-default/60 text-fg-muted text-left">
                  <th class="px-4 py-3 font-medium">Scenario</th>
                  <th class="px-4 py-3 font-medium">Region</th>
                  <th class="px-4 py-3 font-medium tabular-nums">θ_p</th>
                  <th class="px-4 py-3 font-medium tabular-nums">θ_r</th>
                  <th class="px-4 py-3 font-medium tabular-nums">CO₂↓</th>
                  <th class="px-4 py-3 font-medium tabular-nums">Score</th>
                  <th class="px-4 py-3 font-medium tabular-nums">Overhead</th>
                  <th class="px-4 py-3 font-medium tabular-nums">Pauses</th>
                  <th class="px-4 py-3 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                <For each={results()}>
                  {(r) => (
                    <tr class="border-b border-border-default/30 hover:bg-white/[0.02] transition-colors">
                      <td class="px-4 py-3 text-fg-primary">{r.scenarioDescription}</td>
                      <td class="px-4 py-3 text-fg-subtle">{r.region}</td>
                      <td class="px-4 py-3 text-fg-body tabular-nums">{r.threshold}</td>
                      <td class="px-4 py-3 text-fg-body tabular-nums">{r.hysteresisMargin}</td>
                      <td class="px-4 py-3 tabular-nums">
                        <span class={r.co2SavingsPct > 0 ? "text-accent" : "text-alert-red"}>
                          {r.co2SavingsPct.toFixed(1)}%
                        </span>
                      </td>
                      <td class="px-4 py-3 font-mono tabular-nums text-fg-body">{r.score.toFixed(1)}</td>
                      <td class="px-4 py-3 text-fg-body tabular-nums">{r.actualOverheadPct.toFixed(1)}%</td>
                      <td class="px-4 py-3 text-fg-body tabular-nums">{r.numPauses}</td>
                      <td class="px-4 py-3">
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
        </div>
      </Show>

      <Show when={results().length === 0 && !running()}>
        <div class="text-center py-20">
          <div class="text-4xl mb-4 opacity-20">⚡</div>
          <p class="text-fg-muted">No results yet</p>
          <p class="text-xs text-fg-muted mt-1">Click "Run all" to start batch simulation.</p>
        </div>
      </Show>
    </div>
  );
}
