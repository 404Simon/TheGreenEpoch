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
        <h1 class="text-2xl font-bold text-white">Run All Scenarios</h1>
        <button
          onClick={handleRunAll}
          disabled={running()}
          class="px-6 py-3 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium transition-colors"
        >
          {running() ? "Running…" : results().length > 0 ? "Re-run All" : "▶ Run All"}
        </button>
      </div>

      <Show when={running()}>
        <div class="rounded-xl border border-gray-800 bg-gray-900/60 p-6 mb-6">
          <div class="text-center">
            <div class="text-lg text-white font-medium mb-2">
              Running simulations… {done()} / {total()}
            </div>
            <div class="w-full bg-gray-800 rounded-full h-3 overflow-hidden">
              <div
                class="h-full bg-emerald-500 rounded-full transition-all duration-300"
                style={{ width: `${total() > 0 ? (done() / total()) * 100 : 0}%` }}
              />
            </div>
          </div>
        </div>
      </Show>

      <Show when={error()}>
        <div class="rounded-xl border border-red-800 bg-red-950/30 p-4 mb-6 text-red-400 text-sm">
          {error()}
        </div>
      </Show>

      <Show when={results().length > 0 && !running()}>
        <div class="rounded-xl border border-gray-800 bg-gray-900/60 overflow-hidden">
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="border-b border-gray-800 text-gray-400 text-left">
                  <th class="px-4 py-3 font-medium">Scenario</th>
                  <th class="px-4 py-3 font-medium">Region</th>
                  <th class="px-4 py-3 font-medium">θ_p</th>
                  <th class="px-4 py-3 font-medium">θ_r</th>
                  <th class="px-4 py-3 font-medium">CO₂↓</th>
                  <th class="px-4 py-3 font-medium">Score</th>
                  <th class="px-4 py-3 font-medium">Overhead</th>
                  <th class="px-4 py-3 font-medium">Pauses</th>
                  <th class="px-4 py-3 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                <For each={results()}>
                  {(r) => (
                    <tr class="border-b border-gray-800/50 hover:bg-gray-800/30">
                      <td class="px-4 py-3 text-white">{r.scenarioDescription}</td>
                      <td class="px-4 py-3">{r.region}</td>
                      <td class="px-4 py-3">{r.threshold}</td>
                      <td class="px-4 py-3">{r.hysteresisMargin}</td>
                      <td class="px-4 py-3">
                        <span class={r.co2SavingsPct > 0 ? "text-emerald-400" : "text-red-400"}>
                          {r.co2SavingsPct.toFixed(1)}%
                        </span>
                      </td>
                      <td class="px-4 py-3 font-mono">{r.score.toFixed(1)}</td>
                      <td class="px-4 py-3">{r.actualOverheadPct.toFixed(1)}%</td>
                      <td class="px-4 py-3">{r.numPauses}</td>
                      <td class="px-4 py-3">
                        <A
                          href={`/results/${r.id}`}
                          class="text-emerald-400 hover:text-emerald-300 text-xs"
                        >
                          Details →
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
        <div class="text-center py-16 text-gray-500">
          No results yet. Click "Run All" to start batch simulation.
        </div>
      </Show>
    </div>
  );
}
