import { For, createSignal, Show } from "solid-js";
import { A, useNavigate } from "@solidjs/router";
import { useApp } from "../data/store";
import { ScenarioForm } from "../components/ScenarioForm";
import type { Scenario } from "../types";

export function ScenariosPage() {
  const app = useApp();
  const navigate = useNavigate();
  const [showForm, setShowForm] = createSignal(false);
  const [editingId, setEditingId] = createSignal<string | null>(null);

  const scenarios = () => app.allScenarios();

  const handlePlay = (sc: Scenario) => {
    const ti = 0;
    const si = 0;
    navigate(`/simulate/${encodeURIComponent(sc.id)}?threshold=${ti}&start=${si}`);
  };

  const handleEdit = (sc: Scenario) => {
    setEditingId(sc.id);
    setShowForm(true);
  };

  const handleSave = (sc: Scenario) => {
    if (editingId() && editingId()!.startsWith("user-")) {
      app.updateScenario(editingId()!, sc);
    } else {
      app.addScenario({ ...sc, id: `user-${Date.now()}` });
    }
    setShowForm(false);
    setEditingId(null);
  };

  const handleDelete = (sc: Scenario) => {
    const id = sc.id;
    if (id.startsWith("user-")) {
      app.deleteScenario(id);
    }
  };

  return (
    <div>
      <div class="flex items-center justify-between mb-6">
        <h1 class="text-2xl font-bold text-white">Scenarios</h1>
        <div class="flex gap-3">
          <button
            onClick={() => { setEditingId(null); setShowForm(true); }}
            class="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-sm font-medium transition-colors"
          >
            + Add Scenario
          </button>
          <A
            href="/run-all"
            class="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium transition-colors"
          >
            ▶ Run All
          </A>
        </div>
      </div>

      <Show when={showForm()}>
        <ScenarioForm
          initial={editingId() ? scenarios().find((s) => s.id === editingId()) : undefined}
          onSave={handleSave}
          onCancel={() => { setShowForm(false); setEditingId(null); }}
        />
      </Show>

      <div class="grid gap-4 md:grid-cols-2">
        <For each={scenarios()} fallback={<div class="text-gray-500 col-span-2 text-center py-12">No scenarios loaded.</div>}>
          {(sc) => {
            const isUserAdded = sc.id.startsWith("user-");
            return (
              <div class="rounded-xl border border-gray-800 bg-gray-900/60 p-4 hover:border-gray-700 transition-colors">
                <div class="flex items-start justify-between mb-3">
                  <div>
                    <h3 class="font-semibold text-white">{sc.description}</h3>
                    <div class="flex items-center gap-2 mt-1 text-xs text-gray-400">
                      <span class="px-1.5 py-0.5 rounded bg-gray-800">{sc.model}</span>
                      <span class="px-1.5 py-0.5 rounded bg-gray-800">{sc.region}</span>
                      {isUserAdded && <span class="px-1.5 py-0.5 rounded bg-amber-900/40 text-amber-400">custom</span>}
                    </div>
                  </div>
                  <div class="flex gap-1">
                    <button
                      onClick={() => handlePlay(sc)}
                      class="p-1.5 rounded hover:bg-emerald-900/40 text-emerald-400 transition-colors"
                      title="Play"
                    >
                      <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 16 16"><path d="M4 2l10 6-10 6V2z"/></svg>
                    </button>
                    <button
                      onClick={() => handleEdit(sc)}
                      class="p-1.5 rounded hover:bg-gray-700 text-gray-400 transition-colors"
                      title="Edit"
                    >
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                    </button>
                    <Show when={isUserAdded}>
                      <button
                        onClick={() => handleDelete(sc)}
                        class="p-1.5 rounded hover:bg-red-900/40 text-red-400 transition-colors"
                        title="Delete"
                      >
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
                      </button>
                    </Show>
                  </div>
                </div>

                <div class="space-y-1 text-xs text-gray-400">
                  <div class="flex justify-between">
                    <span>Thresholds</span>
                    <span class="text-gray-300">{sc.thresholds.map((t) => `${t}`).join(", ")}</span>
                  </div>
                  <div class="flex justify-between">
                    <span>Hysteresis</span>
                    <span class="text-gray-300">{sc.hysteresis.map((h) => `${h}`).join(", ")}</span>
                  </div>
                  <div class="flex justify-between">
                    <span>Start Times</span>
                    <span class="text-gray-300">{sc.startTimes.join(", ")}</span>
                  </div>
                  <div class="flex justify-between">
                    <span>Overhead Budget</span>
                    <span class="text-gray-300">{sc.overheadBudgetPct}%</span>
                  </div>
                </div>
              </div>
            );
          }}
        </For>
      </div>
    </div>
  );
}
