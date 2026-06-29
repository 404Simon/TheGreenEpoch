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
        <h1 class="text-2xl font-semibold tracking-tight text-fg-primary">Scenarios</h1>
        <div class="flex gap-3">
          <button
            onClick={() => { setEditingId(null); setShowForm(true); }}
            class="px-4 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent/90 active:scale-[0.97] transition-all"
          >
            <span class="mr-1">+</span>Add scenario
          </button>
          <A
            href="/run-all"
            class="px-4 py-2 rounded-lg border border-accent/40 text-accent text-sm font-medium hover:bg-accent-subtle active:scale-[0.97] transition-all"
          >
            Run all
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
        <For each={scenarios()} fallback={
          <div class="col-span-2 text-center py-16">
            <div class="text-3xl mb-3 opacity-30">⚡</div>
            <p class="text-fg-muted">No scenarios loaded</p>
            <p class="text-xs text-fg-muted mt-1">Add a scenario to get started</p>
          </div>
        }>
          {(sc) => {
            const isUserAdded = sc.id.startsWith("user-");
            return (
              <div class="relative rounded-xl bg-surface-2 border border-border-default/60 p-5 hover:border-border-default transition-colors">
                <div class="flex items-start justify-between mb-4">
                  <div class="min-w-0">
                    <h3 class="font-medium text-fg-primary truncate">{sc.description}</h3>
                    <div class="flex items-center gap-2 mt-1.5">
                      <span class="px-2 py-0.5 rounded-md bg-surface-3 text-xs text-fg-subtle">{sc.model}</span>
                      <span class="px-2 py-0.5 rounded-md bg-surface-3 text-xs text-fg-subtle">{sc.region}</span>
                      {isUserAdded && (
                        <span class="px-2 py-0.5 rounded-md bg-accent-subtle text-xs text-accent">custom</span>
                      )}
                    </div>
                  </div>
                  <div class="flex gap-0.5 shrink-0">
                    <button
                      onClick={() => handlePlay(sc)}
                      class="p-1.5 rounded-lg hover:bg-accent-subtle text-accent active:scale-90 transition-all"
                      title="Play"
                    >
                      <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 16 16"><path d="M4 2l10 6-10 6V2z"/></svg>
                    </button>
                    <button
                      onClick={() => handleEdit(sc)}
                      class="p-1.5 rounded-lg hover:bg-white/10 text-fg-subtle active:scale-90 transition-all"
                      title="Edit"
                    >
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                    </button>
                    <Show when={isUserAdded}>
                      <button
                        onClick={() => handleDelete(sc)}
                        class="p-1.5 rounded-lg hover:bg-alert-red-bg text-alert-red active:scale-90 transition-all"
                        title="Delete"
                      >
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
                      </button>
                    </Show>
                  </div>
                </div>

                <div class="space-y-1.5 text-xs text-fg-muted">
                  <div class="flex justify-between">
                    <span>Thresholds</span>
                    <span class="text-fg-body font-medium tabular-nums">{sc.thresholds.map((t) => `${t}`).join(", ")}</span>
                  </div>
                  <div class="flex justify-between">
                    <span>Hysteresis</span>
                    <span class="text-fg-body font-medium tabular-nums">{sc.hysteresis.map((h) => `${h}`).join(", ")}</span>
                  </div>
                  <div class="flex justify-between">
                    <span>Start times</span>
                    <span class="text-fg-body font-medium">{sc.startTimes.join(", ")}</span>
                  </div>
                  <div class="flex justify-between">
                    <span>Overhead budget</span>
                    <span class="text-fg-body font-medium tabular-nums">{sc.overheadBudgetPct}%</span>
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
