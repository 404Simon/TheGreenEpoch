import { createSignal, Show, For, type JSX } from "solid-js";
import type { Scenario, Constants, TrainingProfile } from "../types";
import { useApp } from "../data/store";

interface Props {
  initial?: Scenario;
  onSave: (sc: Scenario) => void;
  onCancel: () => void;
}

const REGIONS = ["DE", "SE", "CN", "IT", "US"];

export function ScenarioForm(props: Props) {
  const app = useApp();
  const isEdit = !!props.initial;

  const [description, setDescription] = createSignal(props.initial?.description || "");
  const [model, setModel] = createSignal(props.initial?.model || "Deepseek");
  const [region, setRegion] = createSignal(props.initial?.region || "DE");
  const [thresholdsStr, setThresholdsStr] = createSignal(props.initial?.thresholds.join(", ") || "550, 600");
  const [hysteresisStr, setHysteresisStr] = createSignal(props.initial?.hysteresis.join(", ") || "500, 550");
  const [startTimesStr, setStartTimesStr] = createSignal(props.initial?.startTimes.join(", ") || "01-01, 02-18, 06-01");
  const [overheadBudget, setOverheadBudget] = createSignal(props.initial?.overheadBudgetPct ?? 200);

  const profiles = () => Object.keys(app.state.profiles || {});

  const handleSubmit = (e: SubmitEvent) => {
    e.preventDefault();
    const thresholds = thresholdsStr().split(",").map((s) => parseFloat(s.trim())).filter(isFinite);
    const hysteresis = hysteresisStr().split(",").map((s) => parseFloat(s.trim())).filter(isFinite);
    const startTimes = startTimesStr().split(",").map((s) => s.trim()).filter(Boolean);

    if (!description().trim() || thresholds.length === 0 || startTimes.length === 0) return;

    props.onSave({
      id: props.initial?.id || `user-${Date.now()}`,
      description: description().trim(),
      model: model(),
      thresholds,
      hysteresis: hysteresis.length === thresholds.length ? hysteresis : thresholds.map(() => hysteresis[0] || 0),
      region: region(),
      startTimes,
      historicalYears: [2022, 2023, 2024, 2025],
      overheadBudgetPct: overheadBudget(),
    });
  };

  return (
    <form onSubmit={handleSubmit} class="mb-6 rounded-xl bg-surface-2 border border-border-default/60 p-5 space-y-5">
      <h2 class="text-lg font-semibold text-fg-primary">{isEdit ? "Edit scenario" : "New scenario"}</h2>

      <div class="grid gap-4 sm:grid-cols-2">
        <div>
          <label class="block text-xs text-fg-muted mb-1.5">Description</label>
          <input
            value={description()}
            onInput={(e) => setDescription(e.currentTarget.value)}
            class="w-full px-3 py-2 rounded-lg bg-surface border border-border-default text-fg-primary text-sm placeholder:text-fg-muted focus:border-accent/50 focus:ring-1 focus:ring-accent/20 transition-all"
            placeholder="e.g. DeepSeek-V3 Germany aggressive"
            required
          />
        </div>
        <div>
          <label class="block text-xs text-fg-muted mb-1.5">Model</label>
          <select
            value={model()}
            onChange={(e) => setModel(e.currentTarget.value)}
            class="w-full px-3 py-2 rounded-lg bg-surface border border-border-default text-fg-primary text-sm focus:border-accent/50 focus:ring-1 focus:ring-accent/20 transition-all"
          >
            <For each={profiles()}>{(p) => <option value={p}>{p}</option>}</For>
          </select>
        </div>
        <div>
          <label class="block text-xs text-fg-muted mb-1.5">Region</label>
          <select
            value={region()}
            onChange={(e) => setRegion(e.currentTarget.value)}
            class="w-full px-3 py-2 rounded-lg bg-surface border border-border-default text-fg-primary text-sm focus:border-accent/50 focus:ring-1 focus:ring-accent/20 transition-all"
          >
            <For each={REGIONS}>{(r) => <option value={r}>{r}</option>}</For>
          </select>
        </div>
        <div>
          <label class="block text-xs text-fg-muted mb-1.5">Overhead budget (%)</label>
          <input
            type="number"
            value={overheadBudget()}
            onInput={(e) => setOverheadBudget(Number(e.currentTarget.value))}
            class="w-full px-3 py-2 rounded-lg bg-surface border border-border-default text-fg-primary text-sm focus:border-accent/50 focus:ring-1 focus:ring-accent/20 transition-all"
          />
        </div>
        <div>
          <label class="block text-xs text-fg-muted mb-1.5">Thresholds (comma-separated)</label>
          <input
            value={thresholdsStr()}
            onInput={(e) => setThresholdsStr(e.currentTarget.value)}
            class="w-full px-3 py-2 rounded-lg bg-surface border border-border-default text-fg-primary text-sm focus:border-accent/50 focus:ring-1 focus:ring-accent/20 transition-all"
            placeholder="550, 600"
          />
        </div>
        <div>
          <label class="block text-xs text-fg-muted mb-1.5">Hysteresis (comma-separated)</label>
          <input
            value={hysteresisStr()}
            onInput={(e) => setHysteresisStr(e.currentTarget.value)}
            class="w-full px-3 py-2 rounded-lg bg-surface border border-border-default text-fg-primary text-sm focus:border-accent/50 focus:ring-1 focus:ring-accent/20 transition-all"
            placeholder="500, 550"
          />
        </div>
        <div class="sm:col-span-2">
          <label class="block text-xs text-fg-muted mb-1.5">Start times (comma-separated, MM-DD format)</label>
          <input
            value={startTimesStr()}
            onInput={(e) => setStartTimesStr(e.currentTarget.value)}
            class="w-full px-3 py-2 rounded-lg bg-surface border border-border-default text-fg-primary text-sm focus:border-accent/50 focus:ring-1 focus:ring-accent/20 transition-all"
            placeholder="01-01, 02-18, 06-01"
          />
        </div>
      </div>

      <div class="flex justify-end gap-3 pt-2">
        <button
          type="button"
          onClick={props.onCancel}
          class="px-4 py-2 rounded-lg border border-border-default/60 text-fg-subtle text-sm hover:bg-white/5 active:scale-[0.97] transition-all"
        >
          Cancel
        </button>
        <button
          type="submit"
          class="px-4 py-2 rounded-lg bg-accent text-fg-primary text-sm font-medium hover:bg-accent/90 active:scale-[0.97] transition-all"
        >
          {isEdit ? "Update" : "Add"} scenario
        </button>
      </div>
    </form>
  );
}
