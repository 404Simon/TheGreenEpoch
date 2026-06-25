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
    <form onSubmit={handleSubmit} class="mb-6 rounded-xl border border-gray-700 bg-gray-900 p-5 space-y-4">
      <h2 class="text-lg font-semibold text-white">{isEdit ? "Edit Scenario" : "New Scenario"}</h2>

      <div class="grid gap-4 sm:grid-cols-2">
        <div>
          <label class="block text-xs text-gray-400 mb-1">Description</label>
          <input
            value={description()}
            onInput={(e) => setDescription(e.currentTarget.value)}
            class="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white text-sm focus:outline-none focus:border-emerald-500"
            required
          />
        </div>
        <div>
          <label class="block text-xs text-gray-400 mb-1">Model</label>
          <select
            value={model()}
            onChange={(e) => setModel(e.currentTarget.value)}
            class="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white text-sm focus:outline-none focus:border-emerald-500"
          >
            <For each={profiles()}>{(p) => <option value={p}>{p}</option>}</For>
          </select>
        </div>
        <div>
          <label class="block text-xs text-gray-400 mb-1">Region</label>
          <select
            value={region()}
            onChange={(e) => setRegion(e.currentTarget.value)}
            class="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white text-sm focus:outline-none focus:border-emerald-500"
          >
            <For each={REGIONS}>{(r) => <option value={r}>{r}</option>}</For>
          </select>
        </div>
        <div>
          <label class="block text-xs text-gray-400 mb-1">Overhead Budget (%)</label>
          <input
            type="number"
            value={overheadBudget()}
            onInput={(e) => setOverheadBudget(Number(e.currentTarget.value))}
            class="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white text-sm focus:outline-none focus:border-emerald-500"
          />
        </div>
        <div>
          <label class="block text-xs text-gray-400 mb-1">Thresholds (comma-sep)</label>
          <input
            value={thresholdsStr()}
            onInput={(e) => setThresholdsStr(e.currentTarget.value)}
            class="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white text-sm focus:outline-none focus:border-emerald-500"
          />
        </div>
        <div>
          <label class="block text-xs text-gray-400 mb-1">Hysteresis (comma-sep)</label>
          <input
            value={hysteresisStr()}
            onInput={(e) => setHysteresisStr(e.currentTarget.value)}
            class="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white text-sm focus:outline-none focus:border-emerald-500"
          />
        </div>
        <div class="sm:col-span-2">
          <label class="block text-xs text-gray-400 mb-1">Start Times (comma-sep, MM-DD format)</label>
          <input
            value={startTimesStr()}
            onInput={(e) => setStartTimesStr(e.currentTarget.value)}
            class="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white text-sm focus:outline-none focus:border-emerald-500"
          />
        </div>
      </div>

      <div class="flex justify-end gap-3 pt-2">
        <button
          type="button"
          onClick={props.onCancel}
          class="px-4 py-2 rounded-lg border border-gray-700 text-gray-300 text-sm hover:bg-gray-800 transition-colors"
        >
          Cancel
        </button>
        <button
          type="submit"
          class="px-4 py-2 rounded-lg bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-500 transition-colors"
        >
          {isEdit ? "Update" : "Add"} Scenario
        </button>
      </div>
    </form>
  );
}
