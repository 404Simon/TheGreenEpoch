import { type JSX } from "solid-js";

interface StatRow {
  label: string;
  value: string;
  unit?: string;
  highlight?: boolean;
}

interface Props {
  rows: StatRow[];
  class?: string;
}

export function StatsPanel(props: Props) {
  return (
    <div class={`grid grid-cols-2 gap-3 ${props.class || ""}`}>
      <For each={props.rows}>
        {(row) => (
          <div
            class={`rounded-lg border px-3 py-2 ${
              row.highlight
                ? "border-emerald-700/50 bg-emerald-950/30"
                : "border-gray-800 bg-gray-900/50"
            }`}
          >
            <div class="text-xs text-gray-500 uppercase tracking-wide">{row.label}</div>
            <div class="text-sm font-semibold mt-0.5">
              {row.value}
              {row.unit && <span class="text-gray-500 font-normal ml-1">{row.unit}</span>}
            </div>
          </div>
        )}
      </For>
    </div>
  );
}
