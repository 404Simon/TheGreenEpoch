import { For } from "solid-js";

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
    <div class={`grid grid-cols-1 sm:grid-cols-2 gap-2.5 ${props.class || ""}`}>
      <For each={props.rows}>
        {(row) => (
          <div
            class={`rounded-lg px-3 py-2.5 ${
              row.highlight
                ? "bg-accent-subtle border border-accent/20"
                : "bg-surface/60 border border-border-default/40"
            }`}
          >
            <div class="text-xs text-fg-muted">{row.label}</div>
            <div class="text-sm font-medium mt-0.5 text-fg-primary tabular-nums">
              {row.value}
              {row.unit && <span class="text-fg-muted font-normal ml-1">{row.unit}</span>}
            </div>
          </div>
        )}
      </For>
    </div>
  );
}
