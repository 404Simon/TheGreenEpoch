import { createSignal, For, Show } from "solid-js";
import { useNavigate } from "@solidjs/router";
import type { Scenario } from "../types";

interface Props {
  scenario: Scenario;
  onClose: () => void;
}

export function RunOptionsModal(props: Props) {
  const navigate = useNavigate();
  const sc = props.scenario;
  const [thresholdIdx, setThresholdIdx] = createSignal(0);
  const [startIdx, setStartIdx] = createSignal(0);

  const handlePlay = () => {
    navigate(`/simulate/${encodeURIComponent(sc.id)}?threshold=${thresholdIdx()}&start=${startIdx()}`);
  };

  return (
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={props.onClose}>
      <div
        class="w-full max-w-md rounded-xl bg-surface-2 border border-border-default/60 p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 class="text-lg font-semibold text-fg-primary mb-1">{sc.description}</h2>
        <p class="text-sm text-fg-muted mb-5">
          {sc.model} &middot; {sc.region} &middot; {sc.historicalYears.join(", ")}
        </p>

        <Show when={sc.thresholds.length > 1 || sc.startTimes.length > 1}>
          <div class="space-y-4">
            <Show when={sc.thresholds.length > 1}>
              <div>
                <label class="block text-xs text-fg-muted mb-2">Threshold / Hysteresis pair</label>
                <div class="space-y-2">
                  <For each={sc.thresholds}>
                    {(th, i) => (
                      <label
                        class={`flex items-center gap-3 px-3 py-2.5 rounded-lg cursor-pointer transition-colors ${
                          thresholdIdx() === i()
                            ? "bg-accent-subtle border border-accent/30"
                            : "bg-surface border border-border-default/60 hover:border-border-default"
                        }`}
                      >
                        <input
                          type="radio"
                          name="threshold"
                          checked={thresholdIdx() === i()}
                          onChange={() => setThresholdIdx(i())}
                          class="accent-accent"
                        />
                        <span class="text-sm text-fg-primary tabular-nums">
                          θ<sub>pause</sub>={th} &middot; θ<sub>resume</sub>={sc.hysteresis[i()]}
                        </span>
                      </label>
                    )}
                  </For>
                </div>
              </div>
            </Show>

            <Show when={sc.startTimes.length > 1}>
              <div>
                <label class="block text-xs text-fg-muted mb-2">Start time</label>
                <div class="space-y-2">
                  <For each={sc.startTimes}>
                    {(st, i) => (
                      <label
                        class={`flex items-center gap-3 px-3 py-2.5 rounded-lg cursor-pointer transition-colors ${
                          startIdx() === i()
                            ? "bg-accent-subtle border border-accent/30"
                            : "bg-surface border border-border-default/60 hover:border-border-default"
                        }`}
                      >
                        <input
                          type="radio"
                          name="startTime"
                          checked={startIdx() === i()}
                          onChange={() => setStartIdx(i())}
                          class="accent-accent"
                        />
                        <span class="text-sm text-fg-primary tabular-nums">{st}</span>
                      </label>
                    )}
                  </For>
                </div>
              </div>
            </Show>
          </div>
        </Show>

        <div class="flex items-center justify-between gap-3 mt-6 pt-4 border-t border-border-default/40">
          <p class="text-xs text-fg-muted">
            θ<sub>pause</sub>={sc.thresholds[thresholdIdx()]} &middot; θ<sub>resume</sub>={sc.hysteresis[thresholdIdx()]} &middot; start {sc.startTimes[startIdx()]}
          </p>
          <div class="flex gap-2">
            <button
              onClick={props.onClose}
              class="px-4 py-2 rounded-lg border border-border-default/60 text-fg-subtle text-sm hover:bg-white/5 active:scale-[0.97] transition-all"
            >
              Cancel
            </button>
            <button
              onClick={handlePlay}
              class="px-4 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent/90 active:scale-[0.97] transition-all flex items-center gap-1.5"
            >
              <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 16 16"><path d="M4 2l10 6-10 6V2z"/></svg>
              Play
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
