import { onMount, onCleanup, createEffect } from "solid-js";
import {
  Chart, LineController, LineElement, PointElement,
  LinearScale, CategoryScale, Tooltip, Filler,
} from "chart.js";
import annotationPlugin from "chartjs-plugin-annotation";

Chart.register(
  LineController, LineElement, PointElement,
  LinearScale, CategoryScale, Tooltip, Filler,
  annotationPlugin,
);

const stateRects = new WeakMap<Chart, { start: number; end: number; color: string }[]>();

const stateBgPlugin = {
  id: "stateBg",
  beforeDraw(chart: Chart) {
    const rects = stateRects.get(chart);
    if (!rects || rects.length === 0) return;
    const ctx = chart.ctx;
    const a = chart.chartArea;
    ctx.save();
    for (const r of rects) {
      if (r.end <= r.start) continue;
      const x1 = a.left + r.start * (a.right - a.left);
      const x2 = a.left + r.end * (a.right - a.left);
      ctx.globalAlpha = 0.12;
      ctx.fillStyle = r.color;
      ctx.fillRect(x1, a.top, x2 - x1, a.bottom - a.top);
    }
    ctx.restore();
  },
};

Chart.register(stateBgPlugin);

interface Props {
  labels: string[];
  co2Data: number[];
  stateSeries?: string[];
  thetaPause?: number;
  thetaResume?: number;
  windowSize?: number;
}

export function CO2Chart(props: Props) {
  let canvas: HTMLCanvasElement | undefined;
  let chart: Chart | undefined;
  let lastKey = "";

  function sliceData(): { labels: string[]; data: number[]; states: string[] } {
    const ws = props.windowSize ?? 0;
    const n = ws > 0 ? Math.min(ws, props.co2Data.length) : props.co2Data.length;
    if (n === 0) return { labels: [], data: [], states: [] };
    const start = ws > 0 ? Math.max(0, props.co2Data.length - ws) : 0;
    const sliced = props.co2Data.slice(start);
    const slicedLabels = props.labels.slice(start);
    const slicedStates = (props.stateSeries ?? []).slice(start);
    if (sliced.length > 2000) {
      const step = sliced.length / 2000;
      const d: number[] = []; const l: string[] = []; const s: string[] = [];
      for (let i = 0; i < 2000; i++) {
        const idx = Math.min(Math.floor(i * step), sliced.length - 1);
        d.push(sliced[idx]); l.push(slicedLabels[idx]); s.push(slicedStates[idx] ?? "");
      }
      return { labels: l, data: d, states: s };
    }
    return { labels: slicedLabels, data: sliced, states: slicedStates };
  }

  function buildRects(states: string[]): { start: number; end: number; color: string }[] {
    if (states.length === 0) return [];
    const rects: { start: number; end: number; color: string }[] = [];
    let runStart = 0;
    let runState = states[0];
    for (let i = 1; i <= states.length; i++) {
      if (i === states.length || states[i] !== runState) {
        const color = runState === "paused" ? "#ef4444" : runState === "running" ? "#22c55e" : "";
        if (color) {
          rects.push({ start: runStart / states.length, end: i / states.length, color });
        }
        if (i < states.length) { runStart = i; runState = states[i]; }
      }
    }
    return rects;
  }

  onMount(() => {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    chart = new Chart(ctx, {
      type: "line",
      data: { labels: [], datasets: [{ data: [], borderColor: "#34d399", backgroundColor: "rgba(52, 211, 153, 0.05)", borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0.1 }] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {
          annotation: { annotations: buildAnnotations(props.thetaPause, props.thetaResume) },
          tooltip: { mode: "index", intersect: false },
        },
        scales: {
          x: { type: "category", display: true, ticks: { color: "rgba(156, 163, 175, 0.6)", maxTicksLimit: 8, font: { size: 10 } }, grid: { color: "rgba(75, 85, 99, 0.15)" } },
          y: { type: "linear", display: true, title: { display: true, text: "gCO₂eq / kWh", color: "rgba(156, 163, 175, 0.6)" }, ticks: { color: "rgba(156, 163, 175, 0.6)", font: { size: 10 } }, grid: { color: "rgba(75, 85, 99, 0.15)" } },
        },
      },
    });
  });

  createEffect(() => {
    if (!chart) return;
    const { labels, data, states } = sliceData();
    const key = `${labels.length}:${data.length}:${props.thetaPause}:${props.thetaResume}`;
    if (key === lastKey) return;
    lastKey = key;
    try {
      chart.data.labels = labels;
      chart.data.datasets[0].data = data;
      const aOpts = chart.options.plugins as any;
      if (aOpts?.annotation) aOpts.annotation.annotations = buildAnnotations(props.thetaPause, props.thetaResume);
      stateRects.set(chart, buildRects(states));
      chart.update();
    } catch (e) {
      console.warn("chart update:", e);
    }
  });

  onCleanup(() => { chart?.destroy(); });

  return <canvas ref={canvas!} class="w-full h-full" />;
}

function buildAnnotations(thetaPause?: number, thetaResume?: number): Record<string, unknown> {
  const a: Record<string, unknown> = {};
  if (thetaPause !== undefined) {
    a.thetaPause = { type: "line", yMin: thetaPause, yMax: thetaPause, borderColor: "rgba(239, 68, 68, 0.7)", borderWidth: 2, borderDash: [6, 3], label: { display: true, content: `θ_pause=${thetaPause}`, position: "start", color: "rgba(239, 68, 68, 0.9)", font: { size: 11 } } };
  }
  if (thetaResume !== undefined) {
    a.thetaResume = { type: "line", yMin: thetaResume, yMax: thetaResume, borderColor: "rgba(34, 197, 94, 0.7)", borderWidth: 2, borderDash: [6, 3], label: { display: true, content: `θ_resume=${thetaResume}`, position: "end", color: "rgba(34, 197, 94, 0.9)", font: { size: 11 } } };
  }
  return a;
}
