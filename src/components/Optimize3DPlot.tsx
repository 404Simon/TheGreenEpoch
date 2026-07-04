import { createEffect, onCleanup } from "solid-js";
import type { SweepPoint } from "../domain/types";
import { dateToDay, dayToDate } from "../domain/optimize";

interface Props {
  data: SweepPoint[];
  optimal: SweepPoint | null;
}

let plotlyPromise: Promise<any> | null = null;
function getPlotly(): Promise<any> {
  if (!plotlyPromise) {
    plotlyPromise = import("plotly.js-dist-min").then(mod => (mod as any).default || mod);
  }
  return plotlyPromise;
}

function renderPlot(Plotly: any, container: HTMLElement, data: SweepPoint[], opt: SweepPoint | null) {
  if (data.length === 0) return;

  const regular = data.filter(p => p !== opt);
  const scores = data.map(p => p.score);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);

  const traces: any[] = [];

  traces.push({
    type: "scatter3d",
    mode: "markers",
    showlegend: false,
    x: regular.map(p => p.thetaPause),
    y: regular.map(p => p.thetaResume),
    z: regular.map(p => dateToDay(p.startTime)),
    marker: {
      size: 4,
      color: regular.map(p => p.score),
      colorscale: "Viridis",
      cmin: minScore,
      cmax: maxScore,
      colorbar: { title: { text: "Score" }, thickness: 15 },
      line: { color: "rgba(0,0,0,0.15)", width: 0.5 },
    },
    text: regular.map(p =>
      [
        `θ_p=${p.thetaPause}`,
        `θ_r=${p.thetaResume}`,
        `Start: ${p.startTime}`,
        `Score: ${p.score.toFixed(4)}`,
        `Savings: ${p.co2SavingsPct.toFixed(2)}%`,
        `Overhead: ${p.actualOverheadPct.toFixed(1)}%`,
        `Iter: ${p.iteration + 1}`,
      ].join("<br>"),
    ),
    hoverinfo: "text",
    hovertemplate: "%{text}<extra></extra>",
  });

  if (opt) {
    traces.push({
      type: "scatter3d",
      mode: "markers",
      showlegend: false,
      x: [opt.thetaPause],
      y: [opt.thetaResume],
      z: [dateToDay(opt.startTime)],
      marker: { size: 12, color: "#ffd700", line: { color: "#fff", width: 2 } },
      text: [
        [
          "★ Optimal",
          `θ_p=${opt.thetaPause}`,
          `θ_r=${opt.thetaResume}`,
          `Start: ${opt.startTime}`,
          `Score: ${opt.score.toFixed(4)}`,
          `Savings: ${opt.co2SavingsPct.toFixed(2)}%`,
          `Overhead: ${opt.actualOverheadPct.toFixed(1)}%`,
        ].join("<br>"),
      ],
      hoverinfo: "text",
      hovertemplate: "%{text}<extra></extra>",
    });
  }

  const allDays = [...regular.map(p => dateToDay(p.startTime))];
  if (opt) allDays.push(dateToDay(opt.startTime));
  const zMin = Math.min(...allDays);
  const zMax = Math.max(...allDays);
  const tickVals: number[] = [];
  const tickText: string[] = [];
  if (zMin === zMax) {
    tickVals.push(zMin);
    tickText.push(dayToDate(zMin));
  } else {
    const nTicks = Math.min(6, zMax - zMin + 1);
    for (let i = 0; i < nTicks; i++) {
      const d = Math.round(zMin + (zMax - zMin) * i / (nTicks - 1));
      tickVals.push(d);
      tickText.push(dayToDate(d));
    }
  }

  const layout = {
    showlegend: false,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 50, r: 30, t: 10, b: 50 },
    font: { color: "rgba(156,163,175,0.8)" },
    scene: {
      xaxis: {
        title: { text: "θ_p (pause threshold)" },
        gridcolor: "rgba(75,85,99,0.2)",
        zerolinecolor: "rgba(75,85,99,0.2)",
      },
      yaxis: {
        title: { text: "θ_r (resume threshold)" },
        gridcolor: "rgba(75,85,99,0.2)",
        zerolinecolor: "rgba(75,85,99,0.2)",
      },
      zaxis: {
        title: { text: "Start date" },
        tickvals: tickVals,
        ticktext: tickText,
        gridcolor: "rgba(75,85,99,0.2)",
        zerolinecolor: "rgba(75,85,99,0.2)",
      },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 0.8 },
      },
    },
    dragmode: "orbit",
  };

  Plotly.react(container, traces, layout, { responsive: true, displayModeBar: false });
}

export function Optimize3DPlot(props: Props) {
  let container: HTMLDivElement | undefined;

  createEffect(() => {
    if (props.data.length === 0 || !container) return;
    props.optimal;

    const c = container;
    getPlotly().then(Plotly => {
      if (!c || !document.contains(c)) return;
      renderPlot(Plotly, c, props.data, props.optimal);
    });
  });

  onCleanup(() => {
    if (container) {
      getPlotly().then(Plotly => {
        try { Plotly.purge(container); } catch {}
      });
    }
  });

  return (
    <div
      ref={container!}
      class="w-full"
      style="height: 420px"
    />
  );
}
