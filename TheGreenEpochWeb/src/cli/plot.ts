import { readFileSync, writeFileSync } from "node:fs";
import { resolve, extname } from "node:path";
import { compile } from "vega-lite";
import { parse, View } from "vega";
import type { SweepPoint } from "../domain/types";

function computeConvergence(points: SweepPoint[]): { iteration: number; bestScore: number; bestSavings: number; bestOverhead: number }[] {
  const byIter = new Map<number, SweepPoint[]>();
  for (const p of points) {
    const arr = byIter.get(p.iteration) ?? [];
    arr.push(p);
    byIter.set(p.iteration, arr);
  }
  const result: { iteration: number; bestScore: number; bestSavings: number; bestOverhead: number }[] = [];
  let globalBestScore = -Infinity;
  for (const iter of [...byIter.keys()].sort((a, b) => a - b)) {
    const iterPts = byIter.get(iter)!;
    const valid = iterPts.filter(p => p.withinBudget && p.co2SavingsPct > 0);
    if (valid.length > 0) {
      const iterBest = valid.reduce((a, b) => (a.score > b.score ? a : b));
      if (iterBest.score > globalBestScore) globalBestScore = iterBest.score;
    }
    result.push({
      iteration: iter + 1,
      bestScore: globalBestScore > -Infinity ? globalBestScore : 0,
      bestSavings: globalBestScore > -Infinity
        ? valid.reduce((a, b) => (a.score > b.score ? a : b), valid[0]).co2SavingsPct
        : 0,
      bestOverhead: globalBestScore > -Infinity
        ? valid.reduce((a, b) => (a.score > b.score ? a : b), valid[0]).actualOverheadPct
        : 0,
    });
  }
  return result;
}

function buildScatterSpec(points: SweepPoint[], budget: number): unknown {
  const colorRange = [
    "#00c9a7", "#36a2eb", "#a29bfe", "#00cec9",
    "#6c5ce7", "#0984e3", "#00b894", "#74b9ff",
  ];

  return {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "title": `Overhead vs CO₂ Savings (budget ${budget}%)`,
    "width": 500, "height": 400,
    "data": { values: points.map(p => ({ ...p, _iterLabel: `Iter ${p.iteration + 1}` })) },
    "layer": [
      {
        "mark": { "type": "rule", "color": "#ff5e7a", "strokeDash": [6, 3], "size": 2 },
        "encoding": { "x": { "datum": budget, "type": "quantitative" } },
      },
      {
        "mark": { "type": "point", "filled": true, "opacity": 0.7 },
        "encoding": {
          "x": { "field": "actualOverheadPct", "type": "quantitative", "title": "Overhead %", "scale": { "zero": false } },
          "y": { "field": "co2SavingsPct", "type": "quantitative", "title": "CO₂ Savings %", "scale": { "zero": false } },
          "color": {
            "field": "_iterLabel", "type": "nominal", "title": "Iteration",
            "scale": { "range": colorRange },
          },
          "opacity": {
            "condition": { "test": "datum.withinBudget", "value": 0.85 },
            "value": 0.3,
          },
        },
      },
    ],
  };
}

function buildConvergenceSpec(convergence: { iteration: number; bestScore: number }[]): unknown {
  return {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "title": "Convergence: Best Score by Iteration",
    "width": 500, "height": 300,
    "data": { values: convergence },
    "encoding": {
      "x": { "field": "iteration", "type": "quantitative", "title": "Iteration", "scale": { "zero": false } },
      "y": { "field": "bestScore", "type": "quantitative", "title": "Best Score", "scale": { "zero": false } },
    },
    "layer": [
      {
        "mark": { "type": "line", "color": "#00c9a7", "point": true, "strokeWidth": 2 },
      },
      {
        "mark": { "type": "text", "dy": -10, "fontSize": 10, "color": "#00c9a7" },
        "encoding": {
          "text": { "field": "bestScore", "format": ".3f" },
        },
      },
    ],
  };
}

function buildHeatmapSpec(points: SweepPoint[]): unknown {
  const valid = points.filter(p => p.withinBudget);
  return {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "title": "Search Grid: θₚ vs θᵣ (color = Score, budget only)",
    "width": 500, "height": 400,
    "data": { values: valid },
    "mark": { "type": "point", "filled": true },
    "encoding": {
      "x": { "field": "thetaPause", "type": "quantitative", "title": "θₚ", "scale": { "zero": false } },
      "y": { "field": "thetaResume", "type": "quantitative", "title": "θᵣ", "scale": { "zero": false } },
      "color": {
        "field": "score", "type": "quantitative", "title": "Score",
        "scale": { "scheme": "viridis" },
      },
      "size": { "value": 60 },
    },
  };
}

async function renderSvg(vlSpec: unknown): Promise<string> {
  const vgSpec = compile(vlSpec as any).spec;
  const view = new View(parse(vgSpec), { renderer: "none" });
  return await view.toSVG();
}

export async function plotCli(inputPath: string, opts: { output?: string }): Promise<void> {
  const outputBase = opts.output ?? "plot";

  interface Result { scenario: { description: string; model: string; region: string }; options: Record<string, unknown>; points: SweepPoint[] }
  let data: Result;
  try {
    const raw = readFileSync(resolve(inputPath), "utf-8");
    data = JSON.parse(raw) as Result;
  } catch (e) {
    console.error(`  Error reading ${inputPath}:`, (e as Error).message);
    process.exit(1);
  }

  if (!data.points || data.points.length === 0) {
    console.error("  No points found in results file.");
    process.exit(1);
  }

  const budget = (data.options?.overheadBudgetPct as number) || 200;
  const convergence = computeConvergence(data.points);

  const plots: { name: string; spec: unknown }[] = [
    { name: "scatter", spec: buildScatterSpec(data.points, budget) },
    { name: "convergence", spec: buildConvergenceSpec(convergence) },
    { name: "heatmap", spec: buildHeatmapSpec(data.points) },
  ];

  const ext = extname(outputBase) ? "" : ".svg";
  const base = extname(outputBase) ? outputBase.replace(extname(outputBase), "") : outputBase;

  for (const p of plots) {
    const path = `${base}_${p.name}${ext || ".svg"}`;
    const svg = await renderSvg(p.spec);
    writeFileSync(path, svg, "utf-8");
    console.log(`  ${path}`);
  }

  console.log(`  Done (${data.points.length} points, ${convergence.length} iterations)`);
}
