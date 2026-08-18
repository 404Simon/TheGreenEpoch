import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { compile, type TopLevelSpec } from "vega-lite";
import { parse, View } from "vega";
import type { CalibrationBundle, CalibrationRow } from "../domain/types";
import type { FixedRow, FixedSummary, ReoptRegionResult } from "./forecast-sweep";
import { GRACE_THRESHOLD, meanStd } from "./forecast-sweep";

const SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json";
const STAR_PATH = "M0,-1 L0.2245,-0.309 L0.9511,-0.309 L0.3633,0.118 L0.5878,0.809 L0,0.382 L-0.5878,0.809 L-0.3633,0.118 L-0.9511,-0.309 L-0.2245,-0.309 Z";
const FAMILY_COLORS: [string, string] = ["#1f77b4", "#ff7f0e"];
const MODEL_COLORS: [string, string, string] = ["#1f77b4", "#ff7f0e", "#2ca02c"];
const PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"];

export type VlSpec = Record<string, unknown>;

export interface FixedFile {
  region: string;
  rows: FixedRow[];
  summary: FixedSummary[];
}

export interface MetricBandRow {
  region: string;
  family: string;
  level: number;
  metric: "savings" | "overhead" | "score";
  value: number;
  lo: number;
  hi: number;
}

export interface DegradationRow {
  region: string;
  family: string;
  level: number;
  degradation: number;
}

export interface DriftPoint {
  label: string;
  family: string;
  thetaP: number;
  thetaR: number;
}

export interface DriftSegment {
  label: string;
  family: string;
  x: number;
  y: number;
  x2: number;
  y2: number;
}

export interface DriftData {
  region: string;
  segments: DriftSegment[];
  points: DriftPoint[];
  baseline: DriftPoint;
  domain: [number, number];
}

export interface RmseRow {
  region: string;
  horizon: number;
  model: "persistence" | "AR(1)" | "AR(7)";
  rmse: number;
}

export interface PlotForecastCliOptions {
  dataDir?: string;
  outDir?: string;
  regions?: string;
  only?: string;
}

function isBandFamily(family: string): boolean {
  return family === "additive" || family === "multiplicative";
}

const BAND_METRICS: { metric: MetricBandRow["metric"]; select: (r: FixedRow) => number }[] = [
  { metric: "savings", select: (r) => r.savings },
  { metric: "overhead", select: (r) => r.overhead },
  { metric: "score", select: (r) => r.score },
];

export function aggregateBand(rows: FixedRow[]): MetricBandRow[] {
  const groups = new Map<string, FixedRow[]>();
  for (const r of rows) {
    if (!isBandFamily(r.family)) continue;
    const key = `${r.region}\u0000${r.family}\u0000${r.param_value}`;
    const arr = groups.get(key) ?? [];
    arr.push(r);
    groups.set(key, arr);
  }
  const out: MetricBandRow[] = [];
  for (const [key, group] of groups) {
    const [region, family, levelStr] = key.split("\u0000");
    const level = Number(levelStr);
    for (const { metric, select } of BAND_METRICS) {
      const { mean, std } = meanStd(group.map(select));
      out.push({ region, family, level, metric, value: mean, lo: mean - std, hi: mean + std });
    }
  }
  out.sort((a, b) => a.level - b.level || a.family.localeCompare(b.family) || a.metric.localeCompare(b.metric));
  return out;
}

export function buildDegradation(summaries: FixedSummary[], region: string): DegradationRow[] {
  const out: DegradationRow[] = [];
  for (const s of summaries) {
    if (!isBandFamily(s.family)) continue;
    if (s.param_value === 0) continue;
    out.push({ region, family: s.family, level: s.param_value, degradation: s.degradation_frac_mean });
  }
  out.sort((a, b) => a.level - b.level || a.family.localeCompare(b.family));
  return out;
}

export function configLabel(family: string, paramValue: number): string {
  if (family === "additive" || family === "multiplicative") return `${family} ${paramValue}×σ*`;
  return `${family} ${paramValue}`;
}

export function buildDrift(result: ReoptRegionResult): DriftData {
  const base = result.baseline;
  const baseline: DriftPoint = { label: "baseline", family: "baseline", thetaP: base.thetaP, thetaR: base.thetaR };
  const points: DriftPoint[] = [];
  const segments: DriftSegment[] = [];
  for (const config of result.configs) {
    const best = config.best;
    if (best === null || !Number.isFinite(best.thetaP) || !Number.isFinite(best.thetaR)) continue;
    const label = configLabel(config.family, config.param_value);
    points.push({ label, family: config.family, thetaP: best.thetaP, thetaR: best.thetaR });
    segments.push({ label, family: config.family, x: base.thetaP, y: base.thetaR, x2: best.thetaP, y2: best.thetaR });
  }
  const all = [base.thetaP, base.thetaR, ...points.flatMap((p) => [p.thetaP, p.thetaR])];
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  const pad = hi === lo ? 1 : (hi - lo) * 0.15;
  return { region: result.region, segments, points, baseline, domain: [lo - pad, hi + pad] };
}

export function modelLabel(row: CalibrationRow): "persistence" | "AR(1)" | "AR(7)" {
  if (row.model === "persistence") return "persistence";
  return row.order === 7 ? "AR(7)" : "AR(1)";
}

export function buildRmse(bundle: CalibrationBundle): RmseRow[] {
  const out: RmseRow[] = [];
  const seen = new Set<string>();
  for (const r of bundle.evaluation) {
    if (r.model === "persistence" && r.order !== 1) continue;
    const m = modelLabel(r);
    const key = `${r.horizon}\u0000${m}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ region: bundle.region, horizon: r.horizon, model: m, rmse: r.rmse });
  }
  out.sort((a, b) => a.horizon - b.horizon);
  return out;
}

export function buildF1Spec(rows: MetricBandRow[]): VlSpec {
  const colorScale = {
    "domain": ["additive", "multiplicative"],
    "range": FAMILY_COLORS,
  };
  return {
    "$schema": SCHEMA,
    "title": {
      "text": "Savings, overhead, score vs forecast-error level",
      "subtitle": "±1 std seed band; additive vs multiplicative noise (level ×σ*)",
    },
    "width": 190,
    "height": 150,
    "data": { "values": rows },
    "facet": {
      "column": { "field": "region", "type": "nominal" },
      "row": { "field": "metric", "type": "nominal", "header": { "labelAngle": 0 } },
    },
    "spec": {
      "layer": [
        {
          "mark": { "type": "area", "opacity": 0.22, "interpolate": "monotone" },
          "encoding": {
            "y": { "field": "lo", "type": "quantitative" },
            "y2": { "field": "hi", "type": "quantitative" },
            "color": { "field": "family", "type": "nominal", "scale": colorScale },
          },
        },
        {
          "mark": { "type": "line", "point": true, "strokeWidth": 2, "interpolate": "monotone" },
          "encoding": {
            "y": { "field": "value", "type": "quantitative" },
            "color": { "field": "family", "type": "nominal", "scale": colorScale },
          },
        },
      ],
      "encoding": {
        "x": { "field": "level", "type": "quantitative", "title": "Noise level (×σ*)" },
      },
    },
    "resolve": { "scale": { "y": "independent" } },
  };
}

export function buildF2Spec(rows: DegradationRow[]): VlSpec {
  return {
    "$schema": SCHEMA,
    "title": {
      "text": "Relative savings degradation ΔS/S₀ vs noise level",
      "subtitle": "level 0 (baseline, zero degradation) omitted for log x-axis; dashed rule = 10% grace",
    },
    "width": 250,
    "height": 210,
    "data": { "values": rows },
    "facet": { "column": { "field": "region", "type": "nominal" } },
    "spec": {
      "layer": [
        {
          "mark": { "type": "rule", "color": "#666", "strokeDash": [4, 3], "size": 1.5 },
          "encoding": { "y": { "datum": GRACE_THRESHOLD, "type": "quantitative" } },
        },
        {
          "mark": { "type": "text", "text": "10% grace", "dx": -4, "dy": -6, "fontSize": 11, "color": "#666", "align": "right" },
          "encoding": {
            "x": { "datum": 5, "type": "quantitative", "scale": { "type": "log", "domain": [0.2, 5] } },
            "y": { "datum": GRACE_THRESHOLD, "type": "quantitative" },
          },
        },
        {
          "mark": { "type": "line", "point": true, "strokeWidth": 2 },
          "encoding": {
            "x": { "field": "level", "type": "quantitative", "scale": { "type": "log", "domain": [0.2, 5] }, "title": "Noise level (×σ*)" },
            "y": { "field": "degradation", "type": "quantitative", "title": "ΔS/S₀", "scale": { "zero": false } },
            "color": { "field": "family", "type": "nominal", "scale": { "domain": ["additive", "multiplicative"], "range": FAMILY_COLORS } },
          },
        },
      ],
    },
    "resolve": { "scale": { "y": "independent" } },
  };
}

export function buildF3Spec(datas: DriftData[]): VlSpec {
  const labels = [...new Set(datas.flatMap((d) => d.segments.map((s) => s.label)))];
  const colorScale = { "domain": labels, "range": labels.map((_, i) => PALETTE[i % PALETTE.length]) };
  const children = datas.map((d) => ({
    "title": d.region,
    "width": 210,
    "height": 210,
    "layer": [
      {
        "data": { "values": d.segments },
        "mark": { "type": "rule", "strokeWidth": 1.5, "opacity": 0.85 },
        "encoding": {
          "x": { "field": "x", "type": "quantitative", "scale": { "domain": d.domain, "zero": false }, "title": "θₚ" },
          "x2": { "field": "x2", "type": "quantitative" },
          "y": { "field": "y", "type": "quantitative", "scale": { "domain": d.domain, "zero": false }, "title": "θᵣ" },
          "y2": { "field": "y2", "type": "quantitative" },
          "color": { "field": "label", "type": "nominal", "title": "Error config", "scale": colorScale },
        },
      },
      {
        "data": { "values": [d.baseline] },
        "mark": { "type": "point", "shape": STAR_PATH, "size": 340, "fill": "#111" },
        "encoding": {
          "x": { "field": "thetaP", "type": "quantitative", "scale": { "domain": d.domain, "zero": false } },
          "y": { "field": "thetaR", "type": "quantitative", "scale": { "domain": d.domain, "zero": false } },
        },
      },
      {
        "data": { "values": d.points },
        "mark": { "type": "point", "filled": true, "size": 110 },
        "encoding": {
          "x": { "field": "thetaP", "type": "quantitative", "scale": { "domain": d.domain, "zero": false } },
          "y": { "field": "thetaR", "type": "quantitative", "scale": { "domain": d.domain, "zero": false } },
          "color": { "field": "label", "type": "nominal", "title": "Error config", "scale": colorScale },
        },
      },
    ],
  }));
  return {
    "$schema": SCHEMA,
    "title": { "text": "Re-optimized (θₚ, θᵣ) under forecast error", "subtitle": "star = error-free baseline; arrows approximated by line from baseline to seed-mean optimum" },
    "hconcat": children,
    "spacing": 40,
  };
}

export function buildF4Spec(rows: RmseRow[]): VlSpec {
  return {
    "$schema": SCHEMA,
    "title": { "text": "Forecast RMSE vs horizon (2025 test)", "subtitle": "calibrated AR(p) vs persistence; log-x horizon" },
    "width": 250,
    "height": 210,
    "data": { "values": rows },
    "facet": { "column": { "field": "region", "type": "nominal" } },
    "spec": {
      "mark": { "type": "line", "point": true, "strokeWidth": 2 },
      "encoding": {
        "x": { "field": "horizon", "type": "quantitative", "scale": { "type": "log" }, "title": "Horizon (5-min steps)" },
        "y": { "field": "rmse", "type": "quantitative", "title": "RMSE (gCO₂eq/kWh)", "scale": { "zero": false } },
        "color": { "field": "model", "type": "nominal", "title": "Model", "scale": { "domain": ["persistence", "AR(1)", "AR(7)"], "range": MODEL_COLORS } },
      },
    },
  };
}

async function renderSvg(vlSpec: VlSpec): Promise<string> {
  const vgSpec = compile(vlSpec as unknown as TopLevelSpec).spec;
  const view = new View(parse(vgSpec), { renderer: "none" });
  return await view.toSVG();
}

function loadJson<T>(path: string): T {
  return JSON.parse(readFileSync(path, "utf-8")) as T;
}

function rsvgCommand(): string {
  const candidates = ["/usr/sbin/rsvg-convert", "rsvg-convert"];
  for (const c of candidates) {
    try {
      execFileSync(c, ["--version"], { stdio: "ignore" });
      return c;
    } catch {
      // try next candidate
    }
  }
  console.error("  rsvg-convert not found (looked at /usr/sbin/rsvg-convert and PATH)");
  process.exit(1);
}

function requireInput(path: string): void {
  if (!existsSync(path)) {
    console.error(`  Missing required input: ${path}`);
    process.exit(1);
  }
}

export function normalizeEps(path: string): void {
  const eps = readFileSync(path, "utf-8");
  const normalized = eps
    .split("\n")
    .filter((line) => !line.startsWith("%%CreationDate"))
    .join("\n");
  if (normalized !== eps) writeFileSync(path, normalized, "utf-8");
}

export async function plotForecastCli(opts: PlotForecastCliOptions): Promise<void> {
  const dataDir = resolve(opts.dataDir ?? "publication/output/forecast");
  const outDir = resolve(opts.outDir ?? "publication/ICREC_Rome/assets");
  const regions = (opts.regions ?? "DE,IT,SE").split(",").map((s) => s.trim()).filter((s) => s.length > 0);
  const figs = (opts.only ?? "f1,f2,f3,f4").split(",").map((s) => s.trim()).filter((s) => s.length > 0);
  if (regions.length === 0) {
    console.error("  No regions selected (--regions)");
    process.exit(1);
  }
  for (const fig of figs) {
    if (!["f1", "f2", "f3", "f4"].includes(fig)) {
      console.error(`  Unknown figure selector '${fig}' (expected f1,f2,f3,f4)`);
      process.exit(1);
    }
  }

  mkdirSync(outDir, { recursive: true });

  const figNames: Record<string, string> = {
    f1: "forecast_savings_overhead_score",
    f2: "forecast_degradation",
    f3: "forecast_reopt_drift",
    f4: "forecast_rmse_horizon",
  };

  const want = (f: string): boolean => figs.includes(f);

  if (want("f1") || want("f2")) {
    for (const region of regions) requireInput(join(dataDir, `fixed_${region}.json`));
  }
  if (want("f3")) {
    for (const region of regions) requireInput(join(dataDir, `reopt_${region}.json`));
  }
  if (want("f4")) {
    for (const region of regions) requireInput(join(dataDir, `calibration_${region}.json`));
  }

  const specs: { fig: string; spec: VlSpec }[] = [];
  for (const fig of figs) {
    if (fig === "f1") {
      const rows: MetricBandRow[] = [];
      for (const region of regions) rows.push(...aggregateBand(loadJson<FixedFile>(join(dataDir, `fixed_${region}.json`)).rows));
      specs.push({ fig, spec: buildF1Spec(rows) });
    } else if (fig === "f2") {
      const rows: DegradationRow[] = [];
      for (const region of regions) {
        const file = loadJson<FixedFile>(join(dataDir, `fixed_${region}.json`));
        rows.push(...buildDegradation(file.summary, region));
      }
      specs.push({ fig, spec: buildF2Spec(rows) });
    } else if (fig === "f3") {
      const datas = regions.map((r) => buildDrift(loadJson<ReoptRegionResult>(join(dataDir, `reopt_${r}.json`))));
      specs.push({ fig, spec: buildF3Spec(datas) });
    } else if (fig === "f4") {
      const rows: RmseRow[] = [];
      for (const region of regions) rows.push(...buildRmse(loadJson<CalibrationBundle>(join(dataDir, `calibration_${region}.json`))));
      specs.push({ fig, spec: buildF4Spec(rows) });
    }
  }

  const rsvg = rsvgCommand();
  for (const { fig, spec } of specs) {
    const base = join(outDir, figNames[fig]);
    const svg = await renderSvg(spec);
    writeFileSync(`${base}.svg`, svg, "utf-8");
    console.log(`  ${base}.svg (${svg.length} bytes)`);
    execFileSync(rsvg, ["-f", "eps", "-o", `${base}.eps`, `${base}.svg`], { stdio: "inherit" });
    normalizeEps(`${base}.eps`);
    console.log(`  ${base}.eps`);
  }
  console.log(`  Done (${specs.length} figure(s) in ${outDir})`);
}
