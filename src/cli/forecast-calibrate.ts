import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { ArCoeffs, CalibrationBundle, CalibrationOrderInfo, CalibrationRow, YearCO2 } from "../domain/types";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(__dirname, "../../public/data");
const OUT_DIR = resolve(__dirname, "../../publication/output/forecast");

export interface HorizonMetrics {
  rmse: number;
  mae: number;
  mape: number;
}

export interface HorizonEvaluation {
  ar: HorizonMetrics;
  persistence: HorizonMetrics;
}

function loadJSON<T>(path: string): T {
  return JSON.parse(readFileSync(resolve(DATA_DIR, path), "utf-8")) as T;
}

function loadRawYear(region: string, year: number): number[] {
  return loadJSON<YearCO2>(`co2/${region}_${year}.json`).carbonIntensity;
}

export function computeAutocorr(series: number[], lag: number): number {
  if (!Number.isInteger(lag) || lag < 1) {
    throw new Error(`computeAutocorr: lag must be a positive integer, got ${lag}`);
  }
  if (!Array.isArray(series) || series.length <= lag) {
    throw new Error(`computeAutocorr: series of length ${series.length} must be longer than lag ${lag}`);
  }
  const n = series.length;
  let sumX = 0;
  let sumY = 0;
  let sumXX = 0;
  let sumYY = 0;
  let sumXY = 0;
  for (let t = lag; t < n; t++) {
    const x = series[t];
    const y = series[t - lag];
    if (!isFinite(x) || !isFinite(y)) {
      throw new Error("computeAutocorr: series contains non-finite values");
    }
    sumX += x;
    sumY += y;
    sumXX += x * x;
    sumYY += y * y;
    sumXY += x * y;
  }
  const count = n - lag;
  const num = count * sumXY - sumX * sumY;
  const den = Math.sqrt((count * sumXX - sumX * sumX) * (count * sumYY - sumY * sumY));
  if (den === 0) return NaN;
  return num / den;
}

interface PooledMoments {
  mean: number;
  n: number;
  std: number;
  gamma: number[];
}

function computePooledMoments(perYearSeries: number[][], maxLag: number): PooledMoments {
  if (!Array.isArray(perYearSeries) || perYearSeries.length === 0) {
    throw new Error("computePooledMoments: at least one series required");
  }
  if (!Number.isInteger(maxLag) || maxLag < 0) {
    throw new Error(`computePooledMoments: invalid max lag ${maxLag}`);
  }
  let total = 0;
  let n = 0;
  for (const s of perYearSeries) {
    if (!Array.isArray(s) || s.length === 0) {
      throw new Error("computePooledMoments: empty year series");
    }
    for (const v of s) {
      if (!isFinite(v)) throw new Error("computePooledMoments: series contains non-finite values");
      total += v;
      n++;
    }
  }
  const mean = total / n;
  const gamma = new Array<number>(maxLag + 1).fill(0);
  let sumSq = 0;
  for (const s of perYearSeries) {
    for (const v of s) sumSq += (v - mean) * (v - mean);
    for (let k = 0; k <= maxLag; k++) {
      let acc = 0;
      for (let t = k; t < s.length; t++) acc += (s[t] - mean) * (s[t - k] - mean);
      gamma[k] += acc;
    }
  }
  const std = Math.sqrt(sumSq / (n - 1));
  return { mean, n, std, gamma };
}

export function fitArPooled(perYearSeries: number[][], order: number): ArCoeffs {
  if (!Number.isInteger(order) || order < 1) {
    throw new Error(`fitArPooled: order must be a positive integer, got ${order}`);
  }
  if (!Array.isArray(perYearSeries) || perYearSeries.length === 0) {
    throw new Error("fitArPooled: at least one series required");
  }
  for (const s of perYearSeries) {
    if (s.length <= order) {
      throw new Error(`fitArPooled: series of length ${s.length} must be longer than order ${order}`);
    }
  }
  const { mean, gamma } = computePooledMoments(perYearSeries, order);
  const m = order;
  const aug = Array.from({ length: m }, (_, i) => {
    const row = new Array<number>(m + 1);
    for (let j = 0; j < m; j++) row[j] = gamma[Math.abs(i - j)];
    row[m] = gamma[i + 1];
    return row;
  });

  for (let col = 0; col < m; col++) {
    let piv = col;
    for (let r = col + 1; r < m; r++) {
      if (Math.abs(aug[r][col]) > Math.abs(aug[piv][col])) piv = r;
    }
    if (Math.abs(aug[piv][col]) < 1e-12) {
      throw new Error("fitArPooled: degenerate input (singular autocovariance matrix)");
    }
    if (piv !== col) [aug[col], aug[piv]] = [aug[piv], aug[col]];
    for (let r = col + 1; r < m; r++) {
      const f = aug[r][col] / aug[col][col];
      for (let c = col; c <= m; c++) aug[r][c] -= f * aug[col][c];
    }
  }

  const phi = new Array<number>(m).fill(0);
  for (let r = m - 1; r >= 0; r--) {
    let acc = aug[r][m];
    for (let c = r + 1; c < m; c++) acc -= aug[r][c] * phi[c];
    phi[r] = acc / aug[r][r];
  }

  const intercept = mean * (1 - phi.reduce((a, b) => a + b, 0));
  return { intercept, ar: phi };
}

export function innovationStd(perYearSeries: number[][], order: number, coeffs: ArCoeffs): number {
  if (!Number.isInteger(order) || order < 1) {
    throw new Error(`innovationStd: order must be a positive integer, got ${order}`);
  }
  if (!Array.isArray(perYearSeries) || perYearSeries.length === 0) {
    throw new Error("innovationStd: at least one series required");
  }
  let sumSq = 0;
  let n = 0;
  for (const s of perYearSeries) {
    if (s.length <= order) {
      throw new Error(`innovationStd: series of length ${s.length} must be longer than order ${order}`);
    }
    for (let t = order; t < s.length; t++) {
      let pred = coeffs.intercept;
      for (let i = 1; i <= order; i++) pred += coeffs.ar[i - 1] * s[t - i];
      const r = s[t] - pred;
      sumSq += r * r;
    }
    n += s.length;
  }
  const dof = n - order - 1;
  if (dof < 1) throw new Error("innovationStd: insufficient degrees of freedom");
  return Math.sqrt(sumSq / dof);
}

export function evaluateHorizon(
  trainCoeffs: ArCoeffs,
  testSeries: number[],
  horizon: number,
): HorizonEvaluation {
  if (!Number.isInteger(horizon) || horizon < 1) {
    throw new Error(`evaluateHorizon: horizon must be a positive integer, got ${horizon}`);
  }
  if (!Array.isArray(trainCoeffs.ar) || trainCoeffs.ar.length === 0) {
    throw new Error("evaluateHorizon: coeffs must contain at least one AR coefficient");
  }
  if (!Array.isArray(testSeries)) throw new Error("evaluateHorizon: testSeries must be an array");
  for (const v of testSeries) {
    if (!isFinite(v)) throw new Error("evaluateHorizon: test series contains non-finite values");
  }
  const order = trainCoeffs.ar.length;
  const start = horizon + order - 1;
  if (testSeries.length <= start) {
    throw new Error(
      `evaluateHorizon: test series of length ${testSeries.length} too short for horizon ${horizon} / order ${order}`,
    );
  }

  let perSe = 0;
  let perAe = 0;
  let perApe = 0;
  let arSe = 0;
  let arAe = 0;
  let arApe = 0;
  let apeCount = 0;
  let count = 0;
  for (let t = start; t < testSeries.length; t++) {
    const y = testSeries[t];
    const dPer = y - testSeries[t - horizon];
    perSe += dPer * dPer;
    perAe += Math.abs(dPer);
    let pred = trainCoeffs.intercept;
    for (let i = 1; i <= order; i++) pred += trainCoeffs.ar[i - 1] * testSeries[t - horizon - (i - 1)];
    const dAr = y - pred;
    arSe += dAr * dAr;
    arAe += Math.abs(dAr);
    if (Math.abs(y) > 1.0) {
      perApe += Math.abs(dPer) / Math.abs(y);
      arApe += Math.abs(dAr) / Math.abs(y);
      apeCount++;
    }
    count++;
  }
  const perMape = apeCount > 0 ? perApe / apeCount : 0;
  const arMape = apeCount > 0 ? arApe / apeCount : 0;
  return {
    ar: { rmse: Math.sqrt(arSe / count), mae: arAe / count, mape: arMape },
    persistence: { rmse: Math.sqrt(perSe / count), mae: perAe / count, mape: perMape },
  };
}

export function runCalibration(
  region: string,
  trainYears: number[],
  testYear: number,
  orders: number[],
  horizons: number[],
  trainSeries?: number[][],
  testSeries?: number[],
): CalibrationBundle {
  const uniqueOrders = [...new Set(orders)].filter((o) => Number.isInteger(o) && o >= 1);
  if (uniqueOrders.length === 0) throw new Error("runCalibration: at least one valid order required");
  const uniqueHorizons = [...new Set(horizons)].filter((h) => Number.isInteger(h) && h >= 1);
  if (uniqueHorizons.length === 0) throw new Error("runCalibration: at least one valid horizon required");

  const train = trainSeries ?? trainYears.map((y) => loadRawYear(region, y));
  const test = testSeries ?? loadRawYear(region, testYear);

  const moments = computePooledMoments(train, Math.max(2, ...uniqueOrders));

  const orderInfo = new Map<number, CalibrationOrderInfo>();
  for (const o of uniqueOrders) {
    const coeffs = fitArPooled(train, o);
    orderInfo.set(o, { coeffs, innovationStd: innovationStd(train, o, coeffs) });
  }

  const sigmaStar = orderInfo.has(1)
    ? orderInfo.get(1)!.innovationStd
    : innovationStd(train, 1, fitArPooled(train, 1));

  const evaluation: CalibrationRow[] = [];
  for (const o of uniqueOrders) {
    const coeffs = orderInfo.get(o)!.coeffs;
    for (const h of uniqueHorizons) {
      const ev = evaluateHorizon(coeffs, test, h);
      evaluation.push({ horizon: h, order: o, model: "ar", rmse: ev.ar.rmse, mae: ev.ar.mae, mape: ev.ar.mape });
      evaluation.push({
        horizon: h,
        order: o,
        model: "persistence",
        rmse: ev.persistence.rmse,
        mae: ev.persistence.mae,
        mape: ev.persistence.mape,
      });
    }
  }

  const ordersRecord: Record<string, CalibrationOrderInfo> = {};
  for (const o of uniqueOrders) ordersRecord[String(o)] = orderInfo.get(o)!;

  return {
    region,
    trainYears: [...trainYears],
    testYear,
    trainMean: moments.mean,
    trainStd: moments.std,
    trainCv: moments.std / moments.mean,
    lag1AutoCorr: moments.gamma[1] / moments.gamma[0],
    lag2AutoCorr: moments.gamma[2] / moments.gamma[0],
    sigmaStar,
    orders: ordersRecord,
    evaluation,
  };
}

function writeCsv(region: string, bundle: CalibrationBundle): void {
  const header = "region,order,model,horizon,rmse,mae,mape";
  const rows = bundle.evaluation.map((r) =>
    [
      region,
      r.order,
      r.model,
      r.horizon,
      r.rmse.toPrecision(6),
      r.mae.toPrecision(6),
      r.mape.toPrecision(6),
    ].join(","),
  );
  writeFileSync(resolve(OUT_DIR, `calibration_${region}.csv`), header + "\n" + rows.join("\n") + "\n", "utf-8");
}

function printSummary(bundle: CalibrationBundle, orders: number[], horizons: number[]): void {
  const line = "\u2500".repeat(52);
  console.log(`\n  ${line}`);
  console.log(`  Forecast calibration \u2500 ${bundle.region} (train ${bundle.trainYears.join(",")} / test ${bundle.testYear})`);
  console.log(
    `  mean=${bundle.trainMean.toFixed(2)}  std=${bundle.trainStd.toFixed(2)}  cv=${bundle.trainCv.toFixed(4)}` +
      `  lag1=${bundle.lag1AutoCorr.toFixed(6)}  lag2=${bundle.lag2AutoCorr.toFixed(6)}  \u03C3*=${bundle.sigmaStar.toFixed(2)}`,
  );
  for (const o of orders) {
    const info = bundle.orders[String(o)];
    if (!info) continue;
    console.log(`  order ${o}  \u03C6=[${info.coeffs.ar.map((v) => v.toFixed(4)).join(", ")}]  \u03C3*=${info.innovationStd.toFixed(2)}`);
    console.log("    h    AR rmse   PERS rmse   gap");
    for (const h of horizons) {
      const ar = bundle.evaluation.find((r) => r.order === o && r.horizon === h && r.model === "ar");
      const per = bundle.evaluation.find((r) => r.order === o && r.horizon === h && r.model === "persistence");
      if (!ar || !per) continue;
      const gap = per.rmse - ar.rmse;
      console.log(
        `   ${String(h).padStart(3)}  ${ar.rmse.toFixed(3).padStart(8)}  ${per.rmse.toFixed(3).padStart(8)}  ${gap.toFixed(3).padStart(6)}`,
      );
    }
  }
  console.log(`  ${line}`);
}

export async function calibrateCli(raw: {
  regions?: string;
  train?: string;
  test?: string;
  orders?: string;
  horizons?: string;
}): Promise<void> {
  const regions = (raw.regions ?? "DE,IT,SE")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  const trainYears = (raw.train ?? "2022,2023,2024").split(",").map((s) => Number(s.trim()));
  const testYear = Number(raw.test ?? "2025");
  const orders = (raw.orders ?? "1,7").split(",").map((s) => Number(s.trim()));
  const horizons = (raw.horizons ?? "1,3,6,12,24,72").split(",").map((s) => Number(s.trim()));

  mkdirSync(OUT_DIR, { recursive: true });

  for (const region of regions) {
    const bundle = runCalibration(region, trainYears, testYear, orders, horizons);
    writeFileSync(resolve(OUT_DIR, `calibration_${region}.json`), JSON.stringify(bundle, null, 2) + "\n", "utf-8");
    writeCsv(region, bundle);
    printSummary(bundle, orders, horizons);
    console.log(`  JSON: publication/output/forecast/calibration_${region}.json`);
    console.log(`  CSV:  publication/output/forecast/calibration_${region}.csv`);
  }
  console.log("\n  Done.\n");
}
