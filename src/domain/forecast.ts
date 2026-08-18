import type { ArCoeffs, CO2Timeline, ForecastModel } from "./types";

export function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return function () {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand: () => number): number {
  const u1 = 1 - rand();
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function sampleNormal(rand: () => number, mean: number, sigma: number): number {
  return mean + sigma * gaussian(rand);
}

export function sampleLognormal(rand: () => number, mu: number, sigma: number): number {
  return Math.exp(mu + sigma * gaussian(rand));
}

export function fitAr(series: number[], order: number): ArCoeffs {
  if (!Number.isInteger(order) || order < 1) {
    throw new Error(`fitAr: order must be a positive integer, got ${order}`);
  }
  if (!Array.isArray(series) || series.length <= order) {
    throw new Error(`fitAr: series must be longer than order (${series.length} <= ${order})`);
  }
  for (const v of series) {
    if (!isFinite(v)) throw new Error("fitAr: series contains non-finite values");
  }

  const n = series.length;
  const mean = series.reduce((a, b) => a + b, 0) / n;
  const centered = series.map((v) => v - mean);

  const gamma = new Array<number>(order + 1);
  for (let k = 0; k <= order; k++) {
    let acc = 0;
    for (let t = k; t < n; t++) acc += centered[t] * centered[t - k];
    gamma[k] = acc / (n - k);
  }

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
      throw new Error("fitAr: degenerate input (singular autocovariance matrix)");
    }
    if (piv !== col) [aug[col], aug[piv]] = [aug[piv], aug[col]];
    for (let r = col + 1; r < m; r++) {
      const f = aug[r][col] / aug[col][col];
      for (let c = col; c <= m; c++) aug[r][c] -= f * aug[col][c];
    }
  }

  const phi = new Array<number>(m).fill(0);
  for (let r = m - 1; r >= 0; r--) {
    let s = aug[r][m];
    for (let c = r + 1; c < m; c++) s -= aug[r][c] * phi[c];
    phi[r] = s / aug[r][r];
  }

  const intercept = mean * (1 - phi.reduce((a, b) => a + b, 0));
  return { intercept, ar: phi };
}

export function forecastInnovationStd(series: number[]): number {
  const { intercept, ar } = fitAr(series, 1);
  let sumSq = 0;
  for (let t = 1; t < series.length; t++) {
    const r = series[t] - (intercept + ar[0] * series[t - 1]);
    sumSq += r * r;
  }
  const dof = series.length - 2;
  return Math.sqrt(sumSq / Math.max(dof, 1));
}

export function applyForecast(realized: CO2Timeline, model: ForecastModel, seed: number): CO2Timeline {
  const carbon = realized.carbonIntensity;
  const n = carbon.length;
  const decision = new Array<number>(n);

  if (model.type === "identity") {
    for (let t = 0; t < n; t++) decision[t] = carbon[t];
  } else if (model.type === "additive") {
    const rand = mulberry32(seed);
    for (let t = 0; t < n; t++) {
      decision[t] = Math.max(0, carbon[t] + model.sigma * gaussian(rand));
    }
  } else if (model.type === "multiplicative") {
    const rand = mulberry32(seed);
    for (let t = 0; t < n; t++) {
      decision[t] = Math.max(0, carbon[t] * Math.exp(model.sigma * gaussian(rand)));
    }
  } else if (model.type === "delay") {
    for (let t = 0; t < n; t++) {
      decision[t] = t >= model.steps ? carbon[t - model.steps] : carbon[t];
    }
  } else {
    const { order, horizon } = model;
    const coeffs = model.coeffs ?? fitAr(carbon, order);
    for (let t = 0; t < n; t++) {
      if (t < horizon + order - 1) {
        decision[t] = carbon[t];
        continue;
      }
      let v = coeffs.intercept;
      for (let i = 1; i <= order; i++) {
        v += coeffs.ar[i - 1] * carbon[t - horizon - (i - 1)];
      }
      decision[t] = Math.max(0, v);
    }
  }

  return {
    zone: realized.zone,
    years: [...realized.years],
    timestamps: [...realized.timestamps],
    carbonIntensity: decision,
  };
}
