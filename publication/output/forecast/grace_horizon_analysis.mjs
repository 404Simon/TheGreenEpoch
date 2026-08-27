#!/usr/bin/env node
// Phase B.5.1 — grace-horizon prediction validation (deterministic analysis).
// Reads ONLY committed artifacts:
//   publication/output/forecast/calibration_{DE,IT,SE}.json
//   publication/output/forecast/multiyear_fixed_summary.json
//   publication/output/forecast/multiyear_summary.json
//   publication/output/forecast/fixed_summary.json          (cross-check, 2025)
//   public/data/co2/{region}_{year}.json                    (year CI mean/median)
// Writes: grace_horizon.json (all computed numbers) and prints a summary table.
// No RNG, no writes outside this directory. Deterministic.
//
// Usage: node grace_horizon_analysis.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../../..");
const FORECAST = resolve(ROOT, "publication/output/forecast");
const CO2 = resolve(ROOT, "public/data/co2");

const REGIONS = ["DE", "IT", "SE"];
const YEARS = [2023, 2024, 2025];
const GRID = [1, 3, 6, 12, 24, 72];
const OUTLIER = { region: "IT", year: 2023 }; // documented B.3.2 outlier (right-censored grace at grid max)

function readJson(p) {
  return JSON.parse(readFileSync(p, "utf-8"));
}

const calibration = Object.fromEntries(
  REGIONS.map((r) => [r, readJson(resolve(FORECAST, `calibration_${r}.json`))])
);
const multiFixed = readJson(resolve(FORECAST, "multiyear_fixed_summary.json"));
const multiSum = readJson(resolve(FORECAST, "multiyear_summary.json"));
const fixedSum = readJson(resolve(FORECAST, "fixed_summary.json"));

// ── per-region calibration facts ──────────────────────────────────────────
function regionFacts(r) {
  const c = calibration[r];
  const phi = c.orders["1"].coeffs.ar[0];
  const sigmaStar = c.sigmaStar;
  const trainMean = c.trainMean;
  const evalRows = c.evaluation.filter((e) => e.order === 1 && e.model === "ar");
  const rmseByH = Object.fromEntries(evalRows.map((e) => [e.horizon, e.rmse]));
  const thByH = Object.fromEntries(
    GRID.map((h) => [h, sigmaStar * Math.sqrt((1 - Math.pow(phi, 2 * h)) / (1 - phi * phi))])
  );
  return { phi, sigmaStar, trainMean, rmseByH, thByH };
}
const FACTS = Object.fromEntries(REGIONS.map((r) => [r, regionFacts(r)]));

// ── year CI statistics (committed data) ───────────────────────────────────
function yearStats(r, y) {
  const d = readJson(resolve(CO2, `${r}_${y}.json`));
  const a = d.carbonIntensity.slice().sort((x, y) => x - y);
  const n = a.length;
  const mean = a.reduce((s, x) => s + x, 0) / n;
  const median = n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2;
  return { mean, median };
}

// ── empirical grace + theta_p per (region, year) ──────────────────────────
const graceBy = Object.fromEntries(
  multiFixed.map((row) => [`${row.region}/${row.year}`, row])
);
const thetaP = Object.fromEntries(
  multiSum.flatMap((s) =>
    s.per_year.map((p) => [`${s.region}/${p.year}`, p.theta_p])
  )
);

const MU_OPTIONS = {
  trainMean: { label: "μ = calibration trainMean (2022–24)", src: "calibration_{region}.json → trainMean" },
  yearMean: { label: "μ = decision-year mean CI (5-min)", src: "public/data/co2/{region}_{year}.json → mean(carbonIntensity)" },
  yearMedian: { label: "μ = decision-year median CI (5-min)", src: "public/data/co2/{region}_{year}.json → median(carbonIntensity)" },
};

// ── build point table ─────────────────────────────────────────────────────
const points = []; // { region, year, g, rmseG, thetaP, S:{mu}, k:{mu}, isOutlier }
for (const r of REGIONS) {
  for (const y of YEARS) {
    const key = `${r}/${y}`;
    const row = graceBy[key];
    const g = row.grace_persistence; // paper uses the persistence/delay family
    const gDelay = row.grace_delay;
    const rmseG = FACTS[r].rmseByH[g];
    const tp = thetaP[key];
    const ys = yearStats(r, y);
    const S = {
      trainMean: Math.abs(tp - FACTS[r].trainMean),
      yearMean: Math.abs(tp - ys.mean),
      yearMedian: Math.abs(tp - ys.median),
    };
    const isOutlier = r === OUTLIER.region && y === OUTLIER.year;
    points.push({
      region: r,
      year: y,
      g,
      gDelay,
      s0: row.s0,
      h72DegFrac: row.degradation_h72_persistence_frac,
      rmseG,
      thetaP: tp,
      yearMeanCi: ys.mean,
      yearMedianCi: ys.median,
      S,
      k: Object.fromEntries(Object.keys(S).map((m) => [m, rmseG / S[m]])),
      isOutlier,
    });
  }
}

// ── fits ──────────────────────────────────────────────────────────────────
function mean(xs) {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
// through-origin: y = k*x ; R² = 1 − Σ(y−kx)² / Σ(y−ȳ)²  (ȳ = mean of y, so the
// through-origin R² is comparable to the intercept R² on the same y-scale).
function fitThroughOrigin(xs, ys) {
  const k = xs.reduce((s, x, i) => s + x * ys[i], 0) / xs.reduce((s, x) => s + x * x, 0);
  const my = mean(ys);
  const sse = ys.reduce((s, y, i) => s + (y - k * xs[i]) ** 2, 0);
  const sst = ys.reduce((s, y) => s + (y - my) ** 2, 0);
  return { k, r2: 1 - sse / sst, sse, sst, n: xs.length };
}
function fitIntercept(xs, ys) {
  const n = xs.length;
  const mx = mean(xs);
  const my = mean(ys);
  const sxx = xs.reduce((s, x) => s + (x - mx) ** 2, 0);
  const sxy = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0);
  const b = sxy / sxx;
  const a = my - b * mx;
  const sse = ys.reduce((s, y, i) => s + (y - (a + b * xs[i])) ** 2, 0);
  const sst = ys.reduce((s, y) => s + (y - my) ** 2, 0);
  return { a, b, r2: 1 - sse / sst, sse, sst, n };
}

const FIT_SAMPLES = {
  all: { label: "all 9 (region, year) points", sel: (p) => true },
  noIT23: {
    label: "8 points, IT-2023 outlier excluded",
    sel: (p) => !p.isOutlier,
  },
};

const fits = {};
for (const mu of Object.keys(MU_OPTIONS)) {
  fits[mu] = {};
  for (const [sampleName, sample] of Object.entries(FIT_SAMPLES)) {
    const sel = points.filter(sample.sel);
    const xs = sel.map((p) => p.S[mu]);
    const ys = sel.map((p) => p.rmseG);
    fits[mu][sampleName] = {
      throughOrigin: fitThroughOrigin(xs, ys),
      intercept: fitIntercept(xs, ys),
    };
  }
}

// ── predicted grace horizon ───────────────────────────────────────────────
// Primary: empirical AR(1) evaluation RMSE crosses k·S (through-origin k from the
// noIT23 fit). Comparison: theoretical AR(1) RMSE crossing the same level.
function predGrace(rmseFn, target) {
  for (const h of GRID) {
    if (rmseFn(h) >= target) return h;
  }
  return null; // never reaches within grid (g_pred > 72)
}
// Continuous crossing of a monotone RMSE curve: linear interpolation in log(h)
// between the bracketing grid points (RMSE is near-log-linear in h).
function predGraceCont(rmseByH, target) {
  if (target <= rmseByH[1]) return 1;
  if (target > rmseByH[72]) return null;
  for (let i = 0; i < GRID.length - 1; i++) {
    const h0 = GRID[i];
    const h1 = GRID[i + 1];
    const r0 = rmseByH[h0];
    const r1 = rmseByH[h1];
    if (target >= r0 && target <= r1) {
      const frac = (target - r0) / (r1 - r0);
      return Math.round(Math.exp(Math.log(h0) + (Math.log(h1) - Math.log(h0)) * frac));
    }
  }
  return null;
}
const kPrimary = fits.yearMean.noIT23.throughOrigin.k;
const predicted = points.map((p) => {
  const target = kPrimary * p.S.yearMean;
  const gPred = predGrace((h) => FACTS[p.region].rmseByH[h], target);
  const gPredTh = predGrace((h) => FACTS[p.region].thByH[h], target);
  const gPredCont = predGraceCont(FACTS[p.region].rmseByH, target);
  const idx = (h) => GRID.indexOf(h);
  const within1 = gPred !== null && Math.abs(idx(gPred) - idx(p.g)) <= 1;
  return {
    region: p.region,
    year: p.year,
    gEmpirical: p.g,
    gPred,
    gPredCont,
    gPredTheoretical: gPredTh,
    target: kPrimary * p.S.yearMean,
    kPrimary,
    within1Step: within1,
    dev: gPred === null ? null : gPred - p.g,
    devCont: gPredCont === null ? null : gPredCont - p.g,
  };
});

// ── theoretical vs empirical RMSE comparison (model under-widening) ────────
const rmseCompare = Object.fromEntries(
  REGIONS.map((r) => [
    r,
    GRID.map((h) => ({
      h,
      empirical: FACTS[r].rmseByH[h],
      theoretical: FACTS[r].thByH[h],
      ratio: FACTS[r].rmseByH[h] / FACTS[r].thByH[h],
    })),
  ])
);

// ── cross-check vs fixed_summary.json 2025 grace levels ────────────────────
const cross2025 = Object.fromEntries(
  fixedSum.map((row) => [
    row.region,
    { persistence: row.graceLevels.persistence.horizon, delay: row.graceLevels.delay.steps },
  ])
);
const multi2025 = Object.fromEntries(
  graceBy
    ? Object.entries(graceBy)
        .filter(([k]) => k.endsWith("/2025"))
        .map(([k, v]) => [
          v.region,
          { persistence: v.grace_persistence, delay: v.grace_delay },
        ])
    : []
);
const crossCheck = REGIONS.map((r) => ({
  region: r,
  fixedSummary: cross2025[r],
  multiyear2025: multi2025[r],
  match: cross2025[r].persistence === multi2025[r].persistence && cross2025[r].delay === multi2025[r].delay,
}));

// ── verdict ───────────────────────────────────────────────────────────────
const r2All = fits.yearMean.all.throughOrigin.r2;
const r2NoIT23 = fits.yearMean.noIT23.throughOrigin.r2;

const out = {
  method: {
    note: "B.5.1 grace-horizon validation. Empirical grace g = grace_persistence (paper uses the persistence/delay family) from multiyear_fixed_summary.json; delay == persistence in all 9 cells (only arma differs, SE-2023: arma 24 vs delay/persistence 12). RMSE at g = empirical AR(1) (order 1, model 'ar') evaluation RMSE at horizon g from calibration_{region}.json; every g lies on the grid {1,3,6,12,24,72} so no interpolation is needed. The calibration bundle is evaluated on test year 2025; that bundle is reused for 2023/2024 exactly as in the B.3.2 multiyear sweep (documented B.3 choice). S = |theta_p − μ|; μ = decision-year mean CI (primary); trainMean and year-median reported as sensitivity. The SPEC relation RMSE(g) ≈ k·S is fit through the origin (y = k·x) and with an intercept; R² = 1 − SSE/SST on the same y-scale for both.",
    mu: MU_OPTIONS,
    grid: GRID,
    outlier: OUTLIER,
    outlierNote:
      "IT-2023 grace = 72 is right-censored on the grid: h=72 persistence degradation is only 3.1% (never crosses the 10% threshold inside the grid), so the empirical grace is not an interior crossing and the point is excluded from the primary fit.",
    signIssue:
      "theta_p < mu in all nine (region, year) cells (the control is pause-mostly / resume-on-clean-windows; e.g. DE 2025 theta_p=272.37 < year mean 339.94). S is therefore taken in absolute value; the excursion magnitude — not its sign — sets the forecast-error budget.",
    theoreticalNote:
      "Theoretical AR(1) h-step RMSE sigma*·sqrt((1−phi^(2h))/(1−phi^2)) grossly under-predicts the empirical evaluation RMSE (ratio ≈ 3–4 at h=72); this is the documented model under-widening (B.1 review M1). The empirical evaluation RMSE is the primary curve; the theoretical curve is reported as comparison and is NOT used for g_pred.",
  },
  calibrationFacts: Object.fromEntries(
    REGIONS.map((r) => [
      r,
      { phi: FACTS[r].phi, sigmaStar: FACTS[r].sigmaStar, trainMean: FACTS[r].trainMean },
    ])
  ),
  points,
  predicted,
  rmseEmpiricalVsTheoretical: rmseCompare,
  fits,
  kPrimary,
  crossCheck,
  verdict: {
    r2All,
    r2NoIT23,
    headline:
      r2NoIT23 >= 0.9
        ? "CONDITIONAL HEADLINE: R² ≈ " + r2NoIT23.toFixed(3) +
          " (8 points, IT-2023 censored outlier excluded). Not a universal law: with IT-2023 included R² drops to " +
          r2All.toFixed(3) + "."
        : "NOT headline; report as design-rule observation.",
    verdictSource: "fits.yearMean.{all,noIT23}.throughOrigin.r2",
  },
  generatedDate: "2026-08-19",
};

writeFileSync(resolve(HERE, "grace_horizon.json"), JSON.stringify(out, null, 2) + "\n");

// ── console summary ───────────────────────────────────────────────────────
console.log("=== B.5.1 grace-horizon prediction — analysis summary ===");
console.log("Primary μ = decision-year mean CI; k from through-origin fit (8 pts, no IT-2023).\n");
console.log(
  "region year  g  RMSE(g)   theta_p  mu(yrMean)  S       k_implied  gPred grid gPred cont within1"
);
for (const p of points) {
  const pr = predicted.find((x) => x.region === p.region && x.year === p.year);
  console.log(
    `${p.region}    ${p.year}  ${String(p.g).padEnd(3)} ${p.rmseG.toFixed(2).padStart(7)}  ` +
      `${p.thetaP.toFixed(2).padStart(8)}  ${p.yearMeanCi.toFixed(2).padStart(9)}  ` +
      `${p.S.yearMean.toFixed(2).padStart(6)}  ${p.k.yearMean.toFixed(3).padStart(9)}  ` +
      `${pr.gPred === null ? ">72" : pr.gPred}  ${pr.gPredCont === null ? ">72" : pr.gPredCont}  ${pr.within1Step ? "yes" : "no"}${p.isOutlier ? "  <-- outlier" : ""}`
  );
}
console.log("\nFits (y = RMSE(g), x = S):");
for (const mu of Object.keys(MU_OPTIONS)) {
  for (const [sn, s] of Object.entries(FIT_SAMPLES)) {
    const f = fits[mu][sn];
    console.log(
      `  ${mu.padEnd(10)} ${sn.padEnd(8)} n=${f.throughOrigin.n}  through-origin k=${f.throughOrigin.k.toFixed(4)} R²=${f.throughOrigin.r2.toFixed(4)}  |  intercept a=${f.intercept.a.toFixed(3)} b=${f.intercept.b.toFixed(4)} R²=${f.intercept.r2.toFixed(4)}`
    );
  }
}
console.log(`\nk (primary, yearMean, noIT23, through-origin) = ${kPrimary.toFixed(4)}`);
console.log(`Verdict: ${out.verdict.headline}`);
console.log("\nTheoretical vs empirical AR(1) RMSE (ratio = empirical/theoretical):");
for (const r of REGIONS) {
  console.log(
    `  ${r}: ` +
      GRID.map((h) => `h=${h} ${rmseCompare[r].find((x) => x.h === h).empirical.toFixed(1)}/${rmseCompare[r].find((x) => x.h === h).theoretical.toFixed(1)}`)
        .join("  ")
  );
}
console.log("\nWrote publication/output/forecast/grace_horizon.json");
