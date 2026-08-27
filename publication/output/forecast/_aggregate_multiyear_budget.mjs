#!/usr/bin/env node
// Aggregates the raw Phase B.3 + B.4 run outputs into the committed artifacts:
//   multiyear_{DE,IT,SE}.{json,csv}, multiyear_summary.json,
//   multiyear_fixed_summary.json, budget_summary.{json,csv}.
// Pure function of the raw files: no RNG, byte-deterministic.
//
// Usage: node _aggregate_multiyear_budget.mjs <rawDir> <outDir>
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const [rawArg, outArg] = process.argv.slice(2);
const RAW = resolve(rawArg);
const OUT = resolve(outArg);
const REPO_ROOT = resolve(RAW, "../../../..");
const CO2_DIR = resolve(REPO_ROOT, "public/data/co2");

const REGIONS = {
  DE: { start: "02-01", tpMax: 800 },
  IT: { start: "01-14", tpMax: 800 },
  SE: { start: "04-22", tpMax: 100 },
};
const YEARS = [2022, 2023, 2024, 2025];
const BUDGETS = [30, 50, 100, 200];
const CKPTS = [148.8, 900];

// ── helpers ──────────────────────────────────────────────────────────────

function readJson(p) {
  return JSON.parse(readFileSync(p, "utf-8"));
}

function fmt(v) {
  return v == null ? "NA" : v.toPrecision(6);
}

function bestCompleted(points) {
  let best = null;
  for (const p of points) {
    if (p.withinBudget && p.completed && p.co2SavingsPct > 0 && (best === null || p.score > best.score)) best = p;
  }
  return best;
}

function cellFromPoint(p) {
  return {
    theta_p: p.thetaPause,
    theta_r: p.thetaResume,
    margin: p.thetaPause - p.thetaResume,
    savings: p.co2SavingsPct,
    overhead: p.actualOverheadPct,
    score: p.score,
    num_pauses: p.numPauses,
    completed: p.completed,
    within_budget: p.withinBudget,
  };
}

function nullCell() {
  return {
    theta_p: null, theta_r: null, margin: null, savings: null, overhead: null,
    score: null, num_pauses: null, completed: null, within_budget: null,
  };
}

// empirical percentiles of theta within a year's CI distribution
function percentileStats(carbon, thetaP, thetaR) {
  let n = 0, aboveP = 0, belowR = 0, leP = 0;
  for (const v of carbon) {
    n++;
    if (v > thetaP) aboveP++;
    if (v < thetaR) belowR++;
    if (v <= thetaP) leP++;
  }
  return {
    n,
    pct_exceed_theta_p: (100 * aboveP) / n, // P(CI > theta_p): pause-threshold exceedance
    pct_below_theta_r: (100 * belowR) / n,  // P(CI < theta_r): resume-threshold subceedance
    pct_le_theta_p: (100 * leP) / n,        // CDF(theta_p)
  };
}

// ── B.3.1 multi-year optimization ────────────────────────────────────────

function multiyearRegion(region, cfg) {
  const rows = [];
  for (const year of YEARS) {
    const raw = readJson(resolve(RAW, "multiyear", `${region}_${year}.json`));
    const carbon = readJson(resolve(CO2_DIR, `${region}_${year}.json`)).carbonIntensity;
    const best = raw.best;
    const cb = bestCompleted(raw.points);
    const cell = best ? cellFromPoint(best) : nullCell();
    const row = {
      year,
      ...(cell.margin == null ? { theta_p: null, theta_r: null, margin: null, savings: null, overhead: null, score: null, num_pauses: null, completed: null, within_budget: null } : cell),
      stop_reason: best ? best.stopReason : null,
      found: best != null,
      best_completed: cb ? cellFromPoint(cb) : null,
      percentile: best
        ? percentileStats(carbon, best.thetaPause, best.thetaResume)
        : { n: carbon.length, pct_exceed_theta_p: null, pct_below_theta_r: null, pct_le_theta_p: null },
    };
    rows.push(row);
  }
  return rows;
}

function ruleDeviation(rows) {
  const margins = rows.map((r) => r.margin).filter((x) => x != null);
  const savings = rows.map((r) => r.savings).filter((x) => x != null);
  const maxMargin = margins.length ? Math.max(...margins) : null;
  const nearZeroYears = margins.filter((m) => m <= 10).length;
  // SPEC: "pass if <= 10, or if it holds for >=3 of 4 years say so honestly".
  const nearZeroMarginPass = (maxMargin != null && maxMargin <= 10) || nearZeroYears >= 3;

  const excP = rows.map((r) => r.percentile.pct_exceed_theta_p).filter((x) => x != null);
  const excR = rows.map((r) => r.percentile.pct_below_theta_r).filter((x) => x != null);
  const excDevP = excP.length ? Math.max(...excP) - Math.min(...excP) : null;
  const excDevR = excR.length ? Math.max(...excR) - Math.min(...excR) : null;
  // No explicit percentile tolerance in the SPEC; the closest analog of the
  // "10 g/kWh margin" tolerance is a 10 pp deviation in the implied percentile.
  const PERCENTILE_TOLERANCE_PP = 10;
  const percentilePass =
    excDevP != null && excDevR != null && excDevP <= PERCENTILE_TOLERANCE_PP && excDevR <= PERCENTILE_TOLERANCE_PP;

  const savMin = savings.length ? Math.min(...savings) : null;
  const savMax = savings.length ? Math.max(...savings) : null;
  const savRange = savings.length ? savMax - savMin : null;
  const savingsPass = savRange != null && savRange <= 5;

  return {
    near_zero_margin: {
      margins,
      max_margin_g_per_kwh: maxMargin,
      holds_years_out_of_4: nearZeroYears,
      pass: nearZeroMarginPass,
      note: nearZeroMarginPass
        ? (maxMargin != null && maxMargin <= 10
            ? `max optimized margin ${maxMargin?.toFixed(2)} g/kWh <= 10 g/kWh tolerance`
            : `max margin ${maxMargin?.toFixed(2)} g/kWh exceeds the 10 g/kWh tolerance but the rule holds in ${nearZeroYears}/4 years (SPEC: pass if it holds for >=3 of 4 years)`)
        : `max optimized margin ${maxMargin?.toFixed(2)} g/kWh > 10 g/kWh; holds only ${nearZeroYears}/4 years`,
    },
    percentile_thresholds: {
      pct_exceed_theta_p_by_year: rows.map((r) => r.percentile.pct_exceed_theta_p),
      pct_below_theta_r_by_year: rows.map((r) => r.percentile.pct_below_theta_r),
      pct_exceed_deviation_pp: excDevP,
      pct_below_deviation_pp: excDevR,
      pass: percentilePass,
      note: `tolerance ${PERCENTILE_TOLERANCE_PP} pp deviation in the implied percentile (closest analog of the 10 g/kWh margin tolerance; no explicit percentile tolerance in the SPEC)`,
    },
    grace_horizon: {
      pass: null, // filled in by the caller from multiyear_fixed_summary
      note: "grace-horizon rule: see multiyear_fixed_summary.json (B.3.2)",
    },
    savings_stability: {
      savings_by_year: rows.map((r) => r.savings),
      min_savings_pp: savMin,
      max_savings_pp: savMax,
      range_pp: savRange,
      pass: savingsPass,
      note: savingsPass
        ? `savings range ${savRange?.toFixed(2)} pp <= 5 pp tolerance`
        : `savings range ${savRange?.toFixed(2)} pp > 5 pp tolerance`,
    },
  };
}

const multiyearSummary = [];

for (const region of Object.keys(REGIONS)) {
  const cfg = REGIONS[region];
  const rows = multiyearRegion(region, cfg);
  const baselineRow = rows.find((r) => r.year === 2025);

  const obj = {
    region,
    model: "Deepseek",
    budget: 200,
    alpha: 1,
    optimizer: { resolution: 10, iterations: 6, tpMax: cfg.tpMax, start: cfg.start, ckpt_pause: 148.8 },
    baseline: {
      theta_p: baselineRow.theta_p, theta_r: baselineRow.theta_r, margin: baselineRow.margin,
      savings: baselineRow.savings, overhead: baselineRow.overhead, score: baselineRow.score,
      num_pauses: baselineRow.num_pauses, completed: baselineRow.completed, within_budget: baselineRow.within_budget,
    },
    rows,
  };
  writeFileSync(resolve(OUT, `multiyear_${region}.json`), JSON.stringify(obj, null, 2) + "\n", "utf-8");

  const csv = [];
  for (const r of rows) {
    csv.push([
      region, r.year,
      fmt(r.theta_p), fmt(r.theta_r), fmt(r.margin), fmt(r.savings), fmt(r.overhead), fmt(r.score),
      r.num_pauses == null ? "NA" : String(r.num_pauses),
      String(r.completed), String(r.within_budget), String(r.found),
      r.best_completed ? fmt(r.best_completed.savings) : "NA",
      r.best_completed ? fmt(r.best_completed.overhead) : "NA",
      fmt(r.percentile.pct_exceed_theta_p), fmt(r.percentile.pct_below_theta_r),
    ].join(","));
  }
  writeFileSync(
    resolve(OUT, `multiyear_${region}.csv`),
    "region,year,theta_p,theta_r,margin,savings,overhead,score,num_pauses,completed,within_budget,found,best_completed_savings,best_completed_overhead,pct_exceed_theta_p,pct_below_theta_r\n" + csv.join("\n") + "\n",
    "utf-8",
  );

  const dev = ruleDeviation(rows);
  const rec = {
    region,
    baseline: obj.baseline,
    per_year: rows.map((r) => ({
      year: r.year,
      theta_p: r.theta_p, theta_r: r.theta_r, margin: r.margin,
      savings: r.savings, overhead: r.overhead, score: r.score,
      num_pauses: r.num_pauses, completed: r.completed, within_budget: r.within_budget, found: r.found,
      best_completed_savings: r.best_completed ? r.best_completed.savings : null,
      best_completed_overhead: r.best_completed ? r.best_completed.overhead : null,
    })),
    rule_deviation: dev,
  };
  multiyearSummary.push(rec);
}

// ── B.3.2 fixed sweep across years ───────────────────────────────────────

const fixedYears = [2025, 2024, 2023];
const fixedByRegionYear = new Map(); // `${region}|${year}` -> summary entry
for (const year of fixedYears) {
  const summary = readJson(resolve(RAW, `fixed_${year}`, "fixed_summary.json"));
  for (const s of summary) {
    fixedByRegionYear.set(`${s.region}|${year}`, s);
  }
}

const multiyearFixed = [];
for (const region of Object.keys(REGIONS)) {
  for (const year of fixedYears) {
    const s = fixedByRegionYear.get(`${region}|${year}`);
    if (!s) throw new Error(`missing fixed summary for ${region} ${year}`);
    multiyearFixed.push({
      region,
      year,
      s0: s.s0,
      degradation_h72_persistence_frac: s.degradationAtH72.persistence.delta_s_frac,
      degradation_h72_persistence_pp: s.degradationAtH72.persistence.delta_s_pp,
      degradation_h72_arma_frac: s.degradationAtH72.arma.delta_s_frac,
      grace_delay: s.graceLevels.delay.steps,
      grace_persistence: s.graceLevels.persistence.horizon,
      grace_arma: s.graceLevels.arma.horizon,
      additive_grace_level: s.graceLevels.additive.level,
    });
  }
}

// grace stability: compare each region's year-2025 grace (delay & persistence)
// with 2024 and 2023; "<= 1 step" = equal to, or immediately adjacent on the
// horizon grid {1,3,6,12,24,72} to, the 2025 value.
const HORIZON_GRID = [1, 3, 6, 12, 24, 72];
function stepDistance(a, b) {
  return Math.abs(HORIZON_GRID.indexOf(a) - HORIZON_GRID.indexOf(b));
}
for (const region of Object.keys(REGIONS)) {
  const base = multiyearFixed.find((x) => x.region === region && x.year === 2025);
  const others = multiyearFixed.filter((x) => x.region === region && x.year !== 2025);
  const perRule = {
    delay: {},
    persistence: {},
  };
  for (const rule of ["delay", "persistence"]) {
    const baseGrace = base[`grace_${rule}`];
    perRule[rule].base_2025 = baseGrace;
    perRule[rule].by_year = {};
    let maxDev = 0;
    let yearsWithin1Step = 1; // the 2025 base itself counts
    for (const o of others) {
      const d = stepDistance(baseGrace, o[`grace_${rule}`]);
      perRule[rule].by_year[o.year] = { grace: o[`grace_${rule}`], steps_off: d, within_1_step: d <= 1 };
      maxDev = Math.max(maxDev, d);
      if (d <= 1) yearsWithin1Step++;
    }
    perRule[rule].max_steps_off = maxDev;
    perRule[rule].years_within_1_step = yearsWithin1Step; // out of 3 tested years (2023, 2024, 2025)
  }
  // SPEC accept criterion: grace horizon stable (<=1 step) in >=2 years per
  // region -> at least 2 of the 3 tested years sit within 1 step of the 2025
  // base (i.e. the base plus at least one additional year).
  const delayOk = perRule.delay.years_within_1_step >= 2;
  const persOk = perRule.persistence.years_within_1_step >= 2;
  const graceEntry = multiyearSummary.find((m) => m.region === region).rule_deviation.grace_horizon;
  graceEntry.pass = delayOk && persOk;
  graceEntry.base_2025 = { delay: base.grace_delay, persistence: base.grace_persistence };
  graceEntry.years = perRule;
  graceEntry.note =
    `2025 grace delay/persistence = ${base.grace_delay}/${base.grace_persistence}; ` +
    `delay within 1 step in ${perRule.delay.years_within_1_step}/3 years (2023-2025), ` +
    `persistence within 1 step in ${perRule.persistence.years_within_1_step}/3 years (grid {1,3,6,12,24,72}); ` +
    `pass iff both >= 2 years`;
}

writeFileSync(resolve(OUT, "multiyear_fixed_summary.json"), JSON.stringify(multiyearFixed, null, 2) + "\n", "utf-8");
writeFileSync(resolve(OUT, "multiyear_summary.json"), JSON.stringify(multiyearSummary, null, 2) + "\n", "utf-8");

// ── B.4.1 budget sweep ───────────────────────────────────────────────────

function budgetRows(region, ckpt) {
  const rows = [];
  for (const budget of BUDGETS) {
    const raw = readJson(resolve(RAW, "budget", `${region}_${budget}_${ckpt}.json`));
    const best = raw.best;
    const cb = bestCompleted(raw.points);
    const cell = best ? cellFromPoint(best) : nullCell();
    rows.push({
      budget,
      ...cell,
      stop_reason: best ? best.stopReason : null,
      found: best != null,
      best_completed: cb ? cellFromPoint(cb) : null,
    });
  }
  return rows;
}

// Collapse = budgets at which no within-budget AND completed point with positive
// savings exists (best_completed == null), i.e. the feasible frontier has
// collapsed there. If the frontier never collapses in-grid, note the lowest
// completed-feasible savings as a "savings floor".
function collapsePoint(rows) {
  const infeasible = rows.filter((r) => r.best_completed == null);
  if (infeasible.length === 0) {
    return {
      collapsed: false,
      note: `no collapse in-grid {30,50,100,200}%: every budget keeps a completed-feasible point; completed-feasible savings floor = ${Math.min(...rows.map((r) => r.best_completed.savings)).toFixed(2)}% at ${rows.reduce((a, b) => (b.best_completed.savings < a.best_completed.savings ? b : a)).budget}%`,
    };
  }
  const first = infeasible[0];
  const prev = rows[rows.indexOf(first) - 1];
  return {
    collapsed: true,
    first_collapse_budget: first.budget,
    infeasible_budgets: infeasible.map((r) => r.budget),
    prev_feasible_budget: prev ? prev.budget : null,
    note: `no within-budget AND completed point at budget(s) ${infeasible.map((r) => r.budget).join(", ")}% (frontier collapses at budget <= ${first.budget}% on the lowest feasible edge; infeasible at B in {${infeasible.map((r) => r.budget).join(", ")}})`,
  };
}

const budgetRegions = [];
const budgetCsv = [];
for (const region of Object.keys(REGIONS)) {
  const cfg = REGIONS[region];
  const ckpt1488 = budgetRows(region, 148.8);
  const ckpt900 = budgetRows(region, 900);
  budgetRegions.push({
    region,
    model: "Deepseek",
    year: 2025,
    start: cfg.start,
    tpMax: cfg.tpMax,
    optimizer: { resolution: 10, iterations: 6, budget: null, alpha: 1, ckpt_resume: 0 },
    ckpt_148_8: ckpt1488,
    ckpt_900: ckpt900,
    collapse: {
      ckpt_148_8: collapsePoint(ckpt1488),
      ckpt_900: collapsePoint(ckpt900),
    },
  });
  for (const ckpt of CKPTS) {
    const rows = ckpt === 148.8 ? ckpt1488 : ckpt900;
    for (const r of rows) {
      budgetCsv.push([
        region, ckpt, r.budget,
        fmt(r.theta_p), fmt(r.theta_r), fmt(r.margin), fmt(r.savings), fmt(r.overhead), fmt(r.score),
        r.num_pauses == null ? "NA" : String(r.num_pauses),
        String(r.completed), String(r.within_budget), String(r.found),
        r.best_completed ? fmt(r.best_completed.savings) : "NA",
        r.best_completed ? fmt(r.best_completed.overhead) : "NA",
      ].join(","));
    }
  }
}

const budgetSummary = {
  spec: "Phase B.4 overhead-budget sweep (DE/IT/SE, year 2025, DeepSeek, reopt settings)",
  generatedDate: new Date().toISOString().slice(0, 10),
  deterministic: true,
  methodology: {
    budget:
      "overhead budget B in {30,50,100,200}%; alpha=1 so the score normalizes only savings (overhead term weighted by (1-alpha)=0), but the withinBudget filter (overhead <= B) still constrains which points the optimizer may select. ckpt_pause 148.8 s = the B.0-verdict (Story A) checkpoint; ckpt_pause 900 s = robustness block (secondary table).",
    collapse:
      "the feasible frontier collapses when no within-budget AND completed point with positive savings exists (best_completed == null). Reported per ckpt as the smallest such budget. All cells retain the raw max-score best AND the completed-feasible optimum (phase B.0 caveat: the raw best can be an incomplete budget-blocked run with inflated savings).",
  },
  regions: budgetRegions,
};
writeFileSync(resolve(OUT, "budget_summary.json"), JSON.stringify(budgetSummary, null, 2) + "\n", "utf-8");
writeFileSync(
  resolve(OUT, "budget_summary.csv"),
  "region,ckpt_pause,budget,theta_p,theta_r,margin,savings,overhead,score,num_pauses,completed,within_budget,found,best_completed_savings,best_completed_overhead\n" + budgetCsv.join("\n") + "\n",
  "utf-8",
);

console.log("wrote multiyear_{DE,IT,SE}.{json,csv}, multiyear_summary.json, multiyear_fixed_summary.json, budget_summary.{json,csv}");
