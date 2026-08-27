# D.10 Claim–Evidence Verification — Audit Report

- Phase: D.10 (e-Energy 2027 paper "Staleness, Not Noise: Why Carbon-Aware LLM Pretraining Does Not Need ML Forecasting")
- Auditor: CLAIM–EVIDENCE VERIFICATION AGENT (read-only; no file modified)
- Date: 2026-08-19
- Scope: every numeric claim in `publication/eenergy/main.tex` vs the committed artifacts under `publication/output/`; correctness of `publication/eenergy/claims_evidence.md`; SPEC Reference facts honored; Known errors not repeated.
- **Verdict: FINDINGS** (no BLOCKING; 2 MAJOR, 4 MINOR, 8 NIT). All headline experimental numbers — savings, overhead, margins, drift, grace horizons, R²/k, adaptive closed loop, DTPR, checkpoint retention, budget sweep — are **exact** matches to the committed JSON. The findings are localized: one figure caption that misstates its own committed figure, two over-generalized sentences, one scope overstatement, and rounding/pedigree nits.

---

## 1. Executive verdict

**FINDINGS** — the core experimental claim surface is clean and fully traceable. Retention, oracle ceilings, drift margins, calibration, grace fits, adaptive table, DTPR β, budget and checkpoint sweeps all match the committed artifacts to reported precision. Two claims are **factually wrong against the committed artifacts** (Fig. 3 caption/text vs the committed figure annotations; "2022–2023 contribute zero recovery at every c" vs SE-2023), and four further claims over-generalize or under-count (7/8 vs 8/8 within-one-step; "135–2700 s" sweep; "five grids / 2022–2026" scope; IT material-widening infeasibility at h≥3). None of these touch the paper's headline numbers, but the MAJORs must be fixed before submission because a reviewer can catch them by opening the committed figure/JSON.

## 2. Scope & method

- Numeric checklist extracted from `main.tex` (abstract, body, all tables, figure captions, algorithm). **Total distinct numeric claims/values enumerated: 92** (counted below by section).
- For each claim, the authoritative value was queried from the committed JSON via `node -e` (key queries listed in §4). Verdict classes: **EXACT** (matches to reported precision), **TOL** (within rounding/tolerance), **MISMATCH** (paper ≠ artifact), **EST** (estimate with cited source, permitted by SPEC D.10).
- `claims_evidence.md` audited row-by-row (§6). Reference facts (§7), Known errors (§8).

## 3. Claim table (exhaustive)

Legend: S0=baseline. All percentages are degradation/savings as in the paper. `…÷…` = derived ratio.

### 3.1 Abstract

| # | Claim (paper) | Artifact source (path.field) | Artifact value | Paper value | Verdict |
|---|---|---|---|---|---|
| 1 | lag-1 autocorr > 0.996 | `calibration_{DE,IT,SE}.json.lag1AutoCorr` | 0.999655/0.998681/0.995950 | "exceeds 0.996" | **NIT** (SE = 0.99595 < 0.996; see F8) |
| 2 | persistence ≈ AR(1), RMSE gap ≤ 0.06 g/kWh | `calibration_{R}.json.evaluation[]` (pers vs ar) | max gap 0.0512 (IT h=72) | ≤ 0.06 | EXACT |
| 3 | 4σ\* additive error < 7% in DE and IT | `fixed_DE.json`/`fixed_IT.json` summary add4 | 0.0241 / 0.0646 | 2.4% / 6.5% (< 7%) | EXACT |
| 4 | six-hour-stale decision costs 36–60% | `fixed_summary.json.degradationAtH72.persistence.delta_s_frac` | 0.358 / 0.422 / 0.605 | "36–60%" | **NIT** (exact 35.8–60.5%; see F9) |
| 5 | grace predicted, k≈0.68·S, R²=0.91, 8 region-years | `grace_horizon.json.{kPrimary, fits.yearMean.noIT23.throughOrigin}` | k=0.68398, R²=0.9134, n=8 | k≈0.68, R²=0.91, 8 | EXACT |
| 6 | closes 95%/80% of recoverable gap at h=1/3 (DE) | `adaptive_summary.json.regions[DE].rows[h=1,3].recovery_vs_oracle` | 0.9538 / 0.7988 | 95% / 80% | EXACT |
| 7 | retention 97.6/94.6/95.2 @900; 91.4/85.3/81.4 @2700 | `checkpoint_summary.json runs[].bestCompleted.savings ÷ baseline.savings` | 97.62/94.61/95.22; 91.35/85.32/81.43 % | same | EXACT |

### 3.2 Introduction

| # | Claim | Artifact | Artifact value | Paper | Verdict |
|---|---|---|---|---|---|
| 8 | DeepSeek ≈ 2.8M GPU-hours | DeepSeek-V3 tech report (cited) | 2.788M H800 | "on the order of 2.8M" | **EST** ✓ marked |
| 9 | 671B MoE, 14.8T tokens, 2048 GPUs | `profiles.json.Deepseek` | 671e9 / 1.48e13 / 2048 | same | EXACT |
| 10 | five grids, 2022–2026, 5-min, 105,120 pts/yr | `public/data/co2/*.json` (25 files), `carbonIntensity.length` | 25 files, 105120 | same | EXACT |
| 11 | 72-step-old decision costs 35.8–60.5% | `fixed_summary.json.degradationAtH72` | 0.358–0.605 | 35.8–60.5% | EXACT |
| 12 | 671B full state ≈ 1.3–1.6 TB | derived 671e9×2 B = 1.34 TB | 1.34 TB | "on the order of 1.3–1.6 TB" | **NIT** (1.6 unsourced; see F12) |
| 13 | checkpoints retain 81–97% | retention (row 7) | min 81.4 max 97.6 | 81–97% | EXACT |
| 14 | contribution 2: 4σ\* <7% DE/IT; 72-step 35.8–60.5% | rows 3,4 | — | same | EXACT |
| 15 | lag-1 ≥ 0.996 | row 1 | — | "≥ 0.996" | TOL (SE 0.99595≈0.9960@3dp) |
| 16 | RMSE gap ≤ 0.06 | row 2 | — | same | EXACT |
| 17 | AR(7) shaves a few g/kWh off short-horizon error | `calibration_DE.json` eval order7 h1 | 2.296 vs 4.214 | "a few g/kWh" | EXACT (qualitative) |
| 18 | checkpoint/restore minutes, not 148.8 s (7B figure) | `constants.json.checkpoint_pause_time`=148.8 | 148.8 | same | EXACT (qualitative) |
| 19 | **sweep 135–2700 s** | `checkpoint_summary.json runs[].ckpt_pause` | {148.8,150,900,2700} | "135–2700 s" | **MISMATCH** (F4) |
| 20 | contribution 1: near-unit-root "across five grids and 2022–2026" | calibration only DE/IT/SE (train 2022–24) | — | "across five grids and 2022–2026" | **MISMATCH** (scope, F5) |

### 3.3 System Model (sec:model)

| # | Claim | Artifact | Artifact value | Paper | Verdict |
|---|---|---|---|---|---|
| 21 | 700 W train / 60 W pause / PUE 1.27 | `constants.json.{gpu_power_train,gpu_power_pause,pue}` | 700/60/1.27 | same | EXACT |
| 22 | checkpoint 148.8 s / 0 s | `constants.json.{checkpoint_pause_time,checkpoint_resume_time}` | 148.8/0 | same | EXACT |
| 23 | 671B MoE state ≈ 1.34 TB BF16 | 671e9×2 B | 1.342 TB | "≈1.34 TB" | **EST** ✓ marked (derived) |
| 24 | sweep {148.8,900,2700}; 150 degenerate & omitted | `checkpoint_summary.json` runs | {148.8,150,900,2700}, ΔS<0.01pp @150 | same | TOL (150 is in artifact; "omitted" = from reporting, F13) |
| 25 | h=72 scale 8.38σ\* ≈ 30.7 g/kWh (DE) | 8.3823×σ\*=3.6584 (from `calibration_DE.json`; formula `σ\*√((1−φ^2h)/(1−φ²))`) | 30.67 | 8.38σ\*≈30.7 | EXACT |
| 26 | margin(h) formula | `adaptive_summary.json.methodology.adaptiveRule` | matches | matches | EXACT |
| 27 | B=200%, α=1 | `reopt_summary.json`/`adaptive_summary.json` methodology | B=200, α=1 | same | EXACT |
| 28 | Kimi-K2 second scale point | `profiles.json.Kimi` | 1.04e12 / 1.55e13 / 256 | "analogous Kimi-K2 profile" | EXACT (no Kimi experiments, as stated) |

### 3.4 Grid CI at 5-min (sec:grid)

| # | Claim | Artifact | Artifact value | Paper | Verdict |
|---|---|---|---|---|---|
| 29 | Table 1 φ / lag-1 / σ\* / RMSE(1) / RMSE(72) | `calibration_{R}.json` | DE 0.9997/0.9997/3.66/4.21/113.4; IT 0.9987/0.9987/4.42/5.02/77.6; SE 0.9960/0.9960/0.77/0.78/7.8 | same | EXACT |
| 30 | max persistence–AR(1) gap 0.0512 (IT h=72) | `calibration_IT.json` eval h72 pers 77.661 vs ar 77.610 | 0.0512 | same | EXACT |
| 31 | AR(7) DE h=1: 2.3 vs 4.2; converges h≥24 | `calibration_DE.json` eval | 2.296 vs 4.214; h24 48.53≈49.76; h72 112.82≈113.41 | same | EXACT |
| 32 | ACF lag-1 ≈ 0.9997 (DE), 24 h ≈ 0.17 | committed figure `figures/ci_trace_acf` (from `DE_2025.json`) | **figure annotates lag-1 ≈ 0.9995, 24 h ≈ 0.72** | 0.9997 / 0.17 | **MISMATCH** (F1) |
| 33 | ACF "decays only slowly, remaining >0.1 at 24 h" | same figure | 0.72 at 24 h | ">0.1 at 24 h" | TOL (true but vacuous; F1) |
| 34 | thresholds "of order 18–270 g/kWh" | reopt θ_p: SE 18.18, DE 272.37 | 18–272 | "18–270" | TOL (rounding) |

### 3.5 Threshold Optimization & Design Rules (sec:design)

| # | Claim | Artifact | Artifact value | Paper | Verdict |
|---|---|---|---|---|---|
| 35 | baseline optima θ/S/O | `reopt_summary.json.regions[].baseline` | DE (272.37,267.73)/43.35/174.3; IT (246.70,230.45)/32.67/194.7; SE (18.18,17.51)/23.17/106.3 | same | EXACT |
| 36 | margins 4.64 / 16.25 / 0.67 | `baseline.margin` | 4.64/16.25/0.67 | same | EXACT |
| 37 | SE margins 0.67–0.75 all four years | `multiyear_summary.json` SE near_zero_margin.margins | [0.75,0.67,0.72,0.67] | 0.67–0.75 | EXACT |
| 38 | DE 15–16 (2022–23); IT 2025 = 16.25 | `multiyear_summary.json` DE per_year margins | 16.00/15.04 (2022/23); IT-2025 16.25 | same | EXACT |
| 39 | percentiles wander 11–15 pp | `multiyear_summary.json.rule_deviation.percentile_thresholds` | 11.29 / 13.71 / 14.82 pp | 11–15 pp | EXACT |
| 40 | θ_p ranges 59.9–71.2 / 60.3–74.0 / 60.0–74.8% | `multiyear_{R}.json` per-year pct_exceed | DE 60.0,59.9,71.2,65.3; IT 69.8,60.3,74.0,68.6; SE 74.8,68.9,71.5,60.0 | same | EXACT |
| 41 | savings ranges 14.95–43.35 / 10.02–32.67 / 16.18–23.17 | `multiyear_summary.json.rule_deviation.savings_stability` | same | same | EXACT |
| 42 | Table 3 margins & grace 22–25 | `multiyear_{R}.json` + `multiyear_fixed_summary.json` | DE 16.00/15.04/4.76/4.64, grace 24/24/24; IT 3.24/8.00/4.72/16.25, grace 72/24/12; SE 0.75/0.67/0.72/0.67, grace 12/12/12 | same | EXACT |
| 43 | grace year-stable (DE 24/24/24, SE 12/12/12, IT 12/24/72) | `multiyear_fixed_summary.json` grace_persistence | DE 24/24/24, SE 12/12/12, IT 12/24/72 | same | EXACT |
| 44 | savings vary 7–28 pp across years | `multiyear_summary.json` ranges | ranges 6.99–28.40 pp | "7–28 pp" | EXACT |
| 45 | budget sweep completed-feasible S | `budget_summary.json regions[].ckpt_148_8[]` | DE 11.51/16.28/29.52/43.35; IT (B100) 19.85, (B200) 32.67; SE 14.59/18.36/22.87/23.17 | same | EXACT |
| 46 | IT frontier collapses B ≤ 50% | `budget_summary.json regions[IT].collapse.ckpt_148_8.infeasible_budgets` | [30,50] | "B ≤ 50%" | EXACT (artifact note text says "≤30%" — F14, paper correct) |
| 47 | IT nominal overhead 194.7%, no headroom | `reopt_summary.json` IT baseline.overhead | 194.68 | 194.7% | EXACT |
| 48 | IT B=100 raw 82.00 vs completed-feasible 19.85 | `budget_summary.json` IT ckpt_148_8 B100 savings / best_completed | 82.00 / 19.85 | same | EXACT |
| 49 | checkpoint table S0/S900/S2700/R900/R2700 | `checkpoint_summary.json` | verified row 7 | same | EXACT |
| 50 | margin widens DE 4.64→16.59 (900)→39.50 (2700) | `checkpoint_summary.json runs[].bestCompleted.margin` | 4.64/16.59/39.50 | same | EXACT |
| 51 | raw best inflated DE 47.91→82.18 | `checkpoint_summary.json runs[900,2700].savings` (completed=false) | 47.9085 / 82.1780 | 47.91 / 82.18 | EXACT |

### 3.6 Forecast Robustness (sec:robustness)

| # | Claim | Artifact | Artifact value | Paper | Verdict |
|---|---|---|---|---|---|
| 52 | add 1σ\*: 0.3/0.4/4.8 % | `fixed_{R}.json` summary add1 | 0.0031/0.0037/0.0483 | same | EXACT |
| 53 | add 4σ\*: 2.4/6.5/40.6 % | summary add4 | 0.0241/0.0646/0.4055 | same | EXACT |
| 54 | multiplicative cheaper (DE 1.5% at 4σ\*) | `fixed_DE.json` summary mult4 | 0.0149 | 1.5% | EXACT (also IT 3.9<6.5, SE 30.6<40.6) |
| 55 | pers 6: 0.7/1.8/2.5 % | summary pers6 | 0.0071/0.0176/0.0254 | same | EXACT |
| 56 | pers 12: 2.4/4.1/5.2 % | summary pers12 | 0.0243/0.0405/0.0517 | same | EXACT |
| 57 | pers 24: 8.0/11.2/14.2 % | summary pers24 | 0.0803/0.1124/0.1415 | same | EXACT |
| 58 | pers 72: 35.8/42.2/60.5 % | summary pers72 | 0.358/0.422/0.605 | same | EXACT |
| 59 | grace 24/12/12 steps | `fixed_summary.json.graceLevels.persistence.horizon` | 24/12/12 | same | EXACT |
| 60 | "even at 12 steps … comparable to a 4σ\* noise" | rows 56 vs 53 | SE: 5.2% vs 40.6% | "comparable" | **MISMATCH** for SE (F7) |
| 61 | 6–15× staleness vs 4σ\* noise (DE/IT) | 35.8/2.4=14.9; 42.2/6.5=6.5 | 14.8× / 6.5× | 6–15× | EXACT |
| 62 | reopt drift margins 16.68/28.36; 23.19/30.15; 3.21/5.49 | `reopt_summary.json` baseline.margin + drift[] margin_drift | 4.64+12.04=16.68, +23.72=28.36; 16.25+6.94=23.19, +13.90=30.15; 0.67+2.54=3.21, +4.82=5.49 | same | EXACT |
| 63 | fixed margin rule fails DE/IT ≥1σ\*; fails IT delay-1; SE survives all | `reopt_summary.json.marginRuleSurvives[]` | DE add1 F, add2 F, delay1 T; IT add1 F, add2 F, delay1 F; SE all T | same | EXACT (SPEC known-error #1 correctly reported) |
| 64 | fit k=0.684, R²=0.913, intercept R²=0.915, n=8; full-9 R²=0.194; trainMean R²=0.86 | `grace_horizon.json.fits` | 0.68398/0.9134/0.9154/8/0.1936/0.8622 | same | EXACT |
| 65 | "7/8 points lie within one grid step" | `grace_horizon.json.predicted[].within1Step` | **8/8 in-sample true** (IT-2023 excluded false) | "7/8" | **MISMATCH** (F3) |
| 66 | Table 5 per-region-year (g, RMSE(g), S, g_pred, dev) | `grace_horizon.json.points[]/predicted[]` | all 9 rows match to 0.1 | same | EXACT |
| 67 | per-point k 0.46–0.92, CV≈0.25 | `grace_horizon.json.points[].k.yearMean` | min 0.4600 (SE-23), max 0.9171 (SE-25), pop-CV 0.249 | same | EXACT |
| 68 | theoretical RMSE under-widens 3.7× at h=72 DE (30.7 vs 113.4) | `grace_horizon.json.rmseEmpiricalVsTheoretical.DE[72]` | ratio 3.6983; 30.666 vs 113.413 | same | EXACT |
| 69 | θ_p < μ in all nine region-years | `grace_horizon.json.points[].S` (S>0 all 9) | yes | yes | EXACT |

### 3.7 Stale-Aware Adaptive Control (sec:adaptive)

| # | Claim | Artifact | Artifact value | Paper | Verdict |
|---|---|---|---|---|---|
| 70 | c\* = DE 0.75, IT 2, SE 1.5 | `adaptive_summary.json.regions[].chosenC` | 0.75/2/1.5 | same | EXACT |
| 71 | Table 7 c-selection mean recovery | `regions[].c_selection.meanRecovery` | DE 0.006/0.006/0.017/0.056/0.012/0.000/0.016; IT 0.004/0.003/0.000/0.000/0.002/0.010/0.025; SE 0.000/0.002/0.022/0.016/0.022/0.029/0.000 | same | EXACT |
| 72 | train signal ≤ 0.056 | `c_selection.meanRecovery` max | 0.0556/0.0248/0.0289 | "≤ 0.056" | EXACT |
| 73 | "2022–2023 contribute zero recovery at every c" | `c_selection.perYear` | **SE 2023 c=0.75 = 0.0171 (non-zero)**; DE/IT 2022-23 and SE 2022 zero | "zero at every c" | **MISMATCH** (F2) |
| 74 | closed-loop table (naive/adapt/oracle/recov/rvo) | `regions[].rows[]` | all 18 rows match (naive/adapt/oracle to 2dp; recov, rvo exact; IT/SE "inc." rows completed=false) | same | EXACT |
| 75 | DE h=1/3 rvo 0.95/0.80 | rows | 0.9538/0.7988 | 0.95/0.80 | EXACT |
| 76 | static oracle recovers 32/48/30% of h=72 loss | (S_oracle−S_naive)/(S0_FF−S_naive) from rows | 31.6%/48.4%/30.2% | 32/48/30% | EXACT |
| 77 | DE h=72 c=6 → S=31.43 completed rvo 0.748; c=8 → 36.29 budget-infeasible | `adaptive_sensitivity_DE.json` | 31.4280, completed, rvo 0.7483; 36.2935, completed=false | same | EXACT |
| 78 | feasible-c envelope: DE ≤6 h72; SE ≤0.75 h72, no recovery (best c=0.75 → S=2.62); IT infeasibility | `adaptive_sensitivity_summary.json.feasibleC` + `adaptive_sensitivity_SE.json` | DE72≤6; SE72≤0.75, c=0.75→S=2.62; IT h=72 {0,3,4,6} | same | EXACT |
| 79 | "IT's material-widening regime c∈{0.25,…,2} is budget-infeasible at h≥3" | `feasibleC` IT | h=3 [0..1.5] feasible (only 2 infeasible); h=6 [0..0.75] | as stated | **MISMATCH** (over-generalized, F6) |
| 80 | model interval under-widens 3.7× (DE h=72) | `grace_horizon.json.rmseEmpiricalVsTheoretical` | 3.6983 | same | EXACT |
| 81 | oracle h=72 center ≈283 vs nominal midpoint 270.1 | oracle θ=(407.47,159.35)→283.41; (272.37+267.73)/2=270.05 | 283.4 / 270.05 | ≈283 / 270.1 | EXACT |
| 82 | DTPR β = 11.26 / 10.20 / 0.75 g/kWh | `adaptive_summary.json.regions[].dtpr.beta` | 11.258/10.197/0.751 | same | EXACT (also β formula verified: τ/3600·θ_p^nom) |
| 83 | DTPR neutral DE/SE within 1 pp; IT infeasible all h | `rows[].savings_dtpr` vs `savings_naive`; `overhead_dtpr` | max |Δ| DE 0.25, SE 0.74 pp; IT overhead_dtpr=200.00>200 all h | same | EXACT (completed_dtpr field does not exist — inferred; F10) |
| 84 | IT nominal overhead 194.7%, ~5 pp headroom | `reopt_summary.json` IT overhead 194.68; budget 200 | 5.3 pp | "~5 pp" | EXACT |

### 3.8 Discussion / Conclusion

| # | Claim | Artifact | Artifact value | Paper | Verdict |
|---|---|---|---|---|---|
| 85 | SE margin 0.67 < σ\* 0.77 → noise-sensitive | `reopt_summary.json` SE margin 0.67; `calibration_SE.json` sigmaStar 0.774 | 0.67 < 0.77 | same | EXACT |
| 86 | SE degrades 40.6% at 4σ\* | `fixed_SE.json` add4 | 0.4055 | 40.6% | EXACT |
| 87 | IT-2023 grace 72 (right-censored; h72 degradation 3.1%) | `multiyear_fixed_summary.json` IT-2023 | grace 72, h72deg 0.0312 | 3.1% | EXACT |
| 88 | SLA: DE 24 steps (2 h), IT/SE 12 steps (1 h) | `fixed_summary.json.graceLevels.persistence.horizon` × 5 min | 24/12/12 steps | same | EXACT |
| 89 | US/CN traces available but unused | `public/data/co2/{US,CN}_*.json` present | present | yes | EXACT |
| 90 | 12× coarser hourly update | 60/5 min | 12 | same | EXACT |
| 91 | conclusion restates rows 3,4,5,6,13 | — | — | — | EXACT |
| 92 | 561M curtailment model (Wiesner) | cited paper (arXiv:2602.22760) | reported figure | 561M | **EST** ✓ marked |

**Tally:** 92 claims → EXACT/TOL 84 · MISMATCH 6 (F1, F2, F3, F4, F5, F6, F7 — seven rows, see below; the mismatch count is 7 distinct rows) · EST 4 (rows 8, 23, 85-bis/561M, and PUE via constants) · NIT-flagged rounding items folded into EXACT/TOL where not misleading.

Re-counting strictly: **7 MISMATCH rows** (19, 20, 32, 33, 60, 65, 79), **4 EST** (8, 23, 92, PUE), remainder exact/tolerance.

## 4. Verification queries (R2)

Key `node -e` queries used (read-only; run from repo root):

```js
// retention (row 7)
node -e 'const c=require("./publication/output/checkpoint/checkpoint_summary.json"); for(const r of c) for(const run of r.runs.filter(x=>x.ckpt_pause===900||x.ckpt_pause===2700)) console.log(r.region, run.ckpt_pause, (run.bestCompleted.savings/r.baseline.savings*100).toFixed(2));'
// => 97.62 91.35 94.61 85.32 95.22 81.43

// oracle ceilings (row 76)
node -e 'const a=require("./publication/output/forecast/adaptive_summary.json"); for(const rg of a.regions){const h=rg.rows.find(x=>x.h===72); console.log(rg.region, ((h.savings_oracle-h.savings_naive)/(h.savings_perfect-h.savings_naive)*100).toFixed(1));}'   // 31.6 48.4 30.2

// calibration (rows 1,2,29,30,31)
node -e 'const s=require("./publication/output/forecast/calibration_DE.json"); console.log(s.lag1AutoCorr, s.sigmaStar, s.evaluation.filter(e=>e.horizon===1||e.horizon===72).map(e=>[e.model,e.order,e.horizon,e.rmse.toFixed(2)]));'

// fixed degradation grid (rows 52–60)
node -e 'for(const R of ["DE","IT","SE"]){const s=require("./publication/output/forecast/fixed_"+R+".json"); for(const fam of [["additive",1],["additive",4],["multiplicative",4],["persistence",6],["persistence",12],["persistence",24],["persistence",72]]){const r=s.summary.find(x=>x.family===fam[0]&&x.param_value===fam[1]); console.log(R,fam[0],fam[1],r.degradation_frac_mean.toFixed(4));}}'

// reopt drift + marginRuleSurvives (rows 62,63)
node -e 'const s=require("./publication/output/forecast/reopt_summary.json"); for(const r of s){const d=r.drift.find(x=>x.family==="additive"&&x.param_value===2); console.log(r.region, "marg1σ=", (r.baseline.margin+d.margin_drift).toFixed(2), "surv1:", r.marginRuleSurvives.find(x=>x.family==="additive"&&x.param_value===1).survives);}'

// grace fits + per-point k/CV (rows 64,65,66,67)
node -e 'const s=require("./publication/output/forecast/grace_horizon.json"); console.log(s.kPrimary, s.fits.yearMean.noIT23.throughOrigin.r2, s.fits.yearMean.noIT23.intercept.r2, s.fits.yearMean.all.throughOrigin.r2, s.fits.trainMean.noIT23.throughOrigin.r2); const ks=s.points.filter(p=>!p.isOutlier).map(p=>p.k.yearMean); console.log(Math.min(...ks), Math.max(...ks), (ks.reduce((a,b)=>a+b,0)/8).toFixed(4));'

// adaptive closed loop + c-selection (rows 70–74)
node -e 'const s=require("./publication/output/forecast/adaptive_summary.json"); for(const rg of s.regions){console.log(rg.region, rg.chosenC, rg.dtpr.beta); console.log(rg.c_selection.perYear["2023"]); console.log(rg.rows.filter(r=>r.h===1||r.h===72).map(r=>({h:r.h,n:+r.savings_naive.toFixed(2),a:+r.savings_adaptive.toFixed(2),o:+r.savings_oracle.toFixed(2),rvo:r.recovery_vs_oracle})));}'
// => SE-2023 c=0.75 meanRecovery = 0.0171 (the F2 mismatch)

// sensitivity money cell + feasibleC (rows 77–79)
node -e 'const s=require("./publication/output/forecast/adaptive_sensitivity_DE.json"); const r=(s.rows||s).find(x=>x.c===6&&x.h===72); console.log(r.savings, r.completed, r.recovery_vs_oracle); const x=require("./publication/output/forecast/adaptive_sensitivity_summary.json"); console.log(JSON.stringify(x.regions.find(z=>z.region==="IT").feasibleC));'

// figure ACF annotations (rows 32,33)
node -e 'const d=require("./publication/data/co2/DE_2025.json"); ...'   // mean-subtracted sample ACF: lag-1=0.9995, lag-288=0.72
grep -o "lag-1[^<]*\|24 h[^<]*" publication/eenergy/figures/ci_trace_acf.svg  // => "lag-1 ≈ 0.9995", "24 h: ACF $\approx$ 0.72"
```

All query outputs are recorded in the transcript of this audit session; the values above are the exact artifact values.

## 5. Findings

### BLOCKING
None.

### MAJOR

**F1 — Fig. 3 caption and §4 text misstate the committed figure's ACF annotations.**
- Location: `fig:trace` caption (`main.tex` lines ~369–370) and §4 text (line ~345); `claims_evidence.md` row D4.
- Claim: "sample autocorrelation to lag 288 (24 h), annotated at lag-1 ≈0.9997 and 24 h ≈0.17"; text "autocorrelation at lag 1 is ≈0.9997 (DE) … remaining >0.1 at 24 h".
- Correct value: the committed figure (`figures/ci_trace_acf.{svg,pdf}`, generated by `make_figures.py fig_ci_trace_acf` from `public/data/co2/DE_2025.json`) is literally annotated **"lag-1 ≈ 0.9995"** and **"24 h: ACF ≈ 0.72"** (verified by extracting the SVG text and re-computing the mean-subtracted sample ACF in Node: lag-1 = 0.9995, lag-288 = 0.72). 0.17 does not occur anywhere in the data/figure.
- The text's 0.9997 matches the *train* calibration lag-1 (0.99965) but is attached to the 2025 figure; the caption's "annotated at … 0.9997/0.17" is simply false.
- Fix: change caption to "annotated at lag-1 ≈0.9995 and 24 h ≈0.72"; change the §4 sentence to "autocorrelation at lag 1 is ≈0.9995 (DE) and decays only slowly (≈0.72 at 24 h)". Update `claims_evidence.md` D4 accordingly. (If the authors prefer 0.9997/0.17 they must regenerate the figure to actually annotate those values — not currently supported by the data.)

**F2 — "2022–2023 contribute zero recovery at every c" is false for SE-2023.**
- Location: §7(c)-selection fragility paragraph (`main.tex` ~line 785); `claims_evidence.md` row G3 ("`c_selection.perYear[2022/2023]` all zero").
- Correct value: `adaptive_summary.json regions[SE].c_selection.perYear["2023"].meanRecovery["0.75"] = 0.0171` (non-zero). DE and IT's 2022/2023 and SE's 2022 are all zero, but SE-2023 at c=0.75 is not.
- Fix: reword to "2022 contributes zero recovery in every region; 2023 contributes zero for DE/IT and ≤0.017 for SE". Update `claims_evidence.md` G3.

### MINOR

**F3 — "7/8 points lie within one grid step" should be 8/8.**
- Location: §6.3 (`main.tex` ~line 635) and `fig:grace` caption (line ~623).
- Correct value: `grace_horizon.json predicted[].within1Step` is `true` for **all 8 in-sample** region-years (DE-23/24/25, IT-24/25, SE-23/24/25); only IT-2023 — the excluded censored outlier — is `false`. `grace_horizon.md` and `phase_b5.md` also carry the erroneous "7/8" prose even though their own tables show 8 "yes" rows.
- Fix: "8/8 in-sample points lie within one grid step (IT-2023, the excluded right-censored outlier, does not)". If the authors find grid-adjacency too generous for DE-2024 (g_pred 72 vs g 24), they must tighten the definition in `grace_horizon_analysis.mjs` and re-derive, not just change prose.

**F4 — "135–2700 s for the sweep here" has no artifact basis.**
- Location: §2 Related Work (`main.tex` line 162).
- Correct value: the sweep is {148.8, 900, 2700} s (artifact runs also contain a degenerate 150 s row). 135 appears nowhere in the artifacts.
- Fix: "148.8–2700 s for the sweep here".

**F5 — near-unit-root claim scoped to "five grids and 2022–2026" but calibrated only for DE/IT/SE.**
- Location: Abstract ("Using five years … across five grids, we show … lag-1 autocorrelation exceeds 0.996") and contribution 1 ("across five grids and 2022–2026").
- Correct value: calibration (`lag1AutoCorr`, RMSE evaluations) exists only for DE/IT/SE, train 2022–24, eval 2025. US/CN and 2026 have no committed calibration.
- Fix: scope to "the three studied grids (DE, IT, SE)" or add committed calibration for US/CN. `claims_evidence.md` A1/B3 already map only to `calibration_{DE,IT,SE}.json`, so the paper's "five grids" wording exceeds its own evidence table.

**F6 — "IT's material-widening regime c∈{0.25,…,2} is budget-infeasible at h≥3" over-generalizes.**
- Location: §7(c) (`main.tex` ~line 826).
- Correct value: `adaptive_sensitivity_summary.json regions[IT].feasibleC` — h=3: [0,0.25,0.5,0.75,1,1.5] feasible (only c=2 infeasible); h=6: [0,0.25,0.5,0.75] feasible. Only the train-selected **c\*=2 is infeasible at all h≥3**; c≤1.5 remains feasible at h=3 and part of the grid at h=6.
- Fix: "the train-selected c\*=2 widening is budget-infeasible at h≥3, and much of the c-grid from h=6". (phase_b1_fix.md §3.4 is accurate; the paper generalized it too far.)

**F7 — "even at 12 steps … comparable to a 4σ\* noise" is false for SE.**
- Location: §6.1 (`main.tex` ~line 569).
- Correct value: pers12 = 2.4/4.1/5.2%; 4σ\* additive = 2.4/6.5/**40.6%**. For SE the 12-step cost (5.2%) is comparable to its *1σ\** noise (4.8%), not 4σ\*.
- Fix: "comparable to a 4σ\* noise in the regions with material margins (DE, IT)".

### NIT

- **F8** — Abstract "lag-1 autocorrelation **exceeds** 0.996": SE lag-1 = 0.99595 < 0.996 (≈0.9960 only at 3-dp rounding). Use "≥ 0.996" (consistent with the intro) or "exceeds 0.995".
- **F9** — Abstract/intro "36–60%" vs the exact 35.8–60.5% (used correctly in contribution 2). Prefer the precise range or "≈36–61%".
- **F10** — `claims_evidence.md` G12 cites `completed_dtpr` which does not exist in the artifact; IT DTPR infeasibility is inferred from `overhead_dtpr = 200.0016 > 200` at every h. Update the field reference to `rows[].overhead_dtpr` (or add the flag to the summary).
- **F11** — §6 degradation grid for SE uses the committed *rounded-threshold* fixed control (19, 18), s0=22.87%, O=99.4% (`fixed_SE.json`), which is exactly the SPEC-Known-error #2 control; DE/IT fixed controls equal the reopt optima. The paper never claims 22.87% is SE's optimum (it correctly reports 23.17%/106.3% in §5), so the Known error is *not* repeated, but the asymmetry is invisible to a reader. Recommend one sentence: "SE's fixed (published) policy uses the previously-published rounded thresholds (19, 18), s0 = 22.87%, distinct from its 2025 reopt optimum."
- **F12** — Intro "full state on the order of 1.3–1.6 TB": the 1.6 TB upper bound is unsourced in `claims_evidence.md` (only the 1.34 TB = 671e9×2 B derivation is listed). Either cite the range or use "≈1.34 TB".
- **F13** — §3 "a 150 s control … is … omitted": the committed `checkpoint_summary.json` *contains* the 150 s runs (ΔS < 0.01 pp vs 148.8 s, verified). "Omitted" is accurate only for the paper's reporting; suggest "computed and discarded (ΔS < 0.01 pp)".
- **F14** — Artifact-only inconsistency (not the paper): `budget_summary.json regions[IT].collapse.ckpt_148_8.note` says "collapses at budget ≤ 30%" while `infeasible_budgets = [30,50]` and `first_collapse_budget = 30`. The paper correctly says "B ≤ 50%". Fix the artifact note text (leave the paper as is).

## 6. Audit of `claims_evidence.md`

- **Legend/paths:** all 11 named artifacts exist at the stated paths with the stated structures (verified by loading each). Correct.
- **Row-by-row correctness:**
  - **Correct and precise:** A1–A7, B1–B5, C1–C7, D1, D2, D3, D5, E1–E14, F1–F10 (except F11), F12, G1, G2, G4, G5, G6, G7, G9, G10, G11, G13, H1–H5. Every field path resolves and every value matches the paper to reported precision. The "type" column correctly flags the four EST rows (B1 DeepSeek, C3 1.34 TB, Kimi, 561M) and the arithmetic rows (C5, 12×).
  - **Wrong:** **D4** (claims the figure is "annotated … 0.9997 / ≈0.17 at 24 h" — the committed figure annotates 0.9995 / 0.72; see F1). **G3** ("`perYear[2022/2023]` all zero" — SE-2023 c=0.75 = 0.0171; see F2). **F11** ("7/8 points within one grid step" — the artifact says 8/8; see F3).
  - **Imprecise:** **G8** ("IT material widening infeasible h≥3" — only c=2 infeasible at h≥3; see F6). **G12** (references a non-existent `completed_dtpr` field; see F10). **A1** ("≥0.9960 at 4-dp report precision" — at 4 dp SE = 0.9959; the table value 0.9960 is a 3-dp rounding, consistent with the SPEC reference fact; recommend stating 3-dp).
  - **Missing:** nothing significant; the "Numbers requiring re-verification" section (§5 of the table) is fully discharged — retention (97.6/94.6/95.2; 91.4/85.3/81.4), oracle ceilings (32/48/30%), 6–15× (14.8×/6.5×), k 0.46–0.92 & CV≈0.25, oracle center 283.4 vs midpoint 270.05 all re-derived and confirmed correct. The one item the table should have flagged but didn't is the Fig. 3 ACF annotation (D4), which it got wrong instead.

## 7. Reference facts (SPEC) consistency

- Perfect-foresight optima (θ/S/O for DE/IT/SE): paper §5 **matches** `reopt_summary.json` and the SPEC reference facts exactly.
- Calibration (σ\*, lag-1, φ): paper Table 1 **matches** `calibration_*.json` (3-dp/4-dp rounding as in the SPEC).
- Fixed-policy degradation (0.31/0.16/4.83 @ σ\*; 2.4/6.5/40.6 @ 4σ\*; 35.8/42.2/60.5 @ h=72; grace 24/12/12): paper **matches** `fixed_summary.json` exactly.
- Reopt drift margins (4.64→16.68→28.36; 16.25→23.19→30.15; 0.67→3.21→5.49): paper **matches** (`baseline.margin + drift[].margin_drift`). The SPEC's explicit correction ("IT also fails at delay-1 — do not repeat the original 'survives all'") is honored: paper states "fails for IT even at delay-1".
- Grace horizons stable within one step in ≥2 years per region: paper **matches** `multiyear_fixed_summary.json`.

## 8. Known errors — NOT repeated (confirmed)

1. "Margin rule survives all delay levels in all regions" — **not repeated**; paper correctly states IT fails at delay-1 (§6.2).
2. SE fixed control (19,18) presented as optimum — **not repeated**; paper's SE headline is the reopt optimum (18.18,17.51 → 23.17%/106.3%) (§5). (NIT F11 asks only for explicit disclosure that the §6 fixed control for SE is the rounded one.)
3. "Gap never exceeds ≈0.05" — **not repeated**; paper states the exact max gap 0.0512 and the "≤0.06" bound.
4. 148.8 s 7B figure applied to 671B unswept — **not repeated**; the checkpoint sweep over {148.8,900,2700} s and the 1.34 TB derivation are front and center (§3, §5).
5. Hanford '16 — **absent** from `main.tex` and `references.bib` (grep verified).
6. "First Pareto frontiers" / "first to use two thresholds" — **not claimed**; paper explicitly disclaims double-threshold novelty (§2: "we make no claim to double thresholds themselves").

## 9. Definition-of-done checklist

- **R1** Comprehensive numeric checklist extracted from `main.tex`: **92 distinct claims/values** enumerated across abstract/body/tables/figure captions (§3).
- **R2** Every checklist item verified against committed JSON via `node -e`/grep (≥30 required; 92 done). Query examples and outputs in §4; exact matches for all headline values, 7 mismatches identified.
- **R3** `review_d10.md` written with verdict (FINDINGS), full claim table (§3), findings (§5), claims_evidence audit (§6), reference-facts/known-errors confirmation (§7–8).
- **R4** Read-only: `git status --porcelain` shows only pre-existing Phase A–D working-tree changes (untracked `publication/`, SPEC edits, `src/cli` edits); no file was modified by this audit.
- **R5** Final message to orchestrator: below.

## 10. Final message to orchestrator

**Verdict: FINDINGS** (no blocking errors). **Claims verified: 92** against the committed artifacts — all headline experimental numbers (baseline optima, margins, drift, calibration, degradation grid, grace horizons, R²/k fits, adaptive closed-loop table, DTPR β, checkpoint retention, budget sweep) are exact matches. **Mismatches:** 2 MAJOR — (F1) Fig. 3 caption/text misstate the committed figure's ACF annotations (paper says lag-1 ≈0.9997 / 24 h ≈0.17; the committed figure says 0.9995 / 0.72), and (F2) "2022–2023 contribute zero recovery at every c" is false for SE-2023 (c=0.75 → 0.0171); 4 MINOR (7/8→8/8 within-one-step; "135–2700 s" vs {148.8,900,2700}; "five grids/2022–2026" scope vs DE/IT/SE-only calibration; IT c-regime infeasibility over-generalized to h≥3); 8 NIT. `claims_evidence.md` is correct except rows D4, G3, F11 (wrong) and G8, G12, A1 (imprecise). All SPEC Reference facts are honored and none of the six Known errors are repeated. **Recommendation (one paragraph):** the paper's empirical core is clean and submittable on the numbers, but before camera-ready fix F1 and F2 (both are immediately checkable against the committed figure/JSON and would surface under adversarial review), reword F3–F7 (cheap one-line edits), and apply the disclosure sentence in F11 about SE's fixed control so the Known-error #2 boundary is explicit. No experiment needs re-running: every mismatch is a wording/caption issue, not a recomputation issue. Apply the corresponding `claims_evidence.md` corrections (D4, G3, F11, G8, G12) so the evidence table stays a faithful index of the paper.
