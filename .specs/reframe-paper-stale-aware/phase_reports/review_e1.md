# Adversarial Review #1 — "Staleness, Not Noise: Why Carbon-Aware LLM Pretraining Does Not Need ML Forecasting"

- Date: 2026-08-19. Reviewer: adversarial audit #1 of 2 (investigate only; read-only).
- Object: `publication/eenergy/main.tex` (+ compiled `main.pdf`, 10 pp, 0 LaTeX errors), committed artifacts under `publication/output/**/*.json`, `public/data/co2/`, SPEC `reframe-paper-stale-aware/SPEC.md`, phase reports B.0/B.1(+fix)/B.3B4/B.5/C/D.10(+fix).
- Protocol mirror: `.specs/forecast-error-sensitivity-study/phase_reports/phase_review.md`.

## 1. Summary & overall assessment

**Recommendation: REJECT in current form / FIX-REQUIRED before review #2.** The empirical artifact surface is
exceptionally clean — every headline number I re-checked (baseline optima, calibration, degradation grid, drift
margins, grace fit, checkpoint retention, budget sweep, adaptive closed loop, c-selection, DTPR, sensitivity) is an
exact match to the committed JSON, all six Known errors are absent, all mandatory citations are present and correctly
positioned, and the paper is honest about most of its own weaknesses (SE noise sensitivity, IT-2023 censoring,
c-selection fragility, three-share h=72 decomposition, checkpoint-inflation artifact). That discipline is real and rare.

However, the **central empirical claim is confounded**: the paper's headline "noise is cheap, staleness is expensive"
compares a small-magnitude error source (4σ* additive noise ≈ 14.6 g/kWh for DE) against a large-magnitude error source
(6-hour staleness ≈ 113.4 g/kWh). On the paper's *own* shared-g/kWh axis (Fig. 6) the two families show the **opposite**
per-unit-magnitude relationship: at matched error magnitude, additive noise degrades savings **more** than a stale
decision (DE 2.41% vs 0.71% at ≈14 g/kWh; IT 6.46% vs 1.76%; SE 40.6% vs ≈5.2% at ≈3 g/kWh). Staleness "wins" only
because it produces 2.5–7.8× larger errors. The design rules (grace horizon SLA, adaptive margin, completion
constraint, checkpoint realism) all survive this reframing, but the title, abstract, contributions, §6.1 "money
result", Fig. 6 caption, and conclusion must be rewritten around **"error magnitude dominates; staleness is the
failure mode that generates large errors"** rather than an intrinsic noise-vs-staleness asymmetry. A second MAJOR is a
reproducibility defect in the headline `recovery` metric: the printed formula, applied to SE, yields +5.1 (not the
reported 0.00) because the clipping and the `naive ≥ perfect-foresight ⇒ 0` guard are unstated. Plus ~10 MINOR/NIT
(abstract "multi-year-stable" overstatement, grace-fit effective independence, "≥ 0.996" precision, the 30.7 vs
113.4 g/kWh internal tension, a dropped region-name in §7.2(c), missing "to our knowledge", undefined score
function, no Phase-E reproducibility statement).

**Strengths**
- Artifact discipline: every checked number is traceable and exact; figures are generated from committed JSON, byte-deterministic; ACF figure annotations now match the figure.
- Honest negative/qualified results: IT delay-1 margin failure, SE rounded-threshold disclosure, IT-2023 censoring with full-sample R²=0.194 shown, c-selection fragility table, "no universal value" for margins, DTPR "completion-infeasible for IT".
- Complete, current positioning (UQ-Advice, DTPR/OPR, LACS, equilibrium, Beyond-MCI, curtailment-LLM, CarbonCast/EnsembleCI/CarbonX, avg-vs-marginal, Green Mirage, QoRa, limits, Let's-wait) with no "first"/"optimal"/"proves" over-claims about the paper's own algorithm.
- Checkpoint realism handled as a design-rule change (Story A) with the completion-constraint artifact made explicit; the completed-vs-raw inflation is documented, not hidden.

**Weaknesses**
- The money-result framing (see MAJOR F1) — the thesis sentence "Staleness, not noise, is the binding constraint" is not what the matched-magnitude data show.
- The headline `recovery`/`recovery_vs_oracle` formulas as printed are not reproducible for SE (and the recovery headline closes a 0.07 pp absolute gap at DE h=1).
- The grace-horizon "prediction" is a between-region level-fit on ~4 distinct RMSE(g) values reused across years; the headline R²=0.91 is not horizon-prediction accuracy, and the year-stability claim is hedged to near-vacuity for IT.
- No Phase-E reproducibility statement (commands/runtimes/commit) and the optimizer score function is undefined in the paper.

---

## 2. Findings table

| # | Check (axis) | Status |
|---|--------------|--------|
| A1 | Abstract vs corrected narrative (recovery_vs_oracle, grace-region scoping, no universal-savings claim, checkpoint retention) | **FINDING (MAJOR F1)** — framing confounds magnitude & source |
| A2 | Title claim "Staleness, Not Noise" vs matched-magnitude data (Axis 1) | **FINDING (MAJOR F1)** |
| A3 | Positioning table honest/complete; no "first"/"proves"/"optimal" on own algorithm (Axis 1) | OK |
| A4 | "No prior work (i)–(iv)" phrasing (Axis 1) | **FINDING (MINOR)** — missing "to our knowledge" |
| A5 | Abstract "multi-year-stable threshold design rules" vs Table 3 (Axis 1) | **FINDING (MINOR)** |
| B1 | Baseline optima θ/S/O (DE/IT/SE) vs `reopt_summary.json` (Axis 2) | OK (exact) |
| B2 | Calibration φ/σ*/RMSE and max persistence–AR(1) gap 0.0512 vs `calibration_*.json` (Axis 2) | OK (exact) |
| B3 | Degradation grid add1/4, mult4, pers6/12/24/72, grace 24/12/12 vs `fixed_*.json` (Axis 2) | OK (exact) |
| B4 | Reopt drift margins 16.68/28.36 … 3.21/5.49 + marginRuleSurvives (IT delay-1 F) vs `reopt_summary.json` (Axis 2) | OK (exact) |
| B5 | Grace fit k=0.684, R²=0.913/0.915/0.194/0.86, per-region-year table vs `grace_horizon.json` (Axis 2) | OK (values exact) — see **MINOR** for independence |
| B6 | Checkpoint retention 97.6/94.6/95.2 @900, 91.4/85.3/81.4 @2700 + margins 4.64→16.59→39.50 vs `checkpoint_summary.json` (Axis 2) | OK (exact) |
| B7 | Budget sweep 11.51/16.28/29.52/43.35, IT B≤50 collapse, raw 82.00 vs 19.85 vs `budget_summary.json` (Axis 2) | OK (exact) |
| B8 | Adaptive table (naive/adapt/oracle/rvo 0.95/0.80), c* 0.75/2/1.5, oracle ceilings 32/48/30%, DE c=6→31.43 rvo 0.748 vs `adaptive_*.json` (Axis 2) | OK (exact) |
| B9 | Known errors (SE rounded control, "survives all", gap 0.051, 148.8 s sweep, Hanford, "first" claims) — none repeated (Axis 2) | OK |
| C1 | Decide-on-forecast/pay-on-realized accounting, hysteresis, grace definition, completion guard (Axis 3) | OK (see **MAJOR F2** for formula) |
| C2 | Recovery/recovery_vs_oracle formulas reproducible (Axis 3) | **FINDING (MAJOR F2)** |
| C3 | §3 "scale of the error a six-hour-old decision faces" = 30.7 vs §6.3 empirical 113.4 (3.7×) (Axis 3) | **FINDING (MINOR)** |
| C4 | Grace-horizon prediction methodology (year-invariant RMSE reuse; level-fit vs horizon accuracy) (Axis 3) | **FINDING (MINOR)** |
| C5 | Limitations discussion (μ, IT-2023, symmetric widening, single-site, ACI-not-MCI, noise boundary) (Axis 3) | OK |
| C6 | Adaptive margin rule, c-selection, feasible-c envelope, DTPR benchmark (Axis 3) | OK (see **MINOR** c*=2 region ambiguity) |
| D1 | Mandatory citations present & positioned; all 45 bibitems cited, 0 undefined (Axis 4) | OK |
| D2 | All 9 figures + 9 tables referenced; captions self-contained (Axis 4) | OK |
| D3 | Abstract length (215 words ≤ 250); page budget (10 pp) (Axis 4) | OK |
| D4 | LaTeX: 0 errors, 0 undefined; 6 small overfull hboxes (≤5.9 pt) (Axis 4) | **FINDING (NIT)** |
| E1 | Data provenance, optimizer settings, model/checkpoint parameters stated (Axis 5) | OK |
| E2 | Score function defined in paper; reproducibility statement (seeds/runtimes/commit) per SPEC E.4 (Axis 5) | **FINDING (MINOR)** |
| E3 | README + `run_eenergy_experiments.sh` document the pipeline (Axis 5) | OK |

---

## 3. Findings (ranked)

### BLOCKING
None. Every headline number re-checked is an exact match to a committed artifact; no claim contradicts its own JSON at the reported precision, and none of the SPEC "Known errors" are repeated.

### MAJOR

**F1 — The money result confounds error *magnitude* with error *source*; at matched magnitude the data contradict the "noise is cheap, staleness is expensive" thesis.**
- Location: title (L18); abstract (L36–56); intro (L92–96, "Staleness, not noise, is the binding constraint"); contribution 2 (L135–138); §6.1 (L557–581, "the paper's central empirical result"); Fig. 6 caption (L552–556, "Noise is cheap; staleness is expensive"); conclusion (L994–998); "6–15×" claim (L577–579).
- Problem: The paper's own shared-axis construction ("a stale decision *is* a persistence forecast", L562) invites a matched-magnitude comparison, and that comparison flips the claimed direction. Using committed values (`calibration_{R}.json` RMSE_persistence vs `fixed_{R}.json` degradation_frac_mean):
  - DE: additive 4σ* = **14.6 g/kWh → 2.41%**; staleness h=6 RMSE = 13.9 g/kWh → **0.71%** (also h=3: 7.2 g/kWh → 0.22% vs 2σ* noise 7.3 g/kWh → 0.94%).
  - IT: additive 4σ* = 17.7 g/kWh → 6.46%; staleness h=6 = 12.5 g/kWh → 1.76%.
  - SE: additive 4σ* = 3.1 g/kWh → 40.55%; staleness h=12 = 2.6 g/kWh → 5.17%.
  So **per unit error magnitude, additive noise is consistently more damaging than staleness** (white noise causes threshold flapping/checkpoint churn; a smooth stale forecast does not). "Staleness is expensive" holds only because h=72 staleness generates a 2.5–7.8× larger error (113.4 vs 14.6 g/kWh DE; 77.7 vs 17.7 IT; 7.8 vs 3.1 SE). The additive family is tested only up to 4σ*; the "noise is cheap" inference generalizes beyond that to a regime that was never measured.
- Evidence: numbers above (independent re-derivation in §4, rows R15–R18).
- Fix: Rewrite the central claim as "**decision-error magnitude dominates; staleness is the failure mode that produces large errors** (near-unit-root drift makes a stale decision a large-magnitude error by construction), so the operational lever is signal freshness — and per unit error magnitude noise is actually worse." Keep the design rules, grace-horizon SLA, adaptive margin, and checkpoint findings (all magnitude-agnostic). Adjust the title ("Staleness, Not Noise" → e.g. "Staleness Makes Errors Large: Why 5-Minute Carbon-Aware Pretraining Needs Fresh Signals More Than Better Forecasts") or, at minimum, add an explicit matched-magnitude sentence + adjust Fig. 6 caption and the "6–15×" sentence to scenario-level phrasing ("at the error magnitudes these failure modes realistically produce").

**F2 — The headline `recovery` metric as printed cannot reproduce the reported values (SE raw recovery is +5.1, reported 0.00); the clip and the `naive ≥ perfect-foresight ⇒ 0` guard are unstated.**
- Location: §7.1 (L701–709) formula; applied in Table 8 (SE rows all 0.00) and in c-selection Table 9; abstract's 95%/80% headline.
- Problem: The paper prints `recovery = (S_adapt − S_naive)/(S0_FF − S_naive)`. For SE h=1 the committed values are S_naive=23.086, S_adapt=21.977, S0_FF=22.868 (perfect-foresight = the rounded-threshold baseline 22.87, which is *below* the naive policy). The formula as printed gives (−1.108)/(−0.217) = **+5.10**; the artifact reports **0.00**. The implementation (`forecast-sweep.ts` methodology: "recovery = clip((…),0,1) … recovery = 0 when S0_FF − S_naive ≤ 1e-3 pp … incl. naive ≥ perfect") clips and zeroes when the ceiling is non-positive. Without stating that, every SE row and the c-selection table (SE's 0.000/0.002/0.022… cells) are unreproducible from the paper text; the same guard silently zeroes `recovery_vs_oracle` when oracle ≤ naive (SE, all h≤24).
- Evidence: `adaptive_summary.json` SE h=1 {naive 23.086, adapt 21.977, perfect 22.868, recovery 0, recovery_vs_oracle 0} vs formula.
- Fix: State the full definition: "recovery = clip((S_adapt−S_naive)/(S0_FF−S_naive), 0, 1), and is 0 whenever S0_FF ≤ S_naive (no positive loss to recover; this is the case for SE, whose perfect-foresight baseline is the rounded-threshold control 22.87, below its naive policy)"; likewise note rvo = 0 when oracle ≤ naive. This is one sentence but it is load-bearing for the headline metric.

### MINOR

**F3 — Abstract/contribution "multi-year-stable threshold design rules" overstates the body.** DE margins span 16.00/15.04/4.76/4.64 and IT 3.24/8.00/4.72/16.25 across 2022–25 (Table 3); only the grace horizon and SE's margin are truly stable, and percentile thresholds wander 11–15 pp (the paper itself says so, L446–451). Recommend "design rules that are stable in *scale* (margin scale, grace horizon) with documented crisis-year exceptions" in the abstract.

**F4 — Grace-horizon prediction: effective independence overstated; R²=0.91 is a level-fit, not horizon-prediction accuracy.** The 8 "region-years" share only 4 distinct RMSE(g) values (DE 49.76 ×3, IT 36.64, IT 20.62, SE 2.58 ×3) because the RMSE(h) curve is the 2025 calibration reused for all years (documented B.3 choice). The headline R² is on the *level* fit RMSE(g) ≈ k·S, not on g_pred vs g (the SLA deliverable), where residuals are 1–2 grid steps and DE-2024 g_pred=72 vs g=24 is "within one grid step" only by the generous grid metric. Fix: state that the fit has ≤4 distinct x-values, give the horizon-space accuracy explicitly (e.g. continuous residuals 2–8 steps from `grace_horizon.json predicted[].gPredCont`), and de-emphasize R²=0.91 in the abstract (or label it "level fit R²").

**F5 — "lag-1 autocorrelation ≥ 0.996" is false at full precision (SE = 0.995950).** Abstract (L41), intro (L100), contribution 1 (L131). The Table 2 value 0.9960 is a 3-dp rounding (the D.10 fix acknowledged this). Fix: "≥ 0.9960" (3 dp) or "≈0.996".

**F6 — §3 (L313–314) "the scale of the error a six-hour-old decision faces" uses the *theoretical* 30.7 g/kWh, contradicting §6.3's own 113.4 g/kWh (3.7× under-widening).** A reviewer cross-checking §3 vs §6.3 catches this. Fix: "the theoretical AR(1) prediction-interval scale (30.7 g/kWh); the empirical staleness error is ~3.7× larger, §6.3" or add a forward reference.

**F7 — §7.2(c) (L833–835) dropped the region name; the c*=2 sentence reads as being about DE, where it is false.** Text: "DE completes through c=6 at h=72; the train-selected c*=2 widening is budget-infeasible at h≥3, and most of the c-grid from h=6; SE completes…". The `c*=2 infeasible at h≥3` claim is an **IT** property (`feasibleC` IT h3 excludes 2; DE c=2 is feasible at *all* h — verified). Fix: "…IT's train-selected c*=2 widening is budget-infeasible at h≥3, and most of IT's c-grid from h=6; DE completes through c=6 at h=72; SE completes only…".

**F8 — "No prior work (i)–(iv)" (L213–217) lacks "to our knowledge".** The claims are supported by the Phase-A survey, but a categorical "No prior work" invites a reviewer to hunt for a counterexample; hedge it.

**F9 — The grace-horizon "year-stable" claim (L471–479) is hedged to near-vacuity for IT (72/24/12 across years = 6× variation).** "Stable within one grid step in at least two of three years" is automatically satisfied for IT (adjacent pairs differ by one grid step) yet 72 vs 12 steps is a 6-hour vs 1-hour SLA difference. Recommend dropping "year-stable" for IT or re-stating as "two of three years (2024–25) agree; 2023 is a censored outlier".

**F10 — Reproducibility: the optimizer score function is undefined and the Phase-E reproducibility statement (SPEC E.4) is still open.** The paper says "a savings-normalized score at α=1" (L404, L416) but never gives the formula; the implementation is `score = (α·(S/100) + 1 − (1−α)·(O/B))/2` (`src/domain/result.ts:24-33`), and at α=1 the overhead term vanishes — which is *why* the optimizer selects budget-blocked points. This makes the "completed-feasible" selection (L419–420) and the IT-82.00-vs-19.85 story hard to reproduce from the text. Fix: state the formula and its α=1 degeneracy, and add the reproducibility statement (commands, runtimes, no-RNG/determinism, commit hash — commit hash may be deferred until de-anonymization).

### NIT

**N1 — Abstract "≤0.017 for SE" (c-selection, L794) vs artifact 0.01708.** 0.0171 ≈ 0.017 is fine, but "≤0.017" is strictly false; use "≤0.018".

**N2 — Six overfull hboxes in `main.log` (≤5.87 pt at L302–304; others 1–3.5 pt).** Cosmetic; fix the worst (the margin-equation paragraph) with a reword or `\sloppy` locally.

**N3 — "105,120 points/year" (L338) is wrong for leap year 2024 (105,408 points).** Add "(105{,}408 in 2024)".

**N4 — Abstract headline "closes 95% and 80% of the recoverable gap at h=1/3 (DE)" is an 0.07 pp absolute effect (43.25→43.32 of 43.35).** The normalized metric is the stated headline, so not an error, but a reader needs the absolute context the Table 8 already provides; consider one clause in §7.1 ("the recoverable gap at h=1 is 0.07 pp of savings").

---

## 4. Verified-items evidence (independent re-checks, all against committed JSON)

> All commands run read-only from repo root. "Exact" = matches the paper to reported precision.

- **R1 Baseline optima** — `reopt_summary.json`: DE (272.37, 267.73, m 4.64) S=43.35 O=174.31; IT (246.70, 230.45, m 16.25) S=32.67 O=194.68; SE (18.18, 17.51, m 0.67) S=23.17 O=106.34. Exact. (paper L407–409, Tab. 3)
- **R2 Calibration** — `calibration_{DE,IT,SE}.json`: lag1 0.999655/0.998681/0.995950; σ* 3.658/4.421/0.774; RMSE(1) 4.213/5.020/0.776; RMSE(72) 113.413/77.610/7.825. Table 2 values 3.66/4.42/0.77, 4.21/5.02/0.78, 113.4/77.6/7.8. Exact. Max persistence–AR(1) gap = 0.0512 g/kWh (IT h=72: 77.661−77.610). Exact.
- **R3 Degradation grid** — `fixed_{DE,IT,SE}.json` (degradation_frac_mean ×100): add1 0.31/0.37/4.83; add4 2.41/6.46/40.55; mult4 1.49/3.92/30.62; pers6 0.71/1.76/2.54; pers12 2.43/4.05/5.17; pers24 8.03/11.24/14.15; pers72 35.83/42.19/60.49. Tab. 6 (0.3/0.4/4.8 … 35.8/42.2/60.5). Exact. SE fixed s0 = 22.8685 (rounded-threshold control, disclosed at L566–568). Grace 24/12/12.
- **R4 Drift + marginRuleSurvives** — `reopt_summary.json`: margins 1σ*/2σ* = 16.68/28.36 (DE), 23.19/30.15 (IT), 3.21/5.49 (SE). survives: DE add1 F, add2 F, delay1 T; IT add1 F, add2 F, **delay1 F**; SE all T. Paper §6.2 matches, incl. "fails for IT even at delay-1" (Known error #1 honored).
- **R5 Grace fit** — `grace_horizon.json`: kPrimary=0.68398; yearMean noIT23 R²=0.9134 (through-origin) / 0.9154 (intercept); all-9 R²=0.1936; trainMean R²=0.8622. Per-point k 0.460–0.917, pop-CV 0.249. `predicted[].within1Step` true for 8/8 in-sample, false only for IT-2023. Exact vs paper L638–653 and Tab. 7.
- **R6 Checkpoint retention** — `checkpoint_summary.json` (bestCompleted.savings / baseline.savings): @900 = 97.62/94.61/95.22%; @2700 = 91.35/85.32/81.43%. Margins 4.64→16.59 (900)→39.50 (2700) DE. Tab. 5 exact. IT-2700 margin 45.66 (not quoted in paper — fine).
- **R7 Budget sweep** — `budget_summary.json`: DE 11.51/16.28/29.52/43.35; IT B100 completed 19.85 vs raw 82.00; B200 32.67; B30/B50 found=false; SE 14.59/18.36/22.87/23.17. Tab. 4 and L507–512 exact.
- **R8 Adaptive closed loop** — `adaptive_summary.json`: c* 0.75/2/1.5; DE h=1 rvo=0.9538, h=3 rvo=0.7988; oracle h=72 ceilings 32.67/25.50/12.93 (→ 31.6/48.4/30.2% of loss vs perfect 43.35/32.67/22.87). Tab. 8 exact; "(DE) 95%/80%" exact.
- **R9 c-selection** — `adaptive_summary.json c_selection.meanRecovery`: DE 0.006/0.006/0.017/**0.056**/0.012/0.000/0.016; IT 0.004/0.003/0.000/0.000/0.002/0.010/**0.025**; SE 0.000/0.002/0.022/0.016/0.022/**0.029**/0.000. Tab. 9 exact. perYear[2023] SE c=0.75 = 0.0171 (matches F2 fix wording; "≤0.017" NIT).
- **R10 Sensitivity money cell** — `adaptive_sensitivity_DE.json`: h=72 c=6 → S=31.43, completed, rvo=0.7483; c=8 → 36.29, completed=false. `feasibleC` IT h3=[0…1.5] (c=2 infeasible), h6=[0,0.25,0.5,0.75]; DE h72=[0…6]; SE h72=[0,0.25,0.5,0.75]. Paper §7.2(c) exact (except region-reference issue F7).
- **R11 DTPR** — β = 11.258/10.197/0.751 g/kWh (= τ/3600·θ_p^nom). DE |Δ|≤0.12 pp, SE ≤0.74 pp vs naive; IT overhead_dtpr = 200.00 at every h (infeasible). §7.3 exact.
- **R12 Multi-year margins/savings** — `multiyear_summary.json`: DE margins 16.00/15.04/4.76/4.64, S 14.95–43.35; IT 3.24/8.00/4.72/16.25, S 10.02–32.67; SE 0.75/0.67/0.72/0.67, S 16.18–23.17. Tab. 3 exact. Percentile wander 11.3/13.7/14.8 pp ("11–15 pp").
- **R13 Grace horizons multi-year** — `multiyear_fixed_summary.json`: DE 24/24/24; IT 72/24/12 (IT-2023 h72 deg 3.12% → censored); SE 12/12/12. L471–479 exact (caveat F9).
- **R14 Oracle h=72 center** — θ=(407.47,159.35) → center 283.4 vs nominal midpoint 270.05. Exact.
- **R15–R18 Matched-magnitude comparison (new, evidence for F1)** — `calibration_{R}.json` RMSE_persistence + `fixed_{R}.json`:
  - DE: (7.2 g/kWh, staleness h3 → 0.22%) vs (7.3 g/kWh, add2σ* → 0.94%); (13.9, h6 → 0.71%) vs (14.6, add4σ* → 2.41%).
  - IT: (8.7, h3 → 0.96%) vs (8.8, add2σ* → 1.85%); (12.5, h6 → 1.76%) vs (17.7, add4σ* → 6.46%).
  - SE: (0.8, h1 → 0.43%) vs (0.8, add1σ* → 4.83%); (2.6, h12 → 5.17%) vs (3.1, add4σ* → 40.55%).
  - Magnitude ratios at the paper's scenario endpoints: 113.4/14.6 = 7.8× (DE), 77.7/17.7 = 4.4× (IT), 7.8/3.1 = 2.5× (SE).
  - Conclusion: per-unit-magnitude, additive noise is **more** damaging; the paper's asymmetry holds only for the *scenario* endpoints, not per error unit.
- **R19 Recovery formula reproduction (evidence for F2)** — `adaptive_summary.json` SE h=1: naive 23.086, adapt 21.977, perfect 22.868 → printed formula gives +5.10; artifact `recovery` = 0, `recovery_vs_oracle` = 0. Formula must include clip + `S0_FF ≤ S_naive ⇒ 0` guard (forecast-sweep.ts methodology).
- **R20 ACF figure annotations** — `figures/ci_trace_acf.svg` annotates "lag-1 ≈ 0.9995" and "24 h: ACF ≈ 0.72" (matches fixed caption L371–373 and §4 text L348–349; the D.10 F1 values are gone).
- **R21 LaTeX/build** — `make` 0 `^!`, 0 undefined refs/citations, 10 pages, abstract 215 words, 45/45 bibitems cited, 6 overfull hboxes (≤5.87 pt).
- **R22 Scope/veracity** — "five grids / 2022–2026" scoped to three studied grids in abstract+contributions (D.10 F5 fixed); "148.8–2700 s" (F4 fixed); "8/8 within one grid step" (F3 fixed); no "Hanford", no "first Pareto", no "first two thresholds"; DTPR double-threshold novelty explicitly disclaimed (L157–160).

## 5. Verdict

**FIX-REQUIRED.** No experiment needs re-running and no headline number is wrong — but the paper's central interpretive claim (F1) is not supported by its own shared-axis data at matched error magnitude, and the headline `recovery` metric (F2) is not reproducible as printed. Both are catchable by a careful reviewer in under an hour (they only need `calibration_*.json` + `fixed_*.json` + the printed formula), and together they strike at the title and the "central empirical result". The reframing for F1 is cheap in experiments (all evidence already committed: the degradation grid *is* the matched-magnitude data; the flapping mechanism is the explanation) but substantial in prose: abstract, intro, contributions, §6.1, Fig. 6 caption, conclusion, and possibly the title must move from "noise is cheap, staleness is expensive" to "error magnitude dominates; staleness is the realistic source of large errors". F2 is a one-sentence definition fix. The remaining MINORs are one-line edits (F3–F10) plus the Phase-E reproducibility statement. **Recommendation: apply the F1 reframing + F2 formula fix + MINORs, rebuild, and only then proceed to review #2 — the artifact core is sound and should survive a second adversarial pass once the framing is honest.**

### DoD status
- R1 Compiled (`make`, 0 errors) and read the full PDF text — DONE.
- R2 Independently re-verified 22 headline numbers/queries (R1–R22) against committed JSON — DONE (≥10 required).
- R3 `review_e1.md` written with all five sections, ranked findings, verdict — DONE (this file).
- R4 Read-only: no file modified by this reviewer (`git status` unchanged; only the report above is added as a new untracked file).
