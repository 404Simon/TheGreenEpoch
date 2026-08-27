# Claims–Evidence Table — "Staleness Makes Errors Large: Why 5-Minute Carbon-Aware Pretraining Needs Fresh Signals More Than Better Forecasts"

Every numeric claim in `main.tex` mapped to its committed artifact path + field.
All paths are relative to the repository root. **Estimate** = not directly in a
committed experiment artifact (sourced from a cited paper or derived with
standard conversion factors); everything else is directly traceable.

Legend:
- `reopt_summary.json` = `publication/output/forecast/reopt_summary.json`
- `fixed_{R}.json` = `publication/output/forecast/fixed_DE.json` (R = DE/IT/SE)
- `calibration_{R}.json` = `publication/output/forecast/calibration_DE.json`
- `adaptive_{R}.json` = `publication/output/forecast/adaptive_DE.json`
- `adaptive_summary.json` = `publication/output/forecast/adaptive_summary.json`
- `adaptive_sensitivity_{R}.json` = `publication/output/forecast/adaptive_sensitivity_DE.json`
- `multiyear_summary.json` = `publication/output/forecast/multiyear_summary.json`
- `multiyear_fixed_summary.json` = `publication/output/forecast/multiyear_fixed_summary.json`
- `budget_summary.json` = `publication/output/forecast/budget_summary.json`
- `grace_horizon.json` = `publication/output/forecast/grace_horizon.json`
- `checkpoint_summary.json` = `publication/output/checkpoint/checkpoint_summary.json`

---

## Abstract

| # | Claim (paper) | Artifact → field |
|---|---|---|
| A1 | lag-1 autocorrelation ≈ 0.996 (abstract) / 0.9960–0.9997 (intro, contribution 1) | `calibration_{DE,IT,SE}.json` → `lag1AutoCorr` = 0.9996547 / 0.998681 / 0.995950 (SE 0.99595 ≈ 0.9960 at 3 dp; "≈0.996" honest at full precision) |
| A1b | matched-magnitude pair in abstract: DE 4σ\* noise 14.6 g/kWh → 2.4% vs 6-step-stale 13.9 g/kWh → 0.7% | `calibration_DE.json` → `evaluation[]` persistence h=6 rmse 13.85; `fixed_DE.json` additive 4 `degradation_frac_mean` 0.0241 / persistence 6 → 0.0071 |
| A2 | persistence ≈ AR(1), RMSE gap ≤ 0.06 g/kWh | `calibration_{R}.json` → `evaluation[]` model `persistence` vs `ar` rmse; max gap = 0.0512 (IT, h=72) |
| A3 | 4σ\* additive forecast error costs < 7% of savings in DE and IT (now framed vs the 35.8–60.5% staleness cost at the magnitudes the modes produce) | `fixed_DE.json`/`fixed_IT.json` → `summary[]` family=`additive` param_value=4 `degradation_frac_mean` = 0.0241 (DE), 0.0646 (IT) |
| A4 | six-hour-stale decision costs 35.8–60.5% | `fixed_summary.json` → `degradationAtH72.persistence.delta_s_frac` = 0.358 / 0.422 / 0.605 (DE/IT/SE) |
| A5 | grace horizon predicted by empirical AR(1) RMSE at a *level fit* k≈0.68·S, R²=0.91, 8 region-years | `grace_horizon.json` → `kPrimary` = 0.68398; `fits.yearMean.noIT23.throughOrigin.r2` = 0.9134, `n` = 8 (level fit, not horizon-prediction accuracy) |
| A6 | closes 95% / 80% of recoverable naive-to-oracle gap at h=1/3 (DE) | `adaptive_summary.json` → `regions[DE].rows[h=1].recovery_vs_oracle` = 0.9538, `h=3` = 0.7988 |
| A7 | completed-feasible retention 97.6/94.6/95.2% @900 s; 91.4/85.3/81.4% @2700 s | `checkpoint_summary.json` → `runs[ckpt=900].bestCompleted.savings` (42.32/30.90/22.06) and `[ckpt=2700]` (39.60/27.87/18.87) ÷ `baseline.savings` (43.35/32.67/23.17) |

## Introduction (sec:intro)

| # | Claim | Artifact → field | Type |
|---|---|---|---|
| B1 | DeepSeek-V3-class run ≈ 2.8M GPU-hours | DeepSeek-V3 technical report (2.788M H800 GPU-hours); cited, not our artifact | **Estimate** (paper source) |
| B2 | 671B-parameter MoE, 14.8T tokens, 2048 GPUs | `public/data/profiles.json` → `Deepseek.modelParams`=671000000000, `datasetTokens`=14800000000000, `gpuCount`=2048 | |
| B3 | data spans five grids, 2022–2026, 5-min, 105,120 pts/yr (105,408 in leap year 2024) | `public/data/co2/{DE,IT,SE,US,CN}_{2022..2026}.json` → 25 files; `carbonIntensity.length` = 105120 (105408 for `*_2024.json`) (data availability only; calibration/evaluation claims cover the three studied grids DE/IT/SE — see A1) | |
| B4 | contribution 2: decision-error magnitude dominates; staleness is the realistic large-error mechanism (h=72 persistence RMSE 2.5–7.8× a 4σ\* noise); at matched magnitude additive noise per-unit more damaging (DE 2.4% vs 0.7% at ≈14 g/kWh) | `calibration_{R}.json` persistence RMSE h=72 (113.4/77.7/7.8) ÷ 4×σ\* (14.6/17.7/3.1) = 7.8/4.4/2.5; `fixed_{R}.json` additive 4 vs persistence 6 degradation (0.0241 vs 0.0071 DE) | |
| B5 | contribution 5: checkpoints retain 81–97%, widen margins, completion constraint required | `checkpoint_summary.json` retention (A7) + `bestCompleted.margin` (B24) + `completed` flags | |

## System Model (sec:model)

| # | Claim | Artifact → field | Type |
|---|---|---|---|
| C1 | 700 W/GPU training, 60 W paused, PUE 1.27 | `public/data/constants.json` → `gpu_power_train`=700, `gpu_power_pause`=60, `pue`=1.27 | |
| C2 | checkpoint 148.8 s / 0 s | `public/data/constants.json` → `checkpoint_pause_time`=148.8, `checkpoint_resume_time`=0 | |
| C3 | 671B MoE state ≈ 1.34 TB BF16 (paper intro uses ≈1.34 TB, 671e9×2 B; no unsourced 1.6 TB upper bound) | `profiles.json` `Deepseek.modelParams` (671e9) × 2 bytes (BF16); standard conversion | **Estimate** (derived) |
| C4 | checkpoint sweep {148.8, 900, 2700} s (150 s degenerate, computed and discarded) | `checkpoint_summary.json` → `runs[].ckpt_pause` ∈ {148.8, 150, 900, 2700}; 150 s ≈ 148.8 s (ΔS < 0.01 pp) | |
| C5 | h=72 interval scale 8.38σ\* ≈ 30.7 g/kWh (DE) | `checkpoint/DECISION.md` §4 (confirmed from `calibration_DE.json` σ\*=3.658, φ=0.999655 via margin formula); `src/domain/adaptive-margin.ts` test (iv) | |
| C6 | margin rule margin(h)=c·σ\*·sqrt((1−φ^2h)/(1−φ²)), symmetric widening | `adaptive_summary.json` → `methodology.adaptiveRule`; implementation `src/domain/adaptive-margin.ts` | |
| C7 | budget B=200%, α=1 | `reopt_summary.json`/`adaptive_summary.json` `methodology`; optimizer settings in `run_*` scripts | |

## Grid CI at 5-minute resolution (sec:grid)

| # | Claim | Artifact → field |
|---|---|---|
| D1 | AR(1) φ, lag-1, σ\* (DE 0.9997/3.66; IT 0.9987/4.42; SE 0.9960/0.77) | `calibration_{R}.json` → `lag1AutoCorr`, `sigmaStar`; φ = `orders["1"].coeffs.ar[0]` |
| D2 | RMSE(h=1) and RMSE(h=72) (4.21/113.4; 5.02/77.6; 0.78/7.8) | `calibration_{R}.json` → `evaluation[]` model=`ar` (order 1) horizon 1/72 `rmse` |
| D3 | persistence–AR(1) max gap 0.0512 (IT h=72) | `calibration_IT.json` → `evaluation[]` `persistence` rmse 77.661 vs `ar` 77.610 at h=72 |
| D4 | lag-1 autocorr ≈ 0.9995, ACF ≈ 0.72 at 24 h | ACF computed over `public/data/co2/DE_2025.json` `carbonIntensity` (figure annotation `ci_trace_acf`) |
| D5 | AR(7) shaves DE h=1 error 4.2→2.3 g/kWh, converges at h≥24 | `calibration_DE.json` → `evaluation[]` order=7 h=1 `rmse` = 2.296 (persistence 4.214); h=72 order7 112.82 ≈ ar 113.41 |

## Threshold Optimization & Design Rules (sec:design)

| # | Claim | Artifact → field |
|---|---|---|
| E1 | baseline optima DE (272.37, 267.73) S=43.35% O=174.3%; IT (246.70, 230.45) S=32.67% O=194.7%; SE (18.18, 17.51) S=23.17% O=106.3% | `reopt_summary.json` → `regions[].baseline.{thetaP,thetaR,margin,savings,overhead}` |
| E2 | margins 4.64 / 16.25 / 0.67 | `reopt_summary.json` → `baseline.margin` |
| E3 | SE margins 0.67–0.75 across 2022–2025 | `multiyear_summary.json` → `SE.rule_deviation.near_zero_margin.margins` = [0.75, 0.67, 0.72, 0.67] |
| E4 | DE/IT margins pushed to 15–16 (2022–23); IT 2025 = 16.25 | `multiyear_summary.json` → `DE.per_year` margins 16.00/15.04 (2022/23), `IT.per_year[2025].margin` 16.25 |
| E5 | percentile wander 11–15 pp | `multiyear_summary.json` → `rule_deviation.percentile_thresholds.pct_exceed_deviation_pp` = 11.29 (DE), 13.71 (IT), 14.82 (SE) |
| E6 | savings ranges DE 14.95–43.35, IT 10.02–32.67, SE 16.18–23.17 | `multiyear_summary.json` → `rule_deviation.savings_stability.{min,max}_savings_pp` |
| E7 | grace 24/12/12 (2025); stability DE 24/24/24, SE 12/12/12, IT 12/24/72 | `fixed_summary.json` → `graceLevels.persistence.horizon`; `multiyear_fixed_summary.json` → `grace_persistence` per (region, year) |
| E8 | budget sweep completed-feasible values (DE 11.51/16.28/29.52/43.35; IT 19.85/32.67; SE 14.59/18.36/22.87/23.17) | `budget_summary.json` → `regions[].ckpt_148_8[].best_completed.savings` |
| E9 | IT frontier collapses at B ≤ 50% | `budget_summary.json` → `regions[IT].collapse.ckpt_148_8.infeasible_budgets` = [30, 50] |
| E10 | IT nominal operating point 194.7% overhead | `reopt_summary.json` → `IT.baseline.overhead` = 194.68 |
| E11 | IT B=100 raw 82.00 vs completed-feasible 19.85 | `budget_summary.json` → `regions[IT].ckpt_148_8[budget=100]` `savings`=82.00, `best_completed.savings`=19.85 |
| E12 | checkpoint table (S₀/S@900/S@2700, retention) | `checkpoint_summary.json` → `runs[].bestCompleted.savings` + `baseline.savings` (retention computed) |
| E13 | margin widens DE 4.64→16.59 (900 s)→39.50 (2700 s) | `checkpoint_summary.json` → `runs[].bestCompleted.margin` (4.64 / 16.59 / 39.50) |
| E14 | raw best inflated (DE 47.91→82.18) at 900/2700 s | `checkpoint_summary.json` → `runs[].savings` (47.91 @900, 82.18 @2700) vs `bestCompleted.savings` (42.32/39.60); `completed`=false |
| E15 | optimizer score score=(α·(S/100)+1−(1−α)·(O/B))/2; at α=1 the overhead term vanishes (score=(S/100+1)/2), so budget binds only via the explicit overhead constraint | `src/domain/result.ts` `computeScore` (L24–33); `reopt_summary.json`/`adaptive_summary.json` `methodology` (α=1, B=200) |

> **F10 residual (tracked):** the Phase-E reproducibility statement (commands,
> runtimes, no-RNG/determinism, commit hash, per SPEC E.4) is a separate
> deliverable being produced by the orchestrator and will be released at
> de-anonymization; it is not part of this claims table. The score-function
> half of F10 is resolved (E15 above).

## Forecast Robustness (sec:robustness)

| # | Claim | Artifact → field |
|---|---|---|
| F1 | additive 4σ\* degradation 2.4 / 6.5 / 40.6 % | `fixed_{DE,IT,SE}.json` → `summary[]` family=`additive` param_value=4 `degradation_frac_mean` = 0.0241/0.0646/0.4055 |
| F2 | additive 1σ\* degradation 0.3 / 0.4 / 4.8 % | `fixed_{R}.json` → additive param_value=1 `degradation_frac_mean` = 0.0031/0.0037/0.0483 |
| F3 | multiplicative cheaper (DE 1.5% at 4σ\*) | `fixed_DE.json` → multiplicative param_value=4 `degradation_frac_mean` = 0.0149 |
| F4 | staleness h=72: 35.8/42.2/60.5%; h=12: 2.4/4.1/5.2% | `fixed_{R}.json` → `summary[]` family=`persistence` param_value 72/12 `degradation_frac_mean` |
| F5 | grace 24/12/12 steps | `fixed_summary.json` → `graceLevels.persistence.horizon` |
| F6 | scenario-level 6–15× ratio (staleness vs 4σ\* noise at the magnitudes each mode realistically produces, DE/IT) | computed = 35.8/2.4 = 14.9 (DE), 42.2/6.5 = 6.5 (IT) from F1/F4 — stated in §6.1 as a scenario comparison, not an intrinsic per-unit property |
| F7 | reopt drift margins DE 4.64→16.68→28.36; IT 16.25→23.19→30.15; SE 0.67→3.21→5.49 | `reopt_summary.json` → `baseline.margin` + `drift[]` family=`additive` param_value=1/2 `margin_drift` |
| F8 | fixed margin rule fails DE/IT ≥ 1σ\* additive; fails IT delay-1 | `reopt_summary.json` → `marginRuleSurvives[]` (additive 1/2 `survives`=false for DE/IT; delay 1 `survives`=false for IT) |
| F9 | grace prediction per region-year (g, RMSE(g), S, g_pred) | `grace_horizon.json` → `points[]` (g, rmseG, S.yearMean, thetaP, yearMeanCi) + `predicted[]` (gPred) |
| F10 | k=0.684, R²=0.913 through-origin; 0.915 intercept; n=8; full-9 R²=0.194; trainMean R²=0.862 | `grace_horizon.json` → `fits.yearMean.{noIT23,all}`; `fits.trainMean.noIT23` |
| F11 | 8/8 in-sample points within one grid step (IT-2023, the excluded right-censored outlier, does not) | `grace_horizon.json` → `predicted[]` (`within1Step` true for all 8 in-sample; IT-2023 `false`), `points[]` IT-2023 `isOutlier`=true, `h72DegFrac`=0.0312 |
| F12 | theoretical AR(1) RMSE under-widens 3.7× at h=72 (DE); 30.7 vs 113.4 | `grace_horizon.json` → `rmseEmpiricalVsTheoretical.DE[h=72]` ratio = 3.6983; theoretical 30.666 vs empirical 113.413 |
| F13 | SE fixed (published) control uses rounded thresholds (19, 18), s0 = 22.87%, distinct from its 2025 reopt optimum (18.18, 17.51 → 23.17%) | `fixed_SE.json` → `policy.{thetaP,thetaR}` = 19/18, `control.savings` = 22.87; `reopt_summary.json` → `SE.baseline.{thetaP,thetaR,savings}` = 18.18/17.51/23.17 |
| F14 | matched-magnitude comparison (F1, §6.1): at equal error magnitude additive noise is more damaging per unit error — DE 14.6 g/kWh→2.4% vs 13.9→0.71%; IT 17.7→6.5% vs 12.5→1.8%; SE 3.1→40.6% vs 2.6→5.2% | `calibration_{R}.json` → `evaluation[]` persistence rmse h=6 (DE 13.85, IT 12.5) / h=12 (SE 2.59) and 4×`sigmaStar` (DE 14.63, IT 17.68, SE 3.09); `fixed_{R}.json` → additive 4 / persistence 6|12 `degradation_frac_mean` |
| F15 | magnitude ratios at scenario endpoints: h=72 persistence RMSE / 4σ\* = 7.8× (DE 113.4/14.6), 4.4× (IT 77.7/17.7), 2.5× (SE 7.8/3.1) | `calibration_{R}.json` → `evaluation[]` persistence rmse h=72 ÷ (4×`sigmaStar`) |
| F16 | grace-fit caveat (F4): 8 region-years share only 4 distinct RMSE(g) values (2025 calibration reused); R²=0.91 is a level fit; continuous residuals −7…+8 steps; DE-2024 grid g_pred 72 vs g=24 (continuous 29) | `grace_horizon.json` → `points[]` rmseG (49.8×3 / 36.6 / 20.6 / 2.6×3); `predicted[]` `gPredCont` (18/29/22/5/17/14/20/18/6) and `devCont` (−6/+5/−2/−67/−7/+2/+8/+6/−6) |

## Stale-Aware Adaptive Control (sec:adaptive)

| # | Claim | Artifact → field |
|---|---|---|
| G1 | c\* = DE 0.75, IT 2, SE 1.5 | `adaptive_summary.json` → `regions[].chosenC` |
| G2 | c-selection mean recovery grid (Table 7) | `adaptive_summary.json` → `regions[].c_selection.meanRecovery` (0:0.0059/0.25:0.0062/0.5:0.0172/0.75:0.0556/1:0.0121/1.5:0/2:0.0158 for DE, etc.) |
| G3 | train signal ≤ 0.056; 2022 contributes zero recovery in every region; 2023 contributes zero for DE/IT and ≤0.018 for SE | `adaptive_summary.json` → `c_selection.meanRecovery` max 0.0556 (DE) / 0.0248 (IT) / 0.0289 (SE); `c_selection.perYear["2022"]` all zero; `perYear["2023"]` zero except `SE` `["0.75"]` = 0.0171 (≤0.018) |
| G4 | closed-loop table (naive/adapt/oracle/recovery/rvo per h) | `adaptive_summary.json` → `regions[].rows[]` {savings_naive, savings_adaptive, savings_oracle, recovery, recovery_vs_oracle, completed_adaptive} |
| G4b | DE h=1 headline closes 0.07 pp absolute (43.25→43.32 vs 43.35 perfect-foresight ceiling) | `adaptive_summary.json` → `regions[DE].rows[h=1]` savings_naive 43.2507, savings_adaptive 43.3197, savings_oracle 43.3230; `S0_FF` 43.35 (reopt) |
| G5 | completion guard changes only budget-exhausted rows; full definition recovery = clip((S_adapt−S_naive)/(S0_FF−S_naive),0,1), 0 when S0_FF ≤ S_naive (SE: perfect-foresight = rounded-threshold control 22.87 < naive); recovery_vs_oracle = 0 when oracle ≤ naive | `adaptive_summary.json` → `methodology.recovery` / `recoveryVsOracle`; verified: SE h=1 naive 23.086, adapt 21.977, perfect 22.868 → unclipped +5.10, artifact recovery 0, rvo 0 |
| G6 | static oracle h=72 recovers 32/48/30% of the loss | computed from `adaptive_{R}.json` h=72: (S_oracle−S_naive)/(S0_FF−S_naive) = (32.67−27.73)/(43.35−27.73)=0.316; IT 0.484; SE 0.302 |
| G7 | DE h=72 c=6 → S=31.43, completed, rvo=0.748; c=8 → 36.29 budget-infeasible | `adaptive_sensitivity_DE.json` → rows c=6 (savings=31.4279, completed, recovery_vs_oracle=0.7483) and c=8 (savings=36.29, completed=false) |
| G8 | feasible-c envelope (DE h=72 ≤6; IT: train-selected c\*=2 infeasible at h≥3, most of the c-grid from h=6; SE h=72 ≤0.75, no feasible recovery) | `adaptive_sensitivity_summary.json` → `feasibleC` per region/h (IT h=3 = [0..1.5], h=6 = [0..0.75]); `adaptive_sensitivity_SE.json` c=0.75 h=72 savings=2.62 |
| G9 | model interval under-widens vs empirical (3.7×, DE h=72) | `grace_horizon.json` → `rmseEmpiricalVsTheoretical.DE[h=72].ratio` |
| G10 | oracle h=72 center ≈ 283 vs nominal midpoint 270.1 | `adaptive_DE.json` → oracle θ=(407.47, 159.35) → center 283.4 (computed); nominal midpoint (272.37+267.73)/2 = 270.05 |
| G11 | DTPR β = 11.26 / 10.20 / 0.75 g/kWh | `adaptive_summary.json` → `regions[].dtpr.beta` |
| G12 | DTPR neutral for DE/SE (within 1 pp), budget-exhausted (infeasible) for IT at all h | `adaptive_summary.json` → `regions[].rows[].savings_dtpr` vs `savings_naive` (max \|Δ\| DE 0.25, SE 0.74); IT `rows[].overhead_dtpr` = 200.0016 > 200 (budget) at all h |
| G13 | IT nominal overhead 194.7%, ~5 pp headroom | `reopt_summary.json` `IT.baseline.overhead` = 194.68 vs budget 200 |

## Discussion (sec:discussion)

| # | Claim | Artifact → field |
|---|---|---|
| H1 | SE margin 0.67 below its σ\* 0.77 → noise-sensitive | `reopt_summary.json` `SE.baseline.margin` 0.67; `calibration_SE.json` `sigmaStar` 0.774 |
| H2 | SE degrades 40.6% at 4σ\* | `fixed_SE.json` → additive param_value=4 `degradation_frac_mean` = 0.4055 |
| H3 | IT-2023 grace 72 (right-censored; h=72 degradation 3.1% < 10%) | `multiyear_fixed_summary.json` IT-2023 `grace_persistence`=72; `grace_horizon.json` points[IT-2023] `h72DegFrac`=0.0312, `isOutlier`=true |
| H4 | grace as SLA: DE 24 steps (2 h), IT/SE 12 steps (1 h) | `fixed_summary.json` → `graceLevels.persistence.horizon` (24/12/12 steps × 5 min) |
| H5 | data spans five grids incl. US/CN (unused in experiments) | `public/data/co2/US_*.json`, `CN_*.json` present; experiments use DE/IT/SE only |

## Estimates flagged for D.10 scrutiny

| Claim | Source / basis | Note |
|---|---|---|
| DeepSeek-V3 ≈ 2.788M H800 GPU-hours | DeepSeek-V3 technical report (arXiv:2412.19437) | Published figure, cited; not re-measured |
| 671B checkpoint state ≈ 1.34 TB BF16 | 671e9 params × 2 bytes | Standard BF16 conversion; consistent with CarbonScaling MoE accounting; validate before camera-ready |
| PUE 1.27 | Jegham et al. (arXiv:2505.09598) via `constants.json` | Provider/paper figure |
| Kimi K2 profile (1T MoE, 15.5T tokens) | Kimi K2 technical report (arXiv:2507.20534) | Cited scale point; no Kimi experiments run |
| 561M curtailment model | Wiesner et al. (arXiv:2602.22760) | Reported figure from cited paper |
| 12× coarser hourly update | 60 min / 5 min | Arithmetic |
| σ\* of savings (positioning-sentence framing) | not used in the paper | superseded by the measured 4σ\* loss figures |

## Numbers requiring re-verification before submission (D.10 owner)

1. **Retention percentages** (97.6/94.6/95.2; 91.4/85.3/81.4): computed as `bestCompleted.savings / baseline.savings` per region; re-derive in the D.10 pass from `checkpoint_summary.json` and round to 1 dp.
2. **Static-oracle ceiling shares** (32/48/30%): computed from naive/perfect/oracle savings at h=72; re-derive from `adaptive_{R}.json`.
3. **6–15× ratio** (now phrased as scenario-level in §6.1): derived (F6); traceable as "14.9× (DE), 6.5× (IT)" = 35.8/2.4 and 42.2/6.5.
4. **k-per-point implied values** in Table 5 (0.46–0.92, CV≈0.25): from `grace_horizon.json` `points[].k.yearMean`; verify CV by hand.
5. **Oracle center 283.4** and nominal midpoint 270.05: computed from θ pairs; verify.
