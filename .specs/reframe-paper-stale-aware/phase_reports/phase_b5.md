# Phase B.5.1 — Validate the grace-horizon prediction (analysis report)

- Date: 2026-08-19
- Agent: analysis agent, Phase B.5.1 (`reframe-paper-stale-aware`, stretch goal)
- Verdict: **CONDITIONAL HEADLINE.** With the documented right-censored IT-2023
  outlier excluded, RMSE(g) ≈ k·S with **k = 0.684**, **R² = 0.913**
  (through-origin, n = 8) for μ = decision-year mean CI; intercept fit
  R² = 0.915. Full-9-point R² = 0.194 (outlier breaks it); μ = committed
  `trainMean` → R² = 0.862 (near-miss). The SPEC's theoretical AR(1) RMSE is
  unusable for the prediction (under-predicts empirical by ≈3.7× at h = 72).
- Deliverables: `publication/output/forecast/grace_horizon.md` (the artifact),
  `grace_horizon.json` (all computed numbers), `grace_horizon_analysis.mjs`
  (deterministic analysis script). **No source files modified.**
- Method: **analysis-only over committed JSON** (`calibration_*.json`,
  `multiyear_fixed_summary.json`, `multiyear_summary.json`, `fixed_summary.json`,
  `public/data/co2/*.json`).

## 1. What was done

1. **Empirical grace horizon** g per (region, year) from
   `multiyear_fixed_summary.json` (`grace_persistence`; `grace_delay` equals it
   in all 9 cells). 2025 values cross-checked against `fixed_summary.json`
   `graceLevels` — exact match for all three regions (DE 24/24, IT 12/12,
   SE 12/12).
2. **RMSE at g** = empirical AR(1) evaluation RMSE (`order:1, model:"ar"`) at
   h = g from `calibration_{region}.json`. All g ∈ grid {1,3,6,12,24,72}, so no
   interpolation was needed for the table; the 2025 calibration bundle is reused
   for 2023/2024 exactly as in the B.3.2 sweep (documented B.3 choice).
3. **Threshold-margin scale S = |θ_p − μ|**, μ = decision-year mean CI
   (primary). Sign issue addressed explicitly: θ_p < μ in all nine cells (e.g.
   DE-2025 θ_p = 272.37 < μ = 339.94), so S uses the absolute value — the
   excursion magnitude sets the error budget. Sensitivity: μ = committed
   `trainMean` (R² = 0.862) and year-median (R² = 0.778).
4. **Fit.** RMSE(g) ≈ k·S through-origin (k = Σxy/Σx²) and with intercept;
   R² = 1 − SSE/SST (SST on the y-scale) in both. Primary sample = 8 points,
   IT-2023 excluded (right-censored: h = 72 degradation 3.1 % < 10 %, grace is
   a grid-boundary artifact, not an interior crossing). Fits for all 9 points
   and all three μ options reported.
5. **Predicted grace** g_pred = smallest h in the grid with empirical
   RMSE(h) ≥ k·S using the global k = 0.684, plus a continuous log-h
   interpolation (g_pred_cont) to de-quantize. Compared to g_empirical.
6. **Theoretical RMSE** σ*·sqrt((1−φ^2h)/(1−φ²)) computed for comparison:
   ratio to empirical grows 1.0 (h ≤ g) → 2–4× at h = 72, and the theoretical
   crossing lies beyond the grid for DE (never crosses k·S) — hence the paper
   must use the **empirical** RMSE curve, not the model formula.

## 2. Method / definitions (as implemented)

- **g**: `grace_persistence` (paper uses the persistence/delay family);
  `grace_delay` identical in 9/9 cells; `grace_arma` differs only in SE-2023
  (24 vs 12).
- **RMSE(g)**: empirical AR(1) (order 1, model "ar") evaluation RMSE at h = g;
  a region's RMSE(h) is year-invariant here (2025 bundle reused), a documented
  B.3 choice.
- **S = |θ_p − μ|**: θ_p from `multiyear_summary.json`; μ = decision-year mean
  CI (primary, same-distribution argument). Absolute value used because
  θ_p < μ in all 9 cells (pause-mostly control regime).
- **Fit**: through-origin y = k·x and intercept y = a + b·x; R² = 1 − SSE/SST
  on the y-scale (comparable); n = 8 (primary) and 9 (full).
- **g_pred**: smallest h ∈ {1,3,6,12,24,72} with empirical RMSE(h) ≥ k·S
  (k = global fitted, primary); g_pred_cont via log-h interpolation; deviation
  g − g_pred reported in steps.

## 3. Per-(region, year) table

Full table with sources in `grace_horizon.md` §2/§6; condensed here
(μ = year-mean; k_implied = RMSE(g)/S; g_pred with global k = 0.684):

| region | year | g | RMSE(g) | S | k_implied | g_pred | g_pred_cont | g−g_pred | within-1 |
|---|---|---|---|---|---|---|---|---|---|
| DE | 2023 | 24 | 49.76 | 59.45 | 0.837 | 24 | 18 | 0 | yes |
| DE | 2024 | 24 | 49.76 | 88.96 | 0.559 | 72 | 29 | +48 | yes |
| DE | 2025 | 24 | 49.76 | 67.57 | 0.736 | 24 | 22 | 0 | yes |
| IT | 2023* | 72 | 77.61 | 16.33 | 4.754 | 6 | 5 | −66 | no |
| IT | 2024 | 24 | 36.64 | 41.59 | 0.881 | 24 | 17 | 0 | yes |
| IT | 2025 | 12 | 20.62 | 35.29 | 0.584 | 24 | 14 | +12 | yes |
| SE | 2023 | 12 | 2.58 | 5.61 | 0.460 | 24 | 20 | +12 | yes |
| SE | 2024 | 12 | 2.58 | 5.26 | 0.491 | 24 | 18 | +12 | yes |
| SE | 2025 | 12 | 2.58 | 2.81 | 0.917 | 12 | 6 | 0 | yes |

\* IT-2023 right-censored outlier (excluded from the fit). 8/8 in-sample points
within one grid position; 3/8 exact (DE-23, DE-25, IT-24).

## 4. Fit results and verdict

| μ | sample | n | through-origin k | R² | intercept (a,b) | R² |
|---|---|---|---|---|---|---|
| year mean CI (primary) | all 9 | 9 | 0.7401 | 0.1936 | (14.68, 0.495) | 0.3283 |
| **year mean CI (primary)** | **8, no IT-23** | **8** | **0.6840** | **0.9134** | **(1.50, 0.660)** | **0.9154** |
| trainMean | 8, no IT-23 | 8 | 0.3782 | 0.8622 | (1.56, 0.364) | 0.8643 |
| year median | 8, no IT-23 | 8 | 0.6904 | 0.7777 | (3.79, 0.628) | 0.7907 |

**Verdict:** R² ≥ 0.9 **holds on the 8-point in-sample fit** (0.913, primary μ)
→ qualifies as a **headline contribution** with the paper carrying three
explicit caveats: (1) the IT-2023 censored outlier (explainable: a year where
staleness never degrades past 3 % at h = 72) breaks the full-sample R² (0.194);
(2) the μ choice matters — with the committed `trainMean` the R² is 0.86, a
near-miss; (3) the fit is on RMSE(g) levels; horizon-space prediction is
coarser (8/8 in-sample within one grid position; continuous residual 2–8 steps). The
fallback framing if reviewers reject the caveats: a design-rule observation
("grace ≈ the h where empirical AR(1) RMSE ≈ 0.7·S"), since k is regionally
stable (0.46–0.92, CV ≈ 0.25) without being a universal constant.

Recommended paper paragraph (headline framing) is in `grace_horizon.md` §5.

## 5. DoD checklist — pass/fail with evidence

- **D1 `grace_horizon.md` exists** — PASS. Per-region/year table (g, RMSE at g,
  S, k, g_pred, deviation) + fit (k, R², n) + clear verdict
  (`publication/output/forecast/grace_horizon.md`).
- **D2 every number traceable to committed JSON; md lists source per column** —
  PASS. §6 of the md maps every column to `calibration_*.json`,
  `multiyear_fixed_summary.json`, `multiyear_summary.json`,
  `fixed_summary.json`, `public/data/co2/*.json` (all committed; verified via
  `git ls-files`). The 2025 grace levels cross-check to `fixed_summary.json`
  exactly. All numbers also in `grace_horizon.json` (script output).
- **D3 μ and S definitions explicit; sign issue addressed** — PASS. §1 of the
  md and §2 here: μ = decision-year mean CI (primary), trainMean and median as
  sensitivity; S = |θ_p − μ| with the θ_p < μ observation explicit in all nine
  cells.
- **D4 R² computed correctly; formula and n stated; both fits reported; verdict
  states R² ≥ 0.9** — PASS. R² = 1 − SSE/SST (y-scale) for both through-origin
  and intercept; n = 8 primary / 9 full; primary R² = 0.913 ≥ 0.9
  (through-origin) and 0.915 (intercept); full-sample 0.194; trainMean 0.862.
  Independent spot-check of k (Σxy/Σx²) and implied k by hand agrees with the
  script.
- **D5 `pnpm test` green + `git diff --stat src/` only pre-existing B.0/B.1
  changes** — PASS. `pnpm test` → 14 files, **205 tests passed**. `git diff
  --stat src/` shows only the B.0/B.1 modifications (forecast-sweep.ts,
  index.ts, optimize.ts); the untracked src files (adaptive-margin.ts,
  checkpoint-sweep.test.ts, adaptive-sweep.test.ts) are B.1/B.0 artifacts, not
  mine. This phase added only `grace_horizon_analysis.mjs`,
  `grace_horizon.json`, `grace_horizon.md` (publication/output/forecast).
- **D6 `phase_reports/phase_b5.md` written** — PASS (this file).

## 6. Deviations / decisions / ambiguities resolved

1. **μ choice (SPEC formula loose).** Chose the decision-year mean CI as
   primary (θ_p and S measured on the same distribution; year-specific per the
   B.3 finding that absolute quantities are year-specific). Reported trainMean
   (committed, R² = 0.86 near-miss) and year-median (R² = 0.78) as sensitivity.
   The verdict therefore depends on this documented choice.
2. **IT-2023 handling.** Its empirical grace (72) is right-censored at the grid
   maximum (h = 72 degradation only 3.1 % < 10 %), so it is not an interior
   crossing and is excluded from the primary fit — standard treatment of a
   censored observation. It is still reported (k_implied 4.75, g_pred 6,
   far off) and the full-9-point R² (0.194) is reported so the honest negative
   is visible.
3. **Theoretical RMSE excluded from g_pred.** σ*·sqrt((1−φ^2h)/(1−φ²))
   under-predicts the empirical evaluation RMSE by up to ≈3.7× at h = 72 (the
   B.1 review M1 under-widening), so the SPEC's "theoretical crossing" is
   beyond the grid for DE and far above g elsewhere. The paper must use the
   **empirical** evaluation RMSE for the grace prediction; the theoretical
   curve is reported as comparison only.
4. **RMSE reused across years.** The calibration bundle is evaluated on 2025;
   it is reused for 2023/2024 (B.3.2's documented choice), so a region's
   RMSE(h) is year-invariant in the table. Noted in the md §1.
5. **Grid quantization.** g_pred on {1,3,6,12,24,72} is coarse (adjacent cells
   span 2–3×); a continuous log-h interpolation (g_pred_cont) is reported so
   the true residual scale (2–8 steps) is visible rather than hidden by the
   "within-1-grid-position" metric (which uses the same convention as B.3.2).
6. **No source changes.** The analysis is a standalone deterministic `.mjs`
   script; no `src/` file touched (D5 evidence).

## 7. Exact commands

```bash
# Regenerate the analysis + grace_horizon.json (deterministic, read-only over committed JSON):
node publication/output/forecast/grace_horizon_analysis.mjs

# Verify determinism (two runs produce byte-identical grace_horizon.json):
node publication/output/forecast/grace_horizon_analysis.mjs >/tmp/gh_run1.txt
cmp publication/output/forecast/grace_horizon.json /tmp/gh_run1.txt 2>/dev/null || true
# (script overwrites the json; re-run then diff a copy kept before the re-run)

# Full suite + no-source-change check:
pnpm test                                        # 14 files, 205 tests passed
git diff --stat src/                             # only pre-existing B.0/B.1 files
```

Artifacts added (this phase): `publication/output/forecast/grace_horizon.md`
(deliverable), `grace_horizon.json` (computed numbers, incl. per-point table,
fits, cross-check, verdict), `grace_horizon_analysis.mjs` (deterministic
script). Nothing committed; `publication/ICREC_Rome/` untouched; no source
file modified.
