# Phase 2 — Forecast-Error Sensitivity Study: Calibration (Tier-2 anchor)

Status: `[x]` implemented · tests green · build green · no new tsc errors.

## What was implemented

### `src/cli/forecast-calibrate.ts` (NEW)
Exports `computeAutocorr`, `fitArPooled`, `innovationStd`, `evaluateHorizon`, `runCalibration`, `calibrateCli` (plus `HorizonMetrics`/`HorizonEvaluation`):

- `computeAutocorr(series, lag)` — Pearson correlation of `series[t]` with `series[t-lag]`; NaN on zero-variance denominator; throws on non-finite/invalid lag/short series.
- `fitArPooled(perYearSeries, order)` — pooled within-year Yule–Walker: autocovariance sums `γₖ = Σ_years Σ_t (y−μ̂)(yₜ₋ₖ−μ̂)` centered on the pooled mean, Toeplitz system solved by Gaussian elimination with partial pivoting (same approach as `fitAr` in `src/domain/forecast.ts`, re-implemented locally per spec); `intercept = μ̂(1−Σφ)`; throws on degenerate (constant/singular) input and any year shorter than `order`.
- `innovationStd(perYearSeries, order, coeffs)` — pooled residual std of the AR fit over within-year residuals only, dof = n − order − 1 (n = total pooled points).
- `evaluateHorizon(trainCoeffs, testSeries, horizon)` — persistence (`ŷ_t = y_{t−h}`) and AR (`ŷ_t = intercept + Σφᵢ·y_{t−h−i+1}`) compared on the same union-valid range `t ∈ [h+order−1, end)`; RMSE/MAE/MAPE with the `|y|>1` gCO₂eq/kWh guard; MAPE of an empty guard set returns 0 (never NaN/inf); throws if the test series is too short.
- `runCalibration(region, trainYears, testYear, orders, horizons, trainSeries?, testSeries?)` — builds the full `CalibrationBundle` (schema below); train/test series are optional so the CLI loads raw per-year files (concatenated, NOT `averageYears`-averaged) and unit tests inject synthetic series. `sigmaStar` = AR(1) innovation std (computed directly when order 1 isn't requested). lag-1/lag-2 autocorr from the pooled `γ₁/γ₀`, `γ₂/γ₀` (same within-year convention as the fit).
- `calibrateCli(raw)` — mirrors `optimizeCli`: parses `--regions/--train/--test/--orders/--horizons` (defaults `DE,IT,SE` / `2022,2023,2024` / `2025` / `1,7` / `1,3,6,12,24,72`), writes `publication/output/forecast/calibration_{region}.json` + `.csv` (`region,order,model,horizon,rmse,mae,mape`, ~6 sig figs, no BOM), prints a compact summary table.

### `src/domain/types.ts`
Added `CalibrationRow`, `CalibrationOrderInfo`, `CalibrationBundle` (consumed by Phase 3/5).

### `src/cli/index.ts`
Registered `forecast-calibrate` (commander, dynamic import, following the `optimize` command pattern; all five flags have defaults).

### `src/cli/forecast-calibrate.test.ts` (NEW, 8 tests)
1. `fitArPooled` recovers φ=0.9 / intercept=50 on a synthetic AR(1) split into 3×2000-point "years" (tolerances 0.05 / 5).
2. `innovationStd` recovers the innovation scale (~1) of the pooled fit.
3. `computeAutocorr` ≈ 0 at a 90° phase shift of a sine (window `n−lag` = 2 full cycles ⇒ ~1e-16), ≈ 1 at lag 1 for a constant-shifted smooth series.
4. `evaluateHorizon`: persistence RMSE at h=1 much smaller than at h=72 on a near-random-walk series.
5. MAPE guard: constant-zero series → all metrics finite (0, no NaN/inf).
6. Smoke: `runCalibration(orders=[1], horizons=[1,3])` on synthetic series → bundle schema keys present, `evaluation.length` = 1×2×2 = 4, models are `{ar,ar,persistence,persistence}`.
7. `fitArPooled` throws on constant (degenerate) input.

## Verification outputs (exact)

### 1. CLI run (defaults) + output dir
`pnpm cli forecast-calibrate 2>&1 | tail -40` (last region shown; full table in §Calibration numbers):

```
  Forecast calibration ─ SE (train 2022,2023,2024 / test 2025)
  mean=23.44  std=8.66  cv=0.3695  lag1=0.995950  lag2=0.992010  σ*=0.77
  order 1  φ=[0.9959]  σ*=0.77
    h    AR rmse   PERS rmse   gap
     1     0.776     0.777   0.001
     3     1.343     1.346   0.002
     6     1.874     1.878   0.004
    12     2.581     2.587   0.005
    24     4.220     4.229   0.008
    72     7.825     7.841   0.016
  order 7  φ=[0.9846, 0.0203, -0.0119, 0.0028, 0.0136, -0.0202, 0.0069]  σ*=0.77
    h    AR rmse   PERS rmse   gap
     1     0.776     0.777   0.001
     ...
    72     7.825     7.840   0.016
  ────────────────────────────────────────────────────
  JSON: publication/output/forecast/calibration_SE.json
  CSV:  publication/output/forecast/calibration_SE.csv
  Done.
```

```
$ ls -la publication/output/forecast/
total 44
drwxr-xr-x 1 simon simon  240 Aug 18 12:34 .
-rw-r--r-- 1 simon simon 1021 Aug 18 12:33 calibration_DE.csv
-rw-r--r-- 1 simon simon 5144 Aug 18 12:33 calibration_DE.json
-rw-r--r-- 1 simon simon 1021 Aug 18 12:33 calibration_IT.csv
-rw-r--r-- 1 simon simon 5151 Aug 18 12:33 calibration_IT.json
-rw-r--r-- 1 simon simon 1041 Aug 18 12:33 calibration_SE.csv
-rw-r--r-- 1 simon simon 5167 Aug 18 12:33 calibration_SE.json
-rw-r--r-- 1 simon simon 5608 Aug 18 12:34 README.md
```

### 2. `calibration_DE.json` sanity
`lag1AutoCorr` = **0.9996547** (≈ 0.999 ✓), `sigmaStar` = 3.658 (plausible: matches AR(1) residual std and the h=1 RMSE ≈ 4.21), `evaluation` fully populated (24 rows). SE `trainMean` = **23.44** (≈ 23 ✓). Cross-checked pooled AR(1)/σ\* against Phase-1 `fitAr`/`forecastInnovationStd` on the concatenated train series — identical (DE φ 0.999655 vs 0.999671; σ\* 3.658 both).

### 3. `pnpm test` — all green
```
 Test Files  9 passed (9)
      Tests  126 passed (126)
```
(118 baseline + 8 new.)

### 4. tsc
`new-tsc-errors=0` (6 pre-existing errors unchanged, identical to baseline).

### 5. `pnpm build`
```
✓ built in 2.95s
```
(green; the pre-existing 500 kB chunk warning is unrelated.)

## Key calibration numbers (train 2022–2024, test 2025)

| Region | μ (g/kWh) | σ\* (g/kWh) | σ*_rel | lag-1 | lag-2 | AR(1) φ |
|--------|-----------|-------------|--------|--------|--------|---------|
| DE | 398.2 | 3.66 | 0.00919 | 0.999655 | 0.999315 | 0.99965 |
| IT | 325.6 | 4.42 | 0.01358 | 0.998681 | 0.997363 | 0.99868 |
| SE | 23.4  | 0.77 | 0.03300 | 0.995950 | 0.992010 | 0.99595 |

Persistence-vs-AR(1) RMSE (g/kWh) on 2025 — gap = persistence − AR(1):

| Region | h=1 | h=3 | h=6 | h=12 | h=24 | h=72 |
|--------|------|------|------|-------|-------|-------|
| DE AR(1) / persistence | 4.21 / 4.21 | 7.24 / 7.24 | 13.85 / 13.85 | 26.54 / 26.54 | 49.76 / 49.77 | 113.41 / 113.43 |
| DE gap | 0.000 | 0.001 | 0.002 | 0.005 | 0.009 | 0.020 |
| IT AR(1) / persistence | 5.02 / 5.02 | 8.69 / 8.70 | 12.49 / 12.50 | 20.62 / 20.63 | 36.64 / 36.67 | 77.61 / 77.66 |
| IT gap | 0.002 | 0.005 | 0.008 | 0.013 | 0.024 | 0.051 |
| SE AR(1) / persistence | 0.78 / 0.78 | 1.34 / 1.35 | 1.87 / 1.88 | 2.58 / 2.59 | 4.22 / 4.23 | 7.83 / 7.84 |
| SE gap | 0.001 | 0.002 | 0.004 | 0.005 | 0.008 | 0.016 |

Finding: persistence is within < 0.1 % of AR(1) RMSE at every horizon (gap only reaches 0.02–0.05 g/kWh at h=72). AR(7) helps a little at h=1 (DE 2.30 vs 4.21) but the DE AR(7) fit is numerically ill-conditioned (near-cancelling coefficient pairs from the near-singular Yule–Walker matrix at lag-1 = 0.9997), so AR(1) is the anchor.

## Chosen sweep levels (documented in `publication/output/forecast/README.md`)

- Additive σ (g/kWh) = `{0, ¼, ½, 1, 2, 4} × σ*` → DE `{0, 0.91, 1.83, 3.66, 7.32, 14.63}`, IT `{0, 1.11, 2.21, 4.42, 8.84, 17.68}`, SE `{0, 0.19, 0.39, 0.77, 1.55, 3.09}`. Justified: level 1 ≈ σ\* ≈ the h=1 forecast RMSE.
- Multiplicative σ = `{0, ¼, ½, 1, 2, 4} × σ*_rel` (log-normal log-std) → DE `{0, 0.0023, 0.0046, 0.0092, 0.0184, 0.0367}`, IT `{0, 0.0034, 0.0068, 0.0136, 0.0272, 0.0543}`, SE `{0, 0.0083, 0.0165, 0.0330, 0.0660, 0.1320}`. Justified: level 1 gives relative error with the same CV as the empirical innovation.
- Delay steps `{1, 3, 6, 12, 24, 72}` (5 min … 6 h); ARMA horizons `{1, 3, 6, 12, 24, 72}` with the calibrated AR(1) coefficients (DE φ=0.99965, IT φ=0.99868, SE φ=0.99595).

## DoD checklist

- [x] `forecast-calibrate.ts` exports `computeAutocorr`, `fitArPooled`, `innovationStd`, `evaluateHorizon`, `runCalibration`, `calibrateCli`.
- [x] CLI command registered and runs with defaults, producing exactly 3 JSON + 3 CSV under `publication/output/forecast/`.
- [x] JSON follows the schema exactly (region, trainYears, testYear, trainMean/Std/Cv, lag1/2AutoCorr, sigmaStar, orders{coeffs,innovationStd}, evaluation rows); DE lag-1 = 0.99965 ≈ 0.999; region means DE 398.2 / IT 325.6 / SE 23.4.
- [x] Test file covers the 4 listed cases (fit recovery, sine/shifted autocorr, persistence horizon growth, MAPE guard) plus smoke/degenerate tests — all pass.
- [x] `README.md` in `publication/output/forecast/` documents levels + justification + MAPE/pooling conventions + exact commands + date.
- [x] `pnpm test` (126) / `pnpm build` green; 0 new tsc errors.

## Deviations

1. **`runCalibration` signature** — spec lists `runCalibration(region, trainYears, testYear, orders, horizons)`. It is implemented as `(…, trainSeries?, testSeries?)` so unit tests can inject synthetic series (per the required smoke test); the CLI omits them and the function loads the raw per-year files itself. Behavior with default args matches the spec exactly.
2. **σ\* magnitude vs spec illustration** — the schema example shows DE `sigmaStar: 52.4`; the real data gives 3.66 gCO₂eq/kWh. This is data-consistent (lag-1 = 0.999655 ⇒ σ\* = σ·√(1−φ²) ≈ 143·0.0263 ≈ 3.8), verified against Phase-1 `fitAr`/`forecastInnovationStd`, so the spec example numbers were illustrative.
3. **Reference means** — paper's grid stats (DE ≈ 380, IT ≈ 309, SE ≈ 23) match only SE (23.4) with the current `public/data/co2`; DE/IT train means are ~5 % higher (398.2 / 325.6). Likely an earlier data snapshot; noted in the README.
4. **DE AR(7) ill-conditioned** — near-cancelling coefficient pairs (e.g. +0.533 / −0.533) from the near-singular Yule–Walker system. Faithful to the requested AR(7) fit and reproducible; documented so Phase 3 uses the defensible AR(1) anchor.
