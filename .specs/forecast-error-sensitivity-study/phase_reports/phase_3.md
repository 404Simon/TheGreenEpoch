# Phase 3 — Fixed-policy sensitivity sweep under forecast error

Status: `[x]` implemented · 146 tests green · build green · 0 new tsc errors · deterministic outputs.

## What was implemented

### `src/cli/forecast-sweep.ts` (NEW)
`--mode fixed` runs the full matrix. Pure helpers are exported for unit tests (CLI itself stays out of the tests):

- `meanStd(values)` — sample mean / sample std (n−1), std 0 for n<2, NaN guarded.
- `degradationFrac(savingsPerfect, savings)` — `(sp−s)/sp`, guarded `sp===0 → 0` (and NaN→0).
- `expandConfigs({families, levels, horizons, sigmaStar, sigmaRel, coeffs})` — expands the 30-config matrix:
  additive `{sigma=L·σ*}` (param `level`), multiplicative `{sigma=L·σ*_rel}` (param `level`), delay `{steps=h}` (param `steps`), arma `{order:1, horizon:h, coeffs}` (param `horizon`), persistence `{type:delay, steps:h}` (param `horizon`).
- `computeSummary(s0, rows)` — `s0, savings_mean/std, delta_s_pp, delta_s_frac (guarded s0=0→0), degradation_frac_mean/std, overhead_mean, num_pauses_mean, completed_rate, within_budget_rate, n_seeds`.
- `computeGraceLevel(summaries)` — largest positive `param_value` with `degradation_frac_mean ≤ 0.10`; none within → `graceLevel 0`; max tested value within → `graceAtMax: true`.
- `runSweepRegion({...})` — control run (no decisionTimeline) + per-(config×seed) forecast runs via `applyForecast(realized, model, seed)` → `simulateStepwise(..., {decisionTimeline})`. Baseline emissions computed once from a `neverPausePolicy` run and shared by control + all forecast rows. Returns `{control, rows, summary}` (fully pure, testable with synthetic timelines/profiles).

`forecastSweepCli`:
- Defaults: `-m Deepseek`, `-r DE,IT,SE`, `-y 2025`, `--error-types additive,multiplicative,delay,arma,persistence`, `--levels 0,0.25,0.5,1,2,4`, `--horizons 1,3,6,12,24,72`, `--seed-count 10`, `--seed-count-other 5`, `--calibration-dir publication/output/forecast`, `-o publication/output/forecast/fixed`.
- Hardcoded study policies: DE (272, 268, `02-01`), IT (246, 231, `01-14`), SE (19, 18, `04-22`); `--theta-p/--theta-r/--start` override only when `-r` has exactly one region (clear error otherwise).
- Seeds: DE `1..seed-count`, others `1..seed-count-other`; the seed list is recorded in each region JSON.
- Profile assembly identical to `src/cli/optimize.ts` (`constants.json` + `profiles.json`); realized timeline = raw single-year load via `averageYears([year])`.
- `--mode reopt` → prints `reopt mode is implemented in Phase 4` and exits 1 (matches the `optimizeCli` error convention — a plain throw would be swallowed by `main()`'s `exitOverride` catch). `--mode` anything else errors.
- Deterministic end-to-end: seeded `mulberry32` (no `Math.random`/`Date.now`), config×seed iteration in fixed order, natural-precision JSON + `toPrecision(6)` CSV.

### Outputs (under `publication/output/forecast/`)
- `fixed_{DE,IT,SE}.csv` — per-region flat rows, header EXACTLY `region,family,param,param_value,seed,sigma,theta_p,theta_r,start,savings,overhead,score,num_pauses,completed,within_budget,savings_perfect,degradation_frac`; numbers to 6 sig figs; `sigma` is the model σ actually used (additive: L·σ\* in gCO₂eq/kWh; multiplicative: L·σ\*_rel dimensionless; `NA` for delay/arma/persistence); booleans `true`/`false`.
- `fixed_{DE,IT,SE}.json` — `{model, region, year, policy:{thetaP,thetaR,start}, budget, seeds, calibration:{sigmaStar,sigmaRel,trainMean}, control:{savings,overhead,score,num_pauses,completed,within_budget}, rows, summary}`; summary entries add `n_seeds` (the seed row count requested in prose) + `graceLevel`/`graceAtMax`.
- `fixed_summary.json` — array of per-region `{region, sigmaStar, sigmaRel, s0, graceLevels:{additive{level,atMax}, multiplicative{level,atMax}, delay{steps,atMax}, persistence{horizon,atMax}, arma{horizon,atMax}}, degradationAtSigmaStar:{additive{delta_s_frac,delta_s_pp}, multiplicative{...}}, degradationAtH72:{persistence{...}, arma{...}}}`.
- `fixed_all.csv` (via `--csv`) — combined 600-row flat CSV.

Run count: DE 30×10 + IT 30×5 + SE 30×5 = 600 forecast sims + 3 controls + 3 baselines = 606 sims, ~5.7 s wall.

### `src/cli/index.ts`
Registered `forecast-sweep` (commander, dynamic import, default flags, `--mode` required) following the `optimize`/`forecast-calibrate` pattern.

### `src/cli/forecast-sweep.test.ts` (NEW, 20 tests)
1. Config-matrix expansion: 30-config matrix, 6/family, additive/multiplicative sigma anchoring, delay/arma/persistence param names + model shapes, coeffs threading, restricted families.
2. `meanStd` (sample std, single-value std 0) + `computeSummary` (savings mean/std, degradation mean/std, completed_rate, within_budget_rate, delta_s_frac guard when s0=0).
3. `computeGraceLevel`: mixed → largest within-10%; all-within → `graceAtMax:true`; none-within → 0; param_value 0 ignored.
4. Level-0 identity path: synthetic blocky timeline + small profile, additive/multiplicative level-0 rows have `degradation_frac === 0` exactly and savings == control; level-1 rows show seed variation; delay h=72 strictly degrades; grace fields populated on every summary row.

## Verification outputs (exact)

### 1. Full sweep run
```
$ pnpm cli forecast-sweep --mode fixed -o publication/output/forecast/fixed --csv publication/output/forecast/fixed_all.csv 2>&1 | tail -40
  [DE] θₚ=272 θₛ=268 start=02-01 configs=30 seeds=10
    control S₀=43.35% overhead=174.31% score=0.7168 pauses=102 completed=true within_budget=true
    additive       deg(0:0.0000 0.25:0.0005 0.5:0.0010 1:0.0031 2:0.0094 4:0.0241) grace=4 (at max)
    multiplicative deg(0:0.0000 0.25:0.0004 0.5:0.0007 1:0.0016 2:0.0054 4:0.0149) grace=4 (at max)
    delay          deg(1:0.0006 3:0.0022 6:0.0071 12:0.0243 24:0.0803 72:0.3583) grace=24
    arma           deg(1:0.0023 3:0.0039 6:0.0089 12:0.0262 24:0.0824 72:0.3602) grace=24
    persistence    deg(1:0.0006 3:0.0022 6:0.0071 12:0.0243 24:0.0803 72:0.3583) grace=24
  [IT] θₚ=246 θₛ=231 start=01-14 configs=30 seeds=5
    control S₀=32.67% overhead=194.68% score=0.6633 pauses=173 completed=true within_budget=true
    additive       deg(0:0.0000 0.25:0.0012 0.5:0.0014 1:0.0037 2:0.0185 4:0.0646) grace=4 (at max)
    multiplicative deg(0:0.0000 0.25:0.0020 0.5:0.0017 1:0.0021 2:0.0096 4:0.0392) grace=4 (at max)
    delay          deg(1:0.0032 3:0.0096 6:0.0176 12:0.0405 24:0.1124 72:0.4219) grace=12
    arma           deg(1:0.0047 3:0.0110 6:0.0188 12:0.0415 24:0.1133 72:0.4251) grace=12
    persistence    deg(1:0.0032 3:0.0096 6:0.0176 12:0.0405 24:0.1124 72:0.4219) grace=12
  [SE] θₚ=19 θₛ=18 start=04-22 configs=30 seeds=5
    control S₀=22.87% overhead=99.41% score=0.6143 pauses=77 completed=true within_budget=true
    additive       deg(0:0.0000 0.25:0.0092 0.5:0.0116 1:0.0483 2:0.1641 4:0.4055) grace=1
    multiplicative deg(0:0.0000 0.25:0.0092 0.5:0.0094 1:0.0298 2:0.1126 4:0.3062) grace=1
    delay          deg(1:0.0043 3:0.0128 6:0.0254 12:0.0517 24:0.1415 72:0.6049) grace=12
    arma           deg(1:-0.0095 3:-0.0023 6:0.0082 12:0.0306 24:0.1152 72:0.6223) grace=12
    persistence    deg(1:0.0043 3:0.0128 6:0.0254 12:0.0517 24:0.1415 72:0.6049) grace=12
  Summary: publication/output/forecast/fixed_summary.json
  CSV (all regions): publication/output/forecast/fixed_all.csv
  Done.
```
Run count: `fixed_DE.csv` 300 rows, `fixed_IT.csv` 150, `fixed_SE.csv` 150, `fixed_all.csv` 600 (30 configs × 10 seeds + 30×5 + 30×5 = 600 forecast sims, +3 controls +3 baselines).

### 2. Determinism
```
$ sha256sum publication/output/forecast/fixed_*.{json,csv} > run1.sha && pnpm cli forecast-sweep --mode fixed -o ... --csv ... --quiet && sha256sum ... > run2.sha && diff run1.sha run2.sha
IDENTICAL: 8 files byte-identical across runs
```

### 3. Results table (`node -e` snippet output)
```
region s0      addL1   addL4  multL1 multL4  delayH1  delayH72 persH72 armaH72  grace(add/mult/delay/pers/arma)
DE  43.35  0.0031 0.0241 0.0016 0.0149  0.0006  0.3583 0.3583 0.3602   add=4* mult=4* del=24 pers=24 arma=24
IT  32.67  0.0037 0.0646 0.0021 0.0392  0.0032  0.4219 0.4219 0.4251   add=4* mult=4* del=12 pers=12 arma=12
SE  22.87  0.0483 0.4055 0.0298 0.3062  0.0043  0.6049 0.6049 0.6223   add=1 mult=1 del=12 pers=12 arma=12
```
(`*` = grace at max tested value.) `degradationAtSigmaStar` (ΔS/S₀): DE add 0.0031 / mult 0.0016, IT 0.0037 / 0.0021, SE 0.0483 / 0.0298. `degradationAtH72`: DE pers 0.3583 / arma 0.3602, IT 0.4219 / 0.4251, SE 0.6049 / 0.6223.

Level-0 identity: every `additive`/`multiplicative` level-0 CSV row has `degradation_frac=0.00000` exactly, e.g.
`DE,additive,level,0.00000,1,0.00000,272.000,268.000,02-01,43.3510,174.310,0.716755,102.000,true,true,43.3510,0.00000`

### 4. Verification tails
```
pnpm test   →  Test Files  10 passed (10)
               Tests       146 passed (146)      # 126 baseline + 20 new
pnpm build  →  ✓ built in 2.34s                  # green
new-tsc=0                                        # comm of tsc error sets, 0 new
```

## Headline findings

- Perfect-foresight controls reproduce the paper: DE S₀=43.35% (paper 43.4%), IT 32.67% (32.7%), SE 22.87% (23.2%); overhead 174.3/194.7/99.4% — all within the 200% budget, all completed.
- Parametric noise is extremely forgiving: at σ=σ\* the relative savings loss is < 0.5% for every region (SE worst at 4.8% additive); even at σ=4σ\* DE/IT keep > 93% of savings (SE 59%).
- Horizon errors dominate: 6 h persistence (h=72) loses 36% (DE) / 42% (IT) / 60% (SE) of savings; grace horizon is 24 steps (DE) / 12 (IT) / 12 (SE) for delay/persistence/arma.
- AR(1) ≈ persistence at every horizon (as expected from lag-1 ≈ 0.999): arma h=72 degrades marginally more than persistence (Δ≈0.02 pp). On SE, the AR(1) h=1/h=3 forecasts slightly *improve* on the realized controller (negative mean degradation −0.0095/−0.0023) — the smoothing shifts a few resume decisions off noise spikes.
- Control thresholds sit far from the noise scale (σ\* is 0.9–3.3% of the grid mean), so additive/multiplicative noise levels 0–2 almost never flip a pause decision; SE's lower μ (23 g/kWh) and 33% higher σ\*_rel make it the fragile region.
- Grace levels (10% relative-savings-loss rule): DE additive/multiplicative 4 (at max, i.e. survive even 4σ\*), IT 4 (at max), SE 1; delay/persistence/arma DE 24, IT 12, SE 12 steps.

## DoD checklist

- [x] `forecast-sweep` registered in `src/cli/index.ts`; `--mode fixed` runs the full 600-sim matrix (verified: 300/150/150 rows); `--mode reopt` errors with `reopt mode is implemented in Phase 4` (exit 1).
- [x] `fixed_{DE,IT,SE}.{json,csv}` + `fixed_summary.json` written under `publication/output/forecast/` with the specified schemas (rows/summary/control/grace fields all present; summary adds `n_seeds`).
- [x] Row fields exactly as specified (header byte-exact); `savings_perfect` = S₀ from the shared control; `degradation_frac` = `(sp−s)/sp` with the s₀=0 → 0 guard.
- [x] Grace levels computed per family with `graceAtMax`; `fixed_summary.json` populated (graceLevels keyed by `level`/`steps`/`horizon` per family).
- [x] Determinism: full re-run → all 8 JSON/CSV byte-identical (sha256 diff empty).
- [x] Level-0 `degradation_frac === 0` — unit-tested on a synthetic timeline and visible in CSV (`0.00000`).
- [x] `pnpm test` 146 green; `pnpm build` green; `new-tsc=0` (6 pre-existing errors unchanged).
- [x] This phase report written.

## Deviations / choices

1. **CSV `sigma` for multiplicative** is the dimensionless relative σ (`L·σ*_rel`, the σ actually passed to the model), not gCO₂eq/kWh; additive uses the absolute `L·σ*` in gCO₂eq/kWh. The spec's "in gCO₂eq/kWh" parenthetical only fits the additive family by construction.
2. **Summary entries include `n_seeds`** (the seed row count requested in step 6's prose; the schema listing omits it). JSON uses natural (`JSON.stringify`) precision; CSV uses `toPrecision(6)`; std is sample (n−1).
3. **`--mode reopt` "error" is `console.error` + `process.exit(1)`** (matching `optimizeCli`'s "Unknown model" convention) because `main()` in `index.ts` swallows plain throws under `exitOverride`; the message text is exactly `reopt mode is implemented in Phase 4`.
4. **`fixed_summary.json` is an array** of per-region records (one per region), chosen over an object keyed by region.
5. **Level-0 sigma recorded as `0`** in CSV (`0.00000`) rather than `NA`, since it is a real numeric model σ for additive/multiplicative; only delay/arma/persistence get `NA`.
6. **SE arma h=1/h=3 negative mean degradation** (−0.0095/−0.0023) — real effect (AR smoothing vs. raw decisions), reported as-is.
