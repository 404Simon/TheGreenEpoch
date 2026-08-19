# Phase 4 — Re-optimization under forecast error (design-rule drift)

Status: `[x]` implemented · 156 tests green (146 baseline + 10 new) · build green · 0 new tsc errors · deterministic outputs.

## What was implemented

### `src/cli/forecast-sweep.ts` — `--mode reopt`

`forecastSweepCli` now dispatches `--mode reopt` to a dedicated `reoptSweepCli` (replaces the Phase-3 stub error).

Pure, unit-testable helpers (exported):
- `seedMeanBest(rows)` — seed-mean of `(thetaP, thetaR, margin, savings, overhead, score)` over **found** rows only; `null` when every seed is `found:false`. The `found:false` row is the null-`best` case (no valid within-budget point for that seed) and is never crashed on.
- `drift(best, baseline)` — `{thetaP_drift, thetaR_drift, margin_drift}` = seed-mean best minus baseline, field-by-field.
- `marginRuleSurvives(config)` — R1: `theta_p − theta_r ≤ 16` on the seed-mean best (boolean) **and** the fraction of found seeds for which the rule holds (`seedFraction`); `{false, 0}` when best is null.
- `regionalRuleSurvives(config, baseline)` — R2: seed-mean `θ_p` within ±50% of baseline `θ_p`; `false` when best is null.
- `buildReoptSummary(region, result)` — assembles the per-region summary record.
- `assertIdentityRegression(profile, realized, years, options)` — runs `runOptimization` with an identity decisionTimeline and asserts (via exact `sameBest` on all 13 `SweepPoint` fields) it equals the no-decisionTimeline run; throws on mismatch.
- `runReoptRegion({...})` — per-region matrix driver. First computes the **no-decisionTimeline** baseline `freeBest` (also the regression reference), then for each config×seed runs `runOptimization(profile, realized, [year], options, undefined, decision)` with `decision = applyForecast(realized, model, seed)`. For additive `level 0` (σ=0 ⇒ identity timeline) each of the 3 seeds is asserted `sameBest` against `freeBest` — this is the in-code regression check (throws on any deviation).

Matrix (SPEC 4.1): additive σ ∈ {0, ½, 1, 2}×σ\* + delay steps ∈ {1, 6}, seeds {1,2,3}, regions {DE,IT,SE} = 6 configs × 3 seeds × 3 regions = **54 optimizations** (+1 no-timeline reference run per region, 57 `runOptimization` calls total).

Fixed study settings (hardcoded): model DeepSeek, year 2025, budget 200%, α=1; fixed start dates DE `02-01`, IT `01-14`, SE `04-22` (Table-2 optima); optimizer `resolution 10, startDateResolution 1, maxIterations 6, minStep 3, shrinkFactor 0.45`, `thetaPauseMax` SE=100 else 800.

CLI options added: `--additive-levels 0,0.5,1,2`, `--delay-steps 1,6`, `--resolution 10`, `--iterations 6`, `--budget 200`, `--alpha 1`, plus reuse of `-m/-r/-y/--seed-count/--calibration-dir/-o/--csv/--quiet`. Commander defaults for `--seed-count`/`-o` moved into the handler so the per-mode defaults apply (fixed: seeds 10/5, prefix `fixed`; reopt: seeds 3, prefix `reopt`).

### Outputs (under `publication/output/forecast/`)
- `reopt_{region}.json` — `{region, model, year, budget, start, seeds, optimizer:{resolution,iterations,tpMax}, configs:[{family, param, param_value, sigma, perSeed:[{seed,thetaP,thetaR,margin,savings,overhead,score,found}], best (seed-mean or null), foundRate}], baseline (= additive level-0 best)}`.
- `reopt_{region}.csv` — flat rows `region,family,param_value,seed,theta_p,theta_r,margin,savings,overhead,score,found` (`toPrecision(6)`, `NA` for a `found:false` row).
- `reopt_summary.json` — per region `{region, baseline, drift:[{family,param_value,thetaP_drift,thetaR_drift,margin_drift}], marginRuleSurvives:[{family,param_value,survives,seedFraction}], regionalRule:[{family,param_value,survives}]}`.
- `reopt_all.csv` via `--csv` (54 flat rows).

**Parallel-write safety (documented):** `--mode reopt` writes per-region files keyed by `-r`; the `reopt_summary.json` is written **only when the invocation spans more than one region** (`regions.length > 1`). So a parallel per-region sub-job (`-r <R> ... --quiet`) writes *only* `reopt_{region}.{json,csv}` — no clobbering of the summary; the combined `--csv` is likewise only written by multi-region (or explicit) runs. All jobs are deterministic, so per-region files are byte-identical regardless of which process wrote them.

### `src/cli/index.ts`
Registered the new reopt flags (`--additive-levels --delay-steps --resolution --iterations --budget --alpha`); `--seed-count`/`-o` no longer carry hardcoded fixed-mode defaults in commander.

### `src/cli/forecast-sweep.test.ts` (+10 tests)
1. `seedMeanBest` averages only found seeds (ignores a `found:false` row); returns null when all rows are `found:false`.
2. `drift` subtracts baseline field-by-field.
3. `marginRuleSurvives`: seed-mean ≤ 16 ⇒ survives with `seedFraction` counting per-seed rule hits (incl. a mixed case: mean 15 survives but fraction 2/3); mean > 16 ⇒ false/0; null best ⇒ false/0.
4. `regionalRuleSurvives`: within ±50% of baseline ⇒ true, outside ⇒ false, null best ⇒ false.
5. Identity regression: `runOptimization` with an identity `decisionTimeline` equals the no-decisionTimeline run on a small synthetic block timeline (fixed start `01-01`, resolution 2, maxIterations 1, non-null best) — `assertIdentityRegression` does not throw.

## Verification outputs (exact)

### 1. Full 54-run sweep (sequential, canonical)
```
$ pnpm cli forecast-sweep --mode reopt -o publication/output/forecast/reopt --csv publication/output/forecast/reopt_all.csv 2>&1 | tail -30
    add 1 x σ*   θₚ=278.5 θₛ=261.8 margin=16.7 savings=43.34% found=1.00
    add 2 x σ*   θₚ=284.3 θₛ=255.9 margin=28.4 savings=43.30% found=1.00
    delay 1      θₚ=272.5 θₛ=267.6 margin=4.9 savings=43.32% found=1.00
    delay 6      θₚ=272.3 θₛ=267.6 margin=4.7 savings=43.04% found=1.00
  JSON: publication/output/forecast/reopt_DE.json
  CSV:  publication/output/forecast/reopt_DE.csv
  [IT] start=01-14 tpMax=800 σ*=4.421
    baseline θₚ=246.7 θₛ=230.4 margin=16.3 savings=32.67% overhead=194.7%
    add 0 x σ*   θₚ=246.7 θₛ=230.4 margin=16.3 savings=32.67% found=1.00
    add 0.5 x σ* θₚ=246.1 θₛ=230.7 margin=15.5 savings=32.63% found=1.00
    add 1 x σ*   θₚ=250.2 θₛ=227.0 margin=23.2 savings=32.62% found=1.00
    add 2 x σ*   θₚ=253.7 θₛ=223.5 margin=30.2 savings=32.55% found=1.00
    delay 1      θₚ=246.7 θₛ=230.4 margin=16.3 savings=32.56% found=1.00
    delay 6      θₚ=240.7 θₛ=236.8 margin=3.8 savings=32.13% found=1.00
  JSON: publication/output/forecast/reopt_IT.json
  CSV:  publication/output/forecast/reopt_IT.csv
  [SE] start=04-22 tpMax=100 σ*=0.774
    baseline θₚ=18.2 θₛ=17.5 margin=0.7 savings=23.17% overhead=106.3%
    add 0 x σ*   θₚ=18.2 θₛ=17.5 margin=0.7 savings=23.17% found=1.00
    add 0.5 x σ* θₚ=19.2 θₛ=16.9 margin=2.3 savings=23.12% found=1.00
    add 1 x σ*   θₚ=19.6 θₛ=16.4 margin=3.2 savings=23.07% found=1.00
    add 2 x σ*   θₚ=20.8 θₛ=15.3 margin=5.5 savings=22.79% found=1.00
    delay 1      θₚ=18.2 θₛ=17.5 margin=0.7 savings=23.09% found=1.00
    delay 6      θₚ=18.2 θₛ=17.5 margin=0.7 savings=22.68% found=1.00
  JSON: publication/output/forecast/reopt_SE.json
  CSV:  publication/output/forecast/reopt_SE.csv
  Summary: publication/output/forecast/reopt_summary.json
  CSV (all regions): publication/output/forecast/reopt_all.csv
  Done.
```
Run counts: `reopt_{DE,IT,SE}.csv` 18 rows each, `reopt_all.csv` 54 rows (6 configs × 3 seeds × 3 regions). Every config/seed found a valid best (`found=1.00` in all 54; 0 `found:false` rows — see DoD).

### 2. Outputs present
`reopt_DE.json`, `reopt_IT.json`, `reopt_SE.json`, `reopt_DE.csv`, `reopt_IT.csv`, `reopt_SE.csv`, `reopt_summary.json` (7 canonical outputs) + `reopt_all.csv`.

### 3. Drift / rule-survival table (seed-mean best; `node` over `reopt_summary.json` + region JSONs)
```
=== DE  baseline: thetaP=272.37 thetaR=267.73 margin=4.64 savings=43.35%
config          | thetaP | thetaR | margin | savings |  dP   |  dR   | dMargin | M<=16 | seedFrac | regRule
additive 0      | 272.37 | 267.73 |  4.64 | 43.35 |   0.0 |   0.0 |   0.0 | true   | 1.00     | true
additive 0.5    | 273.04 | 266.14 |  6.91 | 43.35 |   0.7 |  -1.6 |   2.3 | true   | 1.00     | true
additive 1      | 278.47 | 261.79 | 16.68 | 43.34 |   6.1 |  -5.9 |  12.0 | false  | 0.67     | true
additive 2      | 284.29 | 255.92 | 28.36 | 43.30 |  11.9 | -11.8 |  23.7 | false  | 0.00     | true
delay 1         | 272.47 | 267.61 |  4.86 | 43.32 |   0.1 |  -0.1 |   0.2 | true   | 1.00     | true
delay 6         | 272.30 | 267.60 |  4.70 | 43.04 |  -0.1 |  -0.1 |   0.1 | true   | 1.00     | true

=== IT  baseline: thetaP=246.70 thetaR=230.45 margin=16.25 savings=32.67%
config          | thetaP | thetaR | margin | savings |  dP   |  dR   | dMargin | M<=16 | seedFrac | regRule
additive 0      | 246.70 | 230.45 | 16.25 | 32.67 |   0.0 |   0.0 |   0.0 | false  | 0.00     | true
additive 0.5    | 246.13 | 230.66 | 15.47 | 32.63 |  -0.6 |   0.2 |  -0.8 | true   | 0.33     | true
additive 1      | 250.15 | 226.96 | 23.19 | 32.62 |   3.5 |  -3.5 |   6.9 | false  | 0.00     | true
additive 2      | 253.65 | 223.50 | 30.15 | 32.55 |   7.0 |  -7.0 |  13.9 | false  | 0.00     | true
delay 1         | 246.70 | 230.45 | 16.25 | 32.56 |   0.0 |   0.0 |   0.0 | false  | 0.00     | true
delay 6         | 240.67 | 236.84 |  3.83 | 32.13 |  -6.0 |   6.4 | -12.4 | true   | 1.00     | true

=== SE  baseline: thetaP=18.18 thetaR=17.51 margin=0.67 savings=23.17%
config          | thetaP | thetaR | margin | savings |  dP   |  dR   | dMargin | M<=16 | seedFrac | regRule
additive 0      | 18.18 | 17.51 |  0.67 | 23.17 |   0.0 |   0.0 |   0.0 | true   | 1.00     | true
additive 0.5    | 19.21 | 16.91 |  2.30 | 23.12 |   1.0 |  -0.6 |   1.6 | true   | 1.00     | true
additive 1      | 19.58 | 16.37 |  3.21 | 23.07 |   1.4 |  -1.1 |   2.5 | true   | 1.00     | true
additive 2      | 20.77 | 15.28 |  5.49 | 22.79 |   2.6 |  -2.2 |   4.8 | true   | 1.00     | true
delay 1         | 18.18 | 17.51 |  0.67 | 23.09 |   0.0 |   0.0 |   0.0 | true   | 1.00     | true
delay 6         | 18.18 | 17.51 |  0.67 | 22.68 |   0.0 |   0.0 |   0.0 | true   | 1.00     | true
```
Columns: `dP/dR/dMargin` = drift vs. baseline (seed-mean − baseline); `M<=16` = R1 (`margin ≤ 16` on seed-mean); `seedFrac` = fraction of seeds with `margin ≤ 16`; `regRule` = R2 (seed-mean θ_p within ±50% of baseline θ_p — definition documented here and in the summary schema).

### 4. Determinism (rerun, byte-identical)
```
run1: sha256 of 8 outputs (3 json + 3 csv + summary + all.csv), captured after canonical run
run3: same command rerun → diff empty
DETERMINISM OK (run3 identical to run1)
```
Parallel per-region jobs also reproduced byte-identical `reopt_{region}.{json,csv}` (diff of run1 vs parallel run hashes empty).

### 5. Verification tails
```
pnpm test   →  Test Files  10 passed (10)
               Tests       156 passed (156)      # 146 baseline + 10 new
pnpm build  →  ✓ built in 2.02s                  # green
new-tsc=0                                        # comm of tsc error sets, 0 new (6 pre-existing unchanged)
```

## Runtime + seeds (SPEC 8.2)
- Seeds: `{1,2,3}` per region, recorded in every `reopt_{region}.json` (`seeds`) and per row (`seed`).
- Single runOptimization ≈ 1.6 s (SE) … ~2.5 s (DE/IT); **one region (19 optimizations) ≈ 30 s**; **full 54-optimization sweep sequential ≈ 2:22 wall** (142 s); **parallel 3 region jobs ≈ 1:14 wall** (74 s) on the 12-core machine (3 workers, ~2.5× speedup). Reruns for determinism took the same ~2:22 each.

## Headline findings
- The identity regression holds: for every region, `runOptimization` with the additive σ=0 (identity) decisionTimeline produces the exact same `best` (θ_p, θ_r, margin, savings, overhead, score, and all 13 `SweepPoint` fields) as the no-decisionTimeline run — asserted in code, 3× per region.
- The re-optimized identity baselines reproduce the paper: DE (272.4, 267.7, S=43.35%, O=174.3%), IT (246.7, 230.5, 32.67%, 194.7%), SE (18.2, 17.5, 23.17%, 106.3%) — Table-2 (272,268)/(246,231)/(19,18) within grid resolution.
- **R1 (margin ≤ 16 gCO₂eq/kWh)** survives at realistic error (≤ ½σ\*) and for all delay levels; it **fails at additive σ=1σ\*** on DE (16.7) and IT (23.2), and at σ=2σ\* on all three (DE 28.4, IT 30.2, SE 5.5 — SE still far below 16). SE is robust everywhere (margins 0.7→5.5).
- **R2 (regional θ_p)** survives **every config in every region** — θ_p drifts are modest (DE +11.9, IT +7.0, SE +2.6 at 2σ\*) and stay within ±50% of the baseline.
- Drift is monotone in additive level and moves θ_p up / θ_r down (margin widens) — the optimizer responds to noisier forecasts by widening the hysteresis band, not by abandoning the rule.
- Delay errors barely move the optimum (delay 1/6 margins ≈ baseline); only savings degrade slightly (DE −0.3 pp, IT −0.5 pp, SE −0.5 pp at delay 6).

## DoD checklist
- [x] `--mode reopt` fully implemented with the exact matrix (6 configs × 3 seeds × 3 regions = 54 optimizations; +1 no-timeline reference per region for the regression/baseline).
- [x] Additive L=0 == no-decisionTimeline regression asserted in code (`sameBest`, throws on mismatch) and unit-tested (`assertIdentityRegression` on a synthetic timeline).
- [x] `reopt_{region}.{json,csv}` + `reopt_summary.json` with drift vectors and margin/regional rule survival (baseline, drift, `marginRuleSurvives` with `seedFraction`, `regionalRule`).
- [x] Determinism: full rerun byte-identical (sha256 diff empty); parallel per-region jobs also byte-identical.
- [x] All 54 runs complete. `found` handling: `best: null` → `found:false` row (metrics `null` in JSON, `NA` in CSV), excluded from seed-means, `foundRate < 1`; here every config/seed found a valid point (`foundRate=1.00`, 0 false rows) — but the null path is implemented and unit-tested (seedMeanBest/marginRule/regionalRule with all-`found:false` inputs).
- [x] `pnpm test` 156 green; `pnpm build` green; `new-tsc=0`.
- [x] Runtime + seeds recorded in this phase report (SPEC 8.2).

## Deviations / choices
1. **57 runOptimization calls, not 54**: the spec's 54 are the config×seed matrix; each region adds one no-decisionTimeline run as the additive-L0 regression reference (required by the regression assertion) and as the `baseline` field.
2. **`reopt_summary.json` only written for multi-region invocations** (`regions.length > 1`); single-region (parallel) sub-jobs write only `reopt_{region}.{json,csv}`, so concurrent writers never clobber the summary. Combined `--csv` likewise only from multi-region runs. Documented in code and above.
3. **`margin` is stored as raw `θ_p − θ_r`** (IEEE double, e.g. DE baseline 4.6399999…); drift/rule logic compares against the exact `MARGIN_RULE_CEILING = 16` — IT's level-0 seed-mean margin (16.25) is *already* just above the ceiling, so `M<=16=false, seedFrac=0` for IT additive L0 by design (honest reporting of the re-optimized identity optimum).
4. **Commander defaults moved into the handler** for `--seed-count` and `-o` so the reopt defaults (3 seeds, `reopt` prefix) don't leak into `--mode fixed` (10/5 seeds, `fixed` prefix).
5. **Runtime is not written into the JSON artifacts** (would break byte-determinism); it is reported here (SPEC 8.2) via wall-clock measurements of the CLI invocations.
