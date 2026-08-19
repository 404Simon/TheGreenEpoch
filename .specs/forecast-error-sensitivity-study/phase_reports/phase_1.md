# Phase 1 — Forecast-Error Sensitivity Study: Decision Timeline Core

Status: `[x]` implemented · tests green · build green · no new tsc errors.

## What was implemented

### `src/domain/types.ts`
- Added `decisionTimeline?: CO2Timeline;` to `SimConfig`.
- Added `ForecastModel` union (`identity | additive | multiplicative | delay | arma`), exactly as specified.
- Added `ArCoeffs` and `CalibrationResult` (forward-compatible for Phase 2).

### `src/domain/forecast.ts` (NEW)
- `mulberry32(seed)` — deterministic 32-bit PRNG, floats in `[0,1)`.
- `gaussian(rand)` — Box–Muller, exactly two uniforms per call (`u1 = 1 - rand()`, `u2 = rand()`; never hits `log(0)`).
- `sampleNormal(rand, mean, sigma)` and `sampleLognormal(rand, mu, sigma)` helpers.
- `fitAr(series, order)` — Yule–Walker (normal equations on the Toeplitz autocovariance matrix) solved by Gaussian elimination with partial pivoting; `intercept = mean·(1 − Σφ)`. Deterministic, no RNG. Throws on: non-positive/non-integer order, series not longer than `order`, non-finite values, and singular (zero-variance) autocovariance matrix.
- `applyForecast(realized, model, seed)` — returns a new `CO2Timeline` (copied `zone/years/timestamps`, new `carbonIntensity` array):
  - `identity`: exact copy, no arithmetic.
  - `additive`/`multiplicative`: one `mulberry32(seed)` stream, one `gaussian` per index → same seed ⇒ same `z` sequence regardless of `sigma` (pure scale factor). Values clamped `max(0, …)`.
  - `delay{steps}`: `realized[t-steps]` for `t ≥ steps`, else `realized[t]`.
  - `arma{order,horizon,coeffs?}`: `intercept + Σ ar[i-1]·realized[t-horizon-(i-1)]` for `t ≥ horizon+order-1`, else `realized[t]`; `coeffs` optional (falls back to `fitAr(carbon, order)`). Clamped `max(0, …)`.
- `forecastInnovationStd(series)` — residual std (σ\*) of a fitted AR(1): `sqrt(Σr²/(n-2))`.

### `src/domain/simulation.ts`
- Reads `simConfig.decisionTimeline`; validates `timestamps` and `carbonIntensity` lengths against the realized timeline and throws `Error` on mismatch.
- Added `getDecisionCo2(i)` mirroring `getCo2(i)` (same NaN→mean fallback, but over the decision timeline's values/mean).
- Initial-state check (`policy.evaluate(getCo2(startIdx), false)`) and per-step decision (`policy.evaluate(co2, …)`) now use the DECISION value.
- All energy/emissions/token/time accounting stays on realized `getCo2(idx)`; `maybeYield().carbonIntensity` stays realized.
- Baseline (no `decisionTimeline`) is bit-identical: decision paths short-circuit to `getCo2`/`co2` so the shared `nanFallbacks` counter is never double-incremented (verified by test).

### `src/domain/optimize.ts`
- `runOptimization(profile, timeline, historicalYears, options, onIteration?, decisionTimeline?)` — optional `decisionTimeline` threaded into `dateSimConfig`. Existing callers compile unchanged (parameter is optional).
- **Deviations from spec:** the spec "recommends" `decisionTimeline` after `options`, before `onIteration`. Placing it there would make the existing CLI call (`src/cli/optimize.ts:138`, which passes its `onIteration` callback as the 5th positional argument) a TS type error, violating the hard constraints "existing callers compile unchanged" and "no NEW tsc errors" (and the CLI must not be edited this phase). It was therefore placed **after** `onIteration`. This is the only deviation.

### `src/domain/index.ts`
- Added `export * from "./forecast";`.

## New tests

### `src/domain/forecast.test.ts` (21 tests)
- mulberry32: same-seed determinism, cross-seed difference, range `[0,1)`.
- gaussian: mean≈0 / std≈1 over 10000 draws (|mean|<0.05, std∈[0.9,1.1]); `sampleNormal` mean shift; `sampleLognormal` positivity.
- fitAr: AR(1) recovery (φ=0.9, μ=50, N=2000 → |φ̂−φ|<0.05, |intercept−5|<5); `forecastInnovationStd` recovers innovation scale ≈1; throws on too-short series, constant series (singular), non-finite values, invalid order.
- applyForecast: identity bit-identical; same-seed determinism; cross-seed difference; same `z` sequence across sigma levels (compared against clamped expected values); additive/multiplicative clamp ≥0 (additive also asserts a clamp actually occurred); delay and arma head-fallback; arma without coeffs fits AR on the series.

### `src/domain/simulation-forecast.test.ts` (6 tests)
- Identity `decisionTimeline` ⇒ bit-identical `SimProgress` sequences vs no `decisionTimeline` (deep `toEqual` per step, covering every field incl. `totalEmissionsG`, `numPauses`, `state`).
- Decision/accounting split: realized all-low (100) + decision high-then-low (600,600,30,30,30,30) with `hysteresisPolicy(300,50)` ⇒ pauses occur (decision drives it) while `totalEmissionsG/totalEnergyWh` ≈ 0.1 (accounting at realized intensity 100) and `carbonIntensity` field stays realized.
- Length-mismatch throws for both `carbonIntensity` and `timestamps`.
- `runOptimization` with identity forecast == without (200-point synthetic timeline): identical `points` and `best`.

All 9 spec test groups covered (1 determinism, 2 identity≡realized, 3 decision/accounting split, 4 clamp≥0, 5 head-fallback, 6 length-mismatch throws, 7 AR fit recovery, 8 RNG sanity, 9 runOptimization identity equivalence).

## Verification outputs (exact)

`pnpm test 2>&1 | tail -15`:
```
$ vitest

 RUN  v4.1.10 /home/simon/dev/TheGreenEpoch


 Test Files  8 passed (8)
      Tests  118 passed (118)
   Start at  12:27:29
   Duration  9.10s (transform 2.12s, setup 1.95s, import 3.60s, tests 7.88s, environment 9.56s)
```

`pnpm build 2>&1 | tail -5` (relevant tail; full log ends `✓ built in 2.06s`):
```
[plugin builtin:vite-reporter]
(!) Some chunks are larger than 500 kB after minification. Consider:
- Using dynamic import() to code-split the application
- Use build.rolldownOptions.output.codeSplitting to improve chunking: https://rolldown.rs/reference/OutputOptions.codeSplitting
- Adjust chunk size limit for this warning via build.chunkSizeWarningLimit.
```
(green; the 500 kB chunk warning is pre-existing and unrelated)

`npx tsc --noEmit > /tmp/opencode/tsc_phase1.txt 2>&1; echo exit=$?; rg "error TS" /tmp/opencode/tsc_phase1.txt`:
```
exit=1
node_modules/.pnpm/@testing-library+jest-dom@7.0.1_@testing-library+dom@10.4.1_vitest@4.1.10_@types+node@2_f78b469abc20756ad4bbff29523ba667/node_modules/@testing-library/jest-dom/types/jest.d.ts(1,23): error TS2688: Cannot find type definition file for 'jest'.
node_modules/.pnpm/@testing-library+jest-dom@7.0.1_@testing-library+dom@10.4.1_vitest@4.1.10_@types+node@2_f78b469abc20756ad4bbff29523ba667/node_modules/@testing-library/jest-dom/types/jest.d.ts(9,27): error TS2304: Cannot find name 'expect'.
src/components/CO2Chart.tsx(100,25): error TS2322: Type 'Record<string, unknown>' is not assignable to type '_DeepPartialArray<AnnotationOptions<keyof AnnotationTypeRegistry>> | _DeepPartialObject<Record<string, AnnotationOptions<keyof AnnotationTypeRegistry>>> | undefined'.
src/pages/LiveSimPage.tsx(22,39): error TS2345: Argument of type 'string | string[]' is not assignable to parameter of type 'string'.
src/pages/LiveSimPage.tsx(23,35): error TS2345: Argument of type 'string | string[]' is not assignable to parameter of type 'string'.
vite.config.ts(14,3): error TS2769: No overload matches this call.
```
`diff` of the `error TS` lines against `/tmp/opencode/tsc_baseline.txt` → **IDENTICAL to baseline** (same 6 pre-existing errors, no new ones).

## Design decisions / deviations

1. **`runOptimization` parameter order** — `decisionTimeline` placed **after** `onIteration`, not before. Rationale: the existing CLI passes its callback positionally as arg 5; putting `decisionTimeline` at position 5 would break CLI compilation (function not assignable to `CO2Timeline | undefined`), which the spec forbids this phase. See §What was implemented.
2. **AR fit** via Yule–Walker normal equations (spec's first listed option) with Gaussian elimination + partial pivoting; degeneracy = zero-variance (singular matrix), length ≤ order, non-finite values, invalid order.
3. **Bit-identity preservation** — decision lookup is short-circuited (`decisionTimeline ? getDecisionCo2(…) : getCo2(…)`) so the shared `nanFallbacks` counter is never double-incremented in the baseline path; verified by a deep per-step equality test.
4. `forecastInnovationStd` uses AR(1) residual std with denominator `n-2` (documented choice; Phase 2 may refine).

## DoD checklist

- [x] `types.ts` has `decisionTimeline`, `ForecastModel`, `ArCoeffs`, `CalibrationResult`.
- [x] `forecast.ts` exports `mulberry32`, `gaussian`, `sampleNormal`, `sampleLognormal`, `fitAr`, `applyForecast`, `forecastInnovationStd`, with `ForecastModel` semantics exactly as specified (seeded z stream reused across sigma levels, clamp ≥ 0, head-fallback, arma coeffs optional).
- [x] `simulation.ts` uses decision value for decisions + initial-state check only; accounting stays realized; length-mismatch throws.
- [x] `optimize.ts` threads optional `decisionTimeline`; existing callers compile (one deviation in arg position, documented above).
- [x] `index.ts` exports the forecast module.
- [x] All 9 test groups pass (118 tests, +27 new).
- [x] `pnpm test` and `pnpm build` green; no NEW tsc errors (diff vs baseline identical).

Nothing was silently skipped. `forecastInnovationStd` is implemented and has a sanity test even though Phase 2 will use it.
