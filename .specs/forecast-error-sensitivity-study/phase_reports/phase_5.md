# Phase 5 — Forecast-error sensitivity figures (SVG + EPS)

Status: `[x]` implemented · 167 tests green (156 baseline + 11 new) · build green · 0 new tsc errors · all 8 artifacts produced.

## What was implemented

### `src/cli/plot-forecast.ts` — `pnpm cli plot-forecast`

New CLI that mirrors `src/cli/plot.ts`'s renderer pattern (`compile` → `View(parse(vgSpec), {renderer:"none"})` → `view.toSVG()`), then converts each SVG → EPS via `rsvg-convert -f eps -o <out>.eps <in>.svg` (`execFileSync`, prefers `/usr/sbin/rsvg-convert`, falls back to `rsvg-convert` on PATH, fails loudly otherwise).

Flags: `--data-dir` (default `publication/output/forecast`), `--out-dir` (default `publication/ICREC_Rome/assets`), `--regions DE,IT,SE`, `--only f1,f2,f3,f4`. Missing required input → `Missing required input: <path>` + non-zero exit (verified: exit 1). Unknown figure selector / empty region list also fail loudly.

Pure, exported, unit-testable data-prep + spec builders (no `any`, no `Math.random`):
- `aggregateBand(rows)` — drops raw seed rows; groups the `additive`/`multiplicative` families by level; per metric (savings/overhead/score) emits `{region, family, level, metric, value, lo, hi}` with `value=mean`, `lo/hi=mean±std` (sample std, same `meanStd` helper as the Phase-3/4 summaries). The `fixed_*.json` summary has no overhead/score std, so all three bands are re-aggregated from the raw seed rows in TS.
- `buildDegradation(summaries, region)` — `degradation_frac_mean` per level for both families, **excluding level 0** (zero-degradation baseline cannot sit on a log axis).
- `buildDrift(result)` — per region: baseline point, one point + one segment per `config.best` (already the seed-mean from Phase 4), skipping `best:null` / non-finite; labels `additive L×σ*` / `delay d`; shared x/y domain computed from all points so θₚ=θᵣ stays a 1:1 diagonal.
- `buildRmse(bundle)` — `{region, horizon, model: persistence|AR(1)|AR(7), rmse}`; persistence order-1 rows deduped against the order-7 duplicates.
- `buildF1Spec` / `buildF2Spec` / `buildF3Spec` / `buildF4Spec` — exported pure spec builders.

### Figure layout decisions (with rationale)

- **f1** `forecast_savings_overhead_score`: single layered **facet spec**, `facet.column = region`, `facet.row = metric` → 3×3 grid; area mark = ±1 std band (opacity 0.22), line+point = mean, color by family, `resolve.scale.y = "independent"` so the 9 panels each auto-scale (overhead ≈174, savings ≈43, score ≈0.7 have very different ranges). X = level, linear.
- **f2** `forecast_degradation`: facet column = region; x log scale with explicit `domain:[0.2,5]`; gray dashed **rule at y=0.10** + right-aligned text label "10% grace"; level 0 dropped from the data and documented in the subtitle ("level 0 (baseline, zero degradation) omitted for log x-axis"). Y independent per region.
- **f3** `forecast_reopt_drift`: **`hconcat` of one panel per region** (not a shared facet) because each region needs its own explicit, **equal x/y domain** to keep the θₚ=θᵣ diagonal implicit — a shared facet can't do per-panel equal domains. Segments drawn as `rule` marks from baseline to each seed-mean optimum (vega-lite has no native arrowheads; the line + a note in the subtitle is the documented alternative). Baseline = 5-point star via a custom path (vega has no built-in `star` symbol — verified at runtime, `shape:"star"` throws `Invalid SVG path`). Config points + segments colored by label with one shared legend.
- **f4** `forecast_rmse_horizon`: facet column = region; x = horizon on a log scale, y = RMSE; one line per model (persistence / AR(1) / AR(7)), color-coded.

### `src/cli/index.ts`
Registered `plot-forecast` with the four options and dynamic import of `./plot-forecast`.

### `src/cli/plot-forecast.test.ts` (+11 tests)
`aggregateBand` (band-only families, ±1 std correctness, sort), `buildDegradation` (level-0 dropped, band families only, region tag), `configLabel`/`modelLabel`, `buildDrift` (segment+point per found config, null best skipped, baseline + domain), `buildRmse` (model mapping + persistence dedupe across orders), and one structural test per spec builder (facet/hconcat/layer presence).

## Verification (pasted)

1. CLI run:
```
$ pnpm cli plot-forecast 2>&1 | tail -20
  /home/simon/dev/TheGreenEpoch/publication/ICREC_Rome/assets/forecast_savings_overhead_score.svg (134345 bytes)
  /home/simon/dev/TheGreenEpoch/publication/ICREC_Rome/assets/forecast_savings_overhead_score.eps
  /home/simon/dev/TheGreenEpoch/publication/ICREC_Rome/assets/forecast_degradation.svg (51763 bytes)
  /home/simon/dev/TheGreenEpoch/publication/ICREC_Rome/assets/forecast_degradation.eps
  /home/simon/dev/TheGreenEpoch/publication/ICREC_Rome/assets/forecast_reopt_drift.svg (44341 bytes)
  /home/simon/dev/TheGreenEpoch/publication/ICREC_Rome/assets/forecast_reopt_drift.eps
  /home/simon/dev/TheGreenEpoch/publication/ICREC_Rome/assets/forecast_rmse_horizon.svg (60129 bytes)
  /home/simon/dev/TheGreenEpoch/publication/ICREC_Rome/assets/forecast_rmse_horizon.eps
  Done (4 figure(s) in /home/simon/dev/TheGreenEpoch/publication/ICREC_Rome/assets)
```

2. Artifacts:
```
forecast_degradation.eps  532909  forecast_savings_overhead_score.eps  837333
forecast_degradation.svg   51979  forecast_savings_overhead_score.svg  134651
forecast_reopt_drift.eps  293791  forecast_rmse_horizon.eps           202942
forecast_reopt_drift.svg   44561  forecast_rmse_horizon.svg            60259
```

3. SVG heads start `<svg xmlns="http://www.w3.org/...`; EPS heads start `%!PS-Adobe-3.0 EPSF-`. SVG internals verified: f1 has metric/region facet headers (savings/overhead/score × DE/IT/SE) + family legend; f2 has "10% grace" text + both families; f3 has 3 star paths (`M0,-9.22...`) + 6 config labels; f4 has persistence/AR(1)/AR(7) legend.

4. `pnpm test 2>&1 | tail -6` → `Test Files 11 passed (11) · Tests 167 passed (167)`.
   `pnpm build 2>&1 | tail -2` → `✓ built in 2.05s` (exit 0).
   `npx tsc --noEmit` → exactly the 6 baseline errors; `echo new-tsc=...` → **0**.

5. `--only f3` renders only `forecast_reopt_drift.{svg,eps}` (exit 0); missing input (`--data-dir` pointing nowhere, `--only f4`) prints `Missing required input: ...` and exits 1.

## DoD checklist

- [x] `plot-forecast` CLI registered and runs with defaults.
- [x] All four figures exist as SVG + EPS under `publication/ICREC_Rome/assets/` with the exact names `forecast_savings_overhead_score`, `forecast_degradation`, `forecast_reopt_drift`, `forecast_rmse_horizon`.
- [x] f1: seed band (±1 std) rendered; additive+multiplicative; per-region.
- [x] f2: log-x, 10% grace rule+label, per-region, level-0 excluded (documented in subtitle + this report).
- [x] f3: baseline star + config points + lines from baseline; per-region (hconcat).
- [x] f4: persistence + AR(1) + AR(7) lines, log-x horizon, per-region.
- [x] SVG files non-empty and start with `<svg`; EPS files non-empty and start with `%!PS`.
- [x] `pnpm test` green (167), `pnpm build` green, `npx tsc --noEmit` shows no NEW errors beyond the 6 baseline (new-tsc=0).
- [x] Phase report written.

## Deviations / notes

- f1 uses raw seed rows for all three bands (the summary lacks overhead/score std) — consistent with the "aggregate in TS, drop raw rows" instruction.
- f3 is `hconcat` (not facet) so each region gets an explicit equal x/y domain preserving the θₚ=θᵣ diagonal.
- Arrows not natively supported by vega-lite; segments are `rule` marks from baseline to optimum, noted in the subtitle.
- Star baseline uses a custom SVG path because vega's `shape:"star"` is invalid.
