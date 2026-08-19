# SPEC — Forecast-Error Sensitivity Study

## Goal

Today the simulator is a cheater: the hysteresis policy decides "pause"/"resume"
on the *realized* 5-min CO₂ intensity (`src/domain/simulation.ts:137`). A real
deployment must decide on a *forecast*. This study quantifies how much the
outcomes degrade when the controller acts on an imperfect forecast:

1. **Deployment degradation** — evaluate the already-published optimal
   `(θ_p, θ_r)` policies under forecast error: loss of CO₂ savings, overhead
   drift, feasibility (completion / within-budget).
2. **Design-rule drift** — re-optimize thresholds under each error level: does
   the near-zero-hysteresis rule and the regional threshold guidance survive
   realistic error?

Core mechanic: **decide on the forecast, pay the bill on the realized value.**
Everything seeded and deterministic → fully reproducible.

### Framing of "forecast"

Decision value at step `t` = prediction of `y(t)` using observations up to
`t − h·Δt` (data cutoff `h` steps back, Δt = 5 min):

- `h = 0` → perfect foresight (today's behavior, must remain bit-identical).
- persistence → `y(t − h)` ("the next h steps look like the current value").
- AR(p) → linear recursion on history ending at `t − h`.
- additive/multiplicative noise → `y(t)` perturbed at the decision instant
  (measurement-type error), σ levels anchored to the AR innovation std.

Exploits the paper's own lag-1 autocorrelation 0.999 story: short-horizon
persistence is nearly perfect; the study quantifies *how* nearly.

Status legend: `[ ]` open · `[x]` done · `[~]` blocked

## Phase 0 — Scope (decided)

- [x] Study design: fixed-policy degradation curves **and** per-error re-optimization.
- [x] Forecast tiers: parametric noise sweep **and** calibrated persistence/AR(p) anchor.
- [x] Coverage: DeepSeek V3 × {DE, IT, SE}; train 2022–2024, evaluate 2025.
- [x] Deliverable: results and figures in `publication/output/forecast/` +
      `publication/ICREC_Rome/assets/`. (Paper text was a phase-7 deliverable;
      phases 7–8 are discarded — see below.)
- [x] Determinism: seeded RNG (mulberry32), all seeds recorded in output artifacts.
- [ ] Assumption check (re-opt fixes start date to Table-2 optimum per region; revisit if runtime budget changes).

## Phase 1 — Core implementation

- [ ] 1.1 `src/domain/types.ts`: add `decisionTimeline?: CO2Timeline` to
      `SimConfig`; add `ForecastModel`, `CalibrationResult` types.
- [ ] 1.2 `src/domain/simulation.ts`: read `simConfig.decisionTimeline`
      (fallback = realized) in `simulateStepwise`; decisions AND the
      initial-state check (`:67`) use the decision value; energy/emissions/token
      accounting stays on realized `carbon[idx]`; validate both timelines have
      equal length (throw on mismatch).
- [ ] 1.3 `src/domain/forecast.ts`:
      - seeded PRNG (mulberry32) + Gaussian (Box–Muller) + log-normal samplers
      - `ForecastModel` union: `identity | additive{sigma} | multiplicative{sigma}
        | delay{steps} | arma{order,horizon}`
      - `applyForecast(realized: CO2Timeline, model, seed): CO2Timeline`
        — same timestamps, clamp values ≥ 0, head-fallback (t < horizon/steps →
        realized value), deterministic in seed
- [ ] 1.4 `src/domain/optimize.ts`: accept optional decision timeline and thread
      through the policy-simulation path only (baseline uses `neverPause`, no
      decision dependence → unchanged).
- [ ] 1.5 `src/domain/index.ts`: export the new forecast module.

## Phase 2 — Calibration (Tier-2 anchor)

- [ ] 2.1 `src/cli/forecast-calibrate.ts` + `pnpm cli forecast-calibrate`:
      - fit AR(1) and AR(7) per region on 2022–2024 (OLS/Yule–Walker with
        intercept; deterministic)
      - h-step-ahead evaluation on 2025 vs. persistence: RMSE, MAE, MAPE
      - report innovation std (σ\*), lag-1/lag-2 autocorrelation, AR coefficients
      - flags: `--regions DE,IT,SE --train 2022,2023,2024 --test 2025
        --orders 1,7 --horizons 1,3,6,12,24,72`
      - outputs: `publication/output/forecast/calibration_{region}.json` + `.csv`
- [ ] 2.2 Unit sanity: fit recovers known φ on a synthetic AR(1) series.
- [ ] 2.3 Record the chosen sweep levels (σ values, persistence-vs-AR gaps) and
      justify them from the calibration table (store in calibration README).

## Phase 3 — Fixed-policy sensitivity sweep

- [ ] 3.1 `src/cli/forecast-sweep.ts` + `pnpm cli forecast-sweep --mode fixed`:
      - args: `-m Deepseek -r DE/IT/SE -y 2025 --theta-p --theta-r --start`
        (Table-2 optima), `--error-types`, `--levels`, `--horizons`, `--seeds`,
        `-o json`, `--csv`
      - per (model, error, level, seed): perfect-foresight control run +
        forecast run on the SAME policy
      - row fields: `theta_p, theta_r, start, savings, overhead, score,
        numPauses, completed, within_budget, savings_perfect, degradation_frac`
      - summary: S₀, ΔS (pp), ΔS/S₀, grace level (max σ with ΔS/S₀ ≤ 10%),
        mean ± std across seeds
- [ ] 3.2 Run matrix: families `{additive, multiplicative, delay, arma(1,h),
      persistence(h)}` × levels `{0, ¼, ½, 1, 2, 4 × σ*}` ∪ horizons
      `{1, 3, 6, 12, 24, 72}` × seeds `{5; 10 for DE}`.
      → `publication/output/forecast/fixed_*.{json,csv}`

## Phase 4 — Re-optimization (design-rule drift)

- [ ] 4.1 `forecast-sweep --mode reopt`: run `runOptimization` under the
      forecast timeline; fixed start date (Table-2 optima per region),
      resolution 10, iterations 6, budget 200, tp-max per region (SE 100,
      else 800).
      - configs: additive σ `{0, ½, 1, 2 × σ*}` + delay `{1, 6}` × 3 seeds ×
        3 regions
      - output: best `(θ_p, θ_r, margin, savings, overhead, score)` per config
- [ ] 4.2 Record drift vectors and whether the ≤ 16 gCO₂eq/kWh margin rule
      survives each error level.

## Phase 5 — Plotting

- [ ] 5.1 `src/cli/plot-forecast.ts` + `pnpm cli plot-forecast` (vega-lite,
      mirror `src/cli/plot.ts`); SVG + EPS into
      `publication/ICREC_Rome/assets/forecast_*.{svg,eps}`:
      - f1: savings, overhead, score vs. error level (per region, ± seed band)
      - f2: ΔS/S₀ relative degradation vs. level with 10% grace band (log-x)
      - f3 (reopt): (θ_p, θ_r) drift in threshold plane, arrows from optimum
      - f4: RMSE vs. horizon (persistence vs. AR(1)/AR(7)) per region

## Phase 6 — Reproducibility & cleanliness

- [ ] 6.1 Vitest suite (match existing style in `src/domain/optimize.test.ts`,
      `src/data/simulation.test.ts`):
      - forecast determinism: same seed → identical timeline
      - identity model ≡ realized → `simulateStepwise` outputs bit-identical
      - decision/accounting split: contrived case where decision uses forecast
        but emissions use realized (assert both sides)
      - clamp ≥ 0; head-fallback for delay/AR; length mismatch throws
      - AR fit recovery on synthetic series; RNG distribution sanity (mean ≈ 0)
      - `runOptimization` with identity forecast == without (regression)
- [ ] 6.2 `publication/run_forecast_experiments.sh`: calibration → sweep →
      reopt → plots; `set -euo pipefail`; logs appended to
      `experiments_run.log` (mirror `run_experiments.sh`).
- [ ] 6.3 All artifacts under `publication/output/forecast/`; header comment with
      commands, seeds, and runtime; `.gitignore` large intermediate CSVs if needed.
- [ ] 6.4 `pnpm test` and `pnpm build` green; `tsc` clean.

## Phase 7 — Paper (`publication/ICREC_Rome/main.tex`) — DISCARDED

Superseded by `.specs/reframe-paper-stale-aware/SPEC.md`: the paper is rewritten
there fresh from the committed artifacts; the ICREC paper is archived and
untouched. No paper-editing work is tracked in this spec. Historical note: the
forecast section, calibration table, and claim–evidence mapping were originally
written into `publication/ICREC_Rome/main.tex` (see `phase_reports/phase_7.md`).

## Phase 8 — ACM prep hooks — DISCARDED

No submission-target or venue-specific prep is tracked here; the ACM-specific
figure-conversion and reproducibility-notes tasks are void. Historical note:
originally a non-blocking note list (see `phase_reports/phase_8.md`).

## Reference facts

- Perfect-foresight optima (Table 2, `main.tex`): DE θ=(272,268) S=43.4%
  O=174.3%; IT θ=(246,231) S=32.7% O=194.7%; SE θ=(19,18) S=23.2% O=106.3%.
- Grid stats 2025: DE mean 380 σ=142 CV=0.37; IT 309/87/0.28; SE 23/8/0.35;
  lag-1 autocorrelation 0.999.
- Data: `public/data/co2/{zone}_{year}.json`, 5-min, 105 120 pts/yr.
- CLI entry: `src/cli/index.ts` (commander); test runner: vitest; plots:
  vega-lite.
