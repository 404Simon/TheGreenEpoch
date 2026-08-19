# SPEC — Stale-Aware Hysteresis Control for Carbon-Aware LLM Pretraining

Target: ACM e-Energy 2027 (Winter deadline ~late Jan 2027; e-Energy 2026 already passed). Working paper
lives in `publication/eenergy/` (acmart `sigconf`, ~10 pages). The ICREC workshop paper
(`publication/ICREC_Rome/`) is **archived and untouched** — nothing gets unstashed or ported;
the e-Energy paper is written fresh from the committed artifacts.

Status legend: `[ ]` open · `[x]` done · `[~]` blocked

## Phase A outcome — research is done, read it first

`[x]` A.1–A.6 Literature research (6 buckets + venue survey) — **completed**.
Deliverable: **`.specs/reframe-paper-stale-aware/phase_reports/research_report.md`** (read before any Phase B/C/D work).
It contains: the full e-Energy competitive field (LACS, UQ-Advice, DTPR/OPR, equilibrium analysis,
Moving-Beyond-MCI, curtailment-LLM, CarbonCast/EnsembleCI/CarbonX, …), ~65 existence-verified references
with working `.bib` keys, the gap analysis, and a recommended title/focus reframe.

Critical findings every subagent must respect:

1. **DTPR/OPR** (Lechowicz et al., POMACS '23) already proves optimal *double-threshold* online pause/resume
   with switching cost. We may NOT claim "first to use two thresholds". Our novelty = staleness-aware adaptive
   margin + LLM-pretraining realism (minutes-scale checkpoints) + 5-min design rules.
2. **UQ-Advice** (e-Energy '26) and **equilibrium analysis** (e-Energy '26) are the two most recent anchors a
   reviewer will check. We must cite and position against both. UQ-Advice assumes forecast quality is the
   bottleneck; our thesis is that staleness is. The equilibrium paper challenges average-CI-driven shifting
   entirely; we counter with "pay-on-realized" + demand-side-flexibility framing.
3. **The decisive open question (Phase B.0) decides the paper's direction**: does CO₂ savings survive
   realistic, minutes-scale checkpoint times for a 671B model? Run it first (it is cheap, deterministic,
   uses committed data). Outcome selects one of two honest papers:
   - **Story A "works-with-design-rules"**: savings survive → paper = decision-theoretic control + design
     rules + stale-aware adaptive controller.
   - **Story B "negative result"**: savings collapse → paper = "single-site 5-min temporal shifting of
     frontier LLM pretraining is over-rated; value is spatial/curtailment/coarser decisions".
4. "Hanford '16" is **unverifiable** (absent from DBLP/Crossref/S2/OpenAlex) — drop it; replaced by
   Dodge/Radovanovic/DTPR.

## Phase 0 — Housekeeping & skeleton

- [x] 0.1 `publication/eenergy/` skeleton created: `main.tex`, `references.bib`, `figures/`, `Makefile`,
      `README.md`, `run_eenergy_experiments.sh`.
- [x] 0.2 Baseline facts recorded in this SPEC's Reference facts (below) from committed artifacts.
- [x] 0.3 `publication/ICREC_Rome/` and the stash untouched.

## Phase B.0 — DECISIVE EXPERIMENT: checkpoint-realism sweep (do before anything else)

Why: the committed data uses checkpointPauseTime = 148.8 s (a 7B-model number) applied to a 671B model.
A 671B MoE state ≈ 1.34 TB (BF16) ⇒ real full-state checkpoint/restore is **minutes**, not 148.8 s. This
sweep quantifies the damage and selects Story A vs Story B above.

- [ ] **B.0.1 Parameterize checkpoint times (implement)**
      **Inputs:** `src/cli/optimize.ts`, `src/cli/forecast-sweep.ts`, `src/domain/optimize.ts`, `src/domain/types.ts`,
      `public/data/constants.json`, `public/data/profiles.json`.
      **Steps:**
      1. Add optional CLI overrides `--ckpt-pause <seconds>` and `--ckpt-resume <seconds>` to the `optimize`
         command in `src/cli/index.ts` and to `forecast-sweep` (both modes), defaulting to `constants.json`.
      2. In `src/cli/optimize.ts` and `src/cli/forecast-sweep.ts`, when the override is present, build
         `FullProfile` with `checkpointPauseTime`/`checkpointResumeTime` from the override instead of constants.
      3. Keep `src/domain/optimize.ts` and `simulation.ts` unchanged (they already read checkpoint times from
         `FullProfile`).
      **Artifacts:** modified `src/cli/index.ts`, `src/cli/optimize.ts`, `src/cli/forecast-sweep.ts`.
      **Accept:** `pnpm build` passes; `pnpm cli optimize --help` shows the two new options; running without the
      overrides reproduces current behavior bit-for-bit.
      **Verify:** `pnpm cli optimize -m Deepseek -r DE -y 2025 --start 02-01 --ckpt-pause 148.8 -o /tmp/x.json`
      matches the baseline DE result (θ_p≈272.37, θ_r≈267.73, S≈43.35 %, O≈174.3 %) from
      `publication/output/forecast/reopt_DE.json`.

- [ ] **B.0.2 Checkpoint-time sweep (implement + run)**
      **Inputs:** committed `public/data/co2/{DE,IT,SE}_2025.json`, `public/data/constants.json`,
      `public/data/profiles.json`, B.0.1 CLI.
      **Steps:** for regions DE, IT, SE (DeepSeek profile, 2025 data, optimizer settings identical to
      `reopt_summary.json`: resolution 10, iterations 6, budget 200 %, α=1, fixed start per region from
      `forecast-sweep.ts` `DEFAULT_POLICIES`): run the headline optimization for each
      `--ckpt-pause ∈ {148.8, 150, 900, 2700}` s (resume 0). Record per (region, ckpt): θ_p, θ_r, margin,
      savings %, overhead %, score, numPauses, completed.
      **Artifacts:** `publication/output/checkpoint/checkpoint_{region}.json`,
      `publication/output/checkpoint/checkpoint_all.csv`,
      `publication/output/checkpoint/checkpoint_summary.json` (structure mirrors `reopt_summary.json`).
      **Accept:** every cell recorded; `--ckpt-pause 148.8` reproduces the reopt baseline per region within
      tolerance (see Reference facts). Determinism: running twice yields identical JSON (no RNG in this path).
      **Verify:** `pnpm test`; `cmp` two runs; grep `checkpoint_summary.json` for the four values per region.

- [ ] **B.0.3 Monotonicity + sanity unit tests**
      **Inputs:** B.0.1 implementation, `src/cli/forecast-sweep.test.ts` conventions (vitest).
      **Steps:** add `src/cli/checkpoint-sweep.test.ts` asserting: (i) best savings is monotone non-increasing
      and overhead monotone non-decreasing as `--ckpt-pause` grows (same region/start); (ii) identity: at
      ckpt=148.8 the reported best matches `runOptimization` with constants; (iii) `marginRuleSurvives`
      recomputed under each ckpt (ceiling 16 g/kWh).
      **Accept:** all tests pass; tests are deterministic.
      **Verify:** `pnpm test`.

- [ ] **B.0.4 Decision memo (analysis, no code)**
      **Inputs:** B.0.2 artifacts, `research_report.md` §9.
      **Steps:** compute savings % at 900 s and 2700 s vs 148.8 s per region (absolute pp and relative loss).
      Write `publication/output/checkpoint/DECISION.md` with a table and a verdict:
      **Story A** if DE/IT keep ≥ ~2/3 of 148.8 s-savings at 900 s AND the optimizer still finds a feasible
      within-budget point at 2700 s; **Story B** otherwise. Include the AR(1) interval-width framing
      (width(72 steps) ≈ 8.4σ* ≈ 31 g/kWh for DE) as context.
      **Accept:** `DECISION.md` exists with a defensible verdict and the required table.
      **Verify:** numbers in the table are traceable to `checkpoint_summary.json`.

## Phase B — Experiments

### B.1 Stale-aware adaptive controller (core novelty; independent of B.0 verdict)

- [ ] **B.1.1 Margin-widening rule (implement + unit-test)**
      **Inputs:** `src/domain/forecast.ts`, `src/domain/types.ts`, `src/cli/forecast-sweep.ts`, calibration bundles
      in `publication/output/forecast/calibration_{region}.json`.
      **Steps:** implement margin rule `margin(h) = c · σ* · sqrt((1 − φ^2h)/(1 − φ²))` (AR(1) h-step prediction
      interval), thresholds widened symmetrically around the optimized nominal (θ_p, θ_r); `c` chosen on train
      years (2022–24), validated on test year (2025). Decision still on forecast; emissions on realized.
      Add pure-function module (e.g. `src/domain/adaptive-margin.ts`) with unit tests: monotone in h, = margin(0)
      when h=0, reduces to nominal margin at c=0.
      **Artifacts:** `src/domain/adaptive-margin.ts` + test.
      **Accept:** `pnpm test` green; formula matches the SPEC math; no simulation code touched.
      **Verify:** `pnpm test`.

- [ ] **B.1.2 Closed-loop evaluation across delay sweep (implement + run)**
      **Inputs:** B.1.1, `src/cli/forecast-sweep.ts` (reopt machinery), `src/cli/forecast-calibrate.ts`.
      **Steps:** extend `forecast-sweep --mode reopt` (or a new `--mode adaptive`) to evaluate the adaptive
      margin closed-loop for h ∈ {1,3,6,12,24,72}, regions DE/IT/SE, seeds {10,5}, against three baselines:
      naive-fixed (published θ), perfect-foresight, and static margin-widening oracle (upper bound from reopt
      drift). Report ΔS/S₀ recovery (fraction of lost savings recovered) and gap to the oracle.
      **Artifacts:** `publication/output/forecast/adaptive_{region}.json/csv`, `adaptive_summary.json`.
      **Accept:** deterministic; each (region, h) row has savings, overhead, recovery, oracle-gap; baseline rows
      reproduce `fixed_summary.json` / `reopt_summary.json`.
      **Verify:** `pnpm test`; `cmp` two runs.

- [ ] **B.1.3 Compare vs DTPR-style double thresholds**
      **Inputs:** research_report.md §3 row 3; `src/cli/forecast-sweep.ts`.
      **Steps:** implement a DTPR-flavored threshold benchmark (double thresholds with constant 2β separation
      derived from checkpoint cost, hourly aggregation) and compare its closed-loop savings/overhead against
      B.1.2 on the same traces. Position in Related Work as the theory anchor we build on.
      **Accept:** benchmark runs deterministically and is reported in `adaptive_summary.json`.
      **Verify:** `pnpm test`.

### B.2 Checkpoint realism — superseded by Phase B.0. If B.0 shows collapse (Story B), B.2 reduces to
reporting B.0.2 as the headline sensitivity (per SPEC "report as sensitivity, not a bug"). Otherwise,
re-run headline optimization under 900 s / 2700 s and report as a robustness table.

- [ ] (covered by B.0) Parameterize τ_c, τ_r by model size and storage bandwidth; sensitivity axis
      {150 s / 15 min / 45 min}.
- [ ] (covered by B.0) Re-run headline optimization under realistic checkpoint.

### B.3 Multi-year robustness

- [ ] **B.3.1 Year-stability of design rules (run)**
      **Inputs:** `public/data/co2/{DE,IT,SE}_{2022..2025}.json`, `src/cli/optimize.ts`.
      **Steps:** re-optimize per year 2022–2025 for DE/IT/SE (DeepSeek, budget 200 %, α=1). Verify the design
      rules (near-zero margin, percentile thresholds, grace horizon) are year-stable: for each rule, compute the
      max deviation across years and report pass/fail against a 5 pp savings / 10 g/kWh margin tolerance.
      **Artifacts:** `publication/output/forecast/multiyear_{region}.json/csv`, `multiyear_summary.json`.
      **Accept:** every year×region cell recorded; rule-deviation table present.
      **Verify:** `pnpm test`; numbers traceable to JSON.

- [ ] **B.3.2 Forecast-robustness in ≥ 2 years (run)**
      **Steps:** repeat the noise/staleness fixed-policy degradation (mirror `forecast-sweep --mode fixed`)
      on at least one additional test year per region; confirm grace horizons within one step of 2025 values.
      **Artifacts:** append year column to `fixed_summary.json` or a `multiyear_fixed_summary.json`.
      **Accept:** grace horizon stable (≤1 step) in ≥2 years for each region.
      **Verify:** grep summary.

### B.4 Overhead-budget sweep

- [ ] **B.4.1 Budget sweep (run)**
      **Steps:** re-run headline optimization (DE/IT/SE, 2025) for B ∈ {30, 50, 100, 200} % at ckpt = the
      B.0-verdict checkpoint (148.8 s if Story A is chosen, else 900 s). Report savings/overhead/margin per
      budget; record where the feasible frontier collapses.
      **Artifacts:** `publication/output/forecast/budget_summary.json/csv`.
      **Accept:** one row per (region, budget); collapse point identifiable.
      **Verify:** numbers traceable to JSON.

### B.5 (Stretch) Analytical grace horizon

- [ ] **B.5.1 Validate grace-horizon prediction (analysis)**
      **Inputs:** `calibration_{region}.json` (AR(1) φ, σ*), `fixed_summary.json` (empirical grace levels).
      **Steps:** for each region, check whether the empirical grace horizon (steps h where degradation ≤10 %)
      ≈ the h at which the AR(1) h-step RMSE reaches the threshold-margin scale (RMSE(h) ≈ k·(θ_p − μ)); fit k
      and report R² across regions/years. If R² ≥ ~0.9, this becomes a headline contribution.
      **Artifacts:** `publication/output/forecast/grace_horizon.md` with table + R².
      **Accept:** per-region table + fit quality reported.
      **Verify:** numbers traceable to calibration/fixed artifacts.

## Phase C — Figure pipeline (matplotlib)

- [ ] C.1 Set up `.venv` + matplotlib; `publication/eenergy/figures/make_figures.py` reads committed JSON/CSV
      (no recomputation). Uniform palette, ≥8 pt type, SVG + PDF. (`figures/` dir exists.)
- [ ] C.2 Closed-loop diagram (decide-on-forecast / pay-on-realized) — TikZ.
- [ ] C.3 CI trace sample + autocorrelation (near-unit-root intuition) — from `public/data/co2/DE_2025.json`.
- [ ] C.4 RMSE vs horizon: persistence vs AR(1) vs AR(7) — from `calibration_{region}.json` evaluation.
- [ ] C.5 Pareto frontiers, restyled, multi-year band — from `publication/output/results` (committed).
- [ ] C.6 **Noise-vs-staleness decomposition** (money figure): ΔS/S₀ vs error magnitude for noise AND staleness
      on a shared x-axis — from `fixed_summary.json` (additive/multiplicative/delay rows).
- [ ] C.7 **Grace-horizon map**: empirical vs predicted, per region — from B.5.1.
- [ ] C.8 Reopt drift arrows (θ_p, θ_r plane, per region) — from `reopt_summary.json`.
- [ ] C.9 **Adaptive controller recovery curves** — from B.1.2.
- [ ] C.10 Region × year savings heatmap — from B.3.1.

## Phase D — Paper restructure (acmart sigconf, ~10 pages; direction per B.0.4)

- [ ] D.1 Intro: frontier pretraining emissions + grid flexibility; online decision under forecast uncertainty;
      the central question (noise vs staleness). Contributions list.
- [ ] D.2 Related Work + positioning table (empty row = us). Must cite & position: UQ-Advice, LACS, DTPR/OPR,
      equilibrium analysis, Moving-Beyond-MCI, average-vs-marginal, Green Mirage, curtailment-LLM,
      CarbonCast/EnsembleCI/CarbonX, Carbon-Aware Quality Adaptation, Uncertainty-Aware Decarbonization,
      Wiesner Limitations, Let's Wait Awhile. Use `research_report.md` §11 bib keys.
- [ ] D.3 System Model & Problem Formulation: job model, hysteresis policy, decide-on-forecast/pay-on-realized,
      metrics, grace horizon definition, adaptive margin rule.
- [ ] D.4 Grid CI at 5-min resolution: near-unit-root characterization, persistence ≈ AR(1) ≈ AR(7).
- [ ] D.5 Threshold Optimization & Design Rules (condensed): optimizer, Pareto frontiers, near-zero margin,
      percentile rules, budget sweep, multi-year stability.
- [ ] D.6 Forecast Robustness: noise/staleness decomposition, grace horizon, reopt drift.
- [ ] D.7 Stale-Aware Adaptive Control: algorithm, closed-loop evaluation, recovery vs naive and perfect-foresight,
      gap vs static oracle. Compare DTPR.
- [ ] D.8 Discussion: demand-side flexibility framing, data-freshness SLA for operators, signal choice
      (ACI vs MCI vs excess power — engage Moving-Beyond-MCI + equilibrium critique), marginal-vs-average
      accounting, embodied carbon/water, checkpoint realism, limitations.
- [ ] D.9 Conclusion.
- [ ] D.10 Claim–evidence pass: every number in main.tex traceable to a committed artifact
      (`publication/output/**/*.json`). The SPEC "Known errors" list (below) must be honored.

## Phase E — QA & submission (target e-Energy 2027 Winter, submit ~early-mid Jan 2027)

- [ ] E.1 Adversarial review (mirror `phase_review.md` protocol) × 2; fix all findings.
- [ ] E.2 `make` clean under acmart; 0 LaTeX errors; figures vector-embedded.
- [ ] E.3 `pnpm test` + `pnpm build` green; `tsc` no new errors.
- [ ] E.4 `run_eenergy_experiments.sh` deterministic end-to-end; reproducibility statement
      (seeds, commands, runtimes, commit hash).
- [ ] E.5 Buffer ≥ 2 weeks before Winter 2027 deadline.

## Reference facts (from committed artifacts — used for the claim–evidence pass)

- Perfect-foresight optima (Table 2, ICREC paper; reopt baselines in `reopt_summary.json`):
  DE θ=(272.37, 267.73) margin 4.64, S=43.35 %, O=174.31 %; IT θ=(246.70, 230.45) margin 16.25,
  S=32.67 %, O=194.68 %; SE θ=(18.18, 17.51) margin 0.67, S=23.17 %, O=106.34 %.
- Calibration (train 2022–24, test 2025, `calibration_{region}.json`): DE σ*=3.658, lag-1=0.999655,
  AR(1) φ=0.999655, innovation std=3.658; IT σ*=4.421, lag-1=0.9987, φ=0.99868; SE σ*=0.774, lag-1=0.9960,
  φ=0.99595. Persistence–AR(1) RMSE gap ≤ 0.06 g/kWh at all h ≤ 72.
- Fixed-policy degradation (`fixed_summary.json`): additive/multiplicative at σ*: 0.31 / 0.16 / 4.83 %
  (DE/IT/SE); at 4σ*: additive 2.4 / 6.5 / 40.6 %. Staleness h=72 (persistence): 35.8 / 42.2 / 60.5 %;
  grace horizon 24 / 12 / 12 steps.
- Reopt drift (`reopt_summary.json`): DE margin 4.64 → 16.68 (1σ*) → 28.36 (2σ*); IT 16.25 → 23.19 → 30.15;
  SE 0.67 → 3.21 → 5.49. Margin rule fails DE/IT ≥1σ* additive (SE survives all; IT also fails at delay=1 —
  the SPEC's original "survives all" was wrong, do not repeat).
- Grid data: `public/data/co2/{zone}_{year}.json`, 5-min, 105 120 pts/yr, zones {DE, IT, SE, US, CN},
  years {2022..2026}; ACI (average carbon intensity) — NOT marginal. See Discussion (Moving-Beyond-MCI:
  Electricity Maps discontinued MCI).
- CLI entry: `src/cli/index.ts` (commander); test runner: vitest; existing forecast experiments in
  `src/cli/forecast-{calibrate,sweep}.ts` + `src/domain/forecast.ts`; optimizer in `src/domain/optimize.ts`;
  checkpoint constants in `public/data/constants.json` (148.8 s pause, 0 s resume).

## Known errors in ICREC version to NOT repeat

- False claim: "margin rule survives all delay levels in all regions" — false for IT delay=1
  (margin stays at 16.25 baseline, seedFraction 0).
- SE fixed-policy control used rounded thresholds (19,18) with a non-optimal start date
  (S₀=22.87 % / 99.4 % overhead) ≠ the reopt optimum (18.18, 17.51 → 23.17 % / 106.3 %).
- "Gap never exceeds ≈ 0.05 g/kWh" — actually 0.0512 at IT h=72; phrase as "≈ 0.05… ≤ 0.06".
- Checkpoint time 148.8 s is a 7B-model figure applied to 671B — must be swept (Phase B.0).
- "Hanford '16" citation is unverifiable — drop.
- Do NOT claim "first Pareto frontiers" or "first to use two thresholds" (see Phase A findings).
