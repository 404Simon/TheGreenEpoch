# Phase B.3 + B.4 — Multi-year robustness & overhead-budget sweep (report)

- Date: 2026-08-19
- Agent: implementation/run agent, Phases B.3.1, B.3.2, B.4.1 (`reframe-paper-stale-aware`)
- Story A continues ("works-with-design-rules"). All numbers traceable to
  `publication/output/forecast/{multiyear_*.json,multiYearSummary...}` (see §7).
- Method: **CLI-only + a new standalone aggregation script** — no domain code touched.

## 1. What was done

1. **B.3.1 (year-stability).** Re-optimized DE/IT/SE on each of 2022–2025 with the
   exact reopt settings (DeepSeek, budget 200 %, α=1, resolution 10, iterations 6,
   fixed per-region start DE `02-01` / IT `01-14` / SE `04-22`, tpMax 800/800/100,
   ckpt = constants 148.8 s) via `pnpm cli optimize`. Aggregated into
   `multiyear_{region}.{json,csv}` and `multiyear_summary.json`, including the four
   design-rule checks with pass/fail.
2. **B.3.2 (forecast robustness across years).** Mirrored the committed
   `forecast-sweep --mode fixed` config (default families `additive,multiplicative,
   delay,arma,persistence`, levels `0,0.25,0.5,1,2,4`, horizons `1,3,6,12,24,72`,
   seeds DE=10 / IT=SE=5, budget 200 %) on test years **2025, 2024, 2023**. The 2025
   run reproduces the committed `fixed_summary.json` **byte-identically**
   (`JSON.stringify` equal). Decision models use the 2025 calibration bundle
   as-is (documented choice, task's "keep it simple" option). Aggregated into
   `multiyear_fixed_summary.json`.
3. **B.4.1 (budget sweep).** Re-ran the 2025 headline optimization for
   B ∈ {30, 50, 100, 200} % at **ckpt-pause 148.8 s** (the B.0/Story-A verdict
   checkpoint) and, as the robustness block, at **900 s**. Aggregated into
   `budget_summary.{json,csv}`. Phase-B.0 caveat honored: every cell carries both
   the raw max-score `best` and the completed-feasible `best_completed`.
4. **Reproducibility.** `run_multiyear_budget.sh` re-runs all three phases
   deterministically; `DETERMINISM_CHECK=1` auto-verifies byte-identical artifacts.

## 2. B.3.1 results — per (region, year) optimized cells (2025 = reopt baseline)

All cells `found=true`, `completed=true`, `within_budget=true`.

| region | year | θ_p | θ_r | margin | savings % | overhead % | score | pauses |
|---|---|---|---|---|---|---|---|---|
| DE | 2022 | 449.14 | 433.14 | 16.00 | 14.95 | 117.2 | 0.5747 | 63 |
| DE | 2023 | 316.10 | 301.06 | 15.04 | 28.12 | 185.4 | 0.6406 | – |
| DE | 2024 | 251.61 | 246.85 | 4.76 | 28.33 | 190.9 | 0.6417 | – |
| DE | 2025 | 272.37 | 267.73 | 4.64 | 43.35 | 174.3 | 0.7168 | 102 |
| IT | 2022 | 372.56 | 369.32 | 3.24 | 10.02 | 177.9 | 0.5501 | – |
| IT | 2023 | 296.97 | 288.97 | 8.00 | 25.06 | 187.7 | 0.6253 | – |
| IT | 2024 | 222.27 | 217.55 | 4.72 | 20.94 | 187.1 | 0.6047 | – |
| IT | 2025 | 246.70 | 230.45 | 16.25 | 32.67 | 194.7 | 0.6633 | 173 |
| SE | 2022 | 19.18 | 18.43 | 0.75 | 20.10 | 108.6 | 0.6005 | – |
| SE | 2023 | 18.18 | 17.51 | 0.67 | 16.18 | 134.9 | 0.5809 | – |
| SE | 2024 | 16.48 | 15.76 | 0.72 | 22.42 | 183.7 | 0.6121 | – |
| SE | 2025 | 18.18 | 17.51 | 0.67 | 23.17 | 106.3 | 0.6158 | 85 |

(− = pauses present in JSON, omitted here for width; every cell is fully recorded in
`multiyear_{region}.json`.) **D2 check:** the 2025 rows reproduce `reopt_summary.json`
baselines with θ deviation 0.00 and savings deviation 0.0000 pp (exact).

## 3. B.3.1 design-rule deviation table (pass/fail per region; tolerances 5 pp savings / 10 g/kWh margin)

| Region | Rule | Values across 2022–2025 | Max deviation | Tolerance | Pass/Fail |
|---|---|---|---|---|---|
| DE | Near-zero margin | margins 16.00, 15.04, 4.76, 4.64 g/kWh | max 16.00 | ≤10 g/kWh | **FAIL** (holds 2/4 yr) |
| IT | Near-zero margin | 3.24, 8.00, 4.72, 16.25 g/kWh | max 16.25 | ≤10 or ≥3/4 yr | **PASS** (holds 3/4 yr; 2025 = 16.25) |
| SE | Near-zero margin | 0.75, 0.67, 0.72, 0.67 g/kWh | max 0.75 | ≤10 g/kWh | **PASS** (4/4) |
| DE | Percentile (θ_p exceedance / θ_r subceedance) | 60.0/35.9, 59.9/35.8, 71.2/27.6, 65.3/33.5 % | 11.29 / 8.30 pp | 10 pp* | **FAIL** (θ_p dev > 10 pp) |
| IT | Percentile | 69.8/28.7, 60.3/36.2, 74.0/23.6, 68.6/26.1 % | 13.71 / 12.56 pp | 10 pp* | **FAIL** |
| SE | Percentile | 74.8/20.3, 68.9/24.4, 71.5/21.1, 60.0/30.2 % | 14.82 / 9.95 pp | 10 pp* | **FAIL** |
| DE | Grace horizon | delay/persistence 24/24/24 | 0 steps | ≤1 step, ≥2 yr | **PASS** |
| IT | Grace horizon | 12 (2025), 24 (2024, 1 step), 72 (2023, 2 steps) | 2 steps (2023) | ≤1 step, ≥2 yr | **PASS** (2/3 yr; 2023 outlier) |
| SE | Grace horizon | 12/12/12 | 0 steps | ≤1 step, ≥2 yr | **PASS** |
| DE | Savings stability | min 14.95, max 43.35 % | 28.40 pp | ≤5 pp | **FAIL** |
| IT | Savings stability | min 10.02, max 32.67 % | 22.65 pp | ≤5 pp | **FAIL** |
| SE | Savings stability | min 16.18, max 23.17 % | 6.99 pp | ≤5 pp | **FAIL** |

\* The SPEC gives no explicit percentile tolerance; 10 pp is our documented analog
of the 10 g/kWh margin tolerance (see §6.1). At the stricter 5 pp reading the
outcome is identical (all three already fail).

**Honest headline for the paper (D.5 story):** the *relative* design rules are the
year-stable ones — near-zero margin (DE/IT caveat: energy-crisis years 2022–23 push
DE margins to 15–16 g/kWh, and IT's 2025 reopt operating point is 16.25 g/kWh) and
the grace horizon (stable within the 12/24/72-step scale in ≥2 years everywhere).
The *absolute* quantities are year-specific: optimized savings vary 7–28 pp across
years (grid decarbonization + energy-crisis years) and the implied θ_p percentile
wanders ±11–15 pp. The paper must not claim a universal savings % or a universal
threshold percentile; it can claim stable margin scale + stable grace horizon.

## 4. B.3.2 results — grace-horizon stability (from `multiyear_fixed_summary.json`)

s0 = fixed (published) policy control savings; h72 = persistence-family degradation at h=72.

| region | year | s0 % | h72 deg % | grace delay | grace persistence |
|---|---|---|---|---|---|
| DE | 2025 | 43.35 | 35.8 | 24 | 24 |
| DE | 2024 | 27.24 | 39.7 | 24 | 24 |
| DE | 2023 | 55.48 | 12.7 | 24 | 24 |
| IT | 2025 | 32.67 | 42.2 | 12 | 12 |
| IT | 2024 | 19.42 | 36.1 | 24 | 24 |
| IT | 2023 | 73.75 | 3.1 | 72 | 72 |
| SE | 2025 | 22.87 | 60.5 | 12 | 12 |
| SE | 2024 | 19.82 | 80.2 | 12 | 12 |
| SE | 2023 | 15.86 | 63.1 | 12 | 12 |

**Stability check (grid {1,3,6,12,24,72}; "≤1 step" = same or adjacent grid value):**
DE stable 24/24/24 (0 steps), SE stable 12/12/12 (0 steps); IT = 12 (2025) and
24 (2024) are 1 step apart → 2 of 3 years within 1 step → **PASS** per "≥2 years".
IT 2023 is a genuine outlier (grace 72: that year's CI was so stable that even
h=72 persistence degraded savings by only 3.1 %). Honest reporting: IT grace is
"12–24 steps in 2024–25, up to 72 in 2023".

**Grep-able assertion:**
`grep -o '"grace_delay": [0-9]*' publication/output/forecast/multiyear_fixed_summary.json`
→ `4 × 12, 4 × 24, 1 × 72`.

## 5. B.4.1 results — overhead-budget sweep (2025, DeepSeek)

Headline block at ckpt 148.8 s (Story-A checkpoint); robustness block at 900 s.

### 5.1 ckpt 148.8 s (headline)

| region | budget % | θ_p | θ_r | margin | S % (best) | O % | completed | S % (completed-feasible) |
|---|---|---|---|---|---|---|---|---|
| DE | 30 | 533.76 | 517.76 | 16.0 | 11.51 | 30.0 | yes | 11.51 |
| DE | 50 | 491.33 | 486.31 | 5.0 | 16.28 | 48.8 | yes | 16.28 |
| DE | 100 | 410.57 | 398.97 | 11.6 | 29.52 | 98.6 | yes | 29.52 |
| DE | 200 | 272.37 | 267.73 | 4.6 | 43.35 | 174.3 | yes | 43.35 |
| IT | 30 | – | – | – | **none** | – | – | none |
| IT | 50 | – | – | – | **none** | – | – | none |
| IT | 100 | 283.20 | 258.80 | 24.4 | 82.00* | 100.0 | no | 19.85 |
| IT | 200 | 246.70 | 230.45 | 16.3 | 32.67 | 194.7 | yes | 32.67 |
| SE | 30 | 25.80 | 24.90 | 0.9 | 14.59 | 29.0 | yes | 14.59 |
| SE | 50 | 22.30 | 20.30 | 2.0 | 18.36 | 48.5 | yes | 18.36 |
| SE | 100 | 19.20 | 17.20 | 2.0 | 22.87 | 99.4 | yes | 22.87 |
| SE | 200 | 18.18 | 17.51 | 0.7 | 23.17 | 106.3 | yes | 23.17 |

\* incomplete (budget-blocked) max-score `best` with inflated savings — phase-B.0
artifact; the completed-feasible value (19.85 %) is the honest one.

### 5.2 ckpt 900 s (robustness block; completed-feasible values)

| region | budget % | θ_p | θ_r | margin | S % (best) | O % | completed | S % (completed-feasible) |
|---|---|---|---|---|---|---|---|---|
| DE | 30 | 479.18 | 453.76 | 25.4 | 68.01* | 30.0 | no | 10.42 |
| DE | 50 | 508.00 | 335.30 | 172.7 | 68.05* | 50.0 | no | 15.51 |
| DE | 100 | 379.20 | 370.20 | 9.0 | 59.86* | 100.0 | no | 28.46 |
| DE | 200 | 246.93 | 234.80 | 12.1 | 47.91* | 200.0 | no | 42.32 |
| IT | 30 | – | – | – | **none** | – | – | none |
| IT | 50 | – | – | – | **none** | – | – | none |
| IT | 100 | 283.20 | 258.80 | 24.4 | 81.70* | 100.0 | no | 18.34 |
| IT | 200 | 209.04 | 172.41 | 36.6 | 70.56* | 200.0 | no | 30.90 |
| SE | 30 | 27.80 | 20.90 | 6.9 | 41.08* | 30.0 | no | 13.25 |
| SE | 50 | 21.40 | 18.30 | 3.0 | 76.81* | 50.0 | no | 17.19 |
| SE | 100 | 19.20 | 17.20 | 2.0 | 21.80 | 99.4 | yes | 21.80 |
| SE | 200 | 18.18 | 17.51 | 0.7 | 22.06 | 106.3 | yes | 22.06 |

### 5.3 Collapse points (feasible frontier)

- **IT**: frontier **collapses at budget ≤ 50 %** at both 148.8 s and 900 s — no
  within-budget AND completed point with positive savings exists at B ∈ {30, 50}
  (`found=false`). At B=100 a completed-feasible point exists but savings drop to
  19.85 % (148.8 s) / 18.34 % (900 s) vs 32.67 % at B=200. IT's nominal operating
  point already sits at 194.7 % overhead, so it has no room to tighten.
- **DE**: **no collapse in-grid**; completed-feasible savings fall smoothly
  43.35 → 11.51 % (148.8 s) / 42.32 → 10.42 % (900 s) as B drops to 30 %.
- **SE**: **no collapse in-grid**; completed-feasible savings 23.17 → 14.59 %
  (148.8 s) / 22.06 → 13.25 % (900 s).
- At 900 s every DE/IT cell's max-score `best` is budget-blocked-incomplete with
  inflated savings (phase-B.0 caveat); the completed-feasible columns are the
  paper's B.4 numbers.

## 6. DoD checklist — pass/fail with evidence

- **D1 `pnpm build` + `pnpm test`** — PASS. `pnpm build` → `✓ built in 2.51s`;
  `pnpm test` → `14 files, 205 tests passed` (all pre-existing; no source touched).
- **D2 every (region, year) cell recorded + 2025 reproduces reopt baseline** —
  PASS. 12/12 cells in `multiyear_summary.json` with θ_p, θ_r, margin, savings,
  overhead, score, num_pauses, completed, within_budget (+ `best_completed` and
  percentile fields); verification script reports 2025 rows match `reopt_summary.json`
  with dθ = 0.00 and dS = 0.0000 pp.
- **D3 rule-deviation table with pass/fail, numbers traceable** — PASS.
  `multiyear_summary.json` → `rule_deviation.{near_zero_margin,percentile_thresholds,
  grace_horizon,savings_stability}` per region with per-year values, deviations and
  notes; table reproduced in §3.
- **D4 grace horizons stable (≤1 step) in ≥2 years per region, recorded +
  grep-able** — PASS. `multiyear_fixed_summary.json` has all 9 (region, year) rows;
  DE 24/24/24, SE 12/12/12, IT 12/24/72 (within 1 step in 2/3 years). Grep check:
  `grep -o '"grace_delay": [0-9]*' multiyear_fixed_summary.json` → `4×12, 4×24, 1×72`.
- **D5 budget sweep one row per (region, budget) at 148.8 s + 900 s block;
  collapse identified** — PASS. `budget_summary.csv` = 1 header + 24 rows
  (3 × 4 × 2); `budget_summary.json` has `collapse.ckpt_148_8` and `collapse.ckpt_900`
  per region; IT collapses at ≤50 %, DE/SE do not (see §5.3).
- **D6 determinism** — PASS. `DETERMINISM_CHECK=1 bash run_multiyear_budget.sh`
  → snapshot → full re-run → `cmp` IDENTICAL on all 10 artifacts.
- **D7 protected-domain files unmodified** — PASS. `git diff --stat src/domain/`
  empty (the only `src/domain` change in the working tree is the B.1 addition
  `adaptive-margin.ts`); `src/cli/{index,optimize,forecast-sweep}.ts` mods are
  pre-existing (B.0/B.1). This phase added only `_aggregate_multiyear_budget.mjs`
  + `run_multiyear_budget.sh` (standalone scripts) and output artifacts.
- **D8 `phase_reports/phase_b3b4.md`** — PASS (this file).

## 7. Deviations / ambiguities resolved

1. **Percentile rule tolerance not specified in the SPEC.** Used 10 pp deviation
   (on the implied P(CI > θ_p) / P(CI < θ_r) percentages) as the closest analog of
   the 10 g/kWh margin tolerance; outcome unchanged under a 5 pp reading. Reported
   both `pct_exceed_theta_p` and `pct_below_theta_r` per year plus CDF `pct_le_theta_p`.
2. **Grace-horizon "≤1 step"** operationalized on the horizon grid
   {1,3,6,12,24,72}; IT 2023 (grace 72, 2 steps off) is a genuine outlier and is
   reported as such. IT passes the "stable in ≥2 years" criterion via 2025+2024.
3. **Near-zero margin "≥3 of 4 years" clause** applied for IT (margins ≤10 in
   2022–24, 16.25 in 2025) per the SPEC's explicit instruction to say so honestly.
4. **Savings-stability fails everywhere** — this is a *finding*, not a bug:
   absolute optimized savings reflect each year's grid opportunity (energy-crisis
   2022, decarbonization trend), so the paper's year-stability claim rests on the
   threshold *structure* (margin scale, grace horizon), not the savings level.
5. **B.3.2 decision models** use the 2025 calibration bundle as-is (task's
   "keep it simple" option); the 2025 run was re-executed and is byte-identical to
   the committed `fixed_summary.json`, so the added years are directly comparable.
6. **B.0 caveat honored throughout**: every cell records the raw max-score `best`
   *and* the completed-feasible `best_completed`; at 900 s DE/IT max-score savings
   are inflated-incomplete and the paper must use the completed-feasible column.
7. **Raw evidence** kept under `publication/output/forecast/_raw/`
   (`multiyear/`, `budget/`, `fixed_202{3,4,5}/`); regenerated on every script run.

## 8. Exact commands

```bash
# Reproduce everything (B.3.1 + B.3.2 + B.4.1), deterministic:
bash publication/output/forecast/run_multiyear_budget.sh

# Determinism: snapshot -> re-run -> cmp (10 artifacts byte-identical):
DETERMINISM_CHECK=1 bash publication/output/forecast/run_multiyear_budget.sh

# Single-cell examples (the per-run commands inside the script):
pnpm cli optimize -m Deepseek -r DE -y 2022 --start 02-01 --tp-max 800 --budget 200 --resolution 10 --max-iter 6
pnpm cli optimize -m Deepseek -r IT -y 2025 --start 01-14 --tp-max 800 --budget 50  --resolution 10 --max-iter 6 --ckpt-pause 900 --ckpt-resume 0
pnpm cli forecast-sweep --mode fixed -y 2024 -o /tmp/f24/fixed

# Verification greps / checks:
grep -o '"grace_delay": [0-9]*' publication/output/forecast/multiyear_fixed_summary.json
grep -o '"grace_persistence": [0-9]*' publication/output/forecast/multiyear_fixed_summary.json
node -e 'const s=require("./publication/output/forecast/multiyear_summary.json"); for (const r of s) console.log(r.region, r.rule_deviation.near_zero_margin.pass, r.rule_deviation.percentile_thresholds.pass, r.rule_deviation.grace_horizon.pass, r.rule_deviation.savings_stability.pass)'
pnpm build && pnpm test
```

Artifacts added: `publication/output/forecast/multiyear_{DE,IT,SE}.{json,csv}`,
`multiyear_summary.json`, `multiyear_fixed_summary.json`, `budget_summary.{json,csv}`,
`run_multiyear_budget.sh`, `_aggregate_multiyear_budget.mjs`, `_raw/` evidence.
Nothing committed; `publication/ICREC_Rome/` untouched; protected domain files unmodified.
