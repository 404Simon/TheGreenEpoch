# Phase B.1 — Stale-aware adaptive controller (report)

- Date: 2026-08-19
- Agent: implementation agent, Phase B.1 (`reframe-paper-stale-aware`)
- Verdict: **Implemented, deterministic, baselines exact.** Honest finding: the adaptive
  margin rule recovers a *modest* fraction of the staleness loss; the completion constraint
  (phase B.0 caveat) is the dominant design factor, and the h=72 loss is largely *structural*
  (unrecoverable even by the static oracle).
  > **CORRECTED (per adversarial review M1/M2):** "structural" must be read as "structural for
  > *static-threshold policies and for the train-validated c***". The margin rule is linear in c
  > and *can* express the oracle scale (DE h=72: c=6 → S=31.43, completed, recovery_vs_oracle
  > 0.748); the modest h=72 recovery in §3.2 is c*-specific (c-grid ceiling ≤2 + train-year
  > selection), not a property of the rule. See §3.3 and §5.3 corrections below.
- All numbers traceable to `publication/output/forecast/adaptive_{DE,IT,SE}.json/.csv`
  and `adaptive_summary.json`. Baselines reproduce `fixed_summary.json` / `reopt_summary.json`
  **exactly** (0.0000 pp).

## 1. What was done

1. **B.1.1** `src/domain/adaptive-margin.ts` (pure functions, no simulation imports):
   `adaptiveMargin({sigmaStar, phi, c, h})` implementing the SPEC margin rule
   `c·σ*·sqrt((1−φ^(2h))/(1−φ²))`, `widenedThresholds(nominal, margin)` (symmetric widening
   around the nominal midpoint), and `arScaleFactor(phi, h)` for reporting. Edge cases:
   h=0 → 0; c=0 → 0; φ²≥1 → limit `c·σ*·sqrt(h)` (avoids 0/0); invalid inputs → NaN.
   Unit tests `src/domain/adaptive-margin.test.ts` assert (i) monotone in h, (ii) margin(0)=0,
   (iii) c=0 ⇒ nominal, (iv) DE case 8.3823·σ*≈30.7 g/kWh, (v) midpoint preservation,
   plus the finite-sum identity and φ=1 limit.
2. **B.1.2** Extended `src/cli/forecast-sweep.ts` with `--mode adaptive` (pure-function core
   exported for tests). For DE/IT/SE × h ∈ {1,3,6,12,24,72}: decision-on-forecast
   (AR(1) h-step ahead, `applyForecast {type:"arma", order:1, horizon:h, coeffs}`), pay-on-realized
   emissions, naive-fixed / perfect-foresight / static-oracle baselines, adaptive policy at a
   per-region c* chosen on train years (2022–24), and the DTPR-style benchmark (B.1.3).
3. **B.1.3** DTPR-style double-threshold benchmark with constant separation 2β derived from the
   checkpoint cost, decision on an hourly-aggregated version of the same arma(h) signal.
4. Runner script `publication/output/forecast/run_adaptive_sweep.sh` (deterministic; optional
   `ADAPTIVE_DETERMINISM_CHECK=1` auto-verifies via snapshot → re-run → `cmp`).
5. Test file `src/cli/adaptive-sweep.test.ts`: artifact↔committed-data regression guards (D4–D7)
   plus pure-function tests of `computeRecovery` / `hourlyAggregate`.

## 2. Precise definitions (as implemented)

**Decision model.** Decision timeline at staleness h = AR(1) h-step-ahead forecast
`applyForecast(realized, {type:"arma", order:1, horizon:h, coeffs: calibration.orders["1"].coeffs})`.
This is the `arma(h)` family already used by the committed fixed sweep: for order 1 it evaluates
`decision[t] = intercept + φ·carbon[t−h]` (damped persistence, near-unit-root φ ≈ 1). Emissions are
evaluated on the *realized* 5-min timeline (pay-on-realized). All decision models used here are
deterministic (no RNG); the `seed` field is kept for API parity with the fixed mode and is a no-op
(seed-averaging would add nothing).

**Adaptive rule.** `margin(h) = c·σ*·sqrt((1−φ^(2h))/(1−φ²))`;
`theta_p = nominal.thetaP + margin/2`, `theta_r = nominal.thetaR − margin/2`
(symmetric widening around the nominal midpoint), nominal = reopt baseline
(DE 272.37/267.73, IT 246.70/230.45, SE 18.18/17.51). c=0 ⇒ exactly the reopt nominal.

**Baselines (same decision timeline, test year 2025).**
- naive-fixed: published thresholds (DE 272/268, IT 246/231, SE 19/18) — reproduces
  `fixed_summary.json` arma rows.
- perfect-foresight: identity decision — reproduces `fixed_summary.json` control S₀.
- static oracle: per-h reoptimized static thresholds under the arma(h) decision timeline
  (`runOptimization`, reopt settings resolution 10 / iterations 6 / budget 200 / α=1 / fixed
  start, one seed, deterministic). h∈{1,6} sourced from committed `reopt_{region}.json` delay rows
  — verified **exact to 6 decimals** against a fresh arma(h) optimization (delay(h) ≡ arma(h) at
  these φ); h∈{3,12,24,72} freshly optimized.

**Recovery.** `recovery = clip((S_adaptive − S_naive)/(S0_FF − S_naive), 0, 1)` where
S0_FF = perfect-foresight savings of the naive (published) policy. Convention: recovery = 0 when
`S0_FF − S_naive ≤ 1e-3 pp` (no positive loss to recover, including naive ≥ perfect; SE h=1 arma
slightly *exceeds* the identity control, so recovery is 0 there). `recovery_raw` is the unclipped
value. **Completion guard (added, see §6.1):** an incomplete (budget-exhausted) run's savings is
inflated vs a completed baseline (phase B.0 caveat), so headline recovery = 0 whenever the adaptive
or naive run does not complete; the raw formula is kept as `recovery_raw`.
`oracle_gap = S_oracle − S_adaptive`. Supplementary: `recovery_vs_oracle =
clip((S_adaptive − S_naive)/(S_oracle − S_naive), 0, 1)` = fraction of the *recoverable*
(naive→oracle) gap closed.

**c-selection (train 2022–24, validate 2025).** Per region, `c* = argmax` over the grid
{0, 0.25, 0.5, 0.75, 1, 1.5, 2} of the mean clipped recovery over train years × horizons, ties
broken toward the smallest c. Recovery is evaluated per (year, h) against that (year, h)'s own
naive-perfect gap (self-normalizing across years) and with the completion guard. The chosen c* is
then applied on the test year.

## 3. Results

### 3.1 c* selection (train years)

| region | mean recovery by c (0, 0.25, 0.5, 0.75, 1, 1.5, 2) | c* | mean recovery at c* |
|---|---|---|---|
| DE | 0.0059, 0.0062, 0.0172, **0.0556**, 0.0121, 0.0000, 0.0158 | **0.75** | 0.0556 |
| IT | 0.0035, 0.0034, 0.0000, 0.0001, 0.0023, 0.0100, **0.0248** | **2** | 0.0248 |
| SE | 0.0000, 0.0015, 0.0217, 0.0163, 0.0217, **0.0289**, 0.0000 | **1.5** | 0.0289 |

Note the train-year signal is *weak and noisy*: IT c*=2 is driven by a single 2024 cell
(c=2, h=3 → recovery 0.44); 2022/2023 contribute ~0 for every c in every region. This flatness is
itself a finding (§5).

### 3.2 Test-year (2025) closed loop at c* — per (region, h)

S0_FF = perfect-foresight savings of the naive policy; ceiling = perfect-foresight savings of the
adaptive policy's nominal (reopt) thresholds. `recov` is completion-guarded, `recVsOracle` is the
fraction of the recoverable (naive→oracle) gap closed.

**DE (S0_FF 43.35, ceiling 43.35, c*=0.75):**

| h | naive | perfect | adapt | adaptOverhead | adaptCompl | recovery | recVsOracle | oracle | oracle_gap | DTPR | DTPRoverhead | DTPRcompl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 43.25 | 43.35 | 43.32 | 174.3 | yes | 0.688 | 0.954 | 43.32 | 0.00 | 43.13 | 175.8 | yes |
| 3 | 43.18 | 43.35 | 43.24 | 174.3 | yes | 0.349 | 0.799 | 43.26 | 0.01 | 43.06 | 176.0 | yes |
| 6 | 42.97 | 43.35 | 42.93 | 175.7 | yes | 0.000 | 0.000 | 43.04 | 0.11 | 42.80 | 174.3 | yes |
| 12 | 42.21 | 43.35 | 42.09 | 175.8 | yes | 0.000 | 0.000 | 42.31 | 0.21 | 42.02 | 174.4 | yes |
| 24 | 39.78 | 43.35 | 39.50 | 175.9 | yes | 0.000 | 0.000 | 39.91 | 0.41 | 39.53 | 174.5 | yes |
| 72 | 27.73 | 43.35 | 27.32 | 176.2 | yes | 0.000 | 0.000 | 32.67 | 5.35 | 27.66 | 174.8 | yes |

**IT (S0_FF 32.67, ceiling 32.67, c*=2):**

| h | naive | perfect | adapt | adaptOverhead | adaptCompl | recovery | recVsOracle | oracle | oracle_gap | DTPR | DTPRoverhead | DTPRcompl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32.51 | 32.67 | 32.26 | 199.6 | yes | 0.000 | 0.000 | 32.56 | 0.30 | 33.42* | 200.0 | no |
| 3 | 32.31 | 32.67 | 32.83* | 200.0 | no | 0.000 | 0.000 | 32.35 | −0.48 | 33.96* | 200.0 | no |
| 6 | 32.05 | 32.67 | 33.19* | 200.0 | no | 0.000 | 0.000 | 32.13 | −1.06 | 34.15* | 200.0 | no |
| 12 | 31.31 | 32.67 | 33.87* | 200.0 | no | 0.000 | 0.000 | 31.44 | −2.43 | 32.17* | 200.0 | no |
| 24 | 28.96 | 32.67 | 32.14* | 200.0 | no | 0.000 | 0.000 | 29.32 | −2.81 | 30.28* | 200.0 | no |
| 72 | 18.78 | 32.67 | 22.47* | 200.0 | no | 0.000 | 0.000 | 25.50 | 3.03 | 20.27* | 200.0 | no |

\* incomplete (budget-exhausted) — savings inflated, excluded from recovery.

**SE (S0_FF 22.87, ceiling 23.17, c*=1.5):**

| h | naive | perfect | adapt | adaptOverhead | adaptCompl | recovery | recVsOracle | oracle | oracle_gap | DTPR | DTPRoverhead | DTPRcompl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 23.09 | 22.87 | 21.98 | 129.3 | yes | 0.000 | 0.000 | 23.09 | 1.11 | 22.35 | 119.2 | yes |
| 3 | 22.92 | 22.87 | 21.70 | 121.9 | yes | 0.000 | 0.000 | 22.92 | 1.22 | 22.37 | 115.7 | yes |
| 6 | 22.68 | 22.87 | 21.32 | 121.9 | yes | 0.000 | 0.000 | 22.68 | 1.36 | 22.22 | 114.0 | yes |
| 12 | 22.17 | 22.87 | 17.87 | 143.6 | yes | 0.000 | 0.000 | 22.17 | 4.30 | 22.16 | 106.5 | yes |
| 24 | 20.23 | 22.87 | 23.46* | 200.0 | no | 0.000 | 0.000 | 20.31 | −3.15 | 20.23 | 106.5 | yes |
| 72 | 8.64 | 22.87 | 47.66* | 200.0 | no | 0.000 | 0.000 | 12.93 | −34.73 | 8.52 | 106.5 | yes |

\* incomplete (budget-exhausted) — savings inflated, excluded from recovery.

> **CORRECTED (per adversarial review M2).** The "recovery = 0 at h≥6 (DE) / IT / SE" rows are
> *c*-specific, not a property of the controller: DE h=72 recovers ≈0 *at the train-selected
> c*=0.75*, while the same rule at c=6 recovers 0.237 of the total loss / 0.748 of the recoverable
> gap on the completed 2025 test year (see `adaptive_sensitivity_DE.json` and §5.3). The genuinely
> structural claim is only the *static-oracle* ceiling (32/48/30 % of the h=72 loss unreachable by
> any static-threshold policy) and the *budget-binding* for IT (and SE at h≥24).

### 3.3 Static oracle thresholds (upper bound, per h)

| region | h=1 | h=3 | h=6 | h=12 | h=24 | h=72 |
|---|---|---|---|---|---|---|
| DE (θ_p,θ_r) | (272.47, 267.61) | (272.30, 267.60) | (272.30, 267.60) | (271.76, 268.52) | (271.76, 268.52) | (407.47, 159.35) |
| DE S_oracle | 43.32 | 43.26 | 43.04 | 42.31 | 39.91 | 32.67 |
| IT (θ_p,θ_r) | (246.70, 230.45) | (246.34, 231.08) | (240.67, 236.84) | (239.70, 236.46) | (244.46, 242.03) | (349.81, 128.13) |
| IT S_oracle | 32.56 | 32.35 | 32.13 | 31.44 | 29.32 | 25.50 |
| SE (θ_p,θ_r) | (18.18, 17.51) | (18.18, 17.51) | (18.18, 17.51) | (18.18, 17.51) | (18.67, 18.67) | (36.92, 14.91) |
| SE S_oracle | 23.09 | 22.92 | 22.68 | 22.17 | 20.31 | 12.93 |

The oracle itself recovers only **32 % (DE), 48 % (IT), 30 % (SE)** of the h=72 loss
`(S_oracle − S_naive)/(S0_FF − S_naive)` at h=72: **the h=72 staleness loss is largely
structural, unreachable by any static threshold policy.** The required oracle thresholds at h=72
(θ_p 349–407, θ_r 128–159) are far beyond what the AR(1) interval widening (margins 5.7–35.8 g/kWh
at c=1) can express.

> **CORRECTED (per adversarial review M1).** The claim "far beyond what the AR(1) interval widening
> can express" is FALSE: the margin rule is *linear in c* (`margin(h) = c·σ*·scale(h)`), so it can
> express any margin. A deterministic recovery-vs-c probe on the completed 2025 test year
> (`adaptive_sensitivity_DE.json`) shows the oracle scale is reachable: DE h=72 c=0.75 → S=27.32,
> c=4 → 28.66, **c=6 → S=31.43 (completed, vs oracle 32.67; closes ≈75 % of the recoverable gap,
> recovery_vs_oracle 0.748)**, c=8 → 36.29 but budget-infeasible. The modest h=72 recovery in §3.2
> is therefore a property of the *chosen* c*=0.75 (and of the c-grid ceiling ≤2 + train-year
> selection), NOT of the rule's expressiveness. The rule's genuine limitations remain: (i) at c≈1
> the model interval under-widens vs the empirical staleness error (DE h=72: model interval
> 8.38σ*≈30.7 g/kWh vs AR(1) h-step RMSE 113.4 g/kWh), and (ii) the *symmetric* widening cannot
> express the oracle's center drift at h=72 (DE center 283.4 g/kWh vs nominal midpoint 270.1).
> The static-oracle ceiling itself (32/48/30 %) is unmodified and verified.

## 4. DTPR benchmark

**Derivation (documented in `adaptive_summary.json`).** DTPR's β is the switching-cost parameter;
we map the checkpoint/restore cycle's CO₂ cost to an equivalent CI margin. A pause+resume cycle
costs `P·(τ_pause + τ_resume)` of energy at the checkpoint's CI. Equating that CO₂ cost to the CO₂
saved by shifting one hour of training energy from the resume level to the pause level gives the
band `2β` with `β = (τ_pause + τ_resume)/3600 · CI_ref`, `CI_ref = reopt nominal θ_p`
(τ_pause=148.8 s, τ_resume=0 s, constants.json). Thresholds = `center ± β` around the reopt nominal
midpoint; decision = hourly-aggregated (12×5-min means) version of the same arma(h) signal.

| region | β (g/kWh) | center | DTPR (θ_p, θ_r) | margin 2β | outcome on test year |
|---|---|---|---|---|---|
| DE | 11.26 | 270.05 | (281.31, 258.79) | 22.52 | completes; savings ≈ naive (−0.1…−0.4 pp) |
| IT | 10.20 | 238.57 | (248.77, 228.38) | 20.39 | budget-exhausted at all h (overhead 200%) |
| SE | 0.75 | 17.84 | (18.60, 17.09) | 1.50 | completes; savings ≈ naive (−0.7…−0.1 pp) |

> **(nit n4, fixed).** The IT DTPR row's savings (33.42–34.15 %, `adaptive_IT.json`) *exceeds* the
> perfect-foresight S₀ (32.67 %) because the run is incomplete; the above-perfect savings is an
> **incompleteness artifact** (phase-B.0 inflation) and must not be read as genuine savings anywhere
> DTPR is compared. The `*` marker and `recovery_dtpr` = 0 already encode this; this note makes it
> explicit.

**Positioning (theory anchor).** DTPR/OPR (Lechowicz et al., POMACS '23) proves optimality of
double-threshold online pause/resume with a switching cost; we make **no novelty claim about double
thresholds**. Our contribution remains: staleness-aware *adaptive* margin tied to the AR(1)
prediction interval, LLM-pretraining realism (minutes-scale checkpoint cost enters β), 5-min
realized-vs-forecast evaluation, and design rules. Here DTPR serves as the fixed-separation
baseline: its checkpoint-cost-derived β is *completion-infeasible for IT* (already at 194.7 %
overhead at nominal) and *approximately neutral for DE/SE*. The lesson: constant margins derived
from switching cost do not carry a completion-safety check; the adaptive rule's c-selection (with
the completion guard) at least chooses margins that complete where possible.

## 5. Interpretation (honest headline)

1. **Baselines are exact.** naive-fixed reproduces `fixed_summary.json` arma savings to 0.0000 pp
   at all 18 (region, h) cells; perfect-foresight reproduces the control S₀ (DE 43.3510,
   IT 32.6655, SE 22.8685) to 0.0000 pp. The evaluation harness is faithful.
2. **The completion constraint dominates B.1** (phase B.0 caveat confirmed in closed loop):
   IT's operating point already sits at 194.7 % overhead, so within the train-year c-grid of
   interest (c ≤ 2) any material margin widening is budget-infeasible on the test year (h=1: only
   c ≤ 2 completes; h=3: c ≤ 1.5; h=6: c ≤ 0.75; h=12: c ≤ 0.75 in the material range; h=24/72:
   c ∈ {0.25,…,2} all budget-exhausted; DTPR β is also infeasible). Aggressive widening "recovers"
   savings only by exhausting the budget and not completing — exactly the phase-B.0 inflation
   artifact.
   > **(note, for accuracy).** IT feasibility is *non-monotone* in c at h≥12: some very large c
   > (e.g. h=72 c=6 → S=24.39, O=176.9, completed, recovery_vs_oracle 0.835) complete because an
   > extreme band suppresses almost all state switching and hence most checkpoint overhead. These
   > cells are outside the train-year c-grid (≤2) and were never selected; the budget-bound reading
   > applies to the material-widening regime c ∈ {0.25,…,2}.

   **Feasible-c envelope (per region, per h; from `adaptive_sensitivity_{region}.json`, all cells
   computed on the completed 2025 test year):**

   | region | h=1 | h=3 | h=6 | h=12 | h=24 | h=72 |
   |---|---|---|---|---|---|---|
   | DE | ≤ 8 | ≤ 8 | ≤ 8 | ≤ 8 | ≤ 8 | ≤ 6 (c=8 → 36.29, infeasible) |
   | IT | ≤ 2 | ≤ 1.5 | ≤ 0.75 | {0..0.75, 8} | {0,0.25,6,8} | {0,3,4,6} |
   | SE | ≤ 6 | ≤ 3 | ≤ 2 | ≤ 1.5 | ≤ 1 | ≤ 0.75 (no feasible c recovers: c=0.75 → S=2.62 ≪ naive 8.64) |

   > **(corrected per adversarial review m1).** The earlier draft said "SE completes only up to
   > c=0.75 on 2025"; the committed machinery's envelope is finer (SE h=1 completes through c=6,
   > h=3 through 3, h=6 through 2, h=12 through 1.5 — consistent with the committed c*=1.5 rows in
   > `adaptive_SE.json` completing at h=1,3,6,12). The review's recollection ("SE completes only up
   > to c=1 at h=1,3,6,12") is not supported by the committed machinery; the traceable envelope above
   > is authoritative. The review's substantive point stands and is now explicit: **the train-selected
   > SE c*=1.5 is not budget-feasible on 2025 at h≥24**, and **no feasible c recovers SE h=72**
   > (best feasible c=0.75 → S=2.62, far below naive 8.64).
3. **The adaptive rule recovers the recoverable fraction; `recovery_vs_oracle` is the headline
   controller metric** (m4). DE (the region with budget room) recovers 69 % / 35 % of the (small)
   h=1/3 loss and closes **95 % / 80 % of the naive→oracle gap** there (`recovery_vs_oracle` 0.954 /
   0.799). `recovery` (vs perfect-foresight) is dominated by the structural, oracle-unreachable share
   of the loss; `recovery_vs_oracle` (fraction of the *recoverable* gap closed) is the informative
   controller-quality number. The paper's D.7 and C.9 figure should lead with `recovery_vs_oracle`
   and report both.
4. **The h=72 residual decomposes into three distinct shares** (M2). Report the c*-specific recovery
   *and* the c-sensitivity envelope:
   (a) **Static-oracle-structural share** — unreachable by *any static-threshold policy*: the static
   oracle itself reaches only 32 % / 48 % / 30 % of the h=72 loss for DE/IT/SE (S=32.67 / 25.50 /
   12.93 vs perfect-foresight 43.35 / 32.67 / 22.87). No static-threshold policy can exceed this.
   (b) **Budget-bound share** — IT at 200 % (nominal overhead 194.7 %, ~5 pp headroom; c≤2
   material widening infeasible at h≥3; DTPR β infeasible) and SE at h≥24 (c*=1.5 incomplete).
   (c) **c-selection-fragility share** — DE h=72 *is* recoverable to ≈75 % of the recoverable gap at
   c=6 (S=31.43, completed), but the train-year c-selection (grid ceiling ≤ 2; mean-recovery signal
   ≈ 0 at large c — see §3.1) cannot find that c*. SE h=72 is *not* recoverable at any feasible c
   (budget-bound + static-oracle-structural together).
5. **Design-rule output for the paper (D.7):** report the recovery curves (both metrics), the
   recovery-vs-c envelope, the completion-safety finding (budget headroom is a prerequisite for
   adaptive widening; IT has none at 200 %), the static-oracle ceiling (unreachable by any
   static-threshold policy), the c-selection fragility (train-year signal cannot pick the c≈6–7 that
   would approach the oracle), and the empirical-vs-theoretical interval gap (at c≈1 the model
   interval under-widens; symmetric widening cannot express the oracle's h=72 center drift) as
   limitations. The phase-A claim "recovers most of the loss" must be *calibrated*: the controller
   recovers most of the *recoverable* loss within the grace region (95/80 % of the naive→oracle gap
   at DE h=1/3); the residual at large h is (i) unreachable by any static-threshold policy,
   (ii) budget-bound for IT/SE at 200 % overhead, and (iii) further limited by a c-grid whose
   train-year signal is near-zero.

## 6. DoD checklist — pass/fail with evidence

- **D1 `pnpm build` + `pnpm test`** — PASS. `pnpm build` → `✓ built in 2.44s`; `pnpm test` →
  `14 files, 201 tests passed` (177 pre-existing + 9 margin + 15 adaptive).
- **D2 `adaptive-margin.test.ts` asserts (i)–(v) and passes in isolation** — PASS.
  `npx vitest run src/domain/adaptive-margin.test.ts` → `9 passed`. Asserts monotonicity in h,
  margin(0)=0, c=0⇒nominal, DE h=72 numeric (8.3823·σ*≈30.7), midpoint preservation, plus
  finite-sum identity and φ=1 limit.
- **D3 domain files unmodified** — PASS. `git diff --stat src/domain/{simulation,optimize,
  forecast,result,policy}.ts` → empty (no changes). Only new `src/domain/adaptive-margin.ts`
  added; `src/cli/forecast-sweep.ts` and `src/cli/index.ts` extended (the pre-existing
  modifications to `index.ts`/`optimize.ts` are phase B.0).
- **D4 adaptive artifacts, every (region, h) row complete** — PASS. `adaptive_{DE,IT,SE}.json/csv`
  (6 rows × 3 regions), every row has savings/overhead/score/num_pauses/completed/within_budget
  for naive, adaptive and DTPR, plus perfect, oracle (θ/savings/overhead/score),
  `s0_naive_ff`, `s0_adaptive_ff`, `recovery`, `recovery_raw`, `oracle_gap`, `recovery_dtpr`,
  `recovery_vs_oracle`; `adaptive_summary.json` has methodology + per-region condensed rows.
  Test asserts all fields finite / types correct.
- **D5 baselines reproduce committed data within ±0.5 pp** — PASS, exact (0.0000 pp). naive arma
  rows equal `fixed_{region}.json` arma savings at all 18 cells; perfect-foresight S₀ equals
  `fixed_summary.json` control (DE 43.3510, IT 32.6655, SE 22.8685). Enforced by
  `adaptive-sweep.test.ts`.
- **D6 determinism** — PASS. Two full runs (`pnpm cli forecast-sweep --mode adaptive`) →
  `cmp` IDENTICAL on all 7 artifacts (`adaptive_{DE,IT,SE}.json/csv`, `adaptive_summary.json`).
  Reproducible via `run_adaptive_sweep.sh` (`ADAPTIVE_DETERMINISM_CHECK=1` auto-verifies).
- **D7 oracle for all h + DTPR rows present/deterministic** — PASS. h∈{1,6} oracle rows are the
  committed reopt delay rows (source `reopt_delay`), and a fresh arma(h) optimization matches them
  to 0.000000 pp (delay ≡ arma at these φ); h∈{3,12,24,72} fresh `arma_optimize` rows. DTPR rows
  present with β/θ/center/margin and are deterministic (byte-identical across runs). Tests assert
  both.
- **D8 `phase_reports/phase_b1.md` written** — PASS (this file).

## 7. Deviations / decisions / ambiguities resolved

1. **Completion guard added to recovery and c-selection** (the single most important decision).
   The SPEC's literal recovery formula would reward budget-exhausted runs: for IT and SE at c≥1,
   widening pushes overhead to 200 % and the run stops incomplete with *inflated* savings
   (IT h=12 c=2 → 33.87 %, SE h=72 c=1.5 → 47.66 %). Per the phase-B.0 caveat ("the optimizer needs
   an explicit completion constraint"), headline recovery is set to 0 whenever the adaptive or naive
   run does not complete; the unclipped formula is retained as `recovery_raw`. This makes the
   completion constraint a first-class design rule of the adaptive controller.
   > **(corrected per adversarial review m3).** The guard never erased a genuine completed-run
   > recovery: the only completed IT adaptive run at the train-selected c* (h=1, c=2, O=199.6,
   > S=32.26) is *below* naive (32.51), so its recovery would be 0 even unguarded. The guard only
   > suppresses the inflated, budget-exhausted rows (recovery_raw 0.26–1.89). In short: **the
   > completion guard changes only the budget-exhausted rows; every completed-run recovery is
   > identical with or without the guard** (on every completed row in `adaptive_{region}.json` the
   > guard adds nothing — where `recovery` differs from `recovery_raw` on a completed row, the
   > difference is clipping of the raw formula to [0,1], not the guard).
2. **Train-year c-selection signal is weak/non-robust.** Mean completed-gated recovery is
   ≤0.056 everywhere; IT c*=2 is driven by one 2024 outlier cell; 2022/2023 contribute ~0. The
   chosen c* does not robustly transfer to the test year (IT/SE adaptive at c* is
   budget-exhausted at h≥3/24 on 2025). Reported honestly; the paper should present the full
   mean-recovery-by-c table rather than only c*.
3. **Oracle for h∈{1,6} sourced from committed reopt delay rows** per the task's allowance, and
   verified equal to a fresh arma(h) optimization to 0.000000 pp (all regions, both h) — so the
   arma/delay distinction is immaterial at these φ.
4. **DTPR β derivation.** Used the "margin that equates switching cost to pause savings" option:
   β = (τ_pause+τ_resume)/3600 · θ_p_nominal (per-hour amortization). The alternative (per-5-min
   step: β = (τ/300s)·θ_p) gives ~12× larger, even less feasible bands; both documented, the
   per-hour one reported. τ from `constants.json` (148.8 s / 0 s), honoring the CLI `--ckpt-pause/
   --ckpt-resume` overrides when provided.
5. **SE known-error respected.** Naive-fixed SE uses published (19,18) → S₀=22.87 %
   (reproduced exactly), distinct from the reopt nominal (18.18, 17.51) → 23.17 %; the adaptive
   policy is anchored to the reopt nominal as instructed. The recovery denominator uses S0_FF of
   the *naive* policy (22.87); the adaptive ceiling (23.17) is reported as `s0_adaptive_ff` to keep
   the two honest.
6. **Recovery denominator convention.** `recovery = 0` when `S0_FF − S_naive ≤ 1e-3 pp` (incl.
   SE h=1 where arma(1) naive 23.09 slightly *exceeds* the identity control 22.87 — a real, small
   effect of the damped AR decision, not an error).
7. **Determinism proof not inside `pnpm test`.** A two-run `cmp` needs the ~1.5 min CLI; the
   committed runner script performs it on demand (`ADAPTIVE_DETERMINISM_CHECK=1`), and I verified
   byte-identical output manually (D6 evidence above).
8. **`score` uses the α=1 score** (savings-normalized, budget 200 %) as in the reopt/fixed paths;
   budget is 200 % throughout for consistency with committed artifacts.

## 8. Exact commands

```bash
# B.1.1 tests (isolated)
pnpm test src/domain/adaptive-margin.test.ts          # 9 passed

# B.1.2 + B.1.3 sweep (deterministic)
pnpm cli forecast-sweep --mode adaptive -o publication/output/forecast/adaptive
# or via the committed runner:
bash publication/output/forecast/run_adaptive_sweep.sh
# determinism (manual, verified PASS):
mkdir -p /tmp/d6_snap && cp publication/output/forecast/adaptive_{DE,IT,SE}.json \
  publication/output/forecast/adaptive_{DE,IT,SE}.csv \
  publication/output/forecast/adaptive_summary.json /tmp/d6_snap/
pnpm cli forecast-sweep --mode adaptive -o publication/output/forecast/adaptive --quiet
cmp /tmp/d6_snap/adaptive_DE.json publication/output/forecast/adaptive_DE.json   # (and IT, SE, .csv x3, summary) → IDENTICAL

# recovery-vs-c sensitivity (adversarial review M1/M2/m1; extended c grid 0..8)
pnpm cli forecast-sweep --mode adaptive-sensitivity -o publication/output/forecast/adaptive_sensitivity
# or via the committed runner (ADAPTIVE_SENSITIVITY_DETERMINISM_CHECK=1 auto-verifies):
bash publication/output/forecast/run_adaptive_sensitivity.sh

# artifact regression tests
pnpm test src/cli/adaptive-sweep.test.ts               # 19 passed (15 + 4 sensitivity)

# full verification
pnpm build && pnpm test                                # 14 files, 205 tests passed
```

> **(nits n1/n2/n3, fixed).** (n1) delay(h) ≡ arma(h) is stated "exact to 6 decimals" (D7a asserts
> 6-decimal equality), not "within tolerance". (n2) All CSV columns are formatted with
> `toPrecision(6)`; any claim↔CSV trace must use the JSON artifacts, which carry full numeric
> precision (see `adaptive_summary.json` methodology `reproducibilityNote`). (n3) the summary's
> `generated` key is renamed `generatedDate` (still a date, not a timestamp, to keep the path
> byte-deterministic).

Artifacts added: `src/domain/adaptive-margin.ts`, `src/domain/adaptive-margin.test.ts`,
`src/cli/adaptive-sweep.test.ts`, extended `src/cli/forecast-sweep.ts` (+`--mode adaptive`,
+`--mode adaptive-sensitivity`) and `src/cli/index.ts` (help),
`publication/output/forecast/adaptive_{DE,IT,SE}.json/.csv`,
`publication/output/forecast/adaptive_summary.json`,
`publication/output/forecast/run_adaptive_sweep.sh`,
`publication/output/forecast/adaptive_sensitivity_{DE,IT,SE}.json`,
`publication/output/forecast/adaptive_sensitivity_summary.json`,
`publication/output/forecast/run_adaptive_sensitivity.sh`. Nothing committed;
`publication/ICREC_Rome/` untouched; the five protected domain files unmodified.
