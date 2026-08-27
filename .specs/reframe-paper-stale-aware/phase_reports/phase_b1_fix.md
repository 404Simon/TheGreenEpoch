# Phase B.1 Fix Report — Stale-aware adaptive controller (adversarial review fixes)

- Date: 2026-08-19
- Agent: FIX agent, Phase B.1 (`reframe-paper-stale-aware`)
- Input worklist: `.specs/reframe-paper-stale-aware/phase_reports/review_b1.md` (M1, M2, m1–m4, n1–n4)
- Rule honored: **simulation/domain code and committed experiment numbers are correct — no result
  changed.** Only a new sensitivity artifact, wording, and interpretation were added/corrected.

## 1. What changed

| File | Change |
|---|---|
| `src/cli/forecast-sweep.ts` | New `--mode adaptive-sensitivity` (deterministic per-(region,c,h) sweep on test year 2025, extended c grid {0,…,8}, same arma(h)/reopt-anchor/budget-200 %/completion-guard conventions as `--mode adaptive`); new pure function `runAdaptiveSensitivityRegion` + `adaptiveSensitivityCli`; reworded `adaptiveRuleLimitation`, `recoveryVsOracle`, `completionWarning`/`recovery`, `staticOracle`, added `dtprNote` + `reproducibilityNote`, renamed `generated`→`generatedDate`; mode dispatch. |
| `src/cli/index.ts` | Help text: added `adaptive-sensitivity` mode + `--c-sensitivity` option. |
| `src/cli/adaptive-sweep.test.ts` | +4 regression tests: F1a (artifact complete for 3 regions × 11 c × 6 h), F2 (DE h=72 c=6 money cell ≈ 31.43, completed, recVsOracle ≈ 0.75; c=8 infeasible), F1b (feasible-c envelope; SE h=72 no feasible c; SE train c*=1.5 infeasible at h≥24; IT h=1 ≤ c=2; DE through c=8 at grace horizons), F1c (runner script exists). |
| `publication/output/forecast/adaptive_sensitivity_{DE,IT,SE}.json` | **New** sensitivity artifacts (primary). |
| `publication/output/forecast/adaptive_sensitivity_summary.json` | **New** combined summary (oracles, feasibleC, condensed rows, methodology). |
| `publication/output/forecast/run_adaptive_sensitivity.sh` | **New** committed runner (`ADAPTIVE_SENSITIVITY_DETERMINISM_CHECK=1` auto-verifies). |
| `publication/output/forecast/adaptive_summary.json` | Regenerated via the CLI; **all numeric rows byte-identical to the committed artifact** (verified with `cmp` on the region JSONs and an automated numeric-field comparison of the summary). Only methodology text changed. |
| `.specs/.../phase_reports/phase_b1.md` | Corrections marked inline with `> **CORRECTED (per adversarial review …)**` / `> (… fixed)` notes; no history silently rewritten. |

Protected domain files **unmodified** (verified, §5 below).

## 2. Corrections applied (review findings → fix)

### M1 (MAJOR) — "h=72 loss cannot be expressed by the margin rule" is false
- `phase_b1.md` §3.3 last sentence + executive verdict: corrected with the linear-in-c argument and
  the recovery-vs-c probe result (DE h=72 c=6 → S=31.43, completed, recovery_vs_oracle 0.748;
  c=8 → 36.29 budget-infeasible). Rule limitations narrowed to the two that are real: (i) at c≈1
  the model interval under-widens vs empirical staleness error; (ii) symmetric widening cannot
  express the oracle's h=72 center drift (DE center 283.4 vs nominal midpoint 270.1 g/kWh).
- `adaptive_summary.json` `adaptiveRuleLimitation` reworded to the same effect (M1 wording),
  stating the c-grid ceiling (≤2) + near-zero/noisy train-year signal prevented selecting c≈6–7.

### M2 (MAJOR) — "recovery ≈ 0 at h≥6" is c*-specific, not a rule property
- `phase_b1.md` §3.2 (correction note) + §5.3: h=72 residual now decomposed into three shares
  (see §3 below); the §3.2 rows are labelled as being at the train-selected c*, and the c-sensitivity
  envelope is reported alongside.

### m1 — feasible-c envelope (esp. SE)
- New `feasibleC[h]` per region in the sensitivity artifacts; table added to `phase_b1.md` §5.2.
- Explicit statement: **the train-selected SE c*=1.5 is not budget-feasible on 2025 at h≥24**, and
  **no feasible c recovers SE h=72** (best feasible c=0.75 → S=2.62 ≪ naive 8.64).
- *Discrepancy note:* the review's recollection ("SE completes only up to c=1 at h=1,3,6,12") is
  not what the committed machinery computes (SE h=1 completes through c=6, h=3 through 3, h=6
  through 2, h=12 through 1.5 — consistent with the committed c*=1.5 rows completing at h=1,3,6,12
  in `adaptive_SE.json`). The traceable computed envelope is authoritative and is what the report
  and paper should use.

### m2 — "structural" → "unreachable by any static-threshold policy"
- Applied in `phase_b1.md` (§3.3, §5.3, §5.5) and `adaptive_summary.json` (`adaptiveRuleLimitation`,
  `staticOracleCeiling` in the sensitivity summary).

### m3 — completion guard never erased a genuine completed-run recovery
- Sentence added to `phase_b1.md` §7.1 and `adaptive_summary.json` `recovery`:
  "the completion guard changes only the budget-exhausted rows; every completed-run recovery is
  identical with or without the guard." Verified programmatically on all 3×11×6 sensitivity rows
  (completed rows: `recovery == clip(recovery_raw)`; incomplete rows: `recovery == 0`).

### m4 — headline metric = `recovery_vs_oracle`
- Stated in `phase_b1.md` §5.3, `adaptive_summary.json` `recoveryVsOracle` ("HEADLINE controller
  metric", DE h=1 0.95, h=3 0.80), and the sensitivity methodology. Both metrics remain defined.

### n1–n4
- n1: "verified within tolerance" → "exact to 6 decimals" (`adaptive_summary.json` `staticOracle`,
  `phase_b1.md` §2).
- n2: CSV `toPrecision(6)` → trace via JSON, stated in `adaptive_summary.json`
  `reproducibilityNote` + `phase_b1.md` §8.
- n3: `generated` → `generatedDate` in `adaptive_summary.json` + sensitivity summary (still a date
  to keep the path byte-deterministic).
- n4: IT DTPR "above-perfect savings is an incompleteness artifact" stated in
  `adaptive_summary.json` `dtprNote` + `phase_b1.md` §4.

## 3. New sensitivity artifact

- Generator: `forecast-sweep --mode adaptive-sensitivity` (deterministic; `cmp`-identical across runs).
- Scope: test year **2025**, h ∈ {1,3,6,12,24,72}, c ∈ **{0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8}**,
  budget 200 %, decision = arma(h), reopt nominal anchors, static-oracle ceiling per h
  (h∈{1,6} from committed reopt delay rows; h∈{3,12,24,72} freshly optimized).
- Per (region, c, h) cell: savings, overhead, score, num_pauses, completed, within_budget,
  `recovery` (completion-guarded), `recovery_raw`, `recovery_vs_oracle`, plus
  savings_naive/perfect/oracle and the per-h oracle ceiling row.

### 3.1 DE h=72 — recovery-vs-c (naive 27.73, oracle 32.67, perfect 43.35) — the money cell

| c | S (%) | O (%) | completed | recovery | recovery_vs_oracle |
|---|---|---|---|---|---|
| 0 | 27.82 | 174.8 | yes | 0.005 | 0.017 |
| 0.25 | 27.61 | 176.0 | yes | 0.000 | 0.000 |
| 0.5 | 27.47 | 176.2 | yes | 0.000 | 0.000 |
| **0.75 (c*)** | 27.32 | 176.2 | yes | 0.000 | 0.000 |
| 1 | 27.23 | 176.0 | yes | 0.000 | 0.000 |
| 1.5 | 27.34 | 173.8 | yes | 0.000 | 0.000 |
| 2 | 27.21 | 176.2 | yes | 0.000 | 0.000 |
| 3 | 27.57 | 173.2 | yes | 0.000 | 0.000 |
| 4 | 28.66 | 172.1 | yes | 0.059 | 0.187 |
| **6** | **31.43** | 165.7 | **yes** | **0.237** | **0.748** |
| 8 | 36.29 | 200.0 | no | 0.000 | 0.000 |

**Money cell (F2):** DE h=72, c=6 → `savings = 31.4279703…`, `completed = true`, `within_budget = true`,
`recovery = 0.2365`, `recovery_vs_oracle = 0.7483`. The oracle at h=72 is S=32.67 (θ 407.47/159.35);
the adaptive rule at c=6 closes **≈75 % of the recoverable (naive→oracle) gap**. c=8 → 36.29 is
budget-infeasible (O=200.0, incomplete). This matches the review's independent probe exactly.

### 3.2 IT h=72 — budget-bound with non-monotone large-c completions
c ∈ {0.25,…,2} → O=200.0, incomplete (the material-widening regime is budget-infeasible);
c=3 → S=20.55 completed (recVsO 0.264), c=4 → 22.44 (0.545), c=6 → **24.39** completed
(recVsO 0.835). Extreme c completes because a huge band suppresses nearly all state switching and
hence most checkpoint overhead. These cells lie outside the train-year c-grid (≤2); the budget-bound
reading holds for the material-widening regime and for the train-selected controller.

### 3.3 SE h=72 — unrecoverable at any feasible c
| c | S (%) | O (%) | completed |
|---|---|---|---|
| 0 | 8.64 | 105.7 | yes |
| 0.25 | 6.61 | 129.3 | yes |
| 0.5 | 7.05 | 121.9 | yes |
| 0.75 | **2.62** | 142.7 | yes |
| 1 | 9.85 | 200.0 | no |
| ≥1.5 | 47.66+ | 200.0 | no |

Best feasible c=0.75 → S=2.62 ≪ naive 8.64. No feasible c recovers SE h=72.

### 3.4 Feasible-c envelope (per region, per h; completed AND within budget)
| region | h=1 | h=3 | h=6 | h=12 | h=24 | h=72 |
|---|---|---|---|---|---|---|
| DE | ≤8 | ≤8 | ≤8 | ≤8 | ≤8 | ≤6 |
| IT | ≤2 | ≤1.5 | ≤0.75 | {0..0.75, 8} | {0,0.25,6,8} | {0,3,4,6} |
| SE | ≤6 | ≤3 | ≤2 | ≤1.5 | ≤1 | ≤0.75 |

## 4. h=72 residual — three-share decomposition (F4; guidance for paper D.7)

- **(a) Static-oracle-structural share** — unreachable by *any static-threshold policy*: the static
  oracle at h=72 reaches 32 % / 48 % / 30 % of the loss for DE/IT/SE (S=32.67/25.50/12.93 vs
  perfect-foresight 43.35/32.67/22.87).
- **(b) Budget-bound share** — IT at 200 % (nominal O=194.7 %, ~5 pp headroom; material widening
  infeasible in the train c-grid); SE at h≥24 (c*=1.5 incomplete on 2025).
- **(c) c-selection-fragility share** — DE h=72 recoverable to ≈75 % of the recoverable gap at c=6
  (S=31.43, completed), but the train-year c-selection (grid ceiling ≤2, near-zero/noisy signal)
  cannot find that c*. SE h=72 is not recoverable at any feasible c (shares (a)+(b) combine).

## 5. DoD checklist — pass/fail with evidence

- **F1 Sensitivity artifact exists for all 3 regions, extended c grid, all 6 h; deterministic; reproducible.**
  PASS. `adaptive_sensitivity_{DE,IT,SE}.json` + `_summary.json` exist; each region has
  `cGrid = [0,0.25,0.5,0.75,1,1.5,2,3,4,6,8]`, `horizons = [1,3,6,12,24,72]`, one row per (c,h)
  (198 cells/region). Determinism: two independent CLI runs → `cmp` IDENTICAL on all 4 files;
  committed runner `run_adaptive_sensitivity.sh` with `ADAPTIVE_SENSITIVITY_DETERMINISM_CHECK=1`
  prints "Determinism check PASSED".
- **F2 DE h=72 c=6 (S≈31.4, completed) present and reproduced by a test.**
  PASS. Row present: `savings=31.4279703`, `completed=true`, `within_budget=true`,
  `recovery_vs_oracle=0.7483`. Regression test "F2: DE h=72 c=6 …" asserts `toBeCloseTo(31.43,1)`,
  `completed`, `within_budget`, recVsOracle∈(0.6,0.9), and c=8 infeasible — passes.
- **F3 `adaptiveRuleLimitation` (and related text) reworded per M1; phase_b1.md over-claims corrected and marked.**
  PASS. Summary text now says "linear in c … property of the CHOSEN c* … c-grid ceiling (≤2) …
  c≈6–7 … c=8 budget-infeasible … center drift … unreachable by any static-threshold policy".
  Region JSONs byte-identical; summary numeric rows identical (automated field comparison).
  `phase_b1.md` corrections marked with `> **CORRECTED (per adversarial review M1/M2/m1/m3)**`.
- **F4 h=72 residual decomposed into three shares with numbers; feasible-c envelope reported (esp. SE).**
  PASS. §4 above; feasible envelope in §3.4 (SE h=72 ≤0.75, no recovery; SE c*=1.5 infeasible at h≥24).
- **F5 Headline metric = `recovery_vs_oracle`; completion-guard non-effect sentence added.**
  PASS. `adaptive_summary.json` `recoveryVsOracle` marked "HEADLINE controller metric" with DE h=1/3
  values; completion-guard sentence in `recovery` + `phase_b1.md` §7.1; guard property verified on
  all 594 sensitivity rows.
- **F6 `pnpm build` + `pnpm test` green; protected domain files unmodified; determinism re-verified; no new over-claims; numbers traceable.**
  PASS. `pnpm build` → "✓ built in 2.62s"; `pnpm test` → `14 files, 205 tests passed` (201 + 4 new).
  `git diff --stat src/domain/{simulation,optimize,forecast,result,policy}.ts` → empty.
  Both runners' determinism checks PASSED. Every number in this report traces to
  `adaptive_sensitivity_*.json`, `adaptive_summary.json`, `adaptive_{region}.json`,
  `fixed_summary.json`/`reopt_summary.json`, or `calibration_*.json`.
- **F7 `phase_reports/phase_b1_fix.md` written with DoD evidence.**
  PASS (this file).

## 6. Exact commands

```bash
# sensitivity artifact (deterministic; runner includes auto determinism check)
bash publication/output/forecast/run_adaptive_sensitivity.sh          # or:
ADAPTIVE_SENSITIVITY_DETERMINISM_CHECK=1 bash publication/output/forecast/run_adaptive_sensitivity.sh

# adaptive sweep re-run after summary rewording (numeric rows byte-identical)
ADAPTIVE_DETERMINISM_CHECK=1 bash publication/output/forecast/run_adaptive_sweep.sh

# regression tests (19 = 15 + 4 sensitivity)
pnpm test src/cli/adaptive-sweep.test.ts

# full verification
pnpm build
pnpm test          # 14 files, 205 tests passed

# protected domain files check
git diff --stat src/domain/simulation.ts src/domain/optimize.ts src/domain/forecast.ts src/domain/result.ts src/domain/policy.ts   # empty
```

Nothing committed. `publication/ICREC_Rome/` untouched. The five protected domain files unmodified.
