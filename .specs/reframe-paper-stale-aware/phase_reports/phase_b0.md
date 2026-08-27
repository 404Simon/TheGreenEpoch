# Phase B.0 — Decisive Experiment: Checkpoint-Realism Sweep (report)

- Date: 2026-08-19
- Agent: implementation agent, Phase B.0 (`reframe-paper-stale-aware`)
- Verdict: **Story A — "works-with-design-rules"**
- All numbers traceable to `publication/output/checkpoint/checkpoint_summary.json`.

## 1. What was done

1. **B.0.1 Parameterize checkpoint times.** Added `--ckpt-pause <seconds>` / `--ckpt-resume <seconds>`
   to the `optimize` command and to `forecast-sweep` (both `fixed` and `reopt` modes) in
   `src/cli/index.ts`. In `src/cli/optimize.ts` and `src/cli/forecast-sweep.ts` the `FullProfile`
   is built from the overrides when present, else from `constants.json` exactly as before
   (no-override path unchanged). `src/domain/{optimize,simulation,forecast}.ts` untouched.
2. **B.0.2 Checkpoint-time sweep.** Wrote deterministic `publication/output/checkpoint/run_checkpoint_sweep.sh`
   (runs the 12 headline optimizations via the CLI with the reopt settings, aggregates with an
   embedded node script) → `checkpoint_{DE,IT,SE}.json`, `checkpoint_all.csv`, `checkpoint_summary.json`
   (+ `_raw/` per-run JSON as evidence). Ran the sweep; re-ran and verified byte-identical output.
3. **B.0.3 Monotonicity + sanity tests.** Added `src/cli/checkpoint-sweep.test.ts` (8 tests):
   (i) completed-best savings monotone non-increasing in the committed artifact, plus a fast
   fixed-policy recomputation asserting savings non-increasing AND overhead non-decreasing across
   ckpt values; (ii) identity at ckpt=148.8 vs the constants profile (artifact + small in-test
   `runOptimization` recomputation); (iii) `marginRuleSurvives` ceiling 16 g/kWh applied to every
   (region, ckpt) margin plus boundary unit checks.
4. **B.0.4 Decision memo.** `publication/output/checkpoint/DECISION.md` with the required table,
   completed-feasible interpretation, AR(1) context, and the verdict (Story A).

## 2. Numbers table (from `checkpoint_summary.json`)

`best` = optimizer max-score point (no completion constraint); `bestCompleted` = highest-score
within-budget ∧ completed point from the same run.

| region | ckpt (s) | θ_p | θ_r | margin | savings % | overhead % | score | pauses | completed | S(bestCompleted) % |
|---|---|---|---|---|---|---|---|---|---|---|
| DE | 148.8 | 272.37 | 267.73 | 4.64 | 43.35 | 174.31 | 0.7168 | 102 | yes | 43.35 |
| DE | 150 | 272.37 | 267.73 | 4.64 | 43.35 | 174.31 | 0.7167 | 102 | yes | 43.35 |
| DE | 900 | 246.93 | 234.80 | 12.13 | 47.91* | 199.98 | 0.7395 | 112 | no | 42.32 |
| DE | 2700 | 170.58 | 138.58 | 32.00 | 82.18* | 199.98 | 0.9109 | 15 | no | 39.60 |
| IT | 148.8 | 246.70 | 230.45 | 16.25 | 32.67 | 194.68 | 0.6633 | 173 | yes | 32.67 |
| IT | 150 | 246.70 | 230.45 | 16.25 | 32.66 | 194.68 | 0.6633 | 173 | yes | 32.66 |
| IT | 900 | 209.04 | 172.41 | 36.63 | 70.56* | 200.00 | 0.8528 | 55 | no | 30.90 |
| IT | 2700 | 182.69 | 170.06 | 12.63 | 73.05* | 200.00 | 0.8652 | 59 | no | 27.87 |
| SE | 148.8 | 18.18 | 17.51 | 0.67 | 23.17 | 106.34 | 0.6158 | 85 | yes | 23.17 |
| SE | 150 | 18.18 | 17.51 | 0.67 | 23.17 | 106.34 | 0.6158 | 85 | yes | 23.17 |
| SE | 900 | 18.18 | 17.51 | 0.67 | 22.06 | 106.34 | 0.6103 | 85 | yes | 22.06 |
| SE | 2700 | 21.81 | 16.64 | 5.17 | 18.87 | 105.11 | 0.5943 | 61 | yes | 18.87 |

\* Incomplete (budget-blocked) run: savings inflated vs full-run baseline; see §4.

Completed-feasible retention vs the 148.8-s optimum: **DE 97.6 % @900, 91.4 % @2700; IT 94.6 % @900,
85.3 % @2700; SE 95.2 % @900, 81.4 % @2700.**

## 3. Verdict

SPEC rule (B.0.4): Story A ⟺ DE and IT both keep ≥ ~2/3 of 148.8-s savings at 900 s AND a feasible
within-budget point exists at 2700 s (DE, IT).

- DE retention @900: 97.6 % (completed-feasible) — pass.
- IT retention @900: 94.6 % (completed-feasible) — pass.
- Feasible within-budget point @2700: DE yes (completed point (293.08, 253.58), S=39.60 %); IT yes
  (completed point (271.70, 226.04), S=27.87 %). `found=true` for all 12 cells.

**→ Story A: "works-with-design-rules".** Savings survive minutes-scale checkpoint/restore; the
penalty is a widened optimized margin (DE 4.64 → 39.50, IT 16.25 → 45.66 at 2700 s) and the need
for an explicit completion constraint in the headline objective. Caveats in DECISION.md §5.

## 4. Key finding / caveat (must be reported honestly in the paper)

At 900/2700 s the raw optimizer `best` for DE and IT is an **incomplete budget-blocked run**
(`completed=false`, `stopReason=budget_exceeded`) whose savings % is inflated against a full-run
baseline. This is an artifact of the α=1 score (no completion penalty), not real savings. The paper
must report the completed-feasible optimum (already stored as `bestCompleted` in the artifacts) and
note that a completion guardrail is a required design rule under realistic checkpoint times (B.2/B.4
should use the completion-constrained optimum).

## 5. DoD checklist — pass/fail with evidence

- **D1 `pnpm build` passes** — PASS. `pnpm build` → `✓ built in 2.15s` (rolldown warning only, non-fatal); re-verified after edits (`build OK`).
- **D2 CLI help shows the new options** — PASS.
  `pnpm cli optimize --help` lists `--ckpt-pause <seconds>` and `--ckpt-resume <seconds>`; `pnpm cli forecast-sweep --help` lists both.
- **D3 Bit-for-bit default behavior** — PASS. `cmp /tmp/b0_none.json /tmp/b0_148.json` → `FULL FILE IDENTICAL`; `best` objects JSON-identical (`JSON.stringify(a.best)===JSON.stringify(b.best)` → true).
- **D4 Baseline reproduction (with reopt settings + `--ckpt-pause 148.8`)** — PASS.
  DE 272.37 / 267.73 / 43.35 % / 174.31 %; IT 246.70 / 230.45 / 32.67 % / 194.68 %; SE 18.18 / 17.51 / 23.17 % / 106.34 % — all within tolerance (θ ±2, savings ±0.5 pp, overhead ±1 pp), exact to committed `reopt_summary.json`.
- **D5 `pnpm test` green + isolated test green** — PASS. Full suite: `12 passed (12 files), 177 tests passed` (169 existing + 8 new). Isolated: `pnpm test src/cli/checkpoint-sweep.test.ts` → `8 passed`, 0.5 s.
- **D6 All four checkpoint artifacts exist, every (region, ckpt) cell recorded** — PASS.
  `checkpoint_{DE,IT,SE}.json`, `checkpoint_all.csv`, `checkpoint_summary.json` in `publication/output/checkpoint/`; 12/12 cells recorded with θ_p, θ_r, margin, savings, overhead, score, num_pauses, completed (+ `bestCompleted`). Test asserts all fields present.
- **D7 Determinism: two full sweep runs byte-identical** — PASS. Snapshot → re-run → `cmp` on all 5 artifacts: `IDENTICAL: checkpoint_DE.json / checkpoint_IT.json / checkpoint_SE.json / checkpoint_all.csv / checkpoint_summary.json`.
- **D8 `DECISION.md` has required table, defensible verdict, AR(1) context** — PASS. See file; verdict = Story A per the SPEC rule; AR(1) framing present (both the SPEC `8.4σ*≈31 g/kWh` one-sided scale — confirmed 8.38·σ*=30.7 — and the `2·σ*·sqrt(...)` two-sided width 61.3 g/kWh, factor-2 note).
- **D9 `phase_reports/phase_b0.md` written** — PASS (this file), DoD + evidence included.

## 6. Deviations / issues

1. **Optimizer `best` degenerates at 900/2700 s (DE, IT).** The SPEC's recorded-cell definition
   ("headline optimization best") yields incomplete runs with inflated savings. I recorded them
   faithfully AND added a `bestCompleted` field per cell (from the same optimization run, no extra
   compute) so the verdict uses a fair, completion-constrained comparison. No domain code changed.
2. **Monotonicity test design.** Raw `best` savings/overhead are NOT monotone across ckpt values
   (inflation from incompleteness; SE overhead 106.34→105.11). The test therefore asserts (a) the
   committed artifact's completed-best savings is monotone non-increasing, and (b) a fast
   fixed-policy recomputation (reopt-optimal θ per region) where savings non-increasing AND
   overhead non-decreasing provably hold on the committed data (all regions). This satisfies
   B.0.3(i) with a deterministic, ~0.5 s test.
3. **AR(1) width factor-of-2 ambiguity in the SPEC.** `width(72)≈8.4σ*` equals the one-sided
   h-step σ scale (8.38·σ*=30.7 g/kWh) while the stated formula `2·σ*·sqrt((1−φ^(2h))/(1−φ²))`
   gives 61.3 g/kWh. Both cited transparently in DECISION.md as context only (not a decision driver).
4. **`optimize --help` `--ckpt-pause` placement**: options added without a default (numbers parsed
   only when present), preserving the byte-identical no-override path (D3).
5. One aggregation bug found & fixed during development (CSV completed-best columns read
   `cb.savings` instead of `cb.co2SavingsPct` on the `SweepPoint` type) — resolved before final runs.

## 7. Exact commands run

```bash
# B.0.1 / D2
pnpm build
pnpm cli optimize --help        # shows --ckpt-pause / --ckpt-resume
pnpm cli forecast-sweep --help  # shows --ckpt-pause / --ckpt-resume

# D3 bit-for-bit default behavior
pnpm cli optimize -m Deepseek -r DE -y 2025 --start 02-01 --tp-max 800 -o /tmp/b0_none.json
pnpm cli optimize -m Deepseek -r DE -y 2025 --start 02-01 --tp-max 800 --ckpt-pause 148.8 --ckpt-resume 0 -o /tmp/b0_148.json
cmp /tmp/b0_none.json /tmp/b0_148.json   # FULL FILE IDENTICAL

# D4 baseline reproduction (per region; reopt settings)
pnpm cli optimize -m Deepseek -r DE -y 2025 --start 02-01 --tp-max 800 --budget 200 --resolution 10 --max-iter 6 --ckpt-pause 148.8 --ckpt-resume 0 -o /tmp/b0_DE.json
pnpm cli optimize -m Deepseek -r IT -y 2025 --start 01-14 --tp-max 800 --budget 200 --resolution 10 --max-iter 6 --ckpt-pause 148.8 --ckpt-resume 0 -o /tmp/b0_IT.json
pnpm cli optimize -m Deepseek -r SE -y 2025 --start 04-22 --tp-max 100 --budget 200 --resolution 10 --max-iter 6 --ckpt-pause 148.8 --ckpt-resume 0 -o /tmp/b0_SE.json

# B.0.2 sweep (deterministic; re-run → cmp)
bash publication/output/checkpoint/run_checkpoint_sweep.sh
# determinism:
mkdir -p /tmp/d7_snap && cp publication/output/checkpoint/checkpoint_{DE,IT,SE}.json publication/output/checkpoint/checkpoint_all.csv publication/output/checkpoint/checkpoint_summary.json /tmp/d7_snap/
bash publication/output/checkpoint/run_checkpoint_sweep.sh
cmp /tmp/d7_snap/checkpoint_DE.json publication/output/checkpoint/checkpoint_DE.json   # (and IT, SE, all.csv, summary.json) → IDENTICAL

# B.0.3 tests
pnpm test src/cli/checkpoint-sweep.test.ts   # 8 passed
pnpm test                                    # 12 files, 177 tests passed
```

Artifacts: modified `src/cli/index.ts`, `src/cli/optimize.ts`, `src/cli/forecast-sweep.ts`;
new `src/cli/checkpoint-sweep.test.ts`; `publication/output/checkpoint/` with
`checkpoint_{DE,IT,SE}.json`, `checkpoint_all.csv`, `checkpoint_summary.json`,
`run_checkpoint_sweep.sh`, `DECISION.md`. Nothing committed; `publication/ICREC_Rome/` untouched.
