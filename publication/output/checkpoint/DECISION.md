# DECISION — Phase B.0: checkpoint-realism sweep (Stale-Aware Hysteresis Control for Carbon-Aware LLM Pretraining)

- Date: 2026-08-19
- Scope: sweep of `checkpointPauseTime` ∈ {148.8, 150, 900, 2700} s (resume 0) for the headline
  optimization of DE / IT / SE (DeepSeek profile, 2025 data, reopt optimizer settings:
  resolution 10, iterations 6, budget 200 %, α=1, fixed per-region start, tpMax 800/800/100).
- Every number below is traceable to `publication/output/checkpoint/checkpoint_summary.json`
  (region `baseline` = committed `reopt_summary.json` values; `runs[]` per ckpt value).

## 1. Recorded results (optimizer `best` as-is)

The optimizer returns the max-score point among {within-budget, savings>0} (no completion
constraint). `completed=false` rows are **budget-blocked, incomplete runs**: the job did not
finish training within the 200 % overhead budget, so their reported `savings %` is inflated
(savings are computed against a full-run baseline). They are recorded faithfully below and
interpreted in §3.

### 1.1 DE (start 02-01, tpMax 800)

| ckpt (s) | θ_p | θ_r | margin | savings % | overhead % | score | pauses | completed | ΔS pp vs 148.8 | ΔS rel vs 148.8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 148.8 | 272.37 | 267.73 | 4.64 | 43.35 | 174.31 | 0.7168 | 102 | yes | 0.00 | 0.0 % |
| 150   | 272.37 | 267.73 | 4.64 | 43.35 | 174.31 | 0.7167 | 102 | yes | +0.00 | 0.0 % |
| 900   | 246.93 | 234.80 | 12.13 | 47.91 | 199.98 | 0.7395 | 112 | **no** | −4.56* | −10.5 %* |
| 2700  | 170.58 | 138.58 | 32.00 | 82.18 | 199.98 | 0.9109 | 15 | **no** | −38.83* | −89.6 %* |

\* Incomplete run; see §3.

### 1.2 IT (start 01-14, tpMax 800)

| ckpt (s) | θ_p | θ_r | margin | savings % | overhead % | score | pauses | completed | ΔS pp vs 148.8 | ΔS rel vs 148.8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 148.8 | 246.70 | 230.45 | 16.25 | 32.67 | 194.68 | 0.6633 | 173 | yes | 0.00 | 0.0 % |
| 150   | 246.70 | 230.45 | 16.25 | 32.66 | 194.68 | 0.6633 | 173 | yes | +0.00 | 0.0 % |
| 900   | 209.04 | 172.41 | 36.63 | 70.56 | 200.00 | 0.8528 | 55 | **no** | −37.89* | −116.0 %* |
| 2700  | 182.69 | 170.06 | 12.63 | 73.05 | 200.00 | 0.8652 | 59 | **no** | −40.38* | −123.6 %* |

\* Incomplete run; see §3.

### 1.3 SE (start 04-22, tpMax 100)

| ckpt (s) | θ_p | θ_r | margin | savings % | overhead % | score | pauses | completed | ΔS pp vs 148.8 | ΔS rel vs 148.8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 148.8 | 18.18 | 17.51 | 0.67 | 23.17 | 106.34 | 0.6158 | 85 | yes | 0.00 | 0.0 % |
| 150   | 18.18 | 17.51 | 0.67 | 23.17 | 106.34 | 0.6158 | 85 | yes | +0.00 | 0.0 % |
| 900   | 18.18 | 17.51 | 0.67 | 22.06 | 106.34 | 0.6103 | 85 | yes | +1.11 | +4.8 % |
| 2700  | 21.81 | 16.64 | 5.17 | 18.87 | 105.11 | 0.5943 | 61 | yes | +4.30 | +18.6 % |

SE remains feasible (and *completed*) at every checkpoint time. ΔS pp/rel shown as +loss
(positive = loss). Note the 148.8-s rows reproduce the committed `reopt_summary.json`
baselines exactly (DE 272.37/267.73/43.35 %/174.31 %; IT 246.70/230.45/32.67 %/194.68 %;
SE 18.18/17.51/23.17 %/106.34 %).

## 2. Completed-feasible interpretation (for the verdict)

For a *fair* savings comparison the optimum must both stay within budget **and complete**.
`bestCompleted` (highest-score point with withinBudget ∧ completed ∧ savings>0, from the same
optimization run) per cell:

| region | S₀ (148.8 s) | S(900 s) | S(2700 s) | retention @900 | retention @2700 |
|---|---|---|---|---|---|
| DE | 43.35 % | 42.32 % | 39.60 % | 97.6 % | 91.4 % |
| IT | 32.67 % | 30.90 % | 27.87 % | 94.6 % | 85.3 % |
| SE | 23.17 % | 22.06 % | 18.87 % | 95.2 % | 81.4 % |

A feasible within-budget **and completed** point exists at 2700 s for DE (θ=(293.08, 253.58),
S=39.60 %, O=172.07 %) and IT (θ=(271.70, 226.04), S=27.87 %, O=175.63 %).

## 3. Headline caveat (recorded best degenerates)

At 900/2700 s the raw optimizer `best` for DE and IT is an **incomplete, budget-blocked run**
(completed=false, stopReason=budget_exceeded) whose savings % is inflated against a full-run
baseline (DE 47.91 % → 82.18 %; IT 70.56 % → 73.05 %). This is an artifact of the score
objective (α=1, no completion penalty), not real savings. **Finding:** under minutes-scale
checkpoints the headline objective must carry an explicit completion constraint (or
report `bestCompleted`); this is a genuine design-rule result and must be stated in the paper
(B.2/B.4 follow-ups should use the completion-constrained optimum).

## 4. AR(1) interval-width framing (context only — not the decision driver)

5-min grid CI is a near-unit-root process (DE lag-1 autocorr = 0.999655, σ*=3.658 g/kWh,
calibration 2022–24 / test 2025). The AR(1) h-step prediction-interval scale at the
decision-stale horizon h=72 steps (6 h):

- one-sided h-step σ scale: σ₇₂ = σ*·sqrt((1−φ^(2h))/(1−φ²)) = 3.658 · sqrt((1−0.999655^144)/(1−0.999655²)) = **8.38·σ\* ≈ 30.7 g/kWh ≈ 31 g/kWh** — this is the SPEC's `width(72) ≈ 8.4σ*` figure, confirmed.
- two-sided symmetric interval (full width, formula `2·σ*·sqrt((1−φ^(2h))/(1−φ²))`): **61.3 g/kWh** (exactly 2× the one-sided scale).

I.e. even a 6-hour-stale decision faces AR(1) prediction-interval scales of order 30–60 g/kWh
on a signal whose *optimized thresholds* sit ~4 g/kWh apart (DE margin 4.64) — one motivation
for the stale-aware margin-widening controller (B.1). This framing does **not** drive the B.0
verdict; it is provided for the paper's motivation/context.

## 5. VERDICT — Story A ("works-with-design-rules")

SPEC rule (B.0.4): **Story A** if DE and IT both keep ≥ ~2/3 of their 148.8-s savings at
900 s AND the optimizer still finds a feasible within-budget point at 2700 s (DE and IT);
**Story B** otherwise.

| Criterion | Evidence | Pass |
|---|---|---|
| DE keeps ≥ 2/3 of S₀ at 900 s | completed-feasible 97.6 % (42.32/43.35); recorded best even higher | ✅ |
| IT keeps ≥ 2/3 of S₀ at 900 s | completed-feasible 94.6 % (30.90/32.67); recorded best even higher | ✅ |
| Feasible within-budget point at 2700 s, DE | found=true; completed point (293.08, 253.58), S=39.60 % ≤ budget | ✅ |
| Feasible within-budget point at 2700 s, IT | found=true; completed point (271.70, 226.04), S=27.87 % ≤ budget | ✅ |

**→ Story A.** Even at a 45-minute full-state checkpoint/restore (2700 s, a realistic 671B-MoE
figure), the completed-feasible optimum retains **91.4 % (DE) / 85.3 % (IT)** of the
148.8-s-savings, and SE remains fully feasible and completed throughout (81.4 % retention at
2700 s). The checkpoint cost does **not** collapse the value proposition; instead it (i) widens
the optimized margin (DE 4.64 → 39.50 at 2700 s), and (ii) forces an explicit completion
constraint into the headline objective. Both are design-rule contributions, consistent with the
"works-with-design-rules" paper.

**Caveats carried forward:** (1) the 900/2700 s recorded `best` rows for DE/IT are incomplete
runs — the paper must report the completed-feasible optimum; (2) baseline savings use
θ_p≈272 (DE) where grid percentiles may imply a different feasible regime at stricter budgets —
the B.4 budget sweep should use ckpt = 900 s or 2700 s (Story-A robustness table) as well as
148.8 s; (3) the AR(1) width framing above has a factor-2 ambiguity between the SPEC's `8.4σ*`
(= one-sided σ₇₂) and the `2·σ*·sqrt(...)` formula (= two-sided full width) — resolved here as
context only.
