# Adversarial Review — Phase B.1 Stale-aware adaptive controller

- Date: 2026-08-19
- Reviewer: adversarial review agent (read-only; no files modified)
- Phase under review: `.specs/reframe-paper-stale-aware/phase_reports/phase_b1.md` (B.1.1–B.1.3)
- Scope: formula fidelity, decision semantics, c-selection, completion guard, oracle, recovery metric, DTPR, reproducibility.

## 1. Executive verdict

**ACCEPT** — the implementation is faithful to the SPEC math, fully deterministic, byte-exact against the committed baselines and artifacts, and the honest caveats (weak c-selection, completion guard, empirical-vs-model interval gap) are documented in the artifacts themselves. All headline numbers I independently recomputed match.

However, **one major interpretive claim is wrong and must be fixed before the paper (D.7)**: the report's statement that the h=72 oracle thresholds are "far beyond what the AR(1) interval widening can express" is false — the margin rule is *linear in c*, and a c≈6–7 margin *does* express the required separation and closes ≈75 % of the recoverable h=72 gap for DE on the completed test year. The modest h=72 recovery is therefore substantially an artifact of the **c-grid ceiling (2) + train-year c-selection**, not of the rule's expressiveness. The *static-oracle* ceiling (32/48/30 % at h=72) is real and verified; the budget-bound IT story is real and verified; but "adaptive recovery ≈ 0 at h≥6" is c*-specific, not structural.

No BLOCKING findings. Two MAJOR (narrative calibration), four MINOR, four NIT. A short fix pass (recovery-vs-c curve + two rewordings) is recommended before the claim–evidence pass; it is not required for the phase's DoD.

## 2. Verified-claims table

| Claim | Verdict | Evidence (numbers I recomputed) |
|---|---|---|
| Margin formula = `c·σ*·sqrt((1−φ^2h)/(1−φ²))`, monotone, margin(0)=0, c=0→nominal | **Verified** | Independent recompute (python): DE c=1 → h=1: **3.658** (scale 1.000), h=6: **8.954** (2.447), h=72: **30.666** (8.3823). Matches `adaptiveMargin`, the unit test, and the report (8.3823·σ* ≈ 30.7). `widenedThresholds` preserves midpoint (test (v)). |
| Baselines reproduce committed data exactly | **Verified** | naive arma rows == `fixed_{R}.json` arma savings at all 18 cells, diff **0.00000000 pp** (computed from artifacts). Perfect-foresight S0 = fixed control S0: DE 43.3510 / IT 32.6655 / SE 22.8685. `fixed_DE.json` control = 43.35101808672119. |
| Determinism + reproducibility | **Verified** | Full `--mode adaptive` re-run to `/tmp/review_adaptive` byte-identical (`cmp`) on all 3 region JSONs + summary + DE CSV; second re-run also byte-identical. `adaptive_*` artifacts untouched. |
| Tests pass | **Verified** | `pnpm test src/domain/adaptive-margin.test.ts` → **9 passed**; `pnpm test src/cli/adaptive-sweep.test.ts` → **15 passed**; full `pnpm test` → **14 files, 201 passed**. |
| Protected domain files unmodified | **Verified** | `git diff --stat src/domain/` → empty. Only `src/cli/{forecast-sweep,index,optimize}.ts` + SPEC.md modified (phase B.0/B.1). |
| DE h=1/3 recovery = 0.688 / 0.349; h≥6 = 0 | **Verified** | recovery = (S_adapt−S_naive)/(S0_FF−S_naive): h=1 **0.6876716**, h=3 **0.3488657**, h=6 raw **−0.08883**, h=72 raw **−0.02648**. Matches artifacts exactly. |
| `recovery_vs_oracle` reported and strong in grace region | **Verified** | DE h=1 **0.9538**, h=3 **0.7988** (recomputed). Field present in all rows/CSV/summary. |
| Oracle h∈{1,6} = committed reopt delay rows; fresh arma matches | **Verified** | D7a test asserts 6-decimal closeness (passes). Fresh arma(h) optimizations for h∈{3,12,24,72}: I re-evaluated the recorded oracle thresholds under the arma(h) timeline — recorded == fresh eval **exact (0.00e+0)** at all 12 cells, all `completed=true` (e.g. DE h=72 (407.47,159.35) → 32.670544, overhead 166.6 %). |
| Static oracle h=72 recovers only 32/48/30 % | **Verified** | (S_oracle−S_naive)/(S0_FF−S_naive): DE **0.3161**, IT **0.4838**, SE **0.3019**. |
| c* selections are weak/outlier-driven | **Verified** | DE c*=0.75, meanRecovery 0.05556 = 1/18 — single 2024 cell (h=3, recovery **1.0**); IT c*=2, meanRecovery 0.02479 — single 2024 cell (h=3, recovery **0.4427**); 2022/2023 contribute 0 at every c. |
| IT genuinely budget-constrained | **Verified** | Independent per-c probe (test year): IT at c=0 → overhead 194.7 % completes; **any material widening incomplete** (c=1 at h≥6, c≥2 at h=1, O=200.0). Nominal headroom is only ~5.3 pp. |
| DTPR β = (τ_p+τ_r)/3600·θ_p_nominal | **Verified** | β: DE **11.258**, IT **10.197**, SE **0.751** (computed = artifact). IT DTPR incomplete at all h (O=200.0); DE/SE complete, savings ≈ naive. |
| "Formula cannot express the oracle h=72 margins" (report §3.3) | **REJECTED** | See Finding M1. The rule is c·interval: c≈6–7 reproduces the oracle scale. Independent probe: DE h=72 **c=6 → S=31.43, completed** (recovery_vs_oracle ≈ 0.75); c=8 → 36.29 but budget-infeasible. |
| "Completed-run recovery ≈ 0 at h≥6" (report §5.3) | **Partially rejected** | True at the *chosen* c* (DE h=72: 27.32 < naive 27.73), but **c*-specific, not rule-specific** (DE h=72 c=6 recovers). See Finding M2. |

## 3. Findings

### BLOCKING
None.

### MAJOR

**M1 — "h=72 loss cannot be expressed by the margin rule" is false; the bottleneck is the c-grid + c-selection.**
- Location: `phase_b1.md` §3.3 (last sentence), §5.4; `adaptive_summary.json` `adaptiveRuleLimitation`.
- Problem: The report says the oracle thresholds "(θ_p 349–407, θ_r 128–159) are far beyond what the AR(1) interval widening (margins 5.7–35.8 g/kWh at c=1) can express." The rule is `c·σ*·scale`, *linear in c* — it can express any margin. My independent closed-loop probe on the completed 2025 test year shows DE h=72: c=0.75 → 27.32, c=4 → 28.66, **c=6 → 31.43 (completed, vs oracle 32.67)**, c=8 → 36.29 (incomplete). So the adaptive rule *can* close ≈75 % of the recoverable h=72 gap; the modest recovery is caused by (i) the c-grid ceiling of 2 and (ii) the train-year mean-recovery objective returning ≈0 signal at large c (2022–24 show no h=72 recovery at any c), not by the formula. Additionally, the oracle's center at h=72 drifts to ~283 vs nominal midpoint 270 — a center shift the *symmetric* widening rule cannot express, which the report never mentions.
- Concrete fix: (a) run and commit a recovery-vs-c (and recovery_vs_oracle-vs-c) sensitivity table/curve for each region, h∈{6,12,24,72}; (b) reword to: "at c≈1 the model interval under-widens; the c-grid (≤2) and train-year c-selection never select the c≈6–7 that would approach the oracle, and c=8 is budget-infeasible; the oracle's center also drifts, which symmetric widening cannot express."
- Severity: MAJOR — the paper's most consequential interpretive claim ("h=72 is structural") currently over-claims; it must be narrowed to "structural **for static-threshold policies** and **for the train-validated c***".

**M2 — "recovery ≈ 0 at h≥6" is presented as a property of the controller, not of the chosen c*.**
- Location: `phase_b1.md` §3.2 table, §5.3 ("at h≥6 and for IT/SE the completed-run recovery is ~0").
- Problem: For DE h=72 the completed-run recovery is 0 *at c*=0.75*; the same rule at c=6 recovers ~24 % of the total loss / ~75 % of the recoverable gap. Conflating "the c*-selected controller recovers ~0" with "the controller recovers ~0" would under-sell a mechanism that works when the operator can scale c. The three genuinely distinct effects are: (a) static-oracle-structural share (verified, 32–48 % of h=72 loss unreachable by any static policy), (b) budget-bound share (IT — verified), (c) c-selection fragility share (DE h=72, SE h≥24 — real, and itself a finding: train-year signal cannot pick the right c).
- Concrete fix: restructure §5.3/paper D.7 to separate these three shares; report the c*-specific recovery *and* the c-sensitivity envelope.
- Severity: MAJOR — affects the abstract/narrative (the phase-A "recovers most of the loss" sentence must be calibrated, but *in the right direction*: the recoverable-fraction story is stronger than "recovery ≈ 0 at h≥6" implies).

### MINOR

**m1 — SE train-selected c*=1.5 is not a usable on-test setting (budget-infeasible at h≥24).**
- Location: `phase_b1.md` §3.1, §3.2 (SE rows), `adaptive_SE.json` c_selection.
- Problem: The c-selection procedure is supposed to yield the controller's deployed setting, but for SE it returns a c that is infeasible on the test year for h≥24 (overhead 200 %, incomplete). The report tables this honestly (the `*`), but the paper should also report the feasible envelope (SE completes only up to c=1 at h=1,3,6,12; c=0.75 at h=24; and at h=72 even c=0.75 → S=2.62, far below naive 8.64 — no feasible c recovers SE h=72).
- Fix: add a per-region feasible-c column to the c-selection table; state "train-selected c* is not budget-feasible on 2025 for SE at h≥24" explicitly.
- Severity: MINOR — honesty is present; completeness of the recommendation is not.

**m2 — "structural" should be qualified as "structural for static-threshold policies".**
- Location: `phase_b1.md` §3.3, §5.4; summary `completionWarning`/`adaptiveRuleLimitation`.
- Problem: The oracle ceiling (32–48 %) is an upper bound over *static* thresholds only. A dynamic (e.g., horizon-varying c, or data-driven band-width) policy is outside the class tested; "structural" invites an over-broad reading.
- Fix: use "unreachable by any static-threshold policy" in the paper; keep "structural" only with that qualifier.
- Severity: MINOR (wording), but it matters for the claim–evidence pass.

**m3 — State explicitly that the completion guard never erased a genuine completed-run recovery.**
- Location: `phase_b1.md` §6.1.
- Problem: For IT the only *completed* adaptive run (h=1, c=2, O=199.6, S=32.26) is below naive (32.51); recovery would be 0 even unguarded. The guard only suppresses the *inflated* incomplete rows (recovery_raw 0.26–1.89). A reviewer might suspect the guard masked positive results; it did not.
- Fix: one sentence in §5/D.7: "the completion guard changes only the budget-exhausted rows; every completed-run recovery is identical with or without the guard."
- Severity: MINOR.

**m4 — Lead the paper with `recovery_vs_oracle`, not `recovery`.**
- Location: `phase_b1.md` §5.3; C.9 figure plan.
- Problem: `recovery` (vs perfect-foresight) is dominated by the structural share; `recovery_vs_oracle` (fraction of the *recoverable* gap closed) is the informative controller-quality number (DE h=1: 0.95, h=3: 0.80). It is already computed and reported — make it the headline and define both clearly in D.7.
- Severity: MINOR (presentation; field already exists).

### NIT

- **n1** — `adaptive_summary.json` methodology says delay≈arma "verified within tolerance"; D7a asserts 6-decimal equality — write "exact to 6 decimals".
- **n2** — CSV columns use `toPrecision(6)`, so any claim↔CSV trace must use the JSON artifacts; note this in the reproducibility statement.
- **n3** — `generated: "2026-08-19"` is a date, not a timestamp; irrelevant but could be tightened.
- **n4** — IT DTPR savings (33.42–34.15 %) exceed perfect-foresight S0 (32.67 %) because the run is incomplete; already marked `*`, but state the "above-perfect savings is an incompleteness artifact" explicitly where DTPR is compared.

## 4. Verdict on the "modest recovery" result

The result is **(i) a legitimate, well-executed scientific finding** — provided two interpretive corrections are made. Specifically:

- The **static-oracle ceiling** (32/48/30 % of the h=72 loss) is real, deterministic, and verified: no static-threshold policy recovers the bulk of the h=72 staleness loss. This is a strong, novel, publishable result and is exactly what the SPEC's "upper bound from reopt drift" oracle was meant to quantify.
- The **IT budget-binding** is real and verified (nominal overhead 194.7 %, ~5 pp headroom; any material widening is budget-infeasible). This confirms the phase-B.0 completion-constraint caveat in closed loop — a genuine design-rule result.
- The **DE grace-region recovery** (0.69/0.35 of the small h=1/3 loss; 0.95/0.80 of the recoverable gap) is real.
- The **h≥6 "≈ 0 recovery" is NOT robust**: it is an artifact of the c-grid ceiling (2) and the train-year c-selection returning ≈0 signal, not of the margin formula (which is linear in c and reaches the oracle scale at c≈6–7) nor of the budget for DE. If the paper reports only the c* result without the recovery-vs-c sensitivity, it will mis-state the mechanism.

**What a fix agent should change (if one is dispatched):** (1) add the recovery-vs-c / recovery_vs_oracle-vs-c sensitivity artifact for all regions (h∈{6,12,24,72}); (2) correct the two sentences in §3.3/§5.4 and the `adaptiveRuleLimitation` summary text (M1 wording); (3) restructure §5.3 into the three-share decomposition (M2); (4) add the feasible-c envelope for SE (m1). These are analysis/wording changes — the simulation/domain code and artifacts are sound and need no changes.

## 5. Recommended next steps for the paper

1. **D.7 content**: present `recovery_vs_oracle` as the headline controller metric; show the recovery-vs-c curve; split the h=72 residual into (a) static-oracle-structural, (b) budget-bound, (c) c-selection-fragile shares.
2. **Narrative**: replace "recovers most of the loss" with "recovers most of the *recoverable* loss within the grace region (95/80 % of the naive→oracle gap at h=1/3 for DE); the residual at large h is (i) unreachable by any static-threshold policy (32–48 % ceiling), (ii) budget-bound for IT/SE at 200 % overhead, and (iii) further limited by a c-grid whose train-year signal is near-zero."
3. **C.9 figure**: draw recovery curves (per region, per h) with both `recovery` and `recovery_vs_oracle`, plus the static-oracle ceiling line — this single figure makes the structural-vs-recoverable decomposition visible.
4. **Discussion/limitations**: state the empirical-vs-model interval gap (test-year h=72 AR(1) RMSE 113.4 vs model interval 30.7 g/kWh at c=1) as the reason c≈1 under-widens, and note the symmetric-widening restriction (no center drift) as a rule limitation.
5. **Claim–evidence pass (D.10)**: every new sentence must trace to the JSON; the two corrected sentences in §3.3/§5.4 of the phase report should be fixed in the artifacts so D.10 doesn't resurrect the over-claim.

## DoD self-check
- R1 (read SPEC B.1, phase_b1.md, B.0 report, code, artifacts, research report §8.3): done.
- R2 (independently recomputed ≥3 key numbers): margin formula values (3.658/8.954/30.666), DE h=1 recovery (0.6877), oracle cells (12/12 exact), oracle h=72 recovery (0.316/0.484/0.302), DTPR β — all shown above and matching.
- R3 (ran isolated tests + adaptive sweep re-run + cell compare): 9+15 tests passed; full sweep re-run byte-identical to committed artifacts on 7 files; determinism re-run identical; 201 tests full suite.
- R4 (review_b1.md written with all five sections + verdict): this file.
- R5 (no files modified): `git status --porcelain` identical before/after review (only phase B.0/B.1 agent changes present); all probes under `/tmp/opencode/`.
