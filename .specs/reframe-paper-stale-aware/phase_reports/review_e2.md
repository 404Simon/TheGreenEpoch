# Adversarial Review #2 — "Staleness Makes Errors Large: Why 5-Minute Carbon-Aware Pretraining Needs Fresh Signals More Than Better Forecasts"

- Date: 2026-08-19. Reviewer: adversarial audit #2 of 2 (investigate only; read-only).
- Object: `publication/eenergy/main.tex` (+ compiled `main.pdf`, 10 pp, 0 LaTeX errors), committed artifacts under `publication/output/**/*.json`, `public/data/co2/`, `claims_evidence.md`, phase report `review_e1.md` (review #1) and `phase_e1_fix.md` (the fix agent's report).
- Protocol: verify every review-#1 finding (F1–F10, N1–N4) is resolved correctly without introducing new errors; fresh adversarial pass over the reframed paper.

## 1. Summary & overall assessment

**Recommendation: MINOR-FIXES → READY.** Every review-#1 finding is correctly resolved:
13 of 14 findings are RESOLVED exactly as claimed (F1–F9, N1–N4), and F10 is PARTIAL in the
one way the fix report itself discloses (score function done; the Phase-E reproducibility
statement — commands/runtimes/commit hash — remains open per SPEC E.4, commit hash
explicitly deferred to de-anonymization by review #1). No number, figure, or table value
drifted: I independently re-derived 22 headline numbers against the committed JSON and every
one matches the paper to reported precision. The new title is consistent with the abstract
and body claim strength, and the reframe did not introduce any over-claim ("the data show /
we find" scoping is respected; no "we prove" anywhere).

The fresh pass found **1 MINOR and 4 NITs** — no new MAJOR, no BLOCKING. The single MINOR
(F-N1, "vs. <7% for that 4σ\* noise") is a one-line scoping edit in three locations
(abstract, intro, conclusion) that re-introduces, at the headline level, the exact
cross-region over-generalization the F1 reframe was meant to eliminate: the staleness side
"35.8–60.5%" spans all three regions, but the "<7%" noise side is only true for DE (2.4%)
and IT (6.5%) — SE's 4σ\* additive cost is 40.6% (its own Table 6). Everything else is
source hygiene (a stale title comment, one stale claims-table cell at 4 decimals, one
leftover "only" adverb, and the tracked F10 residual).

**Strengths confirmed from review #1 (still intact)**
- Artifact discipline: all 22 re-derived numbers exact; tables byte-identical to committed JSON; figures from committed artifacts.
- The reframe is executed with unusual discipline: the magnitude thesis + matched-magnitude qualifier appear consistently in the abstract, intro, contribution 2, §6.1, Fig. 6 caption, and conclusion, with the correct committed numbers everywhere (DE 14.6→2.4 vs 13.9→0.7; IT 17.7→6.5 vs 12.5→1.8; SE 3.1→40.6 vs 2.6→5.2; ratios 7.8/4.4/2.5×).
- The additive-ceiling limitation (4σ\*) is stated in §6.1 and in Limitations (4).
- Honest negative/qualified results preserved: IT c\*=2 fragility, SE rounded-threshold disclosure, IT-2023 censoring, c-selection near-zero signal, DTPR IT infeasibility.

## 2. Fix-verification table (F1–F10, N1–N4)

| # | Sev. | Finding (review #1) | Status | Evidence |
|---|---|---|---|---|
| **F1** | MAJOR | Thesis must be "error magnitude dominates; staleness generates large errors; at matched magnitude additive noise per-unit more damaging"; no leftover intrinsic-asymmetry phrasing | **RESOLVED** | Title switched (§L18); abstract (L36–57) carries magnitude thesis + matched pair (14.6→2.4 vs 13.9→0.7, 7.8×, 35.8–60.5 vs <7); intro (L94–103) "inverted, but for a more precise reason than 'noise is cheap'"; contribution 2 (L141–146) "2.5–7.8× … yet at matched magnitude additive forecast noise is per-unit more damaging"; §6.1 (L592–604) full matched-magnitude paragraph + "Staleness 'wins' only through magnitude" + "central empirical result" sentence; Fig. 6 caption (L569–575) "Loss scales with error magnitude; at matched magnitude additive noise is the more damaging per unit error"; conclusion (L1037–1044). 4σ\* test-ceiling limitation stated (L590, Limitations (4)). "6–15×" re-phrased scenario-level scoped to DE/IT (L600–602). No "we prove"/"noise is cheap, staleness is expensive" anywhere (grep 0). Numbers all re-verified vs JSON (§4 R15–R18). **Residual:** the "vs <7%" phrase (L45/99/1042) is region-mixed for SE — see fresh finding F-N1 (MINOR). |
| **F2** | MAJOR | `recovery` definition must include clip, `S0_FF ≤ S_naive ⇒ 0` guard, `recovery_vs_oracle` guard, completion guard; SE h=1 must reproduce 0 | **RESOLVED** | §7.1 (L735–748): `recovery = clip((S_adapt−S_naive)/(S0_FF−S_naive),0,1)`; "Both are 0 when the denominator is non-positive: recovery = 0 if S0_FF ≤ S_naive … the SE case, whose perfect-foresight baseline is the rounded-threshold control 22.87%, below its naive policy"; `recovery_vs_oracle = 0 when S_oracle ≤ S_naive (SE, all h ≤ 12)`; completion guard stated. Reproduced: SE h=1 naive 23.0858, adapt 21.9773, perfect 22.8685 → unclipped +5.108; artifact recovery 0, rvo 0 ✓. The guard scope "(SE, all h≤12)" is exact (SE h=24 oracle 20.31 > naive 20.23; its rvo=0 is from the completion guard) — the fix correctly tightened review #1's suggested "all h≤24". |
| **F3** | MINOR | Abstract "multi-year-stable threshold design rules" overstates | **RESOLVED** | Abstract (L56–57) "threshold design rules stable in *scale* (margin scale, grace horizon) with documented crisis-year exceptions"; conclusion (L1054) "grace horizon stable in normal years (IT-2023 a documented outlier)". Consistent with Table 3 (DE margins 16/15/4.76/4.64, IT 3.24/8/4.72/16.25). |
| **F4** | MINOR | Grace-fit R²=0.91 is a level fit on ~4 distinct RMSE(g) values; horizon-space residuals coarse; R² de-emphasized | **RESOLVED** | Abstract (L51) "at a level fit k≈0.68·S (R²=0.91)"; intro (L118) "(a level fit: R²=0.91)"; contribution 3 (L149) "level-fit R²=0.91"; §6.3 (L668–681) "the 8 region-years share only 4 distinct RMSE(g) values (DE 49.8 g/kWh ×3, IT 36.6/20.6, SE 2.6 g/kWh ×3) because the RMSE(h) curve is the 2025 calibration reused across years—so R²=0.91 is a *level* fit… not horizon-prediction accuracy"; continuous residuals −7…+8 steps; DE-2024 grid 72 vs 24 "within one grid step only on the coarse-grid metric, its continuous prediction being 29". All verified vs `grace_horizon.json` (devCont −7…+8, gPredCont 29). |
| **F5** | MINOR | "≥0.996" false at full precision (SE 0.995950) | **RESOLVED** | Abstract "≈0.996" (L42); intro (L107) & contribution 1 (L137) "0.9960–0.9997"; §4 body "≈0.996–0.9997" (L357). |
| **F6** | MINOR | §3 "scale of the error a six-hour-old decision faces" (30.7) vs §6.3 (113.4) | **RESOLVED** | §3 (L321–324) "reaching 8.38σ\*≈30.7 g/kWh at h=72 for DE—the *theoretical* AR(1) prediction-interval scale of the error a six-hour-old decision faces; the empirical staleness error is about 3.7× larger (Section 6)". 8.38σ\* verified (σ\*=3.658, φ=0.999655 → factor 8.382, 30.66 g/kWh; 113.413/30.66 = 3.70). Forward ref resolves; §6.3 "Empirical, not theoretical, RMSE" (L713–719) states the 30.7 vs 113.4 comparison explicitly. |
| **F7** | MINOR | §7.2(c) c\*=2 sentence missing region (IT property) | **RESOLVED** | §7.2(c) (L875–877) "DE completes through c=6 at h=72; IT's train-selected c\*=2 widening is budget-infeasible at h≥3, and most of IT's c-grid from h=6; SE completes only through c=0.75 at h=72…". Verified: `adaptive_sensitivity_summary.json` feasibleC IT h3=[0…1.5] (excludes 2), h6=[0…0.75]; DE h72=[0…6]; SE h72=[0…0.75]; SE c=0.75 h72 savings 2.62 < naive (no recovery). |
| **F8** | MINOR | "No prior work (i)–(iv)" categorical | **RESOLVED** | Positioning (L221) "To our knowledge, no prior work (i)…(iv)". |
| **F9** | MINOR | IT grace "year-stable" hedged to near-vacuity | **RESOLVED** | §5 heading (L486) "The grace horizon is stable within regions—with a censored outlier"; (L489–493) "IT's 2024–25 horizons (24, 12) are within one grid step of each other, but 2023 (72) is a right-censored outlier—2023 CI never crosses the 10% degradation level even at h=72 (3.1%)…"; conclusion (L1054) "stable in normal years (IT-2023 a documented outlier)". Verified vs `multiyear_fixed_summary.json` (IT 72/24/12; IT-2023 h72 deg 3.12%). |
| **F10** | MINOR | Score function undefined; α=1 degeneracy; reproducibility statement | **PARTIAL** | §5 Optimizer (L417–421) "score = ½(α·S/100+1−(1−α)·O/B)"; "at α=1 the overhead term vanishes (score=(S/100+1)/2), so the budget binds only through the explicit overhead constraint—the source of the budget-blocked runs guarded below". Matches `src/domain/result.ts` computeScore. Determinism claim in §5 Evaluation protocol (L436–439). **The Phase-E reproducibility statement (commands/runtimes/commit hash, SPEC E.4) is still absent** — the fix report explicitly discloses this as an open residual (commit hash deferred to de-anonymization per review #1). Not a regression; tracked. |
| **N1** | NIT | "≤0.017 for SE" strictly false (0.01708) | **RESOLVED** | §7.2 (L837) "≤0.018 for SE"; `claims_evidence.md` G3 "≤0.018 … = 0.0171". Verified perYear[2023] SE "0.75" = 0.01708. |
| **N2** | NIT | Overfull hboxes (worst 5.87 pt) | **RESOLVED** | Rebuilt: **5** overfull hboxes, all ≤3.5 pt (2.86/1.05/3.49/3.09/2.82); 0 LaTeX errors; 0 undefined. Strictly better than the review-#1 baseline (6, ≤5.9 pt). |
| **N3** | NIT | "105,120 points/year" wrong for 2024 | **RESOLVED** | §4 (L347) "(105{,}120 points/year; 105{,}408 in the leap year 2024)". Verified data: DE_2024 len 105,408; DE_2023/2025 len 105,120; 25 files total. |
| **N4** | NIT | 0.07 pp clause | **RESOLVED** | §7.1 Headline (L777–779) "closes a recoverable gap of only 0.07 pp of savings (43.25→43.32 vs. the 43.35 perfect-foresight ceiling)". Verified: naive 43.2507, adapt 43.3197, S0_FF 43.3510 → Δ=0.0690 pp; recovery 0.6877. |

**Count: 13/14 RESOLVED, 1/14 PARTIAL (F10 reproducibility statement, disclosed & tracked).**

## 3. Fresh findings (from the rewrite; ranked)

No BLOCKING, no MAJOR.

### MINOR

**F-N1 — "vs. <7% for that 4σ\* noise" mixes a three-region staleness range against a DE/IT-only noise figure; the abstract's own Table 6 contradicts it for SE.**
- Location: abstract (L45); intro (L99–100); conclusion (L1042).
- Problem: All three state "a six-hour-stale decision costs 35.8–60.5% of savings vs. <7% for that 4σ\* noise". The staleness range 35.8–60.5% spans DE/IT/SE (Table 6: pers72 = 35.8/42.2/60.5), but "<7%" is only true for DE (add4 = 2.4%) and IT (6.5%); SE's 4σ\* additive cost is **40.6%** (Table 6, row SE add4), which is not <7%. This is precisely the cross-region, non-apples-to-apples comparison the F1 reframe was designed to remove, and a reviewer cross-checking the abstract against Table 6 will catch it. §6.1's own honest phrasing is correctly scoped ("costs 6–15× more … in DE and IT", L600–602), so the abstract/intro/conclusion lag behind the body.
- Fix (one line × 3 locations): "…costs 35.8–60.5% of savings in DE/IT/SE vs. <7% for the 4σ\* error in DE and IT (SE's sub-σ\* margin makes its 4σ\* cost 40.6%)", or simply scope: "…in DE and IT".

### NIT

**F-N2 — Old title still in the `main.tex` header comment.**
- Location: `main.tex` L3–4 (comment block): `% Title: "Staleness, Not Noise: Why Carbon-Aware LLM Pretraining Does Not Need ML Forecasting"` plus L5 `% Story A: …`. The `\title`, `\fancyhead[LO]`, and claims table header were updated, but the source-comment block was not. Does not affect the PDF (task's "not comments" rule), but a grader/editor reading the source sees the wrong title and a stale Story-A description.
- Fix: update the comment block to the new title.

**F-N3 — `claims_evidence.md` row G4b oracle value is stale at 4 decimals.**
- Location: `claims_evidence.md` G4b: "savings_oracle 43.3198" (also naive 43.2525, adapt 43.3195); artifact `adaptive_summary.json` DE h=1: naive 43.250739, adapt 43.319698, oracle **43.323040**. The paper's own cells (43.25/43.32/43.32) round correctly and the 0.07 pp / 0.69 / 0.95 numbers all match the artifact, so this is a claims-table drift, not a paper error.
- Fix: correct G4b to the artifact values.

**F-N4 — Leftover "only" adverb from the old "noise is cheap" framing.**
- Location: §6.1 (L584): "a $4\sigma^*$ additive error costs only $2.4\%$ (DE), $6.5\%$ (IT), and $40.6\%$ (SE)". "only" is jarring applied to SE's 40.6% (the region that is noise-sensitive by the paper's own account). Not factually wrong (SE is then immediately explained), but a tone leftover.
- Fix: drop "only" or restructure ("costs 2.4% (DE), 6.5% (IT), and 40.6% (SE)").

**F-N5 (tracked, not a new error) — F10 Phase-E reproducibility statement still open.**
- Location: SPEC E.4; §5 Evaluation protocol (determinism is stated; commands/runtimes/commit hash are not).
- Note: acknowledged in `phase_e1_fix.md` §6.4; commit hash explicitly deferred to de-anonymization. Satisfy before camera-ready; not a submission blocker for the anonymized version.

### Checks with no finding
- Title vs abstract/body claim strength: title "Staleness Makes Errors Large … Needs Fresh Signals More Than Better Forecasts" is matched by the abstract thesis and the §6.1/§6.3 support (near-unit-root → stale decision is a large persistence error; AR(7) vs AR(1) shaves ≤ 2 g/kWh at h=1, decision-irrelevant). No over-strength mismatch.
- No unsupported new claim: "the data show / we find" scoping respected; "central empirical result" is data-anchored; no "prove/optimal/first" on the own algorithm.
- No leftover old-title reference in the .tex body outside the comment (grep: 0).
- Fig. 3 annotation max-gap values (DE 0.02, IT 0.05, SE 0.02) match computed gaps (0.0226/0.0512/0.0154).
- Guard-wording correctness (F2's SE h≤12) verified against the artifact (SE h=24 oracle > naive).

## 4. Re-verified numbers (≥10; all against committed JSON)

> All commands run read-only from repo root. "Exact" = matches paper to reported precision.

| # | Number / query | Value | Verdict |
|---|---|---|---|
| R1 | Baseline optima DE/IT/SE (`reopt_summary.json`) | (272.37,267.73) S43.35 O174.31; (246.70,230.45) S32.67 O194.68; (18.18,17.51) S23.17 O106.34; margins 4.64/16.25/0.67 | Exact |
| R2 | Calibration lag-1 / σ\* / RMSE(1) / RMSE(72) (`calibration_*.json`) | DE 0.999655/3.658/4.213/113.413; IT 0.998681/4.421/5.020/77.610; SE 0.995950/0.774/0.776/7.825 | Exact (Tab. 2) |
| R3 | Max persistence–AR(1) gap | IT h=72: 77.66073 − 77.60958 = 0.0512 g/kWh (≤0.06) | Exact |
| R4 | Degradation grid (`fixed_*.json`): add1/add4/pers6/pers12/pers72 | DE 0.31/2.41/0.71/2.43/35.83; IT 0.37/6.46/1.76/4.05/42.19; SE 4.83/40.55/2.54/5.17/60.49 | Exact (Tab. 6) |
| R5 | Grace horizon 24/12/12 + multi-year (24/24/24, 72/24/12, 12/12/12) | `fixed_summary.json`, `multiyear_fixed_summary.json` (IT-2023 h72 deg 3.12%) | Exact |
| R6 | Reopt drift margins (baseline→1σ\*/2σ\*) | DE 4.64→16.68/28.36; IT 16.25→23.19/30.15; SE 0.67→3.21/5.49; IT delay-1 survives=False | Exact |
| R7 | Grace fit (`grace_horizon.json`): k, R² variants, 4 distinct RMSE(g), residuals | k=0.684; 0.9134/0.9154/0.1936/0.8622; DE 49.76×3, IT 36.64/20.62, SE 2.58×3; devCont −7…+8; DE-2024 gPred 72/cont 29 | Exact |
| R8 | Budget sweep (`budget_summary.json`) | DE 11.51/16.28/29.52/43.35; IT B≤50 none, B100 raw 82.00 vs 19.85, B200 32.67; SE 14.59/18.36/22.87/23.17 | Exact (Tab. 4) |
| R9 | Checkpoint retention + margins (`checkpoint_summary.json`) | 97.62/94.61/95.22 @900; 91.35/85.32/81.43 @2700; DE margin 4.64→16.59→39.50; raw DE 47.91→82.18 | Exact (Tab. 5) |
| R10 | Adaptive closed loop (`adaptive_summary.json`) | DE h1 43.25/43.32/43.32 rec0.69 rvo0.95; h3 rvo0.80; all IT/SE guarded rows 0; c\* 0.75/2/1.5 | Exact (Tab. 8) |
| R11 | c-selection grid (`adaptive_summary.json`) | DE 0.006/0.006/0.017/0.056/0.012/0.000/0.016; IT …0.025; SE …0.029; 2022 all-zero; 2023 SE "0.75"=0.0171 | Exact (Tab. 9) |
| R12 | DTPR (`adaptive_summary.json`) | β 11.258/10.197/0.751; DE \|Δ\|≤0.25, SE ≤0.74 (<1 pp); IT overhead_dtpr 200.0016 all h (infeasible) | Exact |
| R13 | Oracle h=72 ceilings 32/48/30% | (32.67−27.73)/(43.35−27.73)=0.316; IT 0.484; SE 0.295 | Exact |
| R14 | DE h=72 sensitivity c=6/c=8 (`adaptive_sensitivity_summary.json`) | c=6 S=31.43 completed rvo 0.748; c=8 S=36.29 infeasible; feasibleC DE h72 [0…6], IT h3 excludes 2, SE h72 [0…0.75] | Exact |
| R15 | Matched-magnitude pairs (§6.1) | DE 14.63→2.41 vs 13.85→0.71; IT 17.68→6.46 vs 12.5→1.76; SE 3.10→40.55 vs 2.59→5.17 | Exact |
| R16 | Magnitude ratios h=72/4σ\* | 113.4/14.6=7.8×; 77.7/17.7=4.4×; 7.8/3.1=2.5× | Exact |
| R17 | 8.38σ\*=30.7 g/kWh and 3.7× under-widening | factor 8.3824×3.658=30.66; 113.413/30.66=3.70 | Exact |
| R18 | F2 reproduction: SE h=1 formula | (21.977−23.086)/(22.8685−23.086)=+5.108; artifact recovery 0, rvo 0 | Exact (guard) |
| R19 | Abstract headline 0.07 pp | 43.3197−43.2507=0.0690 pp; recovery 0.6877; rvo 0.9538/0.7988 | Exact |
| R20 | Leap-year point count | DE_2024 len 105,408; DE_2023/2025 len 105,120; 25 files | Exact |
| R21 | Multi-year margins/savings (`multiyear_summary.json`) | DE 16/15.04/4.76/4.64 & 14.95–43.35; IT 3.24/8/4.72/16.25 & 10.02–32.67; SE 0.75/0.67/0.72/0.67 & 16.18–23.17; pct wander 11.29/13.71/14.82 pp | Exact (Tab. 3) |
| R22 | Abstract word count / pages / build / tests | 243 words (≤250, excluding ACM Reference Format block); 10 pages; 0 errors; 0 undefined; 5 overfull ≤3.5 pt; `pnpm test` 14 files 205/205; 45/45 bibkeys cited incl. all 19 mandatory | PASS |

## 5. Verdict

**MINOR-FIXES.** The review-#1 worklist is fully discharged (13/14 RESOLVED; F10's second half
is a disclosed, SPEC-tracked residual), no headline number drifted, and the reframed thesis is
now supported by the paper's own shared-axis data at every location that previously over-claimed.
The fresh pass found no BLOCKING and no MAJOR issues — only one MINOR (F-N1, the "vs <7%"
region-scoping at abstract/intro/conclusion, a one-line edit ×3) and four NITs (of which two are
source-hygiene and one is the tracked F10 residual).

**Final acceptance recommendation for submission: ACCEPT for submission after the optional
one-line F-N1 scope fix.** The F-N1 edit (and the NITs) require no experiment re-run and no
artifact change. If the authors choose to ship without F-N1, the paper is still defensible
("that 4σ\* noise" binds to the DE figure immediately preceding), but the sentence is exactly
the kind of thing a careful e-Energy reviewer would bounce, so I recommend taking the 5 minutes.

### DoD status
- R1 Rebuilt (`make`, 0 errors) and read the full PDF text; verified every F1–F10/N1–N4 — DONE.
- R2 Re-checked 22 headline numbers vs committed JSON (R1–R22, ≥10 required) — DONE.
- R3 `review_e2.md` written with verdict, fix-verification table, fresh findings, evidence — DONE (this file).
- R4 Read-only: no repo file modified by this reviewer (only build outputs regenerated by `make`; no tracked file changed, no source edited).
