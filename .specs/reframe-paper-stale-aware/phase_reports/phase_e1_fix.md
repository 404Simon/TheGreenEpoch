# Phase E.1 Fix Report — Adversarial Review #1 (F1–F10, N1–N4)

- Phase: E.1 fix agent ("Staleness Makes Errors Large: Why 5-Minute Carbon-Aware Pretraining Needs Fresh Signals More Than Better Forecasts", ACM e-Energy 2027)
- Agent: FIX AGENT (write access; no experiment re-run, no artifact regeneration)
- Date: 2026-08-19
- Source of worklist: `.specs/reframe-paper-stale-aware/phase_reports/review_e1.md` §3 (F1–F10 MAJOR+MINOR, N1–N4 NIT). No BLOCKING findings; every finding was prose/framing/definition. **No number, figure, design rule, or table value was changed** (verified: all Table/Figure rows byte-identical in source; all new prose numbers trace to committed JSON via `calibration_*.json` + `fixed_*.json`).

---

## 1. The new thesis sentence & the title decision

### 1.1 New thesis (stated verbatim in abstract / intro / §6.1 / conclusion)

> **Decision-error magnitude dominates savings loss; staleness is the failure mode that *generates* large errors (near-unit-root drift makes a stale decision a large-magnitude error by construction), so the operational lever is signal freshness — and, per unit error magnitude, additive forecast noise is actually more damaging than staleness.**

The matched-magnitude qualifier is stated explicitly in every location that previously said "noise is cheap, staleness is expensive" (abstract, intro, §6.1, Fig. 6 caption, contribution 2, conclusion), always with the committed numbers:
- DE: 4σ* = 14.6 g/kWh → 2.4% loss vs 6-step-stale 13.9 g/kWh → 0.71%
- IT: 17.7 g/kWh → 6.5% vs 12.5 g/kWh → 1.8%
- SE: 3.1 g/kWh → 40.6% vs 2.6 g/kWh → 5.2%
- magnitude ratios at the scenario endpoints: 113.4/14.6 = **7.8×** (DE), 77.7/17.7 = **4.4×** (IT), 7.8/3.1 = **2.5×** (SE)
- "6–15×" re-phrased as a *scenario-level* comparison ("at the magnitudes these modes realistically produce").

The additive-test ceiling (4σ*) is stated explicitly in §6.1 as a limitation ("The additive family is tested only up to 4σ*; we do not extrapolate beyond it") and in the Limitations list.

### 1.2 Title decision: **SWITCH to "Staleness Makes Errors Large: Why 5-Minute Carbon-Aware Pretraining Needs Fresh Signals More Than Better Forecasts"**

Rationale:
1. **Honesty.** The old title asserts an intrinsic noise-vs-staleness asymmetry ("Staleness, Not Noise") that the paper's own shared-axis data contradict at matched magnitude. Keeping it would require a disambiguating footnote in the abstract, and review #2 could re-flag it.
2. **Mechanism, not slogan.** The new title encodes the reframed thesis directly: "Staleness Makes Errors Large" is the magnitude mechanism; the subtitle "Needs Fresh Signals More Than Better Forecasts" is the operational takeaway (the data-freshness SLA) and preserves the "does not need ML forecasting" message in positive form.
3. **Punchiness retained.** Same two-line length, same cadence as the original (verb → colon → subtitle).
4. The reviewer's suggested alternative is adopted nearly verbatim (kept "5-Minute Carbon-Aware Pretraining" and "Fresh Signals More Than Better Forecasts").

Consequences: `\title`, `\fancyhead[LO]` ("Staleness Makes Errors Large"), and `claims_evidence.md` header updated. Historical phase reports are untouched (records).

---

## 2. Per-finding resolution table

| # | Sev. | Finding (review_e1.md) | Resolved | Where changed |
|---|---|---|---|---|
| **F1** | MAJOR | Central claim confounds error magnitude with error source; at matched magnitude additive noise is more damaging; staleness "wins" only via 2.5–7.8× larger errors | ✅ | Title (switched, §1.2), abstract (magnitude thesis + matched-magnitude pair + level-fit R² + scale-stable rules), intro ("inverted, but for a more precise reason than 'noise is cheap'"), contribution 2, §6.1 (explicit matched-magnitude comparison, then scenario-level finding, additive-ceiling limitation), Fig. 6 caption ("Loss scales with error magnitude; at matched magnitude additive noise is the more damaging per unit error, while staleness is the failure mode that produces large errors (h=72 RMSE ≈ 7.8× a 4σ* noise for DE)"), "6–15×" sentence (scenario-level), conclusion (magnitude thesis), Discussion ("The low cost of additive noise is a measured boundary condition, not a law"), Limitations (5) |
| **F2** | MAJOR | `recovery` formula as printed gave +5.10 for SE (reported 0.00); clip and `S0_FF ≤ S_naive ⇒ 0` guard unstated | ✅ | §7.1: full definition `recovery = clip((S_adapt−S_naive)/(S0_FF−S_naive),0,1)`, 0 when `S0_FF ≤ S_naive` (SE case: perfect-foresight baseline = rounded-threshold control 22.87% < naive), `recovery_vs_oracle = clip(…,0,1)` and 0 when `S_oracle ≤ S_naive` (SE, all h≤12 — see §3 note), completion guard restated |
| **F3** | MINOR | Abstract "multi-year-stable threshold design rules" overstates | ✅ | Abstract → "threshold design rules stable in *scale* (margin scale, grace horizon) with documented crisis-year exceptions" |
| **F4** | MINOR | Grace-fit R²=0.91 is a level fit on ~4 distinct RMSE(g) values; horizon-space accuracy coarser; R² de-emphasized | ✅ | Abstract → "at a level fit k≈0.68·S (R²=0.91)"; intro + contribution 3 → "a level fit"; §6.3 → "4 distinct RMSE(g) values (DE 49.8×3, IT 36.6/20.6, SE 2.6×3) because the RMSE(h) curve is the 2025 calibration reused across years", "R²=0.91 is a level fit, not horizon-prediction accuracy", continuous residuals −7…+8 steps, DE-2024 grid prediction 72 vs observed 24 "within one grid step" only under the coarse-grid metric (continuous prediction 29) |
| **F5** | MINOR | "≥ 0.996" false at full precision (SE 0.995950) | ✅ | Abstract → "≈0.996"; intro L107 & contribution 1 → "0.9960–0.9997"; §4 lag-1 range → "≈0.996–0.9997" |
| **F6** | MINOR | §3 "scale of the error a six-hour-old decision faces" (30.7) contradicts §6.3 (113.4) | ✅ | §3 margin paragraph → "the *theoretical* AR(1) prediction-interval scale of the error a six-hour-old decision faces; the empirical staleness error is about 3.7× larger (Section ref)" |
| **F7** | MINOR | §7.2(c) c*=2-widening sentence missing the region name (IT property, not DE) | ✅ | §7.2(c) → "IT's train-selected c*=2 widening is budget-infeasible at h≥3, and most of IT's c-grid from h=6; DE completes through c=6 at h=72; SE completes only…" |
| **F8** | MINOR | "No prior work (i)–(iv)" categorical | ✅ | §2 Positioning → "To our knowledge, no prior work (i)–(iv)" |
| **F9** | MINOR | IT grace "year-stable" hedged to near-vacuity | ✅ | §5 → heading "stable within regions—with a censored outlier"; IT: "2024–25 horizons (24, 12) within one grid step of each other, but 2023 (72) is a right-censored outlier…2023 CI never crosses the 10% degradation level even at h=72 (3.1%), so the 'year-stable' reading for IT rests on two of three years with a documented exception"; conclusion "a grace horizon stable in normal years (IT-2023 a documented outlier)" |
| **F10** | MINOR | Optimizer score function undefined; α=1 degeneracy unexplained; reproducibility statement open | ✅ | §5 Optimizer → `score = ½(α·(S/100) + 1 − (1−α)·(O/B))`, "at α=1 the overhead term vanishes (score=(S/100+1)/2), so the budget binds only through the explicit overhead constraint—the source of the budget-blocked runs guarded below" (explains the completed-feasible selection and the IT 82.00-vs-19.85 story). Phase-E reproducibility statement (commands/runtimes/no-RNG) remains open for E.4 (commit hash deferred to de-anonymization, per the review) |
| **N1** | NIT | "≤0.017 for SE" strictly false (artifact 0.01708) | ✅ | §7.2 c-selection → "≤0.018 for SE"; `claims_evidence.md` G3 → "≤0.018 … = 0.0171" |
| **N2** | NIT | Worst overfull hbox (~5.87 pt, margin-equation paragraph); 21 pt overfull introduced during the reframe in the conclusion | ✅ | margin-equation paragraph wrapped in a local `{\sloppy … \par}` (5.87 pt gone); conclusion first sentence reworded (21 pt gone). Final: **5 overfull hboxes, all ≤3.5 pt** (the pre-existing 1–3.5 pt ones) — strictly better than the review's "6, ≤5.9 pt" |
| **N3** | NIT | "105,120 points/year" wrong for 2024 | ✅ | §3 data → "(105{,}120 points/year; 105{,}408 in the leap year 2024)"; `claims_evidence.md` B3 |
| **N4** | NIT | Abstract headline is a 0.07 pp absolute effect | ✅ | §7.1 Headline controller metric → "the DE h=1 headline closes a recoverable gap of only 0.07 pp of savings (43.25→43.32 vs the 43.35 perfect-foresight ceiling)"; `claims_evidence.md` G4b |

**Additional correctness fix made during F2 (noted in the report for transparency):** the guard sentence originally drafted per the review's "(SE, all h≤24)" was corrected to **"(SE, all h≤12)"** after checking `adaptive_summary.json`: SE oracle = naive at h∈{1,3,6,12} (rvo zeroed by the S_oracle ≤ S_naive guard) but oracle 20.313 > naive 20.234 at h=24 (its rvo=0 comes from the completion guard). The paper's guard statement is now exact.

---

## 3. What did NOT change

- **Every number in every table and figure**: `tab:positioning`, `tab:cal`, `tab:rules`, `tab:budget`, `tab:ckpt`, `tab:degrad`, `tab:gracepred`, `tab:adaptive`, `tab:csel` — all rows byte-identical in `main.tex`.
- **All 9 figures** (`fig:loop`, `fig:trace`, `fig:rmse`, `fig:pareto`, `fig:heatmap`, `fig:noise`, `fig:grace`, `fig:drift`, `fig:adaptive`, `fig:alg`) — unchanged, only the Fig. 6 caption text was rewritten.
- **Design rules** (grace-horizon SLA, adaptive margin, completion constraint, checkpoint realism) — all survive, restated only in framing.
- **All 45 bibitems still cited; 0 undefined refs/citations.**
- **No experiment re-run, no artifact regenerated.**

---

## 4. Rebuild & verification (DoD evidence)

Commands (from `publication/eenergy`):

```
make fullclean && make
grep -c "^!" main.log        # 0
grep -c "undefined" main.log # 0
pdfinfo main.pdf | grep Pages# 10
grep "Overfull" main.log     # 5 hboxes, all <= 3.5 pt (was 6 <= 5.9 pt)
```

- LaTeX errors: **0** (`^!` count 0).
- Undefined references/citations: **0**.
- Pages: **10** (unchanged from review baseline).
- Abstract: **238 words** (careful counter; reviewer-style count ≈ 238 ≤ 250).
- Overfull hboxes: **5, all ≤ 3.5 pt** — worst pre-existing 5.87 pt (margin-equation paragraph) fixed with a local `\sloppy`; the 21 pt conclusion overflow introduced during reframing fixed by rewording.
- Mandatory citations: all 19 required keys `\cite`d (verified programmatically — none missing); 45/45 distinct keys cited, matching `references.bib`.
- Forbidden-phrase grep across `main.tex`: no "first to", no "Hanford", no "recovers most of the loss", no "Gap never exceeds", no "noise is cheap, staleness is expensive", no old "≥ 0.996" phrasing.
- Tests: `pnpm test` → **14 files, 205 tests passed** (unchanged).

---

## 5. DoD checklist

| DoD | Requirement | Result |
|---|---|---|
| D1 | F1 reframe applied to title (decided), abstract, intro, contributions, §6.1, Fig 6 caption, "6–15×", conclusion; numbers unchanged; matched-magnitude numbers stated correctly | **PASS** — title switched (§1.2); all seven locations carry the magnitude thesis + matched-magnitude qualifier with the correct committed numbers (14.6→2.4 vs 13.9→0.7 DE; 17.7→6.5 vs 12.5→1.8 IT; 3.1→40.6 vs 2.6→5.2 SE; ratios 7.8/4.4/2.5×); all table/figure values byte-identical |
| D2 | F2 formula definition stated fully in §7.1 | **PASS** — `recovery = clip((S_adapt−S_naive)/(S0_FF−S_naive),0,1)`, 0 when `S0_FF ≤ S_naive` (SE case named with 22.87 < naive), `recovery_vs_oracle` with `S_oracle ≤ S_naive ⇒ 0`, completion guard restated |
| D3 | F3–F10 and N1–N4 all fixed | **PASS** — see §2 table; every finding maps to a diff in `main.tex` (+ `claims_evidence.md`) |
| D4 | `claims_evidence.md` consistent with fixed paper | **PASS** — rows A1/A1b/A3/A4/A5/B3/B4/E15/F6/F14/F15/F16/G3/G4b/G5 and the re-verification section updated; header title updated; every new claim maps to a committed artifact field with matching value |
| D5 | Rebuild: 0 errors / 0 undefined / 10 pages / abstract ≤ 250 / citations intact / no forbidden phrases | **PASS** — 0 / 0 / 10 / 238 / 19+45 keys / none |
| D6 | `pnpm test` green (205) | **PASS** — 14 files, 205/205 |
| D7 | `phase_e1_fix.md` written with DoD evidence | **PASS** — this file |

---

## 6. Residual risks

1. **Title change ripples.** The old title appears in earlier phase reports (B.0–D.10, SPEC headers). These are historical records and were intentionally not rewritten; the fix report documents the change. If a downstream orchestrator keys on the old title, that's the one place to look.
2. **Abstract R² wording.** The abstract keeps "R²=0.91" but now labels it a "level fit" — still the strongest single number for the grace horizon; the body carries the full level-fit/4-distinct-values caveat. If review #2 wants the abstract even softer, the fallback is to drop "R²=0.91" from the abstract entirely (numbers remain in §6.3).
3. **SE "all h≤12" guard wording** (F2): deliberately tighter than the review's suggested "all h≤24" because the artifact shows oracle > naive at h=24 (verified above). Reviewer intent (state the guard) is honored; the wording is now exact.
4. **Phase-E reproducibility statement (F10 second half)** remains open per SPEC E.4 (commands/runtimes/no-RNG/determinism + commit hash). The score-function formula part of F10 is done in §5; the commit hash is deferred to de-anonymization as the review allows.
5. The 5 remaining overfull hboxes (1.0–3.5 pt) are pre-existing cosmetics the review classified as NIT beyond the worst one; no action needed, but noted for completeness.

---

## 7. Re-verification commands

```
cd publication/eenergy && make fullclean && make
grep -c "^!" main.log && grep -c "undefined" main.log && pdfinfo main.pdf | grep Pages
grep "Overfull" main.log
cd /home/simon/dev/TheGreenEpoch && pnpm test
```

Nothing committed. `publication/ICREC_Rome/` untouched. No experiment artifacts regenerated.
