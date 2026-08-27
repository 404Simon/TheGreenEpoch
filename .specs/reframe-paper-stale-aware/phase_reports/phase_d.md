# Phase D — Paper restructure (report)

- Date: 2026-08-19
- Agent: paper-writing agent, Phase D (`reframe-paper-stale-aware`)
- Decisive-experiment verdict honored: **Story A** ("works-with-design-rules")
- Deliverables: `publication/eenergy/main.tex`, `references.bib`,
  `claims_evidence.md`, updated `README.md`, `figures/closed_loop.tex` (input
  fix), this report. Nothing committed; `publication/ICREC_Rome/` untouched.

## 1. Title choice and rationale

**Chosen title:**
> *"Staleness, Not Noise: Why Carbon-Aware LLM Pretraining Does Not Need ML Forecasting"*

Rationale (from research_report §9.2 candidate #1, which the SPEC endorses as
the recommended working title):
- It **leads with staleness** (narrative rule 9) and encodes the paper's money
  result (noise is cheap, staleness is expensive) rather than a neutral method
  name.
- It is **honest to the results**: §4 shows persistence ≈ AR(1) ≈ AR(7) at the
  5-min decision scale (RMSE gap ≤ 0.06 g/kWh) and §6 shows even a 4σ\*
  additive forecast error costs < 7% of savings in DE/IT while a 6-h-stale
  decision costs 36–60%. The abstract immediately converts the negative reading
  into the positive design contribution (grace horizon + adaptive controller),
  and the Discussion scopes the claim (SE's sub-σ\* margin is noise-sensitive;
  ML forecasting would help in sub-margin error regimes).
- It matches e-Energy's appetite for premise-questioning papers (cf. Green
  Mirage, Moving-Beyond-MCI, equilibrium analysis).
- Alternative considered and rejected: the SPEC working title
  ("Stale-Aware Hysteresis Control …") is accurate but does not capture the
  paper's central claim; the abstract retains the adaptive-controller content
  so nothing is lost.

## 2. Abstract gist

"Carbon-aware temporal shifting pauses LLM pretraining when grid CI exceeds a
threshold … the question is not how accurate the forecast is but how fresh the
decision is. Five years of 5-minute ACI across five grids: CI is near-unit-root
(lag-1 ≥ 0.996; persistence ≈ AR(1), RMSE gap ≤ 0.06 g/kWh); a 4σ\* additive
forecast error costs < 7% of savings in DE and IT, while a six-hour-stale
decision costs 36–60%. We formalize the grace horizon (max staleness with
≤ 10% loss), predict it from the empirical AR(1) RMSE (k≈0.68·S, R²=0.91,
8 region-years), and give a stale-aware adaptive hysteresis controller that
closes 95%/80% of the recoverable naive→oracle gap at h=1/3 (DE). Under
minutes-scale checkpoints for a 671B MoE model, completed-feasible retention is
97.6/94.6/95.2% @900 s and 91.4/85.3/81.4% @2700 s (DE/IT/SE)." → Story A +
calibrated controller narrative.

## 3. Section summary (per SPEC D.1–D.9)

1. **Introduction** — frontier emissions (Strubell, Patterson, Wu, Maji
   crossroads, DeepSeek-V3), grid-flexibility framing (Lin & Chien), the
   decide-on-forecast/pay-on-realized problem, the central noise-vs-staleness
   question, 5 contributions (near-unit-root char.; noise-vs-staleness
   decomposition; grace horizon predicted from empirical AR(1) RMSE; stale-aware
   adaptive controller + three-share decomposition; checkpoint-realistic
   evaluation + completion-constraint design rule).
2. **Related Work** — DTPR as theory anchor (no double-threshold novelty claim);
   explicit UQ-Advice relationship ("UQ-Advice assumes forecast quality is the
   bottleneck; we show staleness is"); forecasting thread (DACF/CarbonCast/
   EnsembleCI/CarbonX) positioned as hourly and second-order at the decision
   scale; LLM thread (curtailment-LLM, QoRa, Perseus/Zeus, CarbonScaling);
   accounting thread (Beyond-MCI, avg-vs-marginal, Green Mirage, untangling);
   **full-width positioning table (tab:positioning)** with all mandatory rows
   and a final "This work" row (empty slot reserved for us).
3. **System Model & Problem Formulation** — DeepSeek-V3-class job model (671B
   MoE, 14.8T tokens, 2048 GPUs, 700 W/60 W, PUE 1.27, checkpoint 148.8 s swept
   to 900/2700 s; 1.34 TB state estimate), hysteresis policy + closed-loop
   TikZ figure, staleness model (= persistence forecast), adaptive margin rule
   (eq. 1), metrics (savings %, overhead %, budget, completed-feasible,
   grace-horizon ≤ 10% degradation).
4. **Grid CI at 5-min resolution** — data (§5 grids 2022–2026), near-unit-root
   calibration table, `ci_trace_acf` + `rmse_horizon` figures, persistence ≈
   AR(1) ≈ AR(7) argument, why ML forecasting adds ~nothing at the decision
   scale.
5. **Threshold Optimization & Design Rules** — optimizer + evaluation protocol,
   Pareto (`pareto`), near-zero-margin rule (honest: SE stable, DE/IT
   energy-crisis caveat), percentile NOT year-stable, multi-year rule table,
   grace-horizon year-stable, `heatmap_year_region`, budget sweep with IT
   collapse at B ≤ 50 % (completed-feasible values; raw `best` inflation
   flagged), checkpoint-realism table (retention + margin widening + completion
   constraint as required design rule).
6. **Forecast Robustness** — money figure `noise_vs_staleness` (shared g/kWh
   axis), degradation table, reopt drift + `reopt_drift` figure, grace-horizon
   prediction `grace_horizon_map` + per-region-year table (k=0.684, R²=0.913,
   8 points; IT-2023 censored; μ sensitivity), empirical-not-theoretical RMSE
   (3.7× under-widening).
7. **Stale-Aware Adaptive Control** — controller + pseudo-code, c-selection on
   train years (honest: signal ≤ 0.056, grid ceiling 2), closed-loop
   `adaptive_recovery` + `tab:adaptive` (recovery_vs_oracle headline: DE 0.95
   @h=1, 0.80 @h=3), three-share h=72 decomposition (static-oracle-structural
   32/48/30 %; budget-bound; c-selection fragility with c=6 → S=31.43,
   rvo=0.748), DTPR fixed-separation benchmark (neutral DE/SE, IT
   completion-infeasible), design-rule outputs.
8. **Discussion** — demand-side flexibility framing; why 5-min resolution;
   aggregation/feedback effects (equilibrium engagement); signal choice
   (ACI vs MCI vs excess power — Moving-Beyond-MCI, avg-vs-marginal, Green
   Mirage, untangling); marginal-vs-average accounting; when noise matters (SE
   boundary); embodied carbon/water; checkpoint realism; data-freshness SLA
   (grace horizon as SLA); limitations (μ sensitivity, IT-2023 censoring,
   symmetric widening, single-site, Scope-2 basis).
9. **Conclusion** — 3-sentence calibrated restatement.

## 4. Deviations / decisions

1. **`figures/closed_loop.tex` modified** for correct `\input`. The committed
   file used `\ifdefined\documentclass` to decide standalone-vs-input mode; in
   LaTeX `\documentclass` is always defined, so `\input` executed
   `\documentclass` inside the body ("Can be used only in preamble"). Fixed by a
   `\paperstandaloneinputsentinel` macro defined in `main.tex` before the
   `\input`; standalone compile still works (verified). Documented in the file
   header and README.
2. **Title**: candidate #1 chosen (see §1) over the SPEC working title.
3. **Positioning table** rendered as a full-width `table*` (18 rows + "This
   work" row) because single-column width could not fit 6 columns without
   ugly overfull boxes.
4. **9 figures, 8 images + TikZ**: the task lists 9 Phase-C figures and asks to
   select ~6–8; all 9 are used (closed_loop, ci_trace_acf, rmse_horizon, pareto,
   heatmap_year_region, noise_vs_staleness, grace_horizon_map, reopt_drift,
   adaptive_recovery). Page budget is met.
5. **Checkpoint sweep values**: §3 states {148.8, 900, 2700} s and notes the
   150 s control is numerically degenerate (ΔS < 0.01 pp), which it is
   (checkpoint_summary.json).
6. **Ratio claim scoped**: "a six-hour-stale decision costs 5–25× more than a
   4σ\* noise" was NOT used (SE ratio is only 1.5×); the paper says "6–15× in
   DE/IT" (14.9× / 6.5×), consistent with the SE boundary documented in
   Discussion.
7. **DTPR savings claim scoped** to "within 1 pp of naive" (SE h=1 is −0.74 pp;
   the earlier 0.7 pp phrasing was too tight).
8. **Sukprasert (limitations) DOI** corrected to 10.1145/3627703.3650079
   (verified via Crossref); the research-report §11 only gave the arXiv id.
9. **Minor overfull hboxes** (≤ 6 pt) remain in three intro/related-work
   paragraphs; tables and figures are clean. Cosmetic; left for Phase E polish.
10. **Bib metadata**: 16 `@inproceedings` entries given `address = {New York,
    NY, USA}` and the EuroSys entry given pages to silence ACM-bst warnings;
    build now has 0 bib warnings/errors.

## 5. DoD checklist — pass/fail with evidence

| # | Check | Result | Evidence |
|---|---|---|---|
| D1 | `make` → `main.pdf`, 0 LaTeX errors, no undefined cit/ref | **PASS** | `grep -c "^!" main.log` → 0; `grep -c undefined main.log` → 0; `make` from `publication/eenergy` (after `make fullclean`) → "All targets (main.pdf) are up-to-date". |
| D2 | ≈ 9–11 content pages (sigconf, excl. refs) | **PASS** | `Output written on main.pdf (10 pages …)`; content = pages 1–9, `References` begins on page 10 (pdftotext). |
| D3 | All mandatory citations present + positioning table with our row | **PASS** | 19/19 mandatory keys `\cite`d (grep counts 2–7 each); `tab:positioning` (table\*) lists all 19 + "This work" last row; `comm` on cited-vs-bib keys → empty (45 = 45). |
| D4 | No over-claims; forbidden phrases inspected/reworded | **PASS** | grep: "first to" 0, "Hanford" 0, "survives all delay" 0, "recovers most of the loss" 0, "Gap never exceeds" 0, "two thresholds" 0. "universal" only in negated/scoped contexts ("not a universal value/percentile", "without being universal"); "survives all tested bias levels" appears once, explicitly scoped to SE (SPEC reference facts: SE survives all). |
| D5 | Corrected adaptive-controller narrative | **PASS** | "recovers most of the recoverable loss within the grace region" (main.tex:776); `recovery\_vs\_oracle` is the HEADLINE metric (6 occurrences; fig adaptive_recovery solid line); three-share h=72 decomposition ("three distinct shares", (a) static-threshold-structural 32/48/30%, (b) budget-bound, (c) c-selection fragility); completion guard + completion-constraint design rule (§5, §7); grace prediction uses the empirical RMSE curve ("must therefore use the empirical RMSE curve", 3.7× note). |
| D6 | All cited figures exist; compiles with them | **PASS** | 8 `includegraphics` targets + `closed_loop.tex` all resolve to `figures/*.pdf` (verified file-by-file); `grep -i "not found"` in main.log → none. |
| D7 | references.bib complete; bibtex clean | **PASS** | 45 entries, 45 cited keys, no missing/duplicate (comm → empty); `main.blg`: 0 warnings, 0 errors after clean rebuild. |
| D8 | claims_evidence.md written, traceable rows | **PASS** | `publication/eenergy/claims_evidence.md`: 50+ rows (A–H) each mapping a paper number → artifact path + field; estimates flagged in a dedicated section (1.34 TB state, DeepSeek GPU-hours, PUE, Kimi, 561M). |
| D9 | phase_d.md written with DoD evidence | **PASS** | this file. |
| D10 | Abstract + contributions honest to Story A and calibrated results | **PASS** | Abstract states the checkpoint retention numbers, the calibrated recovery ("recoverable naive-to-oracle gap", 95/80% at h=1/3 DE), the conditional grace prediction (k≈0.68·S, R²=0.91, 8 region-years), and scopes the noise claim to DE/IT; contributions mirror the same calibrated phrasing. |

## 6. Numbers flagged for D.10 scrutiny

1. **Retention percentages** (97.6/94.6/95.2; 91.4/85.3/81.4) — computed as
   `bestCompleted.savings ÷ baseline.savings` from `checkpoint_summary.json`;
   re-derive and round to 1 dp in the claim pass.
2. **Static-oracle ceiling shares** (32/48/30 % at h=72) — computed from
   `adaptive_{R}.json` (oracle/naive/perfect savings); re-derive.
3. **6–15× staleness-vs-noise ratio** — derived (14.9× DE, 6.5× IT); restate as
   the explicit per-region ratios for full traceability.
4. **k-per-point implied values** (0.46–0.92, CV≈0.25) in §6 — from
   `grace_horizon.json` `points[].k.yearMean`; verify CV by hand.
5. **Oracle h=72 center ≈ 283 / nominal midpoint 270.1** — computed from θ
   pairs; verify.
6. **Estimates** (see claims_evidence.md): 1.34 TB checkpoint state (671e9 ×
   2 B), DeepSeek 2.788M GPU-hours (report), PUE 1.27 (Jegham et al.), Kimi K2
   profile, 561M curtailment model — validate against sources at camera-ready.

## 7. Exact commands

```bash
# Build (clean rebuild; D1 evidence)
cd publication/eenergy && make fullclean && make

# D1 / D2 verification
grep -c '^!' main.log          # 0
grep -c 'undefined' main.log   # 0
grep 'Output written' main.log # 10 pages
pdftotext main.pdf - | grep -n '^References$'   # refs start on p.10

# D4 forbidden-phrase grep
grep -n -i -E "first to|Hanford|survives all delay|recovers most of the loss|Gap never exceeds|two thresholds" main.tex

# D3/D7 key alignment
grep -o '\\cite{[^}]*}' main.tex | tr ',' '\n' | sed 's/.*{//;s/}//;s/ //g' | sort -u > /tmp/c.txt
grep -o '^@[a-z]*{[^,]*' references.bib | sed 's/@[a-z]*{//' | sort -u > /tmp/b.txt
comm -3 /tmp/c.txt /tmp/b.txt   # empty
grep -c 'error message' main.blg  # 0

# D6 figure existence
for f in $(grep -o 'includegraphics\[[^]]*\]{figures/[^}]*}' main.tex | sed 's/.*{//;s/}//'); do test -f "$f.pdf" && echo "OK $f" || echo "MISSING $f"; done

# Standalone TikZ still compiles
cd figures && pdflatex -interaction=nonstopmode -halt-on-error closed_loop.tex && grep -c '^!' closed_loop.log
```

Artifacts: `publication/eenergy/main.tex`, `references.bib`,
`claims_evidence.md`, `README.md` (updated), `figures/closed_loop.tex`
(integration fix). Nothing committed; `publication/ICREC_Rome/` untouched.
