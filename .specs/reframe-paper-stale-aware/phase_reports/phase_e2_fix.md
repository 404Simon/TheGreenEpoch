# Phase E.2 Fix Report — Adversarial Review #2 (F-N1 MINOR + NITs a–d)

- Phase: E.2 fix agent ("Staleness Makes Errors Large: Why 5-Minute Carbon-Aware Pretraining Needs Fresh Signals More Than Better Forecasts", ACM e-Energy 2027)
- Agent: FIX AGENT (write access; no experiment re-run, no artifact regeneration, no number/figure/table/structure change)
- Date: 2026-08-19
- Source of worklist: `.specs/reframe-paper-stale-aware/phase_reports/review_e2.md` §3 (fresh findings F-N1 MINOR, NITs a–d). Verdict: MINOR-FIXES → READY.
- Invariants honored: no headline number, figure, table row, recovery formula, or magnitude-thesis wording changed. Only the scoping of the "vs. <7%" comparison (F-N1), two source-hygiene comments (NITs a/b/d), and one adverb removal (NIT c).

---

## 1. Per-finding resolution table

| # | Sev. | Finding (review_e2.md) | Resolved | Where changed | Artifact-verified |
|---|---|---|---|---|---|
| **F-N1** | MINOR | "vs. <7% for that 4σ\* noise" mixes the 3-region staleness range (35.8–60.5%, DE/IT/SE) against a DE/IT-only "<7%" (SE 4σ\* = 40.6%). Must scope or state SE explicitly. | ✅ | Abstract L46–47 → "for that $4\sigma^*$ noise in DE and IT"; intro L100–101 → "for the $4\sigma^*$ error itself in DE and IT"; conclusion L1043 → "versus $<$7\% for the $4\sigma^*$ noise in DE and IT". One-line scoping per the review's own suggested fix ("or simply scope: '…in DE and IT'"). SE's 40.6% remains stated in §6.1 (L584, now without "only"), Table 6 (tab:degrad), and Limitations (4). | Review's R4 fixed grid: DE add4 2.41 / IT 6.46 / SE 40.55 → "<7%" now explicitly DE/IT-only; SE 40.6% stated where the per-region noise cost appears. No number changed. |
| **NIT a** | NIT | Old title still in `main.tex` L3–4 comment block + stale Story-A description | ✅ | Header comment rewritten: `% Title: "Staleness Makes Errors Large: Why 5-Minute Carbon-Aware Pretraining Needs Fresh Signals More Than Better Forecasts"` and Story line updated to the magnitude thesis. `\title` / `\fancyhead[LO]` already correct. grep for "Staleness, Not Noise" / "Does Not Need ML": 0 matches in `main.tex`. | Grep-clean (0). |
| **NIT b** | NIT | `claims_evidence.md` G4b oracle value stale at 4 decimals (43.3198); artifact DE h=1 oracle = 43.323040 | ✅ | G4b → savings_naive 43.2507, savings_adaptive 43.3197, savings_oracle 43.3230, matching `adaptive_summary.json` DE h=1 exactly at 4 dp. Paper cells (43.25/43.32/43.32) and the 0.07 pp / 0.69 / 0.95 numbers unchanged (already consistent). | Re-derived from `adaptive_summary.json`: 43.250739 / 43.319698 / 43.323040 → 43.2507 / 43.3197 / 43.3230. Exact. |
| **NIT c** | NIT | §6.1 L584 leftover over-strong adverb "only" before SE's 40.6% | ✅ | §6.1 (L584) → "a $4\sigma^*$ additive error costs $2.4\%$ (DE), $6.5\%$ (IT), and $40.6\%$ (SE) of savings". "only" dropped; magnitude thesis and matched-magnitude paragraph untouched. | Text-only. |
| **NIT d** | NIT | F10 Phase-E reproducibility statement residual (SPEC E.4) should be disclosed/pointed to | ✅ | Added tracked-residual note to `claims_evidence.md` after E15: the Phase-E reproducibility statement (commands, runtimes, no-RNG/determinism, commit hash) is a separate deliverable being produced by the orchestrator and released at de-anonymization; the score-function half of F10 is resolved (E15). | Note is purely documentary; no numeric claim. |

**Count: 5/5 resolved (1 MINOR + 4 NITs).** No new findings introduced.

---

## 2. What did NOT change

- **No number, figure, table, or section structure changed.** All Table/Figure rows are byte-identical in `main.tex` (diff vs the E.1-fixed baseline touches only: the 3 comment lines, the 3 F-N1 sentence scopes, and the "only" deletion).
- **Recovery formula untouched** (`recovery = clip(…,0,1)` + guards, §7.1).
- **Magnitude thesis untouched** (abstract/intro/§6.1/Fig. 6 caption/conclusion wording preserved; only the noise-side scoping of the "<7%" clause changed).
- **No experiment re-run, no artifact regenerated, nothing committed.**

---

## 3. Rebuild & verification (DoD evidence)

Commands (from `publication/eenergy`):

```
make fullclean && make
grep -c "^!" main.log                      # 0
grep -c "undefined" main.log               # 0
pdfinfo main.pdf | grep Pages              # 10
grep "Overfull" main.log                   # 5, all <= 3.5 pt
```

- LaTeX errors: **0** (`^!` count 0).
- Undefined references/citations: **0**.
- Pages: **10** (one `\looseness=-1` appended at the end of the conclusion paragraph restored the 10th page; the 3 F-N1 scopes alone had pushed ref [45] onto an 11th page — no content removed, only that paragraph's line-breaking loosened).
- Abstract: **≤250 words** under every counting method. Review #2's pre-edit count was 243; this fix adds exactly 4 words ("in DE and IT") → 247 by the reviewer's method. Independent PDF counts: 238 words (alnum tokens, excluding the ACM Reference Format block) / 249 raw tokens — all ≤250.
- Overfull hboxes: **5, all ≤ 3.5 pt** (2.86 / 1.05 / 3.49 / 3.09 / 2.82) — same set as the review-#2 baseline (N2 was already RESOLVED at 5 ≤3.5 pt); no new overfull.
- Citations: **45/45** bibkeys cited (`comm` cited-vs-bib → empty), all **19 mandatory** keys present (johnson2025uqadvice, bostandoost2024lacs, lechowicz2023opr, lechowicz2024ocs, jiang2026equilibrium, wiesner2025beyondmci, sukprasert2024avm, maji2024greenmirage, wiesner2026curtailment, maji2023carboncast, yan2025ensembleci, maji2025carbonx, li2024uncertainty, sukprasert2024limitations, wiesner2021letswait, wiesner2025qora, chung2024perseus, you2022zeus, jiang2025carbonscaling).
- Forbidden/over-claim grep across `main.tex`: 0 for "Staleness, Not Noise", "Does Not Need ML", "costs only", "noise is cheap, staleness is expensive", "we prove". (The single "``noise is cheap''" occurrence at L96 is the deliberate F1 fix phrasing "for a more precise reason than 'noise is cheap'" — unchanged.)
- Tests: `pnpm test` → **14 files, 205/205 passed** (unchanged).

Rendered text check (`pdftotext`): all three F-N1 locations show "…in DE and IT" scoping on the noise side.

---

## 4. DoD checklist

| DoD | Requirement | Result |
|---|---|---|
| D1 | F-N1 + NITs a–d all fixed | **PASS** — see §1 table; 5/5 resolved, verified against committed artifacts and rendered PDF |
| D2 | Rebuild clean: 0 errors / 0 undefined / 10 pages / abstract ≤250 / mandatory citations intact | **PASS** — 0 / 0 / 10 / 247 (reviewer method) / 45=45 incl. 19 mandatory |
| D3 | `pnpm test` green (205) | **PASS** — 14 files, 205/205 |
| D4 | `phase_e2_fix.md` written with DoD evidence | **PASS** — this file |

---

## 5. Residual risks

1. **`\looseness=-1`** added at the end of the conclusion paragraph to keep 10 pages. This is a standard line-breaking hint with no content effect, but it is a fragile layout knob: if any future prose edit lengthens the conclusion, page count may tick to 11 again. Re-check `pdfinfo main.pdf | grep Pages` on any future build.
2. **F10 Phase-E reproducibility statement** (SPEC E.4) remains the only open item — it is explicitly a separate orchestrator deliverable (commands/runtimes/commit hash) to be released at de-anonymization; the score-function part is resolved and the residual is now disclosed in `claims_evidence.md` (NIT d).
3. **Abstract still carries "R²=0.91"** (now labelled "level fit") — the body caveats (4 distinct RMSE(g) values, IT-2023 censoring) carry the full qualifier; unchanged from E.1 as review #2 did not re-flag it.
4. Nothing committed; no experiment artifacts regenerated.

---

## 6. Re-verification commands

```
cd publication/eenergy && make fullclean && make
grep -c "^!" main.log          # 0
grep -c "undefined" main.log   # 0
pdfinfo main.pdf | grep Pages  # 10
grep "Overfull" main.log       # 5, <= 3.5 pt
cd /home/simon/dev/TheGreenEpoch && pnpm test   # 205/205
```

Nothing committed. `publication/ICREC_Rome/` untouched. No experiment artifacts regenerated.
