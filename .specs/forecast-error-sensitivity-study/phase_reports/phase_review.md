# Adversarial Review — Forecast-Error Sensitivity Study

Date: 2026-08-18. Reviewer: adversarial audit (investigate only, no fixes).

## VERDICT: **FAIL** (ready after fixing one major paper claim; code and artifacts are sound)

| Item | Check | Status |
|------|-------|--------|
| A1 | Calibration table `tab:calibration` vs `calibration_{region}.json` | OK |
| A2 | Headline claims vs `fixed_summary.json` / `fixed_*.json` / `reopt_summary.json` | **FINDING (major)** |
| A3 | Persistence–AR(1) gap claim; README "<0.1%" claim | **FINDING (minor)** |
| A4 | `delta_s_frac` at σ\* vs paper %; SE grace exhausted at σ\* | OK |
| B1 | `simulateStepwise` decision/accounting split | OK |
| B2 | ARMA head-fallback bound vs evaluation start index | OK |
| B3 | Fixed-mode control = forecast rows' policy/start; `savings_perfect` ≡ control | OK |
| B4 | `runOptimization` decisionTimeline threading (identity regression) | OK |
| B5 | `run_forecast_experiments.sh` correctness | OK |
| C1 | Full pipeline ×2, byte-identical artifacts, exit 0 | OK |
| C2 | `experiments_run.log` grows; gitignored | OK |
| D1 | `pnpm test` 169/169 (×2, no flakiness) | OK |
| D2 | `pnpm build` green; `tsc` = exactly 6 baseline errors, none new | OK |
| D3 | Spot-checked identity/degradation tests assert correctly | OK |
| E1 | `make` compiles; 0 `^!`; 0 undefined references | OK |
| E2 | Figure refs `forecast_rmse_horizon/degradation/savings_overhead_score/reopt_drift` all exist | OK |
| E3 | New subsection claims only DE/IT/SE | OK |
| E4 | Abstract 170 words ≤ 250 | OK |
| F1 | `git status` clean of temp/log/dist; 2 untracked docs outside manifest | OK |
| F2 | No console debug spew in new CLI modules | OK |

---

## FINDINGS (ranked by severity)

### F1 — MAJOR — Paper contradicts its own artifact: "margin rule survives all delay levels in all regions" is false for IT delay=1

`main.tex:568-569`:
> "The near-zero-margin rule (Δθ ≤ 16 gCO₂eq/kWh) survives all delay levels in all regions and all additive levels in Sweden."

Artifact says otherwise (`reopt_summary.json` → IT `marginRuleSurvives`):
```
{"family":"delay","param_value":1,"survives":false,"seedFraction":0}
```
`reopt_IT.json` best for delay 1: `margin=16.250` (θP 246.7 / θR 230.45, `margin_drift=0`), i.e. unchanged from the 16.25 baseline and > 16. Evidence:
```
$ python3 - <<'EOF'
import json
d=json.load(open("publication/output/forecast/reopt_IT.json"))
for c in d["configs"]:
    if c["family"]=="delay" and c["param_value"] in (1,6):
        print(c["param_value"], c["best"]["margin"])
EOF
1 16.25
6 3.83
```
The following sentence (line 570: "It is exceeded only in the high-variability grids under measurement noise") is also contradicted by IT: its margin exceeds 16 already at **zero noise** (baseline 16.25), which the paper itself concedes two sentences later (line 574-575). Recommended fix: reword to "delay does not inflate the margin (IT's margin stays at its 16.25 baseline, already above the 16 gCO₂eq/kWh ceiling); the rule is newly exceeded only under measurement noise at ≥1σ\*" — or drop "in all regions".

### F2 — MINOR — "gap never exceeds ≈0.05 gCO₂eq/kWh" is exceeded at IT h=72 (0.0512)

`main.tex:492-493` and `512-513`: "never exceeds ≈0.05 gCO₂eq/kWh at any horizon". Actual max persistence–AR(1) gap (order 1, all regions/horizons) = **0.051153** gCO₂eq/kWh (IT, h=72). Evidence:
```
$ python3 … # all 18 order-1 pairs
MAX gap: 0.051153 at ('IT', 72)   MAX ratio: 0.202982%
```
Borderline (≈ softens it), but "never exceeds" is technically wrong. The companion "below 1% of the RMSE" is correct (max ratio 0.203% < 1%). Recommended fix: "never exceeds ≈0.05 gCO₂eq/kWh" → "never exceeds ≈0.06" or "≤0.05 in 17 of 18 cells, 0.051 at IT h=72".

### F3 — MINOR — README "<0.1% of the error at every horizon" is not supported by the artifacts

`publication/output/forecast/README.md:44`: "the RMSE gap is < 0.1 % of the error at every horizon". Max gap/RMSE = **0.203%** (SE h=72; also SE h=3/6/12/24 and IT all > 0.1%). If "of the error" means RMSE, the claim is wrong by 2×; the paper's "below 1% of the RMSE" is the correct one. (If the author meant gap/mean intensity, all cells are <0.1% — then the wording should say so.) Recommended fix: change README to "< 0.3% of the RMSE" or clarify the baseline.

### F4 — MINOR — Fixed-sweep SE baseline does not reproduce `tab:best` SE; only reopt does

The forecast section says it fixes "the Table~\ref{tab:best} DeepSeek V3 policies" (272/268, 246/231, 19/18). DE and IT reproduce `tab:best` exactly (`fixed_summary.json` s0 43.351/32.666 vs 43.4%/32.7%; overhead 174.31/194.68 vs 174.3/194.7). SE does **not**: fixed control S₀ = **22.868**, overhead = **99.4%**, whereas `tab:best` SE says 23.2% / 106.3% — which is exactly the **reopt** baseline (23.169, 106.34, θP/θR = 18.18/17.51). The rounded default (19,18) is not the tab:best optimum. All SE degradation numbers are internally consistent (normalized to 22.87), so no forecast % is wrong, but the paper's "fix the tab:best policies" statement is only approximate for SE. Recommended fix: either use the reopt-optimal SE thresholds (18.2/17.5) as the fixed SE policy, or note that SE's default grid point is 0.3 pp off the optimum.

### F5 — MINOR — Two untracked files outside the deliverable manifest

`git status --short` shows `?? SPEC.md` and `?? publication/ICREC_Rome/acm.md`. Both are untracked and are not in the deliverable list; likely intended (spec / ACM template notes) but verify before commit. No stray temp files, logs, `node_modules`, or `dist` (all ignored).

---

## Verified items (evidence highlights)

- **A1** — every cell of `tab:calibration` matches `calibration_{region}.json` (DE 3.66/4.21/4.21/13.85/13.85/49.77/49.76/113.4/113.4; IT 4.42/…/77.7/77.6; SE 0.77/…/7.8/7.8); lag-1 0.9997/0.9987/0.9960 match `lag1AutoCorr` (0.999655/0.998681/0.995950).
- **A2** — additive σ\* 0.3/0.4/4.8% ✓ (0.312/0.369/4.83), 4σ\* 2.4/6.5/40.6% ✓ (2.41/6.46/40.55), multiplicative 0.2/0.2/3.0% and 1.5/3.9/30.6% ✓, staleness h=72 36/42/60% (ARMA 62% SE) ✓, grace DE/IT ≥4σ\*, SE 1σ\* ✓, grace horizon 24/12/12 ✓, reopt margins 4.6/16.3/0.7 ✓, margin growth 16.7/28.4 (DE) and 23.2/30.2 (IT) ✓ (= baseline + drift), "below 7% up to 4σ\*" for DE/IT ✓ (max 6.46%).
- **A4** — SE 2σ\* additive = 16.41% > 10% ✓; grace additive = 1 (atMax=false) ✓.
- **B1** — `simulation.ts` decisions use `decisionCo2` (line 166, 94-104) while all emissions (lines 148, 209, 221, 226) use realized `co2`; checkpoint transitions (line 148) use realized.
- **B2** — fallback `t < horizon + order − 1` (forecast.ts:119) matches evaluation `start = horizon + order − 1` (forecast-calibrate.ts:191); test "arma head-fallback copies realized values" covers it.
- **B3** — `savings_perfect` is a single value equal to `control.savings` in every row of all 3 fixed JSONs; same policy/start (verified via python).
- **B4** — `runOptimization` passes `decisionTimeline` to the `neverPause` baseline too, but `neverPausePolicy` always returns "continue" (policy.ts:13-20), so the baseline is unaffected; covered by test `runOptimization with identity forecast → produces identical points` and by the live `assertIdentity` for reopt level 0 (would have thrown during runs 1/2).
- **B5** — script runs all 4 steps with correct flags; `set -euo pipefail` + `tee` propagates failure via pipefail (minor quirk: a failed step does not abort later steps inside the group, only the final exit code reflects it).
- **C1** — `bash publication/run_forecast_experiments.sh` exit 0 (163 s, 161 s); `sha256sum` of all 27 artifacts (json/csv/svg/eps) identical across the pre-run state, run 1, and run 2 (`diff hash_before hash_after` → empty).
- **C2** — `experiments_run.log` grew across both runs (4 separators) and is ignored via `publication/.gitignore:*.log` (`git check-ignore -v` confirms).
- **D1** — `pnpm test`: 11 files, 169 tests, passed twice (no order dependence observed).
- **D2** — `pnpm build` exit 0; `npx tsc --noEmit` → exactly 6 `error TS`, byte-identical to `/tmp/opencode/tsc_baseline.txt`, none in `src/domain`/`src/cli` new files.
- **D3** — bit-identical test compares full `SimProgress` sequences via `toEqual`; level-0 degradation test asserts `delta_s_frac ≈ 0`; identity regression asserts `points`/`best` equality.
- **E1** — forced recompile (`touch main.tex && make`): exit 0, `grep -c '^!' main.log` = 0, `Reference.*undefined` = 0, `Citation.*undefined` = 0; all 4 `forecast_*.eps` embedded without missing-file warnings.
- **E2** — 4 committed figure names exist and are referenced.
- **E3** — new subsection mentions only DE/IT/SE (Germany 7×, Italy 8×, Sweden 9×; no CN/US).
- **E4** — abstract = 170 words.
- **F2** — all `console.log` in new CLI modules gated by `!quiet`; data written via `writeFileSync`.

## Ready to commit?
Code, tests, artifacts, and figures: **yes** (deterministic, green, internally consistent).
Paper: **no** — fix F1 (false "survives all delay levels in all regions" claim) before submission; F2/F3/F4 are one-word or one-line accuracy fixes.
