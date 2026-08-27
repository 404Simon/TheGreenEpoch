# Phase C — Figure pipeline (matplotlib) report

- Date: 2026-08-19
- Agent: figure-pipeline agent, Phase C (`reframe-paper-stale-aware`)
- Deliverables: `publication/eenergy/figures/make_figures.py`, `make_figures.sh`,
  `closed_loop.tex`, `README.md`, 8 SVG+PDF figure pairs, `.venv` (git-ignored),
  this report. Nothing committed; `publication/ICREC_Rome/` untouched.

## 1. Environment setup (C.1)

- `python3 -m venv publication/eenergy/.venv`; `venv/bin/pip install matplotlib` →
  **matplotlib 3.11.1** + numpy 2.5.2 (Python 3.13).
- On this Nix host a plain venv launch fails to import numpy:
  `libstdc++.so.6` then `libz.so.1` cannot be resolved. The committed wrapper
  `publication/eenergy/figures/make_figures.sh` builds `LD_LIBRARY_PATH` from the
  `/nix/store` dirs holding `libstdc++.so.6` (gcc-lib) and `libz.so.1` (zlib),
  honoring a caller-provided `LD_LIBRARY_PATH`, and warns if neither is found.
  Verified end-to-end: `bash publication/eenergy/figures/make_figures.sh` runs all
  8 figures and prints a per-figure summary (D1).
- `.venv/` added to `publication/.gitignore` (D1/D2).

## 2. Palette and typography (uniform conventions)

```python
PALETTE = {
  de  = "#2166ac"  it  = "#e08214"  se  = "#4d9221"     # regions (distinct hues)
  persistence = "#7f7f7f"   ar1 = "#2166ac"   ar7 = "#762a83"
  additive    = "#d73027"   multiplicative = "#fd8d3c"
  staleness = "#762a83"     delay = "#762a83"
  recovery  = "#e08214"     recovery_vs_oracle = "#2166ac"
}
```

- Serif text (`font.family = serif`, DejaVu Serif); `font.size` 9, labels 9,
  ticks 8, legends 8; **minimum font size 8 pt** (all explicit `fontsize=` in the
  script are ≥ 8). Light grid (`#bbbbbb`, α 0.45). Every save uses
  `bbox_inches="tight"`, `pad_inches=0.02`. Sizes: column-width ≈ 3.5 in
  (grace_horizon_map, heatmap_year_region) to textwidth ≈ 7.2 in
  (ci_trace_acf, rmse_horizon, pareto, noise_vs_staleness, reopt_drift,
  adaptive_recovery).
- Math is rendered with mathtext (`$\theta_p$`, `$\sigma^*$`, `$\Delta S/S_0$` …)
  — no missing-glyph warnings in the run log (D5).

## 3. Per-figure description and data sources

| Fig | Output files | Description | Data source (all committed) |
|---|---|---|---|
| C.3 | `ci_trace_acf.svg/.pdf` | Left: 48 h (first 576 pts) of 5-min DE 2025 CI (time x-axis). Right: sample ACF of the full year to lag 288 (24 h), annotated `lag-1 ≈ 0.9995` (computed from the committed series: 0.9995) and `24 h ≈ 0.72`. | `public/data/co2/DE_2025.json` |
| C.4 | `rmse_horizon.svg/.pdf` | 3 panels (DE/IT/SE): persistence vs AR(1) vs AR(7) RMSE vs horizon {1,3,6,12,24,72}, log-x. Persistence–AR(1) max gap 0.02 / 0.05 / 0.02 g/kWh (≤ 0.06). | `publication/output/forecast/calibration_{DE,IT,SE}.json` → `evaluation[]` |
| C.5 | `pareto.svg/.pdf` | 3 panels: within-budget (`Budget == ✓ Yes`, `budget_exceeded` dropped) Pareto frontier Overhead % (x) vs CO₂ savings % (y); 2025 run (solid) + 2022–2025 pooled run (dashed), min–max envelope shaded between them; best 2025 point starred. | `publication/output/results/DS_{DE,IT,SE}_all_2025_{800,100}_10it.csv` and `DS_{DE,IT,SE}_all_2022-2025_{800,100}_10it_alpha1.csv` |
| C.6 | `noise_vs_staleness.svg/.pdf` | Money figure. 3 panels; one line per family (additive, multiplicative, staleness=persistence/delay) on the shared x-axis "decision-error magnitude (g/kWh), log". Grace-threshold 10 % line and 1σ* marker shown; DE/IT/SE each plotted. | `fixed_{DE,IT,SE}.json` `summary[]` `degradation_frac_mean`; `calibration_{DE,IT,SE}.json`; `public/data/co2/{DE,IT,SE}_2025.json` (scale μ). See §4. |
| C.7 | `grace_horizon_map.svg/.pdf` | Scatter predicted g (x) vs empirical g (y), log-log, identity line; color per region, IT-2023 marked as censored (hollow ✕); k·S rule annotated (k = 0.684, R² = 0.913, n = 8). | `publication/output/forecast/grace_horizon.json` (`predicted[]`, `kPrimary`, `fits.yearMean.noIT23`) |
| C.8 | `reopt_drift.svg/.pdf` | 3 panels in the (θ_p, θ_r) plane: baseline point + arrows to the drifted points (additive 0.5/1/2 red, delay 1/6 purple); margin annotated at baseline and at 2σ* (DE 4.64→28.36, IT 16.25→30.15, SE 0.67→5.49). | `publication/output/forecast/reopt_summary.json` (`baseline`, `drift[]`) |
| C.9 | `adaptive_recovery.svg/.pdf` | 3 panels, x = h (log {1,3,6,12,24,72}): **`recovery_vs_oracle` (solid, headline per B.1 fix)** and `recovery` (dashed); static-oracle ceiling = perfect-foresight S₀ line at y = 1.0; c* annotated (DE 0.75, IT 2, SE 1.5); budget-exhausted rows marked with red ✕ (recovery = 0 by completion guard). | `publication/output/forecast/adaptive_summary.json` (`regions[].rows[]`, `chosenC`, `s0_naive_ff`) |
| C.10 | `heatmap_year_region.svg/.pdf` | Region × year (rows DE/IT/SE, cols 2022–2025) savings % heatmap + second heatmap for margin θ_p−θ_r, values annotated. | `publication/output/forecast/multiyear_summary.json` (`per_year[]`) |
| C.2 | `closed_loop.tex` (+ compiled `closed_loop.pdf`) | TikZ diagram: grid CI → decision signal (forecast/stale) → hysteresis controller (θ_p/θ_r, adaptive margin) → pause/resume on training → emissions on realized CI (pay-on-realized). Standalone-compilable and `\input`-able (via the `standalone` package). | none (diagram) |

## 4. C.6 shared-axis mapping (D6) — precise definitions

All three families are mapped to the same physical quantity, the **RMS
decision-error magnitude (g/kWh)**:

- **additive** `x(level) = level · σ*` — the `sigma` field of the committed
  additive rows in `fixed_{region}.json` `summary[]` (level × σ*, g/kWh).
- **multiplicative** `x(level) = level · σrel · μ`, `μ = mean(carbonIntensity)`
  of `public/data/co2/{region}_2025.json` (decision-year mean). The multiplicative
  decision perturbs CI by `exp(level·σrel·z)`; the first-order absolute deviation
  at level `L` is `≈ L·level·σrel`, RMS ≈ `μ·level·σrel` (documented
  approximation; σrel = σ*/trainMean).
- **staleness delay h** `x(h) = RMSE_persistence(h)` — the empirical AR(1)
  persistence RMSE at horizon `h` from `calibration_{region}.json`
  `evaluation[]` (order 1, model `persistence`); the stale decision is exactly a
  persistence forecast, so this is its RMS decision error in g/kWh.

`y = ΔS/S₀ = degradation_frac_mean` of the matching `fixed_{region}.json`
summary rows (additive/multiplicative families, and the persistence family for
staleness; delay ≡ persistence in all cells).

Resulting curves: additive and multiplicative live at `x ≲ 18 g/kWh` (≤ 1σ*/4σ*)
with ΔS/S₀ < 10 % in DE/IT; staleness reaches the same loss only at RMSE 12–24 h
(DE/IT/SE grace 24/12/12), i.e. the figure makes "noise is cheap, staleness is
expensive" visible. A mapping note is also embedded in `figures/README.md`.

## 5. Determinism (D3)

- Set in `make_figures.py`: `SOURCE_DATE_EPOCH=0` (stable PDF
  `CreationDate`/`ModDate`), `svg.hashsalt` fixed (stable SVG element ids),
  SVG `Date` metadata pinned to `2026-08-19`. The wrapper also exports
  `SOURCE_DATE_EPOCH` for safety.
- Evidence: ran `make_figures.sh` → snapshotted all 16 SVG+PDF → re-ran →
  `cmp`: **17/17 byte-identical** (16 figure files + `closed_loop.pdf`).
  PDFs are byte-identical (not merely content-identical), so no residual
  nondeterminism is left undocumented.
- `closed_loop.pdf` compiles standalone with **0 errors**
  (`grep -c "^!" closed_loop.log` → 0); `\input` verified through the
  `standalone` package in a test article.

## 6. DoD checklist

| # | Check | Result | Evidence |
|---|---|---|---|
| D1 | venv + matplotlib; wrapper runs end-to-end; README documents install+run | **PASS** | `matplotlib 3.11.1`; `bash publication/eenergy/figures/make_figures.sh` prints 8 per-figure lines + summary; `figures/README.md` written |
| D2 | reads only committed artifacts; no recomputation; no src/ or forecast-data changes | **PASS** | read-set grepped (see §3 table; the only computation = ACF of the committed CI series and frontier envelopes of committed logs); `git diff --stat src/` shows only pre-existing B.0/B.1 files; no `M` on any `publication/output/forecast/` file |
| D3 | byte-identical re-run | **PASS** | `cmp` on all 16 SVG+PDF → identical (SOURCEEPOCH + hashsalt + pinned metadata) |
| D4 | closed_loop.tex compiles standalone, input-ready | **PASS** | `pdflatex closed_loop.tex` → 1 page, 0 `!` lines; `\input` via `standalone` package verified |
| D5 | 8 figures SVG+PDF, fonts ≥ 8 pt, uniform palette, no glyph warnings | **PASS** | all pairs exist (see §3); all explicit `fontsize=`/`labelsize=` ≥ 8; palette in §2; run log has no glyph/missing-font warnings |
| D6 | C.6 shared-axis mapping documented | **PASS** | §4 above (formula per family + source fields) + `figures/README.md` |
| D7 | phase_c.md with DoD evidence | **PASS** | this file |

## 7. Deviations / decisions

1. **C.6 curve source.** `fixed_summary.json` carries only the σ*/h=72 anchors
   (degradationAtSigmaStar / degradationAtH72), not full curves; the per-family
   `degradation_frac` curves live in `fixed_{region}.json` `summary[]`, so the
   figure reads those (same experiment family; σ*/σrel anchors match
   `fixed_summary.json`). Documented in the README and the script docstring.
2. **Multiplicative scale.** The multiplicative family's fractional `sigma`
   field is converted to g/kWh with the documented `level·σrel·μ` choice
   (§4). Alternative (scale = trainMean ⇒ x = level·σ*) collapses onto the
   additive curve; the year-mean choice keeps the two noise families visually
   distinct while remaining physically meaningful.
3. **C.9 interpretation of "perfect-foresight line at S₀".** In recovery units
   both the static-oracle ceiling and perfect-foresight are y = 1.0 (recovery
   normalizes S₀_FF to 1), so one reference line at y = 1.0 is drawn and labeled
   "static-oracle ceiling = perfect foresight (S₀)"; incomplete (budget-exhausted)
   rows are marked with red ✕ so the completion guard is visible.
4. **C.5 pooled frontier** uses the committed `_alpha1` pooled CSVs (default
   α=1); `budget_exceeded` rows dropped per SPEC.
5. **Determinism of PDFs** is achieved by fixing `SOURCE_DATE_EPOCH`; no
   `pdf.metadata` overrides were needed.

## 8. Exact commands

```bash
# setup
python3 -m venv publication/eenergy/.venv
publication/eenergy/.venv/bin/pip install matplotlib

# generate all figures (wrapper resolves libstdc++/libz on Nix, pins SOURCE_DATE_EPOCH)
bash publication/eenergy/figures/make_figures.sh

# determinism (byte-identical)
mkdir -p /tmp/d3 && cp publication/eenergy/figures/*.svg publication/eenergy/figures/*.pdf /tmp/d3/
bash publication/eenergy/figures/make_figures.sh
cmp /tmp/d3/ci_trace_acf.svg  publication/eenergy/figures/ci_trace_acf.svg   # → identical (×16)

# TikZ standalone compile + input check
cd publication/eenergy/figures && pdflatex -interaction=nonstopmode -halt-on-error closed_loop.tex
grep -c '^!' closed_loop.log     # 0

# cleanliness
git diff --stat src/                # only pre-existing B.0/B.1 files
git status --short publication/output/forecast/   # no modified committed data
```
