# Figure pipeline (Phase C) — `publication/eenergy/figures/`

Generates the 8 publication figures (SVG + PDF) for
*Stale-Aware Hysteresis Control for Carbon-Aware LLM Pretraining* from **committed
artifacts only** (no experiment recomputation). Plus `closed_loop.tex`, the TikZ
closed-loop diagram.

## Install

The venv lives outside the repo's git history (see `publication/.gitignore`):

```bash
python3 -m venv publication/eenergy/.venv
publication/eenergy/.venv/bin/pip install matplotlib
```

This creates matplotlib 3.11.x inside the venv. numpy is pulled in as a dependency.

### Nix note (why the wrapper exists)

On this Nix machine the venv's numpy cannot resolve `libstdc++.so.6` / `libz.so.1`
from a plain `venv/bin/python` launch. The wrapper `make_figures.sh` prepends the
matching Nix store dirs to `LD_LIBRARY_PATH`:

1. keeps an `LD_LIBRARY_PATH` already set by the caller;
2. otherwise appends the `/nix/store` dirs of the first `libstdc++.so.6`
   (gcc-lib) and `libz.so.1` (zlib) it finds;
3. if neither exists it warns and tries anyway.

It also pins `SOURCE_DATE_EPOCH=0` so the PDF output is byte-deterministic.

## Run

```bash
bash publication/eenergy/figures/make_figures.sh
```

Prints one line per figure and writes `<name>.svg` + `<name>.pdf` in this directory.
The script is deterministic: re-running yields byte-identical SVG and PDF (SVG ids are
stabilized by a fixed `svg.hashsalt`, the SVG `Date` metadata is pinned, and the PDF
`CreationDate`/`ModDate` come from `SOURCE_DATE_EPOCH=0`).

## Figures and their data sources

| File | Data source (committed) |
|---|---|
| `ci_trace_acf` | `public/data/co2/DE_2025.json` (trace + sample ACF of the committed series) |
| `rmse_horizon` | `publication/output/forecast/calibration_{DE,IT,SE}.json` → `evaluation[]` |
| `pareto` | `publication/output/results/DS_{DE,IT,SE}_all_2025_{800,100}_10it.csv`, `DS_{DE,IT,SE}_all_2022-2025_{800,100}_10it_alpha1.csv` (Budget `✓ Yes` only) |
| `noise_vs_staleness` | `fixed_{DE,IT,SE}.json` `summary[]` degradation curves, `calibration_*.json` (σ*, σrel, persistence RMSE), `public/data/co2/{DE,IT,SE}_2025.json` (year-mean CI) |
| `grace_horizon_map` | `publication/output/forecast/grace_horizon.json` (`predicted[]`, `kPrimary`) |
| `reopt_drift` | `publication/output/forecast/reopt_summary.json` (`baseline`, `drift[]`) |
| `adaptive_recovery` | `publication/output/forecast/adaptive_summary.json` (`regions[].rows[]`, `chosenC`) |
| `heatmap_year_region` | `publication/output/forecast/multiyear_summary.json` (`per_year[]`) |
| `closed_loop.tex` | none (TikZ diagram) |

## C.6 shared x-axis mapping (noise vs staleness)

All three families are plotted against one physical quantity — the **RMS
decision-error magnitude (g/kWh)**:

- additive noise `x = level · σ*` (the `sigma` field of the `fixed_{region}.json`
  additive rows, g/kWh);
- multiplicative noise `x = level · σrel · μ` with `μ` = 2025 decision-year mean CI
  (the multiplicative decision perturbs CI by `exp(level·σrel·z)`; RMS absolute
  deviation ≈ `μ·level·σrel`, documented first-order approximation);
- staleness delay `h` `x = RMSE_persistence(h)` (empirical AR(1) persistence RMSE at
  horizon `h` from `calibration_{region}.json` `evaluation[]`, order 1) — the stale
  decision is exactly a persistence forecast.

`y = ΔS/S₀` is `degradation_frac_mean` from the `fixed_{region}.json` summary rows
(additive/multiplicative) and the persistence-family rows (staleness).

## closed_loop.tex

Standalone-compilable (`pdflatex closed_loop.tex`) and `\input`-able into the paper
via the `standalone` package:

```latex
\usepackage{standalone}
...
\input{figures/closed_loop}
```
