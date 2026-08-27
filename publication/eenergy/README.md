# e-Energy 2027 paper — "Staleness Makes Errors Large: Why 5-Minute Carbon-Aware Pretraining Needs Fresh Signals More Than Better Forecasts"

ACM e-Energy 2027 (Winter) working paper, Story A ("works-with-design-rules"):
savings survive minutes-scale checkpoint/restore. Central thesis (calibrated
after adversarial review): **decision-error magnitude dominates savings loss;
staleness is the failure mode that generates large errors**, so the operational
lever is signal freshness, not forecast accuracy. Per unit error magnitude,
additive forecast noise is actually more damaging than staleness; staleness
dominates in practice because near-unit-root grid drift makes a stale decision
a large-magnitude error by construction.

- `main.tex` — full paper (acmart `sigconf`), 10 pp incl. references.
- `references.bib` — 45 existence-verified entries; every entry cited, every
  citation resolves (0 undefined).
- `claims_evidence.md` — every numeric claim → committed artifact path + field
  (the Phase D.10 claim–evidence table; estimates flagged).
- `figures/` — Phase C figure pipeline (SVG + PDF) + `closed_loop.tex` (TikZ).

## Building the paper

```bash
cd publication/eenergy
make          # latexmk + bibtex → main.pdf (10 pp incl. references)
make clean    # remove build artifacts (keeps main.pdf)
make fullclean
```

Required: `latexmk`, `pdflatex`, `bibtex`, `acmart.cls` (TeX Live 2025+).
Note: `figures/closed_loop.tex` is `\input`-able into the paper (it uses the
`\paperstandaloneinputsentinel` macro defined in `main.tex`); it still compiles
standalone.

## Reproducibility statement (Phase E.4)

Every number in the paper is produced by a deterministic pipeline that reads
only committed artifacts; re-running produces byte-identical output.

**Environment / software** (the versions actually used):
- Node.js 22.x, `pnpm` 11.21.0, TypeScript `tsc` (6 pre-existing baseline errors
  in legacy UI files, none in the experiment code), Vitest 4.x, tsx.
- Python 3.13 + matplotlib 3.11.1 (in `publication/eenergy/.venv`; the wrapper
  `figures/make_figures.sh` resolves libstdc++/libz on this Nix host).
- TeX Live 2025 with `acmart`; `latexmk` + `bibtex`.

**Data provenance.** 5-minute average carbon intensity (ACI, Electricity Maps)
for grids DE, IT, SE, US, CN, years 2022–2026 (`public/data/co2/{zone}_{year}.json`).
Calibration and all headline experiments use DE, IT, SE.

**Seeds / determinism.** No RNG in the optimizer, simulator, or the decision
models used for headline results (`identity`, `delay`, `arma`/persistence are
deterministic; only the additive/multiplicative *noise* families consume RNG,
using `mulberry32` with seeds 1..N — DE 10 seeds, IT/SE 5). Determinism was
verified by running each experiment twice and `cmp`-ing every JSON/CSV artifact
(byte-identical), and by an end-to-end run of `run_eenergy_experiments.sh`
twice (all 11 tracked summary artifacts byte-identical).

**End-to-end reproducibility.**

```bash
bash publication/eenergy/run_eenergy_experiments.sh   # ~6–10 min, deterministic
```

This runs, in order: the checkpoint-realism sweep (B.0), the adaptive
controller + DTPR benchmark (B.1), the adaptive sensitivity sweep (B.1 fix),
the multi-year robustness + budget sweep (B.3/B.4), the grace-horizon
validation (B.5), and `pnpm test`. Observed wall-clock on the author machine:
552 s and 336 s for two consecutive full runs (hardware-dependent; content
deterministic).

**Base commit.** Experiments and paper were produced on top of commit
`3aff0f2` (`docs: rework spec`); the working tree also carries this study's
artifacts and the paper (uncommitted at the time of writing). Record the final
commit hash before submission (de-anonymization).

## Reproducing every number (individual commands)

```bash
# Phase B.0 — checkpoint-realism sweep (decisive experiment, Story A verdict)
bash publication/output/checkpoint/run_checkpoint_sweep.sh

# Phase B.1 — stale-aware adaptive controller + DTPR benchmark + sensitivity
bash publication/output/forecast/run_adaptive_sweep.sh
bash publication/output/forecast/run_adaptive_sensitivity.sh

# Phase B.3 + B.4 — multi-year robustness + overhead-budget sweep
bash publication/output/forecast/run_multiyear_budget.sh

# Phase B.5 — grace-horizon prediction validation
node publication/output/forecast/grace_horizon_analysis.mjs

# Tests + build
pnpm test
pnpm build

# Figures (Phase C) — generated from committed JSON/CSV, no recomputation
bash publication/eenergy/figures/make_figures.sh
```

## Claim–evidence discipline

The paper reads only committed artifacts under `publication/output/**/*.json`
and `public/data/**`; every number in `main.tex` is traced in
`claims_evidence.md`. Anything not directly in an artifact is marked
**estimate** (e.g. the 1.34 TB checkpoint state, the DeepSeek GPU-hour count)
with its published source. The paper was subject to a claim–evidence audit
(`phase_reports/review_d10.md`, fixes in `phase_d_fix.md`) and two adversarial
reviews (`review_e1.md` → `phase_e1_fix.md`, `review_e2.md` → `phase_e2_fix.md`);
all findings were resolved and re-verified.
