# Phase 0 — Scope & Orchestration Baseline

Status: `[x]` decided · orchestrator artifact, not produced by a subagent.

## Decisions (confirmed with study owner)

1. **Fixed-policy θ values (Phase 3):** the *published* rounded integers per
   Table 2 / SPEC reference facts — DE `(272,268)`, IT `(246,231)`, SE `(19,18)`.
   `S₀` is computed fresh at these integer values; `degradation_frac` is relative
   to that fresh `S₀`.
2. **Fixed start dates (Phase 3 + Phase 4 re-opt):** the Table-2 optimal start
   dates per region, recovered by re-running the paper's exact optimizer
   settings (`-m Deepseek -y 2025 --budget 200 --resolution 10 --date-res 7
   --max-iter 10 --alpha 1`, tp-max 800 / 100 for SE):
   - DE: start `02-01`
   - IT: start `01-14`
   - SE: start `04-22`
3. **Experiment matrix:** run the FULL SPEC matrix (Phase 3 + Phase 4), using
   the 12-core machine with process-level parallelism. No reduction.
4. **Table-2 regression check (DONE):** the paper's optimizer was re-run for
   DE/IT/SE and reproduces Table 2 exactly:

   | Region | θ_p | θ_r | Savings | Overhead | Start |
   |--------|-----|-----|---------|----------|-------|
   | DE     | 272.75 | 267.94 | 43.35% | 174.3% | 02-01 |
   | IT     | 246.7  | 230.45 | 32.67% | 194.7% | 01-14 |
   | SE     | 18.8   | 17.8   | 23.17% | 106.3% | 04-22 |

   (Float optima recorded here for reference; published integer policies used
   per decision 1.)

## Baseline snapshot (before any study code)

- `pnpm test` → **91 tests pass** (6 files).
- `pnpm build` → green (vite).
- `npx tsc --noEmit` → **6 pre-existing errors** unrelated to the study
  (jest-dom types, `src/components/CO2Chart.tsx:100`,
  `src/pages/LiveSimPage.tsx:22-23`, `vite.config.ts:14`). DoD for all phases:
  **no NEW tsc errors** in files touched by the study (full baseline in
  `/tmp/opencode/tsc_baseline.txt`).
- Machine: 12 cores, 15 GiB RAM, 421 GB free.

## Reference facts (fed to all phase agents)

- Grid stats 2025: DE mean 380, σ=142, CV=0.37; IT 309/87/0.28; SE 23/8/0.35;
  lag-1 autocorrelation ≈ 0.999.
- Data: `public/data/co2/{zone}_{year}.json`, 5-min resolution, 105 120 pts/yr.
- DeepSeek profile: 2048 GPUs, 14.8T tokens, train 700 W / pause 60 W, PUE 1.27,
  checkpoint pause 148.8 s, resume 0 s. Budget B=200%, α=1.
- CLI entry `src/cli/index.ts` (commander); tests via vitest; plots via vega-lite.

## Orchestration plan

- Phase 1 (core impl) → Phase 2 (calibration) → Phase 3 (fixed sweep) →
  Phase 4 (reopt) → Phase 5 (plotting) → Phase 6 (repro/tests) → Phase 7 (paper).
- Phase 8 (ACM prep hooks) folded into Phase 6/7 notes.
- Each phase: subagent writes code + `phase_reports/phase_X.md`; orchestrator
  reviews; adversarial review + fix agent on failure.
- Single git commit at the very end (no GPG signing, no push).
