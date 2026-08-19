# Phase 6 — Reproducibility & cleanliness for the Forecast-Error Sensitivity Study

Status: `[x]` implemented · 169 tests green (167 baseline + 2 new) · build green · 0 new tsc errors (6 pre-existing) · full pipeline byte-identical across reruns.

## 6.1 — Test coverage audit

Every item from the SPEC was audited against the existing suite. **No gaps → no duplicated tests added.** Only genuinely new code (the EPS determinism fix, §Deviations) got 2 new tests.

| # | Requirement | Covered by (file:test) | Status |
|---|-------------|------------------------|--------|
| 1 | Forecast determinism: same seed → identical timeline | `src/domain/forecast.test.ts:148` "is deterministic for the same model and seed" (+ `:154` "differs across seeds") | covered |
| 2 | Identity model ≡ realized → `simulateStepwise` bit-identical | `src/domain/simulation-forecast.test.ts:54` "produces bit-identical SimProgress sequences vs no decisionTimeline" | covered |
| 3 | Decision/accounting split, contrived case | `src/domain/simulation-forecast.test.ts:72` "decision/accounting split" (`:87` pauses on decision value, `:99` accounts at realized intensity) | covered |
| 4 | Clamp ≥ 0 | `src/domain/forecast.test.ts:171` additive / `:177` multiplicative | covered |
| 4 | Head-fallback for delay | `src/domain/forecast.test.ts:182` "delay head-fallback copies realized values" | covered |
| 4 | Head-fallback for AR | `src/domain/forecast.test.ts:192` "arma head-fallback copies realized values" | covered |
| 4 | Length mismatch throws | `src/domain/simulation-forecast.test.ts:113` carbonIntensity / `:120` timestamps | covered |
| 5 | AR fit recovery on synthetic series | `src/domain/forecast.test.ts:93` "recovers AR(1) coefficients" (+ `:107` innovation scale) | covered |
| 5 | RNG distribution sanity (mean ≈ 0) | `src/domain/forecast.test.ts:64` "mean ~0 and std ~1 over 10000 draws" | covered |
| 6 | `runOptimization` with identity forecast == without | `src/domain/simulation-forecast.test.ts:132` + `src/cli/forecast-sweep.test.ts:455` (runtime assert also enforced in `forecast-sweep.ts:575,710`) | covered |

Added (new code only): `src/cli/plot-forecast.test.ts` — `normalizeEps` strips the cairo `%%CreationDate` header line / leaves non-dated EPS untouched (2 tests).

## 6.2 — `publication/run_forecast_experiments.sh` (new, executable)

Mirrors `publication/run_experiments.sh` style: `#!/usr/bin/env bash`, `set -euo pipefail`, `ROOT`/`OUT_DIR` computed the same way, `PNPM="pnpm"`. Header documents the pipeline (calibrate → fixed → reopt → plots), seeds (fixed: DE 10, IT/SE 5; reopt: 3), policies (DE 272/268 @ 02-01, IT 246/231 @ 01-14, SE 19/18 @ 04-22), budget 200, α=1, and that it appends to `publication/experiments_run.log`.

Steps (each under an echoed section header, all run from `$ROOT` so relative artifact paths are robust):
1. `pnpm cli forecast-calibrate`
2. `pnpm cli forecast-sweep --mode fixed -o $OUT_DIR/fixed --csv $OUT_DIR/fixed_all.csv`
3. `pnpm cli forecast-sweep --mode reopt -o $OUT_DIR/reopt --csv $OUT_DIR/reopt_all.csv`
4. `pnpm cli plot-forecast`

Whole run wrapped in `{ ...; } 2>&1 | tee -a "$LOG"` behind a timestamped separator (`==== <UTC> ==== run_forecast_experiments.sh …`), so stdout+stderr stream to the terminal and are appended to `publication/experiments_run.log`. Total elapsed time recorded at the end (`==== done in … ===== N s elapsed ====`). Idempotent: every run overwrites the same deterministic artifacts. `chmod +x`.

## 6.3 — Artifact hygiene

- All study evidence lives under `publication/output/forecast/` (calibration, fixed, reopt JSON/CSV + `README.md`), the 4 figures as `forecast_*.{svg,eps}` under `publication/ICREC_Rome/assets/` — verified, 23 + 8 = 31 files.
- `git check-ignore publication/output/forecast/calibration_DE.json` → **not ignored** (evidence is committed); only `publication/experiments_run.log` is ignored (via `publication/.gitignore` `*.log`). `publication/.gitignore` untouched.
- No pipeline stray junk: no `dist/`, no `.log` inside `src/`, no temp files. Verified via `git status --short` (§ below).

## 6.4 — End-to-end reproducibility proof

1. Snapshot `sha256sum` of all 23 files under `publication/output/forecast/` + the 8 `forecast_*` assets → `/tmp/opencode/forecast_before.sha256`.
2. Ran `bash publication/run_forecast_experiments.sh` end-to-end twice (second run after EPS fix, timed):

```
$ time bash publication/run_forecast_experiments.sh
… 144.89s user 1.67s system 101% cpu 2:24.01 total
SCRIPT_EXIT=0
```

3. Re-snapshot → `/tmp/opencode/forecast_after.sha256`; `diff` of before/after exits 0:

```
$ diff forecast_before.sha256 forecast_after.sha256 ; echo "exit=$?"
exit=0        # BYTE-IDENTICAL — all 31 files identical before == after
```

Proof (both snapshots, before == after, 31 lines):

```
b74f549b7d524c5c42bc8c718ab3dac2ec86784b0029853ccd3fa4a1d724661f  publication/output/forecast/calibration_DE.csv
62ecfe126cc6fc407221a26f41abb31aba236e9616810c31a667580ac058f5fe  publication/output/forecast/calibration_DE.json
9ece95bbc61dc99f41c6d7802b12dc882be00c2addd1ca0dba01e550751ea5ad  publication/output/forecast/calibration_IT.csv
15c0f7a48d551df5f1bdb101d0fb7cba66cbf8f0bb4a1204c080752c43da34f4  publication/output/forecast/calibration_IT.json
55099432427f3d4787d390946c73e3f80a08ef3130c686ea9166306d910883d6  publication/output/forecast/calibration_SE.csv
8ec6496dcfcca64d928cd1c963b0effb30615c1fd7bae73f6ae4f5f01a0b7e3e  publication/output/forecast/calibration_SE.json
6f79c053f3d6ae8dc19f86b54b9c4e537ff757661add2e13c947a19fbdce9995  publication/output/forecast/fixed_all.csv
1ec43329348835ed5577702903a8e3560b5ae767fa3fdb8b9b07601e581bfe47  publication/output/forecast/fixed_DE.csv
0ebb105444e5bbac332b2fa806b101f2f64ca0972a8c18e1edf5726987461e1c  publication/output/forecast/fixed_DE.json
2aeeeb50e0aa6358d552b790cc760a4e6e6d98f461e4951de03aa9bfa8d91231  publication/output/forecast/fixed_IT.csv
5a51426ebfa180b0570a091cf141c41246f50f7ff03b5a609bad291b098ca73f  publication/output/forecast/fixed_IT.json
5097691c4abfce7031ef8bd11b3debcd14c518bc5ab690cde8966adc4afcfb82  publication/output/forecast/fixed_SE.csv
306e95fc2a4fd360284b960c58877efc67ac35935183188fdf1ab38c1ca6394d  publication/output/forecast/fixed_SE.json
e16d35f2e182cf3c9dde59ca3dc8c125876afe8f970c7b0f9238fddf40ae65b1  publication/output/forecast/fixed_summary.json
428c71e807dc48c3e7704054debccaba9c3307bfbcd4ee947dff43cbb515405a  publication/output/forecast/README.md
abcac7e126592f5d472ccaa3748e0e8815682ae846fb5e0550f2466f91873517  publication/output/forecast/reopt_all.csv
3c3debab779babcd07ee76e33a108609298057e61ba3539070fee346d2c44de7  publication/output/forecast/reopt_DE.csv
d61ede7a8f8168f78b781800fb5cfb2a52219d09ebc290bd45227f6929587a40  publication/output/forecast/reopt_DE.json
8f7c1ae56bf935a81511e51b2a91856d873471a202745d3741be205c737b333d  publication/output/forecast/reopt_IT.csv
52face6e7844bd5a53dd7cdc721a2f5255852f77f66000b3a7cb4977c4191bef  publication/output/forecast/reopt_IT.json
f54a6ff3c17ffd9d4b94a2feb05bfae9b8d0f3a084c29a321d736271ddcfc37e  publication/output/forecast/reopt_SE.csv
d9b44b0b3537605fe9d8890c6b9e38b73e7f5052c63356d4acebbd18a6f2477b  publication/output/forecast/reopt_SE.json
123dee69aac035436f68a38dbad6f2f0c3bc6055d2ac8b9e62d6455c5f3478fc  publication/output/forecast/reopt_summary.json
ef732655ed37943d38a06beabe93e2feba973dcc4d70881d7652f55662e056b9  publication/ICREC_Rome/assets/forecast_degradation.eps
ef5337e9bfb617de4cd6663a97b4daaadc132406d7d8aa56ee07552672e1d9cb  publication/ICREC_Rome/assets/forecast_degradation.svg
d9f882d449fa9345d2c78fab79bb04af10b6ddd5a7b391e0c02f19c27a5c2812  publication/ICREC_Rome/assets/forecast_reopt_drift.eps
03b4da07fd44302e4ac92057b07f1ab207df1d294b7e9fe583ae05bc567f5188  publication/ICREC_Rome/assets/forecast_reopt_drift.svg
423b59474594ff8225970baddde2d8fad8f1f0426b4ae00872f9d7d14598fc54  publication/ICREC_Rome/assets/forecast_rmse_horizon.eps
07e5561a4fe1254cc8d4b43e090d4125a8b6db055808c2c12f7aae2a47fff110  publication/ICREC_Rome/assets/forecast_rmse_horizon.svg
850a99ef8466f805faf9eb386696000d1ec22b36dc1e5a47a5b5e5616b019760  publication/ICREC_Rome/assets/forecast_savings_overhead_score.eps
026dfb3be9a3aeb1558400fd8b5366a87a1b73b7f62f4d4538a73f38467607ef  publication/ICREC_Rome/assets/forecast_savings_overhead_score.svg
```

4. `publication/experiments_run.log` gained 2 timestamped sections (one per end-to-end run), each with a leading `==== <UTC> ==== run_forecast_experiments.sh …` separator and a closing elapsed-time line. Tail:

```
  /home/simon/dev/TheGreenEpoch/publication/ICREC_Rome/assets/forecast_reopt_drift.eps
  /home/simon/dev/TheGreenEpoch/publication/ICREC_Rome/assets/forecast_rmse_horizon.svg (60129 bytes)
  /home/simon/dev/TheGreenEpoch/publication/ICREC_Rome/assets/forecast_rmse_horizon.eps
  Done (4 figure(s) in /home/simon/dev/TheGreenEpoch/publication/ICREC_Rome/assets)

==== done in 2026-08-18T11:24:15Z ===== 144 s elapsed ====
```

5. Verification:
   - `pnpm test` → `11 files passed · Tests 169 passed` (167 baseline + 2 new).
   - `pnpm build` → green (`✓ built in 2.70s`).
   - `npx tsc --noEmit` → exactly 6 `error TS` lines, the same 6 pre-existing errors (jest-dom types, CO2Chart annotations, LiveSimPage, vite.config) — **0 new**.

6. `git status --short` (expected set only — no strays, no tmp, no `*.log` tracked, no node_modules):

```
 M src/cli/index.ts
 M src/domain/index.ts
 M src/domain/optimize.ts
 M src/domain/simulation.ts
 M src/domain/types.ts
?? phase_reports/
?? publication/ICREC_Rome/acm.md
?? publication/ICREC_Rome/assets/forecast_degradation.eps
?? publication/ICREC_Rome/assets/forecast_degradation.svg
?? publication/ICREC_Rome/assets/forecast_reopt_drift.eps
?? publication/ICREC_Rome/assets/forecast_reopt_drift.svg
?? publication/ICREC_Rome/assets/forecast_rmse_horizon.eps
?? publication/ICREC_Rome/assets/forecast_rmse_horizon.svg
?? publication/ICREC_Rome/assets/forecast_savings_overhead_score.eps
?? publication/ICREC_Rome/assets/forecast_savings_overhead_score.svg
?? publication/output/forecast/
?? publication/run_forecast_experiments.sh
?? SPEC.md
?? src/cli/forecast-calibrate.test.ts
?? src/cli/forecast-calibrate.ts
?? src/cli/forecast-sweep.test.ts
?? src/cli/forecast-sweep.ts
?? src/cli/plot-forecast.test.ts
?? src/cli/plot-forecast.ts
?? src/domain/forecast.test.ts
?? src/domain/forecast.ts
?? src/domain/simulation-forecast.test.ts
```

## DoD

- [x] 6.1 audit table: each item → file:test(s) + status (all covered; 2 tests added for new code).
- [x] `run_forecast_experiments.sh` created, executable (`-rwxr-xr-x`), mirrors `run_experiments.sh` style, appends to `experiments_run.log` with timestamps + runtime.
- [x] End-to-end rerun: byte-identical outputs — sha256 before == after for all 31 files (`diff` exit 0), proof pasted above.
- [x] No stray files; `git status --short` is exactly the expected file set.
- [x] `pnpm test` (169) / `pnpm build` green; `npx tsc --noEmit` 0 new errors (6 pre-existing unchanged).

## Deviations

1. **EPS determinism fix (only code change beyond the new script).** First end-to-end run showed all 23 evidence files + all 4 SVGs byte-identical, but the 4 `.eps` differed: `rsvg-convert`/cairo embeds `%%CreationDate: <current time>` in the EPS header. Fixed in `src/cli/plot-forecast.ts` by adding `normalizeEps()` (strips `%%CreationDate`, exported, +2 tests in `plot-forecast.test.ts`) called after each `rsvg-convert`. After the fix, a second full pipeline run was byte-identical across all 31 files. SVG (the actual figure content) was always deterministic; only the header timestamp was not.
2. Snapshot/cmp artifacts and the two run logs were kept in `/tmp/opencode/` (outside the repo), so nothing stray enters the tree.
