# SCM-12 pilot registration, November 14, 2026

The campaign code and fixed protocol were registered before obtaining the external trained RMP checkpoint and D-shared artifact files. This is an exploratory calibration-only instrument. The historical SCM-01 through SCM-11 synthetic pilots retain their original status.

## Locked study

- Model cells: `LOOP-T s1` (`LOOP-T-core4-ds-s1/C3`) and `FF-U s3` (`FF-U-core4-ds-inj-s3/C3`), from the original RMP `phase2.py` registry. Checkpoint and RMP source hashes are frozen in a manifest before capture.
- Dictionary: exactly the RMP-selected D-shared configuration using `select_config`'s `min_ce_recovered >= 0.95` choice and `width,k` ordering. Frozen per-cell decoder feature IDs and their selection provenance are required; the experiment will not choose favorable axes after viewing topology outcomes.
- Tasks: 64 group-unique examples from *selection half* of `monitor_calibration`, balanced 16 per each of `pointer_chase`, `gated_pointer_chase`, `reachability`, `sudoku4`. Seed `20261113` and original SCM-02 task ranking; no reporting-half or held-out evaluation.
- Intervention: after visit 0, token 4, radius 0.05 times that token's clean RMS. Observed downstream token positions 16, 32, and 48. Sample at 32 and 64 angles including duplicated exact endpoint intervention.
- Paired measurements: D-shared decoder-row chart and a seeded random orthogonal chart, retaining the exact same intervention axes within each model. Require identical predictions and logit-change diagnostics under readout-chart replacement.
- Geometry: `SCM01-04` Artin/Alexander/Milnor and winding checks are descriptive for the sampled polygon; `SCM05-11` proof checks are independently reported with their assumptions.

## Primary falsifiers

A model/chart arm may have unresolved crossings, missing Artin witnesses, or poor angular stability. Such cases remain unresolved. If SAE and random charts have indistinguishable valid topology rates, evidence for mechanistically selected topology is null. A change in the model's predictions under a chart-only replacement triggers rejection of the paired campaign. A one-dimensional response verified under the rank-one bound must have trivial braid; a counterexample with source-proven uniform residuals would falsify the mathematical implementation. No causal capability gain is claimed solely from observational association or descriptive sign tests.

## Resource and data contracts

The source checkpoint files and dictionary artifacts remain on the user's RMP machine; SCM-12 does not copy them into Git. All four 64-task capture arms run sequentially under the user-provided CPU/CUDA device. Device preflight and GPU power policies remain the RMP host's responsibility. Source-only numeric validation runs separately and retains its IEEE/BLAS conditional label; captured floating-point topology cannot be certified by a source-only proof.

Run preflight, then freeze a SHA-256 source manifest, execute on the same host, and run paired comparison. Every step refuses overwriting an existing output. The runner stops on missing artifacts or provenance mismatch and preserves any already written partial receipts for audit; only a complete 64-case matched campaign earns a comparative report.
