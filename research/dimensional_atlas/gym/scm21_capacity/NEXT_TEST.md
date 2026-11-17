# Proposed follow-up: SCM-21B on the original 3050 host (not executed)

The SCM-21 CPU audit is retrospective. It reveals a decisive comparability gap to close on the original SCM-20 checkpoints once the unrelated ongoing 3050 work finishes.

**Freeze before inference:** Same SCM-20 causal checkpoints and the three previously generated, hash-verified common-eligible pair sets. Do not retrain the models or revisit development thresholds. Preserve the exact source manifests and first 128 common-eligible IDs per stratum. The publicly known final data can only support a *mechanistic post-hoc diagnostic*, not a newly blinded confirmation.

At zero-based visit 1, compare for each pair: (a) actual native write delta, (b) full state difference projected onto native rank-four write span, (c) full state difference projected onto calibration difference-PCA rank-four, (d) **per-pair norm-matched** difference-PCA patch normalized to (b)'s patch norm, and (e) a per-pair norm-matched random rank-four fiber patch. Use identical recipients, donor labels, direction bases, and model weights. Separately log the full-patch positive control and native sham. Record both patch norms, the norm fraction of donor state differences captured by each basis, first-order logit-gradient alignment and exact future donor outcomes.

Primary estimand: matched-pair contrast of (d) minus (b) at same norm and rank on identical examples. Report exact discordant counts `(A_only,B_only)` per seed and stratum, with an exact paired binomial test and a six-stratum breakdown. Pre-register the test before running any *newly generated* data. Follow-up fresh data should cross all seven EvalMath generators with topology independently of family, deduplicate on actual encoded input as well as public signature, and keep test labels sealed until all adaptation decisions are fixed.

No GPU execution is implied by this document. Avoid displacing the ongoing RTX 3050 run.
