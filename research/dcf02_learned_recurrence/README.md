# DCF-02: Learned recurrent compactification and finite-state quotient

Date: 2026-11-25. Executed exploratory single-thread CPU suite, PyTorch 2.10.0+cpu, NumPy 2.3.5. No GPU, energy, paid compute, trained-language-model or general-matrix-exponent measurement.

## Reproduction

From this directory:
```sh
python train.py --out run
python curriculum.py --prior run --out curriculum_run
python closure.py --out closure_run
python quotient.py --root .
```
The programs save raw JSON, timing samples, and 42 checkpoints (36 trained, six post-hoc SVD derivatives). Full raw receipts, checkpoints and SHA-256 manifest are in the conversation's dcf02_reproducibility.zip artifact, not uploaded to GitHub. The repository contains the executable source and aggregate results. The suite used prospective choices within this exploratory turn, not a blinded preregistered confirmatory study.

## DCF-02A: fixed length

Exact oracle: s in Z/8, op in 0..31, a=2 floor(op/8)+1, b=op mod 8; s'=(a*s+b) mod 8. All maps are permutations. Independent scalar verifier checked 2,048 generated step labels. Three paired seeds 11,23,47; 400 AdamW updates, learning rate .003, batch96, length4, full prefix-state supervision. Separate evaluation RNG by seed and length, 1,024 IID draws per seed per length; random streams may contain duplicate sequences, and operation vocabulary is shared. Dense64 vs rank16 and rank8 vs parameter-matched dense48, plus rank8/rank16 post-hoc SVD of dense64. Models use operation-dependent scale and bias, tied core, tanh and readout. Core rank limits only the shared linear operator; it does not bound the full nonlinear map's complexity.

| arm | total params | core params | L4 % | L8 % | L16 % | latency B1 | B16 | B128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|dense64|9224|4096|54.26|43.23|35.55|1|1|1|
|rank16|7176|2048|75.98|54.04|36.46|1.211|1.148|1.039|
|rank8|6152|1024|77.93|52.38|35.61|1.238|1.166|0.998|
|dense48|6152|2304|43.16|30.53|27.21|0.999|0.961|0.866|
|svd8|6152|1024|54.20|42.42|37.11|1.229|1.161|1.022|
|svd16|7176|2048|54.92|42.90|37.34|1.208|1.188|1.076|

Accuracy is pooled across equal-sized seed streams, not proof of population-level seed generalization. Latency is the median of per-seed/per-length median ratios versus dense64 (lengths4/16, 21 randomized interleaved samples after three warmups); >1 is slower. The rank8 core uses 4x fewer MACs, but end-to-end model latency does not realize those savings at these widths. Training loop wall times total 6.64s, excluding setup/evaluation. No energy result.

Minimum across seeds of the one-sided 95% paired case-bootstrap lower bound (percentage points versus dense64, 2,000 resamples; descriptive, no multiplicity correction):
|arm|L4|L8|L16|
|---|---:|---:|---:|
|rank16|2.15|-2.73|-14.55|
|rank8|12.60|-6.45|-12.41|
|dense48|-24.41|-22.36|-24.22|
|svd8|-5.08|-4.00|-2.05|
|svd16|-1.95|-2.83|-2.15|

Rank8 and rank16 clear the proposed -2pp quality bound at L4, fail at longer lengths. The DCF-01 proposed full quality/energy gate is unpassed. Post-hoc SVD is a separate compression arm and does not reproduce trained-rank gains.

## DCF-02B: equal-label curriculum (FAIL)

Same seeds, initialization, 400 optimizer updates and 153,600 supervised steps, with length/batch/update schedule (4,96,100), (8,48,150), (16,24,150). Independent fresh RNG evaluation, 2,048 cases per seed/length. The fixed-length checkpoints are re-evaluated on identical inputs. Fewer independent sequences, fewer start states and changed gradient correlations confound mechanism. Wall time 10.25s for training loops.

|arm|L4 fixed -> curr|L8|L16|L32|
|---|---:|---:|---:|---:|
|dense64|54.31 -> 25.65|41.86 -> 25.80|36.52 -> 25.08|30.62 -> 24.77|
|rank16|76.89 -> 26.30|52.43 -> 25.21|35.14 -> 24.67|28.32 -> 24.50|
|rank8|78.78 -> 21.16|53.52 -> 18.67|35.22 -> 16.36|26.11 -> 17.07|
|dense48|42.15 -> 16.46|31.02 -> 15.98|26.60 -> 16.15|25.23 -> 15.89|

## DCF-02C: state closure loss (mixed)

Fresh seeds 101,103,107. Both arms have tanh-bounded initial embeddings. Compare CE alone to CE + lambda MSE(h,stopgrad(tanh(embedding(true state)))), lambda=1, on identical batches, 400 updates, length4. This explicitly uses supervised semantic states; no unsupervised latent quotient claim. Fresh eval RNG, 2,048 per seed/length. Training loop wall time 8.34s.

|arm|L4 CE -> closure|L8|L16|L32|
|---|---:|---:|---:|---:|
|dense64|65.69 -> 80.34|48.54 -> 58.87|41.10 -> 48.31|35.07 -> 41.49|
|rank8|79.48 -> 66.10|55.09 -> 47.20|40.06 -> 39.63|32.89 -> 37.19|

Dense64 improves in the mean across all lengths; worst seed case-bootstrap LCB in pp: +8.40, -2.88, -2.20, +0.39. Rank8 worst seed LCB: -34.18, -25.83, -28.86, -25.93. Mean one-step distance to target embedding decreased for all six matched pairs; this is insufficient for long-horizon exactness and rank8 robustness. No uniform contraction certificate.

## DCF-02D: exhaustive finite-state quotient (PASS for 15 selected checkpoints)

For every checkpoint, evaluate all 8*32 possible canonical state-operation inputs. Extract argmax table T and compare each entry to exact oracle f. Seven of 18 DCF-02A checkpoints (including one SVD derivative), zero of 12 curriculum models, eight of 12 closure models pass. Seven passing checkpoints are trained rank8. Minimum target-vs-runner-up logit margin over all 15 passing checkpoints and all 256 entries: 0.19994831. On a common 512-sequence length64 stress fixture, passing checkpoints yield 2,156/7,680 free-running correct vs 7,680/7,680 reset correct and 7,680/7,680 extracted-table correct. There are zero neural-reset/table prefix disagreements across all 42 checkpoints on this fixture. Fixture is shared between checkpoints: do not treat 7,680 predictions as independent unique problems.

**Finite quotient theorem.** If a deterministic table T:S x A -> S agrees with the reference transition f for every pair, and the initial state is represented exactly, then for every finite operation word, repeated application of T yields the reference state. Proof by induction on word length: the empty word returns the initial state; for an extension by operation a, equality at the prefix and pointwise T=f give equality at the next step. This certifies the integer table, independent of floating-point inference. The measured neural margin alone is not a uniform numerical robustness proof across arbitrary kernels, devices or precisions.

A packed generic 8-state x32-op table requires 768 bits (96 bytes); the explicit modular oracle has an even shorter symbolic implementation. The table is an upper bound on representation size, not an information-theoretic lower bound, and it is a task-specific compiled controller. The verifier exploits full knowledge of the finite task domain and exact labels. This does not demonstrate general latent-state discovery, language-model compaction, or a TPU advantage.

## Next registered candidate

DCF-03: extend quotient discovery to partially observed permutation composition with held-out operations and larger state spaces. Separate semantic-supervised state labels from unsupervised quotient discovery, include exact oracle/compiled table and memory-budget controls, hold out operation families and unseen lengths, report exhaustive closure when tractable and verifier coverage otherwise. For neural execution measure fused factorized vs fused dense and dense precomposition on the 5080 at matched task quality, with actual device synchronization and energy telemetry. Require absolute task accuracy floor (e.g. 95% for finite exact tasks) in addition to noninferiority; no model promotion solely because a weak baseline is matched. DCF-02D's selected-checkpoint exactness is a constructive per-checkpoint certificate, not an estimate of training success on an unseen seed family.
