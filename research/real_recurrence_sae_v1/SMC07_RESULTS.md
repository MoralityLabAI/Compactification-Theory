# SMC-07: First source-locked recurrent SAE results and causal controls

**Executed November 15, 2026.** Code and frozen protocol: [SMC-07](README.md). Source-locked hosted run: [GitHub Actions #43702825317](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43702825317). Full 64-puzzle/per-seed/per-puzzle receipt and clustered analysis are together in the artifact **smc07-pinned-64-puzzle-causal-receipt** (artifact ID 14156234539). Pilot and full jobs completed successfully. Earlier initial workflow failures were caused by an optional NumPy-dependent provenance hash and repaired by standard-library FP32 hashing; the final successful full run uses the corrected code.

## Source boundary

- External, independently pretrained **44,001-parameter** TRM-inspired Sudoku network, four iterations of one shared block, 81 tokens × 48 channels; **not** the official 7M TRM and **not** Qwen.
- Upstream commit: f0406195fece4d0db29ced28f3a69d1993d95b1e
- Pretrained weight SHA-256 (checked before weights-only deserialization): 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1
- Fixed 64-puzzle roster hash (checked): eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611
- Split: 32 train puzzle IDs, 16 validation IDs, 16 heldout IDs using ID modulo 4. All tokens from one puzzle remain within its cohort.
- SAE fitting: stage-one recurrent states from *only training puzzles*. Validation reconstruction MSE freezes the fitted SAE checkpoint, and heldout evidence is never used for training. Seeds 71,89,113, 300 optimizer updates each.
- Neural model: hard top-four ReLU SAE 48→96→48, trained in standardized train coordinates. This study does not posit a K6 incidence geometry in the source-model activation dictionary.

## Unsupervised reconstruction and feature persistence

The following numbers come from the **16 heldout puzzles**, including all 81 tokens in the normalized stage-one reconstruction metric.

| SAE fit seed | SAE normalized MSE | PCA4 normalized MSE | Random4 normalized MSE | Matched-token depth1→2 feature overlap | Puzzle-shuffled overlap |
|---:|---:|---:|---:|---:|---:|
| 71 | 0.166308 | 0.526079 | 0.909655 | 0.633295 | 0.121914 |
| 89 | 0.166305 | 0.526079 | 0.909655 | 0.620949 | 0.114390 |
| 113 | 0.169059 | 0.526079 | 0.909655 | 0.664545 | 0.132523 |

SAE average positive activation count is 4.0 in every fitted run. This is the deliberately enforced top-k setting. The SAE's four active coefficients are chosen out of 96 possible dictionary atoms. PCA4 and random4 provide four **fixed subspace** coordinates, so reconstruction comparisons do not control total decoder parameters or index-selection bits.

The average puzzle-clustered difference in top-four overlap between matched depth-one/depth-two hidden states and a puzzle-shuffled control is **+0.51665** (descriptive 95% percentile bootstrap interval **[0.49685, 0.53549]**, 16/16 puzzle groups positive). This demonstrates reproducible recurrence-step feature-index persistence under the *same fitted encoder*. It is not direct evidence that a given SAE atom corresponds to a human-interpretable semantic concept.

## Goal-token reconstruction through the original recurrent teacher

The measurement substitutes one reconstructed stage-one goal token, then runs the **same frozen pretrained block** three more times before reading out the goal logits.

| Seed | SAE output logit MSE | SAE teacher-label agreement | PCA4 output logit MSE | PCA4 teacher-label agreement |
|---:|---:|---:|---:|---:|
| 71 | 0.72610 | 15/16 | 6.49168 | 8/16 |
| 89 | 0.73416 | 15/16 | 6.49168 | 8/16 |
| 113 | 0.67701 | 15/16 | 6.49168 | 8/16 |

The SAE-preserved *ground-truth Sudoku goal cell* correctness is 13/16 for all three seeds in this protocol; PCA4 gets 8/16. As the hidden perturbation sizes also differ (SAE mean delta norm 3.74–3.94, PCA4 5.78), this is a reconstruction/fidelity comparison between different-capacity representations, **not** a matched-norm causal attribution.

## Matched-norm causal feature-direction interventions

At the first missing token of each heldout puzzle, perturb the *original* teacher stage-one state, continue the full recurrence, and measure final goal logit L2 change. All active, inactive and isotropic directions have the **same L2 perturbation radius**. The active SAE direction is the learned decoder atom of the highest active feature at that token. Remote-token intervention uses the same active direction at the next token. Fixed radii are 0.2 and 0.8.

For radius **0.2**, mean final goal-logit changes were:

| SAE seed | Active feature | Inactive feature | Isotropic | Same active direction at remote token |
|---:|---:|---:|---:|---:|
| 71 | 0.15838 | 0.14030 | 0.06967 | 0.04389 |
| 89 | 0.22547 | 0.12566 | 0.09530 | 0.06235 |
| 113 | 0.17153 | 0.10161 | 0.08299 | 0.05020 |

All radius-0.2 active interventions preserved the teacher's goal-cell top prediction (0/16 flips per seed). A larger logit displacement measures **sensitivity**, not a useful or harmful semantic effect.

### 16-puzzle-clustered paired differences

To avoid treating 48 seed-by-puzzle observations as independent, each puzzle's active-minus-control effect is first averaged across three SAE seeds, then the **16 puzzle IDs** are bootstrap-resampled 20,000 times with replacement using fixed seeds. Intervals are descriptive percentile intervals.

| Intervention radius | Active minus control | Mean difference | 95% puzzle-bootstrap interval | Puzzle groups positive |
|---:|---|---:|---:|---:|
| 0.2 | Inactive atom | +0.06260 | [+0.02535, +0.10328] | 11/16 |
| 0.2 | Isotropic | +0.10247 | [+0.05956, +0.14785] | 15/16 |
| 0.2 | Same direction at remote token | +0.13298 | [+0.08474, +0.18307] | 16/16 |
| 0.8 | Inactive atom | +0.33277 | [+0.17214, +0.50489] | 13/16 |
| 0.8 | Isotropic | +0.48422 | [+0.29724, +0.67777] | 15/16 |
| 0.8 | Same direction at remote token | +0.61821 | [+0.40888, +0.83795] | 16/16 |

These are measured, finite-puzzle effects from one source-locked model and a fixed intervention grammar. They show that *highly active learned SAE decoder directions* tend to be more causally sensitive than the selected controls at the same norm and site. They do **not** imply that the decoder atoms form a faithful causal factorization, encode matching geometry, or improve Sudoku reasoning.

## Exact structural control

At the **stage-two pre-LayerNorm2 bus**, adding an equal scalar to all 48 channels of one selected token should be eliminated by LayerNorm. In double-precision replay of two heldout puzzle contexts, the maximum subsequent goal-logit change is **1.33×10^-15**. Matched-norm centered transverse perturbations alter the goal logits by approximately **0.159** and **0.281**. This confirms the expected structural gauge invariance at that named interface; it is distinct from the stage-one SAE feature interventions.

## Evidence scope and next test

Successful hosted jobs are recorded in [SMC-07 Actions #43702825317](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43702825317); source [smc07_real_sae.py](smc07_real_sae.py), [clustered analysis](analyze_smc07.py), and toy-model / analysis regression suites accompany the receipt.

Needed follow-ups: an **equal-parameter/bit-budget** nonlinear baseline, train/validation diversity beyond the highly structured Sudoku generator, a causal ablation-by-feature projection that can be checked against an intervention on the original post-LN state, and tests of actual feature dictionary stability under alternate checkpoint/SAE seeds. The current sample of 16 heldout puzzles cannot establish general model interpretability or performance gains outside the declared tasks.
