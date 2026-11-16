# SMC-17: Inactive-preactivation-matched controls narrow SAE feature causality

**Executed November 16, 2026.** The prospectively frozen [SMC-17 protocol](SMC17_PROTOCOL.md) was committed at [59bad2ad](https://github.com/MoralityLabAI/Compactification-Theory/commit/59bad2ad422faaf3b7bfe7410f7b48656cb279f0), before its fresh heldout state evaluation. The experiment follows [SMC-16 PR #177](https://github.com/MoralityLabAI/Compactification-Theory/pull/177), which retains its 3/4 gate. [SMC-17 PR #179](https://github.com/MoralityLabAI/Compactification-Theory/pull/179) is a distinct stacked draft.

The **source-pinned hosted run [#43800600355](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43800600355) passed** both the four-puzzle pilot and 128-puzzle/full four-pair experiment, including the separately written standard-library receipt auditor and mutation regression tests. Green execution reports scientific outcomes separately: **SMC-17 passed the registered two-family gate for 1 of 4 receiving SAE pairs**. Its other three pair-level failures remain unchanged.

## Source-locked experiment

The original pretrained Sudoku recurrent teacher A SAE71 and independently trained teachers B501/B701 (SAE71 and SAE89) were loaded without altering their weights. All seven original SMC-13C position+visible-digit-conditioned Hungarian assignments were reconstructed and hash-checked before loading any SMC-17 test states.

Two novel full-solution-disjoint 64-puzzle Sudoku cohorts used 30 holes (seeds 202617030000..202617030063) and 34 holes (202617040000..202617040063), with exact unique-solution checks and positive intercalate invariant. Their targets and visible inputs are distinct from all SMC-13, SMC-15 and SMC-16 training/calibration/test families.

An eight-position visible-mask rule selected missing goals. Only a source-top-one/mapped-target-active goal passing the original SMC-16 fully selective erasure constraints was eligible. Its minimum-norm physical encoder-row kill (overshoot 0.02) and original equal-dose tangent sham remained fixed. A further label-blind **131-candidate** search chose an equal-dose tangent sham whose 92 previously inactive SAE *affine encoder preactivation* shifts were most similar to the kill: original sham, positive and negative normalized least-squares directions, and 128 deterministic Gaussian tangents. Every candidate was screened with the original full SAE reencoding guard before any task output was read.

Each arm then changed one stage-one Sudoku token in an unchanged teacher with three FP64 recurrence steps. The primary effect is the correct-digit-minus-clean-rival margin drop of the **selective kill minus the best inactive-matched sham**. Two thousand resamples of 64 independent puzzle IDs give descriptive 95% intervals.

## Complete registered result

| Receiving SAE | Fresh 30-hole effect [95% puzzle bootstrap] | Eligible goals / puzzles | Fresh 34-hole effect [95% puzzle bootstrap] | Eligible goals / puzzles | Both-family gate |
|---|---:|---:|---:|---:|---|
| B501 SAE71 | +1.2241 [0.6419, 1.7793] | 26 / 23 | +0.8454 [−0.1308, 1.8099] | 29 / 23 | **FAIL** |
| B501 SAE89 | +0.2780 [−0.6685, 1.0290] | **17 / 15** | +1.2078 [0.0050, 2.4440] | 30 / 23 | **FAIL** |
| B701 SAE71 | +1.4566 [0.6659, 2.3144] | 42 / 33 | +1.0114 [−0.0029, 2.0580] | 31 / 25 | **FAIL** |
| B701 SAE89 | +1.5261 [0.7874, 2.2830] | 47 / 35 | +1.1524 [0.4095, 2.0061] | 45 / 35 | **PASS** |

**Registered positive gate per pair and family:** at least 24 eligible goals over 20 distinct puzzles, positive 95% paired puzzle-bootstrap lower bound, more clean-correct-to-wrong flips after kill than after the best sham, and pooled inactive shift mismatch at least 10% smaller than the seeded tangent control. Both fresh families must pass. In the B501 SAE71 novel30 condition, kill flips were **1** versus **3** for the optimized sham, hence the gate fails despite a positive mean and interval. B501 SAE89 novel30 fails coverage (17 goals in 15 puzzles) and its interval crosses zero. B501 SAE71 novel34 and B701 SAE71 novel34 have intervals crossing zero. The B701 SAE89 comparison passes both registered cohorts.

The original seeded tangent sham remains in all receipts as a historical within-experiment control. Differences in candidate geometry are selected entirely without task labels.

## Geometry findings: the remaining inactive-feature confound has a rank obstruction

The preregistered matching search reduced the *pooled inactive preactivation residual norm* to the following fractions of the original sham residual:

| Receiving SAE | 30-hole best / original | 34-hole best / original |
|---|---:|---:|
| B501 SAE71 | 0.619 | 0.569 |
| B501 SAE89 | 0.571 | 0.517 |
| B701 SAE71 | 0.571 | 0.600 |
| B701 SAE89 | 0.547 | 0.577 |

Among the **267 eligible pair/family/goal interventions**, every receiving SAE's 92×48 inactive encoder block had **full column rank 48**. Under this full-rank condition, exact inactive-preactivation matching with a four-active-feature-preserving tangent sham would require the sham displacement to equal the kill, contradicting the nonzero kill of an original active feature. The independent [OLM-17 mathematics, PR #100](https://github.com/patrickdugan/MathCan/pull/100) proves this rank obstruction and derives a least-squares residual floor.

Across the eight pair/family conditions, median *least-squares floor / achieved best residual* lies between **0.980 and 0.997**. The least-squares positive direction was selected in **216 of 267** eligible cases. The fixed finite-budget sham therefore approaches the **unconstrained** residual floor closely in most cases, although it remains an approximate finite candidate search and its equal-norm feasibility may be more restrictive than that lower bound.

No posthoc sample filtering, altered search budget or revised scientific gate was used to produce these results.

## Raw receipts and independent replay

- Hosted workflow: [SMC-17 Actions #43800600355](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43800600355), full/pilot successful on commit \`efcc5f65f7693cf40015702f2b3f41b80cdbd21f\`.
- [Original full experimental and audit artifact #14195762984](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43800600355/artifacts/14195762984), ZIP SHA-256 \`d2aa384b276e6a92f05950ca958f2a71525e6841d47bb920b874f3504ed7b9a3\`.
- Raw 18,204,654-byte \`full.json\` SHA-256 \`c3424e3401c7ec60e36de164c4d423fc2dbe393ba4cb3fced1d9c6f5bb8c49ed\`; source \`audit.json\` SHA-256 \`d214624327bd134a38a2c1428673347b0fff11c8bd5bbdba82587638a28f1a7b\`.
- [Machine-readable eight-condition results](SMC17_SUMMARY.json) preserves all accepted and failed gates, coverage, flip counts and source digests. An additional independent archived-JSON numerical replay independently reconstructed every paired effect, cluster bootstrap, inactivity ratio and gate; it agrees with the source reports on all eight conditions.

The hosted independent auditor checks source hashes, full 96-coordinate preactivation displacements, 131 candidate records, source-locked goal selection, registered confidence intervals and gates. It replays the saved numerical observations independently from PyTorch. It does **not** recompute the network forward passes from raw weights and does not supply an Arb interval certificate.

## Interpretation and next rigorous target

The surviving B701 SAE89 pair shows stronger **operational selective-feature causal specificity** under a difficult inactive-feature-matched sham. The other three pairs fail their original preregistered requirements. The estimated Sudoku model capability gap, sparse eligible feature coverage, shared task/teacher architecture and unavoidable remaining inactive-feature displacement constrain any claim of a stable cross-model reasoning feature.

Since the inactive encoder rows have full column rank, the next informative test should examine **task-gradient projections** rather than further inactive-coordinate sham optimization. It should estimate the clean task-margin gradient along the selected active feature direction and quantify the inevitable inactive leakage contribution with second-order residual checks, using fresh cohorts and a new preregistration. The proof-level relationship between the attribution error and the inactive spectral residual is documented in [OLM-17 PR #100](https://github.com/patrickdugan/MathCan/pull/100).
