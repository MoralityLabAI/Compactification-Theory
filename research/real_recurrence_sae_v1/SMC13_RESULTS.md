# SMC-13: Independent recurrent teachers, an orbit-disjoint Sudoku family, and SAE controls

**Executed November 15, 2026.** Completes the next falsification gate after SMC-12's 157/288 trained-vs-1/288 initialized SAE decoder self-selection cones. Companion pure mathematics: [MathCan OLM-13](https://github.com/patrickdugan/MathCan/pull/89).

The results below come from three independently successful source-locked GitHub Actions runs:
- [SMC-13A: newly trained recurrent teacher checkpoints and matched Gaussian SAE controls](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43728006836), [complete checkpoint artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43728006836/artifacts/14166777873).
- [SMC-13B: calibration-only cross-model feature permutation alignment](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43728732508), [complete alignment artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43728732508/artifacts/14167073269).
- [SMC-13C: position-and-visible-clue-conditioned feature transfer](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43729335868), [complete conditioned-alignment artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43729335868/artifacts/14168165450).

## 1. Structural novelty is proved by an exact Sudoku orbit invariant

The original source's pinned full-grid Sudoku generator forms solutions through digit permutation, row-within-band and band permutation, column-within-stack and stack permutation, and transposition of a single canonical cyclic grid. It generates solutions in that grid's transformation orbit.

Define the intercalate count I(G) as the number of alternating two-digit 2×2 rectangles in a completely solved 9×9 Sudoku. This is invariant under *all* permutations of rows and columns, under digit relabeling and transposition. The canonical cyclic grid has I=0, verified by exhaustive counting of all 1,296 candidate rectangles. Thus I(G)>0 is an exact obstruction to belonging to the canonical source generator's solution orbit.

A newly implemented randomized backtracking Sudoku completion, with randomly filled diagonal boxes, produces valid full solutions outside that orbit. It accepts only templates with I>0 and removes clues while checking solution uniqueness with an independent exact Sudoku solver.

The new source-locked cohorts are:

| Cohort | Puzzles | Missing cells | Target intercalate range | Training role |
|---|---:|---:|---:|---|
| Novel teacher train | 192 | 30 | 8–36 | New teacher and SAE training |
| Novel teacher validation | 64 | 30 | 10–28 | Teacher and SAE checkpoint selection |
| Novel feature calibration | 64 | 30 | 8–31 | Feature matching only |
| Novel heldout | 64 | 30 | 9–33 | Evaluation |
| Novel harder heldout | 64 | 34 | 9–31 | Evaluation |
| Canonical heldout | 64 | 30 | 0 | Evaluation |

All 448 noncanonical and 64 canonical complete solutions are distinct, all puzzles are unique-solution, all split seeds are disjoint and no target appears in another cohort. The exact manifest and per-cohort SHA-256 values are preserved in the training artifact. The invariant establishes distribution-level structural novelty relative to the *pinned source generator's orbit*. It does not characterize every possible full-grid example in the upstream model's original, externally conducted training.

## 2. New independent recurrent teacher checkpoints

Two new teachers use the same public MIT-licensed 44,001-parameter 4-step recurrent Sudoku architecture as the source A, but are **initialized and trained from scratch** with seeds 501 and 701. No pretrained source tensor is used to initialize these teachers. Both are trained only on the 192 novel 30-hole puzzles using weighted cross entropy (missing cells weight 3, givens weight 1), AdamW with 600 updates, batch size 16. The selected checkpoint uses *minimum validation missing-cell cross entropy* at scheduled evaluation steps 25,50,...,600. Both teachers selected training step 175.

All scores below are missing-cell accuracy on the designated untouched **64-puzzle cohorts**. The original source teacher is the independently pretrained third-party checkpoint with SHA-256 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1, and is evaluated without further training.

| Teacher | Noncanonical 30-hole | Noncanonical 34-hole | Canonical 30-hole |
|---|---:|---:|---:|
| Source pretrained A | **95.83%** | **91.36%** | **95.73%** |
| New B seed 501 | 46.61% | 40.95% | 47.29% |
| New B seed 701 | 45.26% | 44.53% | 46.30% |

The original pretrained teacher retains high accuracy on a new solution-template family, while the deliberately bounded newly trained teachers are substantially weaker. **This capability mismatch must be considered when comparing their SAEs**. The experiment does not show that source A was trained exclusively on the canonical orbit; it shows its pinned source generator produces that orbit and its external model performs well on the independent solved-template holdout.

The two saved new checkpoint weights and four newly fitted SAE weight snapshots are in the complete artifact, each with a strict weights-only PyTorch reload, model-state tensor-byte SHA-256 and original source/novel-roster provenance.

## 3. Dictionary-wide self-selection is not reasoning-specific

Fit two independent 48→96→48 hard-top-four ReLU SAEs on each new teacher's stage-one training activations, using the same 300-update unsupervised SAE recipe and validation-only selection as SMC-12. Then compute all 96 exact WD column self-selection conditions per SAE with rational arithmetic.

The matched **Gaussian SAE control** has the same train/validation tensor shape (192×81×48 and 64×81×48), source normalization procedure, decoder dimensions, training budget and SAE seeds, but receives independent standard-normal activations rather than any recurrent model's hidden states.

| SAE training data | SAE seed 71 strict self-selection cones /96 | SAE seed 89 /96 |
|---|---:|---:|
| New teacher B 501 | 81 | 75 |
| New teacher B 701 | 76 | 71 |
| IID Gaussian activations | **95** | **94** |

These are exact classifications of trained weights, not independent statistical samples of neural features. The Gaussian reconstruction controls produce **at least as many** self-reinforcing decoder atoms as the new teachers. Consequently, the SMC-12 internal WD diagonal-dominance discovery is a property that reconstruction training can induce even on task-free activations.

MathCan OLM-13 gives a sharper purely algebraic warning: two exact linear identity autoencoder factorizations \(DW=I\) can have **different strict WD self-selection counts** (zero or two in a 1D/two-feature witness). The internal WD cone statistic is invariant under hidden-state basis changes and feature-label permutations, yet it is insufficient evidence of task-semantic factors.

## 4. Calibration-only feature alignment across teachers

Source A's frozen SAE seed 71 comes from its original SMC-11C weights-only snapshot, SHA-locked to the same-fit causal audit. New teacher B SAEs use independently trained weights with seeds 71 and 89.

For each pair, construct binary indicators of positive top-four SAE activations on **64 distinct novel calibration puzzles** shared as external inputs. Center the feature coactivation covariance, then find its deterministic optimal 96×96 one-to-one feature permutation using a Hungarian assignment. This permutation is frozen before testing on three separate 64-puzzle groups. No target solution or heldout puzzle enters alignment fitting.

The heldout mean fraction of source positive feature activations matched by target features is:

| Compared SAE representations | Novel 30-hole | Novel 34-hole | Canonical 30-hole |
|---|---:|---:|---:|
| Source A → B501, two SAE seeds | 35.93–39.51% | 33.46–37.13% | 35.73–39.60% |
| Source A → B701, two SAE seeds | 41.56–42.69% | 38.99–40.28% | 41.80–42.83% |
| B501 SAE71 → SAE89 | 55.08% | 53.25% | 54.76% |
| B701 SAE71 → SAE89 | 59.39% | 57.65% | 59.92% |
| B501 SAE71 → B701 SAE71 | 52.88% | 50.97% | 53.23% |

The random feature-permutation null averages roughly 4%, although the null varies with actual feature occupancy frequencies. A calibration-puzzle-shuffle null and per-puzzle bootstrap intervals are archived for every pairing and test cohort. These results show repeatable *observational coactivation correspondence*, with weaker alignment across the original/new teachers than within the same teacher checkpoint.

## 5. Stronger negative control: condition on observed token information

Shared 81-token position and the visible input digit (0 for a blank) can align otherwise unrelated SAE features. OLM-13F therefore proposes a more discriminating calibration score: center feature coactivation separately **within each observed (token position, input symbol) group**. No hidden solution label is used to define these groups.

A second null independently shuffles target feature vectors **within each such group**, exactly preserving position/digit-conditioned feature marginals. The learned permutation is frozen and tested on the same distinct heldout cohorts as SMC-13B.

| Alignment pair | Unconditioned novel-30 overlap | Position+digit-conditioned | Within-group shuffled null |
|---|---:|---:|---:|
| Source A → B501 SAE71 | 39.51% | **24.83%** | 6.92% |
| Source A → B501 SAE89 | 35.93% | **25.68%** | 7.56% |
| Source A → B701 SAE71 | 41.56% | **19.83%** | 4.84% |
| Source A → B701 SAE89 | 42.69% | **28.81%** | 7.66% |
| B501 SAE71 → B501 SAE89 | 55.08% | 39.40% | 13.40% |
| B701 SAE71 → B701 SAE89 | 59.39% | 28.26% | 8.86% |
| B501 SAE71 → B701 SAE71 | 52.88% | 13.75% | 6.42% |

For the source-to-new comparisons, the conditioned transfer remains higher than the corresponding within-group null. A substantial part of the original unconditioned alignment was explained by shared visible token information. The remaining conditional signal may include common puzzle constraints, different teacher response patterns or other correlated properties; it does not establish an identifiable symbolic Sudoku reasoning feature.

All conditioning uses the calibration cohort alone. An independent mathematical null theorem is provided in MathCan OLM-13F, and the conditional evaluator independently passes tests that erase position/digit-only features.

## 6. Reproduction and evidence boundaries

Executable protocols:
- [Source-registered unique Sudoku families](smc13_family.py)
- [Two independently trained teacher checkpoints and Gaussian SAE controls](smc13_train_teachers.py)
- [Calibration-only Hungarian feature transfer](smc13_transfer.py)
- [Conditioned covariance and within-clue-group shuffled null](smc13_conditioned_transfer.py)
- [Exact mathematical analysis](https://github.com/patrickdugan/MathCan/blob/research/olm13-crossfamily-factorization-20261115/research/orbit_lock/OLM13_ORBIT_AND_FACTOR_GAUGE.md)

Relevant hosted artifact checksums, which are preserved in [SMC13_CERTIFIED_SUMMARY.json](SMC13_CERTIFIED_SUMMARY.json):
- New teacher checkpoint artifact ZIP SHA-256 0be07ad7a159ccd95be47ddb6c58e3d45e1ea8bcf2acc14812c943dd756e43e0; inner full.json SHA-256 1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3.
- Feature transfer artifact ZIP SHA-256 e20641e2e1c90fa9b815f564c6e283a536df0ee8bb0e3fa278a9e4feb7d5f00f; inner alignment.json SHA-256 40bd82bafd672a648b5662ab094a0f2c0ed6d5c6f6c5cdde18be40e4e6eb0022.
- Clue-conditioned transfer artifact ZIP SHA-256 075d50d4d9964139fbcccbcc2f4ff7072951e281278db94a4f4c68bb99ada319; inner conditioned.json SHA-256 3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07.

This is the first independently trained teacher/new-template follow-up in the SMC campaign. The new teachers have much less Sudoku capability than the original source teacher, while the internal WD cone effect is strong on Gaussian data. SAE feature matching is observational even when conditioned on public puzzle information.

A more decisive next gate should compare cross-teacher **causal intervention effects on correct Sudoku digit margins**, using *calibration-only* feature assignments and paired random-feature controls. That would directly test whether the residual activation correspondence predicts how a learned feature influences the task output. A later model-capability-matched replication is needed before claiming transfer of reasoning mechanisms.
