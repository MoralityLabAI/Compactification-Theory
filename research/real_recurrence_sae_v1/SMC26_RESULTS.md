# SMC-26: Four-teacher prospective causal sign utility, with abstention control

**Original science-producing run: 18 November 2026 Manila time (GitHub Actions 17 November UTC). Registered verdict: FAIL, 0/4 independent teacher seeds passed both fresh families.** This result comes from the original fully completed and independently audited [Actions #44130482978](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44130482978), with protocol [SMC26_PROTOCOL.md](SMC26_PROTOCOL.md) committed FIRST at 6bbd2c6 before new model training or new test puzzle generation. The post-result source-only QA workflow is a verification repetition, never a replacement for this originally registered outcome.

## Why this test exists

The preceding SMC-25 study found that a teacher's calibration-weighted median of all nine possible native causal margin effects gave more correct effect signs than a label-uniform median in eight of eight descriptive cells, yet its **registered MAE objective failed for every SAE pair**. The BPD-OUT05 exact rational counterexample showed why sign classification and magnitude loss can rank predictors differently. SMC-26 therefore preregistered a **fixed ternary intervention decision and action cost**, then tested entirely fresh puzzles and added independent teacher seeds.

A positive predicted correct-digit-margin difference D=m(sham)-m(kill) leads to selecting the sham; a negative difference favors the targeted SAE erasure. Predicted magnitudes within ±0.15 logits cause abstention. The true sign class uses exactly the same ±0.15 threshold on the actual native FP64 finite effect. Correct nonneutral action has utility +1, an opposite action -1, abstention on nonneutral effect zero; neutral-case abstention +1 and action -1. Any action costs a further 0.10. An original goal failing physical source admission contributes zero under **every** policy. The primary estimand is per-puzzle mean paired utility over all eight selected blank slots, with 128 independent puzzle IDs as the resampling clusters.

The predeclared null controls are the equal-forward-budget *uniform-nine-label median* and an **always-abstain** policy. Shifted calibrated label probabilities and selecting the model's top predicted digit are additional descriptive comparisons. Native ±0.08 logit tangents and physical SMC-23 33-candidate matched-norm hard shams are generated without test target labels.

## Four source-matched teachers and capabilities

Two original SMC-13 B501 and B701 teacher checkpoints/SAE71 dictionaries are **exact-byte pinned**. Two independently initialized teachers B811 and B919 were trained anew on the original SMC-13 192 training/64 validation puzzles with the same 600-update AdamW model procedure and 300-update Top4 SAE71 training. Only the old validation set selects checkpoints; only the old 64-puzzle feature_calibration roster (512 goal tokens per teacher) selects classwise softmax temperatures. **No fresh test data trains a teacher or SAE**, and there is one SAE71 per teacher to limit fit correlations.

| Teacher | Source validation-selected checkpoint step | Original calibration correct goals | Accuracy | Calibration temperature |
|---|---:|---:|---:|---:|
| B501 | Previously frozen | 235 / 512 | 45.90% | T=2 |
| B701 | Previously frozen | 259 / 512 | 50.59% | T=2 |
| B811 | 175 of 600 | 235 / 512 | 45.90% | T=2 |
| B919 | 150 of 600 | 213 / 512 | 41.60% | T=2 |

All four satisfy the **predeclared 40–60% original-calibration capability match**. They are four independent parameter initialization seeds under the same architecture, data generator and source training roster; this does not constitute four independently sampled task distributions.

The source-locked training artifact preserves canonical FP32 teacher parameter hashes and SAE hashes:
- B811 original **science-producing** teacher SHA256 82948725c7cbb0bfad59fa8d0b0ba2ace288493d3c0e450326e1a228f9c60873; SAE SHA256 fa493c4533119f53c896c16d1899c49c08973a61930131171c42fb2298c366a9.
- B919 original teacher SHA256 51f0141fa07ce29fc1eeb3181d522c27190d87ba74e8a60ab0f69bc80b5c6e63; SAE SHA256 92557e647ab4f194e03a3d32078388fec6e6d149567f64d80135d38a4a5bf9fc.
- Frozen source new-teacher bundle SHA256 9d18efbe4a88fbfce6ffe346f60693b17629eddfd1121068fcab010c332c508d.

Repeated hosted retraining from the same registered seeds can produce different teacher tensor-byte fingerprints between jobs. Consequently the original source checkpoint files above, held by the original complete science run, are the identity of this original test. Subsequent retraining on already examined fixed puzzle seeds is a source/engineering QA repetition; it is neither a second independent prospective test nor a reason to choose a favorable outcome.

## Prospective fresh test splits and source protection

Two independently generated new solution families, each of **128 uniquely solvable, noncanonical 9×9 Sudokus**, were registered before model training: 30-hole grid seed base 202636030000; 34-hole seed base 202636040000; each includes offsets 0 through 127 and dig seed equal to grid seed+1000000. All complete solutions have a positive alternating-rectangle count, with minimum count nine in each new family.

| Family | Frozen source roster SHA256 | Minimum rectangles |
|---|---|---:|
| Novel30 | 8fc327f30a7bf6ba2e398c2c32e3eca133ef55e4f7c790f59d7ee6b4fbaf4bfa | 9 |
| Novel34 | 1cfcbde2cc40e4911b2377dfb7630dd6913dadea48b3155a43daa6b442f99d4d | 9 |

Original source validation verifies distinct inputs/targets relative to the original SMC-13 source inventory plus SMC-25. A subsequent source-only audit extends the explicit inventory comparison to all 640 fixed-seed prior SMC-15/16/23/24/25 fresh puzzle samples. This retrospective integrity extension changes no policy, score, seed or scientific threshold and cannot generate a replacement test puzzle.

Every teacher, family and puzzle has eight input-mask-only missing-cell goals: **8,192 predeclared teacher-goal slots**, with 4,315 (52.7%) admitted by the SMC-23-style exact-target SAE encoder erasure and same-dose hard sham before observing a correct-digit label. Every complete eight-goal puzzle cluster remains in the utility denominator, even if an edit was inadmissible.

The model stage-one original FP32 hidden tensors were sealed in a binary source artifact before target labels were exposed. Original native scoring computed clean, self-patch, kill and sham after three FP64 recurrent blocks. Its maximum cross-host stage-one reconstruction difference was **0.0** in the original run; clean/self-patch full-logit closure and physical-sham admission passed.

## Registered outcome and full eight-cell evidence

Each cell requires at least 200 admitted goals from 80 distinct puzzles. All eight **passed coverage**, with 406–615 eligible goals each and 124–128 independent puzzle clusters. The predeclared utility gate requires **both** corrected 99.6875% two-sided Bonferroni lower bounds (over 8 cells × 2 controls) to exceed +0.02 utility per original blank goal, plus a nonzero calibrated-action rate in the 10–90% interval and calibrated teacher competence within 40–60%. The teacher passes only when both novel families pass. Program success requires three of four teachers including a newly initialized B811/B919.

The primary statistics are the mean over all **128 puzzles** of the original eight-slot per-puzzle utility gain. A positive number favors the calibrated weighted-median policy.

| Teacher | Family | Eligible goals / puzzles | Calibrated action rate | Utility gain vs uniform [99.6875% CI] | Utility gain vs always abstain [99.6875% CI] |
|---|---|---:|---:|---|---|
| B501 | 30 holes | 433 / 126 | 87.3% | +0.1084 [+0.0362,+0.1805] | +0.0520 [-0.0072,+0.1111] |
| B701 | 30 holes | 615 / 128 | 89.3% | +0.1122 [+0.0203,+0.2038] | +0.0304 [-0.0420,+0.1058] |
| B811 | 30 holes | 592 / 128 | 88.3% | +0.1121 [+0.0271,+0.1960] | +0.0085 [-0.0661,+0.0815] |
| B919 | 30 holes | 543 / 128 | 84.3% | +0.0925 [+0.0038,+0.1796] | +0.0256 [-0.0396,+0.0920] |
| B501 | 34 holes | 406 / 124 | **91.9%** | +0.0090 [-0.0682,+0.0798] | +0.0036 [-0.0590,+0.0604] |
| B701 | 34 holes | 593 / 128 | 88.0% | +0.1002 [+0.0129,+0.1843] | +0.0135 [-0.0530,+0.0799] |
| B811 | 34 holes | 614 / 128 | 86.6% | +0.0828 [-0.0104,+0.1761] | +0.0350 [-0.0347,+0.1045] |
| B919 | 34 holes | 519 / 128 | 85.0% | +0.0664 [-0.0102,+0.1459] | +0.0155 [-0.0459,+0.0728] |

All intervals resample **128 whole puzzle IDs 5,000 times**, preserving original eight-goal clusters. Each cell's 99.6875% interval is Bonferroni-adjusted over the original sixteen primary comparisons; this is a deliberately conservative program-level familywise correction for these fixed eight model/cohort cells.

The calibrated action policy has **8/8 positive utility-gain point estimates against uniform** and **5/8 strictly positive adjusted lower bounds** against uniform, of which **3/8** clear the required +0.02 lower-bound margin. **Zero of eight corrected lower bounds against always abstaining are positive**, so every cell fails the simultaneous primary utility test. B501 novel34 also exceeds the original 90% maximum action-frequency criterion. Thus no teacher passes both families and **the original registered scientific verdict is FAIL (0/4)**.

The always-abstain control has positive mean offline utility of roughly 0.025–0.050 per original goal, helped by the fixed true-neutral reward. Calibrated actions are chosen on roughly 84–92% of physically admissible goals, exposing an insufficient abstention strategy under the registered utility. The observed advantage against the label-uniform policy is meaningful as **within-model descriptive evidence**, but its own stand-alone corrected significance is not a substitute for the simultaneous always-abstain criterion.

## Causal and interpretability scope

The model computes nine class-conditional **local** causal tangent effects without any test target digit. It uses a T=2 old-calibration posterior to select a sign with an explicit 0.15 neutral band and an action-cost penalty of 0.10. All native scored effects are finite physical manipulations of one goal-token state, under the same trained recurrent teacher.

The result shows that the calibrated class-aware sign policy can improve over an equal-forward-budget **label-uniform policy** on these Sudoku teachers. The calibrated rule's confident action rate is too high under the declared cost/abstention reward, and the expected conditional decision utility does not exceed always abstaining with corrected uncertainty. A future test can prospectively choose a different abstention threshold **on old calibration only** with independently new hidden targets; changing the threshold on these observed new puzzle scores would invalidate the original registration. The original SMC-25 MAE failure, BPD-OUT05 exploratory sign result and earlier SMC-23/24 failures remain unchanged.

There is no evidence here of full-grid Sudoku improvement or a globally stable mechanistic SAE identity. The sample is four independent model initializations sharing training/validation puzzles and one synthetic Sudoku generator. The independent auditor checks saved arithmetic and model source fingerprints, not a separate model forward implementation.

## Original raw evidence and reproducibility

- Preregistration [SMC26_PROTOCOL.md](SMC26_PROTOCOL.md), committed before new training at 6bbd2c6.
- Input-loader engineering failure [SMC26_PREOUTCOME_AMENDMENT.md](SMC26_PREOUTCOME_AMENDMENT.md), fixes only the old B501/B701 original SAE71 alias before new test data were generated. The first failed [Actions #44129876450](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44129876450) had green training/calibration and no newly scored test data.
- **Original complete science-producing [Actions #44130482978](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44130482978)**, all four stages green.
- [Training/new teacher/calibration artifact #14349821694](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44130482978/artifacts/14349821694): ZIP SHA256 f94c3b8178a4faf45f921d674a0ed7475bd87924bfe4b96de2fa4da535cc54b8, calibration JSON SHA256 145312c04648847c409b506aecdf1a3464af855d4ec8aa638eb0d604185512c9.
- [Target-blind 256-puzzle prediction and binary state artifact #14350026495](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44130482978/artifacts/14350026495): ZIP SHA256 bd0a983c82cc283fec45472d440c06842bd13f48a8757509a6d42e01559abfbb.
- [Original native scored artifact #14350730011](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44130482978/artifacts/14350730011): ZIP SHA256 cdb65b5bdd4c741a6b3efeeba6472e940e81b943f6f9cb4c522a80e612d4dcad, scored JSON SHA256 11f2ec60fc6164db09f8e46a0240e3a55e35f464f5c774e94c79b65fdb18ffe1.
- [Original independent standard-library source/effect/cluster audit #14348279903](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44130482978/artifacts/14348279903): ZIP SHA256 ba119e1b88816fca0b06c012a91a9adf8b5a10958ac230002b742e52a8a48a34, audit JSON SHA256 a193c0f2e77ea53fc371af920d3ecae7042c6904419782c9e94748247c56e59a. Its registered original verdict is FAIL, 4,315 admitted goals, all 8,192 source slots and 256 new puzzles checked.
