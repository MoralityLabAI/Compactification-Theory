# SMC-15: Naturally active cross-model SAE erasures predict Sudoku margin losses

**Executed November 16, 2026.** This independent native-erasure experiment was preregistered before any fresh heldout puzzle states were evaluated: protocol commit [529f828](https://github.com/MoralityLabAI/Compactification-Theory/commit/529f828af6997b676f43b4027e32a33492b2718b). It builds on the injection-only SMC-14 study, which remains a separate [PR #174](https://github.com/MoralityLabAI/Compactification-Theory/pull/174).

The [successful hosted pilot and full evaluation, Actions #43792274428](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43792274428), ran on [SMC-15 PR #176](https://github.com/MoralityLabAI/Compactification-Theory/pull/176). All source checkpoint receipts, all seven original SMC-13C clue-conditioned feature assignment hashes and the fixed calibration-only sign choices were locked before generating the new heldout states. The independently written stdlib auditor passed in hosted CI and reproduced **512 pair/family/puzzle records** again on a separately downloaded full artifact.

## Frozen intervention and denominator

Four original-pretrained A SAE71 to independent-teacher B501/B701 SAE71/89 comparisons were retained. The source SAE's highest strictly positive stage-one feature determines the SMC-13C matched receiving feature. A goal is eligible only when the matched target SAE atom is naturally active and a different active target atom exists. Using the original *clean* SAE coefficient, native erasure subtracts its full physical decoder contribution from one stage-one goal state. A different active target atom receives an equal-physical-L2-dose control subtraction. Both effects use exactly three FP64 recurrent continuation steps, with the target digit and clean rival held fixed. No model, SAE, feature assignment, control choice or calibration sign was fitted on the heldout puzzles.

Two new, noncanonical Sudoku cohorts use seeds 202615030000..202615030063 (30 holes) and 202615040000..202615040063 (34 holes). All 128 puzzles have unique completions, structurally noncanonical full solutions, and distinct source solutions from the SMC-13 train, validation, calibration and test cohorts. Every 64-puzzle manifest and full per-puzzle inputs/targets are retained in the receipt.

The primary estimand is the average *oriented signed agreement* between source and receiving-teacher native erasure margin effects on eligible tokens, minus its dose-matched other-active-feature control, with separate orientations fitted on the 64 SMC-13C calibration puzzles. The paired 95% intervals resample **puzzle IDs** 2,000 times; they are descriptive conditional on this seeded benchmark. Values can lie outside [-1,1] because these are differences between two signed concordance scores.

## Registered primary result

| Receiving teacher/SAE | Novel 30-hole: matched minus control [95% puzzle bootstrap] | Eligible goals / puzzles | Novel 34-hole: matched minus control [95% puzzle bootstrap] | Eligible goals / puzzles |
|---|---|---|---|---|
| B501 SAE71 | +0.5517 [0.2264, 0.8710] | 58 / 38 | +0.6889 [0.4660, 0.9213] | 90 / 54 |
| B501 SAE89 | +0.9630 [0.7200, 1.2093] | 54 / 38 | +0.9388 [0.6176, 1.2632] | 49 / 33 |
| B701 SAE71 | +0.5789 [0.2029, 0.9315] | 76 / 48 | +0.6667 [0.3103, 0.9836] | 57 / 40 |
| B701 SAE89 | +0.6667 [0.3333, 1.0000] | 63 / 42 | +0.9714 [0.6197, 1.2698] | 70 / 45 |

**All four paired teacher/SAE comparisons passed the strictly predeclared gate on both independent fresh families**: at least 24 eligible goals from 20 distinct puzzles, identified calibration orientation on both arms, 2,000 valid puzzle-bootstrap draws, and a positive 95% bootstrap lower bound for matched minus other-active feature effect.

Across the eight pair/family conditions, matched-feature activity is rare: the selected target SAE atom is active on approximately 9.6% to 17.6% of the 512 chosen goals per cohort. The selected native erasure leaves a mean re-encoded matched-feature coefficient between 0% and 5.2% of its original value on eligible tokens, showing that it almost always clears the corresponding SAE activation.

## Functional consequences on eligible goals

The fraction of **all eligible goals** changing from clean-correct to wrong after the matched contribution was erased versus the equal-dose other-active control was:

| Receiving SAE | 30-hole: matched / other active | 34-hole: matched / other active |
|---|---|---|
| B501 SAE71 | 17.2% / 3.4% | 18.9% / 5.6% |
| B501 SAE89 | 25.9% / 1.9% | 24.5% / 2.0% |
| B701 SAE71 | 17.1% / 1.3% | 14.0% / 1.8% |
| B701 SAE89 | 19.0% / 0.0% | 14.3% / 1.4% |

These percentages use *all eligible tokens* as the denominator. Clean-incorrect tokens contribute zero to the clean-correct-to-wrong numerator; exact indicators are included in the raw records.

For comparison, pretrained A was accurate on 95.1% of selected 30-hole and 89.6% of selected 34-hole blank goals. The receiving B501/B701 teachers were only around 43%–48% accurate overall on these goals. The apparent feature-effect correspondence therefore coexists with substantial task capability mismatch.

## Source-locked artifacts and independent replay

- Full original intervention observations: Actions artifact [14192743542](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43792274428/artifacts/14192743542), 3,615,026-byte \`full.json\`, SHA-256 \`34d9224613e2f264b70e5a5ec5e4e080a71a32edb72f6cd63353a321fe05a324\`.
- Hosted independent stdlib \`audit.json\` SHA-256 \`789dec1319b0a1b6707620bdea7b4413e281ed389ac776f71c422a1d42c0c6d9\`; downloaded ZIP SHA-256 \`71ba859a7172d4bd6be74489d1e351f0bc867edd3dca19a503e2c79e6e75a0fc\`.
- Full new-cohort manifest SHA-256: 30-hole \`a0b0d158700c84594c7c3d165041136c093524b3018f1157791b6ee715053e4c\`; 34-hole \`3af2629467a53dd6472cc0a32812febe0d483fd6b1c8967a3ca28ce12c62d909\`.
- Source artifacts verified: original SAE receipt \`7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087\`; new teacher training receipt \`1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3\`; SMC-13C alignment receipt \`3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07\`.

The independent replay verifies every published puzzle-level raw margin response's eligibility, physical-dose equality, fixed goal positions, clean-to-wrong flip consistency, global calibration-only orientation, registered effect score, puzzle-bootstrap intervals, and success gate. It audits numerical outputs and dataset structure; it is a separate algorithm from the PyTorch runner and does not re-execute the source neural networks.

## Interpretation and next discriminating test

SMC-15 shows **task-functional relevance of naturally encoded matched SAE directions under an explicit native decoder-contribution erasure**, strengthening SMC-14's external decoder-injection findings. Near-complete re-encoding suppression and elevated clean-correct-to-wrong flips support an operational role for these naturally active features.

The remaining causal ambiguities are dictionary nonorthogonality, effects of the full hidden-state displacement on neighboring features, and model-head/task-label geometry. An active-other control is physically dose-matched but has a different clean coefficient from the originally matched atom. SMC-13C feature coactivation and Sudoku label structure may produce residual correspondence across teachers. The two SAEs on each B model and both test families share architecture and puzzle generator, making the four comparisons correlated rather than independent replications.

The predeclared primary gate is fully reported without posthoc adjustment or seed selection. Follow-up SMC-16 should test exact activation-preserving feature edit controls and source/target feature-kill specificity, then replicate on capability-matched independently trained teachers. Mathematical companion: [OLM-15 PR #93](https://github.com/patrickdugan/MathCan/pull/93) proves how full erasure effects can differ despite identical symmetric injection responses and gives a curvature-controlled local bound, with exact rational checks passing CI.
