# SMC-25: All-digit causal effects with calibration-frozen probability mixing

**Executed 17 November 2026. Registered scientific verdict: FAIL, 0/4 independently trained B-teacher SAE pairs passed both new cohorts.** The final four-job [source-locked run #44113889861](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44113889861) passed original calibration, source-only prediction, full native FP64 scoring and independent standard-library receipt audit. The registered source protocol [SMC25_PROTOCOL.md](SMC25_PROTOCOL.md) was committed first at 9df8d60, before new source data. The first engineering run #44111941521 halted at a wrong cohort key in a reused stage-one replay helper, prior to scoring any new test effect; its repair and preserved failure are recorded in [SMC25_PREOUTCOME_AMENDMENT.md](SMC25_PREOUTCOME_AMENDMENT.md).

## Question

The earlier [SMC-24](SMC24_RESULTS.md) calibration-frozen state-to-gradient predictor failed across all four B501/B701 SAE fits on new Sudoku. The correct target digit determines the task-margin gradient, so SMC-25 computes **every possible correct-digit causal tangent** before seeing the true target. It then tests whether an old-calibration-frozen distribution over digits can combine these candidates into a better estimate of the heldout native feature-erasure effect than matched-compute alternatives.

Two independent frozen 44,001-parameter recurrent Sudoku teachers (B501 and B701), each with independently fitted Top4 SAE71/89 dictionaries, contribute the four pairs. Both teachers were previously trained in SMC-13. Source pretrained A, stronger in clean Sudoku competence, does not enter this new primary study. The four SAE comparisons reuse two teacher seeds, and all teachers/SAEs share 128 source puzzle cases.

## Registered mechanism and chronology

Each teacher has a temperature fixed from the original 64 SMC-13 calibration puzzles, before generating any test inputs. The preregistered temperature grid is 0.5, 1, 2, 4 and 8. Both B501 and B701 selected **T=2** using negative log likelihood on the 512 original calibration goal labels per teacher. Mean old-calibration NLL at the selected temperature was **1.51103** for B501 and **1.43956** for B701. Calibration clean goal accuracy was respectively 235/512 (45.9%) and 259/512 (50.6%).

The input-only prediction job generates 64 fresh uniquely solvable noncanonical 30-hole Sudoku puzzles from solution seed base **202635030000** and 64 fresh 34-hole puzzles from **202635040000**, each with unique-dig seed solution seed+1,000,000. It checks disjointness from the original SMC-13, SMC-15, SMC-16, SMC-23 and SMC-24 source rosters. Every teacher/SAE fit selects eight input-mask-only missing-cell positions per puzzle, and the largest naturally active SAE feature at each goal. Its physical minimum-norm selective erasure and an equal-dose sham chosen from the same **33 outcome-blind candidate shams** used in SMC-23 are frozen before scoring.

Let the physical erasure-minus-sham vector be v. Two extra full native FP64 three-block recurrences at the canonical FP32 stage-one state with +/-0.08 v/||v|| produce a full ten-logit tangent. Multiplying the central difference by ||v|| gives an approximate finite first-order native response for the actual physical intervention difference. For every hypothetical correct digit y=1..9, its highest CLEAN competing logit c(y) is frozen from all ten output classes, and the candidate effect is the corresponding logit tangent for y minus c(y). No true test digit is needed.

The primary predicted effect is the **T=2 calibrated class-probability-weighted median** of the nine candidate effects. Three equal-native-compute controls use a uniform-nine weighted median, the effect for the clean model's top predicted digit, and a weighted median after a cyclic rotation of the calibrated label probabilities. All seven candidate predictors/aggregations share the original frozen full-logit calculations. The score job alone receives the actual correct digit, reuses the archived FP32 input and physical erasure/sham, computes native finite effect D=m(sham)-m(kill) and preserves the oracle candidate effect of the actual digit as a descriptive finite Taylor check. A separate standard-library auditor replays the source rosters, saved prediction math and every recorded native margin identity.

## Original preregistered criterion

Each of four teacher/SAE fits in each fresh family is one registered cell. Coverage requires >=60 admitted pre-outcome goal interventions across >=30 separate puzzle IDs and >=1,990 of 2,000 valid whole-puzzle bootstrap draws. The primary estimate is the unweighted mean across puzzles of within-puzzle mean eligible-goal absolute-error gain. Positive gain means the calibrated median predicts the actual finite effect better than the named control. For a cell to pass, all **three** Bonferroni-adjusted **99.375% lower** cluster-bootstrap gains must exceed zero and the calibrated median's native effect-sign agreement must exceed the uniform median's on identical admitted goals. A teacher/SAE pair passes only when both new families pass; program success requires at least one fit per independent teacher to pass. Every failed or ineligible cell remains in the denominator.

## Full eight-cell original-result table

| Frozen receiving SAE | Family | Eligible goals / distinct puzzles | Calibrated median MAE | Gain vs uniform median | 99.375% bootstrap CI for gain vs uniform | Calibrated vs uniform effect-sign agreement |
|---|---|---:|---:|---:|---|---|
| B501 SAE71 | 30 holes | 209 / 62 | 1.3628 | -0.0846 | [-0.4396,+0.2779] | 132/209 vs 120/209 |
| B501 SAE89 | 30 holes | 215 / 63 | 1.2050 | +0.1181 | [-0.2532,+0.4928] | 139/215 vs 121/215 |
| B701 SAE71 | 30 holes | 315 / 64 | 1.3706 | +0.1492 | [-0.0774,+0.3826] | 184/315 vs 172/315 |
| B701 SAE89 | 30 holes | 285 / 64 | 1.5119 | -0.0550 | [-0.2939,+0.1957] | 159/285 vs 150/285 |
| B501 SAE71 | 34 holes | 201 / 61 | 1.2817 | +0.2067 | [-0.1648,+0.5642] | 131/201 vs 103/201 |
| B501 SAE89 | 34 holes | 227 / 63 | 1.3175 | -0.0806 | [-0.4012,+0.1939] | 137/227 vs 130/227 |
| B701 SAE71 | 34 holes | 277 / 64 | 1.4960 | -0.0318 | [-0.3753,+0.2539] | 177/277 vs 168/277 |
| B701 SAE89 | 34 holes | 280 / 64 | 1.5149 | +0.0669 | [-0.2236,+0.3279] | 177/280 vs 155/280 |

All eight cells meet the preregistered independent-puzzle coverage and each completed all 2,000 bootstrap draws. There are **2,009** eligible repeated teacher/SAE/goal interventions out of 4,096 preregistered candidate slots, measured on **128 distinct fresh puzzles**. These are 128 independent puzzle inputs under one generator, not 2,009 independent teachers or natural-language tasks.

The calibrated median beats **uniform** MAE in only **four of eight** cell point estimates, beats the clean top predicted digit in **all eight**, and beats the cyclically rotated calibrated distribution in **seven of eight**. Four of eight adjusted CI lower bounds against top-digit are above zero; one of eight against shifted distribution is above zero; **none of eight adjusted lower bounds against uniform median are positive**. Consequently every B-teacher/SAE both-family gate fails. The registered program verdict is **FAIL (0/4 pairs)**, with **complete source and scoring admission**.

The calibrated median had higher descriptive *native effect sign* agreement than the uniform median on **all eight** cells. The original scientific gate required a simultaneous adjusted MAE advantage, so this observation does not count as a pass. A new hypothesis about sign-oriented decisions needs a prospectively fixed sign-loss experiment and new source-disjoint puzzles.

## Finite-tangent error and interpretation

A raw classwise candidate is a two-sided local native logit tangent, while the scored effect uses the actual finite feature erasure and its matched sham. The median absolute difference between the *answer-conditioned* local candidate and the observed finite native effect was about **0.153–0.240 logits** across the eight SAE/family cells, and mean absolute differences **0.295–0.355 logits**. These were scored only after the prediction artifact was sealed; they describe physical finite-dose curvature and local approximation error, without selecting a more favorable predicted arm.

The measured failure constrains the idea that a simply temperature-calibrated label distribution is sufficient for **better finite native effect MAE than an equal-compute uniform-nine median** on these B teachers. The positive sign-agreement description, along with superior MAE versus selecting the single top predicted label, suggests that accounting for class-conditioned effect distributions may carry some task information. Neither result proves a stable semantic mechanism or distinguishes a general Bayesian label-information limit from a poor model probability estimate. Source competence, overcomplete SAE cell changes and native margin nonlinearity remain limitations.

The SMC-25 candidate-effect calculation uses original source models and actual hidden-state perturbations; neither the PR nor its auditor independently proves a global nonlinear Taylor remainder bound or a new exact-real mechanism. The original SMC-24 and SMC-23 negative verdicts remain unchanged, as do older SMC-16/17 receipts and the untouched ARG-03-N sealed 168-case evaluation.

## Reproduce and source receipts

The original exact source checkpoints, three artifact byte SHA-256s and calibration mapping hashes are checked in the hosted jobs. FP32 source states were archived prior to scoring. Maximum cross-runner FP32 stage-one discrepancy across the two teachers and two new puzzle families was **1.43e-6**, within the original numerical admission guard of 1e-4.

- Source-registered protocol: [SMC25_PROTOCOL.md](SMC25_PROTOCOL.md), first commit 9df8d60.
- Engineering amendment: [SMC25_PREOUTCOME_AMENDMENT.md](SMC25_PREOUTCOME_AMENDMENT.md), preserves first scoring KeyError before any task effect was measured.
- [Complete four-job Actions run #44113889861](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44113889861), all phases passed.
- [Calibration temperature artifact #14341163724](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44113889861/artifacts/14341163724), ZIP SHA-256 41f7d998eb5521327180dead295de51c7933c253e9978b406b9931fff739d0a9; calibration.json SHA-256 bba1c7401a200727818f3c9e5a6b2f7189b5f20dd14e25fe88105962e90c3564.
- [Frozen target-blind nine-candidate artifact #14342136984](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44113889861/artifacts/14342136984), ZIP SHA-256 1f76961d51a40c3ebcbeeeade2e3c6732af8286ef0de7bba8e57281d725b7e5c.
- [Native finite output artifact #14341883837](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44113889861/artifacts/14341883837), ZIP SHA-256 4f9b68d4286dd12f7de232e89322a820595f76707a0bf85d1305019f9c446378; scored.json SHA-256 6e58461acd9c012f3ff9324be1420dc693cb3c318419c9a9bebb2376af2c878c.
- [Independent standard-library audit #14341738386](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44113889861/artifacts/14341738386), ZIP SHA-256 56159dd007eae782d7fa0caa58d120bfd4f97e094fd832897a8ddd31f3b35fbf; audit.json SHA-256 ece96aab440f64ac8cfc47bb193081a49f57a257caa108761c768c666c900d8c. The independent auditor reproduced 2,009 admitted goal cases and scientific program success false, using the archived tensors, source generator and output receipts. It does not run a second neural teacher forward pass.
