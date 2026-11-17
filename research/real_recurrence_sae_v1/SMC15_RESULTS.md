# SMC-15: Native tangent conditioning of naturally coactive frozen SAE features

**Executed 17 November 2026. Registered scientific status: UNDERPOWERED.** The stringent predeclared low-singular-value comparison did not reach its minimum sample counts on any of the ten SAE/family cells. All source checks and all 5,120 goal-level native intervention records completed. The source-locked real-model run was [GitHub Actions #44086787126](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44086787126), whose native CPU unit tests, SMC-13 calibration pilot and full 128-puzzle evaluation passed. The exact registration was committed first at 299ad0c in [SMC15_PROTOCOL.md](SMC15_PROTOCOL.md).

Full [uploaded source receipts](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44086787126/artifacts/14329567893) consist of calibration_pilot.json (350,173 bytes) and fresh_full.json (10,386,786 bytes). Uploaded ZIP SHA-256 fcfee553a6351e529dc276e5e15feb9e3cb713d4477b1310eb5ddda1f478e31c; pilot JSON SHA-256 c8883ca53e1804cc5d39da0f32acfed6eb57bc7a94fd059201986f0a7fa21700; full JSON SHA-256 3ad00ce9f190363954b0b90df8ed33966fc687c520ddf9665c6575b62d75070c.

## Source and test boundary

This experiment reuses the original SMC-14 frozen pretrained A (44,001 parameters), independent trained B501 and B701 recurrent Sudoku teachers and five 48→96 top-four SAE dictionaries. Both original source checkpoint/SAE and independent teacher/SAE receipt byte hashes passed. The four original SMC-13C matched and within-clue-shuffled 96-feature calibration permutations were independently reconstructed and matched against the source correspondence SHA-256s before fresh data creation. The engineering pilot inspected four old calibration puzzles only.

Two fresh families each contain 64 independently constructed unique-solution puzzles with 30 or 34 holes. They use the prespecified 202615030000 and 202615040000 seed bases, have positive intercalate counts and are disjoint from 512 previously registered SMC-13 full solution targets. The novel30 manifest SHA-256 is a0b0d158700c84594c7c3d165041136c093524b3018f1157791b6ee715053e4c; novel34 is 3af2629467a53dd6472cc0a32812febe0d483fd6b1c8967a3ca28ce12c62d909. Both cohorts' input and target hashes are preserved in the full receipt.

For every SAE, each puzzle supplies eight distinct visible-only selected missing-cell goals. The two most active feature directions were strictly positive and distinct at all 5,120 observations. Every physical decoder direction was selected before consulting the target digit, and each patch used a source FP32 post-stage-one state with three native FP64 recurrent steps. Zero-dose native self-patches and stage baseline parity were exact at every goal, and each perpendicular-nuisance sham matched its joint intervention's physical dose to at most 1.2e-16.

**Independence:** the experiment measures 128 distinct fresh puzzle cases reused by all teachers and all SAE fits; 5,120 goal-level observations are correlated measurements on those puzzle clusters, and B501/B701 are the two independently trained teachers. An individual top-four SAE weight fit is not an independent teacher.

## Primary measured output-frame geometry

Per goal, the two top-active physical SAE directions are each probed with a central finite difference at physical radius ±0.08. Their class-centered nine-digit native response vectors are normalized and assembled into a 9×2 tangent frame. The smaller singular value is compared to the frozen toy-derived threshold 0.10. A separate physical 48×2 decoder-frame singular value checks the physical origin of any apparent output collapse.

| Frozen teacher / SAE | Novel30 median output sigma-min | Novel30 below 0.10 | Novel34 median output sigma-min | Novel34 below 0.10 |
|---|---:|---:|---:|---:|
| Pretrained A / SAE71 | 0.6729 | 7 / 512 | 0.6288 | 8 / 512 |
| B501 / SAE71 | 0.8271 | 0 / 512 | 0.8378 | 0 / 512 |
| B501 / SAE89 | 0.8112 | 0 / 512 | 0.7939 | 0 / 512 |
| B701 / SAE71 | 0.8098 | 0 / 512 | 0.8300 | 0 / 512 |
| B701 / SAE89 | 0.8113 | 0 / 512 | 0.8012 | 0 / 512 |

The two independently trained teachers exhibit **zero** cases below the registered threshold across their four SAE fits, 4,096 repeated goal/SAE measurements. Source pretrained A has 15 rare cases across 1,024 goal measurements, each in a distinct puzzle. Their physical input-frame sigma-min is **at least 0.9006**, indicating that the near-singular response arises through the **native downstream map to digit logits** on these cases, rather than through decoder-direction collinearity at the injection site.

All five SAE/family cells per cohort are therefore classified UNDERPOWERED. The registration required at least 30 low-conditioning goals and at least ten independent low-conditioning puzzles per cell. Pretrained A has only seven and eight; B teachers have none. No paired bootstrap acceptance criterion was computed or passed. This absence of near-singular directions under the declared threshold is itself a useful limitation on extending the synthetic SAU-CURVE02 example to these trained checkpoints.

## Actual finite-dose causal interventions

At every eligible goal, a second forward pass applies equal physical L2 radius 0.20 along the local smallest- and largest-response right singular combinations, measuring their actual nine-digit-logit displacement. Pretrained A's seven novel30 low-sigma cases have mean puzzle-level min/max finite output effect ratio **0.1250**, against **0.6527** among the higher-sigma puzzles. On novel34, the corresponding means are **0.1522** (eight low-sigma puzzles) and **0.6250**. The predeclared low-group sample gate is not met, so these gaps are descriptive and supply no registered positive inference.

At the 15 rare low-sigma goals, the physical pair's minimum singular values range from 0.9006 to about 0.9974, and individual signed correct-digit margin effects often have opposite signs. This gives concrete candidate states where two distinguishable physical decoder directions have partially cancelling effects on the output under the specified three-step continuation. The task readout is finite horizon and conditional on a fixed Sudoku input. The evidence does not establish a global state quotient, a learned curved feature or a recurrent proof mechanism.

The mean **absolute native joint-minus-additive correct-digit-margin residual** is 0.00641 on pretrained A novel30 and 0.00988 on its novel34, versus approximately 0.00105–0.00130 across the B teacher/SAE cells. This descriptive difference compares teachers with different baseline digit competence and task-margin scales; it cannot be interpreted as a teacher-independent curvature effect.

## Cross-checkpoint natural-coactivation availability

The SMC-14 mapping often transfers an output-influential *decoder direction* to a recipient SAE that does not activate that feature on the same input. The new audit asks the stronger question: from two naturally active source A features at the same blank token, do both matched features appear in the recipient SAE's top four active features?

| Frozen source A SAE71 → target | Novel30 dual active / 512 | Novel34 dual active / 512 |
|---|---:|---:|
| B501 SAE71 | 4 | 8 |
| B501 SAE89 | 4 | 4 |
| B701 SAE71 | 5 | 5 |
| B701 SAE89 | 7 | 3 |

The combined per-comparison rate ranges from 0.6% to 1.6%, and counts must not be pooled as independent tests because the same source puzzles and features are reused. These outcomes constrain the interpretation of SMC-14's successful **directional** causal-margin transfer: naturally participating **feature pairs** correspond much less often than isolated potential intervention directions. The JSON records each coactivation comparison's aggregate numerator and denominator, although the original run did not store a target top-four mask for every token, so the counts require source-model replay for an independent casewise audit.

## Statistical and scientific interpretation

The strict registered test for better finite min/max response separation in high-conditioning goals was **underpowered on both novel cohorts**, hence it supplies no affirmative p-value or valid lower confidence bound. The main result is a bounded picture of the two-feature local native response geometry at five frozen SAE fits across three teacher checkpoints: mostly well-conditioned nine-logit response frames, plus rare output-near-null combinations in pretrained A.

The feature selection is calibration-frozen with respect to target labels, but its local SVD is computed from each test goal's native small-dose output. Therefore its 0.20 finite-dose behavior is a within-goal two-dose verification, and the sigma-conditioned groups are test-derived. A cross-context predictive mechanism would require separate, calibration-frozen predictions of high-risk goal states.

The experiment carries **no measured learned Manifold-SAE model**, no native 7M LS-SAE checkpoint, no Arb exact-real certification and no access to the strictly sealed ARG-03-N test.

## Reproduction and next mechanism

- Registration: [SMC15_PROTOCOL.md](SMC15_PROTOCOL.md).
- Native evaluator and source-lock checks: [smc15_tangent_conditioning.py](smc15_tangent_conditioning.py).
- Six CPU regression tests: [test_smc15_tangent_conditioning.py](test_smc15_tangent_conditioning.py).
- Hosted full run: [Actions #44086787126](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44086787126).
- Independent standard-library raw-receipt auditor: [audit_smc15_receipt.py](audit_smc15_receipt.py), with separate hosted audit workflow. Its case-level source-to-target coactivity verification is explicitly bounded by the original archive's lack of per-token target support masks.

A source-registered follow-up should choose **target-native active features first**, then independently match their physical and observable quotient directions to source A using calibration-only data. It should test task-margin necessity of those naturally active target features and compare actual finite native nullspace edits with matched physical and output-sensitive controls. This changes the question from SMC-14's possible causal response similarity to endogenous feature use. A new protocol and new seed-disjoint Sudoku cohorts are required before testing that hypothesis.
