# Algernon-03f result (H-A3f): once the training loss reaches the answer, the constraint residual repairs it and beats the same-information control

Run 2026-11-10 13:34 to 2026-11-12 02:26 on the local RTX 5080, under
[EXPERIMENT.md, H-A3f](../../../EXPERIMENT.md#h-a3f-h-a3e-with-a-training-loss-that-reaches-the-answer-registered-2026-11-10-after-h-a3es-evaluation-was-read-before-any-h-a3f-run).
It was registered and pushed (`e1c98cb`) after H-A3e's evaluation was read and before any H-A3f run.
- H-A3f is H-A3e with one change, the training loss:
  - Each training case runs all 16 loops with upstream's answer suffix (`### d` and EOS) teacher-forced.
  - The answer digit's cross-entropy is added to the local loss (pi at loops j and j + 1).
  - The answer target is supervision given equally to every arm in training. No arm sees it at evaluation.
- Everything else is H-A3e:
  - sigma_e = 1.0, kappa = 6.92, and noise std 1.0 x 6.92 per slot element for every arm.
  - F is 02D's arm D1 (`80eda0824cff`), frozen. pi is D1's LM head on slot 2. The correctors are bias-free.
- Code: `e1c98cb` for training and evaluation. P was evaluated again.
- Training was stopped twice. Each time it resumed from the last 25-step checkpoint.
  - 2026-11-10 16:24: I at step 288, a GPU power cap. vramorama restarted it from step 275.
  - 2026-11-10 16:55: I at step 425, the card was needed. It resumed 2026-11-11 21:34 from step 425.
  - Each step seeds its own batch, loops and noise from the step number. The resumed schedule is therefore the uninterrupted one, up to GPU kernel nondeterminism.

**The registered decisions: R1, R2 and R3 are met. R4 is not met.** The instrument is valid: P answers all 72 clean cases, and the exact checker accepts every clean decoded trace.

**Clean damage:** C (1.000) and N (0.972) stay above the 0.97 clean floor. I falls below it at 0.944, missing 4 of 72 clean cases.

| arm | clean accuracy | primary accuracy | difference [95% case bootstrap] | rule |
| --- | ---: | ---: | --- | --- |
| P (plain) | 1.000 | 0.626 | C - P = +0.185 [+0.141, +0.231] | R1 needs >= +0.10: **met** |
| I (same information) | 0.944 | 0.714 | C - I = +0.097 [+0.041, +0.154] | R2 needs >= +0.05: **met** |
| N (decode only) | 0.972 | 0.561 | C - N = +0.250 [+0.188, +0.315] | R3 needs >= +0.10: **met** |
| C (constraint residual) | 1.000 | 0.811 | | |
| | | | I - P = +0.088 [+0.032, +0.146] | R4 needs >= +0.10: not met |

The primary condition is sigma_e = 1.0, with the noise after loop j in {4, 8, 12} and four noise seeds: 864 runs per arm.

**The registered predictions** (one of four held):

| prediction | outcome |
| --- | --- |
| P's records equal H-A3e's (primary 0.626) | held: 1,368 of 1,368 records identical |
| C's primary accuracy rises above 0.639, but R1 is not met | **failed.** It rose to 0.811, and R1 is met: +0.185, lower bound +0.141 |
| R4 is met, and I keeps clean runs intact (>= 0.97) | **failed on both.** I - P = +0.088, under the 0.10 margin. I's clean accuracy is 0.944 |
| R2 is not met | **failed.** C - I = +0.097, lower bound +0.041 |

## Answer

- **The registered branch for R1 met applies.** The instrument is valid, and C is above the clean floor.
  - The transition residual repairs D1's answer once its training loss reaches the answer: 0.811 against P's 0.626 at matched corruption.
  - H-A3e's null for C reflects the local loss's reach. That loss scored only pi on slot 2, so the corrector weights for slots 3-5 stayed at zero.
- **C beats the information-matched control (R2).**
  - I receives the same two inputs as C, the map's prediction e and pi's decode p, and combines them freely: G([e, p]).
  - C is I with the weight on e tied to minus the weight on p.
  - Under the same loss, the tied form scores 0.097 higher. Its advantage comes from the residual form, not from information I lacks.
  - This reverses H-A3e, where C - I was -0.166 under the local loss.
- **C does this at no clean cost.** It answers all 72 clean cases. At sigma_e/2 it is level with P (0.968 against 0.963). At 2 sigma_e it keeps its lead (0.722 against 0.542, +0.181 [+0.106, +0.264]).
- **The answer loss costs I its H-A3e repair.**
  - I's primary accuracy falls from 0.804 to 0.714, and its trace acceptance from 0.985 to 0.691.
  - R4 is not met. Its lower bound (+0.032) is above 0, but the difference is under the 0.10 margin.
  - I's clean cost grows from 3 to 4 missed cases.
- **N, without the map, stays below plain recurrence:** N - P = -0.065 [-0.106, -0.021] (post hoc).

## Exploratory: what changed for C and I

This section is a post-hoc reading. None of it is a registered test.

- **C's answer now follows its trace far more often.**
  - C's trace acceptance is unchanged from H-A3e: 0.985, against 0.987.
  - Its answer is right in 0.813 of its 851 valid-trace runs, against 0.638 in H-A3e.
  - On the 576 non-`sink` primary runs, its trace passes in 569, and its answer is right in 0.721 of those (H-A3e: 0.458).
  - C's answer equals pi's final decode in 0.810 of primary runs (H-A3e 0.637), level with P's 0.796.
  - Against its H-A3e records, C gains 159 primary runs and loses 10.
- **C still has a gap.** Its decode is right in 0.985 of primary runs and its answer in 0.811.
- **I lost its decode repair and gained little on the answer.**
  - Its trace acceptance by family: `sink` 0.861, `permutation` 0.615, `long_cycle` 0.597. In H-A3e it was 0.983-0.990 for each.
  - Against its H-A3e records, I gains 74 primary runs and loses 152.
  - Why the answer loss costs I and not C is not established by this run.
- **The answer loss trains the slot 3-5 corrections.**
  - In H-A3e those weights were exactly zero in all three arms.
  - Here their norms are 6.23 (C), 9.88 (I) and 10.54 (N), against 4.61, 3.16 and 2.37 on slot 2.
- **I's clean misses look like H-A3e's.** In all 4 its decoded trace is valid and the generated answer is wrong: one `permutation` case and three `long_cycle` cases. N's 2 clean misses are of the same kind.
- **CPU check on the trained correctors.**
  - It reruns the [03e check](../algernon-03e/report.md#exploratory-c-repairs-what-pi-reads-and-the-answer-does-not-follow) with the H-A3f correctors.
  - Cases: the same 12 generated cases outside the roster (calibration seed 20261108), at j = 4 and sigma = 1, with the fixed noise.
  - Answers right: P 9, C 10, I 7, N 6 of 12 (H-A3e: P 9, C 8, I 12, N 8). Medians:

| arm | slot RMS at loop j - 1 (F's output) | slot RMS at loop j, after the noise | margin at j before correction | margin at j after correction | margin at clean loop j - 1: F alone | the same, after correction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P | 6.95 | 9.7 | -2 | -2 | +13 | +13 |
| C | 6.95 | 9.7 | -2 | +143 | +13 | +13 |
| I | 3.78 | 7.9 | -10 | +62 | -0 | +40 |
| N | 3.81 | 8.0 | -12 | -17 | -5 | -13 |

Relative distance from the clean plain run's slots, ‖s - s_clean‖ / ‖s_clean‖, given for slots 0-1 / slot 2 (what pi reads) / slots 3-5:

| arm | loop j, after correction | loop j + 1, F's output | loop 16 |
| --- | --- | --- | --- |
| P | 2.04 / 0.98 / 1.40 | 0.63 / 0.30 / 0.56 | 0.05 / 0.00 / 0.02 |
| C | 2.37 / 1.51 / 1.71 | 0.77 / 0.39 / 0.66 | 0.27 / 0.61 / 0.45 |
| I | 2.49 / 1.60 / 2.58 | 1.05 / 0.92 / 1.13 | 1.33 / 1.29 / 2.26 |
| N | 2.34 / 1.40 / 2.30 | 0.77 / 0.80 / 1.15 | 1.17 / 1.05 / 1.71 |

- **C still pushes only where the decode is wrong.** Its clean-loop margin stays at +13, as in H-A3e.
  - At loop j it now also writes slots 3-5 (1.71 from the clean run, against 1.40 for the noise alone).
  - By loop 16, its slots 0-1 are closer to the clean run than in H-A3e (0.27 against 0.76). Slot 2 and slots 3-5 are farther (0.61 / 0.45 against 0.47 / 0.21).
  - On these 12 cases, the answer follows pi's decode more often, which the full evaluation shows at scale (0.810 against 0.637).
- **I and N now shrink the slots on every loop.**
  - Their slots after loop j - 1 have a median RMS of 3.8, against 6.95 for P and C. In H-A3e they were 6.6 (I) and 7.1 (N).
  - The noise is fixed in absolute terms (H-A3d's fix), so the same noise is larger relative to their slots than to P's or C's.
  - That is a cost of their learned correction, not a scale the evaluation gives them. It is one candidate reason I lost its decode repair; this run does not test it.
  - I's clean-loop margin is about 0 from F and +40 after its correction. Its decode depends on its own push.
  - C's slots keep F's RMS (6.95).

## Evaluation

**Answer accuracy** (72 cases; primary 864 runs per arm, sigma_e/2 and 2 sigma_e 216 each):

| arm | clean | primary (sigma_e = 1) | sigma_e/2 = 0.5 | 2 sigma_e = 2 |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.626 | 0.963 | 0.542 |
| C | 1.000 | 0.811 | 0.968 | 0.722 |
| I | 0.944 | 0.714 | 0.889 | 0.602 |
| N | 0.972 | 0.561 | 0.782 | 0.495 |

Every arm's generated answer is a well-formed digit in every run.

**Whole-trace acceptance by the exact BN254 R1CS checker** (the 16-step pi-decoded trace against the public machine):

| arm | clean | primary | sigma_e/2 | 2 sigma_e |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.304 | 0.731 | 0.093 |
| C | 1.000 | 0.985 | 1.000 | 0.755 |
| I | 1.000 | 0.691 | 0.958 | 0.394 |
| N | 1.000 | 0.293 | 0.750 | 0.088 |

**Answer = pi's final decode:**

| arm | clean | primary | sigma_e/2 | 2 sigma_e |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.796 | 0.926 | 0.745 |
| C | 1.000 | 0.810 | 0.968 | 0.833 |
| I | 0.944 | 0.804 | 0.912 | 0.782 |
| N | 0.972 | 0.681 | 0.843 | 0.611 |

**By family, primary condition** (24 cases each). The first block is accuracy; the second is trace acceptance.

| arm | `sink` | `permutation` | `long_cycle` | trace: `sink` | trace: `permutation` | trace: `long_cycle` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.503 | 0.375 | 0.542 | 0.205 | 0.167 |
| C | 1.000 | 0.736 | 0.698 | 0.979 | 0.986 | 0.990 |
| I | 1.000 | 0.660 | 0.483 | 0.861 | 0.615 | 0.597 |
| N | 1.000 | 0.451 | 0.233 | 0.569 | 0.205 | 0.104 |

**By noise loop, primary condition:**

| arm | j = 4 | j = 8 | j = 12 |
| --- | ---: | ---: | ---: |
| P | 0.642 | 0.587 | 0.649 |
| C | 0.802 | 0.833 | 0.799 |
| I | 0.642 | 0.719 | 0.781 |
| N | 0.556 | 0.545 | 0.583 |

**Other case-bootstrap differences, primary** (post hoc):
- I - N = +0.153 [+0.095, +0.215].
- C - P on trace acceptance = +0.681 [+0.627, +0.735].
- I - P on trace acceptance = +0.387 [+0.346, +0.429].

**Agreement with P under the primary condition:**
- An arm's decoded trace equals P's in 263 (C), 293 (I) and 327 (N) of 864 runs.
- Whether its answer is correct matches P's in 682, 620 and 700.

## Training

The recipe is H-A3e's with the answer loss added: 600 steps of 8 generated cases per arm, j uniform in 1..7, 25% of cases without noise, Adam at lr 1e-3. The batches and noise draws are H-A3e's. The loss now includes the answer cross-entropy, so its values are not comparable with H-A3e's local-only loss.

| arm | parameters | mean loss per 100 steps | median loss, last 100 steps | steps below 0.001 | max \|w\| | weight norm on slots 0 / 1 / 2 / 3-5 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| C | 27,648 | 4.59, 2.95, 2.26, 1.58, 1.63, 1.26 | 1.487 | 17 | 0.233 | 2.20 / 2.09 / 4.61 / 6.23 |
| I | 55,296 | 4.81, 3.58, 2.46, 1.90, 2.14, 1.80 | 1.358 | 2 | 0.253 | 1.82 / 1.90 / 3.16 / 9.88 |
| N | 27,648 | 6.32, 6.25, 5.83, 5.68, 5.36, 5.18 | 4.844 | 0 | 0.350 | 2.14 / 2.24 / 2.37 / 10.54 |

## Limits

- **One F, one training recipe, one noise level for this loss.**
  - F is D1: GPT-2 small, 6-node maps, one seed.
  - H-A3f ran only at sigma_e = 1. Whether the answer loss also lifts C at sigma\* = 16, where H-A3d found plain accuracy at its floor, is not tested.
- **R2's sign depends on the loss.** C - I is -0.166 under the local loss (H-A3e) and +0.097 under the answer loss (here). The comparison holds for this loss, not for every loss.
- **C still answers wrong in about 1 run in 5** in which its decode is right.
- **Training was interrupted twice and resumed from checkpoints.** The step-seeded schedule makes the resumed batches the uninterrupted ones. I's steps 276-288 were computed twice, and so were the few steps after 425 that ran before the second stop. Each resume restores the optimizer state saved with the checkpoint.
- **I's and N's trained correctors shrink the slots to about half of F's RMS.** Under the fixed absolute noise that makes their corruption larger relative to their state. R2 compares the arms at equal absolute noise, as registered.
- **The exploratory findings come from reading the records after the run,** plus a 12-case CPU check.

## Files

| file | content |
| --- | --- |
| [manifest.json](manifest.json) | code commit, weights and roster hashes, corrector hashes (weights kept outside git), arguments (including `loss`, `sigma_rule` and `noise_ref_rms`), kappa, environment, hardware, commands, receipt hashes |
| [calibrate.json](calibrate.json) | H-A3's sigma calibration, reused. sigma_e is picked from its `plain_accuracy_by_sigma` |
| [scale.json](scale.json) | H-A3b's per-case clean slot RMS on the 48 calibration cases, and r = `slot_rms_mean` |
| [train.json](train.json) | per-step loss for C, I and N, and parameter counts |
| [eval.json](eval.json) | summary (accuracy, trace acceptance, by j, by family, decisions R1-R4 with bootstrap intervals, clean damage) and all 5,472 run records (answer, prediction, decoded trace, exact-check verdict). The generated text is in the run directory's copy only. |
