# Algernon-03 result (H-A3, Phase 4): no arm repairs the noised slot state, because the registered correctors are too weak to act

Run 2026-11-07 00:00 to 04:01 on the local RTX 5080, under
[EXPERIMENT.md, H-A3](../../../EXPERIMENT.md#h-a3-phase-4-constraint-guided-correction-of-d1s-slot-state-under-noise-registered-2026-11-06-before-any-run),
registered and pushed (`f525c9a`) before any run.
- F is 02D's arm D1 (`80eda0824cff`), frozen.
- pi is D1's own LM head on slot 2.
- The correctors are trained on generated `random_function` maps.
- Evaluation uses the frozen roster's 72 calibration and held-out cases at d = 16 loops.

**The registered decisions: R1, R2 and R3 are all not met.** The instrument is valid: P answers all 72 clean cases, and the exact checker accepts every clean decoded trace.

- **Primary condition:** sigma\* = 16, noise after loop j in {4, 8, 12}, four noise seeds, 864 runs per arm.

| arm | primary accuracy | difference vs C [95% case bootstrap] | rule |
| --- | ---: | --- | --- |
| P (plain) | 0.494 | C - P = -0.007 [-0.028, +0.016] | R1 needs >= +0.10: not met |
| I (same information) | 0.505 | C - I = -0.017 [-0.043, +0.008] | R2 needs >= +0.05: not met |
| N (decode only) | 0.500 | C - N = -0.013 [-0.031, +0.006] | R3 needs >= +0.10: not met |
| C (constraint residual) | 0.487 | | |

- **This null comes from the recipe, not from a test of the constraint.**
  - The diagnostic, recorded after training and before any evaluation record was read
    ([EXPERIMENT.md](../../../EXPERIMENT.md#h-a3-training-diagnostic-recorded-2026-11-07-after-training-before-any-evaluation-record-was-read)),
    shows that the trained correctors move a digit logit at slot 2 by at most about 4. The noise moves each digit logit by about 300 (std).
  - [H-A3b](../../../EXPERIMENT.md#h-a3b-h-a3-with-the-correctors-output-in-units-of-the-noise-registered-2026-11-07-before-h-a3s-evaluation-was-read)
    was registered and pushed (`374ed2a`) before this evaluation started. It changes only the corrector's output scale, and it is running.

## Answer

- **The recipe's correctors stay about 100 times too small.** Each is zero-initialised and trained for 600 steps with Adam at lr 1e-3.
- **So nothing can be repaired.** Under the registered recipe, adding the constraint residual through such a corrector does not repair D1's slot state after it is wiped at sigma\*, and neither do the controls.
- **What H-A3 does not decide.** It cannot say whether the constraint residual helps more than the same information given another way (R2). That question needs correctors that can reach the noise, which H-A3b supplies.

## sigma\* calibration

Plain recurrence on 48 generated `random_function` cases (seed 20261108) at d = 16, j in {4, 8, 12}:

| sigma | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| plain accuracy | 1.000 | 0.944 | 0.771 | 0.681 | 0.646 | 0.653 | 0.569 |

- **sigma\* is 16 by the registered rule** (closest to 0.5), the largest value on the grid. At that scale the noise has a per-element std about 16 times the clean slot RMS, so it wipes the state.
- **Accuracy plateaus between sigma 2 and 8.** A plausible reason is that random maps merge states, so many wiped trajectories rejoin the true one before step 16. This was not tested.

## Evaluation

**Answer accuracy** (72 cases; the primary condition is 864 runs per arm, sigma\*/2 and 2 sigma\* are 216 each):

| arm | clean | primary (sigma\* = 16) | sigma\*/2 = 8 | 2 sigma\* = 32 |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.494 | 0.495 | 0.472 |
| C | 1.000 | 0.487 | 0.505 | 0.486 |
| I | 1.000 | 0.505 | 0.528 | 0.505 |
| N | 1.000 | 0.500 | 0.514 | 0.519 |

**Whole-trace acceptance by the exact BN254 R1CS checker:** the 16-step pi-decoded trace, checked against the public machine.

| arm | clean | primary | sigma\*/2 | 2 sigma\* |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.073 | 0.079 | 0.037 |
| C | 1.000 | 0.104 | 0.157 | 0.065 |
| I | 0.958 | 0.101 | 0.139 | 0.046 |
| N | 1.000 | 0.014 | 0.023 | 0.009 |

**By family, primary condition** (24 cases each):

| arm | `sink` (held-out) | `permutation` (calibration) | `long_cycle` (held-out) |
| --- | ---: | ---: | ---: |
| P | 1.000 | 0.330 | 0.153 |
| C | 1.000 | 0.330 | 0.132 |
| I | 1.000 | 0.351 | 0.163 |
| N | 1.000 | 0.358 | 0.142 |

**By noise loop, primary condition:**

| arm | j = 4 | j = 8 | j = 12 |
| --- | ---: | ---: | ---: |
| P | 0.500 | 0.483 | 0.500 |
| C | 0.497 | 0.455 | 0.510 |
| I | 0.507 | 0.465 | 0.542 |
| N | 0.500 | 0.465 | 0.535 |

**What the tables show** (post-hoc reading, not registered tests):

- **Sink maps recover by themselves.** All 24 `sink` cases are answered correctly in every arm, under every noise condition: the trajectory reaches the sink whatever state the noise leaves.
  - So about a third of the primary accuracy is free, and the largest possible gain over P is about 0.5.
  - On the bijective `long_cycle` maps, accuracy is 0.13-0.16, near the 1/6 of a random state.
- **The correctors change trajectories without steering them.**
  - Under the primary condition, an arm's decoded trace equals P's in only 162 (C), 93 (I) and 48 (N) of 864 runs.
  - Yet whether the answer is correct matches P's in 778, 757 and 763 runs.
  - After the noise, small additions change the path, but not towards the true state.
- **Trace acceptance is higher for C than for P** (0.104 vs 0.073 under the primary condition) and lowest for N (0.014). Acceptance is all-or-nothing over 16 steps, and none of this is a registered decision.
- **I alters clean decodes without changing clean answers.** Its corrector, active on every clean loop, made the exact checker reject three clean decoded traces: `calibration/permutation/11`, `calibration/permutation/12` and `test/long_cycle/1`. All three answers were still correct.
  - So I's clean trace acceptance is 0.958.
  - No arm falls below the registered 0.97 clean-accuracy floor.

## Training

- **Recipe:** 600 steps of 8 generated cases per arm, j uniform in 1..7, 25% of cases without noise. The local loss is cross-entropy of pi after loops j and j + 1.
- **The batches, loops and noise are identical across arms.**

| arm | parameters | mean loss, steps 1-100 | mean loss, steps 501-600 | max \|w\| | column norms of W (over 6 x 768) |
| --- | ---: | ---: | ---: | ---: | --- |
| C | 32,256 | 13.63 | 13.64 | 0.223 | 3.0-3.6 |
| I | 59,904 | 13.64 | 13.14 | 0.271 | 6.7-8.5 |
| N | 32,256 | 13.71 | 13.47 | 0.245 | 7.4-8.6 |

- **The loss does not trend.**
- **Adam bounds how far a weight can move:** at lr 1e-3 over 600 steps, each weight moves at most 0.6.
- **Scale on clean D1 runs** (CPU, four calibration cases):
  - the slots' RMS is 5.9-7.3, so the noise at sigma\* = 16 has a per-element std of about 105;
  - slot 2's norm is 166-207, against a noise norm on slot 2 of about 2,900.

## Limits

- **One F and one sigma\*.** F is D1 (GPT-2 small, 6-node maps, one seed). sigma\* came out at the top of the registered grid.
- **The null is about this recipe's corrector scale.** It says nothing about correctors that can reach the noise (H-A3b), or about other forms of correction.
- **pi reads slot 2 only, through D1's own head.** The other five slots are corrected but not read.
- **A third of the evaluation is at ceiling.** The 24 `sink` cases recover under every arm.

## Files

| file | content |
| --- | --- |
| [manifest.json](manifest.json) | code commit, weights and roster hashes, corrector hashes (weights kept outside git), arguments, environment, hardware, commands, receipt hashes |
| [calibrate.json](calibrate.json) | plain accuracy for each sigma, and sigma\* |
| [train.json](train.json) | per-step loss for C, I and N, and parameter counts |
| [eval.json](eval.json) | summary (accuracy, trace acceptance, by j, by family, decisions with bootstrap intervals, clean damage) and all 5,472 run records (answer, prediction, decoded trace, exact-check verdict). The generated text is in the run directory's copy only. |
