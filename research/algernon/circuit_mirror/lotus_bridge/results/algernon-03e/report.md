# Algernon-03e result (H-A3e): at a noise level that damages the state without wiping it, the constraint residual repairs what pi reads but not the answer, and the same-information control repairs the answer

Run 2026-11-10 11:48 to 13:13 on the local RTX 5080, under
[EXPERIMENT.md, H-A3e](../../../EXPERIMENT.md#h-a3e-h-a3d-at-a-noise-level-that-damages-the-state-without-wiping-it-registered-2026-11-10-after-h-a3ds-evaluation-was-read-before-any-h-a3e-run).
It was registered and pushed (`b1bcaa6`) after H-A3d's evaluation was read and before any H-A3e run.
- H-A3e is H-A3d with one change, the noise level:
  - sigma_e = 1.0 is the grid value whose calibrated plain accuracy (0.771) is closest to the midpoint of the grid's highest and lowest accuracies, (1.000 + 0.569) / 2.
  - H-A3 to H-A3d used sigma\* = 16, where plain accuracy is at its floor.
- The noise std is sigma_e x r = 6.92 per slot element for every arm, case and loop, in training and evaluation. kappa = 6.92.
- F is 02D's arm D1 (`80eda0824cff`), frozen. pi is D1's own LM head on slot 2. The correctors are bias-free.
- Code: `b1bcaa6` for training and evaluation. P was evaluated again.
- The run had no crash or restart.

**The registered decisions: R1, R2 and R3 are not met. R4 is met.** The instrument is valid: P answers all 72 clean cases, and the exact checker accepts every clean decoded trace.

**Clean damage:** C and N stay above the 0.97 clean floor. I falls below it at 0.958, missing 3 of 72 clean cases.

| arm | clean accuracy | primary accuracy | difference [95% case bootstrap] | rule |
| --- | ---: | ---: | --- | --- |
| P (plain) | 1.000 | 0.626 | C - P = +0.013 [-0.001, +0.027] | R1 needs >= +0.10: not met |
| I (same information) | 0.958 | 0.804 | C - I = -0.166 [-0.228, -0.102] | R2 needs >= +0.05: not met |
| N (decode only) | 0.986 | 0.546 | C - N = +0.093 [+0.049, +0.140] | R3 needs >= +0.10: not met |
| C (constraint residual) | 1.000 | 0.639 | | |
| | | | I - P = +0.178 [+0.112, +0.243] | R4 needs >= +0.10: **met** |

The primary condition is sigma_e = 1.0, with the noise after loop j in {4, 8, 12} and four noise seeds: 864 runs per arm.

**The registered predictions:**

| prediction | outcome |
| --- | --- |
| P's primary accuracy lies between 0.60 and 0.85 | held: 0.626 |
| C keeps clean runs intact (>= 0.97) | held: 1.000, every clean trace accepted |
| C - P is positive, but R1 is not met | held: +0.013, and the interval includes 0 |
| R2, R3 and R4 are not met | R2 and R3 held. **R4 failed: I - P = +0.178, lower bound +0.112.** R3's lower bound is above 0 (+0.049), but the difference is under its 0.10 margin |

## Answer

- **The constraint residual does not repair the answer at the midpoint either.**
  - C's primary accuracy is 0.639 against P's 0.626.
  - With H-A3d's null at the floor, R1 is now not met at both noise levels.
- **The registered branch for R1 not met said Phase 4 closes with a null for additive linear correctors under this recipe. R4 contradicts that branch as worded.**
  - I is an additive linear corrector from the same family. Under the same recipe, it repairs the answer at matched corruption: 0.804 against 0.626.
  - The null holds for the residual form G(e - p), not for the family.
- **R4 met carries the meaning H-A3d registered for it:** the map's prediction written into the slots repairs D1's state at matched corruption, now at sigma = 1.
  - The noise is fixed at sigma x r for every arm, so I cannot gain by shrinking its slots, which was H-A3c's confound.
  - The repair holds at 2 sigma_e = 2.0, I - P = +0.157 [+0.088, +0.227]. At sigma_e/2 = 0.5, P is already at 0.963 and I is level with it.
- **I pays for the repair on clean runs.** It misses 3 of 72 clean cases (two `permutation`, one `long_cycle`). In all three, its decoded trace is valid and the generated answer is wrong.
- **R2 reversed: the residual's form costs answer accuracy against the free combination of the same inputs,** C - I = -0.166. C is I with the weight on e tied to minus the weight on p.

## Exploratory: C repairs what pi reads, and the answer does not follow

This section is a post-hoc reading. None of it is a registered test.

- **C's decoded trace is almost always right after the noise, and so is I's.**
  - Exact-checker acceptance, primary: P 0.304, C 0.987, I 0.985, N 0.303.
  - pi's decode is right at the noise loop j itself in 0.99 (C) and 0.985 (I) of primary runs, against 0.559 for P.
  - The correction is applied after the noise and before pi reads the slots. At this noise level a push of the size H-A3d measured is enough to flip the decode back.
- **C's answer does not follow its trace. I's does.**
  - When C's trace is valid (853 of 864 primary runs), its answer is right in 0.638.
  - The corresponding share is 0.810 for I and 0.943 for P's 263 valid-trace runs.
  - On the 576 non-`sink` primary runs, C's trace passes in 570 and its answer is right in 0.458, against P's 0.439. I's answer is right in 0.707.
  - C's answer equals pi's final decode in 0.637 of primary runs, against 0.796 (P) and 0.808 (I).
- **The local loss cannot train corrections on slots 3-5.**
  - The loss reads pi on slot 2 after loops j and j + 1, and the slots attend causally.
  - No gradient reaches the part of G that writes slots 3-5. In all three trained correctors, those weights are exactly zero, their initial value.
  - The generated answer reads all six slots. Every arm leaves the noise on slots 3-5 at loop j to F.
  - The CPU check below suggests this gap is not what breaks C's answer. Under plain recurrence, F itself returns slots 3-5 to the clean run within a few loops.
- **CPU check on the trained correctors.**
  - Cases: 12 generated cases outside the roster (calibration seed 20261108) at j = 4 and sigma = 1, with the fixed noise.
  - It reruns the [03d check](../algernon-03d/report.md#exploratory-c-keeps-more-decoded-traces-on-the-machine-and-the-answer-does-not-follow) at H-A3e's noise level, and adds each arm's distance from the clean plain run's slots.
  - Answers right: P 9, C 8, I 12, N 8 of 12. Medians:

| arm | slot RMS at loop j - 1 (F's output) | slot RMS at loop j, after the noise | margin at j before correction | margin at j after correction | margin at clean loop j - 1: F alone | the same, after correction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P | 6.95 | 9.7 | -2 | -2 | +13 | +13 |
| C | 6.95 | 9.7 | -2 | +130 | +13 | +13 |
| I | 6.59 | 9.5 | -2 | +159 | +5 | +101 |
| N | 7.08 | 9.8 | -11 | -14 | -2 | -38 |

Relative distance from the clean plain run's slots, ‖s - s_clean‖ / ‖s_clean‖, given for slots 0-1 / slot 2 (what pi reads) / slots 3-5:

| arm | loop j, after correction | loop j + 1, F's output | loop 16 |
| --- | --- | --- | --- |
| P | 2.04 / 0.98 / 1.40 | 0.63 / 0.30 / 0.56 | 0.05 / 0.00 / 0.02 |
| C | 2.17 / 1.47 / 1.40 | 0.71 / 0.37 / 0.56 | 0.76 / 0.47 / 0.21 |
| I | 3.27 / 1.74 / 1.43 | 1.11 / 0.87 / 0.63 | 2.76 / 1.32 / 0.25 |
| N | 3.56 / 1.40 / 1.43 | 1.03 / 0.76 / 0.63 | 2.76 / 1.20 / 0.43 |

- **The corrections overwrite pi's read rather than restore F's state.**
  - At sigma = 1 the noise costs about 15 logits of margin (+13 to -2).
  - C's correction lifts the margin to +130 and I's to +159, about ten times the loss. Both move slot 2 further from the clean run than the noise alone left it (1.47 and 1.74 against 0.98).
- **C's push vanishes on clean loops; I's does not.**
  - C leaves the clean-loop margin at +13.
  - I lifts it from +5 to +101 on every loop: it runs the public map on its own decode throughout. F's own margin under I falls to +5, and I's slots end farthest from the clean run (2.76 / 1.32 at loop 16).
  - The constant push is the likely source of I's 3 clean misses.
- **Plain recurrence returns to the clean run; C does not.**
  - By loop 16, P's median slots are back on the clean run, slots 3-5 included (0.02).
  - C's slots 0-2 stay off it (0.76 / 0.47), and slots 3-5 follow (0.21).
  - C's decode is right, but its slots are in a state the answer readout handles differently from pi. I's slots are farther off still, yet its every-loop push keeps the answer in step with its decode.
- **N, without the map, pushes the decode away from the true state** on clean loops (-2 to -38), as in H-A3d.

## Evaluation

**Answer accuracy** (72 cases; primary 864 runs per arm, sigma_e/2 and 2 sigma_e 216 each):

| arm | clean | primary (sigma_e = 1) | sigma_e/2 = 0.5 | 2 sigma_e = 2 |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.626 | 0.963 | 0.542 |
| C | 1.000 | 0.639 | 0.958 | 0.542 |
| I | 0.958 | 0.804 | 0.944 | 0.699 |
| N | 0.986 | 0.546 | 0.741 | 0.509 |

Every arm's generated answer is a well-formed digit in every run.

**Whole-trace acceptance by the exact BN254 R1CS checker** (the 16-step pi-decoded trace against the public machine):

| arm | clean | primary | sigma_e/2 | 2 sigma_e |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.304 | 0.731 | 0.093 |
| C | 1.000 | 0.987 | 1.000 | 0.745 |
| I | 1.000 | 0.985 | 1.000 | 0.750 |
| N | 0.986 | 0.303 | 0.648 | 0.130 |

**Answer = pi's final decode:**

| arm | clean | primary | sigma_e/2 | 2 sigma_e |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.796 | 0.926 | 0.745 |
| C | 1.000 | 0.637 | 0.958 | 0.542 |
| I | 0.958 | 0.808 | 0.944 | 0.792 |
| N | 1.000 | 0.796 | 0.894 | 0.810 |

**By family, primary condition** (24 cases each). The first block is accuracy; the second is trace acceptance.

| arm | `sink` | `permutation` | `long_cycle` | trace: `sink` | trace: `permutation` | trace: `long_cycle` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.503 | 0.375 | 0.542 | 0.205 | 0.167 |
| C | 1.000 | 0.500 | 0.417 | 0.983 | 0.990 | 0.990 |
| I | 1.000 | 0.753 | 0.660 | 0.983 | 0.983 | 0.990 |
| N | 1.000 | 0.396 | 0.243 | 0.677 | 0.139 | 0.094 |

N's one clean miss is a `long_cycle` case.

**By noise loop, primary condition:**

| arm | j = 4 | j = 8 | j = 12 |
| --- | ---: | ---: | ---: |
| P | 0.642 | 0.587 | 0.649 |
| C | 0.656 | 0.615 | 0.646 |
| I | 0.747 | 0.830 | 0.837 |
| N | 0.472 | 0.552 | 0.615 |

**Other case-bootstrap differences, primary** (post hoc): N - P = -0.080 [-0.126, -0.037]; I - N = +0.258 [+0.200, +0.317]; C - P on trace acceptance = +0.683 [+0.630, +0.737].

**Agreement with P under the primary condition:**
- An arm's decoded trace equals P's in 263 (C), 264 (I) and 264 (N) of 864 runs.
- Whether its answer is correct matches P's in 821, 624 and 675.

## Training

The recipe is as in H-A3d: 600 steps of 8 generated cases per arm, j uniform in 1..7, 25% of cases without noise, Adam at lr 1e-3. The batches and noise draws are H-A3d's, scaled by sigma_e instead of sigma\*.

| arm | parameters | mean loss per 100 steps | median loss, last 100 steps | steps below 0.001 | max \|w\| | weight norm on slots 0 / 1 / 2 / 3-5 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| C | 27,648 | 1.46, 0.91, 0.37, 0.46, 0.33, 0.13 | 0.003 | 35 | 0.132 | 1.54 / 1.59 / 4.43 / 0 |
| I | 55,296 | 1.60, 0.81, 0.22, 0.27, 0.52, 0.15 | 0.000 | 202 | 0.137 | 3.14 / 4.06 / 5.42 / 0 |
| N | 27,648 | 2.90, 2.92, 2.19, 2.43, 2.12, 1.98 | 1.646 | 0 | 0.274 | 4.03 / 6.31 / 2.96 / 0 |

C and I both drive the local loss near zero, which matches their trace acceptance. N cannot, because without the map it has no way to know the successor.

## Limits

- **One F, one training recipe, two noise levels.** F is D1 (GPT-2 small, 6-node maps, one seed). sigma_e = 1 is the midpoint pick, and sigma\* = 16 (H-A3d) is the floor.
- **The local loss scores pi, which reads slot 2.** It cannot train corrections on slots 3-5, which the answer reads. The registered rules score the answer, so R1-R4 test the corrector together with this loss.
- **I's repair costs 3 clean cases.** It sits just under the 0.97 floor, which the rules report but do not penalise in R4.
- **The exploratory findings come from reading the records after the run,** plus a 12-case CPU check.

## Files

| file | content |
| --- | --- |
| [manifest.json](manifest.json) | code commit, weights and roster hashes, corrector hashes (weights kept outside git), arguments (including `sigma_rule` and `noise_ref_rms`), kappa, environment, hardware, commands, receipt hashes |
| [calibrate.json](calibrate.json) | H-A3's sigma calibration, reused. sigma_e is picked from its `plain_accuracy_by_sigma` |
| [scale.json](scale.json) | H-A3b's per-case clean slot RMS on the 48 calibration cases, and r = `slot_rms_mean` |
| [train.json](train.json) | per-step loss for C, I and N, and parameter counts |
| [eval.json](eval.json) | summary (accuracy, trace acceptance, by j, by family, decisions R1-R4 with bootstrap intervals, clean damage) and all 5,472 run records (answer, prediction, decoded trace, exact-check verdict). The generated text is in the run directory's copy only. |
