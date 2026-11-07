# Algernon-03d result (H-A3d): at one fixed noise scale no corrector repairs the answer, and H-A3c's I result was the noise-scale interaction

Run 2026-11-07 23:29 to 2026-11-08 01:03 on the local RTX 5080, under
[EXPERIMENT.md, H-A3d](../../../EXPERIMENT.md#h-a3d-h-a3c-with-one-fixed-noise-scale-for-every-arm-registered-2026-11-07-after-h-a3cs-evaluation-was-read-before-any-h-a3d-run),
registered and pushed (`b0202d6`) after H-A3c's evaluation was read and before any H-A3d run.
- H-A3d is H-A3c with one change: the noise std is sigma x r for every arm, case and loop, in training and in evaluation. r = 6.92 is the mean clean slot RMS in H-A3b's `scale.json`, so at sigma\* = 16 the std is 110.78 per slot element.
- F is 02D's arm D1 (`80eda0824cff`), frozen. pi is D1's own LM head on slot 2. sigma\* = 16 and kappa = 110.78 are reused from H-A3b; the correctors are bias-free as in H-A3c.
- Code: `b0202d6` for training and evaluation. P was evaluated again. 1,256 of its 1,368 records are identical to H-A3c's; the 112 that differ are noised runs, whose noise changed from P's own slot RMS at loop j to r.
- The run had no crash or restart.

**The registered decisions: R1, R2, R3 and R4 are all not met.** The instrument is valid: P answers all 72 clean cases, and the exact checker accepts every clean decoded trace. **C and I keep clean runs intact. N falls below the 0.97 clean floor.**

| arm | clean accuracy | primary accuracy | difference [95% case bootstrap] | rule |
| --- | ---: | ---: | --- | --- |
| P (plain) | 1.000 | 0.500 | C - P = -0.013 [-0.052, +0.027] | R1 needs >= +0.10: not met |
| I (same information) | 1.000 | 0.455 | C - I = +0.032 [+0.002, +0.061] | R2 needs >= +0.05: not met |
| N (decode only) | 0.917 | 0.506 | C - N = -0.019 [-0.057, +0.022] | R3 needs >= +0.10: not met |
| C (constraint residual) | 1.000 | 0.487 | | |
| | | | I - P = -0.045 [-0.087, -0.005] | R4 needs >= +0.10: not met |

The primary condition is sigma\* = 16, noise after loop j in {4, 8, 12}, four noise seeds: 864 runs per arm.

**The registered predictions:**

| prediction | outcome |
| --- | --- |
| P's primary accuracy stays within 0.05 of 0.494 | held: 0.500 |
| C stays at P, and R1 is not met | held: C - P = -0.013 |
| R4 is not met, and I's primary accuracy falls below 0.65 | held: I is 0.455, below P, and the interval for I - P excludes 0 |
| R2 and R3 are not met | held. R2's lower bound is above 0 (+0.002), but the difference is under its 0.05 margin |

## Answer

- **H-A3c's I result came from the noise-scale interaction.** Under one fixed noise for every arm:
  - I's primary accuracy falls from 0.888 to 0.455, below P's 0.500;
  - its exact-checked trace acceptance falls from 0.999 to 0.139;
  - its training loss no longer falls: means of 15.5-21.1 per 100 steps, against H-A3c's 13.0 falling to 2.4.
- **This is the outcome EXPERIMENT.md registered for R4 not met: no arm under this recipe repairs D1's slot state at sigma\*.** None of C, I or N answers better than plain recurrence at matched corruption.
- **Without the noise interaction, no arm shrinks the state.**
  - A post-hoc CPU check (below) finds every arm's slot RMS before the noise at F's own scale, 6.85-7.02. In H-A3c, I's and N's were 0.92 and 0.66.
  - Shrinking the slots no longer reduces the noise, so training no longer finds it.
- **C and I stay safe on clean runs:** clean accuracy 1.000, and every clean trace is accepted.
- **N damages clean runs.**
  - Clean accuracy is 0.917, and only 9 of N's 72 clean decoded traces pass the checker.
  - pi's decode is wrong somewhere before the noise loop in 86% of N's primary runs.
  - The generated answer, which reads all six slots, mostly stays right.

## Exploratory: C keeps more decoded traces on the machine, and the answer does not follow

This section is a post-hoc reading. None of it is a registered test.

- **C's decoded trace passes the exact checker twice as often as P's.**
  - Primary: 0.156 against 0.075, C - P = +0.081 [+0.059, +0.105] (same case bootstrap). At sigma\*/2: +0.116 [+0.069, +0.167].
  - On the bijective `long_cycle` maps, C's trace passes in 38 of 288 primary runs; P's passes in 1.
  - I's trace passes in 0.139 of primary runs (26 of 288 on `long_cycle`).
- **The repair shows one loop after the noise.** pi's decode is right right after the noise loop j in 0.199 (P), 0.216 (C) and 0.235 (I) of primary runs. At loop j + 1 the shares are 0.198, 0.377 and 0.448.
  - By loop j + 2 the arms are level (0.490, 0.469, 0.461).
- **C's repaired traces do not carry the answer.**
  - When C's trace is valid, its generated answer is right in 86% of primary runs, and in 66% on `long_cycle`.
  - When C's trace is invalid, its answer is right in 0.418 of runs, against P's 0.459 on its own invalid traces.
  - The two effects cancel, and C's accuracy stays at P's.
  - pi reads slot 2 only, while the answer is generated from all six slots. The residual's push repairs what pi reads in some runs, but not what the answer reads.
- **CPU check on the trained correctors.** It used 12 generated cases outside the roster (calibration seed 20261108) at j = 4, with the fixed noise, the same rerun as the [03c check](../algernon-03c/report.md#post-hoc-check-how-i-repairs-and-why-c-cannot). Medians:

| arm | slot RMS at loop j - 1 (F's output) | slot RMS at loop j, after the noise | margin at j before correction | margin at j after correction | margin at clean loop j - 1: F alone | the same, after correction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P | 6.95 | 111.1 | -228 | -228 | +13 | +13 |
| C | 6.95 | 111.1 | -228 | -207 | +13 | +13 |
| I | 6.85 | 111.1 | -216 | -170 | +16 | +23 |
| N | 7.02 | 111.1 | -233 | -225 | +6 | -1 |

- **The pushes at the noise loop are a tenth to a fifth of the noise.** The margin is the true state's logit minus the best other digit's.
  - C moves the margin by about +21 and I by about +46, against a noise cost of about 240.
  - Neither fixes the decode at loop j. Both leave a partial repair that F carries into loop j + 1.
- **I's code no longer dominates F.** On a clean loop, I's correction moves the margin from +16 to +23. In H-A3c it moved it from +10 to +58, on slots about 7 times smaller.
- **N's correction pulls the decode off the true state on clean loops** (+6 to -1). This is the clean damage in the evaluation.

## Evaluation

**Answer accuracy** (72 cases; primary 864 runs per arm, sigma\*/2 and 2 sigma\* 216 each):

| arm | clean | primary (sigma\* = 16) | sigma\*/2 = 8 | 2 sigma\* = 32 |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.500 | 0.505 | 0.477 |
| C | 1.000 | 0.487 | 0.472 | 0.486 |
| I | 1.000 | 0.455 | 0.509 | 0.477 |
| N | 0.917 | 0.506 | 0.495 | 0.495 |

Every arm's generated answer is a well-formed digit in every run.

**Whole-trace acceptance by the exact BN254 R1CS checker** (the 16-step pi-decoded trace against the public machine):

| arm | clean | primary | sigma\*/2 | 2 sigma\* |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.075 | 0.079 | 0.037 |
| C | 1.000 | 0.156 | 0.194 | 0.083 |
| I | 1.000 | 0.139 | 0.218 | 0.065 |
| N | 0.125 | 0.000 | 0.000 | 0.000 |

**Answer = pi's final decode:**

| arm | clean | primary | sigma\*/2 | 2 sigma\* |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.740 | 0.759 | 0.727 |
| C | 1.000 | 0.737 | 0.690 | 0.685 |
| I | 1.000 | 0.911 | 0.935 | 0.889 |
| N | 0.264 | 0.275 | 0.310 | 0.259 |

**By family, primary condition** (24 cases each). The first block is accuracy; the second is trace acceptance.

| arm | `sink` | `permutation` | `long_cycle` | trace: `sink` | trace: `permutation` | trace: `long_cycle` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.351 | 0.149 | 0.201 | 0.021 | 0.003 |
| C | 1.000 | 0.299 | 0.163 | 0.219 | 0.118 | 0.132 |
| I | 1.000 | 0.198 | 0.167 | 0.253 | 0.073 | 0.090 |
| N | 1.000 | 0.358 | 0.160 | 0.000 | 0.000 | 0.000 |

N's clean misses are 5 `permutation` cases and 1 `long_cycle` case.

**By noise loop, primary condition:**

| arm | j = 4 | j = 8 | j = 12 |
| --- | ---: | ---: | ---: |
| P | 0.503 | 0.500 | 0.497 |
| C | 0.483 | 0.476 | 0.503 |
| I | 0.424 | 0.451 | 0.490 |
| N | 0.500 | 0.503 | 0.514 |

**Agreement with P under the primary condition:**
- An arm's decoded trace equals P's in 106 (C), 95 (I) and 0 (N) of 864 runs.
- Whether its answer is correct matches P's in 711, 713 and 755.

## Training

The recipe is as in H-A3 to H-A3c: 600 steps of 8 generated cases per arm, j uniform in 1..7, 25% of cases without noise, Adam at lr 1e-3. The batches and noise draws are H-A3c's, with the noise scaled by r instead of the run's own slot RMS.

| arm | parameters | mean loss per 100 steps | median loss, last 100 steps | max \|w\| | column norms of W |
| --- | ---: | --- | ---: | ---: | --- |
| C | 27,648 | 14.8, 15.7, 15.3, 16.2, 15.0, 15.2 | 15.1 | 0.165 | 1.72-2.08 |
| I | 55,296 | 15.5, 16.4, 16.9, 18.3, 21.1, 20.0 | 19.7 | 0.104 | 0.57-0.99 (e), 0.56-1.13 (p) |
| N | 27,648 | 14.7, 15.6, 15.6, 16.2, 15.2, 15.2 | 15.6 | 0.069 | 0.63-0.98 |

No arm has a step with loss below 0.001. In H-A3c, I had 199 such steps, and its losses fell from 13.0 to 2.4 while N's fell from 12.9 to 8.3.

## Limits

- **One F, one sigma\*, one training recipe.** F is D1 (GPT-2 small, 6-node maps, one seed), sigma\* is at the top of the registered grid, and kappa ties the corrector's reach to one noise std.
- **The matched quantity is absolute corruption.** r is fixed at the mean clean slot RMS. An arm that shrank its slots would now face a larger noise relative to its own signal. None did.
- **pi reads slot 2 only.** C's gain in trace acceptance is a gain in what pi reads. The registered rules score the generated answer.
- **The exploratory findings come from reading the records after the run,** plus a 12-case CPU check.

## Files

| file | content |
| --- | --- |
| [manifest.json](manifest.json) | code commit, weights and roster hashes, corrector hashes (weights kept outside git), arguments (including `noise_ref_rms`), kappa, environment, hardware, commands, receipt hashes |
| [calibrate.json](calibrate.json) | H-A3's sigma\* calibration, reused |
| [scale.json](scale.json) | H-A3b's per-case clean slot RMS on the 48 calibration cases, kappa, and r = `slot_rms_mean` |
| [train.json](train.json) | per-step loss for C, I and N, and parameter counts |
| [eval.json](eval.json) | summary (accuracy, trace acceptance, by j, by family, decisions R1-R4 with bootstrap intervals, clean damage) and all 5,472 run records (answer, prediction, decoded trace, exact-check verdict). The generated text is in the run directory's copy only. |
