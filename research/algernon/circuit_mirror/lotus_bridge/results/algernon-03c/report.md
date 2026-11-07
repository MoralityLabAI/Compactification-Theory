# Algernon-03c result (H-A3c): bias-free C keeps clean runs intact but does not repair; I holds the trace by running the map, helped by a noise-scale interaction

Run 2026-11-07 20:14 to 23:13 on the local RTX 5080, under
[EXPERIMENT.md, H-A3c](../../../EXPERIMENT.md#h-a3c-h-a3b-with-bias-free-correctors-registered-2026-11-07-before-h-a3bs-evaluation-ran),
registered and pushed (`3da47a3`, parameter count corrected in `3e91afc`) before H-A3b's evaluation ran.
- H-A3c is H-A3b with one change: the correctors have no bias, so C's correction is zero whenever e = p.
- F is 02D's arm D1 (`80eda0824cff`), frozen. pi is D1's own LM head on slot 2. sigma\* = 16 and kappa = 110.78, both reused from H-A3b.
- Code: `1ce6afd` for training and evaluation. P's 1,368 records are identical to H-A3b's (and so to H-A3's).
- At 23:02 the evaluation died in arm N with `CUDA error: unknown error`, while NVML's power reading was unreadable for 2 s. vramorama restarted it. The restart reloaded every finished per-run record from `parts/` and computed only N's missing runs.

**The registered decisions: R1, R2 and R3 are all not met.** The instrument is valid: P answers all 72 clean cases, and the exact checker accepts every clean decoded trace. **C no longer damages clean runs.** **I and N fall below the 0.97 clean floor.**

| arm | clean accuracy | primary accuracy | difference vs C [95% case bootstrap] | rule |
| --- | ---: | ---: | --- | --- |
| P (plain) | 1.000 | 0.494 | C - P = -0.034 [-0.068, -0.000] | R1 needs >= +0.10: not met |
| I (same information) | 0.889 | 0.888 | C - I = -0.427 [-0.531, -0.317] | R2 needs >= +0.05: not met |
| N (decode only) | 0.722 | 0.499 | C - N = -0.038 [-0.091, +0.010] | R3 needs >= +0.10: not met |
| C (constraint residual) | 1.000 | 0.461 | | |

The primary condition is sigma\* = 16, noise after loop j in {4, 8, 12}, four noise seeds: 864 runs per arm. The upper end of C - P's interval is -2e-18, zero to rounding.

**The registered predictions:**

| prediction | outcome |
| --- | --- |
| C does not damage clean runs | held: clean accuracy 1.000, every clean trace accepted |
| R1 open | not met: C is no better than P, and slightly worse |
| R2 is not met | held, and reversed: I beats C by 0.43 |
| N stays at or below P | held: N - P = +0.005 [-0.035, +0.043] |

## Answer

- **Without its bias, C does what a residual corrector should on clean runs, and nothing useful under noise.**
  - C's correction vanishes when the decode satisfies the constraint, so clean runs are untouched (1.000; trace acceptance 1.000).
  - Under the primary noise, C's decode at the noise loop is right in 0.207 of runs, against P's 0.199. C does not repair the state where the noise lands.
  - C's training loss never fell: 14.8-16.3 per 100 steps, from first hundred to last.
- **I keeps pi's decoded trace on the machine in 99.9% of primary runs**, against P's 7.3% (case bootstrap: I - P = +0.926 [+0.895, +0.954]). Its answer accuracy is 0.888 in every condition, and its primary answer equals its own clean answer in 99.1% of runs.
- **The registered comparison cannot be read as "the map's information helps and the constraint's form hurts".** A post-hoc check (next section) shows two things:
  - **I runs the map.** I's input e is the map applied to I's own previous decode, which is the gold successor whenever that decode is right. I writes a code for e into the slots on every loop, and that code outweighs F's own slot-2 signal on clean loops as well as noisy ones.
  - **I and N face smaller noise.** The registered noise is sigma times the run's own slot RMS at loop j. Correcting on every loop shrinks F's slot RMS before the noise arrives, and with it the absolute noise. C, which is zero on clean loops, faces the full noise.
- **The brief's warning applies here.** The violation signal reveals the gold successor, and the control arm was given the same information, as the brief requires. The information-matched control, not the constraint arm, exploited it.

## Post-hoc check: how I repairs and why C cannot

This section is a post-hoc reading. It comes from a CPU rerun of the trained correctors on 12 generated cases outside the roster (the calibration seed 20261108), with noise at j = 4 and sigma\* = 16. That rerun logged the slots' RMS and slot 2's digit logits before and after each correction. The margin is the true state's logit minus the best other digit's. Medians over the 12 cases:

| arm | slot RMS at loop j - 1 (F's output) | slot RMS at loop j, after the noise | margin at j before correction | margin at j after correction | margin at clean loop j - 1: F alone | the same, after correction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P | 6.95 | 109.8 | -225 | -225 | +13 | +13 |
| C | 6.95 | 109.8 | -225 | -214 | +13 | +13 |
| I | 0.92 | 15.6 | -18 | +74 | +10 | +58 |
| N | 0.66 | 10.6 | -4 | -3 | +15 | +9 |

- **C's correction is about 10 logits against a noise that costs about 240.**
  - C is zero on clean loops, so its slots keep F's scale (RMS 6.95), and the noise arrives at full size.
  - At the noise loop its push moves the margin by +11 (median), far short of what it needs.
  - Bigger weights would be needed, and C's loss never fell, so training found no direction that pays off.
- **I shrinks the state before the noise arrives.**
  - I adds a code for its predicted state on every loop. F's slot output then has RMS 0.92 instead of 6.95.
  - The registered noise is sigma times that RMS, so I's runs get about 7 times less absolute noise: the slots' RMS after the noise is 15.6 against P's 109.8.
  - I's push, a fixed size set by kappa, moves the margin by about +90. That is enough against the smaller noise.
- **I's code outweighs F on every loop.**
  - On a clean loop, F alone gives the true state a margin of +10 and I's correction brings it to +58.
  - pi's decode is therefore mostly I's code for e = (previous decode) x M, the public map applied to I's own last state.
  - Once the decode at loop j is right, every later e is the gold successor, and I writes it.
- **N shrinks the state as well** (RMS 0.66) but writes a code for its own decode, which the noise has already moved. Its decode at loop j is right more often than P's (0.470 vs 0.199) because its noise is smaller, but nothing pulls it back once it is wrong.
- **H-A3b's correctors shrank the state too.** The same check on H-A3b's biased correctors gives slot RMS at loop j - 1 of 2.70 (C), 1.74 (I) and 0.49 (N), against P's 6.95.
  - H-A3b's N therefore faced about 14 times less absolute noise than P.
  - That qualifies its exploratory trace-acceptance gain. A note is added to the [03b report](../algernon-03b/report.md).

What the registered comparison measured is therefore:
- the constraint-residual corrector, at full noise, against a control that both runs the map itself and shrinks its own noise;
- not the residual's form against its information at matched corruption.

[H-A3d](../../../EXPERIMENT.md#h-a3d-h-a3c-with-one-fixed-noise-scale-for-every-arm-registered-2026-11-07-after-h-a3cs-evaluation-was-read-before-any-h-a3d-run) fixes the noise scale for every arm.

## Evaluation

**Answer accuracy** (72 cases; primary 864 runs per arm, sigma\*/2 and 2 sigma\* 216 each):

| arm | clean | primary (sigma\* = 16) | sigma\*/2 = 8 | 2 sigma\* = 32 |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.494 | 0.495 | 0.472 |
| C | 1.000 | 0.461 | 0.472 | 0.491 |
| I | 0.889 | 0.888 | 0.889 | 0.838 |
| N | 0.722 | 0.499 | 0.620 | 0.500 |

**Whole-trace acceptance by the exact BN254 R1CS checker** (the 16-step pi-decoded trace against the public machine):

| arm | clean | primary | sigma\*/2 | 2 sigma\* |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.073 | 0.079 | 0.037 |
| C | 1.000 | 0.156 | 0.213 | 0.102 |
| I | 1.000 | 0.999 | 1.000 | 0.921 |
| N | 0.944 | 0.075 | 0.204 | 0.042 |

C's trace acceptance beats P's (C - P = +0.083 [+0.061, +0.109], primary) while its answers do not.

**pi's decode after the noise, primary condition** (fraction of runs whose decode is right at loop j, j + 1, j + 2 and at the end; every arm's decode is right before j, except N's in 5% of runs):

| arm | j | j + 1 | j + 2 | loop 16 |
| --- | ---: | ---: | ---: | ---: |
| P | 0.199 | 0.198 | 0.487 | 0.492 |
| C | 0.207 | 0.399 | 0.464 | 0.461 |
| I | 0.999 | 1.000 | 1.000 | 1.000 |
| N | 0.470 | 0.192 | 0.530 | 0.543 |

**Generated answers.**
- P and C produce a well-formed digit in every run.
- I malforms 3 of 72 clean cases (texts like `5->->->...` and `<<<<<<<<<<->->`) in every condition: 36 of 864 primary runs. In 5 more clean cases its answer is a well-formed wrong digit although its decoded trace is accepted. Those 8 cases are the whole of I's clean damage, and they recur under noise.
- N malforms 9 of 72 clean and 109 of 864 primary answers.
- Accuracy on well-formed answers, primary: P 0.494, C 0.461, I 0.926, N 0.571. Among them, the answer equals pi's final decode in 0.751, 0.692, 0.926 and 0.853 of runs.

**By family, primary condition** (24 cases each). The first block is accuracy; the second is trace acceptance.

| arm | `sink` | `permutation` | `long_cycle` | trace: `sink` | trace: `permutation` | trace: `long_cycle` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.330 | 0.153 | 0.201 | 0.014 | 0.003 |
| C | 1.000 | 0.257 | 0.125 | 0.215 | 0.118 | 0.135 |
| I | 0.958 | 0.913 | 0.792 | 0.997 | 1.000 | 1.000 |
| N | 1.000 | 0.361 | 0.135 | 0.208 | 0.014 | 0.003 |

Clean accuracy by family: I 23/24 `sink`, 22/24 `permutation`, 19/24 `long_cycle`; N 24, 16 and 12; P and C 24 each.

**By noise loop, primary condition:**

| arm | j = 4 | j = 8 | j = 12 |
| --- | ---: | ---: | ---: |
| P | 0.500 | 0.483 | 0.500 |
| C | 0.455 | 0.444 | 0.483 |
| I | 0.889 | 0.889 | 0.885 |
| N | 0.486 | 0.469 | 0.542 |

**Agreement with P under the primary condition:**
- An arm's decoded trace equals P's in 101 (C), 63 (I) and 41 (N) of 864 runs.
- Whether its answer is correct matches P's in 709, 482 and 726.
- Its primary answer equals its own clean answer in 46.1% (C), 99.1% (I) and 58.2% (N) of runs; P 49.4%.

**Exploratory differences** (case bootstrap as in R1-R3; not registered tests):

| condition | I - P accuracy | I - N accuracy | N - P accuracy | I - P trace |
| --- | --- | --- | --- | --- |
| primary | +0.394 [+0.289, +0.493] | +0.389 [+0.280, +0.494] | +0.005 [-0.035, +0.043] | +0.926 [+0.895, +0.954] |
| sigma\*/2 | +0.394 [+0.282, +0.505] | +0.269 [+0.162, +0.380] | +0.125 [+0.051, +0.208] | +0.921 [+0.884, +0.954] |
| 2 sigma\* | +0.366 [+0.259, +0.477] | +0.338 [+0.231, +0.449] | +0.028 [-0.028, +0.083] | +0.884 [+0.833, +0.931] |
| clean | -0.111 [-0.194, -0.042] | +0.167 [+0.055, +0.278] | -0.278 [-0.389, -0.181] | 0 |

## Training

Recipe as in H-A3 and H-A3b: 600 steps of 8 generated cases per arm, j uniform in 1..7, 25% of cases without noise, Adam at lr 1e-3. The batches, loops and noise are identical across arms and identical to H-A3b's.

| arm | parameters | mean loss per 100 steps | steps with loss < 0.001 | max \|w\| | column norms of W |
| --- | ---: | --- | ---: | ---: | --- |
| C | 27,648 | 14.8, 15.3, 15.2, 16.3, 15.1, 14.8 | 0 | 0.159 | 2.39-4.21 |
| I | 55,296 | 13.0, 8.2, 3.7, 3.2, 3.0, 2.4 | 199 | 0.099 | 1.21-2.10 |
| N | 27,648 | 12.9, 12.3, 10.1, 9.6, 8.3, 8.3 | 0 | 0.180 | 2.02-2.34 |

I's median loss over its last 100 steps is 0.000. Its means stay above zero because of occasional batches near 5. In H-A3b, with the bias, C's loss rose to 27-29, I's stayed at 22-26 and N's fell to 7.

## Limits

- **One F, one sigma\*, one training recipe.** F is D1 (GPT-2 small, 6-node maps, one seed), and sigma\* is at the top of the registered grid.
- **The noise is relative to each run's own slot RMS.** That was the registered definition from H-A3 on, and it makes the absolute corruption depend on what an arm did before loop j. [H-A3d](../../../EXPERIMENT.md#h-a3d-h-a3c-with-one-fixed-noise-scale-for-every-arm-registered-2026-11-07-after-h-a3cs-evaluation-was-read-before-any-h-a3d-run) repeats H-A3c with one fixed noise scale for every arm.
- **pi reads slot 2 only.** I's trace is right in 99.9% of runs while its answer is wrong in 11%, because the answer is generated from all six slots.
- **The mechanism section is post-hoc,** on 12 generated cases outside the roster and one noise loop (j = 4).

## Files

| file | content |
| --- | --- |
| [manifest.json](manifest.json) | code commit, weights and roster hashes, corrector hashes (weights kept outside git), arguments, kappa, environment, hardware, commands, receipt hashes, the restart note |
| [calibrate.json](calibrate.json) | H-A3's sigma\* calibration, reused |
| [scale.json](scale.json) | H-A3b's per-case clean slot RMS and kappa, reused |
| [train.json](train.json) | per-step loss for C, I and N, and parameter counts |
| [eval.json](eval.json) | summary (accuracy, trace acceptance, by j, by family, decisions with bootstrap intervals, clean damage) and all 5,472 run records (answer, prediction, decoded trace, exact-check verdict). The generated text is in the run directory's copy only. |
