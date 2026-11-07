# Algernon-03b result (H-A3b): correctors in noise units reach the state but do not repair it, and C, I and N all damage clean runs

Run 2026-11-07 04:04 to 20:12 on the local RTX 5080, under
[EXPERIMENT.md, H-A3b](../../../EXPERIMENT.md#h-a3b-h-a3-with-the-correctors-output-in-units-of-the-noise-registered-2026-11-07-before-h-a3s-evaluation-was-read),
registered and pushed (`374ed2a`) before H-A3's evaluation was read. The evaluation lease waited from 06:58 to about 18:57 while another job held the card.
- H-A3b is H-A3 with one change: every corrector's output is multiplied by kappa = sigma\* x mean clean slot RMS = 16 x 6.92 = 110.78.
- F is 02D's arm D1 (`80eda0824cff`), frozen. pi is D1's own LM head on slot 2.
- Code: `29be916` for scale and training. Evaluation ran `1ce6afd`, which only adds `--corrector-bias` with default `on`, so H-A3b's code path is unchanged. P's 1,368 records are identical to H-A3's.

**The registered decisions: R1, R2 and R3 are all not met.** The instrument is valid: P answers all 72 clean cases, and the exact checker accepts every clean decoded trace. **C, I and N all fall below the 0.97 clean floor** and are reported as damaging clean runs.

| arm | clean accuracy | primary accuracy | difference vs C [95% case bootstrap] | rule |
| --- | ---: | ---: | --- | --- |
| P (plain) | 1.000 | 0.494 | C - P = +0.032 [-0.028, +0.095] | R1 needs >= +0.10: not met |
| I (same information) | 0.556 | 0.528 | C - I = -0.001 [-0.073, +0.073] | R2 needs >= +0.05: not met |
| N (decode only) | 0.681 | 0.501 | C - N = +0.025 [-0.084, +0.141] | R3 needs >= +0.10: not met |
| C (constraint residual) | 0.556 | 0.527 | | |

The primary condition is sigma\* = 16, noise after loop j in {4, 8, 12}, four noise seeds: 864 runs per arm.

## Answer

- **The rescaled correctors act on the state, but none repairs it through the constraint.** Under this recipe, an additive linear corrector with a bias, its output in noise units, does not use the transition residual to repair D1's slot state after it is wiped at sigma\*.
- **C's small gain over P is noise-insensitivity, not repair.**
  - C's accuracy is 0.52-0.56 in every condition, clean included.
  - Its answer under the primary condition equals its own clean answer in 85.0% of runs (P: 49.4%).
  - pi's final decode under C is the digit 2 in 69 of 72 clean runs.
  - This is the collapse the training diagnostic predicted
    ([EXPERIMENT.md](../../../EXPERIMENT.md#h-a3b-training-diagnostic-recorded-2026-11-07-after-c-finished-training-before-any-h-a3b-evaluation)). C's input e - p is near zero on clean loops, so its bias, of norm 1.79, adds the same push of norm about 198 to the slots after every loop.
- **The bias is the defect, and it only bites C.**
  - p and e are distributions, so they sum to 1. N's and I's biases fold into per-state codes (W p + b = (W + b 1ᵀ) p).
  - C's input e - p sums to 0, so its bias cannot fold in and stays a constant.
  - [H-A3c](../../../EXPERIMENT.md#h-a3c-h-a3b-with-bias-free-correctors-registered-2026-11-07-before-h-a3bs-evaluation-ran), registered before this evaluation ran, removes the bias and is training now.

## Exploratory: the decode-only arm cleans the decoded trace and breaks the answer text

This section is a post-hoc reading. None of it is a registered test.

- **N re-encodes its own decode.**
  - N writes G(p) = sum over states i of p_i (W_i + b): a learned slot code for the state pi currently reads.
  - Its training loss fell from 11.9 to 7.0. C's rose from 17.3 to 27.3 and I's from 17.6 to 23.6.
- **N's pi-decoded traces satisfy the exact checker far more often than P's.**
  - Primary: 0.399 vs 0.073, N - P = +0.326 [+0.287, +0.367] (same case bootstrap).
  - sigma\*/2: 0.759 vs 0.079, N - P = +0.681 [+0.620, +0.741].
  - Clean: 1.000.
  - On the bijective `long_cycle` maps, primary acceptance is 0.274 against P's 0.003.
- **N damages the generated answer's format, not its content.**
  - About 30% of N's generated answers are not a single digit: for example `0 3`, `4->` or `0->->->->`. That is 23 of 72 clean runs and 255 of 864 primary runs.
  - Where the answer is a well-formed digit, it equals pi's final decode in 99.5-100% of runs.
  - Accuracy on well-formed answers: clean 1.000, sigma\*/2 0.914, primary 0.711, 2 sigma\* 0.545 (P: 1.000, 0.495, 0.494, 0.472).
  - The malformed share depends on the true final state: 65 of 96 primary runs whose answer is 4 are malformed, against 20 of 156 whose answer is 2.
- **N is a cleanup memory, so the gain needs no map.** Re-encoding the decoded state works whenever the decode right after the noise is still correct, which is more often at sigma\*/2.
  - This is the comparison R3 was built for: the decode-only arm beats the constraint arm at keeping the trace on the machine.
  - The registered R3 compares answers, not traces, and is not met.

## Evaluation

**Answer accuracy** (72 cases; primary 864 runs per arm, sigma\*/2 and 2 sigma\* 216 each):

| arm | clean | primary (sigma\* = 16) | sigma\*/2 = 8 | 2 sigma\* = 32 |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.494 | 0.495 | 0.472 |
| C | 0.556 | 0.527 | 0.523 | 0.523 |
| I | 0.556 | 0.528 | 0.523 | 0.523 |
| N | 0.681 | 0.501 | 0.644 | 0.361 |

**Whole-trace acceptance by the exact BN254 R1CS checker** (the 16-step pi-decoded trace against the public machine):

| arm | clean | primary | sigma\*/2 | 2 sigma\* |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.073 | 0.079 | 0.037 |
| C | 0.083 | 0.044 | 0.065 | 0.037 |
| I | 0.250 | 0.227 | 0.236 | 0.181 |
| N | 1.000 | 0.399 | 0.759 | 0.111 |

**Answer = pi's final decode,** among well-formed answers:

| arm | clean | primary | sigma\*/2 | 2 sigma\* |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.751 | 0.750 | 0.736 |
| C | 0.111 | 0.128 | 0.125 | 0.116 |
| I | 0.403 | 0.354 | 0.389 | 0.352 |
| N | 1.000 | 0.995 | 1.000 | 1.000 |

Under C, the generated answer no longer follows slot 2, so C's pi-decoded trace and its answer come apart.

**By family, primary condition** (24 cases each). The first block is accuracy; the second is trace acceptance.

| arm | `sink` | `permutation` | `long_cycle` | trace: `sink` | trace: `permutation` | trace: `long_cycle` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.330 | 0.153 | 0.201 | 0.014 | 0.003 |
| C | 1.000 | 0.378 | 0.201 | 0.090 | 0.042 | 0.000 |
| I | 1.000 | 0.396 | 0.188 | 0.469 | 0.212 | 0.000 |
| N | 0.792 | 0.424 | 0.288 | 0.628 | 0.295 | 0.274 |

N loses 5 of the 24 `sink` cases in every condition, all of them to malformed answers.

**By noise loop, primary condition:**

| arm | j = 4 | j = 8 | j = 12 |
| --- | ---: | ---: | ---: |
| P | 0.500 | 0.483 | 0.500 |
| C | 0.542 | 0.528 | 0.510 |
| I | 0.528 | 0.521 | 0.535 |
| N | 0.521 | 0.483 | 0.500 |

**Agreement with P under the primary condition:**
- An arm's decoded trace equals P's in 4 (C), 36 (I) and 74 (N) of 864 runs.
- Whether its answer is correct matches P's in 706, 699 and 598.
- In H-A3 those counts were 162, 93 and 48, and 778, 757 and 763.

## Training

Recipe as in H-A3: 600 steps of 8 generated cases per arm, j uniform in 1..7, 25% of cases without noise, Adam at lr 1e-3. The batches, loops and noise are identical across arms and identical to H-A3's.

| arm | parameters | mean loss per 100 steps | max \|w\| | column norms of W | bias norm | kappa x bias norm |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| C | 32,256 | 17.3, 25.7, 28.3, 29.3, 29.3, 27.3 | 0.084 | 1.33-1.70 | 1.79 | 198 |
| I | 59,904 | 17.6, 25.5, 26.2, 25.9, 22.2, 23.6 | 0.098 | 1.06-1.75 | 1.81 | 200 |
| N | 32,256 | 11.9, 10.9, 8.7, 8.2, 6.6, 7.0 | 0.138 | 1.24-1.87 | 2.16 | 239 |

H-A3's losses stayed at 13-15 for every arm.

## Limits

- **One F, one sigma\*, one training recipe.** F is D1 (GPT-2 small, 6-node maps, one seed), and sigma\* is at the top of the registered grid.
- **The null is about correctors with a bias.** For C that bias is a constant push, which H-A3c removes.
- **pi reads slot 2 only.** The answer is generated from all six slots, which is why N can keep slot 2 right while breaking the answer text.
- **The N findings are exploratory.** They come from reading the records after the run. The 0.97 clean floor, which N fails, is the registered judgement on it.

## Files

| file | content |
| --- | --- |
| [manifest.json](manifest.json) | code commits, weights and roster hashes, corrector hashes (weights kept outside git), arguments, kappa, environment, hardware, commands, receipt hashes |
| [calibrate.json](calibrate.json) | H-A3's sigma\* calibration, reused |
| [scale.json](scale.json) | per-case clean slot RMS on the 48 calibration cases, and kappa |
| [train.json](train.json) | per-step loss for C, I and N, and parameter counts |
| [eval.json](eval.json) | summary (accuracy, trace acceptance, by j, by family, decisions with bootstrap intervals, clean damage) and all 5,472 run records (answer, prediction, decoded trace, exact-check verdict). The generated text is in the run directory's copy only. |
