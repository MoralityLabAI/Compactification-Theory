# Algernon-03g result (H-A3g): at floor-level slot erasure (`sigma* = 16`), the answer loss still cannot repair the state

Run 2026-11-12 10:21 to 2026-11-12 12:19 on the local RTX 5080, under
[EXPERIMENT.md, H-A3g](../../../EXPERIMENT.md#h-a3g-h-a3f-at-sigma--16-completing-the-noise-by-loss-grid-registered-2026-11-12-after-h-a3fs-evaluation-was-read-before-any-h-a3g-run),
registered and pushed (`845c38a`) after H-A3f's evaluation was read and before any H-A3g run.
- H-A3g completes the fourth cell of `{sigma = 1, 16} x {local loss, answer loss}`:
  - Relative to H-A3d (`sigma* = 16`, local loss): the training loss adds the answer digit's cross-entropy (`--loss answer`), so gradient reaches all six slots.
  - Relative to H-A3f (`sigma_e = 1`, answer loss): `--sigma-rule half` selects `sigma* = 16.0` (`kappa = 110.78`), with fixed noise std `16.0 x 6.92` per slot element.
- F is 02D's arm D1 (`80eda0824cff`), frozen. pi is D1's LM head on slot 2. The correctors are bias-free (`--corrector-bias off`), output in noise units (`--out-scale noise`), with fixed noise reference `r = 6.92` (`--noise-ref fixed`).
- Code: `845c38a` for training and evaluation. P was evaluated again (`1,368` of `1,368` records identical to H-A3d's).
- The run completed cleanly under a 4 GiB `vramorama` lease with no crash or restart.

**The registered decisions: R1, R2, R3 and R4 are all not met.** The instrument is valid: P answers all 72 clean cases, and the exact checker accepts every clean decoded trace. **C keeps clean runs intact (1.000). Both I (0.764) and N (0.528) fall below the 0.97 clean floor.**

| arm | clean accuracy | primary accuracy | difference [95% case bootstrap] | rule |
| --- | ---: | ---: | --- | --- |
| P (plain) | 1.000 | 0.500 | C - P = -0.019 [-0.061, +0.024] | R1 needs >= +0.10: not met |
| I (same information) | 0.764 | 0.528 | C - I = -0.046 [-0.094, -0.002] | R2 needs >= +0.05: not met |
| N (decode only) | 0.528 | 0.505 | C - N = -0.023 [-0.084, +0.032] | R3 needs >= +0.10: not met |
| C (constraint residual) | 1.000 | 0.481 | | |
| | | | I - P = +0.028 [-0.014, +0.071] | R4 needs >= +0.10: not met |

The primary condition is `sigma* = 16`, with noise after loop j in `{4, 8, 12}` and four noise seeds: 864 runs per arm.

**The registered predictions** (three of four held; C's point estimate stayed at P rather than rising slightly above H-A3d's 0.487):

| prediction | outcome |
| --- | --- |
| P's records equal H-A3d's (primary 0.500, 1,368 / 1,368 identical) | **held:** 1,368 of 1,368 records identical |
| C keeps clean runs intact (>= 0.97) | **held:** 72/72 (1.000), every clean trace accepted |
| C rises above H-A3d's 0.487, and R1 is not met | **R1 not met held** (-0.019); C did not rise above 0.487 (0.481) |
| R2, R3 and R4 are not met, and I falls below the 0.97 clean floor | **held:** all three rules not met, and I clean accuracy is 0.764 (17 clean misses) |

## Answer

- **The registered branch for R1 not met applies.** Residual slot repair under `--loss answer` requires partially surviving slot state (`sigma_e = 1`, H-A3f: 0.811 vs 0.626) and fails once corruption reaches the floor (`sigma* = 16`, here: 0.481 vs 0.500).
- **The 2x2 Phase 4 grid (`{sigma = 1, 16} x {local, answer}`) is now closed:**

| corruption | local loss (slot 2 only) | answer loss (local + answer CE) |
| --- | --- | --- |
| `sigma* = 16` (floor, P = 0.500) | H-A3d: C 0.487, I 0.455 (R1-R4 not met) | **H-A3g:** C 0.481, I 0.528 (R1-R4 not met; I clean 0.764) |
| `sigma_e = 1` (midpoint, P = 0.626) | H-A3e: C 0.639, **I 0.804** (**R4 met**; C trace 0.987) | H-A3f: **C 0.811**, I 0.714 (**R1, R2, R3 met**; C clean 1.000) |

- **H-A3d's floor-level null was not caused by the local loss's reach.**
  - Under `--loss answer`, gradient reaches slots 3-5 in all three trained correctors: weight norms on slots 3-5 are 2.57 (C), 3.55 (I) and 2.77 (N), versus identically 0 in H-A3d and H-A3e.
  - Even with nonzero slot 3-5 weights and `e_j = p_{j-1} M` supplying the true state from the last clean loop, a rank-6 linear write scaled by `kappa = 110.78` cannot cancel an isotropic 4,608-dimensional Gaussian perturbation of std `110.78` per element across all six slots.
  - C's training loss never falls across the 600 steps (100-step means: 19.21, 24.54, 25.48, 26.77, 26.60, 26.55).
- **Amplifying `G([e, p])` by `kappa = 110.78` under the answer loss severely damages clean runs for I and N.**
  - C stays at 1.000 on clean cases because `e - p` vanishes on clean D1 loops.
  - I drops from 0.944 clean in H-A3f (`kappa = 6.92`) and 1.000 in H-A3d (`local` loss) to 0.764 here, missing 17 of 72 clean answers (6 `permutation`, 11 `long_cycle`) even though 70 of 72 clean decoded traces remain valid (0.972).
  - N drops to 0.528 clean accuracy (34 misses: 12 `permutation`, 22 `long_cycle`) and 0.347 clean trace acceptance.

## Exploratory: why C loses its H-A3d trace gain under the answer loss

This section is a post-hoc reading. None of it is a registered test.

- **Under `--loss answer` at `sigma* = 16`, C's trace acceptance drops below P's:** 0.034 (29/864) vs P's 0.075 (65/864), `C - P = -0.042 [-0.065, -0.021]`. In H-A3d (`local` loss at `sigma* = 16`), C's trace acceptance was 0.156 (`+0.081` over P).
  - In H-A3d, C wrote zero to slots 3-5 and concentrated its gradient on slot 2 (`pi`'s readout). Here the answer loss drives nonzero writes into slots 3-5 (`norm 2.57`, exceeding slot 2's `1.92`) from noisy `e - p` inputs after loop `j`, perturbing the downstream attention keys/values on loops `j + 1 .. 16` without recovering the answer.
  - Where C's trace does stay valid (29 primary runs), its answer is right in 0.966 (28/29), against 0.86 in H-A3d; on invalid traces it scores 0.465 (P: 0.459).
- **I retains a small trace advantage over P (`0.105` vs `0.075`, `+0.030 [+0.019, +0.042]`),** with right answers on 0.923 of its 91 valid-trace runs, at the cost of 17 clean answer failures where its slot 3-5 pushes (`norm 3.55`) corrupt answer readout despite valid slot-2 traces.

## Evaluation

**Answer accuracy** (72 cases; primary 864 runs per arm, `sigma*/2 = 8` and `2 sigma* = 32` 216 each):

| arm | clean | primary (`sigma* = 16`) | `sigma*/2 = 8` | `2 sigma* = 32` |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.500 | 0.505 | 0.477 |
| C | 1.000 | 0.481 | 0.495 | 0.463 |
| I | 0.764 | 0.528 | 0.528 | 0.537 |
| N | 0.528 | 0.505 | 0.500 | 0.491 |

Every arm's generated answer is a well-formed digit in every run.

**Whole-trace acceptance by the exact BN254 R1CS checker** (the 16-step pi-decoded trace against the public machine):

| arm | clean | primary | `sigma*/2` | `2 sigma*` |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.075 | 0.079 | 0.037 |
| C | 1.000 | 0.034 | 0.028 | 0.000 |
| I | 0.972 | 0.105 | 0.139 | 0.051 |
| N | 0.347 | 0.001 | 0.000 | 0.000 |

**Answer = pi's final decode:**

| arm | clean | primary | `sigma*/2` | `2 sigma*` |
| --- | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.740 | 0.759 | 0.727 |
| C | 1.000 | 0.547 | 0.556 | 0.574 |
| I | 0.750 | 0.696 | 0.671 | 0.662 |
| N | 0.611 | 0.495 | 0.491 | 0.500 |

**By family, primary condition** (24 cases each). The first block is accuracy; the second is trace acceptance.

| arm | `sink` | `permutation` | `long_cycle` | trace: `sink` | trace: `permutation` | trace: `long_cycle` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P | 1.000 | 0.351 | 0.149 | 0.201 | 0.021 | 0.003 |
| C | 1.000 | 0.281 | 0.163 | 0.083 | 0.007 | 0.010 |
| I | 1.000 | 0.438 | 0.146 | 0.233 | 0.059 | 0.024 |
| N | 1.000 | 0.413 | 0.101 | 0.003 | 0.000 | 0.000 |

**By noise loop, primary condition:**

| arm | j = 4 | j = 8 | j = 12 |
| --- | ---: | ---: | ---: |
| P | 0.503 | 0.500 | 0.497 |
| C | 0.483 | 0.479 | 0.483 |
| I | 0.521 | 0.531 | 0.531 |
| N | 0.493 | 0.503 | 0.517 |

**Other case-bootstrap differences, primary** (post hoc):
- I - N = +0.023 [-0.025, +0.071].
- N - P = +0.005 [-0.052, +0.064].
- C - P on trace acceptance = -0.042 [-0.065, -0.021].
- I - P on trace acceptance = +0.030 [+0.019, +0.042].

**Agreement with P under the primary condition:**
- An arm's decoded trace equals P's in 87 (C), 105 (I) and 0 (N) of 864 runs.
- Whether its answer is correct matches P's in 690, 690 and 702.

## Training

The recipe is H-A3d's with `--loss answer` added (equivalently H-A3f's at `sigma* = 16`): 600 steps of 8 generated cases per arm, j uniform in 1..7, 25% of cases without noise, Adam at lr 1e-3.

| arm | parameters | mean loss per 100 steps | median loss, last 100 steps | steps below 0.001 | max \|w\| | weight norm on slots 0 / 1 / 2 / 3-5 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| C | 27,648 | 19.21, 24.54, 25.48, 26.77, 26.60, 26.55 | 27.335 | 0 | 0.107 | 1.99 / 1.77 / 1.92 / 2.57 |
| I | 55,296 | 27.03, 29.42, 26.47, 23.77, 22.27, 20.78 | 21.485 | 0 | 0.100 | 1.14 / 1.31 / 1.17 / 3.55 |
| N | 27,648 | 19.02, 17.73, 16.77, 16.80, 15.99, 15.67 | 15.626 | 0 | 0.113 | 1.39 / 1.15 / 0.93 / 2.77 |

## Limits

- **One F, one linear corrector architecture.** A nonlinear or higher-rank state-replacement module could in principle overwrite all six slots from `e_j = p_{j-1} M` by zeroing the corrupted state first (`s <- 0 + G(e_j)`), whereas additive residual correction `s + kappa G(e_j - p_j)` leaves the 4,602 orthogonal noise dimensions intact. That tests the additive linear residual family registered for Phase 4.
- **Intermediate noise levels between `sigma = 2` (2 sigma_e, where C still beat P by +0.181 in H-A3f) and `sigma = 8` (`sigma*/2`, where C is at 0.495 here)** bound the breakdown transition of additive residual repair somewhere in `sigma in (2, 8)`.

## Files

| file | content |
| --- | --- |
| [manifest.json](manifest.json) | code commit, weights and roster hashes, corrector hashes (weights kept outside git), arguments, kappa, environment, hardware, commands, receipt hashes |
| [calibrate.json](calibrate.json) | H-A3's sigma calibration, reused (`sigma* = 16.0`) |
| [scale.json](scale.json) | H-A3b's per-case clean slot RMS on the 48 calibration cases (`r = 6.9235`) |
| [train.json](train.json) | per-step loss for C, I and N, and parameter counts |
| [eval.json](eval.json) | summary and all 5,472 run records (generated text dropped in repo copy; full copy SHA-256 in `manifest.json`) |
