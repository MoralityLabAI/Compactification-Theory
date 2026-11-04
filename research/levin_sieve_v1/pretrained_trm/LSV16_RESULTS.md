# LSV-16: Source-locked ambient-to-goal robustness results

**Date:** 2026-11-04. **Evidence:** analytic LayerNorm perturbation inequalities, outward-rounded Arb transport through the frozen pretrained 44,001-parameter TRM-inspired Sudoku recurrence, and independently replayed rational comparisons against the archived LSV-15 margins. All results concern exact-real evaluation of the binary-float weights starting at the frozen, binary32 stage-one hidden state. The original source revision, checkpoint fingerprint, 64-puzzle roster, and selected first-missing-cell goal readouts remain fixed.

## Measured certificate outcomes

| Initial stage-one ambient infinity radius | Evaluated cases | Top digit provably unchanged | Top two digits provably unchanged | Denominator bounds failed |
|---|---:|---:|---:|---:|
| 1e-8 | 4 (negative-control smoke) | 0 | 0 | 4 |
| 1e-10 | 4 (negative-control smoke) | 0 | 0 | 0 |
| **1e-14** | **64** | **64** | **57** | **0** |
| **1e-16** | **64** | **64** | **64** | **0** |

At 1e-14, the verified upper bounds on the *centered* final pre-LayerNorm perturbation have minimum approximately 0.01152645, median 0.01335986, and maximum 0.01549817 across 64 cases. At 1e-16, the corresponding range is approximately 0.0001152604 to 0.0001549742. Each case passes only when its bound is strictly smaller than its independently archived exact-rational LSV-15 goal threshold.

The 1e-10 smoke produces very loose bounds (about 212 to 301 at the final goal bus); the 1e-8 smoke cannot establish a positive intermediate LayerNorm denominator lower bound. These are **method failures, not observed adversarial prediction changes**.

## Mathematical statement

For each certified case and its stated radius, every independent perturbation to **all 3,888 coordinates** of the frozen first-recurrence hidden state, with absolute amplitude below or equal to the stated infinity radius, leaves the designated output argmax unchanged after three further exact-real recurrent passes. In the top-two certified cases, it also leaves the specified top-two ordering unchanged. No model target label enters the certificate. The statement concerns invariant predictions, which can include incorrect Sudoku answers.

A proof proceeds by composition of the globally valid `5/4`-Lipschitz GELU inequality, absolute-weight affine error propagation, and a centered local-variance LayerNorm bound. The latter removes the common-mode component algebraically before estimating the induced error. The full derivation and its trust conditions are in [AMBIENT_TO_GAUGE.md](AMBIENT_TO_GAUGE.md).

This is an analytically distinct result from LSV-14's rank-four neighborhood theorem. LSV-14 proves that four smooth goal-history coordinates remain locally necessary at 1e-14, while LSV-16 proves that the *final discrete predicted digit* remains stable at that radius, under the stated exact-real computation.

## Permanent reproducibility receipts

- [Complete 64-case positive-radius receipt](results/lsv16_ambient_64_20261104.json), original JSON SHA-256 `d0b652791c0b21f1f975e18870b1867c12add008b7434f5178a99355c0be9f7f`.
- [Four-case larger-radius negative controls](results/lsv16_ambient_negative_controls_20261104.json), original JSON SHA-256 `7b34fc8e46921cac4d4852d22528d518c286f57d97132e528accb2f175dd964e`.
- [Independent exact-rational checker](verify_transport_receipt.py) validates source and roster identities, saved thresholds, strict inequalities and aggregate claims. It rejects substituted thresholds, corrupted source hashes, and altered winner flags. The intermediate Arb computation remains a trusted component, reproduced from the locked source and checkpoint by [run_transport_gauge.py](run_transport_gauge.py).
- [Successful complete CI run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41835777809): 12 unit tests, both model runs and exact replay passed; 64-case scan wall time 80.263 CPU seconds after setup. The original run artifact is preserved at [GitHub Actions artifact 13326930123](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41835777809/artifacts/13326930123).

For offline audit after checkout:

```sh
python research/levin_sieve_v1/pretrained_trm/verify_transport_receipt.py research/levin_sieve_v1/pretrained_trm/results/lsv16_ambient_64_20261104.json --complete
python research/levin_sieve_v1/pretrained_trm/verify_transport_receipt.py research/levin_sieve_v1/pretrained_trm/results/lsv16_ambient_negative_controls_20261104.json
```

## Next numerical frontier

The huge gap between the observed small input-output gradient and the composed componentwise radius bound indicates where further research can pay off: *goal-directed adjoint transport with a verified second-order remainder*. A future LSV-17 should compute a validated output-margin gradient at the clean stage-one state, then bound the gradient variation across the whole perturbation box. A Taylor estimate with a checked Hessian budget could retain cancellations across the token mixer and extend certificates toward the originally targeted 1e-8 scale. Any such certificate must account for the full 3,888-dimensional input and the exact-real model semantics; sampled gradients and unverified Hessian norms are only diagnostics.

## Reproducible anchor-tolerant v2 proof

A subsequent independent GitHub Actions runner produced binary32 stage-one activations whose final centered inputs did not overlap the original extremely narrow Arb anchor intervals. The strict-anchor implementation rejected this as it should. This is a practical reproducibility limitation: a checkpoint SHA256 and frozen puzzle roster do not guarantee bitwise equality of every intermediate CPU tensor.

The corrected [anchor-transfer theorem](ANCHOR_DRIFT.md) independently encloses `D=||u_new-u_reference||_infinity` and adds that value to the stage-one perturbation transport bound `E` before comparing with the archived LSV-15 strict output threshold. Across all 64 cases, `D` was between **1.8166e-7 and 5.1466e-6**, with median about **9.8853e-7**. Both the old and new result formats are accepted by the independently tested rational replay script; the v2 format further enforces `reported_bound == transported_bound + archived_reference_drift_bound`.

On [passing CI run 41837046193](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41837046193), the anchor-aware remeasurement certified **64/64 top digits and 57/64 top-two orderings at stage-one radius 1e-14**, and **64/64 top digits and 64/64 top-two at radius 1e-16**. All twelve earlier tests plus the new anchor accounting test passed. Larger-radius negative controls remained inconclusive. The v2 source results remain governed by exact-real recurrence from the newly measured binary32 stage-one tensor, while the archived LSV-15 reference margin is reused through a verified drift bound.

- [Anchor-aware complete 64-case receipt](results/lsv16_ambient_64_anchor_v2_20261104.json), JSON SHA-256 `2c4a8dfb48b3bc3b4038d8397be37bde4f3cf480ea86d1636015656998332948`.
- [Anchor-aware four-case negative controls](results/lsv16_ambient_negative_controls_anchor_v2_20261104.json), JSON SHA-256 `9d22839b13e5d0527108178942ab72d45e554e6656fc0f2770df370bd27a0551`.
- [Passing source-controlled checkpoint execution and rational replay](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41837046193), GitHub artifact `13325788206`.

The original v1 proof receipts are kept unchanged as the historical run-specific observations. Their independent rational checks remain valid for those source-generated enclosures, while v2 directly addresses reproducibility of differently realized CPU anchors.
