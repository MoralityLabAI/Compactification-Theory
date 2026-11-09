# RRC-06: raw coefficients to algebraic intermediate traces

**Executed:** 2026-11-09; single CPU-only training seed, frozen evaluation. Twelve new unit tests and source/checkpoint-bound replay passed locally. No GPU or pretrained language-model run. The result is a **negative complete-program result** with measurable partial intermediate learning and a verified first-fault localization rule.

## Experiment

The student receives exactly the 65 coefficient counts \(f_0,\ldots,f_{64}\), transformed by a fixed public log1p normalization and degree index. It predicts both Euler exponents \(a_n\) and the coefficients of the logarithmic derivative \(b_n\). Algebraic teacher values are used only for supervised training and independent evaluation; the neural forward pass has no exact Euler inverse, divisor graph, source-program labels or verifier feedback.

Training used 720 unique periodic product prefixes from periods 2,3,4,6,8,9,10,12 and multiplicity caps unlimited/1. Development had 120 new same-family cases. A separate 72-case unseen-period test uses moduli 5,7,11,13,17,19. The unseen-cap split has 36 cap-2 cases. Local gap and new nonperiodic part-set families were fully withheld from training. Five observationally identical base/future-mutant pairs provide finite-prefix ambiguity controls. Two local-gap test prefixes also match training prefixes; this incidental overlap is recorded and must be excluded from any claim of strict raw-input disjointness.

## Full-program and certificate results

| Split | Exact bounded synthesizer | Recurrent (2 passes) | Feedforward | Shuffled-target recurrent |
| --- | ---: | ---: | ---: | ---: |
| Development, 120 | 120 | 0 | 0 | 0 |
| Unseen periods, 72 | **72** | 0 | 0 | 0 |
| Unseen multiplicity cap, 36 | **36** | 0 | 0 | 0 |
| Local-gap generators, 40 | 3 | 0 | 0 | 0 |
| Nonperiodic part sets, 15 | 0 | 0 | 0 | 0 |
| Out-of-grammar periods, 12 | 0 | 0 | 0 | 0 |
| Future collision cases, 10 | 10 | 0 | 0 | 0 |

These are **accepted executable programs** whose exact Euler residuals vanish on all observed degrees and whose predicted program independently reproduces the complete observed coefficient vector. For every ordinary case accepted by the exact synthesizer, all withheld coefficients through degree 128 agree. Five exact-synthesizer collision acceptances are wrong about the unseen future, as required by the paired ambiguity construction. The neural models emit no accepted program at their registered trained depths and therefore make no such false positive claims.

The exact solver reconstructs both known Rogers–Ramanujan rules: unlimited parts congruent to \(\{1,4\}\pmod 5\) or \(\{2,3\}\pmod 5\). Neither neural model recovers a complete program for those cases.

## Intermediate-level accuracy on 72 unseen-period cases

| Measure | Two-scan GRU | Full-prefix feedforward | Shuffled-target GRU |
| --- | ---: | ---: | ---: |
| Parameters | 2,028 | 2,084 | 2,028 |
| Euler coordinates correct out of 4,608 | 2,838 (61.6%) | **2,919 (63.3%)** | 1,968 (42.7%) |
| Integer log-derivative coordinates correct out of 4,608 | **314 (6.8%)** | 227 (4.9%) | 117 (2.5%) |
| Average first incorrect Euler exponent | **5.86** | 3.49 | 1.79 |
| Fully verified Euler traces | 0 | 0 | 0 |
| Fully verified log-derivative traces | 0 | 0 | 0 |

Both learned models beat the shuffled-target control on intermediate accuracy. Their advantages differ by metric: the feedforward student has more correct Euler coordinates in aggregate, while the recurrent student pushes its first incorrect Euler exponent later and gets more rounded log-derivative coordinates correct. Neither full-trace accuracy nor complete program recovery has been achieved. The decoder's integer rounding is strict; modest floating-point improvements do not count as exact arithmetic.

## Frozen depth ablation on unseen moduli

| GRU scans | Correct Euler coordinates /4,608 | Accepted programs /72 |
| ---: | ---: | ---: |
| 0 (zero-state reference) | 2,423 | 0 |
| 1 | 2,846 | **1** |
| 2 (trained) | 2,838 | 0 |
| 4 | 2,838 | 0 |
| 8 | 2,835 | 0 |

The single accepted case at one scan is a depth-ablation observation, measured after training at two scans. It does not show monotone refinement. More recurrent passes after two scans fail to improve exact completeness, and the parameter-free exact symbolic solver remains vastly stronger. The two-scan GRU requires approximately 245,180 dense MACs/case; the feedforward control takes 1,820, a roughly 135-fold analytic difference, excluding nonlinearities and verification.

## What RRC-06 establishes

The model now starts from the raw coefficient prefix and produces numerical estimates of the two algebraic intermediate vectors without receiving the exact log derivative or prewired divisor graph. This resolves RRC-05's input-contamination concern at the *inference interface*. It also reveals that the hard part is maintaining exact consistency across all 64 coordinates. The independent checker supplies a first-fault certificate that turns a vague failed-program score into a localized algebraic error.

The RRC-05 two-scalar circuit recovered every unseen-period program after it was handed the exact \(b_n\) and divisor graph. The new unaided GRU succeeds on no complete unseen-period program at its trained depth. That comparison changes the availability of mathematical structure and model capacity, so it is **an ablation of supplied information**, not an architecture-controlled capability regression. The original RRC-04 full-program student also had zero successful held-out direct programs, consistent with the present difficulty.

## Reproduce

From the repository root, using Python 3.10+ and **no Python dependencies** for replay:

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_rrc06.py' -v
python research/rrc_compactification/bench06.py --check
```

Optional deterministic CPU PyTorch training (2.10.0+cpu in the registered run):

```sh
python research/rrc_compactification/train06.py --steps 500 --batch-size 64
python research/rrc_compactification/bench06.py --check
```

Frozen weights are portable JSON rather than binary pickles. The trainer checks PyTorch versus standard-library inference on three development cases for each model with maximum absolute logit discrepancy at most 5e-5. A clean rerun must preserve all source hashes, roster hashes and every inference trace fingerprint. The receipt reports metrics and selected per-case first-fault evidence but omits raw full output vectors to keep the source tree small.

## Next falsifiable experiment: RRC-07

Move from per-coordinate classification to an **algorithmic intermediate state** trained with a symbolic residual curriculum. Train a model to propose one correction at a time, where its own previous estimates are fed back; permit exact residuals **only during training and acceptance**. Register a first-fault advancement metric and a train/future-compute matched feedforward unroll before running on a new source-family split. Compare against a hard-coded truncated series solver of equal recurrence depth, and require exact recovery to improve before assigning value to learned iteration. The RRC-06 held-out roster is exposed by this report and becomes development-only for subsequent architectural decisions.
