# SCM-10: Full-shape synthetic CPU research receipt

**Date:** 2026-11-14. **Source:** `scm09_reference.DeterministicRMP` fixed dyadic synthetic model; source-locked architecture of RMP explicit attention. **Hardware:** container CPU. **Arithmetical claim:** conditioned on normal IEEE-754 binary64/BLAS execution of the *verifier* and the correctly interpreted frozen frequency table. All source weights are treated as exact binary rationals. No trained RMP weights or D-shared dictionary were available.

## Full 64-token, 128-hidden, four-head, eight-visit results

| Synthetic case | Visit 1 max Euclidean error | Visit 4 | Visit 8 | Frozen cap gate |
| --- | ---: | ---: | ---: | --- |
| tied, beta=0.10 | 2.18734e-9 | 5.50108e-7 | **6.98376e-4** | <0.01: pass |
| tied, beta=0.25 | 1.30955e-8 | 1.20105e-4 | **38.9105** | <1: fail (inconclusive) |
| untied + embedding reinjection, beta=0.25 | 1.30955e-8 | 1.91978e-6 | **7.34152e-5** | <0.005: pass |

Each maximum covers all 64 token Euclidean balls at that visit. A separate direct PyTorch float64 computation of the same synthetic graph with the frozen source input centers falls inside every reported ball. This is **a diagnostic only**; its runtime numerical error is not independently certified.

The first coordinatewise interval prototype became uninformative after a few full-shape visits. Replacing coordinatewise state propagation by causal tokenwise Euclidean balls and using exact integer Gram norm estimates produces the values above. Tied beta=0.25 is deliberately retained as a falsifying case for usefulness of the current sufficient bounds at later visits.

## Verification and refusal

- 80-digit independent mpmath execution of small tied and untied decoders checks every pre-, middle- and post-visit center/error ball.
- Exp Taylor envelopes, exact-GELU CDF polynomial remainders, exact-dyadic Gram norm bounds, matrix roundoff envelopes, negative/zero arithmetic and softmax simplex sensitivities are regression tested.
- Registered RMP source checks, selection-half monitor-calibration replay, checkpoint/dictionary/chart fingerprints and source-code hashes are preserved by `scm10_audit.py`.
- Full-shape execution was observed on a CPU in approximately 12–16 seconds per synthetic configuration in isolation. Time is diagnostic, not a performance guarantee.

**Scientific status:** useful synthetic numerical error bounds in the full RMP shape; 80-digit independent checks at miniature size; proof conditional on the CPU verifier's IEEE-754/BLAS execution semantics. Float32 PyTorch/CUDA inference, trained RMP clean-state error, nontrivial neural braid topology, neural compactification gains and four-dimensional Seifert-surface embeddings remain untested.

## Reproduction

```bash
python -m unittest discover -s research/seifert_activation/tests -p 'test_scm10_fast.py' -v
python research/seifert_activation/scm10_reference.py --out research/seifert_activation/results/scm10_cpu_reference.json --check
python research/seifert_activation/scm10_audit.py --help
```

The SCM-10 reference checks source hashes, frozen synthetic model identity, verification scope and pass/fail thresholds. Float64 centers are permitted to differ across BLAS vendors within the declared arithmetic contract; the source receipt keeps one host's exact values as evidence. Existing SCM-01 through SCM-09 tests remain immutable. The original registered `LOOP-T` and `FF-U` trained-model files are outside this tool environment.
