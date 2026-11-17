# SCM-09 CPU reference and regression results

**Date:** 2026-11-13. **Source:** local `scm09_reference.py`, synthetic deterministic dyadic weights interpreted as real constants, with exact rational bounds and no trained RMP model weights. The interval evaluator is independent from the 90-digit high-precision regression evaluator. Existing checkpoint and dictionary files reside outside this environment.

| Synthetic RMP graph | Visits | SCM-08 anchor radius at last visit | SCM-09 centered radius at last visit | Improvement factor |
| --- | ---: | ---: | ---: | ---: |
| tied, 8 hidden, 8 tokens | 8 | 0.4591313 | 1.12710e-15 | 4.074e14 |
| untied + reinjection, 8 hidden, 8 tokens | 8 | 0.4559135 | 7.30088e-15 | 6.245e13 |
| tied, 16 hidden, 12 tokens | 8 | 2.4100781 | 6.47780e-15 | 3.721e14 |

All eight visit-specific ideal-real null-braid predicates pass for the deterministic example chart and registered *synthetic* radius `1e-6` times the source upper bound on injection-token RMS.

The exact JSON receipt is `results/scm09_cpu_reference.json`. Reproduce with:

    python research/seifert_activation/scm09_reference.py --out results/scm09_cpu_reference.json --check
    python -m unittest discover -s research/seifert_activation/tests -q

`scm09_audit.py` is a separate read-only bridge to externally stored RMP checkpoints and calibration examples. The original trained weights, actual model behavior under cyclic interventions, and byte-level floating point roundoff are outside this experiment's evidence. Even a source-certified ideal-real null braid is not a proof about a recorded IEEE-754 trajectory.

## Executed regression receipt

Full local command: `python -m unittest discover -s tests -q` within the isolated research track. **190/190 tests passed** (161 inherited and 29 added). The new tests use an independent 90-digit `mpmath` implementation of the RMP real-valued equations, verify certified inclusion on tied and untied cells, and enforce source/geometry refusals. The dedicated SCM-09 CI workflow repeats the new tests and regenerates the JSON fixture.

Synthetic reference SHA-256: `e49e0f53f833da672826c2069edd99ff162b8da63d5ae67b83b64e110f9e9e12`.

The bounded reference supports 16 tokens and 32 hidden dimensions by default. Production RMP's 64-token, 128-dimensional checkpoints are outside the default verified execution budget; the analysis remains an exact-real proof pilot.
