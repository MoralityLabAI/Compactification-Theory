# SCM-05 synthetic CPU receipt

**Date:** 2026-11-13. **Scope:** pure synthetic planar curves and mock/reference geometry; no neural activation checkpoints, causal outcomes, neural Jacobian certificates or 4D Seifert surface invariants.

Executable files: `continuous.py`, `quantization.py`, `scm05_audit.py`, `scm05_reference.py` and their regression suites. Run:

    python -m unittest discover -s research/seifert_activation/tests -q
    python research/seifert_activation/scm05_reference.py

A clean execution at the time of this report passed **78/78** local CPU tests, with **32 new SCM-05 tests** (14 continuous geometry, 8 quantization, 6 audit, 3 provenance, 1 three-strand link). The preexisting suite covered 46 methods. The reference fixture is `results/scm05_cpu_reference.json`. Its SHA-256 is recorded in the GitHub PR metadata after upload.

## Measured synthetic fixtures

| Fixture | Sparse/finely sampled | Observed winding | Conditional certification | Known analytic topology |
| --- | --- | ---: | --- | --- |
| `q(theta)=2 exp(4i theta)` | 4 samples | 0 | refusal | 4 turns |
| `q(theta)=2 exp(4i theta)` | 64 samples | 4 | pass under `L=8.000001` | 4 turns |
| `C=[sigma1^2,sigma2^2]` PL | 97 samples after refinement | pair winding 0 | 3 bits/coordinate sufficient at `R=3` | `mu123=+1` |

The Borromean-type fixture's quantized PL decoder remains connected through a collision-free homotopy. The strict SCM-01 Artin extractor separately rejects its sampled x-coordinate ties. This test deliberately distinguishes a mathematical link preservation witness from a particular crossing-word extractor's numerical preconditions.

The sampling and quantization proof uses rational coordinates obtained from the stored numeric values. The synthetic PL derivative bounds come from exact rational segment differences with outward-rounded Euclidean lengths. For the circle, the derivative upper bound follows from the specified analytic function and the script checks a large-margin sufficient inequality. **Neither case provides a neural derivative bound**.

## Limitations

- Exact rational predicates eliminate floating-point uncertainty in the geometry **of recorded values**. They do not establish a supremum derivative bound for an arbitrary model.
- Numeric neural readout charts can change their link types under projection choice. All results are chart-specific and depend on the model and intervention contract.
- The 3-bit sufficient width concerns chart coordinates of a single synthetic PL link; it is not a claim about bits per network weight, overall latent compression, or a learned quantized model.
- The next experimental step is to run SCM-04's stored capture and SCM-05's read-only audit on the actual fixed RMP checkpoint artifacts, with a separate proof of derivative bounds.
