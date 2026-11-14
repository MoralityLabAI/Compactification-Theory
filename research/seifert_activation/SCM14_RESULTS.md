# SCM-14 executed CPU reference and explicit proof boundary

**Date:** November 14, 2026. **Source:** synthetic dyadic explicit-attention RMP graph from SCM-09; exact-real functional semantics conditional on the prior CPU IEEE-754/BLAS verifier assumptions. RMP C3 trained checkpoints and selected D-shared dictionaries remain external to this environment.

## Results

The local source-only Hessian implementation passes the dedicated analytic, autograd, provenance and conditional bridge tests. The strict full inherited suite is run separately; hosted CI evidence must be checked from the PR rather than inferred from local test output. Its core dependency on SCM-10 was retained without modification.

Full synthetic 64-token, 128-hidden, four-head, eight-visit tied model, beta=.1, source RMS-relative intervention radius 1e-6, token 4 injection after visit zero, observed tokens 16, 32, 48:

| Visit | Uniform global Q/K/V image bound | Certified source-local Q/K/V bound |
| ---: | ---: | ---: |
| 1 | 0 | 0 |
| 2 | 1.28075e-9 | 2.85e-12 |
| 3 | 2.28784e-5 | 2.7547e-10 |
| 4 | 0.255029 | 1.417019e-8 |
| 5 | 2.46518e8 | 6.0317469e-7 |
| 6 | 1.32766e22 | 2.345334116e-5 |
| 7 | 6.9236e35 | 0.00086853079504 |
| 8 | 3.55569e49 | **0.03174224011981** |

Values are per-visit **maximum upper bounds on the projected nonlinear Taylor remainder** over the *entire normalized unit intervention disk*. The source-local bound is always at most the global bound in this frozen reference. A small untied 8-hidden model with input reinjection also passes the separate source reference and derivative checks.

## Independent checks and limits

LayerNorm second derivatives were compared against 70-digit mpmath differentiation. Causal softmax's second derivative inequality was tested with independent PyTorch automatic differentiation. Both tied and untied/reinjected small decoders were checked at multiple unit-disk locations using independently computed float64 output Hessians. Source-only source-local bounds cover these observations; the double-precision calculations are empirical diagnostics of the exact-real proof.

The bridge refuses unproved source scope, negative bounds, malformed token order, missing visits, untrusted float-valued error fields and mismatched strand counts. Its geometry remains **conditional on certified affine anchors/Jacobians**, which SCM-14 has not yet supplied for trained RMP.

**Scientific outcome:** the global Hessian propagation is too loose at late visits, while certified local Q/K/V enclosures reduce the synthetic visit-eight Taylor remainder from approximately 3.56e49 to 0.03174. This certifies a second-order **upper bound**, not an actual nontrivial braid or a compactification capability improvement.

## Run

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s research/seifert_activation/tests -p 'test_scm14*.py' -v
python research/seifert_activation/scm14_reference.py --check \
  --out research/seifert_activation/results/scm14_cpu_reference.json
```

The reference includes source hashes for the new Hessian code and its existing input/interval dependencies. CPU timings depend on the interpreter and BLAS backend and are not a claim about GPU throughput.
