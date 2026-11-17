# SCM-06: exact real-model Lipschitz compiler, CPU receipt

**Executed 2026-11-13** in a clean CPU Python 3.13 / PyTorch 2.10 / SymPy 1.14 runtime. The existing SCM-05 synthetic suite passed 78/78 before work began; after adding the SCM-06 norm, architecture, audit and negative-control tests, 102/102 local tests passed at this stage. Final test count and SHA of the reference are recorded in the PR.

All new evidence is **synthetic or source-level mathematics**, with no trained neural activation observations. The new architecture checker models RMP's exact pre-norm explicit multihead attention graph, including true causal masking and both residual factors. The audit verifies four pinned GitHub blobs to keep that proof aligned with the source.

## Tested observations

- Exact dyadic matrix spectral upper bound: `diag(3,4)` returns exactly `4`; random matrices checked against float64 SVD; a subnormal `float32` entry retains its exact dyadic magnitude. A `float32` SVD rounding mismatch originally exposed an invalid *test comparator* and was fixed by using float64 comparison.
- LayerNorm bound: finite-difference-free automatic-differentiation checks at nearly constant and widely varying inputs stay beneath the analytic `||gamma||inf/sqrt(eps)` bound.
- Explicit causal blocks: random two-point finite changes and direct `torch.autograd.functional.jvp` angular derivatives are bounded for both a tied three-visit mock looper and three untied blocks.
- Strict prefix control: every observed pair entirely before an injected future token receives upper derivative `0` for all visits. A mixed pair retains zero only for the earlier-prefix pair.
- The radius/disk null-braid certificate passes on widely separated synthetic clean token positions with sufficiently small perturbation radius, and refuses a constructed baseline collision.
- Refusals cover training mode, approximate GELU, biased QKV, altered decoder source, missing new capture metadata, and nonfinite tensor coefficients.

## Reference

Run `python research/seifert_activation/scm06_reference.py`. It writes an immutable-style deterministic CPU fixture in `results/scm06_cpu_reference.json` containing exact rational per-visit factors for a three-visit mock model and a 128-hidden/4-head/64-token single-block stress shape. The latter uses **random untrained weights**, and its large global factors illustrate how loose the bound can become.

**Boundary:** The mathematical bounds are for the real-valued architecture with stored binary-float coefficients interpreted as exact rationals. The geometry checks additionally assume that reported capture coordinates equal ideal real activations; a verified forward-error envelope is needed for an end-to-end real-model topological certificate. They are not a formal verification of PyTorch floating-point continuity or end-to-end trained model correctness. Trained `LOOP-T`/`FF-U` checkpoints, D-shared dictionaries, and monitor bundles are not available in this tool runtime, so no real activation/braid or quantization benefit is reported.
