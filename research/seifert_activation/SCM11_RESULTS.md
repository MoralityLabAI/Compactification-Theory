# SCM-11 CPU source intervention response, November 14, 2026

**Scope:** Synthetic, exact-real recurrent source models under the CPU IEEE/BLAS verifier assumptions. No trained LOOP-T/FF-U checkpoint, D-shared dictionary or real-model capture was available. No float32/CUDA implementation-error bound, neural compactification gain or smooth 4D Seifert-surface invariant has been measured.

## Verified theorem and independent tests

The new centered-attention inequality removes a common source-value offset and retains the source attention distribution when bounding perturbed values. A separate LayerNorm equivariance theorem propagates per-token channel-constant shifts by `alpha²` through each pre-norm block. The source-level gauge null-control reports the requisite full-path collision-free condition separately.

Thirteen new local CPU tests pass: analytic attention counterexamples, 60 random high-precision probability changes, source-certification of Q/K/V and p0 upper bounds, 80-digit small-model real-forward intervention containment in tied and untied/reinjection networks, source-audit provenance refusals, no-CUDA/no-link claims, and exact LayerNorm shift equivariance. The SCM-10 predecessor recorded 202 tests; complete combined hosted SCM-11 suite status is maintained by the new PR CI checks.

## Full-shape synthetic comparison

Source: deterministic SCM-09 RMP fixture, 64 tokens, 128 hidden, four attention heads, eight tied visits, beta=0.1. Injection token 4 after visit 0, two coordinate axes, observe tokens 16,32,48, RMS-relative radius 10^-6. Fixed frequency table and model identity are hashed in `results/scm11_cpu_reference.json`.

| Visit | Previous SCM-07/10 upper | SCM-11 refined upper | Source-only null-braid gate |
|---:|---:|---:|---|
| 1 | 0 | 0 | pass |
| 2 | 0.0017581 | 9.13572e-7 | pass |
| 3 | 16.0073 | 8.09771e-6 | pass |
| 4 | 1.56422e8 | 5.33340e-5 | pass |
| 5 | 1.01722e17 | 3.14795e-4 | pass |
| 6 | 6.63157e25 | 0.00175197 | pass |
| 7 | 4.33378e34 | 0.00940259 | pass |
| 8 | 2.83725e43 | **0.0495522** | pass |

The visit-eight ratio is 1.74649e-45 (refined / inherited). This compares **upper bounds**, not actual activation changes. SCM-05's geometric interpretation applies to this frozen readout chart and to the full continuous perturbation disk. SCM-10's tied beta=0.25 clean-state failure remains inconclusive.

The fixed reference uses the SCM-10 source-only centered tubes with outgoing binary64 interval radii. Its CPU BLAS arithmetic semantics remain an **external assumption**, and real neural floating-point capture errors remain unbounded. This experiment does not show that neural reasoning uses knots or that Seifert surface isotopy types differ.

## Reproduction

```sh
python -m unittest discover -s research/seifert_activation/tests -p 'test_scm11_response.py' -v
python research/seifert_activation/scm11_reference.py --check \
  --out research/seifert_activation/results/scm11_cpu_reference.json
python research/seifert_activation/scm11_audit.py --help
```

For a trained RMP source-only audit, provide the original immutable capture receipts, exact RMP source checkout, checkpoint/dictionary artifact roots and `monitor_calibration` bundle. The auditor writes a new source-locked output file and rejects missing sources or hashes. GPU execution and observed link certifications are explicitly false in all receipts.
