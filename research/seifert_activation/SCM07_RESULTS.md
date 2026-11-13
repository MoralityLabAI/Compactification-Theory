# SCM-07 executed CPU reference and proof boundary

**Date:** November 13, 2026. **Code:** `local_jacobian.py`, `scm07_audit.py`, `scm07_reference.py` with 28 new test methods in `tests/test_local_jacobian.py`; 102 historical SCM-01 through SCM-06 methods retained. Local `python -m unittest discover -s tests -q` passed **130/130** after the final four provenance and reference tests were added. (The new test method count is 28.) Python 3.13, PyTorch 2.10 CPU, SymPy 1.14.

The exact mathematical implementation propagates a triangle of tokenwise causal sensitivity rather than a whole-sequence bound. Optional LayerNorm-center bounds use exact binary rational centering energies and outward rational square-root ceilings. Tests compare automatic differentiation and actual two-point perturbations in small pre-norm causal attention blocks to upper predictions. Local-center experiments use the captured float centers with a **synthetically asserted zero center error**, and are NOT trained-model certificates.

## Eight-visit 128d, 4-head, 64-token synthetic reference

`python scm07_reference.py --out /new/receipt/path.json` uses `torch.manual_seed(317)` and one untrained RMP-equivalent block tied for eight visits, with beta 0.25. A single token-four cyclic perturbation is observed through token slots 16, 32 and 48. The table reports the *largest* of the three observed-pair derivative upper bounds at the eighth visit:

| Perturbation amplitude | SCM-06 whole-sequence bound | SCM-07 causal source-global | SCM-07 center-local, **conditional** | Disk inequality at visit eight |
| --- | ---: | ---: | ---: | --- |
| `0.01` | 3.46×10³³ | 4.14×10³⁰ | 1.84×10¹⁰ | fails |
| `0.0001` | 3.46×10³¹ | 4.14×10²⁸ | 2.83 | fails |
| `0.000001` | 3.46×10²⁹ | 4.14×10²⁶ | 0.02662 | passes conditionally |

The center-local numbers are conditional because PyTorch's floating clean centers lack externally certified error radii. Their large contrast with source-global bounds demonstrates *potential numerical tightness*, not end-to-end formal verification. At amplitude `10^-6`, the mock disk criterion passes at every visit, conditional on the clean centers and coordinate readings. At `10^-4`, it passes through visit seven and fails at visit eight. The source-global causal-prefix negative control correctly gives zero for all eight visits when injecting at token 63 and observing 16, 32, 48.

## Verification scope

- Algebraic local LN inequality and attention-row triangle argument are proved in `SCM07_MATH.md`.
- Weight-matrix norm upper bounds use SCM-06's dyadic-integer implementation. The new source-global bounds are valid in exact real arithmetic for the pinned decoder architecture.
- **No trained RMP checkpoint, D-shared dictionary, monitor bundle or activation trace is available in this tool runtime.** No 4D Seifert surface or Khovanov cobordism map is computed.
- The SCM-05/06 numerical gap persists: ideal real activation output coordinates need externally verified forward-error intervals for a final observed-braid claim.

Executable evidence lives in `results/scm07_cpu_reference.json`. All new tests run under the existing `scm03-seifert-activation.yml` workflow inherited on the SCM-06 branch.
