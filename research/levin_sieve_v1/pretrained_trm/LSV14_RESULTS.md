# LSV-14: Uniform goal-history rank over a positive-radius neighborhood

**2026-11-04.** This is a numerical proof for the **exact-real lift** of an externally pretrained 44,001-parameter TRM-inspired Sudoku model. It preserves the LSV-13 SHA256 checkpoint lock, fixed 64-case roster, stage-one hidden-state inputs, and the LSV-12 selection of four Jacobian columns per case.

## Certified full-ambient results

| Infinity-ball radius in all 3,888 hidden coordinates | Rigorously certified rank-four minors | Exact-real final top-one stable | Top-two stable |
|---|---:|---:|---:|
| 1e-12 | 0/64 | 62/64 | 29/64 |
| **1e-14** | **64/64** | **64/64** | **63/64** |
| 1e-16 | 64/64 | 64/64 | 64/64 |
| 1e-18 | 64/64 | 64/64 | 64/64 |
| 1e-20 | 64/64 | 64/64 | 64/64 |

The negative r=1e-12 result is a failure of the conservative interval certificate, without evidence that the model's true Jacobian rank decreases. The r=1e-14 result holds for the entire coordinatewise input box around each of the 64 fixed stage-one hidden states.

The permanent receipt [lsv14_robust_rank_1e14_20261104.json](results/lsv14_robust_rank_1e14_20261104.json) has original source-byte SHA256 `b2bef2927fd69fcd95427a5e68327f43ea400f7d78d2b6c9bb6a79a419688b7f`. Its 64 records contain each of the sixteen outward-rounded derivative enclosures, its interval determinant, the independently evaluated rational Neumann norm and the output-class stability flags. The maximum exact-rational Neumann norm upper bound, displayed to nine decimal places, is `0.596624028`. All 64 bounds are strictly below one.

## Proof and limits

For each selected 4-by-4 Jacobian minor M(z), the Arb engine soundly encloses every entry for all z in the full 3,888-dimensional closed infinity box. An independently checked 24-term determinant expansion excludes zero at r=1e-14 in every case. As an alternative, choose any invertible exact rational reference matrix C, compute its rational inverse, and bound `q = max_k sum_i abs(C_inverse[k,i]) * sum_j sup(abs(M_ij-C_ij))`. If q<1, the Neumann series makes `C^-1*M` invertible throughout the box. The rational reference may come from approximate autograd; numerical accuracy of that reference is unnecessary because the enclosure bounds the exact error.

Consequently the four-time margin-history map has differential rank four throughout every certified region. Every smooth exact factorization of those four real-valued margins in these neighborhoods needs at least four real local state coordinates.

These results certify a local geometric lower bound. They make no claim about minimal bits, global model state, reasoning gains, actual hardware rounding, practical-sized state damage, or biological competence. The guarantee applies to exact-real erf-GELU and LayerNorm with frozen checkpoint weights and observed FP32 initial hidden states, and relies on correctness of the verified operation graph and Arb interval backend.

## Reproduction and audit

- [Initial 1e-12 negative result](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41825389885)
- [Successful r=1e-18 and r=1e-20 verification](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41826176092)
- [Successful r=1e-14 and r=1e-16 verification](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41826940514)
- [Uniform-rank theorem and implementation](ROBUST_RANK_REGION.md)
- [Pure Python rational witness verifier](verify_robust_receipt.py)

To replay the committed finite witness with the Python standard library only:

```bash
python research/levin_sieve_v1/pretrained_trm/verify_robust_receipt.py research/levin_sieve_v1/pretrained_trm/results/lsv14_robust_rank_1e14_20261104.json
```

To regenerate the exact-real numerical enclosures from the source-locked pretrained model, install `torch==2.7.1` (CPU) and `python-flint==0.9.0`, then run:

```bash
python research/levin_sieve_v1/pretrained_trm/run_robust_minor.py --fetch --radius 1e-14 --max-cases 64 --precision 160 --out runs/lsv14/1e-14.json
python research/levin_sieve_v1/pretrained_trm/verify_robust_receipt.py runs/lsv14/1e-14.json
```
