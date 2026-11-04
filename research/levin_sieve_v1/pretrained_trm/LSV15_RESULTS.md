# LSV-15: Correlated LayerNorm intervention certificates, 64 pretrained cases

Date: 2026-11-04. Source commit pinned to the original third-party TRM-inspired Sudoku checkpoint; 44,001 parameters, exact-real lift of binary32 frozen weights, exact binary32 stage-one state, and three additional recurrent passes evaluated with outward-rounded Arb. The 64 first-missing-cell goals and class pairs were frozen in the preceding LSV-13 rank study. No model was retrained.

## Results

| Centered transverse radius at final pre-LayerNorm bus | Proven unchanged top digit | Proven unchanged top two digits |
|---|---:|---:|
| 1e-8 | 64/64 | 64/64 |
| 1e-5 | 64/64 | 64/64 |
| 1e-3 | 64/64 | 64/64 |
| 1e-2 | **64/64** | 58/64 |

An arbitrary per-token shared shift is permitted **at any amplitude in the mathematical exact-real model**. The remaining component must satisfy `||P delta||_infinity <= r` at the specified final pre-LayerNorm interface. The proof uses certified algebraic cancellation of the shared variable and a rational inequality for the transverse component. Results are for the chosen first missing cell in each Sudoku puzzle, irrespective of whether the clean prediction was correct.

The 48 frozen gamma parameters are all nonzero (smallest absolute value about **1.011134**). The corresponding final LayerNorm therefore has an exact 81-dimensional gauge kernel on its 3888-dimensional pre-normalization input, with Jacobian rank **3807** everywhere.

The exact sufficient-radius thresholds from the observed frozen clean margin intervals are:

- All 64 selected winner digits: **every `r < 0.07546051515573275`**.
- All 64 selected top-two digit pairs: **every `r < 0.00140648959646679`**.

The full exact rationals and per-case limiting competitors are saved in [LSV-15b thresholds](results/lsv15_exact_thresholds_64_20261104.json). These values optimize the stated sufficient inequality, not the actual worst-case perturbation radius.

## Evidence and replay

The archived compressed [64-case full proof receipt](results/lsv15_gauge_64_20261104.json.gz) contains every rigorous centered-input interval, the checkpoint weight entries in exact hexadecimal, and all original per-case rational slack values. The uncompressed JSON SHA256 is `af16058f9340f0e413393f2c940a877911149b3906a0fa77a0c2542ac4f3b17e`; its deterministic compressed form has SHA256 `d6d2950aca451ee33eec939b9498577d6c93ca5eb19663a378c8387bd462e12e`. The original GitHub Actions receipt was generated in [run 41831774028](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41831774028). The original exact rational replay checked every entry's implied clean and perturbed bounds.

From the repository root (Python standard library only for the offline verification):

```sh
python research/levin_sieve_v1/pretrained_trm/run_gauge_threshold.py \
  research/levin_sieve_v1/pretrained_trm/results/lsv15_gauge_64_20261104.json.gz \
  --verify research/levin_sieve_v1/pretrained_trm/results/lsv15_exact_thresholds_64_20261104.json
```

For a new proof computation, `python-flint==0.9.0` and CPU PyTorch `2.7.1` are installed in [the workflow](../../../.github/workflows/levin-gauge-quotient.yml) before invoking `run_gauge.py --fetch`. Its numerical component must match the previously pinned checkpoint and roster.

## Trust boundary

The analytic shift invariance and rational optimization formulas are exact mathematical statements. Checkpoint-specific certificates depend on the trusted Arb path that encloses the centered pre-LayerNorm inputs. The standalone verifier recomputes rational bounds from those saved intervals but does not independently reconstruct the entire pretrained network. The result concerns the final LayerNorm preactivation, not arbitrary 3,888-dimensional damage injected after the first recurrence; the earlier full-ambient radius remains `1e-14` under LSV-14. It does not certify finite-precision hardware invariance for unbounded shifts.
