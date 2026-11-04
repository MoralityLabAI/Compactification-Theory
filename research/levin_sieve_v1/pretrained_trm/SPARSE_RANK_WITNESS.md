# LSV-12: Certifying goal-horizon rank with sparse Jacobian minors

**Date:** 2026-11-04. **Mathematical evidence:** a short exact linear-algebra theorem with independently checked rational examples. **Model evidence:** a deterministic greedy selection of `4x4` minors from the *point-autograd* `4x3888` derivative matrices of the hash-pinned pretrained Sudoku checkpoint, evaluated over the unchanged 64-puzzle roster. A verified error bound on those derivatives is still outstanding, so **the numerical model ranks remain observational**.

## Theorem SM-01: a 16-entry sufficient witness for rank four

Let `J` be the *mathematically exact* real `4xn` Jacobian (where `n>=4`) of four declared goal-history observations at a fixed state. Let `I` select four coordinate columns, and write `M=J[:,I]`. Suppose a computed matrix `M_hat` and a valid error bound `epsilon` establish `|M_ij-M_hat_ij|<=epsilon` for all sixteen entries. Then

\[
\boxed{\sigma_4(M)\ge \sigma_4(\hat M)-4\varepsilon.}
\]

Consequently, the strict inequality `4 epsilon < sigma_4(M_hat)` proves `rank J=4`.

**Proof.** With `E=M-M_hat`, the Frobenius norm satisfies `||E||_F <= sqrt(16)*epsilon = 4epsilon`, and `||E||_2 <= ||E||_F`. The singular-value perturbation inequality implies `sigma_min(M) >= sigma_min(M_hat)-||E||_2`. A strictly positive lower bound makes `M` invertible, hence `rank J=4`. QED.

The choice of which columns to check can be data-dependent. Any one verified minor suffices. This reduces the later *rigorous derivative enclosure* from `4x3888 = 15,552` Jacobian entries to **16 chosen entries per case**, provided the full-model point function and the selected columns are frozen before validation.

## Construction

`greedy_minor.py` uses deterministic residual-orthogonalization in the four-dimensional row space to select four columns with large residual norms. This is a practical heuristic for discovering a well-conditioned minor, not a proof that the chosen minor is globally optimal. On each selected minor, the code calculates `sigma_min(M_hat)` and emits the **strict upper limit** `epsilon < sigma_min/4` required of an independently verified *absolute* error on each of its sixteen entries. A selected minor with a small threshold would demand more numerical precision; a larger threshold is easier to certify.

The chooser uses FP64 numbers, and the reported error threshold **does not assert** the autograd entries have already been verified to that accuracy. The eventual numerical proof must account for the exact checkpoint's FP32 weights, FP64 forward arithmetic, approximate GELU, LayerNorm, and reverse-mode differentiation, or instead compute the sixteen mathematical derivatives with a validated high-precision implementation.

The same frozen source `f0406195fece4d0db29ced28f3a69d1993d95b1e`, checkpoint SHA256 `3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1`, and roster SHA256 `eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611` from LSV-09/10 are required. A `64`-case result records every selected hidden-coordinate column index and observed threshold. Train/test labels stay out of selection.

## Reproduction

```sh
python -m unittest discover -s research/levin_sieve_v1/pretrained_trm -p 'test_*.py' -v
python research/levin_sieve_v1/pretrained_trm/run_minor_witness.py --fetch --out runs/lsv12/minors.json
```

The exact unit tests include a sparse four-dimensional witness embedded in a hundred-dimensional state and an independent exact `Fraction` determinant. They also test the explicit singular-value error budget. The pretrained numerical workflow exports an observational receipt and must label the count of *rigorously certified true ranks* as zero until validation of the derivative error.

## Further mathematical opportunity

The small-minor route is an instance of proof-carrying mathematical discovery: an inexpensive floating-point search identifies promising finite witnesses, while expensive rigorous computation is reserved for the few values needed to certify them. For recurrent neural networks, one can pursue a validated 16-entry differentiation engine with interval/ball arithmetic and a pinned reference implementation, then test the certificate on a held-out model family. The full-model arithmetic contract and intervention domain must remain explicit.
