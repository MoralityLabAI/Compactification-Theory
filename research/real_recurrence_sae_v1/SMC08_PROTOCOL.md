# SMC-08: Source-locked Jacobian geometry and globally certified curvature envelope

**November 15, 2026.** Follows merged [SMC-07](SMC07_RESULTS.md). The pure mathematics is [OLM-08 in MathCan](https://github.com/patrickdugan/MathCan/blob/research/olm08-validated-tube-20261115/research/orbit_lock/OLM08_VALIDATED_CAUSAL_TUBE.md).

## Research question

SMC-07 found that, in an independently pretrained 44,001-parameter TRM-inspired Sudoku recurrence, active decoder directions from an unsupervised 48→96 top-four SAE cause larger equal-norm final goal-logit displacements than isotropic and selected inactive decoder directions. The cluster-averaged active-minus-isotropic effect at radius 0.2 was +0.10247, descriptive 95% puzzle-bootstrap interval [+0.05956,+0.14785]. These findings motivate a stricter test of **where the sensitivity comes from**.

At each heldout stage-one goal token, measure the ten-output by 48-input Jacobian of the frozen three-step continuation in FP64. Compare its locally maximally sensitive right singular direction against active SAE, inactive SAE and isotropic directions. Measure the first-order approximation error at fixed radii, then check it against an independently computed *valid exact-real global Hessian ceiling*.

## Pinned provenance and split

This run reuses the SMC-07 source lock unchanged:

- Upstream: NagaSatyaPhanindraVallabhaneni/tiny-recursive-model, commit f0406195fece4d0db29ced28f3a69d1993d95b1e
- Checkpoint size 181147 bytes; SHA-256 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1
- Puzzle roster SHA-256 eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611
- 64 deterministic independent Sudoku puzzles grouped by puzzle ID modulo four; 32 train, 16 validation, 16 final test
- SAE initialization seeds 71,89,113; same architecture, normalization, 300-update budget and validation-only checkpoint selection as SMC-07

The source teacher remains frozen. Source stage-one states are computed in its original FP32 execution and then treated as exact binary anchors for a three-step model with parameters cast to FP64 for numerical differentiation. No Sudoku target or heldout intervention outcome affects SAE fitting.

## Registered measurements

For each of 16 heldout puzzles, identify the first blank cell, freeze all other 80 stage-one token states and let

\[
F_i:\mathbb R^{48}\to\mathbb R^{10}
\]

give the final goal logits after three shared-block iterations. Compute numerical \(J_i=DF_i\) by FP64 autograd and the 10 singular values of its 10×48 Jacobian. Its top right singular vector is a **goal-conditioned oracle control** and represents maximal local L2-to-L2 sensitivity.

From the learned SAE and the same stage-one token, sample exactly four directions:

- Highest active SAE decoder atom, in original hidden units, unit L2-normalized.
- Uniformly chosen inactive SAE decoder atom, unit-normalized.
- Seeded isotropic Gaussian direction, unit-normalized.
- Top right singular vector of J, with deterministic sign convention, unit-normalized.

For each direction, record its linear slope \(\|Ju\|_2\), its slope as a fraction of \(\|J\|_2\), and the actual output displacement at radii 0.02, 0.2 and 0.8. Also report first-order Taylor residual norm, observed secant curvature \(2\|\mathrm{residual}\|/r^2\), and final teacher-label flips. The numerical secant-curvature quantity is **not a uniform Hessian bound**.

## Rigorous global derivative certificate

The separately vendored [OLM-08 exact-rational checker](validated_causal_tube.py) reads exactly stored checkpoint weights and computes dyadic upper bounds on Frobenius norms via integer-square-root arithmetic. For a residual erf-GELU mixer, it applies \(|GELU'|\le5/4\) and \(|GELU''|\le4/5\); for population LayerNorm, it applies the analytic bounds on the Jacobian and Hessian using the model's exact stored gamma and epsilon. It then transports global coefficients through the twelve elementary layers:

\[
A_{j+1}=a_{j+1}A_j,\qquad
B_{j+1}=b_{j+1}A_j^2+a_{j+1}B_j.
\]

The resulting \(M_{\mathrm{global}}=\|W_{\mathrm{head}}\|_{\mathrm{upper}}B_{12}\) is an upper bound on the exact-real three-step map's Hessian operator norm **everywhere**, including the measured intervention tubes. Its certified remainder bound for an exact input shift ru is

\[
\|F_i(h+ru)-F_i(h)-rJ_i u\|_2\le M_{\mathrm{global}}r^2\|u\|_2^2/2.
\]

The exact rational numerator/denominator of M and its approximate decimal exponent are stored in the artifact. The separately measured Jacobian is FP64 autograd; its singular values are **not** interval-certified. A global outer bound that is too large to distinguish experimental effects is recorded as vacuous rather than used to claim a finite-radius dominance certificate.

The existing [LSV-17 signed-adjoint Arb engine](../levin_sieve_v1/goal_taylor_v1/README.md) supplies a path to *localized* tube bounds with verified spectral coefficients. This stage provides a reproducible global certificate and measurements to determine how much localization is needed.

## Falsification and evidence boundaries

A large active-feature effect is not sufficient for interpretability. Its Jacobian slope may merely align with the output gradient. An oracle optimal-Jacobian direction should match the maximum first-order slope, while SAE atoms have measured ratios below one. Rank J is at most ten and leaves at least 38 first-order-null directions in this 48-channel local input.

If the finite-radius output differences are dominated by nonlinear remainder terms, the first-order causal Gram is an inaccurate description at the declared radii. If the global Hessian ceiling is enormous, it is mathematically sound but empirically uninformative. Compare the numerical slopes with the finite effects and preserve this distinction.

The **same 16 puzzle IDs** are reused from SMC-07, so this is a follow-up analysis of an already examined test cohort. Treat its outcomes as explanatory diagnostics; an independent new puzzle-family evaluation would be required for confirmatory generalization.

## Reproduce

CPU-only, Python 3.11, PyTorch 2.7.1 and the already declared pretrained checkpoint lock:

    python -m unittest discover -s research/real_recurrence_sae_v1 -p 'test_smc08_curvature.py' -v
    python research/real_recurrence_sae_v1/smc08_curvature.py --fetch --max-cases 16 --seeds 71 --sae-steps 75 --out runs/smc08/pilot.json
    python research/real_recurrence_sae_v1/smc08_curvature.py --fetch --max-cases 64 --seeds 71,89,113 --sae-steps 300 --out runs/smc08/full.json

The [dedicated workflow](../../.github/workflows/smc08-curvature.yml) runs source-pinned pilot/full jobs and uploads per-puzzle receipts. Scientific conclusions should be drawn from successfully completed and inspected full runs; numerical results are not treated as exact-real interval bounds.
