# LSV-13: Ball-validated differentiation of pretrained recurrent goal-history minors

**Date:** 2026-11-04. **Evidence:** new exact-real computational enclosure pipeline; GitHub Actions checks against pinned third-party Sudoku checkpoint. Mathematical result conditional on an explicitly defined exact-real network and the trusted Arb numerical enclosure engine. See the CI run and pinned receipt before claiming any particular number of successful certified pretrained cases.

## Motivation

[LSV-12](SPARSE_RANK_WITNESS.md) identified 64 promising sparse 4x4 Jacobian minors, but all 64 ranks were observational because PyTorch autograd's error relative to the exact network derivative had not been bounded. The exact-real version of LSV-13 checks those four-by-four minors using **rigorous ball arithmetic**. It aims to prove nonzero determinant directly, which avoids depending on a floating singular value or an assumed autograd tolerance.

## Mathematical object and theorem

Let `z1` be a precisely specified 81-by-48 real tensor, equal coordinatewise to the binary32 values output by the frozen checkpoint's first recurrence under a pinned CPU execution. Let `F:R^3888 -> R^3888` be the *mathematical* four-step model's shared recurrent block with every recorded binary32 coefficient interpreted as an exact real, the exact erf-based GELU function, the exact LayerNorm formula with the binary64 value of `eps`, and no arithmetic rounding during the recurrence. Let `a` and `b` be two **fixed** digit ids, chosen before verification, and let `j` be a fixed Sudoku cell. The goal-history map is

\[
  H(z) = \bigl(g(z),g(F(z)),g(F^2(z)),g(F^3(z))\bigr),
  \qquad g(z) = \bigl(W_{a,:}-W_{b,:}\bigr) z_j + (b_a-b_b).
\]

For four predetermined hidden-coordinate indices \(I\), define the minor \(M= DH(z_1)_{[:,I]}\). If each entry is enclosed in a rigorously computed real ball `B[i,j]`, and the rigorous determinant enclosure `det(B)` excludes zero, then

\[
\boxed{\det M\ne0\quad\implies\quad \operatorname{rank} DH(z_1)=4.}
\]

**Proof.** Each true derivative belongs to the corresponding ball by induction over the operation graph and sound ball evaluation of arithmetic and differentiable special functions. The determinant is a polynomial of the sixteen entries; sound ball arithmetic encloses its exact value, including all dependent uses of the same interval variable. If zero lies outside the determinant ball, the exact minor is invertible. Its four rows are independent and the full Jacobian has rank at least four; since it has exactly four rows, its rank is four. QED.

The constant-rank/implicit-function theorem then prevents any smooth, locally defined three-dimensional encoder with a smooth decoder from reproducing all four goal-history components on a neighborhood of that point. The conclusion is **local**, for the specified goal history and exact-real model, without claiming global dimension minimality for arbitrary non-smooth or discontinuous interfaces.

## Enclosed differentiation formulas

The verifier propagates a center value and four directional derivatives through the exact model. Linear token and channel mixing use Arb ball-matrix multiplication, with coefficients parsed from the checkpoint's binary32 tensors. For GELU,

\[
  G(x)=x\Phi(x),\quad
  G'(x)=\Phi(x)+x\frac{\exp(-x^2/2)}{\sqrt{2\pi}},
  \quad \Phi(x)=\tfrac12(1+\mathrm{erf}(x/\sqrt2)).
\]

For a LayerNorm row of width `d`, set \(\mu=d^{-1}\sum_jx_j\), \(c_j=x_j-\mu\), \(v=d^{-1}\sum_jc_j^2\), \(s=(v+\epsilon)^{-1/2}\). A perturbation direction \(u\) gives

\[
\begin{aligned}
  d\mu &= d^{-1}\sum_j u_j,& dc_j &= u_j-d\mu,\\
  dv &= 2d^{-1}\sum_j c_j dc_j,& ds &= -\tfrac12 s^3 dv,\\
  d\mathrm{LN}_j &= \gamma_j(s dc_j+c_j ds).
\end{aligned}
\]

All operations, including `sqrt`, `exp`, `erf`, matrix multiplication and determinant evaluation, use [FLINT/Arb](https://flintlib.org/doc/algorithms/ball-arithmetic/overview.html) through pinned `python-flint==0.9.0`. Ball radii record both transcendental approximation and arithmetic rounding. Inputs `arb(float(v))` are exact binary floats, not arbitrary strings rounded to decimal.

## Trust and provenance boundaries

The checkpoint is loaded only after the original SHA256 and Git blob hash pass, with `torch.load(..., weights_only=True)` and strict parameter names. The frozen 64-case Sudoku roster and the LSV-12 column selection are validated against their hashes. Winner/rival digits are chosen by the FP64 reference trajectory and subsequently checked against a ball-enclosed exact-real output logit; a rank certificate still applies to the frozen pair if a label identity check fails.

The `z1` input is an *exact real vector chosen from the measured stage-one binary32 state*. LSV-13 does **not** verify that binary32 hardware inference follows the real-arithmetic recurrence, nor does it establish a robustness radius, convergence, post-damage repair, task accuracy, or a global quotient. A proof of rank four also does **not** imply four persistent memory bits: the dimension counts locally smooth coordinates for the chosen four real goal readouts.

### Independent reproduction

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install python-flint==0.9.0
python -m unittest discover -s research/levin_sieve_v1/pretrained_trm -p test_validated_minor.py -v
python research/levin_sieve_v1/pretrained_trm/run_validated_minor.py --fetch --max-cases 1 --precision 128 --out runs/lsv13/verified.json
```

The first gate is one selected case to validate runtime and enclosure tightness. Further cases must retain all source locks, frozen column selections and the complete failed-case ledger. A nonzero numerical FP64 determinant with a ball containing zero remains **unproved**.

## LSV-12 error-budget closure (follow-up)

The validated runner also computes the original `4x3888` point-autograd Jacobian and, for each frozen 4-column selection, takes the *rigorous upper bound* on the absolute difference between the binary64 point-autograd entry and its Arb-enclosed exact-real derivative. The maximum of sixteen such error bounds per case must be strictly less than the originally recorded `sigma_min/4` threshold (stored as a conservative 12-decimal number). This closes LSV-12's original singular-value-perturbation proof obligation if and only if all 64 cases pass, independently of the determinant test. The receipt records both checks separately.
