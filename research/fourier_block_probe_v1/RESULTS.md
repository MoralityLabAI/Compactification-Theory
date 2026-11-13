# FPB-01: verified finite Fourier and heldout probe results

Date: November 13, 2026.
Source results: OpenAI Math #132 (superquadratic sensitivity gap),
and #192 (unbounded signed first-order Fourier / degree ratio).
Evidence class: exact finite reference and frozen synthetic heldout probe.
External source conclusions remain conditional on those manuscripts.

[Theorem and limitations](THEORY.md) |
[Reference](reference.py) |
[Heldout benchmark](benchmark.py) |
[CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43454724201).

## Exact 16/17-bit source

Let h16 be OR over four four-bit blocks. Each block accepts
precisely 0011 or 1100. Exact exhaustive evaluation of all
65,536 assignments gives:

- Exactly 38,416 assignments are rejected.
- Maximum global sensitivity of h16 is 4.
- Adding a direct OR sentinel makes h17's sensitivity exactly 5.
- Eight aligned pair masks and the sentinel mask are nine
  pairwise disjoint sensitive blocks at the all-zero input.

The parity lift G(x,y)=(-1)^h17(x xor y) has 34 input bits.
The analytical Fourier transport theorem says every raw-input
singleton coefficient vanishes, and the source degree 17 becomes
degree 34. The full source degree witness has coefficient 1/4096.
The small-case independent oracle exhausts all 16 Boolean
functions of two inputs and all their lifted Fourier masks.

Exact sign coefficients of the hidden source:

| Fourier mode | Exact coefficient |
|---|---:|
| Mean | -1695/4096 |
| Original 17 singleton inputs (max absolute) | 2401/4096, on sentinel |
| Raw singleton of 34-bit XOR lift (all) | 0 |
| Matched paired sentinel of XOR lift | 2401/4096 |
| Matched fourth-order lift of source input pair {0,1} | -343/4096 |
| Full 34-coordinate Fourier character | 1/4096 |

The source sentinel's first-order coefficient survives as a
*paired second-order character* in the lift. The original
sixteen Rubinstein coordinates each have zero first-order
coefficient. Higher-order paired interactions reveal their
source structure.

A five-bit majority surface mode supplies a finite selector
demonstration. Its singleton sum is 15/8. The whole selector's
singleton sum is 9375/8192 at real Fourier degree 35.
This deliberately small surface is a **structural control**:
it does not exhibit the unbounded violation of #192.

## Exact minimum mean-squared error under linear Fourier features

With independent uniform x,y and hidden sign label G(x,y), the
best population affine regression using the 34 raw sign singletons
is only the intercept mu=-1695/4096. Its MSE is exactly

\[
MSE_{\mathrm{raw}} =
1-(1695/4096)^2
=13904191/16777216.
\]

Adding only the matched sentinel parity feature gives population

\[
MSE_{\mathrm{paired}} =
MSE_{\mathrm{raw}}-(2401/4096)^2
=4069695/8388608.
\]

The squared-error gain equals 5764801/16777216.

For the sign decoder with its optimal coefficient, the exact
population accuracy changes from 5791/8192 (70.69%) with
the intercept alone to 6497/8192 (79.31%) with this one
paired feature. This is a statement about the Fourier
feature readout. Zero singleton coefficients do not preclude
arbitrary nonlinear classifiers.

## Frozen heldout probe experiment

Training: 4,096 samples from a seeded random uniform
pair of 17-bit registers, using seed 940132192.
Heldout: 4,096 independent samples using seed 940132193.
First-order Fourier regression coefficients and intercept
are fitted only on training samples. Each candidate uses
the same calibration/test examples and records all source
and data hashes.

| Probe | Coefficients | Heldout MSE | Heldout classification accuracy |
|---|---:|---:|---:|
| Intercept only | 0 | 0.818208 | 71.36% |
| Raw singleton sign features | 34 | 0.828930 | 71.36% |
| Matched sentinel pair | 1 | 0.485136 | 79.00% |
| All 17 matched pairs | 17 | 0.487637 | 79.00% |
| 17 matched pairs + 24 paired fourth-order features | 41 | 0.326570 | 95.31% |

The 41-feature representation uses additional observation
capacity and a richer registered feature grammar. Probe
performance is evaluated on a purely synthetic XOR-masked
Boolean target, not a trained neural model.

## FPB-5 exact adaptive-query control

The adaptive policy queries x0, then chooses which member of
each next coordinate pair to observe based on the previous
bit. It never reads both members of the same pair.

The reference exhaustively checks all conditional source
states for n=1,2,3,4,5, totaling 62 hidden XOR states.
Every adaptive observation-transcript law is identical
for every possible source z within the same n. An observer
who completes one pair can instead recover that source bit
exactly through xi xor yi. The proof in THEORY.md applies
to all adaptive policies that obey the unmatched-pair
restriction, not only to this finite test policy.

## Verification

- The source-exhaustive and spectral suite ran 15 tests.
- The heldout probe suite ran 8 tests.
- Both jobs passed in
  [GitHub Actions](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43454724201).
- The independent one-bit cancellation lemma was
  Lean-checked in
  [MathCan PR #34](https://github.com/patrickdugan/MathCan/pull/34),
  alongside 583 existing Python regression tests.

The original OpenAI #132 superquadratic family and #192
unbounded-ratio construction have not been reproduced in
this finite campaign. Their reported theorem statements
are inputs to the conditional selector theorem FPB-3.
The only operational intervention results are exact
Boolean masks and synthetic heldout probe observations.
No Q5 trained checkpoint or self-ascription semantics
are evaluated here.
