# SUD-01: semantic uncertainty dimension

This research bridge joins **OpenAI Math #119** (sharp information
contraction on a noisy Boolean cube) to the **entropy/dimension theme of
OpenAI Math #148** (self-similar measures).

[Mathematical derivation](THEORY.md) |
[Executable reference](reference.py) |
[Finite tests](test_reference.py) |
[Computational receipts](RESULTS.md)

For iid balanced semantic bits \(S_t=f(X_t)\), independent bit-flip
observations \(Y_t=X_t\oplus N_t\) of noise rate p, and a separated
two-map Cantor encoding of contraction \(0<r<1/2\), the posterior has
almost-sure Hausdorff dimension

\[
\dim_H\mathcal L(Z\mid Y_1,Y_2,\ldots)
=\frac{H(S\mid Y)}{\log_2(1/r)}
\ge\frac{h_2(p)}{\log_2(1/r)}.
\]

There are two related finite-resource bounds:

\[
\Pr(\widehat S_{1:T}=S_{1:T})\le(1-p)^T,
\qquad
B\ge T h_2(p)-h_2(\delta)-\delta T,
\]

where B is the capacity in bits of an auxiliary message enabling
block recovery with error at most delta.

### Reproduce

From repository root, on Python 3.10+ (stdlib only):

    python -m unittest discover -s research/semantic_uncertainty_dimension_v1 -p 'test_*.py' -v
    python research/semantic_uncertainty_dimension_v1/reference.py --n 4 --p 0.1 --exhaustive

The latter checks all 12,870 balanced four-input Boolean functions.
A one-coordinate function attains the error and entropy minima.

**Proof-status boundary.** The proof note supplies a conditional
derivation for observation-dependent measures; it is not an imported
formalization of OpenAI #148, which addresses stationary measures.
The Python receipts are finite numerical checks and include no trained
model evaluations. A later experiment should validate the encoding
metric and the observation channel before transferring these bounds
to LoopedTransformers Q5 or the Compactification Theory RC-01 ledger.
