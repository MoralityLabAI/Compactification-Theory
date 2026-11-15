# NOR-01: numerical-range gated recurrent \(L_p\) orbit sketches

**Scope:** deterministic linear recurrent-model surrogate, a new analytic theorem in [MathCan NOR-01](https://github.com/patrickdugan/MathCan/pull/45), and empirical finite-window checks. This is neither a trained-model result nor a constructive implementation of OpenAI Math #094's general existential dimension bound.

## Provenance

Two OpenAI papers from `openai/math`, pinned at `4eaee87548250b2682affda7d05d947c05554bae`:

- [#094: Subpolynomial dimension reduction in Lp, September 23, 2026](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/preprints/Subpolynomial-dimension-reduction-in-Lp-September-23-2026/build/paper.tex).
- [#325: A direct proof of the complete Crouzeix inequality, September 26, 2026](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/preprints/A-direct-proof-of-the-complete-Crouzeix-inequality-September-26-2026/build/main.tex).

## Applied theorem

For a real matrix \(A\), complex numerical radius \(w(A)\le q<1\), and initial vector \(x\), the complete Crouzeix theorem applied to \(P(z)=z^t\) gives \(\|A^t x\|_2\le 2q^t\|x\|_2\).

An isometric Gaussian map sends a real Hilbert space into real \(L_p\). Paper #094 then gives an existential finite-set distortion-\(D\) embedding of the \(T\) observed orbit states and the zero anchor into \(\ell_p^d\), with

\[
d\le \exp(C_{p,D}(\log(T+1))^{\gamma(p)}),\quad
\gamma(p)=\begin{cases}2-p,&1<p<2,\\ 1-2/p,&p>2.\end{cases}
\]

Clip the unobserved states \(t\ge T\) to zero. If \(E_T\) is the actual worst additive error among the finite observed pair distances including the anchor, every pair of times (even beyond the measured window) obeys

\[
|\|z_s-z_t\|_p-\|A^s x-A^t x\|_2|
\le E_T+4q^T\|x\|_2
\le4(D-1+q^T)\|x\|_2.
\]

This proof is developed in MathCan PR #45. Its Lean module certifies the combinatorial \(E+2B\) stitching lemma over integer metrics. The imported analytic inequalities remain paper-level.

## Reference surrogate and controls

32 nonidentical rational upper-Jordan blocks \(J_i=[[a_i,b_i],[0,a_i]]\) use \(a_i=3/8-i/1024\), \(b_i=1-((5i)\bmod7)/256\). Their complex numerical ranges are explicit disks. The block-diagonal operator has exact \(w(A)=7/8\) and spectral radius \(3/8\), while \(\|A\|_2=9/8\). Thus the experiment includes nonnormal transient amplification.

The sketch is seeded Gaussian with \(k\) shared rows. For \(p=4\) it uses the exact Gaussian fourth-moment normalization \(\mathbb E[G^4]=3\); a finite random sample does not supply the existential worst-case dimension guarantee. An \(L_2\) sketch is retained as a baseline. All prefix pairs are checked, followed by extra future pairs; the all-horizon claim follows analytically from the radius certificate, rather than from extrapolating those observations.

A negative control confirms that the spectral radius \(3/8\) alone gives the wrong one-step estimate. Another block family with \(\rho<1\) but \(w>1\) is correctly rejected by the gate.

## Reproduce

Python 3.10+ with standard library only:

```bash
python -m unittest discover -s research/nor01_numerical_orbit_sketch -p 'test_orbit_sketch.py' -v
python research/nor01_numerical_orbit_sketch/orbit_sketch.py --check
python research/nor01_numerical_orbit_sketch/orbit_sketch.py --check --p 2
```

The frozen default receipt is [results/seed325094_p4.json](results/seed325094_p4.json).

Default \(p=4\), seed 325094: 64 ambient coordinates, 24 sketch coordinates, cutoff 18, 16 additional future checks, observed worst prefix additive error \(0.31840137\), tail radius bound \(0.18079023\), all-horizon derived additive bound \(0.67998182\), and finite-probe worst error \(0.31840137\). These numeric observations are floating-point estimates; no interval enclosure certifies their last bits. The rational \(w(A)\) and block-power formulas have independent exact derivations.

## Research continuation

Extract Jacobians of a trained looped transformer and establish certified numerical-range enclosures across actual traces. Nonlinear trajectories require their own Jacobian-variation and perturbation bounds. A further constructive finite-dimensional embedding experiment should compare random sketches to the existential \(L_p\) frontier on held-out trajectories. Avoid treating this finite surrogate as evidence of learned reasoning gains.


## NOR-02: matrix-valued curvature filter (15 November 2026)

[Pure mathematical proof and Lean scope](https://github.com/patrickdugan/MathCan/blob/research/nor01-orbit-stitching-20261115/research/nor02/FILTERED_ORBIT_CURVATURE.md) extend NOR-01 using OpenAI #325's complete matrix-valued inequality. The two coefficient matrices of \(C(z)=\bigl(\begin{smallmatrix}0&1\\z&0\end{smallmatrix}\bigr)\) do not commute, so this benchmark tests a genuinely matrix-valued temporal filter \(P_r(z)=(z-q)^rC(z)\).

With numerical range inside \(D(a,b)\) for \(a,b>0\), \(q=a+b<1\), the exposed-point zero supplies the sharp scalar extremal
\[
\sup_{z\in D(a,b)}|z|^t|z-q|^r
\le (b/a)^{r/2}q^{t+r}
\sqrt{\frac{t^t r^r}{(t+r)^{t+r}}}.
\]
Under the explicit radial-attainment condition from the note, equality holds. The complete polynomial bound is then \(2\) times this quantity, uniformly in matrix coefficient dimension. A direct polynomial-difference argument strengthens the clipped-tail pair bound to \((1+q)B_T\).

The optimized all-horizon sketch budget is
\[
\max\{E_T,\ E_{0,T}+B_T,\ (1+q)B_T\}
\]
with \(E_T\) measured over the observed window and \(E_{0,T}\) measured against the zero anchor. MathCan/FilteredOrbit.lean proves the corresponding generic discrete stitching theorem over natural-number metrics with independent late-pair budgets, not the full analytic theorem.

### Held-out finite controls and exact geometry

The new 32-block model has 64 base dimensions and two coefficient channels, hence **128 Hilbert dimensions** before sketching to **24 L4 coordinates**. Each rational block numerical-range disk lies inside \(D(3/8,1/2)\); block 0 realizes that disk exactly. The shared complex numerical radius is \(7/8\), while the first block has operator norm \(9/8\).

A seeded order-2 filter at cutoff 18 gives a measured prefix max error \(0.019230627339\), anchor error \(0.017641810799\), curvature tail radius \(0.007150104311\), and all-horizon optimized budget \(0.024791915111\). A disk-only bound on the same tail radius gives \(0.180790227001\), approximately **25.28×** larger. The coefficient triangle bound gives \(1.038131381609\). These are bounds of different mathematical strength, not model-quality improvement measurements.

All prefix pairs are measured; 18 additional time points are probed as an independent finite diagnostic. The error theorem covers all later times analytically. The observation errors come from floating-point calculation, and the random sketch has no guaranteed subpolynomial dimension rate. No results are claimed for a trained transformer.

Run with Python 3.12 standard library:

    python -m unittest discover -s research/nor01_numerical_orbit_sketch -p 'test_*.py' -v
    python research/nor01_numerical_orbit_sketch/filtered_orbit.py --check
    python research/nor01_numerical_orbit_sketch/filtered_orbit.py --check --order 3 --p 2

The frozen receipt is [results/nor02_filtered_p4.json](results/nor02_filtered_p4.json). Ten new tests include exact rational disk inequalities, independent binomial-versus-Jordan evaluation, noncommutation, an extremal-attainment check, filter orders 1–4, the scalar constant-2 witness, and two additional seeded controls. The CI job compares selected receipt fields to the committed reference using strict numeric tolerances.

**Open gate:** certified complex numerical ranges for trained looped-model Jacobians, with an explicit nonlinear remainder and held-out-depth test, are required before this can become an applied mechanistic-interpretability claim.

## NOR-03: real trained RRC-25 nonlinear GRU

[Trained model interval experiment, verification protocol, and early results](NOR03.md). This replays the original frozen 1,000-parameter GRU, propagates exact rational intervals through sigmoid/tanh, and computes the true nonlinear 12-step block Jacobian product at trained hidden states. Source-locked controls distinguish the amplification at hidden zero from later contraction. Direct interval products test six source-family inputs at eight zero-input steps; a bounded 32-step continuation audit tests two of those families in CI. The pure-math proof and the quadratic remainder theorem live in [MathCan NOR-03](https://github.com/patrickdugan/MathCan/tree/research/nor01-orbit-stitching-20261115/research/nor03).
