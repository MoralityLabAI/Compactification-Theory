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
