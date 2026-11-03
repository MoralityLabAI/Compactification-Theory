# LTR-03: spectral companion, finite prototype

This companion develops the operator stage of Logos–Topos Reconstruction. The Lean foundation is in [OntologicalProofs PR #5](https://github.com/patrickdugan/OntologicalProofs/pull/5), under `Neoplatonic/LTR/` and `research/logos_topos/`.

## What runs

`spectral.py` builds a finite spectral triple from an explicitly marked finite metric space. It uses exact rational arithmetic, with no GPU or third-party Python dependency. Eight local unit tests passed on 2026-11-02; the dedicated GitHub workflow repeats them.

For each unordered pair x != y, take a two-dimensional Hilbert-space summand with

    D_xy = [[0, 1/d(x,y)], [1/d(x,y), 0]],
    pi_xy(a) = diag(a(x), a(y)).

Set H to the direct sum of these summands, A = C(X), and D to the block direct sum. The representation is faithful for at least two points. D is self-adjoint. In finite dimension, resolvent compactness and bounded commutators hold automatically. Its eigenvalues are the pairs ±1/d(x,y), with multiplicity.

The commutator seminorm is exactly

    L_D(a) = max_(x != y) |a(x)-a(y)| / d(x,y).

Consequently the Connes distance between point states equals d. The upper bound follows from the corresponding pair in the maximum. For the lower bound use a(z) = d(z,y): the triangle inequality gives L_D(a) <= 1, and |a(x)-a(y)| = d(x,y). The implementation constructs this exact witness for every pair. This is a standard finite metric-to-operator construction; novelty is not claimed for it.

## A reproducible presentation-sensitivity example

Take the labelled binary completion samples 000, 001, 011 and the first-disagreement ultrametric with weights 1/2, 1/4, 1/8.

| Sentence order | Pair distances | Dirac eigenvalues |
|---|---|---|
| 0, 1, 2 | 1/8, 1/4, 1/4 | -8, -4, -4, 4, 4, 8 |
| 2, 1, 0 | 1/2, 1/2, 1/4 | -4, -2, -2, 2, 2, 4 |

The same logical samples produce different spectra after reordering coordinates. The exact metric-recovery theorem holds in both cases. A separate test permutes point names: it preserves the spectrum while changing distances between named points. Thus the spectrum alone loses marking information, and a chosen coordinate filtration contributes real geometric data.

The tests cover metric recovery through 16 points, the ultrametric inequality, spectral symmetry, scale changes, and invalid inputs. These are finite implementation checks, not certificates about all completions of an arithmetic theory.

## Infinite-dimensional direction

Pearson and Bellissard construct spectral triples from weighted rooted trees associated with ultrametric Cantor sets. Their construction is the relevant precedent: completion spaces have zero-dimensional topology, and the natural logical refinement structures are trees. See [arXiv:0802.1336](https://arxiv.org/abs/0802.1336), published in J. Noncommutative Geometry 3 (2009), 447–480.

An LTR extension needs to specify an effective filtration of the sentence algebra, retain the sentence-to-clopen marking, and say which interpretation maps preserve the filtration up to a controlled distortion. For a chosen infinite Dirac operator, self-adjointness on a stated domain, compact resolvent, and bounded commutators require proofs. The finite block construction makes none of those infinite-dimensional claims.

## A precise operator obstruction from incompleteness

Let X_T be the compact completion space of a consistent c.e. extension of Q. For every sentence phi, let p_phi = 1[U_phi] in C(X_T). In any faithful representation of C(X_T),

    ||1-p_phi|| = 0  iff T entails phi;
    ||1-p_phi|| = 1  otherwise.

A uniform norm approximation with absolute error strictly below 1/2, taking sentence codes as projection names, would therefore decide theoremhood. Essential undecidability forbids such a procedure. This written reduction concerns the marked projection algebra. It does not yet encode theoremhood in the eigenvalues of a canonical Dirac operator.

Faithful representation is compatible with failure of effective norm computation. An eventual computable spectral presentation must state what its names encode, which operations are computable, and where the undecidability resides.

## Research target

Construct an interpretation-compatible marked spectral representation of a useful class of finitely axiomatized theory networks. Establish a recovery theorem and controlled dependence on filtration changes. Then reduce a logical decision problem to a precisely defined operator observable with a computable map on finite input descriptions.

The symmetry obstruction in `Neoplatonic/LTR/MetricObstruction.lean` rules out demanding that a nondiscrete homogeneous completion space's metric be invariant under every homeomorphism. The admissible equivalences must preserve additional structure.

## Run

    python3 -m unittest discover -s research/logos_topos_spectral -p 'test_*.py' -v
    python3 research/logos_topos_spectral/spectral.py

No theological predicates enter the construction or its tests.


## v0.6: sentence projections and exact Dirac commutator norm

The companion files nullary_projections.py and test_nullary_projections.py
audit logical projection observables against finite Dirac blocks, with
all values represented by fractions.Fraction.

For a finite model space X and a Boolean truth locus U, its projection
p_U is the 0/1 indicator, p_U^2=p_U, and the uniform norm satisfies

    norm(p_U) = 0 if U is empty, and 1 otherwise.

The norm of the refutation projection is zero iff the underlying
sentence holds in every listed model. Across all 256 Boolean truth
tables on an eight-state space, exact tests verify the full norm gap.

For any unequal states x,y and mass m, the two-point Dirac block

    D_xy = [[m, 1/d(x,y)], [1/d(x,y), -m]]

has a commutator with diag(p_U(x),p_U(y)) of norm

    abs(p_U(x)-p_U(y)) / d(x,y).

The commutator vanishes iff the states agree on that sentence. This
has been checked exhaustively across all pairs of eight states and
all 256 Boolean truth tables.

For all 32 five-bit states and each atomic predicate P_n, the exact
finite Dirac direct-sum commutator seminorm is 2^n, with the same
answer for mass zero and mass eight. The resulting implementation
is a finite version of the Lean atomic-filtration theorem in
OntologicalProofs PR #5.

**Verification:** the expanded suite passed 30 tests in
[GitHub Actions run 41571682988](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41571682988).

Mathlib kernel proofs for the continuous function algebra, norm gap,
first-order syntactic pullback and the 2^n atomic grade are in
OntologicalProofs. This exact finite prototype supplies concrete
operator-norm audits. It is separate from the infinite direct-sum
operator-domain formalization, which remains open.
