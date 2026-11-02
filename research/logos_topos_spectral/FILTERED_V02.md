# Filtered spectral reconstruction, v0.2

The new `filtered.py` and `test_filtered.py` implement marked finite spaces, completion maps, and an exact semantic depth profile. The mathematical statement and infinite-dimensional proof are in [OntologicalProofs, FILTERED_SPECTRAL.md](https://github.com/patrickdugan/OntologicalProofs/blob/research/ltr-reconstruction-2026-11-02/research/logos_topos/FILTERED_SPECTRAL.md).

## Main identity

Let h_f(n) be the least number of source observations needed to determine the first n target observations. With distance 2^(-k) at first disagreement k, the exact finite identity is

    Lip(f) = max_{h_f(n)>0} 2^(h_f(n)-n).

The empty maximum is zero. This handles constant maps correctly. A delay c means h_f(n) <= n+c for every n. Equivalently, Lip(f) <= 2^c. Delays add under composition, while optimal Lipschitz bounds multiply submultiplicatively. The optimal bound for pulling back unit-Lipschitz observables is the same Lip(f).

## Local checks actually run

Thirteen new tests passed locally. They exhaust all 81 maps from a four-state marked source into a three-state marked target, including nonsurjective and constant maps. Composition checks cover all 648 pairs with maps onward to a two-state target. Distance-to-target observables attain the computed optimal pullback bound. The six previous logical-network tests remain in OntologicalProofs; this repository retains its eight earlier spectral tests as well.

A sharp fixture drops the first two bits of all 32 five-bit source states. The eight three-bit target states all extend. Its depth profile is (0,3,4,5), its least delay is 2, and both its distance expansion and its optimal observable pullback bound equal 4.

## Infinite-dimensional support

The companion note uses Christensen--Ivan, *Sums of two-dimensional spectral triples*, Theorem 3.1, arXiv:math/0601024, for the established infinite-dimensional construction. On a countable dense list of unequal pairs it uses blocks

    D_j = [[2^j, 1/d_j], [1/d_j, -2^j]].

Their commutators agree with those of the mass-free blocks. The squared eigenvalues are 4^j + d_j^(-2), giving the tail estimate

    ||(1-P_N)(1+D^2)^(-1)|| <= 1/(1+4^N).

At N=8 this is 1/65537. Python checks exact finite block identities and this bound against sampled finite blocks. It does not compute an infinite arithmetic completion or certify full infinite operator analysis.

## Scope

The dyadic distance here is 2^(-k), matching mathlib PiNat and starting at distance 1. The v0.1 `spectral.py` uses weights 2^(-k-1), so its distances are uniformly half as large. All map Lipschitz ratios and delay results agree between these conventions; absolute eigenvalues scale accordingly. The two APIs retain their explicit conventions.

The construction is functorial on marked observable algebras with controlled Lipschitz pullbacks. It does not supply canonical Hilbert-space intertwiners or a spectrum invariant under arbitrary changes of presentation. The mass terms alone change the spectrum without changing the commutator seminorm. The specific interpretation-compatible theorem combines established geometry with an explicit logical filtration; novelty has not been established by literature review.

Run:

    python3 -m unittest discover -s research/logos_topos_spectral -p 'test_*.py' -v
    python3 research/logos_topos_spectral/filtered.py
