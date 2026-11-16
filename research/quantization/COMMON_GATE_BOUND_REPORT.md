# Common Taylor gate bound and decoded spatial capture

2026-11-16. There are 23 new replayed theorem roots, bringing the research total
to 181 before the separate analytic continuation. The unchanged Hodge root is separate.

`CommonGateBound` proves the uniform constant

    A_D = 2^(k+1) D^k,   D >= 1.

An injective index layout with each coordinate degree at most k has at most one
compatible Taylor entry at each axis degree. The proof filters incompatible and
below-diagonal entries, bounds each remaining power by D^k, and applies the
hockey-stick identity and the bound on binomial coefficients. It does not assume
a matrix norm. The previous unfiltered `rowBudget` is deliberately not bounded
by this constant: its incompatible entries must first be removed.

For |v_gamma| <= C and |h| <= D, every concrete transport coordinate is bounded
by C*A_D. With C=2, all amplitude candidates use the common gate constant 2*A_D.
It does not depend on the streamed site count or the coordinate being updated.

`CommonGateCapture` uses this same constant in the actual Hamming-gated update.
Its frame, exact scheduled stream, robust sentinel stream, and exact/robust spatial
capture theorems reproduce the already-defined concrete clipped reference recurrence.
The frame width remains p+a*J with D<2^p. Layout injectivity and coordinate-degree
bounds are explicit hypotheses, rather than assumed norm bounds.

Validation: Lean 4.32.0-rc1 and preserved Lean 4.35 compilation and leanchecker
both pass. OxiLean verifies 3537 declarations, with zero unsupported
or rejected. All 3858 exported constants match native types,
values and universe parameters. Every root has the same native/exported axiom
inventory, contained in propext, Quot.sound and Classical.choice. All eight
acceptance tamper controls pass. All sources from the preceding four research
corpora are checked against their accepted hashes.

Export SHA-256: `afd5dee654f3a6dbd99050f6ce3efbd2f247a2f9189872834c4dc0c19257365b`.

Copy the included helper scripts into CAMPAIGN_WORK and run
`run_common_replay432.sh`, `check_common435.sh`, and `test_common_acceptance.py`.
See `pinned432_common/ACCEPTANCE.json` and `BUILD_RECEIPT.json` for exact evidence.

This stage proves uniform gate bounds and exact scalar circuit semantics.
The full multivariate smooth approximation and dense network resource theorems
remain open at this checkpoint.
