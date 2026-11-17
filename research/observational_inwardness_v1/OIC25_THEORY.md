# OIC-25: A pairwise adversarial pruning theorem and the five-phase query threshold

**11 November 2026.** Evidence: exact finite enumeration by two implementations, an exhaustive negative search without the new pruning rule, and a complete constructive decision tree checked by an independent simulator. The finite result is **computer-assisted**, not a Lean proof or an analytic formula for arbitrary cycle size.

## The game

Fix an integer \(n\ge2\). The hidden world is \(w=(d,z_0)\in\mathbb Z_n^2\). At query time \(t\), a controller selects one phase address \(m_t\in\mathbb Z_n\) and receives a binary report whose clean value is

\[
 h_t(w,m_t)=[z_0+t d\equiv m_t\pmod n].
\]

It knows \(n\) but not \(d\) or \(z_0\), and it must identify \(d\). It may choose the next address after observing earlier reports. At most one report may be substituted by its complementary bit. Each query costs one returned bit and one addressed sensor operation. A direct binary address code costs \(\lceil\log_2 n\rceil\) additional transmitted bits per online query.

For a fixed received history define \(M_0\) to be the candidate worlds requiring zero substitutions to explain it and \(M_1\) the worlds requiring exactly one. If \(C_y(t,m)\) is the set of worlds whose clean report for query \(m\) is \(y\), then

\[
 M_0'=M_0\cap C_y, \qquad
 M_1'=(M_1\cap C_y)\cup (M_0\cap C_{1-y}).
\]

A decision leaf is valid exactly when \(M_0\cup M_1\) contains candidates with one common shift. The exact feasibility recursion is the AND over all possible reports nested inside an OR over sensor addresses. Impossibility is therefore certified by evaluating the complete Boolean finite game.

## OIC-T44: Pairwise residual-disagreement obstruction

Suppose two worlds \(w=(d,z_0)\) and \(w'=(d',z'_0)\), with \(d\ne d'\), remain consistent with the *same* received history, having used respectively \(i,j\) of their allowed \(e\) substitutions. Define

\[
 H_{w,w'}(t,Q)=\#\{s\in\{t,\ldots,Q-1\}:
 z_0+s d\not\equiv z'_0+s d'\pmod n\}.
\]

Then a sufficient condition for **the nonexistence of a guaranteed successful continuation** is

\[
\boxed{H_{w,w'}(t,Q)\le (e-i)+(e-j).}
\]

**Proof.** At any common future query address, the two clean bits can differ only if the two hidden phases occupy different positions. Hence along *every* possible future adaptive action path, their clean report strings have Hamming distance at most \(H_{w,w'}(t,Q)\). A common corrupted report can be formed by allocating these at most \(H\) disagreements between the two worlds, using at most \(e-i\) flips for the first and \(e-j\) for the second. Equal received histories force the same next sensor address under any adaptive policy. The report construction can therefore continue step by step until the horizon while retaining both distinct shifts as possible. No decoder can then guarantee the correct shift. QED.

This is a state-dependent coding obstruction, generalizing the simple rule that two binary codewords with distance at most \(2e\) have overlapping radius-\(e\) Hamming balls. It is sound to prune any finite Bellman state containing such a pair. Computing the bound depends only on the specified latent trajectories, not on the controller's preferred addresses.

## OIC-T45: Exact five-phase one-substitution query complexity

For \(n=5\), one substituted binary report, no erasure, and unknown \((d,z_0)\), the exact minimum adaptive query horizon is

\[
\boxed{Q^*_{\mathrm{flip}}(5)=14.}
\]

**Computer-assisted lower bound.** The recurrence above was exhaustively evaluated through \(Q=13\), quotienting only by the previously established simultaneous phase-translation symmetry \((d,z_0,m)\mapsto(d,z_0+c,m+c)\). Two independent implementations agree that \(Q=13\) has no winning decision tree:

- The Python implementation maintains the explicit OIC-23 candidate-mask transition, separately implements canonicalization and action ordering, and applies OIC-T44. It visits **379,887** canonical states at \(Q=13\) and returns UNSAT.
- A compiled C++ implementation visits **2,597,699** canonical states at \(Q=13\) with the pairwise obstruction **disabled**, and independently returns UNSAT. With the obstruction enabled it visits **379,889** states.

The unpruned run ensures the lower-bound result does not rely on the new obstruction. Complete search of this finite Boolean game gives \(Q^*\ge14\).

**Constructive upper bound.** At \(Q=14\), the Python and C++ implementations construct identical complete decision trees. The tree has **662 nodes**, as independently checked by OIC-23's verifier. A separate trajectory-level simulator replays **327 legal true-world/fault paths**, for all 25 hidden worlds and each possible online substitution, and verifies the correct shift at every leaf. This gives \(Q^*\le14\). Together these prove the stated exact finite result, conditional on the correctness of the independently audited executable enumerators. QED.

The earlier OIC-23 empirical pattern \(Q^*(n)=2n+2\) holds for \(n=2,3,4\), whose thresholds are respectively 6, 8 and 10. It **fails at \(n=5\)**, which requires 14 queries. OIC-24's previously inconclusive \(Q=12\) run is superseded by this search.

## Resource accounting

For the winning five-phase policy, the online communication budget under direct binary addressing is

\[
14\text{ returned sensor bits}+14\lceil\log_2 5\rceil
=14+42=\boxed{56\text{ link bits}}.
\]

The decision tree itself has 662 nodes and must also be stored, generated, or otherwise charged. The exact candidate-state solver holds two 25-bit sets (50 logical bits), while the naïve stored policy needs more space because it branches on histories. The computation cost of finding the policy and the online sensing budget are different resources. The result refers to a synthetic labeled-sensor interface and makes no claim about real neural states or intrinsic phenomenal inwardness.

## Reproduction

The source files are `oic25_independent.py` (Python reference) and `oic25_fast.cpp` (compiled, independently implemented search). Both use the same public sensor specification. `oic25_reference.py --check` verifies the saved compressed policy tree using the original OIC-23 verifier and repeats the negative Python search. A separate C++ CI step verifies \(Q=13\) with pruning disabled and reconstructs the \(Q=14\) policy. The complete canonical policy JSON has a SHA-256 pinned in `results/oic25_reference.json`.

~~~sh
python research/observational_inwardness_v1/oic25_reference.py --check
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic25_exact.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic25_exact.py' -v
g++ -std=c++17 -O3 research/observational_inwardness_v1/oic25_fast.cpp -o /tmp/oic25_fast
/tmp/oic25_fast 13 5000000 1 3000000 0
/tmp/oic25_fast 14 5000000 1 3000000 1 /tmp/oic25_n5_q14.json
~~~

The C++ arguments are: horizon, memo-node cap, heuristic mode, reserve capacity, pairwise-pruning flag, optional winning-tree path. A cap hit is explicitly **inconclusive** and exits nonzero. Running without pruning at the negative horizon supplies an independent check on the key lemma's use in the solver.
