# OIC-33: Mandatory-address constraints for fault-tolerant active sensing

**Status, November 11, 2026:** proved necessary action restriction for the
finite cyclic-monitor game, independent small-state tests, exhaustive n=7
Q=14 UNSAT replay and n=7 Q=20 independently checked policy. The n=7 Q=19
minimum remains unresolved. This is a specialized finite-state search cut,
not a theorem about pretrained models or intrinsic consciousness.

## Game and belief state

The hidden world is `w=(d,z)` with `d,z` in Z_n. At time `t`, a marker query
`m` produces clean bit `b_t(w,m)=[z+t*d=m (mod n)]`. One adversarial bit
substitution is allowed over the episode. Given a common observed history,
the belief state is `(M0,M1)`, where `M0` contains worlds with no substitutions
used and `M1` contains worlds with exactly one. The sets are disjoint.

The OIC-T44 pairwise residual-disagreement obstruction used the following
state-independent capacity for worlds with different shifts:

\[
D_{ab}(t,Q)=\sum_{s=t}^{Q-1}
       [z_a+s d_a\not\equiv z_b+s d_b\pmod n].
\]

Let `r_a=1-e_a` and `r_b=1-e_b` be their remaining substitution budgets,
where `e_w=0` for a candidate in `M0` and `e_w=1` for one in `M1`.
OIC-T44 proves the node is losing if `D_ab <= r_a+r_b`.

## OIC-T70: Mandatory-address lemma

**Theorem.** Suppose `a,b` have distinct shifts, are both compatible with
one reported history, and

\[
D_{ab}(t,Q)=r_a+r_b+1.
\]

If their present phases `p_a=z_a+t d_a` and `p_b=z_b+t d_b` are distinct,
then any winning policy must choose

\[
\boxed{m_t\in\{p_a,p_b\}.}
\]

**Proof.** A marker outside both positions returns the *same* clean value
zero in both worlds. The adversary can choose the common report zero without
spending any of the remaining substitutions. Both candidates survive this
reported branch with unchanged individual fault budgets. Because their
current phases differ, the remaining number of time steps at which their
hidden phases could be distinguished decreases by one. Consequently
`D_ab(t+1,Q)=r_a+r_b`. The original OIC-T44 criterion gives an
indistinguishable future receipt, regardless of the controller's subsequent
adaptations. Hence a policy choosing this address fails on an admissible
branch. QED.

At a given information state, **every** tight pair shares the same current
public history, so its required address restriction applies to the same
next sensor choice. Define

\[
A_t=\bigcap_{\substack{a,b: d_a\ne d_b\\
D_{ab}=r_a+r_b+1\\p_a\ne p_b}}
\{p_a,p_b\}.
\]

The empty intersection over no qualifying pairs is all of `Z_n`.

**Corollary (multihypothesis intersection cut).** If `A_t` is empty,
no guaranteed correct continuation exists. For a nonempty `A_t`, a complete
search may restrict candidate addresses to `A_t` without losing any winning
policy. This cut combines multiple distinct hidden-world pairs at a shared
sensor decision. It is a *necessary* test; nonempty `A_t` is not sufficient
for success.

## OIC-T71: Affine gauge equivariance

For prime n and unit `u`, relabel worlds and queried markers under

\[
(d,z,m)\mapsto(ud,uz+c,um+c).
\]

This transformation preserves phase equality, the number of future phase
disagreements, the fault layer of each candidate and the terminal criterion
that exactly one shift remain. It sends the permitted set `A_t` to `uA_t+c`.
Therefore OIC-T70/T71 can be combined soundly with OIC-28's 42-element
seven-phase affine memo quotient, provided query addresses are transformed
back during policy extraction. QED.

## Computational checks and evidence boundary

The independent standard-library reference `oic33_tight_reference.py`
recomputes future phase-disagreement counts, mandatory-address intersections
and the original finite Bellman recurrence for n=2–4. Four tests compare
original and filtered decisions, inspect small-state forbidden-action
witnesses, and verify affine equivariance. They run under both ordinary and
optimized Python. The n=7 C++ code precomputes the finite pair masks.

With the same action heuristic and existing OIC-T44 cut, the new solver
reduces the n=7 Q=14 complete UNSAT run from 6,719,523 visited states to
1,673,753. It reduces the Q=20 SAT run from 1,094,946 to 270,328.
The Q=20 decision tree is independently checked against every one of the
49 hidden worlds and each of its 21 allowable zero-or-one-flip placements.
The historical Q=14 rejection and Q=20 success remain unchanged.

**Open:** OIC-33 does not establish whether Q=19 succeeds or fails.
The enlarged search exceeded its explicit state or wall limit. A cap is
an engineering boundary, never a mathematical UNSAT certificate.
