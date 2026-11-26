# Strict-prefix recurrence, nilpotence and certified memory compaction

Development mathematical note, 2026-11-26. Motivated by https://arxiv.org/abs/2611.34923.
Companion branch: `research/molt-goblin-01-20261126` in HybridTRMLDT
(exact scalar witness), LoopedTransformers (neural pilot) and DataGoblin (array tasks).

## General DAG stabilization

Let G be a finite directed acyclic graph in topological order 0,...,T-1. Let P(i)
be the predecessors of i. With fixed inputs and parameters, define a deterministic map

    F_i : product_{j in P(i)} X_j -> X_i
    z_i^(r+1) = F_i((z_j^r)_{j in P(i)}).

No previous self-state, pass index or mutable external input is allowed. Each X_i is
nonempty. Define d_i = 1 + max_{j in P(i)} d_j (empty max = 0), and D=max_i d_i.
The unique fixed point is obtained recursively in topological order:
z_i* = F_i((z_j*)_{j in P(i)}).

Proof: induct along the topological order. A source is constant and reaches its unique
fixed state after one update. For a non-source all parents equal their fixed states
after max parent depth, hence its next value is z_i* and stays there. Therefore for
all r>=d_i, z_i^r=z_i*. Any fixed point must have the same source coordinates and,
recursively, the same non-source coordinates. D bounds stabilization from every
initial state, and D<=T. The bound is worst-case tight for a longest chain with
constant source and identity propagation initialized to a distinct value. Parallel
and topological evaluation agree in exact arithmetic; finite precision may differ.

## Linearization and nonnormal transport

If coordinate spaces are finite-dimensional and the maps differentiable, J(z)=DF(z)
is strictly block lower triangular, with off-diagonal block (i,j) zero when j is not
a predecessor. A product of k such Jacobians (possibly at different points) has
(i,j) support only on paths of length k: expand the block product. Hence every product
of D Jacobians is zero, since no path has D edges. At a fixed point J^D=0 and rho(J)=0.
This says nothing about transient norms: for a length-2 chain a scalar edge of weight
M gives ||J||=|M|, arbitrarily large; longer chains give powers M^k. This isolates
finite memory depth from BPD transport strength and numerical precision.

## Versioned cache invalidation and exact compaction

If only a subset C of local inputs/maps changes, and graph, other maps and input
dependencies remain fixed, only nodes reachable from C (including C) can have changed
fixed states. Proof by topological induction outside that descendant closure.
If the dependency graph changes, take descendants in the union of old and new graphs
(or conservatively invalidate the entire epoch); changes to global parameters may
affect every node. Reuse only same-version or dependency-proven clean cache entries.

For a proposed quotient pi_i:X_i->Y_i, exact compaction requires maps barF_i satisfying

    pi_i(F_i(z_parents)) = barF_i((pi_j(z_j))_{j in P(i)})

on the stated reachable domain. In addition every certified readout A must factor
through the quotient: A(z)=barA(pi(z)). Then induction gives commutation of every
finite pass and the stabilized readout. Equal dimension, small reconstruction error
or matched logits on exposed data alone cannot prove this factorization. For an
approximate quotient, use separately justified local Lipschitz/error bounds and
margin certificates for every discrete decision; otherwise retain original certified
state. An epoch-changing LDT refinement invalidates affected cached state; it does
not inherit stabilization across the changed map.

## Next steps

Formalize the generic finite DAG theorem in Lean, then prove the path-support product
lemma and exact quotient commutation. Feed these as proof goals into LBF and DataGoblin
only after an actual Lean kernel check; this note is a handwritten proof, not a
machine-checked theorem. Use an exact graph witness separately from trained-model
results, and measure cache hit rate, recomputation, numerical discrepancy and verified
utility under matched resource budgets. No learned compression or causal claim here.
