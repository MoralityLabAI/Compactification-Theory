# Strict-prefix recurrence, nilpotence and certified memory compaction

Development mathematical note, 2026-11-26. Motivation: [Mixture-of-Loops Transformer](https://arxiv.org/abs/2611.34923).
Companion branch `research/molt-goblin-01-20261126` contains an exact scalar fixture in HybridTRMLDT,
a neural pilot in LoopedTransformers and generated array tasks in DataGoblin. The proofs below are
handwritten. Lean kernel checks and learned-compaction evidence remain future work.

## Mathematical object and implementation contract

The object is a finite dependency DAG with a fixed deterministic local transition at each node.
An evaluated epoch fixes the input, parameters, public LDT state and every other external quantity
read by those transitions. Nodes can represent sequence positions or a finer computation graph.

For a prefix-memory block, the graph must include the inputs used by its gate, retrieval scores,
normalization and output write. State-dependent donor selection is allowed when every possible
dependency belongs to one fixed acyclic superset. Counting only the finally selected edges can
understate the true depth: a selector that inspected all earlier positions depends on all of them.
The fixed null-memory slot supplies the empty-prefix case.

The theorem uses a homogeneous recurrence starting from `z^0`. Its depth count begins with the
first application of the same map `F`. The motivating paper has a special first loop before its
repeated transition; its loop indices require that extra initialization offset. Apply the bound
to the implemented transition and its declared clock. Current-loop computation inside a node
can be incorporated into its local map, provided previous-loop self-state is absent.

## General DAG stabilization

Let `G` be a finite DAG in topological order `0,...,T-1`, with `T >= 1`. Let `P(i)` be the
fixed dependency superset of node `i`. Each coordinate space `X_i` is nonempty. Define

```text
F_i : product_{j in P(i)} X_j -> X_i
z_i^(r+1) = F_i((z_j^r)_{j in P(i)})
d_i = 1 + max_{j in P(i)} d_j       (empty max = 0)
D = max_i d_i
```

The unique fixed point is obtained recursively in topological order:

```text
z_i* = F_i((z_j*)_{j in P(i)})
```

**Proof.** A source has no variable arguments, so its first update equals its unique fixed state.
Suppose the parents of `i` have reached their fixed states by `max_j d_j`. The next update of `i`
then equals `z_i*`, and subsequent updates preserve that value. Topological induction proves
`z_i^r = z_i*` for every `r >= d_i`. Applying the same induction to any candidate fixed point
establishes uniqueness. Thus every initialization stabilizes within `D <= T` updates.

The bound is tight for a longest chain whose source becomes a constant and whose other maps copy
their predecessor. Initialize the chain to a different value. The new source value advances by
one edge per synchronous update, reaching the final node on update `D`.

For example, with `P(0)={}`, `P(1)={0}`, `P(2)={0}` and `P(3)={1,2}`, the certified depths are
`(1,2,2,3)`. Three parallel updates suffice from arbitrary initialization. Topological evaluation
computes the same fixed state by evaluating each local map once, after its predecessors.

### Scope checks

| Implementation choice | Mathematical consequence |
| --- | --- |
| A fixed gate reads only declared predecessor states | Included in `F_i`; the same depth proof applies |
| Gate scores read every earlier state before selecting a sparse set | The certificate uses the full score-dependency graph |
| A position reads its own previous-loop state | A self-edge leaves the DAG theorem's scope |
| Forbidden tokens enter an unmasked softmax denominator | They are real dependencies even when their values are later zeroed |
| Random masks or cache approximations change each pass | The fixed-map assumption must be replaced by a separate analysis |
| Public lattice state or weights change | Begin a new epoch and recompute the affected dependency closure |

The scalar fixture certifies its finite exact arithmetic domain. Neural code also needs numerical
parity checks between stabilized parallel and topological evaluation. Different reduction orders
can produce floating-point differences; tolerances belong to the numerical receipt.

## Linearization and transient transport

Assume finite-dimensional coordinate spaces and differentiable local maps. Then `J(z)=DF(z)` is
strictly block lower triangular, and block `(i,j)` vanishes whenever `j` is outside `P(i)`.
Expanding a product of `k` such Jacobians, even at different evaluation points, shows that block
`(i,j)` can be nonzero only along a dependency path of exactly `k` edges. No path has `D` edges,
so every product of `D` Jacobians vanishes. At a fixed point, `J^D=0` and its spectral radius is zero.

Transient gains can still be large. A two-node chain with one scalar edge of weight `M` has
`||J||=|M|`. Longer chains can transport factors of `M^k` before the product vanishes. BPD transport
measurements therefore need actual state displacement and numerical error controls even when
finite stabilization holds.

This nilpotence concerns derivatives with respect to the recurrent state. Parameter derivatives
include direct dependence of every local map on shared weights. They can remain substantial.
When all relevant maps are differentiable and the forward graph is unchanged, a full unroll through
the stabilization depth and the topological computation represent the same input-output function.
Gradient parity then requires preserving those derivatives. Detaching carried activations or
stopping before depth `D` changes the training calculation and needs its own recorded recipe.

## Incremental recomputation across epochs

Suppose a set `C` of local maps or fixed inputs changes while all other dependencies remain fixed.
Let `A` contain `C` and every descendant of `C`. Nodes outside `A` keep their old fixed states:
their maps and every predecessor outside the changed closure are unchanged, by topological induction.

An exact incremental evaluator can retain clean boundary states and recompute affected nodes in
topological order. A synchronous affected-only evaluator uses the longest path through the affected
subgraph as a conservative number of updates, treating retained boundary values as fixed inputs.
It must read previous-pass affected values consistently. Mixing updates within a pass defines a
different schedule whose result needs its own argument.

When edges change, include every node whose parent set or local map changed in `C`, then use the
descendant closure in the union of the old and new graphs. The union gives a conservative influence
set even when its edges no longer share one topological order; actual recomputation uses the new
DAG. A global parameter update can put every node in `C`.

| Cache field | Proof obligation |
| --- | --- |
| `frame_id`, `problem_version`, `epoch_id`, `public_state_digest` | Premises and external context used by the local maps are fixed |
| `checkpoint_digest`, `map_version` | The relevant parameters and local function agree |
| `parent_ids`, `dependency_digest` | Every input capable of changing the node is covered |
| `arithmetic_mode`, `value_digest` | Exact-domain or numerical replay convention is explicit |
| `certificate_id` | Any certified readout can be checked against the same context |

A cache lookup is exact only when these dependencies are unchanged or a proof establishes that
the changes cannot affect the value. A TTL or a similar embedding is a heuristic reuse rule and
belongs to an approximate-cache arm. A verified LDT refinement creates a new epoch; the bound
starts again for the affected computation under its new fixed context.
If a premise is retracted or the candidate universe changes, the public-state system must
reinitialize a new environment-owned frame from valid assumptions. The recurrence theorem
does not restore candidates eliminated by an earlier lattice update. Preserve the earlier
frame as evidence and maintain common root lineage across the revisions.

## Exact memory quotients

Let `pi_i : X_i -> Y_i` be a proposed compact representation. Exact compaction needs local maps
`barF_i` such that, on a stated invariant domain containing the initial states and every intermediate
state used by the computation,

```text
pi_i(F_i(z_parents)) = barF_i((pi_j(z_j))_{j in P(i)})
```

Every certified readout `A` must also factor through the quotient:

```text
A(z) = barA(pi(z))
```

Start the compact recurrence at `y^0 = pi(z^0)`. Induction on the pass number gives
`y^r = pi(z^r)`, and the readout equation then preserves each certified answer. The compact local
maps still use the declared DAG, so the same depth proof applies. This establishes a commutation
property of the computation and its observations.

An equivalent construction criterion is that each `F_i` respects the product of the equivalence
relations induced by the parent quotients, while `A` is constant on the relevant quotient classes.
Small reconstruction error on exposed examples supplies an empirical approximation measurement;
the universal commutation obligation requires a separate proof or finite-domain check.

### Approximate compaction

Suppose a reconstruction map `rho_i` exists and
`||F_i(rho_parents(y)) - rho_i(barF_i(y))|| <= epsilon_i` uniformly on the certified domain.
If the original local map satisfies parentwise Lipschitz bounds `L_ij`, an envelope for the
reconstruction error at the stabilized states obeys

```text
e_i <= epsilon_i + sum_{j in P(i)} L_ij e_j
```

The finite DAG makes this a finite propagation bound even when some `L_ij > 1`. Large path gains
can still make the resulting envelope unusable. To preserve a discrete winner under a readout
score error of at most `eta` per candidate, the winning score margin must exceed `2 eta`.
Other predicates require their own margin or exact verifier. Retain the certified full state
whenever a required decision lacks such a certificate.

## Proof and implementation milestones

| Milestone | Output | Acceptance condition |
| --- | --- | --- |
| Finite witness | Exhaustive small-DAG fixtures and explicit invalid graph cases | Parallel/topological equality and correct depth bounds on the declared domain |
| DAG theorem | Lean statements for existence, uniqueness and finite stabilization | Actual kernel replay with pinned toolchain and dependency versions |
| Invalidation theorem | Descendant-closure lemma and incremental evaluator | Recomputed fixed states agree with complete recomputation after declared local edits |
| Quotient theorem | Local congruence and readout-factorization lemmas | Kernel-checked preservation for a concrete compact representation |
| Numerical bridge | Neural parity, cache-cost and readout-error receipts | Declared tolerances hold under the measured resource envelope |

Lean development should first formalize a finite node type with a well-founded predecessor relation
and local product-state maps. Existence comes from well-founded recursion; stabilization follows
by induction on graph depth. Keep the Jacobian path-support theorem as a separate finite-dimensional
result, so the basic stabilization proof remains independent of differentiability.

DataGoblin can turn graph changes, invalidation sets and quotient checks into executable training
tasks. LBF may receive the corresponding proof obligations after their actual Lean statements and
kernel checks exist. HybridTRMLDT owns the epoch and membrane interfaces; LoopedTransformers supplies
the neural forward and gradient comparisons. These handoffs preserve the distinction between exact
finite evidence and measurements of trained systems.
