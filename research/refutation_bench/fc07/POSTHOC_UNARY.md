# FC-07 post-hoc shortcut: unary-implication path search

**Added November 13, 2026, after FC-07's first preregistered
model comparison.** The new control is separate from the frozen
test and cannot be counted as an originally registered comparator.
It explains a remaining simple structural shortcut in the
generated Horn graphs.

## Why it works

Every source graph contains a canonical path of unary rules

```text
P_0 -> P_1 -> ... -> P_d,
```

where `P_0` is the initial fact, `P_d` is the goal and
`d <= 5`. The generator adds alternative rules whose
consequents may be real goal-chain atoms, making the
*ordinary* backward ancestry set contain all reachable
distractors. Yet every such alternate goal producer
has **at least two premises**:

```text
(P_(i-1) AND distractor_j) -> P_i.
```

Some alternatives require the unreachable guard as
a third premise. Every extra one-premise decoy/cycle
rule concludes a **decoy** or the unreachable guard,
never a goal-backbone atom.

Now restrict the source graph to rules with exactly
one antecedent. Each is an ordinary directed edge.
Perform breadth-first search from the given fact to
the goal, recording the original rule indices for
the predecessor chain.

For every *positive* task, the canonical backbone
is an available unary path, so this restricted
search finds a legal trace of at most d<=5 actions.
It follows the original source indices despite
their shuffled order and randomly renamed atoms.

For every *negative* task, the source makes one
essential canonical edge conjunctive by adding
the unreachable guard. All alternate producers
of that backbone atom also require the guard.
No other one-premise rule can derive that atom.
Therefore the unary subgraph cannot contain
a path to the goal, and no positive certificate
is released.

The algorithm uses only the supplied facts,
rule premises, rule heads and goal. Its runtime
with a precomputed adjacency list is
`O(|atoms| + |rules|)`; it needs **zero trainable
parameters** and **zero recurrent neural cells**.
The independent FC-05 trace checker still
replays the exact proposed rule sequence,
and Lean 4.19.0 compilation remains the
certificate authority.

## Meaning for FC-07

The preregistered FC-07 experiment did find
learned rule selection that beat the **registered
full-backward-ancestry** baseline, which
succeeded on 47/96 same-depth and 22/96
deeper valid cases before the model test.
The learned LR-01 actor scored
94/96, 85/96, 93/96 and 80/96 across
two seeds and both test-depth groups,
kernel-compiling 352 model-selected positive
proofs and issuing no false finite
certificates. Its CPU-cost criterion already
failed at familiar depth.

The new unary-path control exposes the
limitation of that result: a *different,
simpler symbolic graph operation* can
obtain the exact goal path on the same
restricted grammar without any learned
rule ranking. This is an additional
post-hoc audit, so it does not rewrite
the registered first-run criterion or
its originally frozen test scores.

## Next design gate

To assess learned proof search more
seriously, a newly registered curriculum
should remove the guaranteed unary goal
path, introduce genuine conjunctive
requirements on all goal-producing rules,
and allow mutually exclusive cheap-looking
alternatives with shared subgoals. Exact
bounded shortest-proof BFS remains the
correct symbolic upper bound. Include
unary-path BFS, backward ancestry, and
a predeclared compute-limited symbolic
best-first policy as baselines.

Such an evaluation can test a neural
model's ability to choose among
competing multi-premise derivations
under a strict action budget. It
remains a finite Horn surrogate until
a verified parser connects these
arguments to real mathematical texts.
