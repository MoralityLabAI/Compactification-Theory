# FC-06 post-hoc structural shortcut: zero-neural backward goal slicing

**Analyzed November 13, 2026, after the first FC-06 held-out neural run.**
This analysis was absent from the preregistered control set. It is a
stronger counter-control, and the original first neural test, raw
predictions and scores remain unchanged.

## Exact syntactic theorem

The generated source has one initial fact F (spine atom 0), one goal
spine of length d, and rules that derive each spine atom i=1..d
from its predecessor i-1. All **additional** producers in the source
conclude a decoy atom, never a spine atom. The unreachable guard
has only a self-loop as a head producer. The negative version adds
that guard as an indispensable premise of one spine rule.

Given only the *syntactic* rule list, compute the backward ancestry
set B by starting with the goal and repeatedly adding all premises
of every rule whose conclusion is already in B. This requires a
finite graph traversal, O(V+E) with an adjacency index. For a
positive source graph, the only rules concluding an ancestor of the
goal are the actual goal-spine rules. Thus

```
B = {spine atom 0, spine atom 1, ..., spine atom d}.
```

No distractor is in B. A deterministic policy that takes a currently
legal novelty rule **only if its consequent lies in B** therefore
must take the unique next backbone edge. From the initial fact,
it reaches the goal in exactly d≤5 actions. This argument
holds independently of atom renaming and rule permutation.

For a negative source graph, B includes the unreachable guard
due to its extra premise. No legal novelty rule derives that guard
because its only producer is the self-loop. The policy cannot
complete a false goal; the independent verifier rejects it.

The graph predicate is a syntactic reverse slice: it is not
a general formal proof oracle. If decoy branches could reach
goal ancestors through additional alternative producers, this
simple slice would include irrelevant rules and a five-action
budget could again become challenging.

## Executable post-hoc control

The [posthoc_backward_slice.py](posthoc_backward_slice.py)
implementation reads only original atoms, goal, facts and rules,
builds reverse goal ancestry, and selects the first applicable
novel-consequent rule within that ancestry. It receives no
source gold labels or original minimal proof traces. Every
output passes through the independent FC-05 trace verifier,
and accepted traces are rendered as explicit Lean source.

[Post-hoc CI run #43416698132](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43416698132)
passed all four regression tests and recompiled **all 192
positive held-out proofs** under Lean 4.19.0, with zero
missed positives or false formal certificates. Both
inspected endpoint theorem declarations reported no
additional Lean axioms.

| Policy | Trainable parameters | Held-out valid proofs | Neural cell calls |
| --- | ---: | ---: | ---: |
| FC-06 learned LR-01 actor | 16,226 plus separate gate | 192/192 per seed | 2,112 shallow, 2,784 deep |
| Backward goal-slice graph rule | **0** | **192/192** | **0** |
| Preregistered random legal | 0 | 6/192 | 0 action-scoring cells |
| Preregistered goal-first | 0 | 30/192 | 0 action-scoring cells |

The learned student met both preregistered criteria against the
two frozen heuristic controls. This new control demonstrates
that its **exact proof-selection capability is not superior to a
simple symbolic graph algorithm** on FC-06's restricted family.
The actor's learned ranking could be implementing a useful
approximation of reverse goal reachability. This experiment
does not isolate that mechanism from other graph cues.

## Consequence for a future FC-07

A meaningful follow-up must introduce alternative rules that
can conclude goal ancestors, decoy subgoals with false prospects,
shared dependencies, and legal cycles. Its exact minimal
proof must remain within a fixed strict budget, while the naive
backward-ancestry set includes more legal alternatives than
the model can afford. The stronger static backward-slice
baseline must be **preregistered** and evaluated alongside
the random and goal-first policies. A learned advantage requires
measured certificate recall or cost better than this
syntactic control, while preserving the dual exact verification
gate and unseen provenance families.

A raw manuscript still requires a verified mathematical
encoding before any finite graph result can certify it.
