# FC-05 post-hoc diagnosis: the proof-action budget saturates the graph

**Written 2026-11-13 after the first FC-05 held-out evaluation**, which
was already committed in the unchanged source dataset at
`e2fd9f6a0bc7466475541a7ee2afd1320b4e0366`.
This finding is not part of the preregistration and cannot be used as
confirmatory evidence for learned rule selection. The first CI result
remains preserved unchanged.

## Exact finite bound

Every FC-05 task has `depth = d`, with `2 <= d <= 8`, exactly
`|A| = d + 4` distinct proposition atoms, and one initial fact.
There is a designated guard atom G which is never a fact and which
appears as a rule head **only** in the self-dependency `G -> G`.
Therefore G cannot enter the least Horn closure `Cl(F)`, regardless
of the order in which rules are fired.

Consider any strategy that repeatedly applies an *applicable* rule
whose head is not already known. Starting from initial fact set F,
every such application adds a new member of `Cl(F)`. No strategy can
do this more than

```text
|Cl(F) \ F| <= |A| - |F| - 1
              = (d + 4) - 1 - 1
              = d + 2 <= 10.
```

After all applicable novelty actions have been exhausted, the known
set is closed under every rule. By the minimality of the Horn closure,
the final set equals `Cl(F)`. Hence if the goal is entailed, **every
legal-novel action order** derives it within the FC-05 budget of ten
rules. There is no opportunity for a model to improve the success
rate of the random or immediate-goal policies.

The proof does *not* rely on the model's output, a particular seed,
or the source's stored reference proof trace. The extra post-hoc
regression test `test_posthoc_saturation_any_legal_order_has_ten_step_budget`
also generates a deliberately reversed rule-priority trace on every
positive held-out graph, replays it through the independent verifier
and checks the same ten-action upper bound.

## What the first CI actually established

[First model CI run #43399759225](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43399759225)
passed both training and frozen evaluation, including Lean 4.19.0
kernel compilation of **318** model-proposed positive proof terms.
It did not use the reference proof terms to construct those outputs.
In all four split/seed comparisons, the learned model and both
10-action heuristic controls had **identical positive certification
coverage**, for the mathematical reason above.

| Seed | Split | Certified positive / 96 | Certified negative / 96 | Abstained |
| --- | --- | ---: | ---: | ---: |
| 17 | same depth | 96 | 96 | 0 |
| 17 | deep | 87 | 90 | 15 |
| 29 | same depth | 96 | 96 | 0 |
| 29 | deep | 39 | 87 | 66 |

The three positive-proof algorithms are all gated by the *same* FC-04
prior stopping decision, so their certified coverage is identical.
The different training seeds primarily change **whether the positive
attempt is initiated**, beyond the control of the proof-head policy.

The learned model's forward cost is higher: on seed 17, same-depth
tests required 2,790 shared-cell calls for learned rule ranking versus
990 for a random legal rule sequence, with identical certification
counts. On the deeper cases the corresponding counts were 4,610 vs
1,626. This makes the lack of learning advantage more consequential.

## Correct research claim

The architecture successfully **acquires and verifies positive
certificates**, compiling the original selected inference traces
under the Lean kernel. Negative releases remain checker-gated.
All accepted formal certificates are sound within the bounded Horn
fragment, and the first full evaluation emitted no false certificates.

The experiment supplies **no evidence that the trained proof head
selects better action sequences** than a simple legal-rule baseline,
because its maximum permitted action count makes every ordering
succeed when the source claim is entailed.

## Design requirement for a future FC-06

A new, independently preregistered test must put some reachable
distractor nodes ahead of the goal under an action budget **strictly
smaller** than the maximum reachable novelty count, while preserving
at least one short derivation for every positive example. Freeze a
fixed budget before test release, and ensure that a random/legal
baseline fails a measurable fraction of feasible positive tasks.
Train/validation must remain family-isolated and the exact checker
must retain the same no-repair contract. A meaningful trial should
compare learned proof length or success at a fixed strict budget,
with an exact shortest-derivation oracle as a labelled upper bound.

Changing the current ten-action budget or test grammar after seeing
the FC-05 results would invalidate their original registered comparison.
