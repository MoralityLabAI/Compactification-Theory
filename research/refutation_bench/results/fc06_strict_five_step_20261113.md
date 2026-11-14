# FC-06: first frozen strict-budget learned proof-selection result

**November 13, 2026. First preregistered neural test: completed and passed CI.**
The trained LR-01 rule actor satisfied both of the registered
learning-vs-heuristic and neural-compute-normalized criteria in both
held-out splits for two seeds. A **post-hoc stronger symbolic
backward-slice control** later exposed a simple zero-parameter
algorithm that also solves every valid test case.

## Source and leakage boundary

- Preregistration committed first at
  `81f53bcf606e45d8b3fea9acba7907641c339cab`.
- Frozen TRAIN-ONLY source checkout
  `1bf514325788b633432eb9f3e49627bd923a216e`;
  source train/dev SHA-256:
  `c841727d6a3e82c5ce5769b2b324e9bde078d15930d03fad5462292657d73ccf`.
  This revision predates the test generator; training CI asserted
  that the new test module did not exist.
- Frozen TEST source checkout
  `63c7c411e32a25b447b1ca92ae0db9e33b135d27`;
  source test SHA-256:
  `6d570d3c6b21466c8df0032cac0cab115ebf3a586e82160f18b0a96f9efb9d9e`.
  This source was accessed in a dependent job only after
  hashed model artifacts were produced.
- [Exact training source CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43414157412)
  passed seven tests and Lean-compiled 480 positive training/dev
  reference proofs.
- [Exact held-out source CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43414735943)
  passed seven tests and Lean-compiled all 192 valid test
  references; all 192 invalid test records have independently
  checked Boolean countermodels.

Every valid graph contains 12 proposition atoms, one initial
fact, an essential goal chain of d=2–5 rule applications, 10-d
reachable distractors, and an unreachable guard. The shortest
verified positive proof is at most **five** actions while all
ten novelty consequences would require ten. Train/dev groups
are separated from four unseen test families by WL graph
fingerprints, checked against earlier FC-02 through FC-05 data.

## Neural training, first frozen CI

[First full two-job model run #43415640416](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43415640416)
passed eight training-only unit tests, trained both frozen actors,
evaluated every predeclared comparator on the new test cases and
compiled **384 model-proposed Lean theorem terms** in Lean 4.19.0.
The two inspected endpoint theorems report no extra axioms.
CI retains the original per-case JSONL predictions, all failed
heuristic attempts, generated Lean and kernel receipt in artifact
`fc06-five-action-43415640416` (30 days).

The FC-04-style negative/provisional gate was an independently
frozen, **11,521-parameter** four-pass LR-01 network trained on
FC-02 only. Both seeds selected a threshold of **0.55** from
FC-02 development. A separate copied trunk and new 4,705-parameter
ranker were trained jointly on **1,149 exact FC-06 train-only
proof-prefix states** for 300 AdamW updates. The actor has
**16,226 parameters**, giving 27,747 parameters for the two
neural components together. Both seeds attained 100% on a
128-state training-prefix diagnostic; this was in-sample.

No shortest-proof oracle, gold label, minimum-step target or
test-family name entered inference. The actor ranked source
rules under a mask of currently applicable novel consequences.
Every positive claim required a bounded, independently replayed
source-rule trace, then compilation of the exact same trace
as a Lean theorem. Negative claims required complete verified
Boolean countermodels. Unsuccessful proposals abstained.

## Frozen held-out source results

Each test split contains **96 valid and 96 invalid** cases,
which both seeds evaluate separately.

| Method | Same-depth positive proofs /96 | Deep positive proofs /96 | Negatives certified /96 (both splits) |
| --- | ---: | ---: | ---: |
| **Jointly trained LR-01**, seed17 | **96** | **96** | **96** |
| **Jointly trained LR-01**, seed29 | **96** | **96** | **96** |
| Uniform random legal | 4 | 2 | 96 |
| Immediate-goal-first | 26 | 4 | 96 |
| Exact shortest-proof solver (symbolic ceiling) | 96 | 96 | 96 |

Across the two seeds there were **384** positive Lean-compiled
model proof terms and **384** exact negative countermodels.
The generated proofs include seed/case repetitions over the
same 192 underlying positive test problems. The number of
false *formal certificates* was **zero**, and the learned
method abstained on zero of the 768 seed/case decisions.

The registered learning criterion requires the trained actor
to exceed BOTH heuristic controls on each test split for
both seeds. It passed. Paired, among 96 valid examples in
each split, the learned model alone succeeded on:

| Held-out split | Learned-only vs random | Learned-only vs goal-first |
| --- | ---: | ---: |
| Same depth | 92 | 70 |
| Deeper depth | 94 | 92 |

Neither heuristic ever certified a valid case that the learned
method failed. The paired differences were the same for
both trained seeds. Every output was checked against the
source-independent formal certificate contract.

### Neural compute and certificate efficiency

The model's rule-priority head spends four recurrent cell
passes at every chosen action, while the heuristics require
no neural action-scoring calls. All methods pay the same
negative/provisional stopping-gate cost.

| Condition | Total recurrent cell invocations /192 | Positive certificates per 1,000 calls | Model + checker CPU ms/case |
| --- | ---: | ---: | ---: |
| Learned, same depth | 2,112 | **45.45** | 4.93 |
| Goal-first, same depth | 960 | 27.08 | 2.18 |
| Random legal, same depth | 960 | 4.17 | 2.18 |
| Learned, deeper | 2,784 | **34.48** | 6.54 |
| Goal-first, deeper | 1,056 | 3.79 | 2.37 |
| Random legal, deeper | 1,056 | 1.89 | 2.37 |

The registered *neural-call normalized* efficiency criterion
against these two heuristics passed for both seeds and splits.
All timings are single GitHub Actions CPU-runner measurements;
there is no independent device benchmark or FLOP-matched
model comparison here.

## Critical post-hoc stronger control

[Post-hoc graph analysis](../fc06/POSTHOC_BACKWARD_SLICE.md)
identifies a simple structural shortcut. Every rule whose
consequent lies on the goal backbone is itself a goal-spine
rule; distractor rules only conclude distractor nodes.
A syntactic backward dependency slice from the goal
therefore selects exactly the useful chain. A deterministic
**zero-parameter backward-slice** rule policy proves every
valid source graph within five steps, regardless of atom
renaming or source rule order.

[Independent post-hoc CI #43416698132](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43416698132)
passed four tests, verified **192/192** held-out positive
certificates and Lean-compiled all 192 of those simple
algorithmic proofs. It has zero neural cell calls.

This control was introduced **after** the first registered
neural evaluation; it is not included in the preregistered
success criterion or used to change the initial test data.
It narrows the interpretation: the recurrent actor learns
to approximate a useful graph-dependency procedure and
transfers across the generated structural family labels,
but **a simpler symbolic reverse-graph traversal achieves
perfect correctness at lower cost**. No genuine neural
advantage over graph algorithms is shown.

## Future experiment

The next benchmark should include decoys inside the backward
ancestry of the goal through alternative goal-producing
rules, shared premises, and misleading cycles. Some legal
goal-relevant rules must waste the strict budget. A
reverse-slice control must be preregistered alongside
random legal, immediate-goal and exact shortest-proof
oracles, with compute accounting. Only improved
certificate yield over this stronger structural
baseline would substantiate learned search selection.

All conclusions in this report concern finite synthetic
Horn entailment. The exact symbolic proof solver is already
100% accurate; human mathematical manuscripts still need
an independently verified source-to-logic interpretation.
