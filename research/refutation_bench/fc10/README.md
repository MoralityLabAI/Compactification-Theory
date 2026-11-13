# FC-10: genuine Lean 4.19.0 goal-state and tactic-choice benchmark

FC-09's exact symbolic AND/OR planner was faster and more reliable
than the recurrent student on tiny, already formalized Horn graphs.
FC-10 moves the next stage into **actual Lean elaboration and tactic
application**. It is a deliberately small first Lean search gym, with
generated Nat/Prop/List theorem templates, distinct from the earlier
graph replay tasks.

The [preregistration](PREREGISTRATION.md) was committed as
`32330e86366f3ea6378ceb1bf6a4b189610b4483` before
generating any FC-10 theorem or training/evaluating a model.

## Compiler-origin training inputs

Every theorem starts with an explicit Lean hypothesis telescope and
goal, followed by these commands:

~~~lean
trace "FC10_BEGIN:fc10_train_0000"
trace_state
trace "FC10_END:fc10_train_0000"
-- Followed by a known valid reference tactic, compiled by Lean.
~~~

Those markers surround **Lean 4.19's actual elaborated goal-state
text**, including local hypotheses and its `⊢` turnstile.
[core.py](core.py) parses the raw compiler log and fails if any
goal is absent, duplicated or fails to contain exactly one
turnstile. The actual goal state is the only textual input
to the trained tactic-selection model. It cannot reconstruct
a goal from the source theorem title or reference answer.

## Frozen train/test provenance

| Source revision | Tasks | Canonical real goal-state JSONL SHA256 |
| --- | ---: | --- |
| Train/dev **before test existed**: `631ccbb1dd393d2608b5d2c7e154aeb3ce5099b0` | 132 train + 24 dev | `3d2c5b6055ffb0bf0118c69f9a6ea405a92a922f7f869b591e26ae5feb9cb03b` |
| Held-out: `171d23e84dfc5af3ed8723ac99f75badbaf6a619` | 48 test | `8e1792c737c47124b8962e3377ec4be12ec91c083b9d87644e5293fc892b0eed` |

[Training source CI #43454981592](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43454981592)
passed six tests and compiled all **156 original theorem declarations**,
then extracted 156 actual goal states. [Held-out source CI
#43455352149](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43455352149)
passed five tests, compiled all 48 held-out theorem references, and
extracted 48 independent compiler-origin goal states.

The tactic registry includes `omega`, `rfl`,
`assumption`, `constructor <;> assumption`,
explicit conjunction/or discharges, `simp_all`,
`simp [List.append_assoc]`, and the elimination
of False. Training families include Nat arithmetic,
definitional equality and propositional introduction/projection.
The test generator uses distinct Nat, nested conjunction and
list associativity templates. Source-signature SHA256s are distinct
across the 204 declarations, although this is **not**
an isomorphism-level novelty guarantee.

The compiler may use normal Lean logical principles
such as `propext` or `Quot.sound` for automation.
These are part of the trusted Lean setup; compiler
acceptance and absence of `sorryAx` are recorded
rather than claiming that every tactic is axiom-free.

## Exact tactic applications and models

The stacked [LoopedTransformers FC-10 PR](https://github.com/MoralityLabAI/LoopedTransformers/pull/31)
imports the **original** LR-01 `TypedLoop` cell
and ranks twelve candidate tactic scripts using the
compiler-emitted goal text. Two seeds and recurrence
depths 1/4 are trained on 132 training goals, with
24 source-dev states reserved for diagnosis.
The train-only model job explicitly asserts that
this repository revision has no `test_data.py`.

A dependent test job loads frozen checkpoint SHA256
and the later test source, regenerates real Lean goal
states, and evaluates the selected candidate by
running the **actual Lean compiler** on the
original theorem declaration with that tactic.
Failed candidate scripts are counted as failures.
The baseline includes a train-only lexical
nearest-state classifier, full sequential registry
search through actual compiler calls, and
Lean-native `first | ...` backtracking.

The benchmark is small and synthetic. Its value is
a real tactic-state input/Lean verification loop,
which can later be extended to imported theorem
obligations under explicit source revisions.
There is no claim of proof validity for
unformalized papers.
