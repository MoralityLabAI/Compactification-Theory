# FC-11: actual MathCan multi-step Lean tactic search

[FC-11 preregistration](PREREGISTRATION.md) was committed
as `b57e2500e829b12fdfc257077b8e410171c93b98`
before importing training trajectories, building the
scratch evaluator, or scoring an LR-01 model.

FC-10 produced a real Lean tactic-state pipeline for
generated Nat/Prop/List goals, with a negative
out-of-family model result. FC-11 moves to **original
research proofs** in `patrickdugan/MathCan`, using the
exact MathCan source revision and Lean version.

## Original MathCan source, frozen selection

The original repo is pinned at commit
`c84e2025334e9a967c1297425a490f98d5b3d189`,
which declares Lean **4.35.0**. All original proof
scripts remain unchanged in MathCan.

| Data | Modules | Theorems | Real Lean tactic-stage states |
| --- | --- | ---: | ---: |
| Train | FiniteHorizonGap, TOECosmology, TOEGraph, DeletionMemory | 16 | **31** |
| Held-out | Intervention, HookState, GraphAuthorization | 14 | **29** |

The train-selection revision, which precedes
the held-out selector, is
`2e7e749d4c7f8610a4083f4f8551d084102ea55a`.
It lacks `test_data.py`. The independently
certified held-out selector was frozen at
`b821e0d85423606cf971b1f80db8092ede693d81`.

Canonical real-state JSONL hashes:

- Train: `47fae96eac1f9b176519faaa0a64cbf016203b6f23ba56e40a2295e9fedaa97d`
- Held-out: `a8e7bee63f80c03c640ccdbb9bada1f8d943e33fb403e37ad1fc7dbf536e11ad`

[Original train-source CI #43540113159](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43540113159)
passed six tests, kernel-compiled the original
selected proof bodies and extracted exactly 31
compiler-printed intermediate tactic states.
[Held-out source CI #43540299693](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43540299693)
passed five tests and compiled the selected
original source files, extracting 29 held-out
intermediate states.

## Actual tactic trajectories

`source_core.py` inserts Lean `trace_state`
immediately before each original **top-level**
tactic in an instrumented source copy, with
unique theorem/step markers. It preserves
the source telescope, namespace, declarations,
earlier definitions and reference tactic
body. The parser accepts only a matching
compiler-emitted marker and a real goal
turnstile. Multiple simultaneous goals are
recorded explicitly for later unsupported-goal
accounting.

The model sees the printed goal-state text.
The original reference proof script is retained
solely for source certification and train-only
prefix alignment.

## Eight registered tactic actions

`omega`, `decide`, `rfl`, `simp_all`,
`aesop`, `assumption`, `intro`,
`constructor`.

`scratch.py` compiles a source-specific
copy of an original theorem with a candidate
action. A scratch run may append `sorry`
to expose the *next* Lean goal for another
action choice. These preview runs are
**not** theorem certificates. A released
proof requires a **separate compiler run
without any `sorry`**, with all proposed
actions from the model, in sequence.

The train-only probe program attempts every
one of these eight candidates on every
training compiler state, using the original
source prefix as the train-only alignment.
Each action therefore receives positive
or negative training evidence from actual
Lean 4.35 tactic application. A failed
candidate has no positive source label.

## Learner and strong symbolic control

The [stacked LoopedTransformers FC-11 policy](https://github.com/MoralityLabAI/LoopedTransformers/tree/feat/lr01-fc11-imported-tactic-search/experiments/refutation_transfer)
uses the original LR-01 `TypedLoop` cell
and the original **62,401-parameter**
token/action scorer configuration, with
eight active candidate tactic nodes.
Two source-trained seeds (17, 29),
four tied recurrence passes, 300
AdamW updates and batch16 use the
real train-only tactic-effect outcomes.

On each held-out original theorem the
model observes the initial **actual Lean
goal state**, chooses a tactic, and
kernel-compiles that proposed prefix.
An unfinished valid prefix may be
recompiled as an explicitly untrusted
scratch state to obtain a new Lean
goal for the next model choice.
After at most three model actions,
a positive release requires a
complete Lean4.35 proof.

The source-only Lean-native control
tries the same registered tactic scripts
with `first | ...` over up to
three goal transitions, using a
per-theorem measured time allowance
from the model's own forward/scratch
tactic workload. Every baseline
theorem must also compile without
`sorryAx`. The report records
kernel success, rejected actions,
multigoal abstentions, start-up
cost, and deadline overruns.

The train selector is frozen before
test selection, but the upstream
MathCan source is public and its
original theorems are accessible
in the pinned repository. The
access boundary is a documented
**train-only selector**, rather
than a private blind test.

## Reproduce source audit

From Compactification-Theory root
with the pinned MathCan checkout
and original Lean 4.35 executable:

~~~sh
python -m research.refutation_bench.fc11.train_data \
  --source ../MathCan-pinned --out /tmp/fc11_train
for x in /tmp/fc11_train/*.lean; do
  lean "$x" > "${x%.lean}.log" 2>&1
done
python -m research.refutation_bench.fc11.train_data \
  --out /tmp/fc11_train --extract
~~~

A similar `test_data` invocation
certifies the fourteen held-out
theorems separately. Complete
source manifests and raw trace
logs are GitHub Actions artifacts.

The first FC-11 results concern
a small set of existing Lean
research lemmas, not the correctness
of any unformalized manuscript.
