# FC-10: first kernel-checked REAL Lean tactic-state transfer

**Date: November 13, 2026. First preregistered model test: completed.**
[GitHub Actions #43456144563](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43456144563)
passed both train-only and held-out evaluation jobs. The preregistered
four-pass model-success criterion **FAILED** against a train-only
lexical nearest-goal control. Lean's own source-registry tactic search
proved all 48 held-out theorems.

Unlike FC-02 through FC-09, the student inputs here were **actual
Lean 4.19.0 compiler-emitted tactic goal states**, obtained by
`trace_state`, parsed directly from compiler logs and protected by
frozen SHA256 checks. The source theorems were still small *generated
Nat/Prop/List templates*, not arbitrary imported research proofs.

## Immutable protocol and independent source traces

- [Preregistration](../fc10/PREREGISTRATION.md) first committed at
  `32330e86366f3ea6378ceb1bf6a4b189610b4483`,
  before any FC-10 source theorem or model test existed.
- TRAIN/DEV-only source commit:
  `631ccbb1dd393d2608b5d2c7e154aeb3ce5099b0`.
  The test generator did **not** exist at this revision.
- [Training source CI #43454981592](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43454981592)
  passed six source tests, compiled **156** Lean reference theorems
  and extracted **132 train + 24 development** actual goal states.
  Frozen real-goal JSONL SHA256:
  `3d2c5b6055ffb0bf0118c69f9a6ea405a92a922f7f869b591e26ae5feb9cb03b`.
- TEST source commit:
  `171d23e84dfc5af3ed8723ac99f75badbaf6a619`.
  [Held-out source CI #43455352149](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43455352149)
  passed five tests and Lean-compiled **48** distinct reference
  theorems from four newly held-out source-generator families,
  each yielding one actual elaborated goal state.
  Frozen test goal-state JSONL SHA256:
  `8e1792c737c47124b8962e3377ec4be12ec91c083b9d87644e5293fc892b0eed`.
- [First complete model CI #43456144563](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43456144563)
  passed seven learner unit tests, trained four models with real
  Lean goals, SHA-froze checkpoints, and **then** mounted the
  held-out source in its dependent job. Every candidate model
  tactic was applied to the original theorem through a new
  Lean compiler invocation, with no fallback to its reference
  proof on failure.

The source action space contains exactly **12 pinned tactic
scripts**: `omega`, `rfl`, `assumption`,
`constructor <;> assumption`, conjunction/or introduction
and projections, `simp_all`, `simp [List.append_assoc]`
and `False.elim`. One reference action is known to prove
each theorem. There can be multiple valid tactics for a goal;
the test score uses actual Lean compiler acceptance,
rather than requiring the reference label to match.

The compiler requires no `sorry` or `sorryAx`. Some tactics
use Lean's normal logical principles including `propext`
and `Quot.sound`; therefore no claim of an axiom-free
foundation is made beyond ordinary trusted Lean.

## Genuine LR-01 model and training

The student directly imports the original LR-01
`TypedLoop` from `LoopedTransformers/experiments/lean_reasoner/train.py`.
A deterministic BLAKE2b hash encoder converts the compiler-emitted
goal text into up to 40 token nodes. Twelve tactic candidates become
typed rule nodes with goal/context incidence. The tied cell
runs **one or four passes**, and a small rule scoring head
ranks the candidate scripts.

- **62,401 trainable parameters**, independent of recurrence count.
- CPU PyTorch 2.5.1; two CPU threads; deterministic algorithms.
- Model seeds: **17** and **29**.
- 250 AdamW updates, batch24, lr=0.002, train only on
  132 source-train goal/reference-action pairs.
- Dev: 24 goals from 2 disjoint theorem families, scored without
  weight selection or source-test access.
- Each of the four arms obtained **0% agreement with the single
  reference tactic label on development**, which deliberately
  includes previously unseen action-template families. This
  diagnostic is not a kernel-proof success rate.
- Training cost: 0.94–0.95 seconds for one-pass models and
  2.56–2.65 seconds for four-pass models on the CI runner.

The separate **lexical baseline** uses a token-Jaccard
nearest training goal, with deterministic class-prior
tie-breaking. It is fit only on the original 132
compiler-emitted training states and is frozen
before held-out evaluation.

## First held-out, actual-kernel proof success

Every row below reports **Lean-accepted proofs out of 48**
original, distinct held-out theorem declarations.
All model choices are source-only one or three
candidate tactics, and unsuccessful selections abstain.

| Selector | One candidate /48 | At most three candidates /48 |
| --- | ---: | ---: |
| Train-only lexical nearest-goal baseline | **36** | **36** |
| LR-01, 1 pass, seed 17 | 33 | 36 |
| LR-01, 4 passes, seed 17 | 31 | 36 |
| LR-01, 1 pass, seed 29 | 35 | 36 |
| LR-01, 4 passes, seed 29 | 35 | 36 |
| Sequential full registry, 12 actions | **48** | n/a |
| Lean-native `first | ...` full registry | **48** | n/a |

The full sequential registry search tested **204**
actual tactic candidate attempts under Lean,
eventually compiling all 48 theorems.
The batch `first | ...` source was also
accepted by Lean for all 48 declarations.

**Preregistered success criterion FAILED:** neither four-pass
seed beat the lexical top-1 baseline's 36 proofs.
Longer recurrence did not improve certified
top-1 proof coverage. At three allowed attempts,
every neural arm reached the same 36/48
as the lexical baseline.

### By held-out theorem family

All four new test families have 12 examples:

| Test theorem family | Lexical top1 | LR-01 1-pass s17 | LR-01 4-pass s17 | LR-01 1-pass s29 | LR-01 4-pass s29 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nat inequality with shifted terms | 12 | 12 | 12 | 12 | 12 |
| Nat subtraction arithmetic | 12 | 12 | 12 | 12 | 12 |
| Nested conjunction projection | 12 | 9 | 7 | 11 | 11 |
| List append associativity | **0** | **0** | **0** | **0** | **0** |

All models solved 24/24 arithmetic test cases. The
largest model-specific error variation was in
the conjunction-projection family. The entire
list-associativity family failed because
its successful tactic `simp [List.append_assoc]`
was **absent from training**. No model
learned to select that unfamiliar action from
the registered vocabulary. Multi-try evaluation
did not recover any of those twelve cases.

This is an explicit negative demonstration of
**out-of-family tactic transfer**. The exact
symbolic registry still recovered all twelve.

## Actual Lean compilation cost

The one-candidate policy evaluated each test
declaration separately with a fresh Lean
compiler process, which costs about
18.0 seconds in aggregate per policy
(roughly 375 ms per attempted theorem).
The lexical baseline's
36 proved theorems cost 18.054 seconds
total; model one-pass/four-pass policies
cost 17.97–18.08 seconds in this run.

Full sequential registry search performed
204 separately compiled tactic attempts,
costing **73.47 seconds**.
By contrast, one Lean process using
Lean's own `first | tactic_0 | ...`
backtracking compiled all 48 theorems
in approximately **0.973 seconds**.

These are different execution modes.
The batched `first` baseline amortizes
Lean process startup, while the
standalone candidate checks repeatedly
pay that startup. They cannot support
a precise FLOP-matched neural/symbolic
cost claim. They do show that actual
Lean-native tactic search can be
both successful and efficient
on this small problem class.

## Independent artifact audit

The evaluation archive contains a full
`FC10_report.json`, **ten** neural
and lexical top1/top3 JSONL policy
files with 48 cases each (480
original case-policy outcomes),
a sequential 48-case symbolic result,
the `first` full-batch Lean source,
compiler messages and real held-out
goal traces. It also references
the frozen four-model weight artifact.

An independent post-run archive
audit, using the original test
metadata and source traces,
reconstructed and SHA256-matched
**all 209 unique (goal, tactic)
Lean candidate trial sources**;
it found no conflicting compiler
outcomes for any repeated candidate,
no malformed policy row, and no
checkpoint hash mismatch.
It verified that every policy
success count matches the saved
score ledger, every source test
case appears once in each file,
and the native batch contains
exactly 48 theorem declarations.

The audit source/receipt is linked
in the [FC-10 independent audit artifact](fc10_independent_audit.json).
The primary **kernel authority** remains
the successful GitHub Actions Lean4.19
compiler run, not the local
metadata consistency audit.

## Limitations and next stage

This is a genuine Lean *goal-state* and
tactic-application pipeline,
yet its theorem statements are
synthetic templates with limited
actions and only a one-tactic
proof-search horizon.
Training/held-out source families
are distinct by template name
and textual signature hash,
without proof of non-isomorphism
or semantic novelty.
Numeric irrelevant source
hypotheses introduce family-specific
contextual regularities that may
simplify classification.
The existing 132-goal training
set contains no examples of the
successful held-out list tactic.
A robust next corpus must train on
all relevant tactic types while
holding out new theorem structures.

The natural next step is to **import
actual Lean declarations from
OntologicalProofs, EvalMath,
MathCan and other pinned user
repositories**, capture
intermediate multi-step tactic
states, and train/evaluate with
independent source-family splits.
Compare against optimized Lean
symbolic automation under measured
CPU budget. Kernel compilation
remains the release gate, and
no result from these generated
templates certifies an untrusted
manuscript.
