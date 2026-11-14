# FC-10 preregistration: real Lean tactic-state policy transfer

**Frozen November 13, 2026, before creating any FC-10 Lean task or observing held-out scores.**

FC-09 established that a source-only AND/OR planner solved all 96 valid Horn
goals on both held-out splits faster than the LR-01 recurrent actor, despite
complying with its per-example time allowance. FC-10 changes domain to
**actual Lean 4.19.0 elaborated goal states** and kernel-checked tactic
applications, rather than continuing finite Horn graph benchmarks.

## Source contract

Generate actual Lean declarations in a small, auditable subset of Nat,
Prop, and List obligations. Each source declaration has a known correct
reference tactic **from an explicit finite registry**. The source
generator must place a unique event marker and Lean `trace_state`
**before** the reference proof tactic. CI compiles every theorem in
Lean 4.19.0 and records compiler-emitted goal states. A parser must
read the actual Lean trace output with a goal turnstile; synthetically
reconstructed theorem titles are not accepted as goal-state inputs.

Primary tactic vocabulary, frozen in this order:
`omega`, `rfl`, `assumption`, `constructor <;> assumption`,
`exact h.1`, `exact h.2`, `exact Or.inl h`,
`exact Or.inr h`, `exact And.intro h.2 h.1`,
`simp [List.append_assoc]`, `simp_all`,
`exact False.elim h`.

Training/dev/test are separated by **unseen theorem generator template
families**, rather than only shuffled theorem instances. Train on 132
examples from 11 grammars, validation on 24 examples from 2 held-out
grammars, and a separately committed test with 48 examples from
four additional families. Each task has a stable hash, reference
tactic, original theorem signature, and source-origin family.
Formula/hypothesis names may vary deterministically to prevent simple
per-task ID memorization. Root seed **20261113**.

CI MUST freeze the train/dev-only source and model checkpoint
before the test generator module is added or checked out.
Conservative normalized signature fingerprints may not occur
across splits. A duplicate must fail the source audit.

## Neural learner and baselines

The primary learner imports the **original LR-01 TypedLoop**
from `LoopedTransformers/experiments/lean_reasoner/train.py`.
It encodes actual Lean-printed goal-state text as hashed word/subword
features on graph nodes, and candidate tactic slots as typed rule
nodes. Tied-weight recurrence is evaluated at **one** and **four**
passes on two fixed seeds 17, 29. Train only from reference
tactic labels on frozen training states, 250 AdamW CPU steps,
batch 24, learning rate 0.002. Use validation for reporting
and one fixed threshold (top-1); no test-derived hyperparameters.
Save exact train manifest and checkpoint SHA256.

Strong non-neural controls:
1. Deterministic exhaustive Lean tactic-application search in the
   listed registry order, trying each source tactic with the
   **actual Lean kernel**, stopping at the first successful
   certificate. Report attempted candidate count, wall clock
   milliseconds and accepted Lean proof.
2. `first | ...` ordered tactic search within the same Lean
   elaboration context, compiled in a single Lean process;
   report full CPU elapsed and successful theorem counts.
3. A simple lexical goal-state-to-tactic baseline fit on train
   only, using class priors/goal term tokens. Its code and scoring
   must be frozen before held-out test access.

All positive claims **must compile under Lean** with the
model-selected tactic; an unproved goal is an abstention.
No gold reference tactic, training labels, or exact
answer can be substituted at test inference.
Do not claim general theorem proving from synthetic
tactic templates or a small held-out sample.

Primary metric: Lean-accepted proofs out of 48 held-out
goals with one **model-selected tactic**; compare against
the first-tactic-only symbolic/lexical controls, and
report full-registry symbolic search separately.
Secondary: average tactic candidates executed, CPU
milliseconds per certified proof, total Lean invocations,
false reported certificates (must be zero), train and
test family breakdown, and seed/pass sensitivity.

**Success criterion:** at least one of the four-pass LR-01
seeds must exceed the train-only lexical baseline's
kernel-confirmed top-1 proof count on the same 48
goals, with zero false certificates. Exact full
tactic search is expected to solve every reference
task and is the correctness/CPU upper benchmark.
A negative result is research data.

This is a staged migration toward actual Lean states.
These first synthetic Lean declarations are still
small theorem templates. The next release should add
large imported theorem obligations from real project
repositories, with exact revisions and trusted Lean
compilation of the source declarations.
