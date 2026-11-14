# FC-12: local-context typed Lean tactic actions for imported research proofs

[Original preregistration](PREREGISTRATION.md) was frozen at
`d6fae0ddb9ce60036cff09e5ec93820ed3879af8` before
building FC-12 data or scoring a model.

FC-11 harvested 31 real intermediate tactic states from imported MathCan
theorems and learned from eight bare action scripts. It verified four
and one of fourteen held-out original theorem proofs in two recurrent
seeds. A corrected Lean-native symbolic search compiled five of fourteen
with a generous, post-hoc time allowance. The model showed no robust
proof-search advantage, and eleven train states had no progressing
action in that bare vocabulary.

FC-12 generates **parameterized tactic scripts from actual Lean 4.35
compiler-printed local contexts**. It uses twelve frozen action families,
including local-hypothesis application, local equality rewrites,
existential/conjunction case destruction, and induction on locally
typed List variables. Every candidate is checked by Lean4.35 in the
original theorem context.

The source remains the actual unmodified
[patrickdugan/MathCan](https://github.com/patrickdugan/MathCan)
repository at commit
`c84e2025334e9a967c1297425a490f98d5b3d189`.
The original references are 100%-compiled proof oracles and are never
used as fallback after a model failure.

## Train-only source and positive-action prerequisite

The registered train-only selector contains 27 original research
theorems across seven MathCan modules: FiniteHorizonGap,
DeletionMemory, TOEGraph, TOECosmology, Intervention, HookState,
and GraphAuthorization.

[Original source CI #43545821464](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43545821464)
passed its exact source audit and compiled all 27 original proof bodies,
extracting **51 genuine single-goal intermediate Lean states**.
Frozen compiler goal-state JSONL SHA256:

`403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0`

Only the declared *train-only* source and original proof prefix may
create supervised action effects. All twelve families are probed against
each actual source state using at most four concrete parameter arguments
generated solely from its printed hypotheses. Scratch runs may use
`sorry` to expose a subsequent state; they never count as formal
positive theorem certificates.

The initial [train-only action run #43546204691](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43546204691)
tested 961 concrete Lean tactics and correctly failed the mandatory
coverage checkpoint because bare `exact_local` had zero successful
training examples. It found 13 accepted rewrites, one successful
`rcases`, and 30 successful list-induction steps. The
[pre-test amendment](PRETEST_AMENDMENT_A.md) records the observed
train-only failure and permits typed application of local
hypotheses to other local arguments, such as
`exact (h word valid).symm`.
The corpus and held-out selector remain unchanged until the
mandatory positive-action check passes.

## Candidate generation and theorem verification

The fixed twelve action families are:

| Family | Example from compiler-visible context |
| --- | --- |
| omega | `omega` |
| rfl | `rfl` |
| simp_all | `simp_all` |
| assumption | `assumption` |
| intro | `intro` |
| constructor | `constructor` |
| exact_local | `exact h` or `exact (h word valid).symm` |
| apply_local | `apply h` |
| rw_local | `rw [h]` or `rw [← h]` |
| rcases_local | `rcases witness with ⟨fc12_left, fc12_right⟩` |
| induction_list | `induction xs` |
| simpa_local | `simpa using h` |

The local argument parser uses only the actual `trace_state` telescope
and deterministic original context order. It rejects arbitrary Lean
injections, source proof text, and more than four candidates for any
family. The complete test pipeline applies model-selected tactics on
a scratch copy of the original theorem to observe any remaining goals.
A positive theorem release requires a **separate successful Lean
compilation of the original theorem with the chosen tactics and
no `sorry`**. Failed/unfinished attempts abstain.

The trained actor uses the unchanged original `TypedLoop`
shared recurrence and exact **62,401-parameter** architecture.
The [early model smoke CI #43546651741](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43546651741)
passed its parameter audit, local-script syntax tests, and
native sorry-free Lean theorem compilation.

## Held-out comparison protocol

After the train-only action effects and checkpoints are frozen,
the separate held-out selector will import **16 original MathCan
theorems from four previously unused modules**: Hodge,
TOEDressedMatter, TOELorentzBivector, and TOETensorCompletion.
It will use the exact same source SHA and compiler version,
with a source-family-disjoint split.

A recurrent model ranks typed family IDs, then uses the same
deterministic concrete arguments and Lean compiler response
as a source-only symbolic baseline. Both have at most three
successful goal transitions and twelve candidate tactic
attempts per theorem. The symbolic baseline additionally
receives only the model's measured per-theorem wall-time
allowance, never its predictions.

Learning and runtime efficiency are separate preregistered
outcomes. The theorem oracle remains 16/16 under its original
complete reference scripts. This benchmark concerns fully
formalized MathCan propositions, and provides no proof of
an unformalized mathematical conjecture.
