# FC-13: original Lean 4.35 transfer under a hard deadline

This source-side branch is stacked on FC-12 and keeps FC-12's training,
original held-out results, and 64-case independent replay immutable.

The [FC-13 preregistration](PREREGISTRATION.md) fixes 16 new original
theorems from the first four source-order declarations in each of:
HookMultiTick, Settlement, TOEJordan, and TOENonAbelian. All four modules
import original MathCan dependencies. FC-13's source extractor preserves
those imports, adds read-only goal-state traces to copies of the unchanged
reference proofs, and compiles them under the pinned MathCan Lake project.

[Original source certification CI #43554159459](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43554159459)
passed, compiling 16 original theorems and extracting 18 authentic Lean
goal states across the selected theorem prefixes. The original compiler-goal
JSONL SHA256 is
`3df327272bc4768eac2fdfaa60a2cdbead06443e48153781683ee03944568554`.

The learned controller uses only the previously trained FC-12 model
weights and its twelve-family context-instantiated tactic grammar.
The new native and recurrent comparison is fixed at 20 seconds per
theorem, with a subprocess-level hard timeout, a 12-call ceiling
including scratch and final compiler processes, and a separate
sorry-free original-theorem release gate.

The [corresponding model branch](https://github.com/MoralityLabAI/LoopedTransformers/tree/feat/lr01-fc13-hard-deadline-transfer)
owns the policies and evaluation; this source branch owns the independent
theorem selector, original source build and genuine compiler states.
