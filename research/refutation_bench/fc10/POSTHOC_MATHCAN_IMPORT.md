# FC-10 post-hoc: real MathCan research-theorem goal-state harvesting

**Executed after FC-10's first preregistered model evaluation.**
This is a distinct, exploratory source import, and the original
FC-10 132/24/48 generated source splits, frozen model checkpoints,
first-run scores and preregistered criterion remain unchanged.

## Independent external source and kernel version

The source was checked out from the existing public
[MathCan](https://github.com/patrickdugan/MathCan)
repository at **immutable commit**
`c84e2025334e9a967c1297425a490f98d5b3d189`.
That source specifies `leanprover/lean4:v4.35.0`
in its exact `lean-toolchain` file.

The probe imports the original source texts of
`MathCan/FiniteHorizonGap.lean`,
`MathCan/DeletionMemory.lean` and
`MathCan/TOECosmology.lean`. All three
are standalone `import Init` modules.
It makes separate instrumented copies,
prepends `import Lean` to access
`trace_state`, inserts a before-proof
goal trace after the original `:= by`,
and leaves each original theorem
proof's tactic body unchanged.
Nothing was committed to or altered
in the MathCan repository.

[Post-hoc GitHub Actions #43458916540](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43458916540)
passed the instrumentation unit tests,
installed the pinned Lean **4.35.0**
compiler and successfully compiled
all three instrumented modules,
including their unmodified
reference theorem proofs.
The exact compiler stdout was
parsed into **seven actual elaborated
goal states** and archived.

Frozen source-state JSONL SHA-256:

`627e3251e1aaeff810177628c6651a427a9b9a20049ae4111bb9ad347a9b103c`

## Seven kernel-backed original theorem goals

| File | Original theorem | Goal type and context |
| --- | --- | --- |
| FiniteHorizonGap | `epsilon_sixth_clearance` | Nat inequality from `6 * ε ≤ δ` |
| FiniteHorizonGap | `epsilon_sixth_equiv` | Equivalence between a clearance inequality and `6 * ε ≤ δ` |
| FiniteHorizonGap | `full_horizon_recovers` | Equality of two lists from finite-prefix agreement |
| DeletionMemory | `correct_decoder_separates` | Injectivity of a terminal encoder on two distinguishable inputs |
| TOECosmology | `anchored_present` | Vanishing of an anchored redshift mode at its present point |
| TOECosmology | `curvature_mode_absorbed` | Integer polynomial identity for a curvature-mode shift |
| TOECosmology | `matter_mode_absorbed` | Integer polynomial identity for a matter-mode shift |

The actual compiler-emitted state of the first
source theorem is:

~~~lean
delta epsilon : Nat
h : 6 * epsilon ≤ delta
⊢ delta + 2 * epsilon ≤ 2 * (delta - 2 * epsilon)
~~~

The DeletionMemory imported theorem begins:

~~~lean
Source Memory : Type
encode : Source → Memory
decode : Memory → Source
x y : Source
hx : decode (encode x) = x
hy : decode (encode y) = y
hxy : x ≠ y
⊢ encode x ≠ encode y
~~~

These are source **research theorem states**
rather than generated FC-10 synthetic templates.
The first is an arithmetic tactic problem; the
second requires a genuine multi-step argument.
There are also Nat, Int, List and abstract Type
obligations in the imported set.

## Reproducibility and separation

The [instrumentation source](imported_probe.py)
checks the exact source toolchain and theorem
names before inserting markers. Its parser
requires one actual `⊢` goal between
a matching pair of compiler-emitted
`FC10_IMPORTED_BEGIN` and `FC10_IMPORTED_END`
markers for each theorem, and rejects
compiler errors or `sorryAx`.

The [independent source CI workflow](../../../.github/workflows/refutation-fc10-imported-mathcan.yml)
checks out MathCan at the immutable
source commit, compiles the instrumented
files under Lean4.35.0 and uploads the
complete source copies, proof-compiler
logs, seven real goal states and
SHA256 metadata. The artifact is
`fc10-posthoc-mathcan-goals-43458916540`.

This source probe does **not** score the
frozen FC-10 neural tactic ranker on
the imported theorems and therefore
makes no learned transfer claim on
MathCan. That requires a separately
predeclared train/test split of
imported math sources, a multi-step
tactic-action vocabulary, and a
kernel-gated proof-search evaluation
with MathCan's exact Lean version.
