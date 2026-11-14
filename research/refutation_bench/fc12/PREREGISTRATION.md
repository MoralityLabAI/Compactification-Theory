# FC-12 preregistration: typed, parameterized Lean tactic actions

**Frozen November 14, 2026, before FC-12 source selection, probes, training, or held-out results.**

FC-11 demonstrated actual compiler-state acquisition and independent sorry-free
kernel proof verification on original MathCan declarations. Its learned policy
proved 4/14 or 1/14 original test theorems using eight bare tactic scripts; the
corrected Lean-native search proved 5/14 with a relaxed time allowance.
Eleven of 31 training tactic states had no progressing action within the
registered script vocabulary. FC-12 addresses **action expressivity** by
constructing parameterized, typed tactic candidates from the *actual compiler-
printed goal context*, including local hypotheses, equalities and list variables.

## Original source and independent families

The ONLY upstream research source is `patrickdugan/MathCan` at immutable commit
`c84e2025334e9a967c1297425a490f98d5b3d189`, Lean
`leanprover/lean4:v4.35.0`. All original proof bodies remain unchanged.
The train-only selector and its extractor must be frozen and kernel compiled
**before** any FC-12 held-out selector module is committed. The source repository
itself is public, so the split protects selection and model inputs, not
cryptographic access to all upstream theorem source files.

Training theorem-module families, in the registered order:
`FiniteHorizonGap`, `DeletionMemory`, `TOEGraph`, `TOECosmology`,
`Intervention`, `HookState`, and `GraphAuthorization`.

Held-out original theorem-module families, never previously used in an FC-10/11
neural evaluation: `Hodge`, `TOEDressedMatter`,
`TOELorentzBivector`, and `TOETensorCompletion`.
Take exactly **four named theorems per held-out module** (16 total).
Tests are formal source-family holdouts, not unseen mathematical proof structures
in the wider world. Do not change held-out cases after scoring.

Original theorem state collection instruments successive top-level reference
tactics using the FC-11 compiler trace protocol. It records *real* Lean
elaboration states, including the number of simultaneous goals and exact source
file/theorem/step SHA256. The source-level reference proofs must all compile
without `sorryAx`. A model never receives gold future source tactics at
inference. Reference original proofs are an explicit 100% oracle ceiling.

## Registered twelve typed tactic families

Freeze these family IDs and their order before training:

1. `omega`
2. `rfl`
3. `simp_all`
4. `assumption`
5. `intro`
6. `constructor`
7. `exact_local` (instantiate `exact h` using local hypothesis names)
8. `apply_local` (instantiate `apply h`)
9. `rw_local` (instantiate `rw [h]` or `rw [← h]` for local equalities)
10. `rcases_local` (instantiate `rcases h with ⟨x,y⟩` for local conjunction/existence)
11. `induction_list` (instantiate `induction xs` on local `List` variables)
12. `simpa_local` (instantiate `simpa using h` for local hypotheses).

Only a **lexical parse of the actual Lean state** may generate parameter
arguments. No reference proof text, filename, source declaration name,
test labels or gold tactic metadata may choose arguments at inference.
Keep a deterministic maximum of **four concrete scripts per family**
(in the goal-state hypothesis order, with explicit forward/reverse rewrite
tie order). Log how many arguments were generated and why any family had
zero candidates. A parameterized tactic must be checked by the *actual
Lean4.35 compiler* on the original theorem context; malformed candidates
are failed actions.

## Genuine train-only action exposure

Probe all twelve families against every **single-goal** state from the pinned
TRAIN theorem trajectories, using at most four concrete arguments per family.
Scratch probing may append `all_goals sorry` ONLY to obtain next-state
compiler text, and must be marked untrusted. A probe succeeds as supervision
when Lean accepts the script and either closes the goal or changes the actual
remaining goal-state text. A full positive theorem certificate requires
a **separate compiler call with the original theorem and its proposed
action sequence without any sorry**.

Publish complete positive and negative action-effect matrices by family,
including candidate-source SHA256, actual Lean compiler return codes, and
wall time. Before model training require at least one compiler-accepted,
goal-progressing TRAIN example for each of `exact_local`, `rw_local`,
`rcases_local`, and `induction_list`. If this train-only prerequisite
fails, report that outcome and correct the source training curriculum
**before freezing the train-only source**; never select held-out test
examples to repair action coverage.

The model may train from reference-stage *states* and compiler-probed
candidate outcomes. Its loss uses positive/negative tactic-family labels
only. It must never use the original reference tactic string as the correct
label when a real compiler probe failed.

## Neural policy and strict proof gate

Use the original LR-01 `TypedLoop`, 62,401 parameters, 4 shared passes,
12 candidate family nodes, hashed original Lean goal text (max112 tokens).
Seeds 17 and 29; PyTorch CPU 2.5.1, deterministic two-thread runtime,
300 AdamW updates, batch16, learning rate .002, weight decay .01.
Freeze checkpoint SHA256 and all action-effect training artifacts before
opening the separate held-out selector. No test-driven threshold selection.

For each of 16 held-out original theorem statements, the policy proposes
up to **three successful goal transitions**, using compiler feedback to
reject malformed/tactic-failing concrete instances. Maximum four argument
candidates per tactic family and maximum twelve actual Lean candidate
probes in total per theorem. After each successful but unfinished tactic,
obtain the next REAL compiler goal from a separately marked untrusted
scratch `sorry` call. More than one open goal can cause an explicit
`multigoal_unsupported` abstention. Report every failed tactic and
argument trial. Only a separately compiled, fully closed theorem using
the original statement and the model-selected script with **no `sorry`**
is a positive certificate.

## Matched native symbolic search

Freeze a non-neural symbolic policy before test scores. It enumerates
the **identical instantiated candidate scripts** from the exact same
goal-state parser in the fixed twelve-family order. It may use compiler
feedback to reject failing actions and advance through real goals, up to
three goal transitions and twelve actual candidate compiler probes.
Charge the same Lean subprocess startup costs and record actual per-case
wall time. Evaluate both a fixed 12-probe cap and a **model-derived
time allowance**, where the symbolic planner receives only the scalar
total measured time spent on a given model's tactic inference and compiler
calls, not its chosen tactics or logits. Report any deadline overrun.

Additionally run the source reference proof as a distinct 16/16 exact
kernel oracle, never an inference fallback. Report train-only lexical
tactic-family retrieval, if implemented, separately from native search.

## Predeclared outcomes

Primary: number of original held-out theorem declarations which compile
in Lean4.35 using the policy's complete selected tactics, for each seed.
Compare to the *same case/time-allowance* native search, preserving all
16 per-theorem outcomes, names, files, actual compiler goal SHA, model
tactic lists and exact kernel verdicts.

Independent cost: total wall-clock milliseconds per certified theorem,
including neural forward, scratch compiler and fresh final Lean compile,
versus native search including its actual compiler trials.

**Registered model success** requires strictly more kernel-certified
original theorems than the CPU-allowance native control for **both**
seeds, with zero false certificates. A separate compute success requires
lower measured ms per certified original theorem in both seeds. If either
fails, publish the negative result. The original source oracle is always
16/16, and a model does not prove an unformalized mathematical claim.
