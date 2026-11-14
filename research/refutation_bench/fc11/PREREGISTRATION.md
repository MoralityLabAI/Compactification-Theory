# FC-11 preregistration: imported Lean multi-step tactic search

**Frozen November 14, 2026, before FC-11 instrumentation, corpus extraction, or new model results.**

## Objective

FC-10 trained a 62,401-param tied LR-01 model on actual Lean-printed goal
states from synthetic theorem templates. Its four-pass tactic policy did
not beat lexical retrieval, and native Lean tactic search solved every
test theorem. A post-hoc import under MathCan's own Lean4.35 toolchain
extracted seven real research theorem goals.

FC-11 evaluates **intermediate tactic states from existing MathCan theorem
proofs** and tests whether a recurrent policy can choose an efficient
sequence of actions whose **complete submitted Lean proof compiles**,
compared with a measured CPU-constrained Lean-native tactic search.

## Immutable source and train/test split

Pinned source: `patrickdugan/MathCan` commit
`c84e2025334e9a967c1297425a490f98d5b3d189`, toolchain
`leanprover/lean4:v4.35.0`. Keep all original source files unchanged.

Train-family module files:
- `MathCan/FiniteHorizonGap.lean`: arithmetic and prefix-equality proofs.
- `MathCan/TOECosmology.lean`: anchored modes and algebraic rewrites.
- `MathCan/TOEGraph.lean`: graph holonomy and cyclic contradictions.
- `MathCan/DeletionMemory.lean`: exact deck lemmas and decoder injectivity.

Holdout-family module files:
- `MathCan/Intervention.lean`: observational-equivalence compositionality.
- `MathCan/HookState.lean`: hook-state path identities and potential bounds.
- `MathCan/GraphAuthorization.lean`: repair-budget reasoning.

The source extractor selects named, predeclared theorem declarations
inside those modules, preserving their original statements and
reference proof tactics. It adds `trace_state` before successive
**top-level original tactic commands** and parses the actual Lean
compiler output. For multi-goal states record all goals, with
the count explicit. No reconstructed theorem signature is accepted
as a model input. The original lemma body remains unchanged.

Pin and independently CI-check the **train-only extractor revision**
before adding any held-out file selection or evaluation code.
The upstream repository contains both source families, so the
access boundary is enforced by a train-only selection manifest,
the external source commit, and recorded file access. Training
must not load the held-out module files as dataset inputs.
The held-out source contains real mathematical theorems whose
proofs exist in the same publicly accessible revision: this is
a family-held-out evaluation, **not a private blind challenge**.

## Tactic actions and model supervision

Freeze a general tactic vocabulary before held-out evaluation:
`omega`, `decide`, `rfl`, `simp_all`, `aesop`, `assumption`,
`intro`, `constructor`.
Each action must be attempted on **multiple actual training Lean
states** through a compiler probe, yielding at least one
train exposure for the action, including negative outcomes.
A class is not treated as learned merely because a tactic
string appears in the vocabulary. Train-label success is
determined by actual Lean effects or kernel completion,
not the original theorem's reference tactic identity.

The first experiment trains a 62,401-param goal/tactic graph
ranker built from the original `TypedLoop`, using hashed
Lean-printed state tokens. Keep seeds 17 and 29,
four tied inference passes, CPU PyTorch2.5.1,
300 AdamW steps, batch16 and lr=.002.
Use train state-action success labels from the
source-only corpus. Freeze before test extraction.

## Closed-loop multi-step evaluation

For each selected held-out *original* theorem statement,
start at its actual initial Lean tactic state. The policy
selects one tactic from the registered vocabulary.
Apply the chosen tactic to a scratch copy of the original
theorem **through Lean**. A completed original theorem
must return successful Lean compilation with no `sorryAx`;
that is the only formal success outcome.

If the candidate leaves a goal, collect the actual
new goal state from a scratch Lean prefix probe.
Allow at most **three model-selected tactic actions**
per theorem, and stop on a kernel-accepted proof,
failed tactic, or attempt limit. The scratch probe may
use `sorry` solely to expose intermediate states
for subsequent inference. It is strictly untrusted:
never release a positive theorem certificate based
on a scratch run containing `sorryAx`.

For multiple open goals the policy may abstain
in this initial experiment, with an explicit
`multigoal_unsupported` reason. Record every
tactic and actual compiler response, including
all failed prefixes.

## Strong Lean-native baselines and cost

1. A fixed tactic registry `first | ...` native
   Lean backtracking baseline compiled against
   the original theorem. Repeat up to 3 goal
   transitions with a fixed script, all in one
   Lean process per tested module.
2. A CPU-limited symbolic `first` search using
   up to the **same measured wall-time allowance**
   as the neural inference plus its scratch
   tactic probes on that theorem. Record any
   overrun and subprocess startup overhead.
3. One-action `aesop` and `omega` controls,
   submitted to the original kernel.
4. An original-source reference proof oracle,
   reported as 100% correctness only,
   excluded from neural/symbolic comparisons.

The symbolic search is allowed a max wall time,
and its actual elapsed must be recorded. The
neural policy's extra CPU includes recurrent
forward calls, scratch Lean subprocess probes,
and full final Lean compilation; always report
total measured time per certified theorem.
No claim of FLOP equality follows from a
deadline-based comparison.

Primary metric: **number of kernel-accepted
original held-out theorems** out of the frozen
set, under at most three model actions.
Secondary: per-module proof count, action
sequence lengths, multi-goal abstentions,
candidate failures, scratch `sorry` call
counts, CPU time per accepted theorem,
Lean-native full-registry success/time,
and model/lexical disagreement.

The first model success criterion is greater
kernel-accepted coverage than the bounded
CPU-matched native-tactic baseline, with
zero false claims. Report a negative or
partial result if not met. A distinct compute
success requires lower total measured time
per accepted theorem, separately.
This is a narrow imported Lean corpus and
does not certify open conjectures.
