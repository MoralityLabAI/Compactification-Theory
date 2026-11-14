# FC-03: preregistered out-of-family Horn transfer

**Protocol frozen 2026-11-13, before FC-03 model training or any test-score inspection.**
Protocol ID: FC03-PREREG-v1. Source ancestor: RefutationBench FC-02 at
`57dc81bda04d2dc05b108edac19e6667405b78e7`.
Any post-registration change of scoring, generator grammar, seeds, hyperparameters,
or comparisons must be logged in a dated amendment and called exploratory.

## Questions and primary contrasts

On an exact propositional Horn fragment, does a four-pass weight-tied LR-01
graph cell generalize to new rule topologies better than a compute-matched
four-block untied LR-01 model?

**Primary comparison**: `tied4_width24` against `untied4_width24`.
Identical 4 graph-cell invocations, same width, same batch/optimizer/steps,
same node/edge features and readout. The untied model has more parameters.
Report exact parameter counts and measured inference milliseconds alongside
the fixed equal cell-invocation budget; do not call these parameter-matched.

**Parameter-matched sensitivity**: `untied4_width12` compared with
`tied4_width24`. Width 12 is chosen algebraically from the known parameter
formula before the test (11,593 versus 11,521 parameters); invocation count
matches, while width/compute do not. Report parameter and cost differences.

**Predeclared extrapolation**: train `tied4_width24` at four passes, evaluate
the **same trained weights** at 8 and 12 inference passes. The increased compute
is explicitly charged. These are secondary outcomes and cannot justify
revising the primary four-pass comparison.

## Frozen source data and unseen families

Training and model-selection development are the *already frozen* FC-02
deterministic corpus at generator commit `7aca2bc3fc5586c5b7997fba868bd179db761121`
and cases SHA-256
`8e86a41dd93ff162072799332ebbef4c7134951798eb4af7ebb4ab044bebd56f`.

- Train: existing FC-02 1,152 examples, depth 2–4; structural families
  `linear`, `sidechain`, `fanin`.
- Development: existing FC-02 192 examples, depth 5–6; families
  `ladder`, `diamond`. Record outcomes without selecting checkpoints,
  thresholds, pass counts, or learning rate from development.
- New **same-depth family-held-out test**: 192 cases at depths 2–4.
- New **deep family-held-out test**: 192 cases at depths 6–8.

Both test splits use four **entirely new** structure grammars:
`fork_join`, `anchored_cycle`, `alternate_routes`,
`guarded_ladder`. Each has 48 cases per split, balanced 24 valid/24 invalid.
Root generation seed `20261113`; per-split deterministic seeds must be fixed
in source. Cases have 6–12 atoms, at most 20 rules, and are grouped by
isomorphism-invariant WL fingerprints across all 1,920 FC-02 and FC-03 cases.
Fingerprints are a conservative collision/group guard, not a complete
isomorphism oracle.

The same-depth test isolates held-out structural topology at a comparable
training depth. The deep test adds depth shift on those same structural
families. Explicitly report that test families span both test sets for the
depth-controlled comparison. Distinct test instances and fingerprints
are required. The test generator is public; this is a *pre-registered*
evaluation with no unseen private exam.

## Ground truth and certificate gate

For every task, a separate exact Horn forward-chaining implementation
and exhaustive Boolean truth tables (<=12 atoms) must agree on entailment.
Each invalid case carries a **complete** assignment satisfying all
premises and falsifying the conclusion; the independent checker verifies it.
Each valid example must construct and compile a Lean 4.19.0 proof from explicit
premises, without `sorry`. The generated source will be checked in source
CI; assert no hidden global axioms through inspection of generated theorems.

At inference, neural outputs propose a complete Boolean assignment from the
twelve atom logits. **The exact checker must reject** any assignment that fails
the premises/conclusion test. Never correct or complete a model assignment
using ground truth.

Primary metric: classification accuracy on all 192 examples per test split,
per seed, with counts and paired per-instance outcomes. Report:
- false-refutation rate among 96 valid examples;
- certified-recall rate among 96 invalid examples;
- fraction of negative predictions with accepted countermodels;
- verdict macro-F1, including a 50% majority-label control;
- exact parameter count, cell invocation count, train seconds, inference
  time and CPU process memory;
- confusion matrix and paired tied-versus-untied McNemar counts.

The finite exact forward-chaining oracle has 100% accuracy by design.
Do not suggest a neural advantage over the exact oracle. Keep the
model's *predicted* refutation distinct from a *certified* counterexample.
A deployable selective auditor may issue counterexamples only after
verification.

## Frozen learner configuration

- Python 3.11, pinned CPU PyTorch 2.5.1, deterministic algorithms on.
- LR-01 `TypedLoop` implementation from
  `MoralityLabAI/LoopedTransformers/experiments/lean_reasoner/train.py`.
- Encoder: existing FC-02 `formal_student.encode`, 5 atom features and 4
  rule features, real proposition/rule incidence edges.
- Training labels: exact closure of **training only**; target includes
  atom reachability and goal truth, as in FC-02.
- AdamW lr=0.002, weight_decay=0.01, 350 updates, batch=48, grad-norm cap 1,
  existing FC-02 objective; no early stopping.
- Model seeds 17 and 29. Same training-example minibatch indices for the tied
  and untied arms at each seed. Report seeds separately and their aggregate.
- 2 CPU threads. All tests performed in a separate CI job that reads *frozen*
  trained checkpoint SHA-256 hashes from the training job. The training job
  has no FC-03 test file access and may access only FC-02 train/development.
- Save original predictions, raw scores and JSON receipts, not only summary.

No use of FC-03 test labels for training, thresholds, checkpoint selection or
architecture changes. The first sealed test run is primary; reruns after code
fixes must preserve the original failed run and be marked as amendments.

## Decision criterion

Report the paired mean accuracy advantage on the same-depth unseen families
and deep held-out families, with per-seed differences and exact paired
discordance counts. A claimed *generalizable* tied-cell benefit requires a
consistent positive advantage in **both** test depths and both seeds,
alongside non-increasing false-refutation rate and no certificate-soundness
violation. Failure to satisfy any part is inconclusive or negative.

The result is about bounded Horn entailment graphs. It says nothing
directly about RH, Collatz or a paper-level mathematical proof without a
validated semantic translation and a raw-source evaluation.
