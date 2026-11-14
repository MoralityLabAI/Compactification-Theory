# FC-06 preregistration: strict-budget learned rule selection

**Registered November 13, 2026, before FC-06 source generator, training and test.**
This protocol is immutable for the first CI result. Amendments after test inspection
must preserve the original and be labelled exploratory.

## Source and exact structural contract

FC-05's ten-step budget covered every legal novelty-action sequence. FC-06 uses a
strict **five-rule-action budget**. Every task has **12 atoms**: one initial
fact, one goal chain of d essential steps, (10-d) reachable distractors, and
one unreachable guard. For a true entailment the shortest legal proof must
have exactly d steps, where d <= 5, while the closure has **10 novel atoms**.
Thus all true tasks satisfy min_steps <= 5 < available_novel_steps = 10.
Incorrect legal choices may exhaust the five-step budget.

Negative tasks block an essential spine inference with the unreachable guard,
and decoy rules cannot repair the goal. Every negative receives a complete
Boolean countermodel checked exhaustively.

- Train: **768** examples, depths 2-4, families star_decoys, ladder_decoys,
  fork_decoys. Root seed 20261113; split offset 11203.
- Dev: **192** examples, depths 2-4, a new cross_decoys family; offset 22411.
- Test_same: **192** examples, depths 2-4; offset 33629.
- Test_deep: **192** examples, depths 4-5; offset 44839.
- Both tests use four new families: zigzag_lures, nested_fork,
  late_spoke, alternating_gate. Each contributes 24 valid and 24
  invalid to each split. Training/dev families stay absent from tests.
- Source graph WL fingerprints must be unique across FC-02, FC-03, FC-04,
  FC-05, FC-06 train/dev, and both tests. WL is a conservative grouping
  proxy, not an isomorphism proof.

The TRAIN-ONLY source generator and this preregistration must be committed
and pinned BEFORE the test module exists. The training GitHub Actions job
checks out this earlier commit and verifies test generator absence.
The dependent evaluation job downloads SHA-identified frozen checkpoints
before checking out the separately pinned test generator. Any source
generator failures must fail CI rather than silently rewrite labels.
The random-legal baseline test result is always retained even if weak.

## Independent proof certificates

Positive claims must pass Horn operational closure, exhaustive Boolean
truth-table entailment, and an exact minimal-proof BFS during source audit.
Source CI compiles every generated positive reference theorem with Lean
4.19.0. Negative claims receive complete verified finite countermodels.

Inference selects original source-rule indices in order under budget 5.
The independent FC-05 trace checker, called with explicit budget=5, verifies
rule existence, premise availability, novel conclusion, no post-goal steps
and successful goal derivation. It never chooses or repairs a model step.
Only accepted actual model-selected traces are rendered into Lean terms,
and the dependent CI kernel-compiles every emitted positive theorem.
Negative releases still require exact verification of the full predicted
Boolean assignment; any failed positive search abstains.

## Architecture and frozen training recipe

Frozen FC-04 gate: LR-01 LoopReach width24, four trained passes, 11,521
parameters. Train from FC-02 train only: 350 AdamW steps, batch48, lr
0.002, weight_decay 0.01, grad clip1, two CPU threads, deterministic
PyTorch 2.5.1 CPU, seeds **17 and 29**. Select gate confidence threshold
from fixed grid [.55,.70,.85,.93,.98] using FC-02 dev ONLY.

Proof actor: a SEPARATE copy of the LR-01 tied recurrent cell initialized
from this gate's weights, plus the FC-05 4,705-parameter Linear(96,48),
GELU, Linear(48,1) rule ranking head. Fine-tune actor AND head on FC-06
training-only positive proof-prefix states. Teacher is deterministic exact
minimum-length BFS using the 5-step limit. Cross-entropy targets original
source-rule IDs masked to legal new consequents. Actor: four tied passes
per action, 300 AdamW updates, batch48, lr0.002, weight_decay0.01,
grad clip1, deterministic minibatch seed 6154 + model seed. No changes
based on dev/test outcomes. The original negative/provisional gate stays
frozen. No gold label, source ID, family, minimum proof length or
certificate enters the actor's input features.

## Registered comparator and result contract

All arms use identical frozen FC-04 decisions for negative certificates
and positive eligibility. A positive attempt starts after two consecutive
confidence checkpoints meet the calibrated threshold. Compare these
five-action rule policies:

1. learned_joint: masked neural argmax among currently legal novel-rule
   IDs, recomputing four shared cell passes at every action.
2. random_legal: deterministic uniform choice, seeded by
   2917 + stable SHA256(case ID).
3. goal_first: if a goal-producing rule is legal, apply it; else
   the same deterministic random legal choice.
4. shortest_oracle: exact BFS proof, as a labelled non-neural oracle
   upper bound; NEVER invoked in model inference.
5. fc04_provisional: previous stopping-gate positive eligibility,
   reported separately as unverified.

Report every seed/test split/family: positives certified out of 96 valid,
negatives certified out of 96 invalid, positive attempts, action-success
rates on eligible positives, abstentions, false formal certificates,
raw failed traces, paired learned-vs-heuristic wins/losses on the same
cases, action lengths, model recurrent cell calls and CPU time,
Python checker time, Lean-compiled output counts and bytes. Record
all predictions, including failures. Refutation or positive acceptance
without exact verification fails the experiment.

**Primary learning success:** The learned policy must certify more eligible
positives than BOTH random_legal and goal_first on EACH test split for
BOTH seeds, with zero false formal certificates. Failure yields a
negative or mixed result. **Compute compactification** separately
requires more certified positive proofs per 1000 neural cell calls
than the best non-neural baseline after charging the learned policy's
extra action-scoring calls.

Tests are PUBLIC deterministic Horn logic, not a private blind
mathematics challenge. The exact solver has 100% correctness in this
fragment and is available as a separate symbolic upper bound.
Nothing here proves or disproves RH, Collatz, Fermat, or P versus NP.
