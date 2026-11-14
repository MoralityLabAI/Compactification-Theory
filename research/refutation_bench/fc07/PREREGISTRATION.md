# FC-07 preregistration: misleading goal ancestry with exact proofs

**Frozen November 13, 2026, before any FC-07 source generation, model
training, or test-score inspection.** This protocol defines the first
test's outcome measures. Any changes following a test score are
exploratory amendments and must preserve the first-run record.

## Question

FC-06 trained a 16,226-parameter recurrent proof actor to succeed
under five legal Horn rule applications, yet a zero-neural backward
slice of the goal's syntactic ancestors solved every test positive.
FC-07 tests whether a learned policy can choose useful rules
**when the simple ancestry slice itself contains reachable decoys**.

Preregister the FC-06 static backward-ancestry algorithm as the
**primary non-neural baseline**, alongside uniform random legal
selection, immediate-goal-first, and an exact shortest-proof oracle
clearly labelled as a 100%-correct symbolic reference.

## Finite exact task construction

Every graph has 12 distinct atoms: one initial fact; backbone atoms
0,1,...,d with goal d; 10-d reachable distractor atoms; and one
unreachable guard. The canonical backbone contains rules
(i-1) -> i, so a valid goal has a shortest proof of exactly
d legal novel steps, where **2 <= d <= 5**. All distractors
are reachable from supplied premises independently of the goal,
and positive closure contains exactly **10 novel atoms**.

Introduce alternative rules **with the same backbone conclusions**
whose prerequisites include a reachable distractor, e.g.
(i-1 AND distractor_j) -> i, some of which are cyclic or
depend on shared subgoals. Every distractor must appear as a
premise of at least one alternative backbone producer, so it lies
in the *syntactic backward ancestry* of the goal. A reverse
ancestry slice will therefore contain at least ten reachable
propositions and can waste a five-action proof budget.

For invalid cases, the primary producer and **every** alternative
producer of one essential backbone atom must require the
unreachable guard. The exact checker must confirm the goal is
underivable. The guard can occur in cycles but can never be
derived solely from the initial fact. No alternate rule may
circumvent the blockade.

Cap at 12 atoms and 20 implication rules. Each positive source
must satisfy:

- Exact BFS minimum proof length **d <= 5**.
- Ten reachable novelty consequences; a strict five-rule budget.
- Every distractor is in the naive goal-ancestor set.
- Independently verified propositional entailment, and a
  Lean 4.19.0 proof under precisely its stated hypotheses.

Each invalid source must have an independently verified complete
Boolean countermodel, and bounded model proof search must fail
when no proof exists.

## Family and source split

The new FC-07 corpus must be disjoint from all prior
FC-02 through FC-06 cases by a conservative WL structural
fingerprint, and disjoint across its own train/dev/test.
No data may be selected or rejected based on observed
classifier or heuristic performance.

- Train **768** cases, depth 2-4; three new grammar families:
  `braided_ancestors`, `shared_subgoals`, `cycle_traps`.
- Dev **192** cases, depth 2-4; `gated_forks`, separate family.
- Test-same **192** cases, depth 2-4; four previously unused families:
  `reciprocal_forks`, `delayed_subgoal`,
  `false_shortcuts`, `nested_switches`.
- Test-deep **192** cases, depth 4-5; same four test
  family names but different cases and WL fingerprints.
- Every test family supplies exactly 24 valid and 24 invalid
  cases per split. Train and dev are balanced by family.
- Root seed **20261113** and frozen offset values:
  train **53107**, dev **64211**, test-same **75323**,
  test-deep **86453**, using deterministic attempt-index
  seeds with duplicate-WL rejection.

**Access isolation:** the train/dev-only generator and this
registration must be committed first, prior to adding any
test-generator code. GitHub Actions training checks out only
that exact train-only revision and asserts that the test
module does not exist. It freezes SHA256-identified network
weights and selected confidence thresholds. A *separate,
dependent* evaluation job obtains those immutable artifacts
before checking out the pinned source test revision.

## Frozen neural learner

Replicate the FC-06 recipe with exactly the same data access
boundaries, independent of later test results:

- PyTorch CPU 2.5.1 with deterministic algorithms, two CPU
  threads, seeds **17** and **29**.
- Train an 11,521-parameter `LoopReach(4, width=24)`
  LR-01 gating model on original FC-02 train only:
  AdamW lr .002, weight decay .01, 350 steps, batch 48,
  grad clip 1. Freeze it. Choose θ from the original
  [.55,.70,.85,.93,.98] FC-04 threshold grid using
  FC-02 development labels only.
- Initialize a separate LR-01 `JointActor` from the
  frozen gate weights, with a 4,705-parameter
  `Linear(96,48)->GELU->Linear(48,1)` rule head.
  Train **both** actor trunk and head with supervised
  minimum-length next-rule targets from **FC-07 train
  positives only**. The teacher finds a deterministic
  minimum rule sequence via BFS on source train only;
  its traces cannot appear in model inputs.
- Actor training: **300** AdamW steps, batch48,
  lr .002, weight decay .01, grad clip1; fixed
  sampler seed 6154 + model seed. No checkpoint selection,
  tuning or training augmentation based on test data.
- Recompute four tied LR-01 cells after each chosen
  action; mask to currently applicable rules with
  *novel* consequences. Up to **five rule applications**.
  Inputs are explicitly whitelisted atoms, facts,
  source rules, current known-set and goal only.

The original FC-04 gate decides whether positive proof
search starts, using two stable high-confidence checkpoints,
or instead releases a separately verified complete
Boolean negative countermodel. Failures of positive
search are **abstentions**, not theorem certificates.
No oracle may repair a model-proposed trace.

## Registered comparison and scoring

On each new test split for each seed evaluate precisely
these five-rule action policies, with identical frozen
FC-04 positive eligibility and negative verification:

1. `learned_joint`: jointly trained LR-01 actor.
2. `random_legal`: seeded uniform choice among legal
   novel-consequent rules, seed 2917 + SHA256(case_id).
3. `goal_first`: take legal goal-producer immediately,
   else the same deterministic random chooser.
4. `backward_slice`: calculate *syntactic* backward
   goal ancestry from **all source rules** and choose
   the lowest-index legal novel rule with its head in
   that set. This is the exact zero-neural FC-06
   post-hoc control, included a priori.
5. `shortest_oracle`: exact BFS, used in source truth
   and reporting only; strictly absent from inference.

Every accepted positive trace must pass FC-05's independent
strict rule replay with explicit **budget=5**, and the
**exact same model-chosen trace** must compile as a Lean
4.19.0 theorem in evaluation CI. Every accepted negative
must verify as a complete Boolean countermodel satisfying
all premises and falsifying the goal. Invalid or
unfinished traces abstain. Check and report zero false
formal certificates, failed first-rule attempts,
full source-rule traces and reason codes.

Primary metric: **positive proof certified / positive
eligible** among 96 valid cases on each test split.
Also report positive and negative certified recall,
overall certificate coverage, abstentions, false
certificate count, paired learned-only/baseline-only
success counts, first fault positions, total model
forward calls and milliseconds, independent checker
time, trace bytes, source-family outcomes, and total
recurrent cell invocations. Report verified positives
per **1,000 neural cell calls**, with a zero-neural
control's proof search/graph time charged separately.
Raw per-case reports and full Lean proofs are retained.

**First learning criterion:** outperform `backward_slice`
on positive eligible proofs on **both seeds and both
test splits**, with zero false certificates. Report
any mixed/negative result honestly. **Compute criterion:**
model-selected proofs must have lower total measured
model+verification cost per certified positive than
the backward-slice baseline on both seeds/splits.
The criteria are separate and failure of either is
documented. Do not infer an advantage over the
exact minimal-proof oracle, which scores 100% on
these bounded tasks by construction.

The test corpus is public and preregistered rather
than a cryptographically sealed blind challenge.
No conclusion about correctness of real manuscripts
or famous open conjectures follows without a
separately verified semantic bridge.
