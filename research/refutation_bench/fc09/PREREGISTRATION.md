# FC-09 preregistration: CPU-budget-matched LR-01 versus AND/OR Horn planning

**Frozen November 13, 2026, before FC-09 generation, model training, or any test score.**
This protocol defines the primary comparison. Preserve every first-run result, including
failures; subsequent modifications are exploratory and require a separate amendment.

## Question and setting

FC-08 learned rule selection exceeded a preregistered 24-popped-state
symbolic best-first search, but failed the CPU efficiency criterion.
FC-09 compares the same compact 16,226-parameter tied LR-01 actor
against a stronger, preregistered **conjunction-aware AND/OR heuristic
planner** under a *per-case measured CPU wall-time budget* equal to
the recurrent actor's additional rule-scoring time. Both policies
share the same FC-04 gating and independent exact proof-checking
costs; the symbolic policy sees only the actor's **time allowance**,
never its predictions, traces, probabilities, or ground truth.

This is a public, finite **12-atom Horn entailment** fragment.
Exact symbolic shortest proof BFS solves 100% by construction,
so the experiment cannot establish a neural mathematical
accuracy advantage over an unrestricted symbolic solver.
It tests bounded proof selection under comparable runtime.

## Frozen dataset structure and split

Valid tasks require conjunction. They have 12 distinct atoms,
one initial fact, a goal with a shortest verified proof of
d = 3, 4, or 5 novel-rule applications, 10 reachable novel
consequences, and an unreachable guard. Every valid goal
producer has at least two antecedents, and there is **no
unary-rule path** to the goal. Every reachable distractor
appears as a premise of some alternate goal producer,
and additional **conjunctive alternative producers of
intermediate goal ancestors** can create shared subgoals,
costly false routes, and legal cycles.

Every positive must have a shortest exact proof of length
d<=5, verified by independent BFS and Lean 4.19.0. Every
negative task guards every goal producer, admits no
valid proof and has a complete independently verified
Boolean countermodel. No result may be accepted if
operational Horn closure disagrees with exhaustive
Boolean truth tables. All model-produced positive
rule traces must pass the separate FC-05 checker
with budget 5 and compile in Lean. Failed proposals
abstain, with no reference proof repair.

- Train **768** balanced cases, depths 3-4,
  families `convergent_and`, `crossed_dependencies`,
  `cyclic_shared`.
- Dev **192** balanced cases, depths 3-4,
  separate family `nested_conjunction`.
- New test_same **192** balanced cases, depths 3-4.
- New test_deep **192** balanced cases, depth 5.
- Each test split has four previously unused families:
  `tight_alternative_and`, `shared_bridge`,
  `many_headed_goal`, `recursive_lures`,
  each contributing 24 valid and 24 invalid records.
- Root seed `20261113`. Source split offsets:
  train **506311**, dev **617533**, same **728747**,
  deep **839963**. Candidate seeds depend only on
  attempt index and split offset. Conservative WL
  fingerprints must be unique across all FC-02
  through FC-08, FC-09 train/dev, and FC-09 test.
  Reject structural duplicates without inspecting
  any model output or heuristic performance.

**Access isolation:** commit this registration first,
then train/dev generator and its exact CI. Freeze and
pin the train/dev-only commit *before adding the FC-09
test generator*. Model training CI uses that earlier
source and asserts the test module absent. After
both model checkpoints and stopping thresholds are
frozen, a dependent CI job checks out the separate
pinned test revision. Preserve source JSONL SHA-256
and exact source-CI receipts.

## Fixed LR-01 recipe

- Python 3.11, PyTorch 2.5.1 CPU, deterministic algorithms,
  two CPU threads, seeds **17** and **29**.
- Separate 11,521-param `LoopReach(4,width=24)` gate
  trained on original frozen FC-02 training cases:
  350 AdamW steps, batch48, lr=.002, weight_decay=.01,
  max grad norm1. Gate threshold from original
  `[.55,.70,.85,.93,.98]` grid on FC-02 dev **only**.
  Freeze it before anything from FC-09 test is visible.
- Independent LR-01 actor copied from that gate, with
  original 11,521-param tied trunk and 4,705-param
  rule-ranking head, jointly fine-tuned on FC-09
  **train-only positive shortest-proof prefixes**.
  300 AdamW updates, batch48, lr=.002,
  weight_decay=.01, clipping1, minibatch seed 6154+seed.
- Actor sees only original graph syntax, given facts,
  updated proved atoms, and goal. Mask to applicable
  novel-conclusion rules. Up to five model-chosen
  rule actions, charging **four recurrent cell
  invocations for each attempt**. No gold, minimum
  trace, family or source split enters inference.

All methods receive the *same* frozen FC-04 gate's
eligibility for a positive attempt. Negative claims
are emitted only with complete Boolean certificates
verified independently.

## Preregistered CPU-budget-matched symbolic primary control

An AND/OR planner searches *sets of already proved
atoms*. Each expansion applies exactly one source
rule whose premises are all known and whose head is
novel. It uses a min-heap with priority
`(g + h_max(known), g, insertion_id)`, where g is
the number of selected rules and `h_max` is the
relaxed conjunctive max-distance to the goal:

- Every known atom has cost 0.
- Repeatedly relax each Horn rule with
  `cost(head) <= 1 + max(cost(p) for p in premises)`.
- Initialize unknown atoms to infinity, propagate
  until finite-cost fixed point. This is a lower
  bound on additional *number of novelty actions*
  because shared subproofs may be reused. If the
  goal is already known, h_max=0.
- A state's successors are visited in the original
  source rule order. Deduplicate atom sets. Disallow
  paths longer than the fixed five-action budget.
- Stop on verified goal, heap exhaustion, or when
  the *actual measured per-example search time*
  reaches its frozen allowance.

**Timing allowance:** for each positive-eligible
case, first run the frozen neural actor and record
*only the elapsed perf_counter time spent on its
four-pass rule-scoring calls*, excluding shared
gate, source encoding, Python certificate replay,
and Lean compilation. Supply that scalar time
to the symbolic search, without any model trace,
probability or answer. The planner may spend at
most that additional rule-scoring wall-time budget
on its own heap search; allow one first-state
check even if the budget is tiny and log all
actual overrun. The search checks the deadline
**before every expansion**; runtime jitter and
overshoot must be reported separately.

This is per-item **measured CPU-runner wall time**,
not formal exact FLOP equality. Run all comparisons
in a pinned two-thread CPU GitHub Actions job;
record model-specific proof-scoring milliseconds,
symbolic search milliseconds, popped states,
budget overruns and effective time ratio.
Measure shared gate and both exact checkers
separately. The policy may not inspect model
predictions even though its allowed runtime is
taken from the actor's measured overhead.
No batch of neural tasks may be used to tune the
time budget or model hyperparameters.

## Secondary controls and cost accounting

- `bounded_24`: unchanged original FC-08
  24-state best-first as a historical comparison.
- `backward_slice`, `unary_bfs`,
  `random_legal`, `goal_first`: the
  exact old source-only methods.
- `exact_shortest_BFS`: unrestricted symbolic
  oracle, 100% correct by construction, excluded
  from primary neural efficacy claims.
- An independent `hmax_24` comparator uses
  the same new AND/OR heuristic but has a fixed
  24-pop cap instead of a runtime limit.
  This distinguishes improved heuristic quality
  from the CPU-matching allowance.

All procedures retain original source-rule lists
and full failed traces for independent checker
replay. Every accepted model positive is
compiled by Lean 4.19.0 in the separate test CI.

Report for each seed/test split/family:
positive-eligible count; verified positive
certificates out of 96 valid; verified negative
countermodels out of 96 invalid; false formal
certificates; abstentions; paired model-only
versus symbolic-only successes on identical
source cases; exact symbolic popped-state
counts; recurrent cell calls; actual model
forward and symbolic CPU-runner wall time
per case; time allowance used and overrun;
Python checker time; cost per positive
certificate; trace and certificate sizes.
Publish raw JSONL, all hashes and Lean logs.

**Primary learning criterion:** the learned actor
certifies more positive goals than **CPU-time-
matched AND/OR h_max search** in both test splits,
for both seeds, with zero false formal
certificates. Report mixed/negative if any
condition fails.

**Independent CPU-efficiency criterion:** the
learner must achieve a lower *total measured
wall time per certified positive* (shared gate
+ candidate search + exact verification) than
the symbolic primary control on both splits
and seeds. Do not conflate that with nominal
runtime matching of only the extra actor.
If an exact unrestricted BFS solves every case
faster, report it explicitly as a substantive
limit on neural computational value.

The test labels are public, the synthetic
grammar is narrow, and all source premises
are already formal. No claim about paper
proof validity, RH, Collatz or P≠NP follows.
