# FC-08 preregistration: conjunctive competing proof search

**Protocol frozen 2026-11-13, before FC-08 generator construction, training or test inspection.**
The first run is primary. Repairs after a result must preserve the original test, source hashes,
and raw first-run outcomes; any new comparisons are explicitly post-hoc.

## Hypothesis and limits

FC-07's model beat the preregistered backward dependency slice, but a later
zero-parameter BFS over unary implication rules found a complete proof on
every held-out valid graph. FC-08 removes the guaranteed unary goal path:
every valid goal must require **multiple antecedents** and competing
alternative derivations with shared premises. Shortest proofs remain
explicitly checkable in Lean and bounded at **five rule applications**.

The primary question is whether an LR-01 tied recurrent actor selects
a certified proof more frequently than the frozen **compute-bounded symbolic
best-first** baseline under the same five-rule output budget, for every seed
and both held-out splits. A secondary question compares cost per certified
proof, charging the actor's repeated 4-cell calls and the symbolic
baseline's graph search time.

These are public, deterministic propositional Horn tasks, with an exact
symbolic solver already available. Results do not transfer by themselves
to proofs claimed in real mathematical manuscripts.

## Source grammar and verifier contract

Every task has 12 atoms: one initial fact, a canonical proof backbone
of depth **d = 3, 4, or 5** with branching and at least one irreducible
AND-rule, exactly `10-d` reachable distractors, and one unreachable guard.
All **ten** non-guard atoms beyond the fact are reachable for valid tasks.
The positive goal has shortest proof length exactly d, verified by
independent BFS, and every valid goal derivation must contain at least
one conjunctive rule. There must be **no path to the goal using only
one-premise implications**.

Decoys appear as premises of alternative producers for actual goal
ancestors. Alternative producers must not collapse to one-premise
shortcuts, and must not create a sub-five-step bypass of the registered
canonical minimum. They may share useful prerequisites and form benign
or unreachable cycles. The canonical rule for the goal itself must
have at least two distinct prerequisites. The test code must validate
that all distractors belong to syntactic backward ancestry of the goal.

For invalid cases, all rules producing the goal have an extra
unreachable guard premise. This gives a concrete exact Boolean
countermodel while keeping the rest of the graph satisfiable.

At most 12 atoms and 20 Horn implication rules. Independent operational
Horn closure and exhaustive truth-table entailment must agree on all
labels. Every positive reference proof compiles under Lean 4.19.0 with
explicit assumptions and without `sorry`; negatives have complete
assignments that the independent checker verifies. The inference proof
contract remains the FC-05 strict replay checker with `budget=5`;
an invalid original rule sequence is rejected, never repaired. Model
positives must additionally compile under Lean 4.19.0 from the *same*
model-selected rule sequence. Negative releases require independently
checked complete Boolean countermodels.

## Source splits and leak control

- Train **768** balanced cases, depth 3–4, three grammar families:
  `braided_and`, `shared_fork_and`, `cycle_and`.
- Dev **192** balanced cases, depth 3–4, distinct family `cross_and`.
- Test familiar **192** balanced cases, depth 3–4.
- Test deep **192** balanced cases, depth **5**.
- Each test set contains four unseen grammar families, with 24 positives
  and 24 negatives per family:
  `paired_requirements`, `nested_and_switch`,
  `goal_misdirection`, `multi_support_cycle`.
- Root deterministic seed **20261113**, source offsets train **101311**,
  dev **202627**, test-familiar **303941**, test-deep **405179**.
  Attempts produce deterministic graph candidates; WL duplicate
  fingerprints are rejected against every FC-02 through FC-07 source
  and all other FC-08 records, without looking at model performance.
- The test generator must be a **later commit** than a frozen
  train/dev-only revision. The train GitHub Actions job checks out
  only that earlier commit and asserts test source absence.
  Its checkpoint hashes and stopping thresholds are frozen before
  a dependent evaluation job accesses the pinned test generator.

No model checkpoint, hyperparameter or inference pass count may be
selected from FC-08 test outcomes. Only FC-02 development selects the
original FC-04 gate threshold.

## Frozen actor training

- Python 3.11, PyTorch **2.5.1 CPU**, deterministic algorithm mode, two
  CPU threads; seeds **17** and **29**.
- Gate: original LR-01 four-pass `LoopReach(width=24)` (11,521 params),
  trained on frozen FC-02 train only for 350 AdamW updates, batch48,
  lr .002, weight_decay .01, gradient norm max1. Freeze gate weights.
  Calibrate theta only on FC-02 dev from
  `[0.55,0.70,0.85,0.93,0.98]` with FC-04's registered utility.
- Actor: separate copied LR-01 four-pass cell with 4,705-parameter
  goal-conditioned rule head, **16,226 total actor parameters**,
  jointly fine-tuned on FC-08 train-only *positive* shortest-proof
  prefixes (minimum BFS source teacher is training-only).
  AdamW 300 updates, batch48, lr .002, weight_decay .01, gradient
  norm max1, minibatch seed 6154 + model seed.
- At test time action selection sees only facts, source rule incidence,
  goal, and current-known state, with a legal novel-rule mask.
  No gold, family ID, source proof trace or minimum proof length enters
  graph inputs. Every proof action costs four recurrent passes.
  At most **five original source-rule actions** are selected.

## Preregistered non-neural baselines

All arms share *identical* frozen FC-04 positive eligibility and
negative-countermodel verification. The positive action budget is 5:

1. `learned_joint`: the jointly trained LR-01 policy; a new graph
   forward pass at each candidate legal rule choice.
2. `random_legal`: uniform random among currently legal novel rules,
   deterministic per case (2917 + SHA256(case_id)).
3. `goal_first`: an immediately applicable goal-producing rule if
   any; otherwise the same random legal selector.
4. `backward_slice`: FC-06 syntactic goal ancestry and original
   lowest-index currently legal rule with head inside the ancestry.
5. `unary_bfs`: shortest BFS in the graph of one-premise rules only.
   It must return no positive certificate for this test by construction.
6. `bounded_best_first`: symbolic state-space search with maximum
   **24 popped states** from a fixed min-heap. States comprise the
   set of proved atoms plus original rule-ID trace. Apply only legal
   rules with novel heads until either the goal is reached within
   five actions or 24 states have been expanded. Priority is
   `(steps_so_far + relaxed_distance(goal), steps_so_far, tie_ID)`,
   where relaxed distance ignores conjunction by treating any one
   antecedent as sufficient, computed once on the full source
   graph. Equal priorities break on insertion order, with legal
   successor rules considered in their original listed order.
   No gold labels or reference traces enter search. Report actual
   states popped, graph time, failed traces, and budget misses.
7. `exact_shortest_oracle`: exhaustive BFS with no state-expansion
   cap, source truth/reference only; **never used by neural inference**.

The symbolic best-first search has no neural cell calls; all procedures
still pay the same frozen gate calls and exact checker time. Time costs
and rule-space expansions are reported separately to make this
comparison intelligible.

## Registered metrics and success

Each seed/test family/depth must report positive-eligible count,
certified positives out of 96 valid, certified negative countermodels
out of 96 invalid, abstentions, false certificate counts, mean
model actions, failed-trace first-fault reasons, paired per-case
learned-only vs baseline-only successes, mean and p95 recurrent
cell invocations, symbolic popped-state counts, model milliseconds,
graph-search milliseconds, exact checker milliseconds, and Lean
compilation counts. Save raw source-rule traces and predictions
including failures.

**Primary learning criterion:** the learned actor certifies **more
eligible positive cases** than the *compute-bounded best-first*
baseline on **both** held-out test splits for **both seeds** with
zero false finite formal certificates. If any comparison fails,
report mixed/negative. Compare other baselines as secondary.

**Compute criterion:** the learned policy must also have a **lower
total measured forward/search/checker CPU cost per certified
positive** than the bounded-best-first baseline across those four
conditions; report it independently of learning. Never call an
exact shortest-proof oracle result a neural win.

The test corpus is public/preregistered, and all valid cases are
solvable exactly in this bounded Horn fragment. Inference cannot
verify mathematical semantics of an arbitrary paper without a
separately verified source-to-formula representation.
