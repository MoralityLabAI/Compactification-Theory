# FC-07: misleading goal ancestry and a preregistered symbolic control

[FC-07 preregistration](PREREGISTRATION.md) was frozen in
`d3216371ae480b7e1b42cb913a33847ed5ad47c2`, before
the new generators, models or test scores.

The previous FC-06 graph had a unique producer for every
goal-chain atom, making a zero-parameter backward dependency
slice perfectly effective. FC-07 introduces alternative
producers for actual goal ancestors, shared decoy subgoals
and misleading cycles. Each distractor appears as a premise
of at least one alternative backbone rule, so the backward
slice includes branches the five-rule budget cannot afford.

## Exact data and source provenance

All source graphs have 12 atoms, one initial fact, exactly
ten reachable novelties in the positive cases and a strict
five-action limit. A positive goal always has a certified
minimum-length proof of d=2–5 actions. Negative problems
guard every producer of one essential backbone node and
have an independently verified complete Boolean countermodel.

| Split | Cases | Goal depth | Structural families |
| --- | ---: | --- | --- |
| Train | 768 | 2–4 | braided_ancestors, shared_subgoals, cycle_traps |
| Development | 192 | 2–4 | gated_forks |
| Test at familiar depth | 192 | 2–4 | reciprocal_forks, delayed_subgoal, false_shortcuts, nested_switches |
| Test at greater depth | 192 | 4–5 | Same four unseen families, different WL-disjoint cases |

All prior FC-02 through FC-06 cases are excluded through
conservative isomorphism-invariant WL fingerprint grouping.
Each test family supplies exactly 24 valid and 24 invalid
examples to both test splits.

- Train-only frozen source commit:
  `f470a97502b853addf2b0f67c9f6225001074a90`;
  canonical train/dev JSONL SHA-256
  `f725c31a18aa84b2b831b90d8d2f6ca62c2d20d40530b8f8284429d8c80e1359`.
- Test source frozen **after** train-only commit:
  `6cee4fdba91ba9c13ad47c492a127c7dc1884090`;
  test JSONL SHA-256
  `f7481ed2f489c43eb95fe74b1544aad3ef59898b29d10fd1455874ad990d4cdb`.
- [Train-only source CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43421898810)
  passed six tests, compiled 480 positive Lean 4.19.0
  theorems, and checked 480 negative countermodels.
- [Test source CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43422531533)
  passed seven tests, compiled 192 positive Lean theorems,
  and checked 192 negative countermodels.

All positive reference claims are parameterized by their
original Horn hypotheses and implication rules; two audited
endpoint Lean declarations report no extra axioms.
The independent propositional checker is the formal authority
for negative certificates.

## Symbolic baseline registered before model evaluation

The **primary** non-neural comparison is the exact
backward-ancestry policy from FC-06, applied with the
same five-action cap: build a syntactic backward
dependency set from *all* source implications, then
choose the lowest-index legal rule whose conclusion
belongs to that set.

The frozen source-level audit, before model evaluation,
reported the following initial outcomes:

| Proof selector | Same-depth correct /96 valid | Deeper correct /96 valid |
| --- | ---: | ---: |
| Backward goal ancestry | **47** | **22** |
| Exact shortest-proof oracle | 96 | 96 |

This proves that FC-07's ancestry selection has become
nontrivial. The full source result was kept without
selecting test cases based on a desired neural/heuristic
gap. Random legal and goal-first controls are also
included in the paired neural evaluation.

## First trained-student outcome

[First frozen CI #43422916239](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43422916239)
kernel-compiled **352 student-proposed positive proof
terms**, with 379 exact negative countermodels across
two seeds and two held-out test groups. The learned
actor proved 94/96 and 93/96 positives at familiar
depth, and 85/96 and 80/96 at greater depth.
The preregistered backward-slice control proved
47/96 at familiar depth and 22/96 or 21/96
at greater depth under the same stopping gate.
No false finite formal certificates were issued.

The [first-run research ledger](../results/fc07_misleading_ancestry_20261113.md)
records all paired positive wins, false-proof
checks, CPU and recurrent-call costs.
The **learning criterion passed** on both seeds
and depths; the separate **compute-cost
criterion failed**, because backward slicing
was cheaper per certified positive at familiar
depth, even though the student was more
efficient at greater depth.

## Post-hoc stronger control

[The post-hoc unary-implication BFS](POSTHOC_UNARY.md)
restricts the graph to one-premise Horn rules
and finds a proof path from the initial fact to
the goal. The canonical backbone uses only
single-premise rules, while every alternate
backbone producer needs a conjunction.
That zero-parameter control certified
**192/192 distinct positive test cases**
and compiled them in [separate Lean CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43424550518).
It was introduced after the first registered
FC-07 model evaluation, which remains
unchanged. Consequently, the learned
model beats the registered backward
slice but is weaker than this simpler
symbolic path algorithm.

## Reproduction

From the repository root:

~~~sh
python -m unittest research.refutation_bench.tests.test_fc07_train -v
python -m unittest research.refutation_bench.tests.test_fc07_test -v

python -m research.refutation_bench.fc07.train_data \
    --out /tmp/FC07_train.jsonl \
    --lean /tmp/FC07_train.lean \
    --receipt /tmp/FC07_train_receipt.json

python -m research.refutation_bench.fc07.test_data \
    --out /tmp/FC07_test.jsonl \
    --lean /tmp/FC07_test.lean \
    --receipt /tmp/FC07_test_receipt.json

lean /tmp/FC07_train.lean
lean /tmp/FC07_test.lean
~~~

The private LoopedTransformers FC-07 branch uses the same
source-pinned FC-06 model recipe, with newly trained
source-family-isolated actor weights and a separate
evaluation job. Its accepted positive rule sequences
must pass the exact FC-05 replay checker and compile
as Lean terms from the *original proposed actions*.
Every issued negative conclusion must carry a verified
complete Boolean assignment.

The exact shortest-proof BFS solves these small Horn
tasks by design. Results about the neural policy concern
finite graph search under a strict budget and are
separate from evaluating real mathematical manuscripts.
