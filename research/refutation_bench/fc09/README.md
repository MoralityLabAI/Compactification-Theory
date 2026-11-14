# FC-09: CPU-budget matching for conjunctive proof selection

The [preregistration](PREREGISTRATION.md) was committed as
`37ceea8c6e5310fb9fb6fb43fd2aab1e9f3210a4`, before
FC-09 source generation, neural training or model test scores.

The motivation is FC-08: a trained tied LR-01 actor successfully found
more short proofs than a 24-popped-state symbolic search, yet its
extra repeated neural forwards cost more CPU per certified proof
in most comparisons. FC-09 includes a stronger source-only
**AND/OR h-max** search and a separate **per-case measured
actor-rule-scoring-time allowance**. No model prediction,
rule list, gold verdict or reference proof is passed to the
symbolic planner; it receives only a time allowance scalar.

## Exact formal input

There are 12 atoms per graph, one given fact, an unreachable guard,
and 10 reachable novel conclusions in every valid example. The
valid goal has a shortest proof of three, four or five novel Horn
rule applications, requires at least one conjunction, and has
no unary-only proof. Shared prerequisites and competing producers
for goal ancestors provide misleading alternatives. Negative
tasks guard all goal producers and have complete checked
Boolean countermodels.

| Split | Records | Minimum proof depth | Source families |
| --- | ---: | --- | --- |
| Train | 768 | 3–4 | convergent_and, crossed_dependencies, cyclic_shared |
| Development | 192 | 3–4 | nested_conjunction |
| Test familiar | 192 | 3–4 | tight_alternative_and, shared_bridge, many_headed_goal, recursive_lures |
| Test deep | 192 | 5 | The same four unseen family names, with disjoint instances |

All source labels pass independent Horn closure and exhaustive Boolean
truth tables. Positive reference proofs derive the goal from exactly
the explicit hypotheses and rules and are compiled by Lean 4.19.0.
Invalid records carry complete Boolean witnesses checked against
every premise. Conservative WL structural fingerprints are disjoint
across FC-02 through FC-08, FC-09 train/dev, and the held-out tests.

The train-only revision was fixed **before** test_data.py or the
new test control code existed. Model CI checks out that revision
and asserts both are absent. A dependent test job reads the
immutable checkpoint hashes before checking out the test revision.

| Source | Frozen commit | Canonical JSONL SHA-256 |
| --- | --- | --- |
| Train/dev only | `315119ead7264fbfbf5e58ee1b7b5c8b512d475d` | `680318ea334bba0777a048d3299a9e490207faa3a8ead549393616790a3761d5` |
| Held-out | `0b045fade18ca725ad9adc536ad48d6f58935fb3` | `89719bf1af2927fbd957f6735d15e4ec30598696b173a3a365bcd39bbfd5440d` |

[Successful train source CI #43448030483](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43448030483)
ran six new tests and Lean-compiled **480** valid source proofs.
[Successful test source CI #43449281226](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43449281226)
ran eight new tests, checked 192 negative assignments and
Lean-compiled **192** valid held-out reference proofs.
Endpoint axiom inspections showed zero extra Lean axioms.

An earlier source-only CI run #43447734094 failed because an
alternative goal producer shortened the intended minimum proof.
Its prerequisites were tightened **before** any held-out test source
existed or model training occurred. The exact shortest-proof
auditor now guards this property.

## First symbolic results, recorded before model testing

The source's [controls.py](controls.py) provides h-max A* over
sets of proved atoms. For any proof state, the heuristic derives
the relaxed shortest cost to an atom by repeated updates of
`cost(head) <= 1 + max(cost(prerequisites))`,
minimizing over its producers. The conjunction-aware value is a
lower bound on additional rule applications. Source rules are
expanded in original listed order, and repeated known-atom
sets are deduplicated.

Fixed at five output rule applications and at most 24 popped states,
the original source-only pre-model receipt reports:

| Control | Familiar positive /96 | Deep positive /96 |
| --- | ---: | ---: |
| Conjunction-aware h-max A* | **96** | **70** |
| Original FC-08 best-first | 91 | 22 |
| Goal-first | 57 | 10 |
| Backward goal slice | 57 | 11 |
| Random legal | 37 | 4 |
| Unary-rule BFS | 0 | 0 |
| Exact unrestricted shortest-proof oracle | 96 | 96 |

The held-out source cases were **not** changed after measuring these
controls. The zero-neural stronger A* is the key comparison.

## First frozen neural comparison

[First correctly pinned model CI #43449947846](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43449947846)
passed both dependent jobs and Lean-compiled **324
model-selected positive theorem terms**,
with another **383** verified negative
countermodels and zero false finite
formal certificates across 768
seed/case evaluations.

| Test group | Learned seed 17 | Learned seed 29 | CPU-budgeted symbolic h-max |
| --- | ---: | ---: | ---: |
| Familiar | 96/96 | 96/96 | **96/96** |
| Deep | 64/96 | 68/96 | **96/96** |

The **learning and CPU-efficiency criteria
both failed**. The symbolic AND/OR search
used only **9.2–13.5%** of its time allowance,
with zero timeout or overrun, and its
full CPU-runner time per case was
approximately 4.5–4.8 ms versus
9.2–11.3 ms for the learned actor.
The [first-run results ledger](../results/fc09_cpu_matched_andor_20261113.md)
records every paired comparison,
exact source hash, CPU budget and
Lean proof receipt. The time match
was a *maximum allowance*, and this
stronger symbolic algorithm solved
all valid cases well below it.

## Per-case CPU-time matching

The neural model follows the original FC-08 recipe: an 11,521-parameter
LR-01 frozen gate trained on FC-02, theta selected on FC-02 dev;
a separate 16,226-param copied LR-01 actor trained jointly on
FC-09 train-only shortest proof prefixes, with seeds 17/29 and
unchanged 350/300 optimizer step counts.

At positive eligibility, run the actor first and record actual
**rule-scoring model forward milliseconds**. The source AND/OR search
then gets *only that numeric time allowance*; it sees no neural
actions or probabilities. It checks the deadline before expansions
and during successor generation. The first root state may be
checked even when the time allowance is too small, and all
overshoots are measured. Compare accepted source-rule traces
under the same exact five-action checker, with independent
kernel compilation of the learner's accepted positive proofs.

Both policies pay the same frozen gate and negative verifier costs.
The runtime-match heuristic, fixed 24-pop h-max, old 24-pop
best-first, random, goal-first, backward ancestry and unary
BFS remain independent controls. Full CPU model/graph/checker
time and paired success outcomes are recorded in the test
artifact.

The experiment concerns public *already formalized* Horn logic.
It does not translate untrusted human mathematical manuscripts
into certified statements, and unrestricted symbolic BFS still
solves this finite fragment exactly.
