# FC-09: first registered CPU-time-budgeted AND/OR comparison

**Date:** November 13, 2026. **First correctly pinned two-job CI passed.**
The stronger symbolic AND/OR h-max search **outperformed the trained
LR-01 actor** in proof coverage and total measured CPU-runner time.
Both preregistered neural success criteria **failed**. The exact
certificate gate retained zero false finite formal certificates.

## Preregistration, train/test isolation and exact source

- FC-09 registered **before** source generation, model training or
  test inspection at commit
  `37ceea8c6e5310fb9fb6fb43fd2aab1e9f3210a4`.
- Frozen train/dev-only source:
  `315119ead7264fbfbf5e58ee1b7b5c8b512d475d`;
  canonical 960-case JSONL SHA-256
  `680318ea334bba0777a048d3299a9e490207faa3a8ead549393616790a3761d5`.
- Frozen test source:
  `0b045fade18ca725ad9adc536ad48d6f58935fb3`;
  canonical 384-case JSONL SHA-256
  `89719bf1af2927fbd957f6735d15e4ec30598696b173a3a365bcd39bbfd5440d`.
- [Exact train-source CI #43448030483](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43448030483)
  passed six tests and Lean 4.19.0 compilation of **480 positive**
  reference theorems.
- [Exact test-source CI #43449281226](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43449281226)
  passed eight tests, checked 192 complete negative
  countermodels and Lean-compiled all **192 positive**
  test reference theorems.
- [First correctly pinned model CI #43449947846](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43449947846)
  passed the train-and-freeze and dependent held-out
  evaluation jobs, including Lean compilation of
  **324 actual model-selected positive proof terms**.
- The immediately preceding model CI #43449815264 had
  a mechanical unfilled test-digest placeholder and
  stopped before publishing any valid held-out scores.
  Its training-only job passed. The correction consisted
  solely of pinning the source test commit and canonical
  dataset SHA; it did not change model weights, scoring
  algorithms, source cases or preregistered criteria.
- An initial source-only CI #43447734094 detected an
  alternate producer that shortened the registered
  minimum proof. The generator was repaired **before**
  any held-out generator or model training existed.
  The corrected train source was then frozen first.

The new data have 12 atoms, a guaranteed shortest
three-to-five-step proof of a positive goal, multiple
competing conjunctive goal producers, and ten reachable
novel conclusions. There is no unary-only proof path.
Invalid claims admit complete exact countermodels.
Every positive theorem is checked from the graph's
explicit assumptions; no extra axioms were found in
the two endpoint Lean audits. All candidate training
and held-out source WL fingerprints are disjoint.

## Frozen neural recipe

Both seeds **17** and **29** trained a separate
11,521-parameter four-pass LR-01 verdict/eligibility
gate on old FC-02 train, calibrated exclusively
on FC-02 development at theta **0.70** and **0.55**.
A separate **16,226-parameter** joint LR-01
proof actor copied the original tied cell and
trained a 4,705-parameter goal-rule scorer jointly
with its 11,521-parameter trunk using **1,344
training-only positive proof-prefix states**
and 300 fixed AdamW updates.

The frozen actor's first 128 in-sample training-prefix
accuracy was **77.3438%** for seed17 and **79.6875%**
for seed29. This is a training diagnostic,
not a held-out performance measurement.

The model receives only source graph syntax,
current known set and goal; reference proofs,
source labels and family metadata are excluded
from the model's inputs. The original source
test checkout became available only in a
dependent job after SHA256-identified weights
were uploaded.

## Conjunction-aware symbolic search and actual CPU allowance

The primary comparator is h-max AND/OR A*
over currently derived atom sets. For each
state it relaxes a Horn rule according to

```
cost(head) <= 1 + max(cost(premises))
```

and minimizes over alternative rule producers.
The cost is a lower bound on novelty actions
needed to derive the goal, with admissible
sharing of subproofs. Source rules are
enumerated in their original order, and
duplicate known-atom sets are deduplicated.

For each positive-eligible test case, the
symbolic planner receives a **single scalar**:
the actor's measured forward-pass
rule-scoring wall time. It receives none of
the actor's predictions or proposed actions.
It may take no more than five source-rule
actions and checks the time deadline before
expanding each new state, recording overrun.
Both methods additionally pay the same
FC-04 gate and exact certificate-checker cost.

The symbolic planner actually used **only
9.2–13.5%** of its available rule-scoring
time across the four test conditions.
It encountered **zero timeouts and zero
deadline overruns**. Consequently the
CPU-time allowance was *nonbinding* for
these symbolic instances. This is an
important qualification: the experiment
establishes that the symbolic planner is
both more effective and much faster on
this graph family; it is not a measurement
of neural and symbolic methods forced to
consume exactly equal CPU time.

## First frozen test outcomes

Each test split contains 96 true entailments
and 96 false entailments. Both policies
receive the *same* positive eligibility and
negative-verification signals from the
frozen model gate.

| Seed / test depth | Positive eligible /96 | Learned positive Lean proofs | CPU-budgeted AND/OR positive proofs | Learned exact negatives /96 | Learned abstentions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 / familiar | 96 | 96 | **96** | 95 | 1 |
| 17 / deep | 96 | 64 | **96** | 96 | 32 |
| 29 / familiar | 96 | 96 | **96** | 96 | 0 |
| 29 / deep | 96 | 68 | **96** | 96 | 28 |

Across 768 seed/case decisions the learner
produced **324 kernel-compiled positive
proof terms** and **383 complete, verified
negative countermodels**. It abstained on
**61** decisions. Every released positive
theorem followed the original model-selected
five-or-fewer-rule trace, passed strict
independent replay and compiled under
Lean 4.19.0. All verified negative claims
had complete satisfying premise assignments
with false conclusions. **Zero false formal
certificates** passed either gate.

The primary **learning criterion FAILED**:
the model never exceeded CPU-matched
AND/OR success. Deep cases had zero
learned-only wins and **32 and 28 symbolic-only
wins** for seeds 17 and 29. On familiar
depths, both succeeded on the same
96/96 positive cases.

### A separate symbolic expansion-cap sensitivity

The fixed source-only control, measured
*before* model evaluation, found:

| Fixed 24-pop control | Familiar positive /96 | Deep positive /96 |
| --- | ---: | ---: |
| Conjunctive h-max A* | **96** | **70** |
| Previous FC-08 relaxed best-first | 91 | 22 |
| Goal-first | 57 | 10 |
| Backward ancestry | 57 | 11 |
| Random legal | 37 | 4 |
| Unary-path BFS | 0 | 0 |

Under the actor's measured CPU-rule-scoring
allowance the h-max planner was permitted
to expand beyond 24 states, reaching
**96/96 deep** while consuming only
about **2.0 milliseconds of graph-search
time per positive-eligible case**.

## CPU and memory-relevant accounting

The model performs four tied-cell
forward passes per proof-selection action.
The graph planner is pure Python over
12-bit known-atom sets, and its
Python time is separately charged.

| Seed / depth | Learned model + checker ms/case | AND/OR model-gate + search + checker ms/case | Symbolic search ms/eligible case | Symbolic time used / allowed |
| --- | ---: | ---: | ---: | ---: |
| 17 / familiar | 9.176 | **4.478** | 0.964 | 9.42% |
| 17 / deep | 11.030 | **4.671** | 1.974 | 13.46% |
| 29 / familiar | 9.370 | **4.622** | 0.960 | 9.20% |
| 29 / deep | 11.328 | **4.767** | 2.018 | 13.36% |

Mean total elapsed CPU-runner wall milliseconds
**per verified positive certificate**:

| Seed / depth | Learned ms/proof | Symbolic ms/proof |
| --- | ---: | ---: |
| 17 / familiar | 18.35 | **8.96** |
| 17 / deep | 33.09 | **9.34** |
| 29 / familiar | 18.74 | **9.24** |
| 29 / deep | 31.99 | **9.53** |

The preregistered **compute-efficiency
criterion FAILED** in all four conditions.
The h-max planner certifies more deep
proofs with fewer total milliseconds
and without a neural proof actor.

All times were measured on one pinned
two-thread CPU GitHub Actions runner
using perf_counter wall time. They
are host-dependent and do not
establish a general hardware-neutral
FLOP comparison.

## Independent replay of archived output certificates

An additional post-run container audit independently
loaded the **original source-test CI ZIP**
and the **original model-evaluation CI ZIP**,
without using the model runner or its internal
source certificate-checker implementation.

It checked all **6,144 response rows** from
32 distinct per-policy JSONL files,
requiring exactly 192 unique case IDs
per output. For each released positive
certificate it replayed the original
source-rule IDs in order, checking every
antecedent, novelty, goal derivation and
the five-step cap. For every negative
certificate it required an assignment
of all twelve atoms, all facts true,
all Horn implications satisfied and
the asserted goal false. The results
agreed with the independent source
labels throughout: **324 valid model
positive proofs, 383 full negative
countermodels and zero false
formal certificates**.

For the 4 CPU-budgeted symbolic conditions,
every `symbolic_allowed_ms` matched
the corresponding frozen model's
`actor_forward_ms` for the same case.
There were **zero overrun cases**.
The audit also confirmed that the
generated Lean file has **324 actual
theorem declarations** and the
SHA256 specified in the original score
receipt. It did not substitute any
source reference theorem to
repair a model trace.

The [machine-readable independent
audit receipt](fc09_independent_artifact_audit.json)
is committed alongside this report.
Its original ZIP sources remain in
their separately linked CI artifacts.

## Artifact audit and limits

The full successful-run artifact
`fc09-cpu-matched-43449947846`
contains **32 per-policy, seed and
depth prediction JSONL files**,
each with 192 case rows, for
**6,144 original response records**.
An independent archive audit verified
every file's row count, the 324 model
theorem declarations, the exact Lean
source SHA-256
`20ea7e2c1aa3e9c55de3a9b07eb83c01380b0d9fb63b26e3d80ec6aadccf519a`,
and the kernel log's two no-extra-axiom
endpoint checks. The JSON report also
preserves every source-family score,
paired proof success count, time
allowance, actual search time,
overrun, search states popped and
raw failed action list.

FC-09 is a **clear negative result
for learned computational superiority**
over a source-only conjunctive AND/OR
planner in this finite, already
formalized Horn family. Even with
limited search-time allowance the
symbolic procedure solves all
valid entailments and is cheaper.
The original LR-01 actor can
produce formal positive and negative
certificates, but exact symbolic
planning is preferable for these
twelve-atom tasks.

This result says nothing about
the mathematical validity of a
human's Riemann, Fermat, Collatz
or P≠NP manuscript. It also
does not characterize larger,
more expressive logics where exact
search may become expensive.
A future trial should grow the
number of atoms, introduce
stronger interference between
AND/OR subproofs, and benchmark
optimized symbolic solvers and
neural proposals on genuinely
held-out theorem-prover tasks
under fixed hardware/cost budgets.
