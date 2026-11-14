# FC-08: first preregistered conjunctive proof-search experiment

**13 November 2026. First full frozen CI: passed.** The LR-01 actor
beat the preregistered **24-state compute-bounded best-first** baseline
on positive certificate count for each of two seeds and both unseen
test depths. The separate preregistered CPU-efficiency criterion
**failed** in three of four conditions. These are public, already
formalized 12-atom Horn proof tasks, with an exact shortest-proof
symbolic oracle available.

## Registration and exact source provenance

The first protocol commit, b6c8ffbef6606e89737bbfc32d73b8732b765e8d,
preceded both the source generator and model training. It fixed
the five-action cap, the four test-family grammars, model seeds,
training budget, original negative/positive stopping gate and
primary comparator: best-first symbolic search with **24 popped states**.

- Train/dev-only source commit:
  51d0082f27e4c6214ec0cebb4886df44a4962321
- Canonical 960-case train/dev JSONL SHA-256:
  0d51135f606d44085f2bf0643e44469c36cbebae2469c5ceb8dcbb52b75f763a
- Test source commit:
  1408daf07c34a133a794662c31f8fbdf789ac840
- Canonical 384-case test JSONL SHA-256:
  3b7d93c04e6565fc8e46e30947c0347d75ad6b61d575aeb7a598e6c189969dd1

[Exact train source CI #43442906680](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43442906680)
passed six regression tests, verified 480 negative countermodels
and Lean-compiled all **480** positive reference theorems.
[Exact test source CI #43444039591](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43444039591)
passed seven source tests, checked 192 negative countermodels
and compiled all **192** valid test theorem references.
Audited endpoint theorem declarations report no extra Lean axioms.

Every positive goal requires a conjunction, with a shortest
source proof of three, four or five novel rule actions.
The graph contains ten reachable novel consequences plus
an unreachable guard. Each distractor participates in an
alternative producer of the actual goal, and **no unary-only
path can derive that goal**. Positive and negative source
families are WL-disjoint from prior FC-02 through FC-07
and from FC-08 train/dev.

## Source-only symbolic controls measured before model inference

The following outcomes were frozen in the test-source
receipt before the two-job model experiment. Each split
contains **96 valid** and **96 invalid** entailments.

| Rule selector, 5-action cap | Familiar depth, valid /96 | Deeper depth, valid /96 |
| --- | ---: | ---: |
| Best-first, at most 24 states expanded | **89** | **28** |
| Goal-first legal rule | 68 | 18 |
| Random legal rule | 53 | 5 |
| Backward goal ancestry | 48 | 9 |
| Unary-only BFS | 0 | 0 |
| Exact shortest-proof solver (oracle upper bound) | 96 | 96 |

The test corpus was kept without model-score-informed filtering.
The best-first graph priority uses source-only relaxed distances
that ignore conjunction when calculating a lower bound, with a
24-state cap and the same five-action output limit. It remains
a **heuristic search**, and exact unrestricted BFS solves all
finite examples.

## Frozen real LR-01 training

[First successful model CI #43444713589](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43444713589)
passed two dependent jobs. The training job checked out
**only** the train/dev source revision above, asserted
that test_data.py and controls.py did not exist, and
wrote SHA-256-identified gate/actor checkpoints.
Its dependent evaluation job then checked out the
pinned test source, used that source's exact
best-first implementation, and compiled every
accepted model-selected positive proof in Lean 4.19.0.

For both seeds **17** and **29**, a separate
11,521-parameter tied LR-01 gating model trained on
FC-02 train and selected theta=**0.55** using FC-02
development alone. A **16,226-parameter** proof actor
copied the recurrent trunk and trained its 4,705-param
rule-ranking head jointly with its 11,521-param trunk.
The actor received **1,344 training-only positive
proof-prefix states**, with 300 AdamW steps after
the gate's 350 fixed steps. Both models therefore
have **27,747 neural parameters combined**.

The in-sample first-128-prefix next-rule accuracies
were **81.25%** (seed17) and **84.375%** (seed29).
The checkpoints were frozen before test access;
no held-out label entered the action-ranking policy.

At inference, positive outputs were accepted only
after independent rule-by-rule trace replay under
the exact five-action bound, then Lean compilation
of their original model-generated rule IDs.
Negative outputs required a complete independently
verified Boolean assignment satisfying every
premise and falsifying the conclusion.

## First held-out model versus bounded symbolic search

Both the symbolic controls and actor were evaluated
only on *the same* positive-eligible cases under
identical frozen gating decisions.

| Seed / test depth | Eligible positives /96 | Learned positive Lean proofs | Bounded best-first positive proofs | Learned exact negatives /96 | Learned abstentions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 / familiar | 96 | **94** | 89 | 96 | 2 |
| 17 / deeper | 96 | **72** | 28 | 96 | 24 |
| 29 / familiar | 96 | **96** | 89 | 92 | 4 |
| 29 / deeper | 96 | **90** | 28 | 96 | 6 |

The learned actor produced **352 Lean-compiled positive
theorem terms** and **380** verified negative
countermodels across the four conditions.
There were **36 abstentions** in 768
seed/case evaluations, and **zero false finite
formal certificates**. The 352 positive terms may
include proofs of identical underlying problems
by separate seeds. The symbolic best-first
control was bounded at 24 popped graph states
per example, by preregistration.

The **registered learning criterion passed**:
learned positive proof count exceeded best-first
in both test splits for both seeds.

### Paired successes on identical valid examples

| Seed / test depth | Learned-only proof successes | Best-first-only successes | Both certify |
| --- | ---: | ---: | ---: |
| 17 / familiar | 5 | 0 | 89 |
| 17 / deeper | 45 | 1 | 27 |
| 29 / familiar | 7 | 0 | 89 |
| 29 / deeper | 62 | 0 | 28 |

The small same-depth net gains of five and seven
are more modest than the deeper gains. Only two
model seeds were evaluated, and these are
reproducible synthetic tasks with public labels.

## Full recurrent and CPU-cost accounting

The recurrent actor performs four additional
tied-cell passes at each selected action.
The bounded best-first baseline performs
source-graph search without a neural rule
scorer; both pay the same frozen FC-04
gating inference and exact certificate costs.

| Seed / split | Learned cells /192 | Best-first cells /192 | Learned CPU ms/case | Best-first CPU ms/case |
| --- | ---: | ---: | ---: | ---: |
| 17 / familiar | 2,450 | 1,078 | 3.619 | 1.619 |
| 17 / deeper | 2,926 | 1,006 | 4.323 | 1.533 |
| 29 / familiar | 2,472 | 1,096 | 3.731 | 1.637 |
| 29 / deeper | 2,918 | 998 | 4.051 | 1.406 |

The exact symbolic baseline popped **1,282** states
across the 96 valid familiar-depth cases and
**2,183** states across the 96 valid deep cases
under the 24-state-per-case limit.

Charging the total measured source-graph/neural
forward plus Python checker CPU time:

| Seed / split | Learned CPU ms / positive proof | Best-first CPU ms / positive proof |
| --- | ---: | ---: |
| 17 / familiar | 7.391 | **3.492** |
| 17 / deeper | 11.527 | **10.511** |
| 29 / familiar | 7.462 | **3.531** |
| 29 / deeper | **8.642** | 9.639 |

The **registered compute-efficiency criterion failed**
in three of four comparisons. The model's
additional recurrent calls cost more CPU even when
they improved positive certificate recall.
One deeper seed (29) amortized the compute
over enough extra successful proofs to beat
best-first CPU per certificate.

These timing numbers came from a single hosted
CPU run with two pinned threads and torch 2.5.1.
They are not a cross-device speed benchmark.

## Artifact audit and interpretation

The downloaded CI artifact includes **24 raw prediction
JSONL files of 192 rows each** (4,608 decisions),
the full score receipt, an exact model-generated
Lean source file and the kernel log. A separate
archive verification checked every output case ID
for uniqueness within its file, every proposed
rule list against the five-action cap, the
positive certificate flags and the Lean source
SHA256. All checks passed; **352** explicit
theorem declarations are present.

FC-08 demonstrates that a trained tied recurrent
rule policy transfers to new **conjunctive** Horn
structures and outperforms a preregistered
bounded symbolic best-first search in certified
positive coverage. The result depends on the
symbolic comparator's **24-pop search budget**
and the actor's extra compute. It does not
show advantage over unlimited exact BFS,
which is already 100% correct and may be
computationally cheap on twelve atoms.

The next stronger comparison should
preregister *equal CPU time or expansion
cost* rather than equal output-rule budget
alone, and include more sophisticated
symbolic AND/OR hypergraph planning.
All real-manuscript claims still need a
verified source-to-formal semantic bridge.
