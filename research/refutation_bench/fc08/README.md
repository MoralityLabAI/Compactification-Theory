# FC-08: conjunctive goal search and preregistered bounded best-first

FC-07's original neural model beat a preregistered backward-ancestry
graph selector, but a later zero-parameter unary-implication BFS found
all source proofs. FC-08 removes that shortcut, requiring actual
conjunctive premises for every positive goal.

The [FC-08 protocol](PREREGISTRATION.md) was frozen as commit
`b6c8ffbef6606e89737bbfc32d73b8732b765e8d`
**before** either new source generator or model trial. The train/dev
source was certified and pinned before a separate test module existed.

## Source corpus

All graphs have exactly 12 atoms and at most 20 Horn rules.
Every positive goal has a shortest proof using exactly 3–5
legal novel-rule applications. Its goal is produced only by
multi-premise rules. There are ten reachable new conclusions,
including distractors that occur in alternative producers of
the actual goal, so a five-action budget is binding.

| Split | Cases | Shortest valid depth | Distinct families |
| --- | ---: | --- | --- |
| Train | 768 | 3–4 | braided_and, shared_fork_and, cycle_and |
| Development | 192 | 3–4 | cross_and |
| Held-out familiar | 192 | 3–4 | paired_requirements, nested_and_switch, goal_misdirection, multi_support_cycle |
| Held-out deeper | 192 | 5 | Same four unseen families, disjoint examples |

Every split is label-balanced, and all conservative structural
WL fingerprints are distinct from FC-02 through FC-07 and from
the other FC-08 splits. The source checker independently confirms
every entailment with exhaustive Boolean truth tables, verifies
complete countermodels for invalid cases, and checks positive
minimal proofs using exact bounded BFS. The resulting theorem
terms compile in Lean 4.19.0.

| Source | Frozen commit | JSONL SHA-256 |
| --- | --- | --- |
| Train/dev only | `51d0082f27e4c6214ec0cebb4886df44a4962321` | `0d51135f606d44085f2bf0643e44469c36cbebae2469c5ceb8dcbb52b75f763a` |
| Held-out test | `1408daf07c34a133a794662c31f8fbdf789ac840` | `3b7d93c04e6565fc8e46e30947c0347d75ad6b61d575aeb7a598e6c189969dd1` |

[Train source CI #43442906680](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43442906680)
passed six new tests and compiled 480 valid training/development
reference proofs. [Test source CI #43444039591](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43444039591)
passed seven tests, compiled all 192 positive held-out references
and checked 192 negative countermodels. Endpoint axiom audits
reported no extra axioms.

## Symbolic controls registered before model scores

The [source implementation](controls.py) uses only atoms,
facts, original source rules and goal. The fixed best-first
arm pops at most 24 graph-known-state nodes and produces
at most five original-rule actions. It prioritizes states
by path length plus relaxed goal distance, treating an
individual conjunctive prerequisite as sufficient for the
distance lower bound. Its exact search and step cost are
recorded separately from the neural model.

The first complete source receipt recorded these positive
proofs, out of 96 valid cases in each held-out split:

| Policy | Familiar depth | Greater depth |
| --- | ---: | ---: |
| Bounded best-first (24 states) | **89** | **28** |
| Goal-first | 68 | 18 |
| Uniform random legal | 53 | 5 |
| Backward goal ancestry | 48 | 9 |
| Unary-only BFS | 0 | 0 |
| Exact shortest-proof oracle | 96 | 96 |

The test set was kept without model-informed filtering or
altering its initial baseline scores.

## First frozen LR-01 outcome

[First model CI #43444713589](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43444713589)
passed both training-only and held-out jobs. The student
generated **352 positive Lean-compiled proof terms**
and **380 exact negative Boolean certificates**
across two seeds and two held-out depth groups,
with zero false formal certificates.

| Seed | Familiar positive /96 | Deeper positive /96 | Best-first familiar/deeper |
| --- | ---: | ---: | --- |
| 17 | **94** | **72** | 89 / 28 |
| 29 | **96** | **90** | 89 / 28 |

The preregistered **learning criterion passed** against
24-popped-state symbolic best-first across both seeds
and depths. The independent **CPU-cost criterion failed**
in three of four comparisons; the learned actor
requires more neural passes per action.
The [full first-run receipt](../results/fc08_conjunctive_first_run_20261113.md)
includes original source hashes, paired outcome
counts, verifier costs, abstentions, and archive checks.

## Neural trial and dual exact certificates

The stacked [LoopedTransformers FC-08 PR](https://github.com/MoralityLabAI/LoopedTransformers/pull/29)
uses the original tied LR-01 recurrent cell. An independently
trained 11,521-parameter verdict/eligibility gate is
calibrated on **FC-02 development only**. A separate
16,226-parameter actor jointly fine-tunes the recurrent
cell and a goal-conditioned rule-ranking head using
**FC-08 training-only shortest proof-prefix states**.

A GitHub Actions training job checks out the pinned
train-only commit, explicitly asserts the absence
of the test generator and its symbolic controls,
and freezes SHA-256-identified actor weights.
A dependent test job checks out the held-out source
after training finishes and scores the learned actor
against the original source's best-first code.

Positive output requires the **model's exact proposed
rule sequence** to pass the strict independent
five-step trace checker and compile in the Lean
kernel; otherwise the system abstains. Negative
output requires a complete, independently verified
Boolean countermodel. All raw failed traces and
both checker/model CPU times are saved.

This tests learned selection on already formalized,
finite Horn arguments. It does not validate
a manuscript-to-formula translation or resolve
a famous open problem.
