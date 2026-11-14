# FC-07: first preregistered learned proof selection against symbolic ancestry

**First frozen test on November 13, 2026: model CI passed, registered
learning criterion passed, registered CPU-cost criterion failed.**

This report preserves the first successful held-out evaluation and the
unchanged preregistration. All numbers concern exact finite
propositional Horn entailment graphs. An exact symbolic
shortest-proof oracle solves 100% of these tasks.

## Source and frozen train/test separation

- [Preregistration](../fc07/PREREGISTRATION.md) was committed first
  as `d3216371ae480b7e1b42cb913a33847ed5ad47c2`,
  before either generator or any model trial.
- TRAIN-ONLY source commit:
  `f470a97502b853addf2b0f67c9f6225001074a90`,
  960-case JSONL SHA-256
  `f725c31a18aa84b2b831b90d8d2f6ca62c2d20d40530b8f8284429d8c80e1359`.
  Training checks out this revision and explicitly asserts that
  the test generator does not exist.
- TEST source commit:
  `6cee4fdba91ba9c13ad47c492a127c7dc1884090`,
  canonical 384-case JSONL SHA-256
  `f7481ed2f489c43eb95fe74b1544aad3ef59898b29d10fd1455874ad990d4cdb`.
  Evaluation downloads frozen SHA-256 model checkpoints before
  checking out this exact test source.
- [Train source CI #43421898810](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43421898810)
  passed 6 tests and compiled 480 positive Lean 4.19.0 theorems.
- [Test source CI #43422531533](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43422531533)
  passed 7 tests and compiled 192 positive reference Lean theorems.
  All 192 invalid sources have complete exact Boolean countermodels.
- [First frozen neural CI #43422916239](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43422916239)
  passed the train job and independent held-out evaluation job,
  compiled **352 model-selected positive Lean theorem terms**
  from the actual accepted action traces, and confirmed
  zero false finite formal certificates.
- The full first-run evidence is in the GitHub Actions artifact
  `fc07-ancestry-43422916239`: `FC07_report.json`,
  16 per-case prediction JSONL files, all accepted Lean
  proof terms and the Lean kernel receipt. 30-day retention.

The new graph family uses alternative producers for real
goal ancestors. Every decoy becomes syntactically
goal-relevant in the backward dependency closure, though
the direct backbone still admits a five-step shortest
proof. Train graph families are separate from test families
and all previous FC-02 through FC-06 records by conservative
WL fingerprint grouping.

## Frozen model recipe

Two seeds (17 and 29) train the original 11,521-parameter
LR-01 four-pass gate on frozen FC-02 train, then calibrate θ
on FC-02 dev only. Both thresholds were **0.55**.

An independently copied LR-01 trunk plus a 4,705-parameter
rule head was jointly trained on 1,149 FC-07 training-only
positive proof-prefix states for 300 fixed AdamW steps.
The proof actor has **16,226 parameters**; together with
the separate frozen gate this is 27,747 model parameters.
Both seeds scored 100% on their recorded 128-state
in-sample training probe, an intentionally weak diagnostic.

Test inputs to the actor are limited to graph atom marks,
rule incidence, current known set and goal. Proof oracle
traces, source family, minimum length and gold verdicts
are absent from model inputs. The actor chooses up to
five legal original rule IDs, and the independent checker
replays exactly those IDs without a proof-repair oracle.
Lean recompiles the resulting accepted theorem terms.
Every negative claim requires a verified complete
Boolean assignment falsifying the goal.

## First held-out paired results

Each test split contains 96 valid and 96 invalid cases.
The four policy outputs share the exact same frozen FC-04
eligibility/negative-checking run for fair paired comparison.

| Seed / split | Positives eligible /96 | Learned positives /96 | Random /96 | Goal-first /96 | Preregistered backward-slice /96 | Negatives certified /96 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 / same depth | 96 | **94** | 41 | 65 | 47 | 96 |
| 17 / deep | 96 | **85** | 16 | 29 | 22 | 94 |
| 29 / same depth | 96 | **93** | 41 | 65 | 47 | 95 |
| 29 / deep | 85 | **80** | 15 | 28 | 21 | 94 |

Across the 768 seed-case predictions, the learned actor produced
**352 kernel-compiled positive theorem terms** and **379
independently verified negative countermodels**. The total
is **731 accepted certificates**, with **37 abstentions**
and zero false finite formal certificates. The 352
positive terms include proofs of the same underlying
cases under different trained seeds.

The registered first learning criterion required greater
positive certified recall than backward-slice on **both**
test depths for **both** seeds, with zero false
formal certificates. **It passed.**

### Paired positive proof outcomes

| Seed / split | Learned-only vs backward-slice | Backward-slice-only | Net learned advantage |
| --- | ---: | ---: | ---: |
| 17 / same depth | 47 | 0 | +47 |
| 17 / deep | 65 | 2 | +63 |
| 29 / same depth | 46 | 0 | +46 |
| 29 / deep | 60 | 1 | +59 |

The archived receipt also includes paired results against
random-legal and goal-first. The model performs better
than both on each split and seed.

### Held-out source-family breakdown

For seed17, positive certificates out of **24 valid**
cases in each of the four new test families:

| Family | Learned same | Backward same | Learned deep | Backward deep |
| --- | ---: | ---: | ---: | ---: |
| reciprocal_forks | 24 | 12 | 22 | 4 |
| delayed_subgoal | 24 | 14 | 24 | 7 |
| false_shortcuts | 22 | 7 | 21 | 3 |
| nested_switches | 24 | 14 | 18 | 8 |

For seed29, the corresponding learned counts were
24,24,21,24 on same depth, and 21,21,19,19 on
deep. The primary registered comparison holds
across different generated structural families,
although the number of trained seeds is small.

## Full CPU and recurrent cost comparison

The rule head spends four recurrent cell passes per
new action. The backward-slice baseline uses a
zero-neural graph traversal. All methods still pay
the *same frozen FC-04 recurrent gate* and
exact-negative-checker cost.

| Seed / split | Learned cells /192 | Backward-slice cells /192 | Learned model+checker CPU ms / case | Backward-slice CPU ms / case |
| --- | ---: | ---: | ---: | ---: |
| 17 / same | 2,198 | 1,026 | 8.173 | 3.731 |
| 17 / deep | 2,886 | 1,126 | 10.824 | 4.142 |
| 29 / same | 2,214 | 1,042 | 8.311 | 3.819 |
| 29 / deep | 2,712 | 1,204 | 10.267 | 4.470 |

Dividing total measured CPU time by verified *positive*
certificates gives:

| Seed / split | Learned ms per positive certificate | Backward-slice ms per positive certificate |
| --- | ---: | ---: |
| 17 / same | 16.69 | **15.24** |
| 17 / deep | **24.45** | 36.15 |
| 29 / same | 17.16 | **15.60** |
| 29 / deep | **24.64** | 40.87 |

The preregistered **compute-efficiency criterion failed**
because the learned method spends more CPU time per
certified positive than backward-slice at same depth
for both seeds. It is more efficient per certificate
on the deeper tasks, because backward-slice has
much lower certificate recall there.

All CPU figures are one hosted GitHub Actions run,
not FLOP-normalized measurements or an estimate
of cross-device variance. The archival JSON
contains the recorded CPU cost and raw traces.

## Separate post-hoc unary-path control

After the first frozen model test, a
[post-hoc structural audit](../fc07/POSTHOC_UNARY.md)
identified another simple symbolic
solution. The canonical goal chain is
made of **single-premise implications**;
the alternative goal producers that
confused full backward ancestry all
use conjunctions. Breadth-first search
in the unary-rule subgraph finds the
real goal path in at most five steps.

[Independent post-hoc CI #43424550518](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43424550518)
passed four regression tests and
compiled **192/192** such positive
proofs in Lean 4.19.0, with zero
parameters, zero neural cell
invocations and no false certificates.

This stronger non-neural method
was discovered *after* the originally
registered test, so it does not alter
FC-07's frozen learned-vs-backward-
slice criterion. It materially limits
any claim of general neural proof-
search superiority: the source
grammar retains an exact and cheaper
unary-graph shortcut.

## Meaning and limitations

This is the first RefutationBench FC experiment
to show an LR-01 recurrent proof actor beating a
**preregistered non-neural reverse-graph-ancestry
control** on separate, unseen graph families
with a genuine binding five-action budget.
The traces were independently checked and
compiled under Lean. That supports a concrete
claim about learned rule selection under an
explicit small computational budget.

The model still operates on complete, already
formalized Horn graphs. Exact symbolic BFS
supplies 100%-correct proofs; the experiment
does not show neural superiority over optimal
symbolic proof search. Its positive source
proofs and negative countermodels are
synthetic formal obligations, unrelated to
whether a human manuscript has a valid
semantics-preserving formal encoding.

A follow-up should compare with compute-bounded
best-first search, cost-aware reverse slicing,
and genuine longer proof search on shared
subgoals. Such comparators must be
preregistered before any new test, with
checkpoint, verifier and CPU-cost receipts.
