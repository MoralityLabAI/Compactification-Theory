# FC-05: first complete model-proposed positive and negative certificates

**13 November 2026.** First full held-out training/evaluation CI was
successful. Its registered **operational** criterion was met: all
seed/split combinations produced at least one independently verified
Lean-compiled positive proof, and no certificate on either polarity
was false. The registered **learned-policy advantage** criterion was
**not** met. A post-hoc combinatorial argument shows why the fixed
ten-rule budget makes the task insensitive to legal rule ordering.

## Immutable provenance

- [Preregistered FC-05 protocol](../fc05/PREREGISTRATION.md) committed
  `e6ab5716e3316441adad131b087a50d1ca0f2734` before source
  implementation and model evaluation.
- Old FC-02 training source at
  `7aca2bc3fc5586c5b7997fba868bd179db761121`; dataset
  SHA-256 `8e86a41dd93ff162072799332ebbef4c7134951798eb4af7ebb4ab044bebd56f`.
- New FC-05 source pinned to
  `e2fd9f6a0bc7466475541a7ee2afd1320b4e0366`;
  test JSONL SHA-256
  `faced2fce3ecc176a86871211ec013c1a8deb1a83f3b5e89f9cb650c979dd722`.
- [Source CI #43399156073](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43399156073)
  passed eight source tests, checked all 384 finite cases, and
  compiled all 192 valid Lean reference terms. Original source CI
  #43398968509 failed on a test-only overlong reference trace; the
  test was corrected to truncate at the first goal derivation.
- [First full model CI #43399759225](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43399759225)
  passed seven train-only unit tests, trained and SHA-froze weights,
  then accessed the held-out source in its dependent evaluation job.
  It independently checked the model-selected action sequences
  and compiled **318 model-proposed Lean theorem terms** with no
  errors. Both sampled endpoint axiom inspections say the
  generated proofs depend on no extra axioms.
- Complete evaluation artifact `fc05-dual-certificates-43399759225`
  retains `FC05_report.json`, twelve per-case prediction JSONL
  files, `FC05_model_proposed.lean`, and the kernel compilation log
  (30-day retention on GitHub). The 318 theorem terms count accepted
  **seed/case evaluations** and may include multiple terms proving
  the same underlying proposition across different seeds.

The test corpus contains 192 valid and 192 invalid cases across four
new graph families. Each depth-split has 96 valid/96 invalid. None
of the FC-05 source grammar was present at training or stopping
threshold calibration.

## Model and algorithm

Two newly trained **11,521-parameter** four-pass LR-01 recurrent
models received 350 AdamW steps each on 1,152 older FC-02 train cases.
After training, the FC-04 development-only threshold grid yielded
θ=**0.70** (seed 17) and θ=**0.55** (seed 29).

Each model then froze its trunk and trained a **4,705-parameter**
rule-priority head on 1,728 exact training-only positive
proof-prefix states, using 250 AdamW steps. Total neural parameters:
**16,226**. The proof head ranks currently legal rules at each step
using four tied-cell passes, and can select at most ten original
rule IDs. It never receives reference source proof traces or gold
labels at evaluation.

For a stable positive candidate, the independent FC-05 checker
replays the *same model-proposed action list*; failed traces become
abstentions. All accepted traces are rendered into explicit
proposition/implication Lean theorem terms and compiled in CI.
Negative claims still require exact complete Boolean countermodels
through the FC-04 negative gate.

## First frozen held-out results

Counts are out of 96 **valid** positives and 96 **invalid** negatives
in every split. These four lines are for the learned proof head.

| Seed | New family depth | Positive Lean certificates | Negative exact countermodels | Abstentions | Mean rule applications | Shared-cell invocations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 2–4 | 96 | 96 | 0 | 2.34 | 2,790 |
| 17 | 6–8 | 87 | 90 | 15 | 3.89 | 4,610 |
| 29 | 2–4 | 96 | 96 | 0 | 2.35 | 2,790 |
| 29 | 6–8 | 39 | 87 | 66 | 1.71 | 3,128 |

The 318 positive certificates came from the sum of the four
positive-success counts (96+87+96+39). Negative certificates were
**369** across the four seed/split conditions. Total accepted
certificate results were **687 of 768 seed-case decisions**. There
were **zero false formal certificates** because every one passed
its corresponding exact checker and positive terms were compiled
by Lean. The symbolic Horn solver remains 100% accurate on this
finite fragment.

### Frozen heuristic controls

The uniformly random legal-action baseline and the
immediate-goal-first baseline, both with the same ten-step
proof budget and exactly the same FC-04 eligibility gate, had
**identical positive and negative certificate success counts**
in all four comparisons. All eight paired positive discordance
counts are **0/0**, as recorded in `FC05_report.json`.

| Condition | Learned positive /96 | Random legal /96 | Immediate goal /96 |
| --- | ---: | ---: | ---: |
| Seed17, same depth | 96 | 96 | 96 |
| Seed17, deep | 87 | 87 | 87 |
| Seed29, same depth | 96 | 96 | 96 |
| Seed29, deep | 39 | 39 | 39 |

The learned policy used considerably **more** computation:
on seed17 same-depth it needed **2,790** recurrent cell calls,
versus **990** for either heuristic policy with no neural
per-action scoring. On seed17 deep it used **4,610** versus
**1,626** calls for the heuristics. Model-forward milliseconds
per case were 13.14 versus 3.48 (same depth) and 23.74
versus 5.68 (deep), measured on one CI CPU runner.

The model's per-proof trace is short (mean 14–28 UTF-8 bytes
depending on split/seed), and the maximum of ten rule actions
applies to all methods. The learned head's shorter average
trace lengths in a few comparisons are too small to justify
a generalization claim, especially given its added inference
expense and equivalent certificate recall.

## Why this control saturates: a post-hoc theorem

The new tasks have `d+4` atoms with `d <= 8`, one
initial fact, and an unreachable guard atom. Any applicable
rule whose head is not yet known increases the known set by
one reachable atom. No such strategy can make more than

`|Cl(F) \ F| <= (d+4) - 1 - 1 = d+2 <= 10`

novel updates before all reachable consequents are exhausted.
If the source claim is valid, the goal is therefore guaranteed
to appear under **every legal novelty rule ordering** within
the experiment's ten-action budget. [The post-hoc analysis](../fc05/POSTHOC_SATURATION.md)
proves the bound and records an adversarial ordering regression
test. This diagnosis followed the test run; the first test and
its original predictions remain immutable.

The equality to heuristic baselines is a structural property
of this dataset and budget. **FC-05 certifies a working dual
proof channel, with no evidence for superior learned proof
search**. It cannot serve as a positive-result benchmark for
the trained rule head in its current form.

## Necessary next stage

FC-06 should register a new test *before* inference, with
reachable distractors and a strict action budget smaller
than the worst-case number of derivable atoms. Every valid
example must still admit a verified short proof and an exact
solver upper bound; random and goal-first policies must have
nontrivial measured failure. Comparisons must charge
the learned head's extra recurrent passes and keep all
failed traces in the receipt.

No result in FC-05 concerns truth or falsity of RH, Collatz,
Fermat or P versus NP. Formalizing an untrusted human
manuscript is a separate, uncompleted semantic task.
