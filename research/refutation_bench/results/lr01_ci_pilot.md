# LR-01 recurrent student on RefutationBench v0.1 + sourced expansion

**Date:** 2026-11-12  
**Status:** Genuine CPU training/inference in GitHub Actions, **negative transfer result**
for this small pilot. This receipt is descriptive; the source-hashed v0.1
deterministic harness baseline in \`results/baseline_receipt.json\` is unchanged.

## Cross-repository provenance

- Student implementation: private
  \`MoralityLabAI/LoopedTransformers/experiments/refutation_transfer/\`,
  [PR #22](https://github.com/MoralityLabAI/LoopedTransformers/pull/22).
  It imports \`TypedLoop\` directly from the repo's LR-01 Lean Reasoner
  implementation and adds a hashed-text encoder, verdict and pointer heads.
- Bench revision:
  \`27a12f866204855e224c5f6bd00888ab41d6fa5c\` (public source).
- Student revision with control logging:
  \`cd819bb5e21a18131944340d6217085826824a59\`.
- [Verified CI run 43236451992](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43236451992)
  checked the external checkout SHA, passed six new tests, trained the models,
  independently re-scored a dev response file and uploaded 14-day checkpoint
  and report artifacts.
- [Earlier successful CI run 43235983713](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43235983713)
  used the same training algorithm, before adding log-only control printing.

## Experiment specification

Six synthetic training cases; nineteen public development examples, with
disjoint source-family IDs. No test split, checkpoint selection by dev results,
or pretrained LR-01 checkpoint. This uses **36,682 trainable parameters**, a
shared tied graph cell, multi-task verdict/flaw/evidence loss, 120 optimizer
updates, PyTorch 2.5.1 CPU, and two seeds. One-pass and four-pass inference
use the same parameter count. The deeper arm uses more compute per example.

All predictions use public claim and evidence text; the scorer uses gold
answers after inference. Any "supported refutation" score uses the existing
evidence-ID annotation proxy. The student emits a canned human-review notice
rather than independent mathematical proofs.

### Latest complete CI run (commit \`cd819bb\`)

| Predeclared arm | Train accuracy | Dev accuracy | Dev macro-F1 | False refutations among 9 non-refuted cases | Annotation-supported rate among 10 refuted cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant: always refuted | n/a | 10/19 (52.6%) | 0.172 | 9/9 (100%) | 0 |
| LR-01 1 pass, seed 17 | 4/6 | 11/19 (57.9%) | 0.345 | 8/9 (88.9%) | 1/10 |
| LR-01 1 pass, seed 29 | 4/6 | 10/19 (52.6%) | 0.172 | 9/9 (100%) | 1/10 |
| LR-01 4 passes, seed 17 | 5/6 | 11/19 (57.9%) | 0.345 | 8/9 (88.9%) | 0/10 |
| LR-01 4 passes, seed 29 | 4/6 | 10/19 (52.6%) | 0.179 | 8/9 (88.9%) | 2/10 |
| LR-01 4 passes, seed 17, train-label shuffle | 4/6 | 10/19 (52.6%) | 0.172 | 9/9 (100%) | 0/10 |

The nominal 1-pass gain for seed 17 is one correctly classified development
case beyond the constant baseline. Four recurrent passes do **not** improve
development accuracy for either evaluated seed. A shuffled training-label
control nearly matches the recurrent models. The evidence-ID match scores
do not certify semantics, and high false-refutation rates are disqualifying
for a reliable auditing agent.

An earlier run at the same data pin and optimizer settings produced 10/19
for both 4-pass arms, so there is observed one-case variability between jobs.
Treat near-baseline differences cautiously; exact CPU workload replay and
separate held-out families are warranted.

## Interpretation

The successful CI job proves that the small LR-01 recurrent cell can be trained
and connected to the existing public RefutationBench scorer under GitHub Actions
resource limits. It **does not** establish that the student acquired a
generalizable refutation skill. The six-item supervision set is too sparse,
the training families are synthetic, and the audit examples are curated
summaries with public answer keys. A useful next run should train on many
independently generated, *exactly verified* proof-vs-invalid-inference pairs,
add non-refuted and insufficient controls, and reserve newly acquired original
manuscripts as a raw-source blind set.

Checks and artifacts live at the private workflow link above; visibility to
that repository is needed to inspect them.
