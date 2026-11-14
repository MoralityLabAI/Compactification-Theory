# FC-03: first preregistered tied-versus-untied LR-01 comparison

**Date:** 2026-11-13 (UTC+08). **Status:** First frozen-checkpoint evaluation
completed, two-job GitHub Actions CI **passed**. The preregistration's
criterion for a consistent tied-cell advantage was **not met**.

## Precise provenance

- [FC-03 preregistration](../fc03/PREREGISTRATION.md) was committed first at
  `650bd2cc5f0fd628784f672248c3a3e362e9dcdf), before the new
  corpus, model experiment and any FC-03 test scores were available.
- FC-02 training source: pinned commit
  `7aca2bc3fc5586c5b7997fba868bd179db761121);
  canonical source-corpus SHA-256
  `8e86a41dd93ff162072799332ebbef4c7134951798eb4af7ebb4ab044bebd56f`.
- FC-03 test generator source: pinned
  `c859f04cda1d142d4eb560148414b82d53f80705);
  test JSONL SHA-256
  `ddc04ec8553a24269b16b53e27d28b521bc2f928978bfd5ee8a9e5fc36463fdb`.
- [Source verification CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43352394607)
  passed six tests, independently checked 384 entailment/countermodel
  problems, and kernel-compiled the 192 positive Lean 4.19.0 proofs.
  Two audited endpoint declarations report no extra axioms.
- [First model CI run 43352937705](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43352937705)
  has a successful **train-frozen** job followed by a successful
  **evaluate-frozen** job. Training checked out only the old FC-02 source.
  The evaluation job downloaded hashed training artifacts and subsequently
  checked out the pinned FC-03 generator. The run saved the complete
  per-case predictions and `fc03_test_report.json` as
  artifact `fc03-paired-43352937705` (30-day retention).
- Training: six models; seeds 17 and 29, 350 AdamW steps, batch 48,
  CPU PyTorch 2.5.1, two CPU threads, fixed architecture.
  Four cell invocations for every primary and control arm.

The two test splits each contain 192 publicly generated, precisely audited
Horn entailments: 96 valid, 96 invalid. `test_same` uses unseen graph
topologies at depth 2–4. `test_deep` uses those four held-out topologies
at depth 6–8. These are preregistered public tests, not hidden
human-mathematics manuscripts or a secret blind set.

## Primary equal-operator-depth comparison

Four LR-01 cell invocations with 24-dimensional states in each model.
The tied cell has **11,521 parameters**; the four independently weighted
cells have **45,073**. The untied width-12 control has **11,593 parameters**,
within 0.63% of the tied model, but has lower arithmetic cost.

| Arm | Seed | New family depth 2–4 | New family depth 6–8 | False refutations / 96 deep valid | Verified counterexamples / 96 deep invalid |
| --- | --- | ---: | ---: | ---: | ---: |
| Tied, width 24 | 17 | 189/192 (98.44%) | 108/192 (56.25%) | 84 | 17 |
| Untied, width 24 (compute matched) | 17 | 188/192 (97.92%) | 109/192 (56.77%) | 83 | 10 |
| Untied, width 12 (parameter matched) | 17 | 188/192 (97.92%) | 108/192 (56.25%) | 84 | 17 |
| Tied, width 24 | 29 | 188/192 (97.92%) | 108/192 (56.25%) | 84 | 15 |
| Untied, width 24 (compute matched) | 29 | 189/192 (98.44%) | 108/192 (56.25%) | 84 | 17 |
| Untied, width 12 (parameter matched) | 29 | 189/192 (98.44%) | 108/192 (56.25%) | 84 | 17 |

The registered decision criterion required the tied model to improve
accuracy across **both** held-out test depths and **both** seeds, with
no increased false-refutation rate and no certificate-soundness violation.
It fails the accuracy requirement. On the held-out same-depth families,
all arms reach 97.9–98.4% accuracy; on the deep families all primary and
control arms remain near 56%. Exact Horn forward chaining solves every
case by construction.

### Exact paired prediction discordances

The archived report includes per-example paired McNemar counts.
Counts here use **tied correct / untied incorrect** (`b`) and
**untied correct / tied incorrect** (`c`), per 192 paired cases:

| Held-out set | Seed | Untied width 24 b/c | Untied width 12 b/c |
| --- | --- | --- | --- |
| Same depth | 17 | 1 / 0 | 1 / 0 |
| Same depth | 29 | 0 / 1 | 0 / 1 |
| Greater depth | 17 | 0 / 1 | 0 / 0 |
| Greater depth | 29 | 0 / 0 | 0 / 0 |

There is no consistent paired tied-cell advantage; the differences
are zero or a single case in each 192-case comparison. These counts are
taken directly from `paired_primary_and_sensitivity` in the archived
FC-03 score receipt.

## Registered inference-depth extension with fixed weights

The tied model was **trained at four loops**. The next two conditions
reuse its exact checkpoint, with eight or twelve inference calls.
Results from both seeds and both test splits are reported without
selecting a preferred pass count after inspecting test labels:

| Seed | Inference calls | Same-depth correct /192 | Deep correct /192 | Deep false refutations /96 valid | Deep certified counterexamples /96 invalid |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 4 | 189 | 108 | 84 | 17 |
| 17 | 8 | 189 | 163 | 23 | 83 |
| 17 | 12 | 108 | 102 | 0 | 0 |
| 29 | 4 | 188 | 108 | 84 | 15 |
| 29 | 8 | 189 | 149 | 43 | 90 |
| 29 | 12 | 158 | 175 | 10 | 79 |

Increasing recurrence helps the long-chain evaluation in several conditions
and deteriorates performance in others. Seed 17 at twelve passes
classifies 90 of the 96 deep invalid cases as established and produces
**zero verified deep refutations**. Seed 29 at twelve passes reaches
175/192 deep accuracy with 79 validated countermodels. This dependence
on seed/pass count rules out a monotone "more recurrence always helps"
claim. Selection of pass count using these test outcomes would break
the preregistered held-out evaluation contract.

The measured model-only forward costs on CPU at depth 6–8 for the tied
seed-17 arm are approximately 0.107 ms/case (4 passes), 0.201 ms/case
(8 passes), and 0.299 ms/case (12 passes). These are single-run
wall-clock measurements, not benchmarked FLOP-equivalent timing studies.
Untied width-24 depth-four costs ranged from 0.104 to 0.106 ms/case;
the width-12 parameter control cost approximately 0.091–0.092 ms/case.

## Certificate release policy

For each neural claim of invalidity, the learner proposes a complete
truth assignment. The exact checker accepts it only if all premises
evaluate true and the proposed conclusion evaluates false. False
refutations of valid formulas cannot pass that exact verification
in this bounded fragment. This supports **selective certified
counterexamples with abstention** on unverified proposals; it does
not raise the neural classifier's raw accuracy. The reported
certified-recall figures account for all 96 invalid cases, including
ones on which the neural student predicted validity.

The manuscript task still requires an independently verified semantic
bridge from source text to a formal object. FC-03 evaluates a synthetic
finite Horn surrogate and supplies no resolution of famous open problems.

## Interpretation and next tests

A shared recurrent operator acquired high same-depth transfer to
unseen rule families, and its longer computation sometimes transported
reachability across deeper dependencies. Similar results for matched
untied depth-four controls constrain any claim about tying itself.
Longer-loop instability suggests testing a **validation-trained halting
criterion** under new held-out families and reporting total computation
plus certified coverage. A distinct independent test grammar must be
registered before further confirmatory conclusions.
