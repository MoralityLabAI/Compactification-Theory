# FC-02: Lean-certified Horn curriculum and LR-01 depth extrapolation

**Date:** 2026-11-12. **Status:** Genuine CPU neural training and exact
certificate scoring in GitHub Actions. This is an exploratory, fully public
formal-domain transfer experiment; it is not a new mathematical proof and not
a blind natural-language refutation benchmark.

## Immutable inputs and kernel audit

The committed generator in
[formal_curriculum](../formal_curriculum/README.md) constructed 1,536
argument graphs in seven source-disjoint graph families:

- Train: 1,152 problems of implication depths 2–4.
- Development: 192 problems at depths 5–6.
- Structural OOD test: 192 problems at depths 7–8.
- 768 positives and 768 negatives, balanced within each split.

The independent bounded truth-table solver verified entailment and concrete
countermodels for all 1,536. Lean 4.19.0 compiled 768 positive proof terms
with explicit theorem hypotheses, and both endpoint axiom audits report no
extra axioms. This is limited to parameterized propositional Horn logic.

[Source CI run 43240137061](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43240137061)
passed eight added unit tests and uploaded a generated-data receipt.

Generator checkout used by the learner:
7aca2bc3fc5586c5b7997fba868bd179db761121

Frozen generated data SHA-256:
8e86a41dd93ff162072799332ebbef4c7134951798eb4af7ebb4ab044bebd56f

Generated Lean file SHA-256:
fe66347cd1a12f5e8b09d4b06ed1508cb12c0c466377250e9c830d9e271166c1

## Real model and provenance

The student is an **11,521-parameter** compact neural graph model that
directly imports the existing shared-weight LR-01 TypedLoop implementation
from private MoralityLabAI/LoopedTransformers. Weights were initialized
from scratch, optimized on train only with 350 AdamW steps, and evaluated
on structurally disjoint development/test graphs using two seeds (17, 29).

[LoopedTransformers PR #22](https://github.com/MoralityLabAI/LoopedTransformers/pull/22)
contains the model and
[successful CI run 43241554303](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43241554303)
contains the checkpoint hashes, prediction JSONL and the report artifact
(with the private repository's access requirements).

The four-pass arms achieved 100% on the first 192 training examples.
The policy predicts which atoms are reachable, and proposes a Boolean
assignment when it regards the goal as unreachable. The independent
checker accepts or rejects the exact assignment *as supplied*. It never
repairs a failed countermodel. Gold and family labels do not enter the
model input.

## Recurrent extrapolation result

The following numbers are from **the same model weights** for a given
seed, with extra inference passes beyond the four passes used in training.

| Train loops | Inference loops | Seed | Dev accuracy | Structural OOD test accuracy | OOD false-refutation rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 17 | 51.0% | 50.0% | 100.0% |
| 4 | 8 | 17 | 89.6% | 91.1% | 0.0% |
| 4 | 12 | 17 | 79.2% | 85.4% | 0.0% |
| 4 | 4 | 29 | 51.0% | 50.0% | 100.0% |
| 4 | 8 | 29 | 99.5% | 52.1% | 95.8% |
| 4 | 12 | 29 | 95.3% | 87.5% | 19.8% |

A one-pass student achieved 52–54% on the OOD tests across seeds.
Both four-pass models achieved 100% sampled training accuracy, and at
inference depth four classified every OOD case as refuted. Doubling or
tripling shared-cell inference passes restored long-chain reachability
in several conditions, with strong initialization and pass-count effects.
At 12 passes, OOD accuracy rose by 35.4–37.5 percentage points for the
two reported seeds. This indicates transferable shared-operator dynamics
on these bounded generated arguments; additional pass counts do not
guarantee monotone improvement.

## Independently verified proposed countermodels

Both OOD splits are balanced: 96 valid entailments and 96 invalid.
For a purported refutation, an independent exact Boolean checker
evaluates every premise under the model's proposed complete assignment
and requires the conclusion to be false. The model receives no repair
from the checker.

| Four-pass-trained model | OOD correct refutation certificates / 96 invalid |
| --- | ---: |
| Seed 17, infer 4 | 67/96 (69.8%) |
| Seed 17, infer 8 | 64/96 (66.7%) |
| Seed 17, infer 12 | 46/96 (47.9%) |
| Seed 29, infer 4 | 66/96 (68.8%) |
| Seed 29, infer 8 | 79/96 (82.3%) |
| Seed 29, infer 12 | 56/96 (58.3%) |

The checker rejects any false refutation on a **valid propositional
entailment**, by construction. If this exact checker is the release gate,
the system can issue only verified countermodel claims and abstain on
unverified proposals. Certificate recall remains incomplete. Its soundness
guarantee applies to the displayed, bounded propositional formula only;
a source manuscript's semantic encoding is a separate responsibility.

## Research limits

- The synthetic structural OOD test is committed as deterministic
  generator code. It is public, and an experiment was added after
  inspecting the initial four-pass failure. The extra-loop findings
  are exploratory and cannot count as a pristine sealed test.
- Training, dev and OOD test use disjoint graph families and nonrepeating
  WL fingerprints; WL is a conservative grouping approximation.
- Depth and graph-family shift occur together. The experiment alone
  cannot isolate which factor causes each gap.
- The 1-pass and 4-pass models have the same parameters but different
  training and inference compute. The 8/12-pass arms spend additional
  inference compute with fixed weights.
- Exact forward chaining and exhaustive propositional verification have
  100% accuracy in this finite fragment. The neural model has no
  accuracy advantage over this available exact oracle.
- The original source manuscripts remain excluded from training. This
  result establishes no performance on uncurated human mathematical
  papers, on arbitrary Lean declarations, or on semantic proof checking.

Future benchmarks should pre-register unseen proof families, compare
compute-normalized recurrent depth against untied baselines, and certify
the actual semantics of source-derived formal encodings. A separate
proof-search policy can then learn to choose verification actions under
a budget instead of approximating a known finite SAT solver.
