# SHAFT-COMP-22: Factorized behavioral reconstruction and the canonical goal-latch gauge

**Registered 2026-11-11, before training or evaluating the new decoders.**

The previous COMP-21 aux-state GRUs are frozen. This is a prospective new-decoder/new-trajectory test on existing model seeds. The same models were evaluated in COMP-21; accordingly the model-level evidence remains retrospective, and results cannot be presented as a novel-model holdout. All tasks are consented, local, synthetic, and retain their full weights.

## Exact finite system and a gauge issue

For six input tokens, the state of each family is (goal g, latch l, previous token p). Output under query q equals the goal bit g[q] XOR l, and the state after this query updates to latch l XOR T(p,q) and previous token q.

Families and triggers:
- repeat_toggle: T(p,q) = 1[p == q].
- successor_toggle: T(p,q) = 1[q == (p+1) mod 6].

The ideal **two-query final signature** for ordered (a,b) is

    B2(g,l,p)[a,b] = g[b] XOR l XOR T(p,a).

This is the exact behavioral factorization. A trained decoder with canonical goal bits, latch bit, and six-way previous-token head can reconstruct it deterministically, without fitting a 36-output behavioral probe directly.

But the behavioral transformation (g,l,p) ~ (g XOR 63, l XOR 1, p) is a gauge symmetry: the output table depends on g XOR (63 if l else 0). Define effective = g XOR (63 if l else 0). The 768 triples have at most 384 distinct future-behavior classes, parametrized by (effective, p). This differs from recovering the specific *oracle* canonical latch and goal labels.

The exact standard-library reference will enumerate both families, verify 36-bit reconstruction for all 768 states, prove the gauge equivalence under all future words of length up to three exhaustively and by transition congruence for arbitrary length, and establish whether (effective,previous) is a minimal behavioral quotient. Prove sufficiency and a concrete alias/split witness in Lean.

## Frozen model cohort

Use only the four aux_state models that passed COMP-21's native 90% capability gate:
- repeat_toggle: seeds 2202 and 2203.
- successor_toggle: seeds 2212 and 2213.

Load weights from immutable COMP-21 saved .pt checkpoints and verify each base model SHA-256 against original COMP-21 receipts. Record previous accuracy at lengths 12 and 32. Do not retrain the base model. Training-only auxiliary heads are excluded from confirmatory feature computations; report their frozen accuracy as context.

## Four registered decoders

Fit all heads on the same 2,304 independent frozen recurrent states at depths 4,8,12, 12 independent prefix histories per each of 64 goals per depth, using random seeds disjoint from COMP-21 probe training and COMP-22 evaluation. Each head uses Adam LR 0.04, 400 optimization steps, BCE for bits or CE for six-way previous-token classification.

1. direct36: 24->36 linear probe trained on native frozen-model final bits for all ordered two-query continuations.
2. canonical: 24->6 goal bits + 24->1 oracle latch bit + 24->6 previous-token classifier, reconstruct B2 through the fixed trigger.
3. behavioral: 24->6 native immediate query bits representing the effective table + 24->6 previous-token classifier, reconstruct B2 through same trigger. This is invariant to complementing goal and toggling latch.
4. oracle: exact B2 computed using true goal/latch/previous labels, used solely as a reference, never as a learned decoder.

Canonical and behavioral factorizations have fewer trainable parameters than direct36; and their previous-token classifiers have access to privileged automaton state labels. Therefore any apparent neural advantage requires this qualification.

## Fresh evaluation

For each frozen model independently sample 32 natural prefix histories per goal at depths 12 and 40, with evaluation seeds disjoint from probe training and all COMP-21 evaluation seeds. This yields 2,048 states per depth per model and 16,384 total evaluation states.

Compare each decoder's 36-bit signature with:
A. native frozen-model two-query behavioral bits, the **primary target**;
B. exact automaton two-query final bits, a secondary target.

Report micro bit accuracy and per-state exact 36-bit accuracy by seed/family/depth. Registered **primary contrast:** canonical minus direct36 in exact native-signature accuracy, pooled over four models at depth 40. Behavioral factorization versus direct36 is secondary. Report canonical goal 6-bit agreement, latch bit, previous-token class accuracy, effective table bit/exact accuracy, and oracle-vs-native agreement.

The reconstructed output uses Boolean operations on predicted bits only. True labels are excluded from test-time model predictions.

**Uncertainty and decision:** 1000 stratified bootstrap resamples of states within model/depth for the depth-40 canonical-minus-direct36 exact accuracy difference. Support requires a positive 95% lower limit and positive improvements in at least three of four frozen model seeds. Otherwise classify result negative or inconclusive, regardless of exact finite theorem.

## Scope

Archive full per-state receipts, model and trained-probe hashes, and executable source. Rerun independently and report whether bytes are identical. Outcomes concern only four previously studied synthetic GRUs. No agent persistence, distributed-device mechanism, unknown hidden objective, or erased-weight recovery is evaluated.
