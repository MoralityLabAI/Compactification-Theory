# SMC-15: Fresh-cohort active SAE contribution erasure

**Pre-registration:** November 16, 2026. This protocol is committed as the first SMC-15 change, before any fresh-cohort intervention. This branch is stacked on the unmerged [SMC-14 PR #174](https://github.com/MoralityLabAI/Compactification-Theory/pull/174); SMC-14 results are prior observations, never an independent SMC-15 replication.

## Frozen sources and question

Reuse exactly SMC-14's pretrained source A, B501/B701 independent recurrent teachers, original pretrained SAE71, both new SAEs (71/89), three SHA-pinned artifact receipts, and original **SMC-13C** 64-puzzle position+visible-clue-conditioned Hungarian permutations. Verify every pretrained model/checkpoint/SAE receipt and all *seven* original conditioned mapping hashes using SMC-14 helpers **before constructing new evaluation hidden states**. Never train or refit a model, SAE, mapping, or intervention direction.

SMC-14 showed signed effects for *injected decoder directions* across independent teachers. Only 8.6–16% of its matched target features belonged to a receiving teacher's naturally active top-four features. SMC-15 measures **erasure of a naturally encoded contribution**. Physical ablation uses the exact source FP32 top-four SAE coefficient and decoder column, including the original training-only hidden-state scaling. This is an operational counterfactual, not an identified proof of unique internal causal necessity.

Primary pairs, retained in fixed order: A-SAE71 to B501-SAE71, B501-SAE89, B701-SAE71 and B701-SAE89. No seed selection after outcomes.

## Newly committed, disjoint test rosters

Construct two independent 64-puzzle families using `smc13_family.new_full_solution` and `dig_unique`, with the source-fixed rules for unique solution and exact hole count:

- `smc15_novel_30`: seed base **202615030000**, 64 successive seeds, 30 holes.
- `smc15_novel_34`: seed base **202615040000**, 64 successive seeds, 34 holes.

Use `seed+1000000` for digging. Require each solution's alternating-rectangle count >0, exactly one Sudoku completion, and disjoint complete solutions and visible puzzles from **every** SMC-13 train, validation, calibration and test cohort and between the two SMC-15 groups. Publish per-cohort SHA-256 manifests of `fingerprint(rows)`. Generate **four-puzzle prefixes for smoke tests** without scoring SMC-15's full gate.

## Deterministic intervention

Choose eight blank goal tokens per puzzle with the original SMC-14 visible-mask rule. From the *clean FP32* stage-one state, encode both source and receiving model using their unmodified top-four SAEs. Source feature `i` is its strictly positive, uniquely highest SAE feature. Target `j=pi(i)` uses the frozen conditioned Hungarian assignment.

The candidate is eligible only when source `i` is strict positive, target `j` is active with `z_j>0`, and at least one **different** receiving SAE atom `k` is also active. Select `k` by a deterministic seed indexed by pair, cohort, puzzle and goal, using only target SAE activations and the selected feature ID. No gold digit, causal outcome, or clean prediction may select the comparison feature.

In the target model remove the **full native physical decoded contribution**
`delta_j = z_j * (scale * decoder[:,j])`
from that one stage-one goal token. This is a counterfactual erasure on the original full hidden state, without replacing it with an SAE-only reconstruction. For the active-other-atom negative control, subtract an equally strong physical L2 dose along the *unit physical decoder direction* of `k`. The source model likewise erases its native top feature. Hold all other 80 stage-one tokens unchanged at injection, then use exactly three FP64 recurrence steps, without renormalizing or retraining. Report post-erasure re-encoded `j` activation ratios separately: the subtraction is an operational decoded-contribution erasure, which may re-encode an active `j` because SAE dictionaries are nonorthogonal.

For each endpoint, measure the change in **correct digit minus highest clean competing digit** margin, using the clean rival fixed for that teacher. Record the selected token's clean correctness, flip from clean correct to wrong after ablation, its feature coefficient, physical ablation dose, and post-erasure re-encoded coefficient. Keep ineligible examples in raw receipts, with explicit flags and no imputed zero causal necessity.

## Calibration sign, scoring and null

Determine one global sign for the matched erasure arm and another for the norm-matched other-active-feature arm **only on the original 64-puzzle SMC-13 calibration cohort**, by the sign of the sum of source-target erasure-effect products over eligible goals. Freeze both signs before evaluating the fresh cohorts. Report an unidentified sign and abstain when there are no eligible calibration cases or zero total product.

Define `sgn_e(x)=0` for `abs(x)<=1e-8`, otherwise `+1/-1`. Each eligible goal's signed correspondence is `sgn_e(source_margin_drop) * sgn_e(target_margin_drop) * calibration_sign`. The primary paired per-goal difference is matched correspondence minus active-other-feature correspondence, evaluated on **the same matched-active eligibility mask**. Every fresh family reports eligible goals, contributing puzzle clusters, average signed correspondence for both arms, and paired difference.

Resample **64 independent puzzle IDs** with replacement 2,000 times, taking the pooled eligible-goal difference within each sampled cluster multiset. Mark draws without eligible goals as invalid; a positive gate must have 2,000 valid draws. Also report the unweighted average of per-puzzle eligible rates as a sensitivity check. Full per-goal data and paired controls remain in the artifact.

**Predeclared pair-level success criterion:** on *each* fresh 30-hole and fresh 34-hole family, at least **24** eligible goals across **20** distinct puzzles, an identified calibration sign for both arms, and a puzzle-cluster bootstrap 95% lower endpoint for the matched-minus-other-active signed concordance **strictly greater than zero**. Retain failures and per-pair results. This is a descriptive finite-cohort criterion without population inference or correction for four comparisons.

Secondary outcomes: clean-correct goal accuracy, matched and other-active clean-correct-to-incorrect flip rates, uncalibrated correct-digit margin drops, remaining matched-feature re-encoding, and coverage across all selected goals. Record the original SMC-14 heldout effects as prior context only; SMC-15 uses fresh data.

## Interpreting the result

A positive result would show that a frozen cross-model matched atom, **when naturally encoded by the receiving teacher**, contributes to the correct-digit margin more coherently with the source than a physical-dose-matched alternative active atom under the declared calibration and controls. Even that is an intervention-specific feature-contribution observation rather than a unique mechanism or a claim of the same Sudoku skill. A null or low-coverage result constrains the SMC-14 interpretation.

Proof companion: OLM-15 in MathCan will state a finite erasure Taylor bound, the exact local SAE re-encoding condition, and a counterexample to active-feature necessity from activation alone. Independent receipt audit, distinct test roster hashes, full results and explicit failure states are required before claiming an empirical success.
