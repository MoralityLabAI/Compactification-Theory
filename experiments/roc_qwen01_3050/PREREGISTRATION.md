# ROC-QWEN-01 / RTX 3050, 16 November 2026

This protocol replaces earlier hardware instructions. Only the inspected RTX 3050
Laptop UUID is permitted. Decimal allocation ceiling 3,400,000,000 bytes;
allocator cap 3,200,000,000 bytes. One NF4/FP16 text model, batch one, no model
backward, no dense neural Jacobians, no simultaneous model copies. The smoke
precedes task generation. No optional expansion is registered for this campaign.

## Sources and inferential scope

Checkpoint: `Qwen/Qwen3.5-0.8B-Base`, immutable revision in model_manifest.json.
Mathematics: MathCan PR 99 at `37bbac210c175ccb98e0ff0d4aa8bdd2be0764e6`,
`research/observer_calculus/ROC01_AMENDMENT_2026-11-16C.md` (Git blob
`6b7e0a0a402bdde65c4be22df574237ba6017edb`). Do not form T-I for this residual
experiment. A/B/C, full recurrent closure, T-B bounds, and five mechanism tags
are unavailable. Synthetic projector identities are separate exact arithmetic.
Their companion MathCan branch is `research/roc-qwen01-3050-routing-20261116`,
commit `4c49600431ca2f96d59cf9b971e5e39407125137`.
Six-direction Holm is required for any future complete card; this study does
not have that card or substitute residual measurements for A/B/C.

## Population, splits, and boundary

Generate 128 train, 64 calibration, 96 held-out cases, seeds 2026111601/02/03.
Each split balances copy/flip rules, answer labels, and delays 2..8 as closely as
its size permits. Independent nonce/distractor draws prevent identical prompts.
Prompts encode a binary initial cue, a rule, 2..8 intervening steps and a release.
Answers are actual-tokenizer verified single tokens ` A` and ` B`. No truncation:
every prompt must fit 64 tokens. The initial smoke may use a shorter prompt.
The final input token is the selected token. Capture decoder block outputs after
blocks 6, 12 and 18 (one-based, before final normalization), fp16 NumPy memmap.

## Observer fitting and nulls

CPU-only ranks {1,2,3,4,7,8,16}. PCA, task-only, causal and Haar random observers
share the same training activations. Task-only and causal use identical 100-step
Adam budgets, initialization and ridge readout regularization 0.01. All fitting
uses training labels only; calibration selects no hyperparameters in this pilot.
Task labels are +/-1. Input comparator is ridge regression on rule, cue, their
interaction, delay and a constant. Cohort-specific and leave-one-rule-out task
comparisons are reported separately. One checkpoint is one model realization.

The causal observer is a **local directional-sketch observer**, not an estimated
full gradient. At layer 12, measure 16 registered directions: eight training PCA
directions and eight SHA-keyed orthogonal random directions. Probe the first
eight training and first eight calibration states, with central differences at
epsilon 0.25 and 0.5. Accept a direction only when the estimates differ by at
most max(0.02, 0.25*max(abs(d1),abs(d2))). Failed directions remain public and
are zeroed in the sketch with the accepted fraction recorded; no stability
claim is made for rejected directions. Fit causal U from training-state sketches
only. Calibration sketches are for external directional validation.

Write g_i = sum of accepted directional derivatives times their orthonormal
directions. This is a projection onto a measured 16-dimensional span; sensitivity
outside that span is unknown. The residual adaptation of amendment C is

    mean task MSE + mean [(dark_energy_i(U) / null_median_i(k))^2]

where dark_energy=1-||U^T g_i||^2/||g_i||^2. Null medians come from 128 independent
SHA-keyed ambient-dimension Haar/QR rank-k bases at each state and rank and are
frozen for all observer candidates. Zero gradient/constant null abstains. This
is an explicitly labeled residual-output objective, not amendment C's B penalty.
Every extracted state gets 128 Haar bases per rank for residual-energy comparisons;
measured-sketch states additionally get conditional directional energy nulls.
No output sensitivity is imputed to states without measured derivatives.
Random-observer task comparisons use one train-fitted random observer per rank,
with labels used only for evaluation outside train. Add-one p values count ties;
Holm controls the four observer comparisons for each registered residual metric.
These are descriptive single-checkpoint conditional tests, not model replication.

## Physical intervention and prediction test

First run layer 12, rank 7, radius 0.5. Held-out cases are never used for fitting.
For each observer and recipient, pick the first opposite-answer training donor
with the same rule. A SHA-keyed rank-k dark frame V orthogonal to U defines an
explicit artificial isometry. For donor difference z, the arms are
`U V^T z` (dark_to_observed), `V U^T z` (observed_to_dark), and `U U^T z`
(observed_to_observed), each normalized to L2=0.5 before FP16 rounding.
Zero directions fail explicitly. Matched random coefficients use the same
target frame and radius; observed-to-dark uses the same coefficients as
observed-to-observed. Log actual FP16 displacement and source/target leakage.
These names specify donor routing, not recurrent Jacobian blocks.

The hook is an intermediate residual boundary, not a separable observer
readout. Therefore pure `ablate_readout` is **unsupported** and the four-arm
ROC completeness gate is false. Publish a separate exploratory full centered
observed-residual erasure, with its own norm-matched random control, so it cannot
be mistaken for radius-0.5 donor arms or pure readout ablation.

For each direction also measure central derivatives at 0.25 and 0.5 and use the
same stability gate. The output is correct-minus-other candidate logit margin.
Train a CPU linear margin model on training activations in U; predict margin
change as its linear coefficient applied to physical displacement. Primary
prediction comparisons give every predictor the same actual causal-arm delta.
At layer 12/rank 7 also fit each observer and readout using one rule alone and
evaluate task and causal predictions on the other rule. Compare
held-out prediction absolute errors, task accuracy and actual effects, paired
by state/rule/delay, for causal, task-only, random, PCA, and input comparator.
The input comparator predicts zero effect for a hidden-state intervention
(unchanged input). Report all results, including failures and no advantage.

## Runtime and evidence

All model forwards use inference_mode(), use_cache=False, logits_to_keep=1.
Full prefix is recomputed, with no reuse of mutated caches: results are
conditional residual-state effects, not interventions on the complete hybrid
recurrent state. One forward at a time. GPU references die before each next
probe. Every batch records allocated/reserved/peak bytes, device, elapsed time,
successful passes and failures. Allocator OOM aborts the run with a failure
receipt, never drops a row. Forward receipts are keyed by checkpoint revision,
token IDs, delta, layer, backend, source hashes and settings; matching cached
passes are reused. Commit this preregistration and executable sources before
task extraction. Results identify that source commit and all file hashes.
