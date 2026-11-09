# RRC-07: raw-only iterative refinement and separately charged exact feedback

**Frozen before the first full evaluation; 2026-11-09.** The RRC-06 test roster and its scores are already known. RRC-07 therefore uses new seeded cases for an independent *instance* split; its periodic/gap grammar families are known from earlier stages and are not claimed as newly blind families. A single training seed, no hyperparameter search or test-tuned checkpoint selection.

## Scientific question

Can a small learned recurrent pass move the first **exact** Euler-exponent error later and reduce the number of expensive exact corrections needed to reach a certified program? Compare to a similar-parameter feedforward student and a shuffled-supervision control. A separate residual-feedback arm measures the maximum repair advantage achievable by asking an exact formal-series verifier to reveal the first fault. It is not a raw-only model.

## Information and split contracts

The only neural features are the public 65 raw coefficient counts, transformed to `log1p`, local log differences and degree, and the three-class logits of the frozen, unchanged RRC-06 recurrent checkpoint (`rrc06_recurrent.json`). The recurrent student sees previous **own class probabilities** and its own hidden state on later passes. It cannot invoke Euler inversion, series residuals, source-program metadata, or the verifier during inference. Training uses exact Euler exponents only as supervised targets. The baseline checkpoint already received RRC-06 training labels.

The RRC-07 training function calls `new_roster(include_test=False)` and constructs **600** unique in-family product prefixes from caps unlimited/1 and periods 2,3,4,6,8,9,10,12, seed `2026110917`. Development is **96** newly sampled unique prefixes with the same grammar. Final evaluation creates 72 new product cases with periods 5,7,11,13,17,19; 30 cap-2 cases; 30 gap-partition cases; 12 irregular-part-set cases; and five base/late-mutant pairs at degrees 67,79,91,107,123. The latter share the complete observed prefix but diverge later. As the family labels were explored before RRC-07, claims are restricted to newly sampled instances. Record prefix aliases with both RRC-07's training data and the previously released RRC-06 roster.

## Frozen models and cost

- Tied-weight GRU correction: input 9, hidden 32, class head 3, **4,227** FP32 parameters, 32 FP32 live hidden coordinates. Frozen RRC-06 base logits are external to this count and counted in full for deployment.
- Flattened feedforward correction: features 65×6, hidden 7, class head 195, **4,297** FP32 parameters (1.66% more). It receives exactly the same raw features and initial logits.
- Shuffled-label GRU control: 4,227 parameters, targets permuted across training cases.
- Training: 240 Adam updates, batch 64, learning rate .003, CPU one thread, fixed seeds, second-pass loss and 0.15 first-pass deep supervision. RRC-06 is frozen. Train the recurrent model at two passes. Ablate the same weights at **0/1/2/4/8** passes without refitting.
- Exact residual queries, symbolic grammar fitting, full-series coefficients, decoder code, verifier CPU, and neural MACs must be accounted separately. A positive raw-only result requires a later first-error location and better complete-program recovery than the feedforward control on the same cases. A pass that reduces coordinate accuracy or is less efficient is a failure, to be retained.

## Exact feedback and falsification

Given a *proposed* integer Euler vector, the verifier returns the exact full degree-indexed residual vector `R_n = n f_n - Σ_{k=1}^n f_{n-k} Σ_{d|k} d a_d`. It identifies the first nonzero residual `m`; earlier coordinates are already correct, so `R_m / m` is the exact correction at `m`. The feedback arm changes only this coordinate, then runs a fresh residual verification. One entire residual evaluation is charged **per query**, including the terminal acceptance check. Full-series Euler inversion does not appear inside this arm, but the residual oracle is sufficiently informative to reconstruct the full truncated exponent vector. Report its improvement as *algorithmic oracle assistance*, never as autonomous learned recovery.

Evaluate correction budgets 0/1/2/4/8/16/32/64 on the new held-out-period set, starting independently from frozen RRC-06, learned recurrent two-pass, and feedforward raw estimates. Compare exact symbolic inversion/synthesis as a stronger one-shot baseline. Report the first error, all 64 coordinates, accepted finite programs, their 65–128 exact predictions, bad future acceptances, and oracle edit/query counts. All proposed programs must pass coefficient residual and independent product replay before acceptance.

## Acceptance rules

1. Unit tests independently verify the first-fault identity at several degrees, idempotence of repaired prefixes, budget limits, and exact-series input validation.
2. The 64-edit upper budget must recover the exact truncated exponent vector for any supported integer prefix, including out-of-class families, although out-of-class **program acceptance** can remain zero.
3. Every accepted program must exactly reproduce all observed coefficients. Any future collision false acceptance is reported. A finite certificate does not prove an infinite identity.
4. Both members of every collision pair must give identical neural predictions and edits; outputs depend only on observed coefficients.
5. Replaying saved checkpoints under NumPy-accelerated CPU inference, with scalar standard-library parity checked to 1e-10, must give the retained deterministic source-and-checkpoint-hashed JSON receipt. An independent PyTorch-vs-portable parity test is mandatory.
6. The earlier RRC-06 regression, RRC-01 and RRC-05 exact tests remain available. CI installs NumPy (CPU-only), runs the new tests and the replay without model training.

The test ledger deliberately has an unambiguous failure criterion: if recurrent direct accuracy or query economy fails to improve against controls, report the failure. The exact symbolic algorithm is already capable of solving these finite-product instances, so an accuracy advantage over it is mathematically unavailable on its declared grammar.
