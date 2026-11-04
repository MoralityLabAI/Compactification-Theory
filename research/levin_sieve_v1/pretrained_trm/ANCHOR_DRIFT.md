# LSV-16a: portable prediction certificates across differing FP32 activation anchors

Date: 2026-11-04. The original LSV-16 certificate was generated using a binary32 stage-one hidden state produced by a specific CPU runner. Its frozen checkpoint and Sudoku roster do not by themselves imply bitwise identical stage-one activations on all CPU runtimes. A subsequent CI replay found disjoint, extremely narrow Arb enclosures for nominal clean pre-LayerNorm states. That check was correctly conservative: it prevented reusing a point-targeted witness without verifying its relation to the new point.

## Theorem AD-01: reference anchor tolerance

Let `u_ref` be a certified centered input to a fixed final LayerNorm at the first run's declared goal cell. Assume the archived LSV-15 inequalities establish that its selected top digit is preserved whenever the centered perturbation has infinity norm strictly less than `T_top1>0` (and the designated top-two order when the perturbation is below `T_top2>0`).

For a second CPU run, let `u_new` be its exact-real clean centered pre-LayerNorm state at the same cell and `u_pert` any perturbed state obtained from the second run's stage-one hidden state after the allowed recurrent updates. Suppose rigorous interval computation proves

`||u_new-u_ref||_infty <= D` and `||u_pert-u_new||_infty <= E`.

Then **for every perturbation covered by E**, the archived top digit is unchanged whenever `D+E<T_top1`, and the top two ordering is unchanged whenever `D+E<T_top2`.

**Proof.** The triangle inequality gives `||u_pert-u_ref||_infty <= ||u_pert-u_new||_infty + ||u_new-u_ref||_infty <= D+E`. Apply the previously proved strict LSV-15 margin inequality. QED.

## Verifying D

For each centered feature, the archived Arb witness provides a closed interval `[a_j,b_j]` containing the old exact-real `u_ref_j`. The new Arb evaluation provides `[c_j,d_j]` containing `u_new_j`. An exact upper bound on their possible separation is

`D_j = max(|c_j-b_j|,|d_j-a_j|)`.

Take the maximum over the 48 features and round *upward* to a recorded decimal rational. All reported radii in the source-locked run use the sum of the rounded-up transported error and this rounded-up drift. The independent standard-library verifier checks exact additivity, archived source identity, original goal thresholds and success counts. The Arb producer remains trusted for the intermediate range propagation, and CI reruns the same producer.

## Meaning

This is a general reference-transport lemma, built from triangle inequality and validated interval arithmetic. It preserves the meaning of the original fixed reference certificate across variations of the computed hidden state, provided the drift is bounded. Source revision and model hash still identify the fixed learned weights and task roster; the additional quantity D accounts for the realized CPU inference state. The theorem concerns exact-real recurrence from the *measured binary32 stage-one states* and does not itself certify the rounding behavior of hardware recurrence.
