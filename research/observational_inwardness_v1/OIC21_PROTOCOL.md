# OIC-21: Active sensor design and learned latent-generator recovery

**Prospective / unexecuted.** OIC-20 establishes the exact identification limits of multiple *labeled* phase sensors. The next learned-model task evaluates whether a recurrent observer can discover which sensor to query under a resource budget, with no privileged initial phase at inference.

## Comparison

Train on n=4–6 with five seeds and initial-phase holdouts; test unseen n=7,8,12,16,24,32. Compare a cost-regularized adaptive controller that chooses one sensor position per step, a fixed optimally placed marker set from OIC-20, a random fixed marker set of the same size and total transmitted bits, and the original one-marker OIC-19 predictor. Do not grant the learned model a labeled multi-sensor observation without counting its extra bits and readout cost.

The controlled inference target is the full modular shift d, but report correct observable gcd class separately. The unrestricted target is identifiable only under sufficiently rich marker sets. For OOD n, separate prediction of a known n from inference of n. Do not choose sensor sets after observing held-out outcomes.

## Diagnostics

Report per-n minimal sensor count, total active queries, total observation bits, exact class distribution, phase-posterior entropy, shifted gauge equivalence, decoder calibration, single raw-substitution/erasure robustness and equal-budget CPU wall time. Use original OIC-T26 gcd classes as the single-sensor ceiling. Evaluate observation history likelihood under alternate d values rather than a hidden-phase classification that is impossible from the interface.

## Acceptance

Require all registered n and initial phases to be included; zero passive family leakage; at least 90% shift identification accuracy under the minimum-marker sensor family and high-probability recovery on noisy histories at a stated bit budget. Any architecture containing an exact cyclic hypothesis enumeration or a precomputed optimal marker design must be labeled as structure-assisted. Publish a finite exact query-policy oracle and archive all learned-model weights, receipts and evaluation seeds.
