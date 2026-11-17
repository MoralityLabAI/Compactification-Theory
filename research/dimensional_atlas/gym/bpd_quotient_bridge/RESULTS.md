# SCM21 to BPD-CQ-01: public-source audit interpretation

**Source:** Compactification-Theory merged PR #207, `research/dimensional_atlas/gym/scm21_conservative_causal/results/summary.json`, Git blob SHA `a765043e0837268d52c0548e26d477ba1e813da9`. The root experiment remains scientifically negative for native-write mediation, with clean accuracy preserved. This follow-up is a read-only count-and-dose audit, not an extra trained experiment.

All figures below pool the *same* six seed/stratum blocks of 128 common-eligible recipient/donor pairs from SCM21. They are repeated task strata within three trained seeds, **not 768 independent model replications**. At zero-based visit 1:

| Frozen causal-model intervention | Correct / 768 | Weighted mean patch norm |
| --- | ---: | ---: |
| Native VPD write | 0 | 0.701675 |
| Native 4D write-span projection | 0 | 1.4048 |
| Difference-PCA 4D | 147 | 7.732769 |
| PCA 4D | 101 | 7.2865 |
| Head-readable full difference | 413 | 10.7871 |
| Full CLS donor state difference | 432 | 11.4545 |
| Random fiber 4D, per-pair norm-matched to native | 0 | 0.701675 |

Difference-PCA dose is **11.02×** the native-write dose *on average*. Its 147 successes and the native's zero imply at least 147 difference-PCA-only successes among identical paired evaluations. This is a **direction-and-dose combined observation**. Per-pair norm-matching of both methods is required before concluding that the difference-PCA directions are intrinsically superior.

The BPD-CQ-01 finite-horizon theorem supplies the appropriate mathematical distinction. Global sparse-basis optimality under expected L0 gives a family of equivalent bases. A current-output readout may identify fewer intervention differences than the stacked present-plus-future readouts. Component-mask invariance can depend on both readout and horizon, and restricted observers can create cancellation-based families that do not define Boolean coordinate blocks.

The result does not identify actual BPD neural optimum bases, infer a trained-model observability Gramian, or prove a causal semantic variable from the SCM21 summary. That next test requires a frozen native parameter site, a calibrated local Jacobian/quotient, equal physical patch doses and actual task-outcome comparison on independent generator clusters. Do not unseal ARG-03-N's 168-case test.
