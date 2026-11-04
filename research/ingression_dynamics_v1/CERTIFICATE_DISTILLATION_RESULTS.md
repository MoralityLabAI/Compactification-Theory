# QO-14: Trained recurrent certificate-student results

**Post-outcome research note, November 4, 2026.** The [source-frozen protocol](CERTIFICATE_DISTILLATION.md) defines the objective and data splits. The [complete machine-readable result](../../results/certificate_student/receipt.json) contains all training weights, validation histories, held-out predictions summarized by metrics, 216 policy checkpoint records and 72 stream hashes. It is regenerated from source on every QO-14 CI run.

## Model fit on disjoint rate families

Both models trained on exact QO-13 horizon-two certificate-reachability action values, using public QO-07 evidence features only.

| Metric | Linear student | Eight-hidden-unit tanh RNN |
|---|---:|---:|
| Trainable weights | 26 | 138 |
| Selected epoch, using validation only | 52 | 45 |
| Best validation weighted mean squared loss | 0.00939 | 0.00148 |
| Held-out supervised mean absolute Q-value error | 0.05177 | 0.03427 |
| Informative held-out action disagreements | 3 / 8 | 3 / 8 |
| Mean exact-teacher action regret over held-out states | 0.001922 | 0.001922 |
| Model multiply–accumulates per inference | 24 | 240 |

The experiment generated **720 training**, **180 validation**, and **540 supervised test** evidence records. Already-certified records were excluded from student training and teacher-choice evaluation, leaving 689 training examples, 180 validation examples and 505 test examples. All the declared rate parameter families and Bernoulli seeds are disjoint.

### The decision-label bottleneck

Among the training examples, **only eight** had unequal exact action values. The validation set had **one** such state, and the supervised held-out test set had **eight**. There were 46 positive-value training records before excluding already-certified states, but only 15 remained after excluding the 31 already-certified records.

Consequently the low weighted validation error and improvement in mean value prediction carry little evidence that the RNN learned a reliable *choice of probe* near a difficult certificate boundary. Both students disagreed with the exact teacher on three of the eight informative test states. The evaluated mean action regret was equal.

The current result is a bounded demonstration of recurrent training with full BPTT, rather than a model trained on a rich distribution of active decision boundaries. The gradients passed numerical central-difference tests in the repository.

## Paired policy outcomes

Policies consumed 80 queries per trajectory across six held-out rate pairs and three fresh seeds, for **18 matched case-seed groups and 72 total trajectories**. The final checkpoint aggregated over these 18 groups:

| Policy | Correct certificates | Unresolved | Mean true frozen TV error | Mean certification gap |
|---|---:|---:|---:|---:|
| Existing QO-08 bottleneck | 6 | 12 | 0.35028 | 0.30807 |
| Exact QO-13 horizon-two teacher | 6 | 12 | 0.35028 | 0.30807 |
| Trained linear student | 5 | 13 | 0.35421 | 0.30569 |
| Trained eight-unit RNN | 5 | 13 | 0.35089 | 0.30802 |

**Zero false certificates and zero anytime-interval coverage failures** occurred in all 216 checkpoint records. Those counts are specific to this synthetic sample and are not frequentist guarantees of absolute absence of errors.

### Where the policies actually deviated

Across all 18 complete 80-query trajectories, the exact horizon-two teacher saw sufficient public evidence to evaluate action values 963 times, but its maximum predicted two-step certification probability exceeded the prespecified 0.05 action threshold only **13 times**. It consequently followed the bottleneck fallback nearly everywhere. It produced an *identical complete query/response transcript* to bottleneck in **13 of 18** case-seed groups.

The trained RNN made a nonfallback model-selected probe **122 times**, and the linear student did so **435 times**. The recurrent student matched bottleneck's entire transcript in nine of 18 groups; linear matched in one. The difference illustrates how small action-value prediction errors close to a near-zero acquisition threshold can change many probe selections without improving the eventual certificate.

In the student policy simulations, total registered model inferences were 967 for the RNN and 975 for the linear baseline, across 18 trajectories each. Their deterministic multiply–accumulate counts therefore total 232,080 (RNN) and 23,400 (linear). The teacher used exact rational Bellman calculations for its 963 action-value evaluations; those entail substantially more work than a single student forward pass. This ledger counts mathematical operations and teacher calls, rather than calibrated wall-clock CPU, memory traffic, or training amortization.

## Decision-theoretic interpretation

A related [MOR-12 Lean theorem](https://github.com/patrickdugan/OntologicalProofs/pull/20) states that a uniform per-action prediction error at most epsilon implies a true horizon-two action-value regret at most twice epsilon. This is a **conditional** theorem. The QO-14 supervised metric is a sample mean; it does not establish a universal uniform epsilon guarantee on unseen evidence states.

The current result therefore supports a specific conclusion: a tiny trainable recurrent controller can approximate the finite exact value function with fewer arithmetic operations per query, while the tested model's probe-selection performance remains unproven. The source-fixed policy test found a one-certificate deficit relative to bottleneck and the exact teacher.

## Registered next mathematical questions

A more informative follow-up is to generate a **decision-boundary-enriched** training distribution with a frozen, independent test generator, then compare the percentage of states whose teacher actions differ and the conditional regret specifically on those states. Another is to replace absolute Q-value regression with a margin-aware action-ranking objective and evaluate threshold calibration. Those would be new, separately registered experiments.

Any such work must preserve held-out rate/seed separation, the QO-07 confidence certificate, predeclared query budgets, and complete accounting for recurrent inference, teacher label generation and model training. These data currently concern two named states and one action, rather than a recurrent model grafted into Qwen3.5 or a substrate-independent biological system.
