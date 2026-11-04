# QO-14 results: A recurrent student learned value, while action supervision collapsed

**Retrospective analysis, November 4, 2026.** The protocol was frozen in [RECURRENT_CERTIFICATE_DISTILLATION.md](RECURRENT_CERTIFICATE_DISTILLATION.md) before the full experiment. Every result below is sourced from the [complete source-hashed CPU receipt](../../results/recurrent_certificate_distillation/receipt.json), which also contains the exact frozen reservoir and trained readout coefficients. These are synthetic resettable Bernoulli systems; no external language model or real neural controller was trained.

## Teacher/holdout distribution

Training examples: **864** from 8 Bernoulli rate pairs and 12 source seeds. Held-out examples: **378** from 6 *disjoint* rate pairs and 7 disjoint seeds, with identical evidence-count schedules. The QO-14 teacher adapter obtained first-action Q labels by replaying hypothetical first observations into the original unchanged QO-13 horizon-one solver, and checked their maximum against QO-13's horizon-two optimum. This preserves QO-13's previously committed source-hash and confidence-evidence contract.

The teacher revealed a major structural issue:

| Evidence property | Training | Held-out |
|---|---:|---:|
| Total states | 864 | 378 |
| Teacher predicted positive certification probability | 90 | 16 |
| Already QO-07 certified | 76 | 14 |
| **Unequal exact first-action Q values** | **3** | **0** |

Almost every state gives a tie between the two first probes. The held-out action-regret metric is therefore identically zero regardless of which first probe is selected. Reporting action accuracy as a high percentage would be misleading; it is recorded as **undefined** on the zero-sized decision-sensitive test subset.

## Learned model and first-action regression

The student is a **24-hidden-unit tanh recurrent reservoir with a trained ridge readout**, processing the two named-cell evidence tokens sequentially. Its input and recurrent matrices are fixed after initialization, so this is reservoir/echo-state-style readout learning rather than end-to-end recurrent weight optimization.

| Held-out metric | Flat linear ridge | Reservoir RNN ridge readout |
|---|---:|---:|
| Bellman value RMSE | 0.2058 | **0.1863** |
| Both first-action Q values RMSE | 0.1917 | **0.1747** |
| RMSE restricted to positive teacher targets | 0.3190 | **0.3005** |
| Teacher-relative first-action regret | 0 (all ties) | 0 (all ties) |
| Float32-equivalent model bytes | 168 | 3,560 |
| Estimated multiply-adds per inference | 40 | 1,680 |

The reservoir learned a more accurate **scalar certification-probability approximation** under the registered split. Its first-action ranking cannot be evaluated meaningfully on this test set, because the exact teacher supplied no non-tied action labels.

## Paired rollout on unseen true-rate pairs

Four frozen held-out rate pairs, new seeds 809, 811, 821, 823 and checkpoint budgets 24/64/96 were used for four policies: existing QO-07 bottleneck, flat linear student, reservoir student, and exact horizon-two teacher. Total: **192 checkpoint trials** and **11,776 addressed Bernoulli observations** (each checkpoint restarts a fixed-seed rollout, with deterministic prefix reproduction).

At the 96-query budget across 16 case-seed groups, correct certificates:

| Policy | Correctly certified | Result |
|---|---:|---|
| Bottleneck | 4/16 | Highest practical certificate count, tied |
| Exact QO-13 horizon-two teacher | 4/16 | Matches bottleneck |
| Linear ridge student | 2/16 | Lower |
| Reservoir ridge student | 2/16 | Lower |

The exact teacher often selected the bottleneck fallback because both first-action Q values tied or were below the predeclared value threshold. The students sometimes selected different cells from their approximate Q values and lost certificates. There were **zero false certificates and zero true-rate confidence-coverage failures** in the complete 192 trials.

For the exact teacher, QO-13's full adaptive Bellman search cost hundreds of exact gap evaluations per rollout at 96 queries. The recurrent model used around 100,000 estimated multiply-adds per 96-query rollout in several cases, with no extra Bellman search during deployment. Those are different units of computational work; measured CPU latency, code/feature-extraction cost and energy are future requirements for an economic comparison.

## Interpretation and next falsifier

The teacher signal is **mostly first-action symmetric** under this particular two-addressed-cell, one-action, horizon-two and fixed-feasibility-threshold experiment. When a two-probe certificate event depends on one observation from *each* independent cell, sampling them in either order yields the same joint likelihood and the same expected success probability. Other patterns, including repeated sampling of one cell, can break that symmetry. The sparse observed training non-ties (3 out of 864) and zero held-out non-ties fit this explanation; the experiment has not proven equality for every possible evidence history.

A mathematical companion can prove commutation of the two independent Bernoulli observation orders under a shared terminal success predicate. The next **action-learning** experiment should use an interface with asymmetric probe costs, heterogeneous actions, or horizon-three histories in which two first actions have distinct exact teacher values. Such a dataset must be registered and evaluated on new rate families. Simply improving Q-value regression loss on mostly tied labels does not establish better experiment selection.

The QO-07 mixture confidence sequence continues to determine every valid certification independently of student prediction. The model's predicted Q values and planning Beta probabilities remain **heuristics** with no calibration theorem. This report makes no claim about biological agency, compact AGI, or transfer to a real pretrained transformer.
