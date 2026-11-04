# QO-12 result: strict delayed information, limited policy improvement

**Post-hoc analysis dated November 4, 2026.** The [complete source-bound receipt](../../results/two_step_gap/receipt.json) and [pre-outcome protocol](TWO_STEP_CERTIFICATION.md) preserve every registered synthetic seed, case, policy and budget. The numerical results in this note are descriptive and were read after the first full paired experiment.

Across five finite synthetic cases, three new seeds (449,457,461), four policies, and cumulative checkpoints (24,64,96), the verified study completed **60 trajectories, 180 checkpoint records and 5,760 Bernoulli queries**. Every policy used the same per-cell random stream prefixes. The report records **zero incorrect certificates** and **zero confidence-sequence coverage failures**.

At the 96-query checkpoint, aggregated over all 15 case/seed pairs:

| Policy | Correctly certified | Mean certification gap | Mean frozen one-step TV error | Mean extra depth-two gap calls |
|---|---:|---:|---:|---:|
| QO-08 bottleneck | 3/15 | 0.3240 | 0.3218 | 0 |
| QO-11 expected-gap + bottleneck | 3/15 | 0.3181 | 0.3210 | 0 |
| QO-12 depth-two on plateau, bottleneck fallback | 3/15 | 0.3227 | 0.3202 | ~1,147 |
| QO-12 depth-two on plateau, width fallback | 3/15 | 0.3246 | 0.3209 | ~1,212 |

The QO-12 lookahead policies also run QO-11's positive one-step acquisition calculation, so their total lookahead cost is larger than the last column. Actual CPU wall time and neural inference cost were not measured. The machine receipt includes exact integer optimizer-call counts per trajectory and signed paired differences for every checkpoint.

The two-cell **mathematical plateau witness** is established: with two unobserved cells, QO-07's ideal unrounded alpha=1/20 confidence update gives zero one-step expected reduction and strictly positive two-step expected reduction **1/320**. The executable outer-rounded version independently confirms a smaller positive two-step value. The first-stage policy may choose the second cell based on the first observed hit/hold.

The **empirical result is negative for certification advantage** at the registered 96-query horizon. The two-step variants tied the bottleneck baseline in certificate count while consuming substantial extra planning work. The reduction in mean actual frozen TV prediction error for two-step+bottleneck was small and came with a larger mean certification gap than QO-11's one-step policy. Two-cell feasible and infeasible plateau cases account for the three certificates under every policy; none of the three retrospective three-state cases yielded a certificate by 96 queries.

This result does not show whether deeper recourse would help under longer sampling budgets, other statistical margins, or learned neural representations. A better planning objective may value **probability of crossing a certification boundary** instead of expected reduction of the scalar gap, and may use horizon truncation or a proven admissible potential. Any further study should predeclare fresh case families and count both Bernoulli queries and planning CPU cost.

The deterministic two-stage monotonicity theorem and ideal plateau arithmetic are formalized separately in [OntologicalProofs MOR-10](https://github.com/patrickdugan/OntologicalProofs/pull/16). The anytime confidence guarantee remains a separate QO-07 assumption on resettable IID Bernoulli streams.
