# DISTILL-08 / incumbent-relative research deviations

Tested commit: `7edc62c4e818694edfa5a2f80987f562b426b996`. [Residual workflow](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/46036888147) succeeded (4 targeted tests and complete campaign). [Full offline workflow](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/46036888100) passed 57 tests.

The full run used 24 training roots, 8 calibration roots, and 8+8 fresh ordinary/shifted evaluation roots. It collected 393 novel sufficient states reached by the provisional residual policy outside the incumbent/oracle trajectory union. All 393 were retained; final 2,239 states carry 11,092 exact legal-action labels. Positive incumbent-continuation opportunities occur in 540 retained states, and positive optimal-continuation residuals occur in 630. Those counts are separate target definitions, not an assertion of subset containment or a measure of real research success. The 600-coefficient ridge model and 32 learned gate buckets are synthetic reference policies.

Evaluation replayed 117,472 world transitions. Actor-only process CPU totaled 1.515 s over all reported evaluation arms/caps. No descriptive 25 ms envelope overruns (0); this is not a matched CPU-time budget. Complete trajectory work caps and priced utility were used in all comparisons.

| Policy | Ordinary 8,192 | Shifted 8,192 | Ordinary 32,768 | Shifted 32,768 |
|---|---:|---:|---:|---:|
| reuse_first | 24.2872 | 24.9947 | 24.2872 | 24.9947 |
| budget_rule | 24.3535 | 25.0603 | 24.3535 | 25.0603 |
| always | 22.8500 | 23.4967 | 22.1838 | 23.0147 |
| fixed_gate | 24.0430 | 24.6916 | 24.0430 | 24.6916 |
| fixed_shuffled | 23.9818 | 24.6916 | 23.9818 | 24.6916 |
| fixed_forced_base | 23.9818 | 24.6916 | 23.9818 | 24.6916 |
| calibrated | 24.2872 | 24.9947 | 24.2872 | 24.9947 |

## Findings

- The fixed residual gate raises ordinary raw utility from 24.71875 to 24.82470, but extra work makes priced utility 0.24418 lower. In the shifted cohort it exactly reproduces the incumbent raw utility and loses 0.30305 priced units. Its shuffled-label and forced-incumbent controls match each other in raw utility and charged work on these aggregates. All nine calibration configurations at each cap fail the positive-priced-utility rule. The selected configuration is disabled at both caps.
- Always scoring loses every root under both caps. No learned promotion.
- The budget rule is the successful analytic ablation: identical raw expected utility to reuse-first in each cohort, with 271.58571 and 268.95476 fewer mean work units, respectively, giving +0.06631 and +0.06566 priced utility. It wins 8/8 ordinary and 8/8 shifted roots at either cap. It makes one mean deviation per trajectory. This is an analytic reference-policy improvement under the declared work metric, not a learned improvement or native production deployment.
- At the fixed gate, the ordinary learned policy is slightly better than the forced-base overhead control, yet it still fails to pay for its own computation. The next target is an inexpensive action-specific exception detector trained against the budget rule; use a small number of features available before posterior acquisition, and compare against a frozen analytic alternative on fresh roots. A stronger candidate could reserve expensive search for states where it can certify a positive lower bound on marginal value.

The outcome was not tuned after evaluation. Full row receipts and local weights were not uploaded by the hosted workflow; `DISTILL_08_HOSTED_SUMMARY.json` archives the printed aggregate and original report/fit hashes. Rebuild with the registered command for full evidence. No API calls, GPU training, provider response training, native DataGoblin update, or downstream training admission.
