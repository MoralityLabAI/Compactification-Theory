# H-M2 report audit for M-VOI-01

**Status:** existing CPU research report found in the user's file library, **not independently re-executed from raw H-M2 sources in this repository**. Source report: `H_M2_COMPOSITION_REPORT.md` (2026-11-03), original code reported SHA-256 `b740a0c59d3c09c477092f3c0188511ce51d92a3f8c9ca223aa6aeaf8c3b448a`.

### Evidence reported by the original experiment

The report describes 900 training worlds (12,617 teacher-labeled states), 150 validation worlds, and two 650-world evaluation sets. It reports an action classifier with 15,052 weights and **58.50% validation teacher-action agreement**, and a strong two-production macro comparator.

| Metric | Held-out macro | Held-out learned | Confirmation macro | Confirmation learned |
| --- | ---: | ---: | ---: | ---: |
| Weighted accuracy | 0.8160 | 0.8129 | 0.8161 | 0.8109 |
| Realized utility | 0.7493 | 0.7343 | 0.7511 | 0.7325 |
| Exact unseen composed-program recovery | 0.2031 | 0.2734 | 0.2109 | 0.3099 |

- Reported learned-minus-macro confirmation utility: **-0.0185**, paired 95% CI **[-0.0329,-0.0033]**.
- Learned-minus-macro exact composed-program recovery: **+9.90 percentage points** on the confirmation composed-target subset (exploratory secondary result, unadjusted for multiple comparisons).
- Mean confirmation probe/construction cost: **2.1757 learned** versus **1.8056 macro**.
- Shuffling policy evidence decreased the learned policy's utility by **0.0477**, a larger effect than seen in H-M1. This supports reliance on observations within this implementation, pending intervention controls and raw-record verification.
- Original report claims 38 experiment tests, 5 registry tests, an audit of 9,100 evaluation records, and 168 replay checks. These are **historical report claims**, not independently run by this addendum.

### Methodological qualification

H-M2's held-out families are operator pairs already present as candidate constructions in the public DSL, so success tests novel *composition of existing operators*, not novel operators. Its generator renormalizes target programs within eligible held-out families, while the nominal Bayes planner uses an unconditioned public prior. The exact planner is consequently optimal only for its *assumed* model, not necessarily the shifted evaluation distribution.

The report is consistent with a metacognitive **utility-control defect**: the learned model sometimes finds the structurally correct rule but spends too much searching, and underperforms on familiar primitive tasks. Action-value estimation and cost-aware stop/revision advantages are therefore the first causal hypotheses worth testing.

### Required next independent work

Recover `experiments.h_m2_composition` and matching pre-run lock, `policy.pt`, manifest, evaluation records and audit files. Verify the locked source and rerun against unchanged original worlds before attributing the reported numbers to this repository. In a newly frozen M-VOI-01 experiment, compare the information-matched planner, original teacher, macro policy and learned value-controller under identically priced probes, construction actions and output labels. Keep H-M2 confirmation untouched as historical evaluation.

**GPU and Silico spending: unrun.** This document adds source-aware context to `M_VOI_01_GOODFIRE_PROTOCOL.md`.
