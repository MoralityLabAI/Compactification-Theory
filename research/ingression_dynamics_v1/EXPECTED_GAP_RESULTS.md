# QO-11 paired experiment result

Full source-hashed result: [receipt](../../results/expected_gap_pilot/receipt.json). Hypotheses, proof and fixed pilot contract: [EXPECTED_GAP_THEORY.md](EXPECTED_GAP_THEORY.md).

**Completed November 4, 2026:** 100 synthetic paired trajectories, 300 checkpoints and 19,200 Bernoulli queries. At 192 queries, correctly certified case-seed pairs out of 20: bottleneck **9**, expected-gap-with-bottleneck-fallback **7**, expected-gap-with-width-fallback **6**, width **0**, round-robin **0**. Mean robust certification gaps: 0.2673, 0.2932, 0.3064, 0.3497 and 0.3441 respectively.

Each one-step-lookahead trajectory evaluated 2,304 hypothetical optimizer outcomes. Zero incorrect certificates; **13 confidence-coverage failure checkpoint records** were retained. Existing QO-08 baselines had zero lookahead optimizer calls. The new heuristic did not improve over bottleneck under matched budgets.

The key theoretical point persists: nesting the uncertainty rectangle makes the optimized gap nonincreasing along a valid evidence path. This theorem gives no claim of optimal probe selection. The experiment suggests that delayed information matters; a two-step planning policy or a trained recurrent proposal needs an additional independent evaluation.

Cases are small synthetic state/action hit/hold models, with three retrospective QO-10 cases and two source-frozen new ones. This does not measure neural or biological systems. Planning CPU wall time was not measured; exact optimizer-call counts are included in the receipt.
