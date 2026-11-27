# SG-04/v0.1: certified bounded spectral search

Companion to [SG-03](../sg03/README.md). In reverse routing, the state (t,S) is the number of visits left and the union of already consumed labels. The partial suffix objective g charges each component at its last use. For any real multipliers eta_i, relax each remaining per-visit cardinality constraint by adding eta_i(k_i-|A_i|). The unconstrained component objectives separate. An active label can take every positive adjusted utility. An inactive label either stays unused or chooses a last consumer l, takes every earlier positive adjusted utility, and pays lambda*b_j*(x_l-x_0) exactly once. The sum of these maxima plus sum eta_i*k_i is an upper bound for every legal completion. The minimum over valid multiplier vectors and the cost-free top-k bound stays admissible. This pilot uses six fixed multiplier vectors: zero and the kth local utility plus shifts -6,-3,0,3,6 (k=0 uses maximum +1). No floating optimization or rounding claim; all generated inputs, bounds and objectives are small exact JS integers.

Bounded search tracks each open state and its certified upper bound. It safely removes a state with no better suffix objective at identical (t,S), or an upper bound no larger than the incumbent. For a memory-capped frontier, every evicted state's upper bound is retained in a scalar maximum. Thus max(incumbent, live frontier upper bounds, evicted upper bounds) remains an upper bound on the optimum. Eviction can prevent later certification even when the incumbent happens to be optimal. Search expands complete action sets, stops before exceeding its transition cap, and never uses the learned prediction as a proof bound. A zero gap certifies optimality in the abstract finite task.

Run node sg04.mjs > sg04.json. Same source executed in a CPU JS isolate before commit. 48 independent training tasks (8 components, 6 visits), 72 fresh exposed evaluation tasks (9 components, 7 visits), 24 larger exposed tasks (10 components, 8 visits). All use planted persistent, rotating, mixed regimes and nonuniform integer bit costs/gaps; "in_domain" denotes the same generator families with a larger dimension than training. The ranker fits ten linear coefficients on exact prefix-DP slack targets, and the shuffled-label ranker fits targets permuted within each training root. Both see full offline utilities, not future live activations. Compare weak priority, strong priority, learned slack priority, shuffled slack priority, and best-first frontier capped at 8 nodes. All arms use the same strong pruning/certificate and initial incumbent (best of local, global fixed pool, reverse myopic). Thus the weak arm tests ordering only; beam8 is a capped best-first frontier rather than layerwise beam. Caps are 1024, 4096, 16384 generated action transitions. Bound/DP preprocessing, initial heuristic, sorting, learned score overhead and retained frontier memory are separate costs, so equal edge counts are not equal CPU time. Exact oracle computes all prefix states for evaluation only; the bounded search is passed no exact values. No native model, GPU, paid inference, confirmation, or claim of hardware savings. All evaluation roots are exposed development, with complete root/task/arm outputs.

Exact finite checks: 48 tiny tasks, 1578 brute schedules, 3072 small state bounds; 86016 training and 516096 evaluation bound checks; 1440 bounded certificates verified by independent exact prefix DP and last-use schedule scoring. All passed. Initial regret 3.7188; average root slack weak 170.8333, strong 27.5625. Train labels 6912 are correlated within roots. No inference from individual states. The ranker is an offline development policy; compare all baselines before promotion.

## Aggregate results
### Transition cap 1024
- weak: regret 3.4063, certified gap 7.9167, 50/96 optimal, 39/96 certified, mean transitions 829.34
- strong: regret 2.5625, certified gap 4.9167, 59/96 optimal, 47/96 certified, mean transitions 795.25
- learned: regret 2.0000, certified gap 4.7917, 67/96 optimal, 47/96 certified, mean transitions 788.34
- shuffled: regret 2.5729, certified gap 4.9271, 58/96 optimal, 46/96 certified, mean transitions 797.59
- beam8: regret 2.5625, certified gap 4.9167, 59/96 optimal, 47/96 certified, mean transitions 795.25

### Transition cap 4096
- weak: regret 1.8021, certified gap 2.3958, 79/96 optimal, 79/96 certified, mean transitions 1728.28
- strong: regret 0.1979, certified gap 0.2813, 94/96 optimal, 94/96 certified, mean transitions 1249.44
- learned: regret 0.0938, certified gap 0.2188, 95/96 optimal, 94/96 certified, mean transitions 1269.47
- shuffled: regret 0.2083, certified gap 0.2917, 93/96 optimal, 93/96 certified, mean transitions 1318.75
- beam8: regret 0.1354, certified gap 1.0938, 94/96 optimal, 79/96 certified, mean transitions 1161.47

### Transition cap 16384
- weak: regret 0.0938, certified gap 0.2396, 95/96 optimal, 95/96 certified, mean transitions 2511.84
- strong: regret 0.0000, certified gap 0.0000, 96/96 optimal, 96/96 certified, mean transitions 1346.16
- learned: regret 0.0000, certified gap 0.0000, 96/96 optimal, 96/96 certified, mean transitions 1367.59
- shuffled: regret 0.0000, certified gap 0.0000, 96/96 optimal, 96/96 certified, mean transitions 1436.88
- beam8: regret 0.1146, certified gap 1.0729, 94/96 optimal, 79/96 certified, mean transitions 1177.25

All results are in results.json; no hidden test roots or teacher weights are included. This is a finite planning model, not a proof of the original Euclidean spectral-flat theorem.
