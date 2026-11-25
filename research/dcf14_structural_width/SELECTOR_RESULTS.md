# DCF-14C: Held-out algorithm selection

2026-11-25. Exploratory CPU pilot. 90 development problems (grid, sparse, dense, cycle, bipartite; seeds1601–1606), 72 test problems (ladder, clique_chain, hub_core, chordal_cycle; seeds1701–1706), widths12,16,24. Known scopes, independent fair binary priors, noisy unary and pair parity likelihoods. DCF-14B exact VE16 reference. Reward +1 correct, -4 wrong, 0 abstention. Four candidate operators VE5, cutset, BP20, abstain. Three timing repetitions per operator, median. Selector predicts regret + .03*milliseconds from nine structural features: n, edge count, min-fill width, log2 estimated work, max degree, query degree, greedy cut count, cut feasibility, unary evidence strength. RandomForestRegressor 64 trees, depth4, min leaf4, seed71, one job. Structural rule chooses VE5 if width<=4, otherwise cutset if feasible and cut count <= threshold, else BP. Threshold chosen on development objective (4). No test tuning. Feature extraction and selection time charged to both routers, operator timing measured separately (sum, not end-to-end replay); feature time single sample, prediction time single sample. Fixed policies do not pay redundant feature computation. No training-cost amortization included, so learned result is favorable in this respect.

|policy|expected utility|regret|mean ms|objective|
|---|---:|---:|---:|---:|
|always cutset|.62261|0|3.83143|.11494|
|heuristic|.62261|0|4.55762|.13673|
|learned|.62261|0|7.01686|.21051|
|always BP|.61814|.00446|9.74709|.29688|
|always VE5|.12393|.49868|.33231|.50865|

Learned chooses VE5 12/72, cutset60/72; heuristic18/72 and54/72. All test cases solvable by cutset, limiting operator-selection diversity. All 54 independent small-system brute posterior checks across both splits agree within 9.992e-16. Mean feature overhead .77057 ms. Promotion FAIL: learned selector preserves exact optimal utility but loses to fixed cutset and heuristic on objective in every held-out family mean. No claim of general learned routing failure; fixed operator set, narrow simulator, unoptimized Python and small test. Timings subject to host noise; no matched peak memory or hardware claim. First execution finished but JSON serialization failed on numpy integer; rerun same seeds and settings with serialization-only fix. Both source hashes retained. Report committed; executable, receipts and manifests in dcf14c_reproducibility.zip conversation artifact. No paid compute. DCF-15/16 integration remains outstanding.

Next: eliminate duplicated structural analysis by reusing the elimination plan; compare bounded exact inference against a compiled simple dispatcher under held-out families with genuinely different optimal operators. A learned router should be retained only if it adds value to that baseline.
