# SG-03/v0.1: optimal source-bound spectral routing

[SG-02](../sg02/README.md) prices a shared label with bit weight b_j at its last consumer's distance from a fixed producer. Here local utility is additive and known for every visit, component and root. Schedules have an exact per-visit read budget k_t; sites are fixed sorted coordinates, bit prices are positive integers and lambda is nonnegative. We maximize total utility minus lambda times assigned cost. This is offline planning with full future and past utility information, not native model inference, estimated entropy, physical bandwidth or online routing.

## Exact oracle
Let V_(t+1)(S) be the best suffix objective with union S. Starting with V_(T+1)(empty)=0, assign A of cardinality k_t and relax V_t(S union A) with V_(t+1)(S)+U_t(A)-lambda*(x_t-x_0)*sum_(j in A minus S)b_j. Each component's cost is charged precisely once at its last consumer. This is a complete enumeration of legal actions and suffices by induction since earlier costs depend only on S. Maximize over final unions and backtrack. Time O(T 2^d max_t binom(d,k_t)) plus mask-table preparation and optional state sorting; value storage O(2^d), backpointers O(T 2^d). Implementation sorts active states for deterministic tie-breaking/beam pruning; exact oracle can avoid that sorting. Independent brute-force schedules confirm 96 small tasks and 11664 schedules. Every test schedule is checked against a separate last-use cost computation and the oracle upper bound.

## Development campaign
Run node sg03.mjs > sg03.json. Same source ran in an isolated JS CPU runtime before commit. No weights, external packages, paid inference or GPU. Three planted regimes (persistent, rotating, mixed), 24 independent train and 24 fresh exposed evaluation generator roots each. Eight components, six visits, 2 or 3 reads per visit, nonuniform integer bit costs and coordinate gaps, lambda 1..3. Teacher supervision consists of 432 reverse decisions and 3456 correlated component labels. One 10-coefficient logistic ranker fits only train roots; its frozen reverse-greedy rollout sees the current union and all local utilities. A separate matched ranker shuffles earlier/later feature identities in train and eval. Local utility, fixed global dictionary, analytic reverse-myopic, analytic lookback, beam-8, and exact teacher are all included. The lookback bonus is a hand-chosen 0.25 heuristic, not optimized. All choices and seeds are development-only; no sealed confirmation. The reported "exact" counts objective ties, not exact agreement of tied schedule identities. All root outcomes are archived with task input and output masks. Regret is exact objective gap in assigned utility units; no significance or mechanistic inference. No native BPD or Inkling data, no EvalMath-ready claim. Use a new independent calibration set before any native run. The geometry-aware controller is an offline utility model with assigned prices; it is not a trained neural recurrence.

## Result
- local: objective -7.1667, mean regret 54.5556, cost 83.7639, utility 161.1111, exact 0/72
- fixed_pool: objective 42.5694, mean regret 4.8194, cost 42.5000, utility 125.8472, exact 32/72
- myopic: objective 32.8750, mean regret 14.5139, cost 40.0139, utility 108.5972, exact 19/72
- lookback: objective 33.8750, mean regret 13.5139, cost 38.7778, utility 107.9028, exact 15/72
- learned: objective 36.3194, mean regret 11.0694, cost 51.2361, utility 137.0833, exact 15/72
- shuffled: objective 33.3750, mean regret 14.0139, cost 52.2361, utility 136.0278, exact 16/72
- beam8: objective 45.9583, mean regret 1.4306, cost 43.6250, utility 129.2778, exact 58/72
- oracle: objective 47.3889, mean regret 0.0000, cost 45.5556, utility 134.7500, exact 72/72

The result is a small-domain oracle and a distillation development fixture. Compare all controls, including strong beam search, before promoting a learned policy. Do not treat repeated decisions/components as independent roots.
