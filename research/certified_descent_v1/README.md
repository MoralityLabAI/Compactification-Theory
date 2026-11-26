# Certified growing-arity descent as a compactification objective

2026-11-26. Motivation: [Cogentic discrete logarithm result](https://github.com/google-research/google-research/blob/4a7bc8ab0bc770e71a259b7966c500496b99010a/cogentic_research_papers/discrete_logarithm/README.md). The mathematical abstraction and toy routing experiment are independent of the paper's technical proof.

## State and resource accounting

Use a multiset frontier F_t of unresolved certified subproblems. An action picks a node v, proposes a certificate with child list, and updates the frontier only after the exact verifier accepts. A terminal answer requires all pending nodes to close. The public state includes remaining verifier budget and memory budget. Charge attempted verification (including rejected attempts), proposal generation, and cache misses separately.

For positive sizes d and exponent s>=0, if sum_i d_i^s <= d^s at every accepted step, total frontier potential never increases. At any depth there are at most N^s nodes for root size N. For an additive cost, the stronger amortized condition c(v)+sum P(child_i)<=P(v) gives a telescoping work bound. This is a conditional invariant; a learned representation cannot certify it by itself.

For recurrent policy h_{t+1}=F_theta(h_t, frontier_t, budget_t, receipt_t), learn an action value for expected remaining total work, with a separate exact verifier. BPD proposal features may help choose among candidates; candidate rank changes cannot bypass the verifier. Cache canonical states by domain, representation and certificate version, with an explicit memory cap. Measure peak bytes, calls, coefficient operations and wall time rather than treating fewer passes as lower cost.

## ALG-DESCENT-01 development receipt

A stdlib Python polynomial routing proxy used 100 roots (five small primes, five degrees 8 through 128, four seeds), 300 rollouts. Public ordered linear factors were supplied to every policy; all costs are successful schoolbook verifier coefficient multiply-adds. Actions are balanced contiguous partitions with arities 2,3,4,8. At degree 128, fixed binary costs 9726 (depth 7), maximum arity costs 9672 (depth 3), and DP-optimal costs 9450 (depth 4). At degree 64, those costs are 2718,2664,2580. Source SHA256 6510d6cbe9b4b3a77b571c8204e99ec3ce744771ce23a0cea5590a430ceb7a17; stdout JSON SHA256 5ef7b4e3fcdac30cca257c5bf7ecc321a36d3d6b08447ac8dec0964872e5de09.

The proxy supplies factors, excludes proposal construction, uses fixed balanced partitions, and its cost is determined by degrees; it is not evidence for learned routing, speedup of algebraic discovery, or a DLP algorithm.

## Proposed prospective experiment

1. Freeze disjoint generator seeds and separate held-out degree and arity regimes. Add heterogeneous certified transformations and public costs that cannot be inferred from degree alone. Include invalid and costly decoy actions; ensure each solvable root has a certified path.
2. Compare fixed binary, max arity, cheapest immediate, exact small-state DP, learned TRM with identical public candidates, and BPD/MMS routing using matched parameter/training/compute budgets. Keep proposal and verifier call budgets equal and separately report cache allocations. Run cache-off and shuffled-BPD controls.
3. Score verified full closure first, then total attempted work among matched solved roots, worst-tail cost, peak bytes and wall time. Report root-paired uncertainty and seed-level variation. A learned model that only matches the trivial degree-only DP proxy has no demonstrated advantage. Register thresholds before training.
