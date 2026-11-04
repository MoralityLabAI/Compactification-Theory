# QO-13: Budgeted probability of obtaining a QO-07 certificate

**November 4, 2026.** Finite synthetic resettable hit/hold systems. No neural or physical experiment. Companion to [QO-12](TWO_STEP_RESULTS.md).

The new objective is the probability that *QO-07's existing certificate* becomes feasible (robust upper R*<=tau) or infeasible (optimistic lower L*>tau) within h extra addressed probes. Let E denote full public evidence, with beta-binomial e-process intervals, counts and hits. Beta(1,1) is used **only to predict possible next bits for planning**.

    V_0(E) = 1 if E is certified, else 0
    V_h(E) = 1 if certified, else
             max_c [(1-p_c(E)) V_(h-1)(E(c,0))
                    + p_c(E) V_(h-1)(E(c,1))]
    p_c(E) = (hits_c+1)/(draws_c+2).

This is a classical finite Bellman recursion with adaptive second and later actions. Invalid hypothetical intersections have terminal value 0 and remain in the probability calculation. Induction proves 0<=V_h<=1 and V_(h+1)>=V_h, because success is absorbing. The QO-07 confidence-sequence coverage guarantee remains a **separate statistical premise**, based on fixed stationary IID Bernoulli streams and predictable probe choices.

**Exact delayed certificate witness.** Two cells, one action, alpha=1/20, 10-bit outward rational QO-07 endpoints, tau=1/5, and histories of 18 all-holds at cell 0 and 18 all-hits at cell 1 have current optimistic lower bound 49/256. The one-probe chance of certificate is zero; the exact optimal two-probe chance is **19/21**. Independently enumerating all QO-07 decision trees with replayed observations verifies this.

**Work gate.** An exact horizon 2 or 3 Bellman tree is built when each addressed cell has >=8 samples and an optimized error bound is within 1/20 of the threshold. Otherwise the controller uses unchanged bottleneck probing. Horizon 3 is clipped to 2 for interfaces with >2 cells. Every Bellman candidate, optimizer call, invalid branch and fallback is counted.

**Fixed paired pilot.** Four source-defined synthetic cases (two retrospective QO-12, two new), seeds 541/547/557, 24/64/96 observations, and policies bottleneck/certificate_h2/certificate_h3. Exactly **36 trajectories, 108 checkpoints, 3,456 Bernoulli observations**, matched per-cell random prefixes. The planned [receipt](../../results/certificate_reachability/receipt.json) preserves all positive/negative results and coverage failures.

Literature: [Cheng & Huan, optimal sequential Bayesian experimental design](https://arxiv.org/abs/2509.21734); [Howard et al., confidence sequences](https://doi.org/10.1214/20-AOS1991). Both mathematical mechanisms predate this integration.
