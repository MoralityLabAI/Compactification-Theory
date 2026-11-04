# QO-12: Exact two-step certification value and delayed-information plateau

**Protocol frozen 2026-11-04**, following [QO-11](EXPECTED_GAP_RESULTS.md). These are finite digital Bernoulli hit/hold systems with separately addressable, resettable microstates. The statistical guarantee continues to come from the unchanged [QO-07](ANYTIME_ACTIVE_QUOTIENT.md) beta-binomial e-process.

## Bellman formulation

Write E for the current public evidence ledger (per-cell hit counts, total counts, and nested confidence intervals). For a fixed class budget k let G(E) be the exact QO-07 certification gap R*(E)-L*(E). Given a candidate addressable probe c, QO-07's exact next-confidence-set update gives E(c,y) for Bernoulli outcome y. For selecting probes **only**, the declared Beta(1,1) posterior predictive is p_c(E)=(hits_c+1)/(count_c+2).

For h in {0,1,2} define

    V_0(E)=G(E)
    V_h(E)=min_c [(1-p_c(E)) V_(h-1)(E(c,0))
                       +p_c(E) V_(h-1)(E(c,1))].

Crucially, after observing the first y, the second cell c' is optimized anew on E(c,y), rather than fixed in advance. The exact two-step acquisition for c is the difference between G(E) and the first-stage expectation of V_1(E(c,y)).

**QO-12/T1 (finite Bellman monotonicity):** When all considered hypothetical evidence intersections remain nonempty, each QO-07 interval shrinks or remains fixed. Hence G(E(c,y))<=G(E). By convexity and minimization, 0<=V_2(E)<=V_1(E)<=G(E). This is an elementary dynamic-programming statement with written proof: the one-step inequality follows by averaging nonincreasing gaps; the second follows from V_1(E(c,y))<=G(E(c,y)) and minimizing. The controller ignores candidate policies whose hypothetical confidence intersection becomes empty and retains QO-07's separate validity checks.

**QO-12/T2 (strict delayed-value witness):** At n=2, d=1 with no observations, both QO-07 intervals are [0,1], R*=1/2, L*=0. A first probe of either cell has zero one-step value because the other interval remains [0,1]. For alpha=1/20 and two cells, an *ideal unrounded* one-trial e-process interval after a hit is [1/80,1] and after a hold is [0,79/80]. Querying the **other** cell second yields gap 79/160 if the two outcomes agree, 1/2 otherwise. Under independent Beta(1,1) planning predictions, agreement occurs with probability 1/2. Therefore

    V_1=1/2, V_2=159/320, G-V_2=1/320 >0.

The executable QO-07 implementation rounds endpoints conservatively outward, so its exact rational expected gain is slightly **smaller** than 1/320 while remaining strictly positive. This demonstrates delayed information value in the narrow synthetic contract. It is not a new general dynamic-programming theorem.

## Practical probe selector

[Two-step module](two_step_gap_probe.py) computes the exact depth-two adaptive Bellman policy with rational arithmetic and reuses identical leaf-gap results through a local cache. The fully general two-step routine is testable separately. The registered pilot's controller uses the expensive depth-two evaluation **only if every valid one-step candidate has zero predicted gap improvement**. On a positive one-step score, it uses the existing QO-11 selector. When depth-two is also flat or has invalid hypothetical branches, the controller falls back to QO-07 bottleneck or width. Each extra optimizer evaluation, counterfactual branch and fallback is counted.

The Beta planning model may be misspecified. Its predictive probabilities are never used to assert confidence coverage. Every real decision still passes through QO-07's anytime-valid certificate, under its IID stream and predictable sampling assumptions.

## Frozen paired study

[Casebook](two_step_gap_pilot.py): two **new** two-state one-action synthetic plateau cases, plus three **retrospective QO-11** cases (action0 singleton, action1 pair, two-action interaction). Three fresh seeds 449,457,461; checkpoints 24,64,96. Four policies: unchanged QO-08 bottleneck, unchanged QO-11 expected-gap+bottleneck, and QO-12 two-step-on-plateau with bottleneck or width fallback. Every policy shares deterministic per-cell Bernoulli response prefixes.

Total registered experiment: **5 cases x 3 seeds x 4 policies = 60 trajectories, 180 checkpoint records, 5,760 Bernoulli queries**, with 30 two-step planner receipts. Record every certificate, unresolved result, confidence failure, actual frozen prediction error, certified gap, paired positive and negative comparison, and compute work in exact leaf-gap optimizer calls. Do not infer statistical population superiority, neural transfer or biological cognition from this synthetic pilot.

The full receipt is emitted from CI and committed with source hashes at [results/two_step_gap/receipt.json](../../results/two_step_gap/receipt.json). Its verifier reproduces every matched trajectory byte-for-byte. Earlier QO-11 negative results are preserved and the old baseline runners remain unchanged.

**References:** Standard finite-horizon Bellman recursion; [QO-07](ANYTIME_ACTIVE_QUOTIENT.md); [QO-11](EXPECTED_GAP_THEORY.md); Howard et al. (2021), https://doi.org/10.1214/20-AOS1991. A companion Lean MOR-10 module formalizes the rational delayed-value witness and convex continuation bound, separately from e-process coverage.
