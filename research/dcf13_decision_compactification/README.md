# DCF-13: Decision-directed compactification and exact hybrid inference

2026-11-25. CPU-only synthetic study following DCF-12. 432 arm-condition cells over three suites of 3,072 independent decisions each (9,216 generated episodes; arms share episodes). Primary, elimination follow-up and hybrid follow-up locally hash-locked before their respective runs. No external preregistration. No paid compute or real-model/hardware claim.

## Task and exact relevance theorem

Uniform independent K=8 or 12 latent bits. Each episode has 24 noisy Boolean observations with known factor scopes and bit-flip probabilities: one unary read per variable, then pair/triple XOR, AND or majority predicates. A query asks the parity of two bits. Four regimes: stable four-bit blocks; per-episode rewired blocks; rewired blocks with two random cross-block triple constraints; and ambiguous triple sensors, independently choosing XOR or majority with probability 1/2. The correct likelihood marginalizes the unobserved sensor type. All informed methods receive scopes, noise rates, and the true mixture weight; nominal uses weight zero. No learned graph discovery or autonomous model identification. In particular this is static inference over fresh episodes rather than the recurrent context inference in DCF-09–12.

Reward +1 for correct, -4 for wrong, 0 for abstention. Bayes acts when p>=.8 or p<=.2. Expected utility uses the full reference posterior, realized utility uses the generated true bit, and regret is reference-optimal expected utility minus the policy's expected utility. This synthetic loss makes errors consequential; it is not a measure of general intelligence. Because the prior factorizes and observation likelihoods have disconnected scopes, the posterior factorizes over the evidence hypergraph components. For query parity, its signed expectation is the product of the signed expectations on each relevant component. Thus query-directed enumeration ignores disconnected factors exactly. A capped arm abstains if any relevant component exceeds its cap. A fixed4 arm drops all cross-block factors and uses the resulting approximate distribution. Full enumeration streams assignments; it does not materialize a joint belief table.

## Primary (DCF-13A)

Seeds1009/1013/1019, 128 episodes per width/seed/regime, 24 conditions x6 arms. Numba FP64 routines; warmup then three interleaved timing repeats, median per condition. Timings exclude generation, compilation and decision scoring; include per-episode graph discovery, enumeration and prediction. State scope cap is an algorithmic bound, not matched total memory. Gate adaptive8 regret<=.02 and time<=full, per condition.

|Arm|Expected utility|Realized utility|Regret|Coverage|Mean ms/128|Median paired speedup vs full|
|---|---:|---:|---:|---:|---:|---:|
|full|.561995|.569010|0|.772461|36.993|1|
|fixed4|.286188|.288086|.275807|.491536|.221|116.90|
|adaptive4|.417989|.432943|.144005|.584310|.240|154.72|
|adaptive8|.515100|.528646|.046894|.706055|.886|64.17|
|adaptive_full|.561995|.569010|0|.772461|5.861|15.08|
|nominal8|.452727|.453125|.109268|.700521|.846|60.96|

Adaptive8 skips 9.60% of episodes; max posterior discrepancy on non-skipped informed adaptive arms 1.521e-14. Joint gate passes 19/24; time alone 22/24. Main failure is bridged K12, where 76.82% of queries exceed cap8; mean expected regret .37516. At K8 the same bridge regime fits cap8 but provides almost no enumeration saving. Full adaptive expected utility matches full enumeration in every condition. Stable fixed partitions are already exact; adaptive discovery overhead can be wasteful. The full enumeration baseline is deliberately simple, not a lower bound on inference cost.

## DCF-13B: stronger exact elimination and sensor uncertainty

Fresh RNG offsets with same system seeds/regimes; 24 conditions x8 arms. Greedy minimum union-scope elimination keeps query variables last, multiplies local factors and sums out one variable at a time. Scope cap includes the eliminated variable; if exceeded, abstain. Exponential cost depends on induced width rather than component cardinality. No assertion of optimal elimination ordering. Timing as above. Count live factor table entries separately (max148 FP64 entries, 1184 bytes, excludes metadata and scratch); not total working memory or a matched-RSS study.

|Arm|Expected utility|Regret|Coverage|Mean ms/128|
|---|---:|---:|---:|---:|
|full|.566724|0|.785156|34.769|
|adaptive8|.522035|.044689|.717448|.829|
|adaptive_full|.566724|0|.785156|5.475|
|ve4|.562089|.004635|.779297|1.510|
|ve6|.566724|0|.785156|1.511|
|ve_full|.566724|0|.785156|1.514|
|ve_nominal|.508789|.057935|.786458|1.523|
|ve_robust|.453167|.113557|.611979|2.957|

VE6 matches all 3,072 exact actions with zero skips; max encountered scope5, median paired speedup18.36x vs full. VE4 skips .6836% and passes the primary-style regret/time threshold in 21/24 conditions (post-hoc descriptive). Independent NumPy full joint enumeration checks 96 episodes, max abs discrepancy 7.661e-15. All informed elimination arms assert posterior agreement on non-skipped episodes. Timing not matched to an optimized external library.

For uncertain ambiguous factor weights in [0,1], each likelihood is bounded pointwise by its endpoint min/max. Products and sums preserve these inequalities; lower/upper masses for each class bound posterior probability by L1/(L1+U0), U1/(U1+L0). Accept only if the lower probability for a class is >=.8. This is valid for the specified independent conditional likelihood family, not unknown arbitrary correlations or all model misspecification. Bounds contain all correct-reference probabilities and five grid weights on each of 96 independent audit fixtures; theorem uses nonnegative mass inequalities, not the grid alone. FP64 numerical tolerance only, no interval-certified rounding. On ambiguous regime alone (768 episodes):

|Arm|Expected utility|Realized utility|Coverage|Wrong decisions|
|---|---:|---:|---:|---:|
|known mixture|.467546|.492188|.707031|33|
|nominal parity|.235806|.269531|.712240|68|
|robust bounds|.013320|.014323|.014323|0|

Robust policy is drastically overconservative; zero observed wrong answers does not imply zero risk. Oracle mixture weight is supplied, not learned. A future calibration experiment should estimate the mixture with independent validation and tighten uncertainty sets.

## DCF-13C: frozen hybrid, new seeds

Use relevant component enumeration when all queried components have <=4 variables; otherwise exact min-scope elimination with cap6. Fresh seeds1103/1109/1117, 128 episodes/condition, 24 conditions x4 arms, five interleaved timing repeats. Threshold and cap selected after follow-up, frozen before fresh data. Gate: exact actions everywhere and time <= faster of component_full and VE6 in >=18/24 conditions.

|Arm|Expected utility|Mean ms/128|Median paired speedup vs full|
|---|---:|---:|---:|
|full|.565353|35.153|1|
|component_full|.565353|6.161|13.87|
|ve6|.565353|1.563|17.79|
|hybrid|.565353|.676|23.16|

|Regime|Component ms|VE6 ms|Hybrid ms|Hybrid fallback fraction|
|---|---:|---:|---:|---:|
|stable|.262|1.369|.277|0|
|rewired|.322|1.541|.334|0|
|bridge|23.712|1.708|1.700|.95964|
|ambiguous|.348|1.634|.393|0|

All 3,072 hybrid decisions identical to full, no skips, max scope5, max hybrid posterior discrepancy1.4655e-14. Mean time lowest across regimes, but strict per-condition latency gate FAIL (6/24 vs required18/24): graph routing overhead loses to pure enumeration in many easy conditions. Speedups conditional on this mix of task structures. No guarantee on unseen higher-treewidth graph families or unobserved factor scopes. No neural training or accelerator measurement, and no general matrix exponent conclusion.

## Recovery, reproduction and DCF-13D (2026-11-25, repository run)

The 13A/B/C receipts were missing from the branch. They were regenerated from the hash-locked sources: decision-level results reproduce exactly, and timings move with the host (13C latency gate 7/24 here vs 6/24 preliminary). DCF-13D completes the Phase-1 checklist on fresh seeds: a cost-estimated hybrid (exact, but its latency gate fails at 4/36); min-fill and weighted min-fill orderings with budgeted pre-execution abstention; exact Bayesian model averaging over a shared unknown sensor law in Bernstein form; and shared-parameter credal certificates. The certificates raise safe coverage from 1.65% (13B pointwise bounds) to 39.8% with zero negative-utility actions. See [RESULTS.md](RESULTS.md), [THEORY.md](THEORY.md) and [PROTOCOL_INDEX.json](PROTOCOL_INDEX.json) (named so because `protocol.json`, the hash-locked 13A protocol, collides with `PROTOCOL.json` on case-insensitive filesystems); the integer core is kernel-checked in [MathCan `DCFDecision.lean`](https://github.com/patrickdugan/MathCan/blob/main/MathCan/DCFDecision.lean).

Reproduce 13D (after the 13A/B/C commands below; NumPy 2.3.5, Numba 0.65.1, pytest):

    sha256sum -c PRE_RUN_13D_SHA256   # fg.py line fails after later DCF-14 changes; the 13D run used the fg.py blob of the commit that added PRE_RUN_13D_SHA256
    python audit13d.py calibrate      # dev seeds only; the committed calibration13d.json is what the run used
    python audit13d.py run
    python summarize13d.py
    python -m pytest tests


## Reproduce

Python3.13.5, NumPy2.3.5, Numba0.65.1. From this directory:

    sha256sum -c PRE_RUN_SHA256
    python gym.py
    sha256sum -c FOLLOWUP_PRE_RUN_SHA256
    python eliminate.py
    sha256sum -c HYBRID_PRE_RUN_SHA256
    python hybrid.py

Raw receipts include per-episode action and realized utility, all timing samples, and independent audit results. Reproducibility archive excludes generated compiler caches. Reported work counters differ across methods (likelihood evaluations vs table construction/multiplication counts), so no equal-operation speed claim. All tests ran sequentially on one CPU thread, no paid compute.

## DCF-14 candidate

Use independently generated higher-treewidth graphs and partial/unknown factor scopes; add bounded elimination orders, cutset conditioning or adaptive search, plus a learned proposal with exact or interval verification. Calibrate ambiguous likelihood weights and measure coverage/regret with confidence bounds on fresh systems. Compare with optimized exact VE and component inference under a predeclared utility/latency objective, amortized preparation, and measured peak memory. Freeze an external registration if possible before opening held-out graphs. Avoid treating this finite static factor-graph result as evidence of a general LDT or TPU advantage.
