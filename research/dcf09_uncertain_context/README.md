# DCF-09: Uncertain-context predictive states and budgeted routing

Date 2026-11-25. Six fresh synthetic systems, 576 primary and 108 follow-up metric cells. CPU-only; no trained model or accelerator claim. Protocol/source SHA-256 locked locally before each run; no independent preregistration. Follow-up designed after primary results.

## Information boundary

Reuse DCF-07 graph generator (independent, reciprocal pairs, disjoint XOR triples, connected XOR cycle), K=6,8, seeds 601,607,613. **True graph library and bit noise rates supplied equally to all arms.** This isolates context inference and state approximation; no autonomous graph discovery. Hidden context changes with probability .125 to one of the other three uniformly. Clean, noisy15, noisy30 (uniform wrong label), missing50, hidden (no label), bursty30 (alternating .04/.65 hazard every 16 steps, .30 label error). All arms know label error rate and use assumed hazard .125, so the full-state reference is exact under the assumed model and misspecified for bursty30. Hidden bit reset independent fair. Actions/read indices observed; predict current read before its value; context label available before prediction. Privileged comparator sees true context.

## Filters

Context-specific mixture stores q(z) and product of cluster beliefs conditional on z. At context transition, mix the old conditional marginals using p(z_old | z_new, history) then re-factorize. Update label posterior, run each context transition and predict, aggregate read likelihood, then update q(z) and conditional beliefs using the observed bit. A single cluster spanning all bits gives exact Bayesian recursion. Approximate clusters discard cross-cluster correlation (including correlation from mixing contexts). Shared arm retains one factored bit belief, updates context weights using observation likelihood, then averages conditional posterior marginals back to a single shared state. MAP uses label-only posterior, one context per step, no observation-driven context update. Particle arm samples label-conditioned context transitions, importance-weights by label and observation likelihood, uses stratified resampling every step; two independent particle seeds per condition, packed uint16 context/bit code.

State cap counts actual persistent belief array payload per trajectory: mixture FP64 8*(4+4*sum 2^cluster_size), shared/MAP 8*(4+sum 2^cluster_size), particles 2*N bytes. Excludes common model parameters, kernel tables, allocator overhead, RNG state, temporary arrays and output histories. Equal upper caps, not equal occupied bytes or compute. Particle transition tables are exponential in K in this implementation; no scalable particle-memory claim. Full reference 2080/8224 bytes at K6/8. Full K6 fits 2560 cap, but primary compact candidate pool excludes full single-cluster reference; explicit full-state controls added on follow-up.

## Primary protocol and results

Greedy pooled graph partition caps 1,2,3,4. For each system, mode and budget choose lowest mean NLL over four validation regimes (noisy15, noisy30, missing50, hidden), 16 words x32/regime, no test selection. Test 48 words x80/regime, six regimes/system. 320,640,1280 particles at caps 640,1280,2560. Cache identical selected partitions across caps. 576 metric cells includes reference cells and repeated selected configurations; not independent sample count. Main comparison averages 24 system-regime pairs, two particle repeats averaged per pair. Lower predictive conditional KL to same-information full-state reference is better (nats/read).

|Cap bytes|MAP KL|Shared KL|Mixture KL|Particle KL|Mean mixture bytes|Mixture wins vs MAP/shared/particle|
|---:|---:|---:|---:|---:|---:|---|
|640|.020604|.008089|.006112|.011232|544|24/24,20/24,24/24|
|1280|.020604|.008089|.004967|.005750|650.67|24/24,24/24,13/24|
|2560|.020604|.008089|.004967|.003103|650.67|24/24,24/24,4/24|

MAP/shared mean bytes 160/186.67; particle occupies full cap. Mixture beats hard selection, and beats tested particles at smallest cap; at large cap particles improve beyond tested compact partitions. At 2560 cap an exact full-state K6 filter should supersede approximations if prediction quality alone is the objective. This is not an equal-compute or globally optimized Pareto claim. All timings single CPU runs with concurrent jobs; no controlled latency benchmark.

K6 at 640 cap, KL .004577 mixture vs .021996 MAP (79.2% lower), .008182 shared, .008751 particles. K8 corresponding .007646/.019212/.007995/.013713. Full-state context accuracy with hidden labels .33385 vs MAP .264 (chance .25); context remains uncertain. Exact same-information vs privileged predictive KL .009666/.018158/.010080/.040854 for noisy15/noisy30/missing50/hidden. This quantifies uncertainty cost separately from representation approximation.

24 independent explicit joint-state mass-transport checks of full filter, including context pre/post probabilities and predicted bit, max abs disagreement 1.3878e-15. 32 primary causal-prefix tests across unique selected configurations and a particle configuration, max delta 0. These audit recursion under assumed transition model, not global correctness under misspecification.

## Follow-up: posterior-weighted dynamic routing

Fresh RNG streams. Expanded fixed and dynamic candidate pool shared: pooled and context-specific graph partitions with cap 1..4, full if feasible. Budgets 640/1280 bytes. Validate fixed choice on noisy30/hidden, 16 words x32; estimate per-context partition NLL on separate clean static-context validation, 16 words x32. Test 24 words x64 per regime/system. For each trajectory and time step select retain/current, posterior-expected best, or fixed baseline partition minimizing q-weighted local validation NLL plus switching fee .02/8 and mu times posterior-expected conditional projection KL/8. mu=0 dwell-only, mu=1 correlation. Horizon fixed at 8 (no learned dwell-time estimator). Exact entropy projection prices the current approximate conditional distribution, not the true lost information. Charge actual occupied bytes, switches, projection context evaluations and timing; both dynamic arms compute projection estimates even when mu=0 (overhead not optimized). Compare full MAP/shared/mixture controls with actual size on same fresh streams; full controls not included in the dynamic cap comparisons if infeasible.

|Cap|Arm|KL|Mean bytes|Switches/word|Discarded KL/word|Projection evals/condition|CPU seconds/condition|
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|640|fixed|.003995|522.67|0|0|0|.0890|
|640|dwell|.003146|529.37|3.299|.043706|5759|.1775|
|640|correlation|.003075|529.78|2.774|.012010|5906|.1710|
|1280|fixed|.003640|736|0|0|0|.0994|
|1280|dwell|.002443|716.06|3.931|.055757|9205|.2277|
|1280|correlation|.002343|729.24|3.240|.014467|9487|.2041|

Each row mean 12 system-regime pairs. Correlation beats fixed 10/12 and 11/12, dwell 6/12 and 7/12, at the two caps. At 1280 cap KL falls 35.6% vs fixed, discarded conditional information falls 74.1% vs dwell, CPU time about 2.05x fixed. No throughput win. Full shared K6 544 bytes KL .004291, K8 2080 bytes KL .003235 on follow-up. Full mixture reference zero KL by definition, 2080/8224 bytes. All 108 follow-up metric cells finite; all cap-bound routing cells comply with persistent-array cap. Limited systems/seeds, no statistical significance or uniform guarantee. No learned graph library, real LDT model, hardware energy or generalization claim. A context-posterior mixture helps in this domain, but route overhead and approximate loss proxies still limit efficiency.

## Reproduction

Python with NumPy and PyTorch (frozen DCF-07/08 dependency imports torch; no neural training here). From directory:

    sha256sum -c PRE_RUN_SHA256
    bash run_all.sh
    python audit.py
    sha256sum -c FOLLOWUP_PRE_RUN_SHA256
    while read k seed; do python adaptive.py --k "$k" --seed "$seed" --out "followup/${k}_${seed}"; done < cases.txt

Primary first process group hit a wall-clock timeout during K8 runs. Partial receipts/logs preserved; reran K8 with identical source/seeds to completion, no changed protocol or result-based tuning. Raw receipts, source, pre-run hashes and logs in conversation archive dcf09_reproducibility.zip. No paid compute.

## DCF-10 candidate

Evaluate amortized context/partition routing under a stricter wall-clock or operation budget: cache projection statistics, calculate loss proxy from action-specific conditional predictive likelihood, implement event-triggered route reconsideration, and test against optimized fixed mixtures and particles on fresh systems. Include matched working memory and time, not only persistent arrays. Add library mismatch and learned noise/hazard rather than supplying true parameters; report failure/abstention. 
