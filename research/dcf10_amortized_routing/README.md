# DCF-10: Amortized correlation-aware routing

2026-11-25. Six fresh systems, 24 paired budget/scenario conditions, 96 metric cells. CPU exploratory experiment following DCF-09. Protocol/source hashed before execution locally; no external preregistration. True graph library, bit noise rates, context hazard and label corruption supplied to all arms; library mismatch and parameter learning deferred.

## Exact screening rule

For current partition a0 and alternative a, the routing objective is L(a)+[fee+mu*D(p||projection_a(p))]/T, with switching charge zero for a0. Given fee, mu>=0 and T>0, if L(a)+fee/T >= L(a0), the alternative cannot strictly beat retention. With ties resolved in favor of retention, pruning such candidates preserves the decision within the original candidate set (retain, local expected-NLL minimizer, fixed choice). Nonnegativity is implemented by the inherited projection cost clamp. This is objective-relative, not a guarantee of predictive optimality. Event arm applies the same screen every fourth step or whenever the unpriced expected gain exceeds .012. The expected-loss score is still computed each step. All dynamic arms avoid regrouping if there is no switch. No approximate cache of posterior-dependent entropies; static intersection lookup tables are cached in inherited projection code.

## Protocol

K=6,8, seeds 701,709,719. Common DCF-09 graph library and simulator; noisy30 and hidden labels with correct assumed hazard .125. Candidate pool is shared: union/context partitions up to cap4 and full if feasible. Per-context NLL estimated on clean static 24x40 validation words; fixed partition selected on distinct noisy30/hidden validation. Test 32x96 words per scenario/seed, identical fixtures reused across budgets and arms. Fee .02, mu 1, horizon 8. Equal upper persistent state caps 640/1280 bytes, charge FP64 conditional factors and four context weights plus 8 bytes per trajectory for route identifier (also charged to fixed). Excludes working arrays, model/kernel tables, output and allocator state. One warm run and three interleaved randomized timed repetitions per arm, sequential single-threaded CPU, no concurrent experiment jobs launched. Timer covers filtering/routing loops, not kernel construction/calibration. Raw samples retained. No energy/GPU measurements. Gate event KL <= 1.10*step KL and <= fixed KL, plus median runtime <= fixed. No test tuning. Exact same-information full joint reference, conditional KL in nats/read.

## Results

Means across 24 conditions (means of within-condition median timings):

|Arm|KL|Seconds|Candidate projection context evals|Mean bytes|Switches/word|Discarded KL/word|
|---|---:|---:|---:|---:|---:|---:|
|fixed|.00511306|.102135|0|562.67|0|0|
|step|.00346779|.195111|18135.50|635.29|2.779|.014722|
|screen|.00346779|.177905|571.67|635.29|2.779|.014722|
|event|.00355583|.172143|322.00|633.89|1.839|.009565|

Screen eliminates 96.85% of candidate context projection evaluations, with all 24 route paths and predictions identical to step. Event eliminates 98.22%, with mean KL 2.54% higher than step and 30.46% lower than fixed. Counts exclude actual switch materialization; those costs are timed. Median paired time ratios relative to step: screen .9130, event .8979. Median paired ratios relative to fixed: 1.3406, 1.3952. Event faster than step 22/24 and fixed 2/24; quality gate 22/24, joint gate 2/24 (FAIL). Strict predictive improvement over fixed 18/24. Quality failures at 1280 cap; worst event/step KL ratio 1.424. All state caps pass; event sometimes occupies more bytes than fixed within cap. Not matched actual occupied memory or equal computation.

|Cap|Fixed KL|Step/screen KL|Event KL|Event mean seconds|Event quality gate|Event joint gate|
|---:|---:|---:|---:|---:|---:|---:|
|640|.00518314|.00413679|.00418885|.153010|12/12|1/12|
|1280|.00504298|.00279878|.00292281|.191275|10/12|1/12|

## Replay and profiling

24 independent replay fixtures (8x32) compare fixed against frozen DCF-09 filter and step against frozen DCF-09 dynamic: max absolute prediction delta 0. All 24 causal prefix checks and repeat determinism checks pass. Representative diagnostic profile, K8 seed709 1280/noisy30, 32x96: fixed 384 kernel calls, step/screen 1516, event1536. Kernel cumulative profiled times .148/.402/.398/.399 seconds; projection cumulative times 0/.116/.059/.025. Profile timings instrumented and separate from primary measurements. One tracemalloc diagnostic peak including initialization and output arrays: fixed 290064, step 351728, screen 341457, event 337493 bytes; not RSS or comprehensive working-memory bound. Kernel rows identical across arms (4*n*l). Main overhead from partition fragmentation and per-group indexing, not remaining projection count. A GPU implementation may differ. Strict below-parity runtime target not achieved. No claim of actual LDT hardware or general matrix exponent implications.

## Reproduction

Python 3.13.5, NumPy 2.3.5, torch 2.10.0+cpu (frozen dependency imports). From directory: `sha256sum -c PRE_RUN_SHA256; python router10.py; python audit10.py`. Results and all six execution logs in conversation archive. Frozen DCF-09 dependency copies included. No paid compute. Next candidate DCF-11: partition-cohort routing and a fused masked kernel; price number of distinct layouts, not just individual loss; compare to shared fixed backend, with time/working-memory bounds. Separate graph-library mismatch and learned noise/hazard experiments.
