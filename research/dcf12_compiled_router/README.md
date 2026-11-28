# DCF-12: Fully compiled collective routing

2026-11-25. Six fresh synthetic systems, 120 paired conditions, 720 primary metric cells. CPU-only exploratory follow-up to DCF-11. Primary source, protocol and dependencies hashed locally before execution; no external preregistration. Independent audit and particle/table diagnostics separate from primary selection. No paid compute.

## Method

Keep the DCF-11 finite layout-subset objective and candidate choices (current, per-row expected-NLL minimizer, validation-selected fixed). Precompute old/new factor offsets and marginal index maps for each nonempty intersection of a pair of partitions. Each new factor is the product of corresponding old-factor intersection marginals. The projection cost is the sum of their entropies minus the sum of old-factor entropies, weighted across contexts by the current posterior. Compile cost construction, entropy, materialization, schedule, label update and entire recurrent loop using Numba, reusing DCF-11's fused inference kernel. No deliberate objective approximation. Exponential subset scheduling remains; candidate library size 4–10. Projection tables scale as O(P^2 K 2^m) for P layouts, K variables, max cluster size m.

## Protocol

K=6,8; seeds901,907,911. True graph library, bit noise rates, hazard .125 and label error supplied to every arm. Noisy30 and hidden scenarios; same-information exact joint reference. DCF-11 calibration on fresh systems (24x40 static clean validation words/context, separate 24x40 uncertain-context words; select fixed by validation NLL). Persistent state caps 640/1280 bytes including context weights and route identifier; padded allocated arrays 552/680/1064 bytes. Test 64 steps at B=1,8,32,128,256. Fixture seeds differ by batch/scenario; common fixture across arms/caps. Reconsider every 4 steps; fee .02, mu1, horizon8. Collective gamma=.00125*B in both backends; illustrative price, not hardware-calibrated. Independent gamma0; fixed no routing. Three policies x two backends: DCF-11 compiled inference with Python routing vs fully compiled DCF-12. One warmup and three randomized interleaved timing repeats, single-thread CPU, no concurrent experiment jobs. Loop timing excludes validation, JIT and table construction; new loop includes buffer initialization, old loop timer excludes it. Gate: collective KL <=1.1*independent and median time <=fixed under equal allocated state cap. No test tuning.

## Primary results

All 720 cells finite, repeat deterministic within backend and cap-bound. All 360 matched old/new paths identical; max prediction discrepancy 7.771561e-16, floating-point agreement rather than bitwise identity. Mean ms of condition medians (24 conditions per B), paired speedup median separately:

|B|old fixed|new fixed|old independent|new independent|old collective|new collective|collective speedup|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|1.4121|.0897|2.9468|.1222|3.2305|.1219|27.438|
|8|1.7722|.3009|5.3327|.4449|4.5588|.4349|10.577|
|32|2.4738|.9217|8.4711|1.4828|6.5357|1.4097|4.419|
|128|5.2044|3.4189|14.3709|5.4875|11.6331|5.2838|2.047|
|256|9.1758|6.8225|21.4003|11.1940|17.5091|10.5022|1.646|

Aggregate median paired speedup fixed2.740, independent4.938, collective4.419. Collective throughput B*64/mean time rises from .525M to 1.56M recurrent row-steps/s; leveling off after B128 on this CPU. At B1 collective fee is constant so routes match independent.

|B|fixed KL|independent KL|collective KL|joint gate /24|
|---:|---:|---:|---:|---:|
|1|.00336947|.00348947|.00348947|0|
|8|.00284294|.00262969|.00249992|2|
|32|.00339185|.00307199|.00319121|2|
|128|.00325047|.00287011|.00293349|2|
|256|.00346200|.00307106|.00314296|1|

|Compiled policy|mean KL|mean ms|active layouts|active bytes|
|---|---:|---:|---:|---:|
|fixed|.00326335|2.3107|1|658.67|
|independent|.00302647|3.7463|1.5911|657.11|
|collective|.00305141|3.5505|1.1453|660.67|

Collective reduces excess layouts above one by75.4%; mean KL rises .824% vs independent and falls6.49% vs fixed. Median paired collective/fixed runtime1.403. Quality gate111/120; joint latency-quality gate7/120 (FAIL). Six passing cells tie fixed KL and one is slightly worse. Zero cells achieve strictly better KL (difference >1e-12) than fixed at <= fixed latency. Timing-only passes not evidence of a quality gain; small timings noisy. Common allocated cap does not imply equal active payload.

## Independent audit and diagnostics

64 randomized old/new partition fixtures compare compiled projection with inherited factor projection and direct joint enumeration: max KL error3.83027e-15, factor error3.33067e-16. 24 future-observation flip checks preserve all rows' routes and predictions through index8. On fresh streams, eight focal trajectories per system/cap/scenario (192 total) tested solo and in batch32. Independent routing matches solo routes, max prediction delta8.88178e-16. Collective changes routes48/192, max probability delta .255555. Mean focal realized NLL change +.0006174 nats/read, p95 +.010356, range -.018495 to +.023388, 23 positive and23 negative beyond1e-12. Finite paired diagnostics, no fairness guarantee or claim of privacy leakage.

12 table configurations: inference tables21,280–176,080 bytes, projection tables13,984–238,960, combined35,264–415,040 bytes; median configuration median prep time4.871ms, five repeats. Static storage amortizable but not free. State caps exclude lookup tables, working memory, temporary arrays, allocator, history outputs, model and JIT overhead. No matched peak working-memory claim.

Post-primary particle diagnostic on B32 fixtures, two particle seeds per condition, 48 cells. Frozen DCF-09 packed uint16 particle implementation, equal persistent cap; not optimized or matched working-memory/time comparison:

|cap|fixed KL|independent KL|collective KL|particle KL|particle mean runtime ms|
|---:|---:|---:|---:|---:|---:|
|640|.00378443|.00353311|.00364508|.00966991|105.04|
|1280|.00299926|.00261087|.00273734|.00483397|196.61|

Particle timing descriptive single-run Python/NumPy baseline, no speedup claim against optimized particles. Known graph/rates/hazard and short task families limit generalization. Model mismatch and learned parameters deferred.

## Reproduce

Python3.13.5, NumPy2.3.5, Numba0.65.1, PyTorch dependency imports. In repo, bash prepare.sh copies frozen DCF-11 and DCF-10 dependencies and checks manifests. Archive already contains dependencies. Run OPENBLAS_NUM_THREADS=1 python router12.py; python audit12.py; python diagnostics12.py. Archive dcf12_reproducibility.zip contains full report, exact sources, raw primary receipts, timing samples, logs, audits, diagnostics and SHA256SUMS. Archive SHA256 3002509d41893709f1d3256b2f606c3dec505e08a800d413cae9976eb795fe28. No accelerator or energy claim.

## DCF-13 candidate

Stop optimizing routing overhead in this weak-benefit gym. Use richer coupled dependencies and tasks with independently verified action utility, budgeted abstention and graph-library mismatch. Compare optimized fixed/particle/full-state baselines under total working-memory and latency budgets. Promote only on strictly better task quality at matched latency or strictly better latency at matched quality.
