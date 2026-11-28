# DCF-11: Collective layout economics and compiled mixed-layout filtering

2026-11-25. Six fresh synthetic systems (K=6,8; seeds811,821,823), 24 paired cap/scenario conditions, 168 metric cells. CPU only. Primary spec and sources hashed locally before execution, not externally preregistered. Known true graph library, rates, hazard and label corruption supplied to all arms as in DCF-10.

## Objective and implementation

At scheduled reconsideration steps, each row offers retain, its local expected-NLL minimizer, and validation-selected fixed partition. Charge estimated context-weighted NLL plus (switch fee .02 + expected conditional projection KL)/8 for changes. Independently minimize row cost (gamma=0), or minimize sum of row costs + gamma times number of active layouts over the offered choices. For a candidate nonempty set S, rowwise minima within S yield the best assignment constrained to S. Enumerating all nonempty subsets and minimizing sum of row minima plus gamma*|S| is exact for this finite, one-step objective (gamma>=0): any assignment defines a feasible subset, and any subset's minimizing assignment uses at most that subset. Infeasible row/layout entries have infinite cost. Individual retain screening is invalid for this collective problem: a locally worse assignment may eliminate a layout. The solver is exponential in offered layout count (candidate library <=10 here), not proposed as a scalable general optimizer. Gamma .04 and .16 are illustrative prices, not measured hardware costs. Reconsider once per four steps; no DCF-10 override rule. No test-based choice between gamma values.

Numba compiled CPU kernel processes all rows, contexts and layouts in one call per step using layout-dependent offset, gather and parity tables. It updates only the affected clusters. Both grouped NumPy and compiled backends use identical padded FP64 state arrays, context mixing kernel, route candidate generation and projection code. Grouped backend calls frozen DCF-08 Kernel separately per active layout and context. Fusion comparison includes native compilation and implementation differences; does not isolate fusion alone. Every route has one shared padded allocation sized for largest candidate partition under cap, 32 context weight bytes + 8 route bytes + 32*max sum(2^cluster_size) belief bytes. Thus allocated persistent arrays are equal across arms within each condition (552 bytes at cap640; 680 or 1064 at cap1280); excludes working buffers, lookup tables, allocator, histories, model and JIT code. Active payload is smaller for some partitions. JIT compilation, kernel preparation and validation excluded from timing; one warm run and 3 randomized interleaved timed repeats; time includes filtering, schedule generation, projection and switching. Test 32 trajectories x64 steps per scenario (30% wrong context labels or entirely hidden), 24x40 validation fixtures, caps640/1280. Candidate selection based on validation only. Full same-information joint oracle supplies predictive KL. A wall-clock tool timeout was reported at the end of the last multi-system invocation, but all six complete per-system receipts were present and parsed; no missing condition. No paid compute.

## Primary results

Means across 24 conditions; timings are means of within-condition medians (ms), paired ratios are reported separately.

| Arm | Predictive KL | Mean time ms | Active layouts | Kernel calls |
|---|---:|---:|---:|---:|
| fixed grouped | .004155 | 72.865 | 1.000 | 256 |
| fixed compiled | .004155 | 2.384 | 1.000 | 64 |
| independent grouped | .003595 | 113.359 | 1.708 | 437.33 |
| independent compiled | .003595 | 8.758 | 1.708 | 64 |
| cohort .04 grouped | .003798 | 87.692 | 1.172 | 300 |
| cohort .04 compiled | .003798 | 6.463 | 1.172 | 64 |
| cohort .16 compiled | .003822 | 6.059 | 1.078 | 64 |

Matched grouped/compiled median paired speedups: fixed28.68x, independent13.92x, cohort .04 14.11x. 72 backend equivalence tests: identical paths, max prediction delta 1.110223e-15. Independent compiled improves KL vs fixed in 13/24 strictly, equals or improves in 21/24. Collective .04 and .16 respectively improve strictly in 5/24 and 4/24; equal or improve in 23/24 and 24/24. Median paired runtime ratios vs independent compiled .876/.843; faster 15/24 and 17/24. Mean KL rises ~5.6% and ~6.3% vs independent. Excess layout count above one drops 75.7% and 89.0%. The strict pre-run gate (KL <=1.10*independent and runtime <=fixed compiled) fails 0/24 for both gamma arms; quality condition alone 18/24 and 17/24. Median paired runtime vs fixed 2.274x and 2.297x. Thus neither compiled collective route earns a latency promotion. The fixed backend also benefits from compilation. Inference no longer fragments into layout-specific Python calls, but scheduling and projection remain substantial. Collective routing often defaults to fixed, particularly at 640 bytes, and cannot claim universal improvement. No real hardware layout prices, GPU/TPU, energy or language model measurements.

| Cap / scenario | fixed KL | independent KL | cohort .04 KL | cohort .16 KL |
|---|---:|---:|---:|---:|
|640 noisy30|.006135|.005887|.006149|.006135|
|640 hidden|.002531|.002531|.002531|.002531|
|1280 noisy30|.005814|.003940|.004500|.004608|
|1280 hidden|.002141|.002022|.002014|.002014|

Cohort assignment depends on other trajectories' observed past; predictions may change with batch composition. This is a deliberate collective policy, not a per-request invariant. Further independent checks: exhaustive assignment enumeration across 64 random small cost fixtures agrees exactly; 12 frozen DCF-09 fixed-filter replay fixtures max delta 7.77e-16; 12 future-observation flip checks preserve all predictions and routes through index8 (delta0). All 168 cells finite, cap-bound and repeat deterministic. Timing is a single CPU host, small batches only. No statistical generalization claim.

## Reproduce

Python3.13.5, NumPy2.3.5, Numba0.65.1, torch2.10.0+cpu (dependency import). In repository directory, run bash prepare.sh to copy the frozen DCF-10 sources and verify the pre-run hashes, then python router11.py and python audit11.py. In conversation archive the dependencies are already copied; run sha256sum -c PRE_RUN_SHA256. Raw receipts, logs, audit and frozen source copies retained in dcf11_reproducibility.zip (SHA256 65a67e1fd7481ad82190dcebb8f8c8d4c367c0aaad032c17a49f46d1c3b1df18). DCF-12 candidate: compiled projection and cost construction, bounded layout menus, batch-size sweeps with scaled gamma and measured overhead, plus comparison with fixed mixture/particle controls under matched working memory. Include per-request fairness and batch composition sensitivity. Advance only if matched compiled latency-quality gate passes.
