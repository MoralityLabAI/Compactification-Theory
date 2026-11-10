# RRC-22: real CPU latency, memory and integer-width audit of selective symbolic deferral

**Executed November 10, 2026, one CPU host; observed RRC-21 evaluation roster reused.** This is a systems remeasurement of the *already exposed* 480 positive cases plus 16 known-negative and 12 future-collision controls. It trains no network, selects no hyperparameters, and preserves the RRC-21 calibration and three prior model checkpoints. The selected RRC-21 policy is **selective full** (OOD/entropy deferral with up to 33 certified candidate checks). The comparison is the **complete certified class search** on every case, with no neural model invocation.

## Actual measured CPU execution

Environment: Linux x86_64, Python 3.13.5, AMD EPYC 9V74 80-Core Processor (5 CPUs available to the container). The measurement is portable standard-library Python, with ordinary Python implementations of the model's GRU/MLP matrix operations and of the integer interpreter. Three measured, deterministically shuffled passes followed a 12-case warmup per arm. Each arm saw all 508 cases in each pass, without concurrent experiment processes; setup and classifier compilation are separately timed. The reference measurements were collected in three sequential arm batches on this host.

| Policy | Correct positive futures /480 | Median warm wall ms/case | P90 warm wall ms/case | Total wall ms, 3×508 | Cold setup ms | Peak process RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| Complete certified search | **480** | **0.303** | **0.705** | **573.992** | 426.492 | 108,432 |
| Selective full deferral | 457 | 1.140 | 7.777 | 5,582.573 | 644.982 | 108,432 |
| OOD-only full deferral | 450 | 1.129 | 7.392 | 5,351.472 | 649.256 | 104,844 |
| Selective strict-12 | 324 | 0.936 | 7.992 | 5,681.960 | 616.920 | 109,584 |
| Neural top-12 on all cases | 374 | 6.727 | 7.232 | 10,399.013 | 663.212 | 109,584 |

**Decisive observation:** on this CPU and degree-12 graph grammar, complete symbolic search is about **9.7× faster** in aggregate than selective full deferral despite requiring more exact graph candidates. Of selective full's 5,582.573 measured milliseconds, 5,266.823 are attributed to detection, inference and ordering; only 315.750 milliseconds are spent executing its exact verifier. For full search the corresponding figures are 232.257 milliseconds building the public class order and 341.735 milliseconds executing exact candidates. The selective policy saves 25.985 milliseconds of measured exact verification over all three passes while adding more than five seconds of routing/inference work. Thus the saved integer arithmetic fails to amortize the routing cost.

**Memory interpretation:** each peak RSS above is a *fresh worker process* measured via Linux `ru_maxrss`. The shared exact polynomial class compiler dominates the peak, making complete and selective-full both approximately **105.9 MiB** (108,432 KiB). Post-setup resident memory is 103,304 KiB for full symbolic and 104,584 KiB for selective full; the latter keeps JSON-deserialized model weights and calibration data. The separate Python allocation peak (`tracemalloc`) is sampled on 64 stratified cases, while process RSS is measured across all 508. Neither number is a GPU/VRAM estimate. The cold setup figures include roster generation and the certified class map, and for selective include frozen checkpoints/calibration loading. Historical *training* and construction of the calibration are separate costs.

## Actual integer operands and graph work

Instrumented re-execution counts each exact two-demonstration candidate on the 480 positive cases, with independently verified arithmetic parity across all 256 graph/topology combinations. It records every multiplication and addition, the maximum observed intermediate integer width and operand bit-length proxies. Instrumentation is excluded from the wall-timed runs.

| Policy | Exact graph queries | Integer multiplies | Integer adds | Sum of multiplication operand-bit products | Maximum intermediate bits |
|---|---:|---:|---:|---:|---:|
| Complete certified search | 4,859 | 796,556 | 811,508 | 5,757,272 | 101 |
| Selective full deferral | 4,252 | 700,652 | 718,604 | 5,596,056 | 101 |
| OOD-only full deferral | 3,974 | 672,800 | 686,000 | 5,512,684 | 101 |
| Selective strict-12 | 3,266 | 471,466 | 480,274 | 3,266,264 | 101 |
| Neural top-12 | 2,796 | 549,740 | 560,060 | 7,989,249 | 101 |

The bit-product sum means `Σ bit_length(a) × bit_length(b)` over observed multiplication operands, including zero-valued operands. It is a **bit-work proxy**, not actual CPython instruction counts or an estimate of asymptotically optimal integer multiplication. The addition proxy similarly sums maximum operand widths and is stored in the JSON receipt. Since candidate programs vary in their reciprocal and convolution structure, fewer graph queries need not imply a proportionate reduction in integer work: the selective full policy cuts the graph-query count by **12.5%** but the observed bit-product proxy by only **2.8%**. The maximum intermediate coefficient in the metered run has 101 bits, so Python's compact exact-integer operations remain cheap relative to repeated interpreted neural inference.

## Verification and costs beyond the timed path

- **Exact scores:** complete search 480/480 positive futures; selective full 457/480; neural top-12 374/480; strict selective top-12 324/480; OOD-only full 450/480. Every result matches the RRC-21 source-bound receipt.
- **Selective deferral:** 311/480 positive cases deferred, of which 280 follow the OOD fast path. Only 200 cases invoke the frozen models; complete search invokes none. The OOD path performs a nearest-reference comparison against 420 known-only reference vectors of width 10. This is real CPU routing work.
- **Queries and exactness:** each paid graph candidate executes two separate degree-12 demonstration series. The independent future grade is performed only *outside* the timed policy path. Six paired future mutations still give the same candidate traces and disclose finite-prefix uncertainty.
- **Neural cost:** selective full invokes two 6,240-parameter GRUs and one 6,235-parameter MLP on only in-domain cases, with the RRC-21 estimated **112,552 dense MACs** per such case, excluding nonlinearities and Python dispatch. The three existing models and their original training costs were paid in earlier experiments. Full certified search uses neither neural weights nor neural forward MACs.
- **Compiler cost:** the class map is generated on setup from exact degree-12 coefficient polynomials. It performs 256 graph normalizations and consumes the largest transient memory spike. This shared cost is separately recorded; amortization across requests changes the economics of both methods.

## Reproduction and limitations

The source-locked [`rrc22_deterministic.json`](results/rrc22_deterministic.json) compares all semantic outputs and per-arm traces against RRC-21, including measured integer bit widths. It is byte-identically reproducible by standard-library Python. [`rrc22_reference_profile.json`](results/rrc22_reference_profile.json) records machine-specific `perf_counter_ns`, `process_time_ns`, cold setup and isolated peak memory measurements. Its timing values are observations, **deliberately excluded from byte-for-byte CI thresholds**. Run the profiler on a different CPU or native optimized model runtime to establish that machine's result.

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_meter22.py' -v
python research/rrc_compactification/bench22.py --check
python research/rrc_compactification/bench22.py --smoke
python research/rrc_compactification/bench22.py --profile --rounds 3 --out /tmp/rrc22_machine_profile.json
```

The three profiling batches are stored as provenance in the reference profile; exact latencies vary with scheduler noise and runtime version. This is a systems test of a known RRC-21 roster at a *short fixed horizon*, with a Python-model inference implementation and a bounded supplied graph interpreter. It does not measure compiled ONNX/TensorRT models, GPU execution, longer generating functions, end-to-end external-API costs or training amortization. Exact degree-12 certificate acceptance is a finite equality check, not a proof of equal infinite series.

**Decision:** On this workload the complete finite symbolic search should be the deployed CPU default. A neural selective policy becomes interesting again only when the interpreter grows materially more expensive, or model inference is compiled and sufficiently cheap to amortize its detection and routing costs. RRC-23 should test a wider grammar/greater degree and include a measured break-even analysis before another model-training campaign.
