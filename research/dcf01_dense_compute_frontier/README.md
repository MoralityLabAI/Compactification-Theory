# DCF-01: Dense compute frontier and recurrent compactification

Date: 2026-11-25. Status: conditional theory, executed synthetic CPU pilot, prospective model experiment.

## Premise and source boundary

OpenAI's November 24 manuscript claims the arithmetic exponent of general square matrix multiplication equals 9/4 over every field. This note uses the October upper bound and November equality as conditional premises, without auditing the lower-bound proof or claiming independent verification. The arithmetic exponent is an infimum over asymptotic algorithms, not a pointwise wall-clock lower bound; no practical crossover or floating-point stability follows from it.

Sources:
- https://github.com/openai/math/blob/main/history.md
- https://github.com/openai/math/tree/main/preprints/The-matrix-multiplication-exponent-is-nine-fourths-November-24-2026
- https://github.com/openai/math/tree/main/preprints/Matrix-Multiplication-Nine-Fourths-October-2-2026
- https://eccc.weizmann.ac.il/report/2026/281/ (Shpilka, constructive upper bound exposition)

## Batch-dependent upper bound (derived)

For X of shape B by d and W of shape d by d, let B <= d. Pad d to a multiple of B, split X into B-by-B blocks, and split W into corresponding blocks. There are O((d/B)^2) square block products and O(d^2) addition work. Given an O_eps(n^(9/4+eps)) square algorithm, total operations are O_eps(d^2 B^(1/4+eps)); per token O_eps(d^2 B^(-3/4+eps)). If B >= d, group into d-row blocks: per token O_eps(d^(5/4+eps)). This is an upper bound, not an optimal rectangular exponent or a fixed-weight lower bound. B=1 recovers ordinary quadratic matrix-vector cost. If B scales as d^a, 0<=a<=1, the ideal upper-bound per-token width exponent is 2-3a/4 (up to eps).

For L recurrent passes, multiply the work by L, conditional on sequential execution and unchanged shapes. At fixed B, halving d allows 4x passes at the same leading cost; with square batches co-scaling with d and comparing equal token counts, it allows 2^(5/4) passes per token. The 2^(9/4) factor compares total work for one square batch with fewer tokens, so it cannot serve as an equal-token budget.

An exact rank-r factor W=UV permits classical work ~2Bdr multiply-accumulates and storage 2dr scalars. A dense conventional layer uses Bd^2 MACs and d^2 scalars. Under equal normalized constants, the factorized classical arm costs less than the hypothetical tiled dense upper-bound proxy for r < d/(2 B^(3/4)); this is a proxy crossover, not a hardware prediction. At r=d/16 it is B=16. Real constants, rectangular algorithms, rank requirements, precision and kernel fusion can shift this entirely.

## Economic model

Let P_eff(B,d,precision) denote measured effective arithmetic throughput and H measured memory bandwidth. A roofline lower estimate for one pass is max(arithmetic_work/P_eff, external_bytes/H); add non-overlapped launches, communication and scheduling where measured. For an uncached weight matrix external weight bytes per token are q*d^2/B, q bytes per scalar, plus activation and KV traffic. For rank r, replace d^2 by 2dr. Tied recurrence reduces unique weight storage, while weights may be streamed on each pass if they cannot remain resident. Charge all passes and routing/halting, all activation/KV transfers and time in the batch queue. Distinguish prefill and decode; training also charges gradient and optimizer costs.

Batching can help any accelerator or inference provider. A Google TPU-specific advantage needs measured costs and service quality relative to GPU/other accelerators; no vendor moat or guaranteed inference economy follows from the algebraic exponent alone. The strongest economic metric is joules or dollars per correctly completed task at matched accuracy and latency SLO, including retries, queueing, model loading and depreciation where available.

## Executed pilot

Run from repository root:
    python research/dcf01_dense_compute_frontier/benchmark.py > cpu_receipt.json

Sandbox CPU, x86_64, Python 3.13.5, NumPy 2.3.5; BLAS thread environment set to 1 before importing NumPy. Seeds 11, 23, 47; widths 256, 512, 1024; batches 1,16,128; loops 1,4; 21 interleaved timed samples per arm after warmup. FP32 random factors rank d/16; compare tanh(z @ W) with tanh((z @ U) @ V), W=U@V precomputed outside timing. Both execute the same nonlinear recurrence in real arithmetic; roundoff permitted. Weight generation, training and model loading excluded. Ordinary NumPy kernels only.

54/54 configurations passed relative L2 < 1e-4; worst observed 1.081197e-6. 54/54 had median speedup >1. Overall median of row-wise speedups: 3.667826x. Both unique weight storage and classical MACs decrease exactly 8x in this constructed arm. Negative control: identity has best rank-d/16 relative Frobenius approximation error sqrt(15/16) ~= 0.968246 by Eckart-Young. This is an analytic control, not an empirically trained arm.

| B | loops | median speedup | min | max |
|---:|---:|---:|---:|---:|
|1|1|3.306|1.420|4.373|
|1|4|3.109|1.434|7.373|
|16|1|4.350|2.955|5.213|
|16|4|4.434|3.618|6.349|
|128|1|3.174|2.654|4.069|
|128|4|3.302|2.897|21.696|

Timing is a finite single-host pilot, not a fitted exponent or a transferable hardware estimate. No trained model, energy data, GPU or TPU measurement. The raw local receipt was retained in the conversation artifact, while this repository records aggregates and executable reproduction.

## DCF-02: prospective quality-matched accelerator experiment

Run a dense teacher, tied dense student, tied factorized student, untied matched parameter/FLOP controls and a random-feature/control router on the same held-out reasoning tasks. Start with exact algorithmic composition (modular arithmetic, permutation composition, graph reachability) and an existing NOR/AEG task slice. Freeze splits, seed families and training budget before tuning. Evaluate unseen lengths and unseen task families, and report exact verifier success as well as calibration. Rank ratios 1/2,1/4,1/8,1/16; depth 1,2,4,8,16; widths 128,256,512,1024 where feasible. Separate inference-only factorization from retrained factorized student. Record actual parameter counts, weight bytes, pass counts, realized MACs and measured latency. Use identical data, optimizer budgets and decoding policy; report all seeds and failed arms.

Register three or more seeds and a held-out paired task bootstrap, with seed-level sensitivity. First gate: exact recurrence equivalence relative error <1e-4 for synthetic fixtures. Second: quality noninferiority with a predeclared 2 percentage point margin and one-sided 95% paired confidence bound on the held-out accuracy difference; power the sample before opening held-out data. Third: at least 20% lower joules per correct task and median latency within 10% of the matched baseline under a common batch and precision, plus p95 latency and peak VRAM reported. Failure or underpowered results remain recorded. Treat these thresholds as proposed, subject to registration and hardware access.

Target RTX 5080 locally, then same precision on a second accelerator only if available; no paid compute required for DCF-01. Measure warm and cold residency separately, B=1,4,16,64,128 where supported; torch CUDA events with synchronization, warmup and randomized paired runs; report device, runtime, driver, dtype, clocks/power method, and sample distributions. Add energy via supported telemetry with idle baseline and limitations. Compare throughput at matched p95 latency, with concurrent requests for the service case. Export training recipe, checkpoint provenance, hardware receipts, negative outcomes and verifier metadata to Data Goblin when models actually exist.

## Interpretation

The pilot is a constructive instance of cheaper execution for a structured operator. It establishes neither generic low-rank representability nor an end-to-end reasoning advantage. The conditional asymptotic batch bound and the independently runnable finite pilot are separate results.
