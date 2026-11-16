# NOR-20: wider coefficients and independent Linux/Windows CPU verification

**Protocol fixed 16 November 2026, before generating NOR-20 sources or measuring any NOR-20 benchmark.** Continues [NOR-19 #181](https://github.com/MoralityLabAI/Compactification-Theory/pull/181) and [MathCan NOR-19 #101](https://github.com/patrickdugan/MathCan/pull/101).

## Scope

NOR-19's streamed public verifier lowered peak traced Python allocations by ~52.5% against eager candidate caching while preserving a ~4.33× native CPU speedup. Its 64 cases came from the original NOR-18 population and were measured on Linux. NOR-20 tests wider input coefficients and a second CPU environment: GitHub Actions **Ubuntu 24.04 x86-64 and Windows Server 2025 x86-64**, both with CPython 3.12 from setup-python. Record precise runner details and the Python compiler. The study cannot claim an independent CPU microarchitecture unless the actual runner hardware is known.

The exact RRC25 grammar remains frozen: base B and U/V pairs selected from I, N, R, S; delays 64/80/96/112; spacings 24/32; amplitudes -2/-1/1/2. Each source has two degree-192 public demonstrations and a degree-256 public query INPUT. Its query OUTPUT and planted program identity are never placed in the fixture. Every policy evaluates the same sixteen public template candidates and exactly checks output-unanimity.

## Registered fresh 64-case source distribution

Four deterministic source cohorts, each with sixteen public cases: one for every combination of four coefficient-width tiers and four source families.

- **Nominal signed nonzero coefficient bit-widths:** 4, 8, 12, 16, with uniform integer magnitude 1 through (2^width−1) and independent sign. Sparse families also permit exact zero coefficients.
- **Families:** wide_dense has every nonconstant coefficient nonzero; wide_sparse has 20% nonzero nonconstant coefficients; wide_late has its first nonzero source coefficient at a uniformly sampled degree 57–78 and later 30% nonzero coefficients; wide_rational is the exact reciprocal of a short normalized polynomial with several signed wide coefficients in degrees ≤16. For wide_rational the actual generated coefficients may be much wider than the nominal seed width. Record actual input, output and inverse coefficient widths separately.
- **Seeds:** for replicate r=0..3, width index wi=0..3 and family index fi=0..3, the RNG seed is corpus.SEED+94120000+1000003*r+10007*wi+421*fi. Draw a six-field RRC25 program and fresh public inputs deterministically, then require that the native public header decoder returns exactly one candidate. Skip ambiguous headers and duplicate complete-public sources; the maximum is 500 draws per combination. Record accepted attempt index and public header. Fail on exhaustion.

All 64 are new source inputs, with sixteen per nominal bit tier and sixteen per family. Check public source hashes against the original NOR-18 288-case evidence and NOR-18R/NOR-19 benchmark fixtures. Source generation never reads prior benchmark outcomes. Produce the fixture once in a source-only CI job, upload it, and have the Linux and Windows jobs download **exact same fixture bytes**. Compare SHA-256 across jobs, with no network or secret labels inside verifier code.

## Fixed compared algorithms

The original NOR-19 five modes: independent16 native full template checks, cached16 eager two-demo full outputs, stream16_queryfirst, **primary stream16_verifyfirst**, and chunk4_verifyfirst. Every mode computes all sixteen candidate hypothetical public query outputs from scratch. The original native checker supplies the public exact acceptance-mask reference. All 64 masks and output-unanimity decisions must agree across modes and operating systems.

Per host: two warmups and seven measured repetitions per case/mode, deterministic shuffled mode order, garbage collection before each sample and disabled during each timed segment. Measure nanosecond query time, verification time, certificate time and total time; keep every timing sample. Report medians per source, paired native/stream CPU speedup and streamed/cached Python peak allocation ratio.

Measure complete and verifier-only peak Python temporary allocation in separate untimed tracemalloc passes per case/mode. Linux reports isolated process peak RSS when supported; Windows RSS may be unavailable and is separately labeled. Measure actual input/inverse coefficient bit lengths and exact observed reciprocal-recurrence operand bitwidth-product work outside timed code. Source generation and fixture decoding are excluded from timing.

## Preregistered cross-host criterion

A cross-environment replication succeeds only when **both OS runners** independently satisfy:

1. Parity: all 64 source masks and output-unanimity certificates match the original native exact checker, with source program identities and true query outputs absent.
2. Memory: median of 64 per-case streamed/eager-cached complete tracemalloc peak ratios ≤0.80; stream16_verifyfirst lower on at least 48/64 cases.
3. Runtime: median of 64 paired native/stream complete CPU time ratios ≥2.0×.
4. Width coverage: all four nominal input widths and all four source families appear in every cohort, with actual bit widths and per-family performance reported.

The test must report each machine's results, including any host-specific failure. Four-cohort blocked bootstrap with 4,096 seeded draws gives per-host percentile intervals for memory and runtime ratios. Those intervals condition on this source distribution and runner, and are not evidence of cross-microarchitecture guarantees.

This is an exact synthetic-program-family verifier experiment. Python tracemalloc excludes native allocator and process overhead; it cannot prove constant total RAM, theoretical bigint time, or GPU speedup.

## Reproduction

    python -m unittest research.nor20_crosshost.test_crosshost -v
    python research/nor20_crosshost/benchmark.py --make-fixture nor20_public_fixture.json
    python research/nor20_crosshost/benchmark.py --fixture nor20_public_fixture.json --check --out nor20_benchmark.json

The workflow runs source fixture creation and hashing before any benchmark job, then freezes original CPU/memory results and audits the complete original first-run records before merge.


## First executed cross-host outcome

The first preregistered cross-host campaign passed [GitHub Actions #43807479206](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43807479206). Its original **64-case public-only fixture**, Linux result, Windows result and cross-host comparison JSON were preserved byte-for-byte by a temporary write-capable publisher, then published in the source-controlled [nor20_original_crosshost.zip](results/nor20_original_crosshost.zip). The temporary publisher was deleted after its successful archival commit. The [frozen first-result receipt](results/nor20_first_frozen_receipt.json) contains the exact four SHA-256 hashes; the [independent auditor](receipt_audit.py) replays full native verification, all timing sample arithmetic and the reciprocal operand widths from the archive.

The original public fixture is **5,416,693 bytes**, SHA-256:

    de7ed801e0e8708afebdbe3699d9074375b156795b40372eb77e2a4a13218c19

Identical bytes and public case masks were provided to both OS jobs. None of the fixture's 64 sources overlaps the 288-case NOR-18 original public manifest. Each of the four independent RNG cohorts contributed one case from each of the sixteen nominal bit-width × source-family combinations.

### First Linux and Windows measurements

| CPU public verifier statistic | Ubuntu 24.04 x86-64 | Windows Server 2025 AMD64 |
| --- | ---: | ---: |
| Original independent sixteen-check median complete CPU time | 29.330 ms | 22.213 ms |
| Eager cached sixteen-check median complete CPU time | 5.778 ms | 4.625 ms |
| **Streaming verify-first median complete CPU time** | **5.570 ms** | **4.337 ms** |
| **Median paired native/streaming speedup** | **4.382×** | **4.331×** |
| Eager-cached median complete peak Python allocation | 282.02 KiB | 282.01 KiB |
| **Streaming median complete peak Python allocation** | **172.09 KiB** | **172.08 KiB** |
| **Median of paired stream/cache peak allocation ratios** | **0.55063** | **0.55061** |
| Number of individual source cases with lower streaming peak | 64/64 | 64/64 |
| All exact public acceptance masks and certificates equal | 64/64 | 64/64 |
| **Preregistered CPU and memory criterion** | **Pass** | **Pass** |

The 4,096-draw four-cohort-block bootstrap percentiles for the paired allocation ratio were **[0.54192, 0.56381]** on Ubuntu and **[0.54191, 0.56381]** on Windows. For the native-to-streaming complete runtime ratio, the conditional intervals were **[4.344, 4.445]** and **[4.285, 4.409]**, respectively. These are source-cohort resampling intervals, not confidence bounds across different CPUs.

The Linux runner used CPython 3.12.16 compiled with GCC 13.3.0, on Linux kernel 7.0.0-1017-azure. Windows used CPython 3.12.10 built with MSVC 19.43 and reported an AMD64 Family 26 Model 2 processor. Each runner reported four logical CPUs. This is an **operating-system/compiler comparison with different hosted environments**; the source-locked experiment cannot isolate whether differences in absolute time arise from the OS, compiler, CPU microarchitecture or scheduling.

### Wider coefficients and the memory frontier

The wide-rational sources reached a median per-case **maximum public input width of 2,193.5 bits**, even though the nominal seed polynomial coefficient widths were only 4–16 bits. That is exactly the failure mode for a simple source-bitwidth-to-intermediate-bitwidth estimate. The observed reciprocal transform on an already rational source may simplify again, so the inverse recurrence's maximum bit width needs to be recorded separately.

With 16-bit nominal width across all four families, the Linux median complete temporary allocation rose to about **426 KiB** for eager caching and **289 KiB** for streaming, with a paired ratio of **0.586**. In the four-bit stratum the corresponding medians were **200 KiB** and **107 KiB**, with a paired ratio of **0.511**. Thus absolute allocation grows with wider coefficients and the fractional reduction varies by tier, while the streaming implementation remained cheaper on **every** measured case. The large-rational family independently retained the same speed/memory direction.

The observed improvement under this new synthetic generator strengthens the evidence that coefficient streaming reduces Python allocation without sacrificing cached unary computation. It does not establish a hardware-independent guarantee, a constant-total-memory bound, or transfer to non-enumerated theorem-proving tasks.

### Original artifact and permanent replay

- Public fixture: **5,416,693 bytes**, SHA-256 de7ed801e0e8708afebdbe3699d9074375b156795b40372eb77e2a4a13218c19
- Linux complete benchmark: **683,486 bytes**, SHA-256 6c033440be72ae470769e7a152286677cba8fa3b676e9f9207d0c335ab1ec01d
- Windows complete benchmark: **683,275 bytes**, SHA-256 85686b13b4526d926f10b0b4f476bea82ab9d2f88dbc63d02704268f9ef88158
- Independent two-host comparison: **1,296 bytes**, SHA-256 3a048cfbafb7c9e87f28ed04d7d7e2ecd1b06b3be5c513351aa7fb11fa6a07c4

The permanent source CI first reads the checksum-pinned ZIP and checks **4,480 original nanosecond timing samples**, **1,280 complete and verifier-only tracemalloc records**, **640 mode-specific reciprocal integer workloads** and **16,384 deterministic bootstrap draws**, while recalculating all 128 native public masks. A second hosted source/Windows/Linux benchmark remains reproducible from the registered generator and fixture serialization. That rerun may have different raw times and Python memory allocations, which are reported separately from the original immutable first-run evidence.

The companion [MathCan NOR-20 PR #104](https://github.com/patrickdugan/MathCan/pull/104) proves an observed operand-bitwidth-product upper bound, width-aware packed candidate-array savings, and magnitude-independent exact integer streaming comparisons.
