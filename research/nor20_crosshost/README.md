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
