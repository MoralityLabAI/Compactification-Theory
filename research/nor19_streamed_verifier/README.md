# NOR-19: bounded candidate-output buffers in the cached exact verifier

**Preregistered 16 November 2026 before any NOR-19 CPU or memory measurement.** Extends [NOR-18R #178](https://github.com/MoralityLabAI/Compactification-Theory/pull/178) and [MathCan reciprocal reuse #98](https://github.com/patrickdugan/MathCan/pull/98).

The NOR-18R verifier reuses the four unary public-series transforms per degree-192 demonstration, achieving a measured **4.33× median paired complete CPU speedup** over sixteen independent checks. Its Python-traced peak temporary allocation was larger than the native checker (128.2 versus 74.1 KiB on the original 64 sources), partly because the cache materialized all sixteen complete candidate outputs for both demonstrations while their public query predictions were also live. NOR-19 tests reducing the number of materialized public demonstration candidate-output arrays while retaining the same exact public acceptance masks and certified future outputs.

## Source lock and disjoint confirmation fixture

The exact original NOR-18 288-case public source stream remains SHA-256 \`9905aa23466b98ba2068e6daf9206d35c453b8b3b75bd27c3ab7ff8f0db4041b\`; its RRC25 trained GRU checksum is \`54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b\`.

Regenerate the complete 288-case NOR-18 roster and independently verify **every** regenerated public source digest against the original archived JSON, before extracting any timing fixture. Choose the next 64 source entries, which NOR-18R's [first timing fixture](https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/nor18r_runtime/results/nor18r_original_hosted.zip) did **not** time: for each of four source cohorts, late_source cases at positions 8–15, signed at 52–55 and rational at 64–67. This provides **32 late_source + 16 signed + 16 rational**, all distinct complete-public inputs, with no overlap with NOR-18R's original 64-case benchmark. The sources originate from the same frozen NOR-18 generator and source split; confirmation is across **previously unbenchmarked cases**, not across a new generator distribution.

The new public fixture stores only two degree-192 input/output demonstration pairs, the degree-256 public query INPUT, the public header, family, original source index, and the frozen exact acceptance-mask and output-unanimity receipts. **No planted program identities or withheld query OUTPUTS are serialized.** All five verifiers must match the original sixteen-bit acceptance mask and output-unanimity certificate for every source. An independent audit must rederive these masks using the original exact RRC25 checker.

## Registered implementations

All implementations perform one complete **sixteen-candidate public query-input simulation** and account for its time, then check all sixteen exact degree-192 public demonstration templates. They differ in how those template outputs are materialized and in the ordering of query/verification work:

| Implementation | Materialized demonstration outputs | Peak design bound | Query ordering |
| --- | --- | --- | --- |
| \`independent16\` | Native exact checker, one candidate per call | Native control | Query first |
| \`cached16\` | Original NOR-18R cached two-demo sixteen-output batch | 32 candidate-output vectors | Query first |
| \`stream16_queryfirst\` | Four unary transforms once per demo; compare each candidate coefficient by coefficient, abort on first mismatch | **Zero** complete candidate-output vectors | Query first |
| \`stream16_verifyfirst\` | Same coefficient-streamed exact checker, process demo 1 then demo 2 and release each workspace | **Zero** complete candidate-output vectors | Verify first |
| \`chunk4_verifyfirst\` | Four shared unary transforms per demo, materialize at most four complete candidate outputs at a time, discard after comparing | **Four** candidate-output vectors | Verify first |

Every cached/streamed implementation evaluates four unary transforms on each of the two demonstrations, including one reciprocal transform R, retaining the full exact integer recurrence semantics. Query simulation is recomputed per algorithm and never reused across modes. The abstract materialized-output count is an **algorithmic workspace model**, not a measured count of Python allocation or total live series (which also include the four unary transforms and query predictions). The pointwise streaming predicate must be provably equal to full finite-series equality. Each implementation's public mask/certificate must match all four other implementations and the original NOR-18 reference.

## Timing and memory protocol

Run on a hosted Linux x86-64 CPython 3.12 CPU. Per public source and algorithm: **two warmups and seven measured repetitions**, each mode presented in a deterministically shuffled order per repetition. Collect garbage before each timing, disable the collector within the timed segment and measure with \`perf_counter_ns\`. Time the query simulation, exact public verification, certificate selection and combined operation, recording the execution order for the verify-first methods.

Measure peak temporary Python allocation using \`tracemalloc\` **in a separate untimed complete execution** for each source/algorithm, starting after fixture loading and stopping afterward. Also measure Python-traced peak for each verifier-only phase separately and record isolated worker \`ru_maxrss\` for all five algorithms. A zero increase over the process baseline on the small fixture is interpreted as a measurement-resolution limit, not evidence of identical temporary allocations.

Quantify total R reciprocal invocation counts from the frozen four-op grammar: one query inversion common to all methods, one R per public demonstration for each cached/streamed/checker variant, and 8+16 if the base operator is R for every native independent demonstration. Compare the exact big-integer operand-width products recorded by the prior NOR-18R audit on the public sources.

## Primary success rule and controls

The **predeclared primary implementation** is \`stream16_verifyfirst\`. It succeeds on this fixed source benchmark only if:

1. Every public acceptance mask and certificate exactly matches the original native checker and the original source-locked NOR-18 JSON for all 64 cases.
2. The **median of per-case ratios of untimed Python \`tracemalloc\` complete-operation peaks**, streamed / cached, is at most **0.80**, and streamed has a lower peak on at least **48/64** cases.
3. The median of paired complete-operation runtime ratios, independent16 / streamed, is at least **2.0×**, so the new memory-bounded checker retains substantial measured CPU acceleration over the original repeated-check implementation.

Report actual means/medians, the paired runtime and peak-allocation ratios, per-family results, and a deterministic 4096-draw four-source-cohort block-bootstrap percentile interval for each ratio. The chunked implementation and query-first streaming control are secondary, and any hindsight-best mode may be described only as exploratory. A failure on any primary threshold is reported as a failed preregistered result even if another variant performs better.

This test measures one CPython/GitHub-hosted CPU environment. Peak allocation may vary with Python's allocator, source coefficient bit widths, and generated source family. The result does not establish a zero-memory verifier, constant total series memory, GPU improvement, or production theorem-prover safety.

## Reproduction

\`\`\`sh
python -m unittest research.nor19_streamed_verifier.test_streamed -v
python research/nor19_streamed_verifier/benchmark.py \
  --make-fixture /tmp/nor19_public_fixture.json --check
python research/nor19_streamed_verifier/benchmark.py \
  --fixture /tmp/nor19_public_fixture.json \
  --repetitions 7 --warmups 2 --check --out /tmp/nor19_full.json
\`\`\`

The first completed CI will freeze the original full benchmark and public-only source fixture SHA-256 checksums. Each new result must pass an independent source/semantic, timing and allocation audit before merging.
