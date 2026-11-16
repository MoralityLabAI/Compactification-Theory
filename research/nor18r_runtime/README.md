# NOR-18R: executable CPU, bigint, and memory audit of cached exact verification

**Preregistered 16 November 2026 before opening measured NOR-18R CPU timings.** Follows merged [NOR-18 #175](https://github.com/MoralityLabAI/Compactification-Theory/pull/175) and [MathCan cached-verifier work theorem #94](https://github.com/patrickdugan/MathCan/pull/94).

NOR-18 measured **coefficient-operation work**, finding that one cached degree-192 sixteen-template public screen gave the entire certifiable 48/192 late-source subset at an eight-unit proxy cap. The mathematical work saving from sharing four unary public transforms is exact for that declared proxy. NOR-18R measures real Python 3.12 CPU execution, time per checked public case, Python-traced peak temporary allocation, isolated-process maximum resident memory and the actual integer operand sizes encountered by the reciprocal (\`R\`) recurrence.

## Frozen test sources and execution

Use the **original NOR-18 held-out 288-case record**, preserved at \`research/nor18_costed_prefix/results/nor18_full.zip\`. Require the original uncompressed 12,896,423-byte JSON SHA-256:

\`9905aa23466b98ba2068e6daf9206d35c453b8b3b75bd27c3ab7ff8f0db4041b\`.

Regenerate the exact public source cases from NOR-18's frozen generator and seeds. Verify **every** reconstructed public source hash against its corresponding archived first-run record before selecting the benchmark subset. In each of four cohorts, take the first eight late_source, first four signed and first four rational entries in the original cohort order. This fixes **64 public-distinct benchmark sources**, with 32 late_source, 16 signed and 16 rational cases. Store only degree-192 public input/output demonstrations, degree-256 *public query input*, the publicly derived header, source family and archival source digest. The planted program and withheld future query output remain excluded from every benchmark fixture and timed path.

All measured algorithms use the *identical* two degree-192 public demonstrations and query input and must produce the **same 16-bit exact acceptance mask and the same unanimous-future-output certificate**. Recovered masks must also match the original frozen NOR-18 full-mask receipts; each public simulated future must agree with the reference RRC25 exact interpreter in standalone parity tests.

## Preregistered executable comparisons

1. **\`independent16\`:** Call the original \`compound25.verify_template\` sixteen times, using native \`execute\` and recalculating its base, U and V unary transformations for each candidate.
2. **\`cached16\`:** Call the existing \`nor17.candidate_outputs\` once per demonstration at degree 192, reusing the four unary transforms across all sixteen templates, then compare each candidate output with the observed public demonstration.
3. **\`prefix160\`:** Construct the two cached degree-160 source-prefix screens, intersect the survivors, and call the *same native* full exact \`verify_template\` on each surviving candidate.
4. **\`prefix176\`:** Repeat with a degree-176 screen and exactly the same downstream native full checks.

Every algorithm also independently computes all sixteen hypothetical degree-256 outputs from the known **public query input**. The benchmark reports **query simulation time and public verification time separately**, as well as combined execution time, so comparisons of complete certificates include shared query computation costs. It reads neither future query outputs nor private program identities. Query simulations for every mode are recomputed and metered within that mode, with no cross-mode cache credit.

For each of the 64 cases, run two warmups and **seven measured repetitions per mode**. Randomize the order of the four modes deterministically inside each repetition, so CPython allocator warming and CPU frequency drift do not always favor a particular algorithm. Before each sample, perform \`gc.collect()\`; temporarily disable garbage collection during the timed segment. Report per-case median query, verification and combined \`perf_counter_ns\` times; paired speedup \`median(independent16)/median(cached16)\` and a 95% **four-cohort block bootstrap** confidence interval with 4096 deterministic seeded draws. Give per-family medians and the fraction of source cases where cached execution is faster. CPU architecture, Python/compiler version and host information must accompany every source receipt.

Measure peak Python-traced temporary allocation **in an untimed separate run per case and algorithm** using \`tracemalloc\` starting before the algorithm and stopping afterward. Also launch isolated subprocesses per mode, reading their process maximum resident set size via \`resource.getrusage(RUSAGE_SELF).ru_maxrss\` on Linux; report pre-run baseline RSS, peak RSS, and peak-minus-baseline, with clear caveats that allocator reservations and process startup can make the RSS increments zero. These memory metrics are separate from timings, and neither is a hardware-independent proof.

## Integer costs and safety interpretation

Replay one exact reciprocal formal-series inversion for each degree-192 demonstration input and the degree-256 public query input. Verify the replay against the exact original \`graph11.inv\` result. Record the **actual number of Python big-integer multiplications and the sum of the products of the operands' positive bit lengths** in that inverse recurrence. Scale this observed per-source integer workload by the *exact* reciprocal invocation counts implied by each algorithm's public base, U and V calls (and cache reuse) to provide a per-method bigint operand-width ledger. Record maximum intermediate and output coefficient bit lengths and maximum Python integer storage size. This bit-length product sum is an **operand-size proxy**, distinct from native CPU instructions, Python multiplications per second and asymptotic multiplication bit complexity.

The prior NOR-18 coefficients-only prediction is that cached sixteen-template verification does fewer repeated reciprocal-series operations and may be faster in real CPU time. **Primary registered performance criterion:** cached16 has the same exact masks/certificates for every source, a paired median complete-public-certificate speedup of **at least 1.5×**, and is faster on **at least 48 of 64 cases**. Report those components independently. All other comparisons, including prefix160/prefix176, memory, process RSS and family-specific effects, are exploratory. This is measured performance on a pinned GitHub Actions CPU runner, not a claim about 5080/3050 GPUs, other machines or mathematical generalization.

## Reproduce

\`\`\`sh
python -m unittest research.nor18r_runtime.test_runtime -v
python research/nor18r_runtime/benchmark.py --check \
  --make-fixture /tmp/nor18r_fixture.json
python research/nor18r_runtime/benchmark.py --check \
  --fixture /tmp/nor18r_fixture.json \
  --repetitions 7 --warmups 2 --out /tmp/nor18r_full.json
\`\`\`

The hosted workflow separately verifies the source manifest, original public-mask parity, all 64×4×7 timing samples, memory ledgers and test-source integrity. Full original benchmark evidence is frozen by SHA-256 after the first successful run. Unexpected errors, parity failures or insufficient measurements block merge.
