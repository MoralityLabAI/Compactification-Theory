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


## First executed CPU result (original source locked)

[Original green hosted CPU and memory benchmark](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43797844269), artifact ID 14195445905, completed on 16 November 2026. Its public-only 64-source fixture and complete 7×4×64 timing records are **preserved byte-for-byte** in [results/nor18r_original_hosted.zip](results/nor18r_original_hosted.zip) and fingerprinted in [results/nor18r_first_frozen_receipt.json](results/nor18r_first_frozen_receipt.json).

The original JSON fingerprints are:

- Public-only NOR-18 source fixture: 1,306,733 bytes, SHA-256 \`5d75f2971d8b94daa9b65caae68e65921f2a90cd267481c7bb80a881e937d186\`.
- Complete first measured CPU benchmark: 580,813 bytes, SHA-256 \`59d48c8c9fe1bdf80ee2662d0c18f069f8414a92d48ad7980e3bb19b2a10ad98\`.
- Original NOR-18 288-case source record: SHA-256 \`9905aa23466b98ba2068e6daf9206d35c453b8b3b75bd27c3ab7ff8f0db4041b\`.

The [independent archived-receipt auditor](receipt_audit.py) regenerates exact acceptance masks and candidate query predictions from the fixture's public data using the original corpus interpreter, checks 1,792 original nanosecond timing samples, all 256 separately measured peak allocation records, per-case inverse-operand bit sizes, and the deterministic 4,096-draw cohort-block bootstrap. It imports no benchmarking implementation. The separate native-parity test verifies the four original public algorithms and future-label isolation, while the full hosted benchmark is reproducible without changing frozen source selection.

### Hosted platform and CPU measurements

Measured on GitHub Actions Linux \`7.0.0-1019-azure\`, x86-64, CPython 3.12.16 compiled with GCC 15.2.0, reporting four logical CPUs. Garbage collection was collected immediately before each timed run and disabled during the measured segment. Each case/mode was warmed twice, timed seven times, and mode order was deterministically shuffled. Source generation and JSON loading were excluded from timings.

| Public checker, including identical public query simulations | Median per-case complete CPU time | Median public-verification time | Median separately traced peak temporary allocation |
| --- | ---: | ---: | ---: |
| Native 16 independent checks | 23.913 ms | 21.291 ms | 74.1 KiB |
| **Four-unary cached sixteen-candidate batch** | **4.327 ms** | **2.266 ms** | 128.2 KiB |
| Degree-160 prefix cache + independent survivor checks | 12.331 ms | 9.706 ms | 112.8 KiB |
| Degree-176 prefix cache + independent survivor checks | 8.836 ms | 5.744 ms | 120.6 KiB |

The **median of the 64 paired complete-time speed ratios** was **4.32878×**, with cached full-batch execution faster in **64/64** cases. The preregistered speed criterion (median paired speedup at least 1.5× and cached faster on at least 48 of 64 public inputs) **passed**. The four-cohort block-bootstrap 95% interval was **4.280× to 4.396×**. This interval resamples the same four fixed generator cohorts and is conditional on this first hosted execution. It is not a cross-machine confidence interval or an independent-replication result.

Per-family median paired speedups were 4.396× for late_source (32 cases), 4.323× for signed (16) and 4.204× for rational (16). Every mode exactly reproduced every publicly accepted template and every logical output-unanimity certificate on all sixty-four benchmark sources.

### Exact integer operand workload and memory qualification

The independent RRC25 reciprocal algorithm executes a quadratic number of Python bigint products per inversion. The observed multiplication counts and operand-bitwidth products, including the once-per-mode public query simulation, were:

| Method | Observed R integer products across 64 cases | Sum of positive operand bitwidth products |
| --- | ---: | ---: |
| Native sixteen checks | 31,750,144 | 845,306,615 |
| Cached sixteen-candidate batch | **4,476,928** | **163,683,594** |
| Prefix160 plus native survivors | 18,724,608 | 227,918,645 |
| Prefix176 plus native survivors | 16,549,888 | 215,773,495 |

The second column is an **observed operand-size proxy**, computed from the exact integers passed into the original reciprocal recurrence. It measures multiplication count and operand bit widths, not the actual dynamic native integer multiplication instruction count or wall-clock cost of each operation. The call-multiplicity identity is kernel-proved in [MathCan NOR-18R PR #98](https://github.com/patrickdugan/MathCan/pull/98): eight native R inversions per public demonstration from U/V, plus sixteen more when the fixed base is R, versus one cached R inversion.

Caching uses **more temporary Python allocation** in these source-locked tests, with median independently measured \`tracemalloc\` peaks of 128.2 KiB versus 74.1 KiB for the native repeated checker. Isolated-worker maximum RSS was 29,364 KiB for every mode, with no measurable rise over that worker's fixture-loaded baseline. That process-level peak does not resolve the temporary Python allocation difference; it also excludes the cost of loading source files on a fresh machine.

NOR-18's arithmetic proxy predicted a cached-work benefit. NOR-18R establishes that the specifically tested CPython implementation also runs faster on this original held-out source subset, while revealing an allocation tradeoff. Generalization to a fresh source generator, other Python versions, different CPU microarchitectures, GPU implementations or production proof tasks remains to be tested.

### Independently reproduce the frozen evidence

\`\`\`sh
python research/nor18r_runtime/receipt_audit.py --check \
  --archive research/nor18r_runtime/results/nor18r_original_hosted.zip
\`\`\`

The complete archived fixture deliberately excludes all withheld query-output labels and planted program identities. It is suitable for independent public-only verifier microbenchmarks without unsealing NOR-18's future truth.
