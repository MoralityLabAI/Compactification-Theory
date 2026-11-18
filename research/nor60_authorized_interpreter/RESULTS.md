# NOR60 first result: exact authorized execution with support compression and local unary reuse

**The first registered run passed all five engineering gates.** Across the
60-source workload, the sum of complete signed-verifier source medians fell
from **8,286.439526 ms** to **4,673.229136 ms** when the optimized interpreter
used NOR59's warm indexed Merkle template. The ratio was
**0.5639610500187702**, a **43.603895% reduction** for this fixed workload.
Every original output vector and full indexed Merkle root agreed. The
independent archive audit replayed all 60 original interpreter outputs and
authenticated all 300 signed bundles. [1], [2], [3]

This result closes the NOR execution and verification engineering sequence at
its measured boundary. The reusable implementation is the fixed optional
NOR45 policy `_interpreter_policy="support_memoized"`; the original policy
remains the default. The [Authorized Exact Gym](../authorized_exact_gym/README.md)
provides public development tasks from the resulting signed-execution
contract. This report proposes no further numbered NOR experiment.

## Registered panel and five gates

The panel contains 2 observed degrees, 512 and 1024, crossed with source
strides 1/4/8, five fixed authorized program profiles, and two sign-family
replicates. Full vector lengths are 769 and 1537. All 60 public inputs were
uploaded before reference computation and CPU measurement. Their exact
vectors are distinct from all 588 original NOR51-NOR59 vectors, including the
retained NOR57 failure. A content hash that excludes experiment metadata
establishes this exclusion separately from schema-bearing source IDs. [1], [3], [4]

| Registered gate | Frozen requirement | First observed result | Outcome |
|---|---|---|---|
| Sparse interpreter improvement | Median sourcewise optimized-interpreter/full-root ratio strictly below 0.70 on all 40 sparse cases | **0.20419599493475604** | PASS |
| Complete workload improvement | Combined-interpreter/warm-root ratio of sums of source medians strictly below 0.80 on all 60 cases | **0.5639610500187702** | PASS |
| P4 repeated-R improvement | Median sourcewise optimized-interpreter/full-root ratio strictly below 0.65 on all 12 P4 cases | **0.11972620591979487** | PASS |
| Exact signed execution and security | Complete vector/root parity on all 60 cases, checked support and passing source-free controls | **60/60**, with **43/43** preflight tests passing | PASS |
| First-source and evidence integrity | Original 60 inputs, 588 exclusions, five samples per arm, five signed sessions per source, unchanged measured code and retained first evidence | **1,200** primary samples, **300** bundles, **1,324** journal events | PASS |

Each source has five samples per arm. A source's median is computed within
its arm. The sparse and P4 gates then take the median of paired sourcewise
ratios. The workload gate divides the sum of combined-arm source medians by
the sum of baseline source medians. These two estimands weight the workload
differently. Their definitions and thresholds were committed before the
source panel existed. [2], [4]

## Four-arm result

Every arm verifies the same original program-authority certificate,
verifier-signed precommit and provider-signed root within a repetition. All
arms recompute the full authorized integer output and validate the original
full-index Merkle commitment. The factorial comparison changes the
interpreter policy and root construction independently. [2], [4]

| Interpreter and root | Sum of 60 source medians, ms | Ratio of sums to baseline | Median paired sourcewise ratio |
|---|---:|---:|---:|
| Original interpreter, original root | 8,286.439526 | 1.000000000000 | 1.000000000000 |
| Original interpreter, warm template | 8,166.188945 | 0.985488269042 | 0.979600803588 |
| Optimized interpreter, original root | 4,751.423296 | 0.573397450267 | 0.259004922735 |
| Optimized interpreter, warm template | 4,673.229136 | 0.563961050019 | 0.213505137249 |

The interpreter change accounts for 3,535.016230 ms of the reduction in the
sum-of-medians workload. Adding the warm root to that interpreter saves a
further 78.194160 ms. The complete combined change saves 3,613.210390 ms.
These differences come from the same first paired run. [2]

### Sparse and dense sources

| Source group | Cases | Interpreter/full-root median paired ratio | Combined median paired ratio | Combined ratio of sums | Cold combined single-call median ratio |
|---|---:|---:|---:|---:|---:|
| Dense, stride 1 | 20 | 0.978560716037 | 0.983864173643 | 0.702158821358 | 0.986106322905 |
| Stride 4 | 20 | 0.236668007461 | 0.207001024209 | 0.158314117480 | 0.256105693635 |
| Stride 8 | 20 | 0.153227564757 | 0.107321802608 | 0.090455182993 | 0.177032133871 |
| All sparse, strides 4/8 | 40 | 0.204195994935 | 0.157365450499 | 0.128001167673 | 0.219017138179 |

For the 40 sparse sources, the baseline sum is 1,994.517477 ms and the
combined sum is 255.300566 ms. The inverse computation uses lengths 193 or
97 at full length 769, and lengths 385 or 193 at full length 1537. It expands
the exact result back into every original position before the signed output
check. [2]

Dense sources retain the full inverse length. Their near-one median paired
ratio shows that a typical dense case changes little. Their ratio of sums is
lower because the expensive P4 cases benefit from unary reuse. Dense cases
contribute 6,291.922049 ms to the baseline sum and 4,417.928570 ms to the
combined sum. This concentration of saved time explains the difference
between dense medians and workload totals. [2]

### Program profiles and P4

Each profile has 12 cases. P4 requests R for its base, left term and right
term. The optimized interpreter computes that inverse once within the
authenticated invocation. Across one complete evaluation of the 60-source
panel, the declared program structure contains 84 R calls in the original
interpreter and 60 in the optimized interpreter. All unary calls decrease
from 216 to 180. Source-free instrumentation verifies invocation-local
reuse and checks that a changed source receives a newly computed result. [2], [5]

| Profile | Original-interpreter/warm-root ratio of sums | Optimized-interpreter/full-root median paired ratio | Combined ratio of sums |
|---|---:|---:|---:|
| P0 | 0.967759944824 | 0.271994848373 | 0.603789597286 |
| P1 | 0.959987234659 | 0.263002790144 | 0.755008746839 |
| P2 | 0.971989512972 | 0.268376045328 | 0.863205599118 |
| P3 | 1.022998730933 | 0.270824714390 | 0.636970172290 |
| P4 | 0.999805188890 | 0.119726205920 | 0.299255710484 |

The P4 primary group includes eight sparse cases, so its 0.119726 median
contains the effects of compression and repeated-R reuse. The four dense P4
cases keep the reciprocal length unchanged and use the original full root in
the interpreter comparison. Their median paired ratio is
**0.3779104278756591**; the ratio of sums is **0.36658863220404964**.
Original dense-P4 source medians total 2,847.919486 ms, and optimized
interpreter/full-root medians total 1,044.014909 ms. This secondary contrast
supports the reuse mechanism while retaining the optimized path's input
scanning and dispatch costs. It was outside the registered pass thresholds.
[2], [4]

The root-only change also varies across profiles. P3's ratio of sums is
1.022999. Reporting that subgroup preserves the measured variation in this
single-host run. [2]

## Descriptive uncertainty

The registered secondary calculation resamples ten degree/profile blocks,
each containing all six stride/replicate cases, for 4,096 seeded bootstrap
draws. The reported intervals describe the ratio of sums of source medians.
The independent auditor recomputed every interval from the original samples.
[2], [3], [4]

| Arm versus original interpreter/full root | Observed ratio of sums | Descriptive 95% block interval |
|---|---:|---|
| Original interpreter, warm template | 0.985488269042 | [0.965806779175, 0.998846740973] |
| Optimized interpreter, original root | 0.573397450267 | [0.374990214345, 0.815881380084] |
| Optimized interpreter, warm template | 0.563961050019 | [0.369656049554, 0.797667982811] |

Only ten blocks and one hosted machine contribute to these intervals.
Program cost differs substantially across the panel. The workload was fixed
by registration; this resampling does not establish a distribution of
performance over arbitrary authorized programs or other machines. [2], [4]

## Geometry setup and cold template cost

The warm path builds six templates after clearing the geometry cache and
before any timed verifier call. Their combined recorded construction time is
**8.738125 ms**. Canonical template encodings total **360,924 bytes**; that
encoding size does not measure the complete Python object heap. All cached
nodes depend solely on trusted length, position and support geometry. [2], [4]

| Full length | Stride | One-time build, ms | Canonical template bytes | Dynamic supported leaves | Static nodes |
|---:|---:|---:|---:|---:|---:|
| 769 | 1 | 0.950522 | 18,124 | 769 | 502 |
| 769 | 4 | 1.362894 | 45,772 | 193 | 1,270 |
| 769 | 8 | 1.037921 | 56,140 | 97 | 1,558 |
| 1,537 | 1 | 0.863652 | 36,520 | 1,537 | 1,013 |
| 1,537 | 4 | 1.738171 | 91,816 | 385 | 2,549 |
| 1,537 | 8 | 2.784965 | 112,552 | 193 | 3,125 |

After each source's primary samples, one combined call reconstructs its
template inside the timed call. These 60 cold-template samples total
**4,755.577124 ms**. Dividing that sum by the baseline sum of source medians
gives **0.573898730459401**. The median of paired cold-sample/baseline-median
ratios is **0.2922103043909299** across all 60 cases. These are secondary
single-call measurements taken in an already running process. They leave
process startup and CPU cache initialization outside their scope. [2], [4]

Charging all six one-time template builds once to the combined
sum-of-medians workload yields a descriptive ratio of
**0.5650155590117559**. This accounting calculation uses the already measured
setup values and source medians; it adds no timing run. [2]

## What the timer includes

Each primary timer spans one complete call to
`verify_authorized_execution`: the original authority-signature verification,
NOR44 precommit/provider verification, independent source/session binding,
full integer program execution, and validation of the computed original
full-index Merkle root against both signed references. The result is returned
before the timer stops. Each repetition passes the same signed bundle to all
four arms, with deterministic source/repetition-seeded interleaving. [2], [4]

The reference vector and original root are constructed before those calls.
Five signed bundles are also prepared outside the measured region. The
benchmark checks each returned vector and root after its timer stops, then
writes its journal entry. Durable journal flushes and decimal serialization
of full vectors are outside the primary timing. Four untimed parity calls
per source precede the five primary repetitions. The cold-template diagnostic
follows them. [4], [6]

The recorded reference-execution total is 8,088.139124 ms, and independent
reference-root construction totals 173.946422 ms. Recorded setup for the 300
signed bundles totals 675.667286 ms. That setup counter starts inside each
bundle loop, after fixture signer initialization and construction of the
provider's output tree. Those earlier operations also occur outside primary
timing and have no separate phase counter. The complete CPU job lasted
198 seconds, including environment setup, source-free tests and evidence
handling. [1], [2], [6]

The run recorded CPython **3.12.16**, `cryptography==46.0.4`,
`gmpy2==2.3.2`, four logical CPUs and a process peak RSS of **76,108 KiB**.
The hosted workflow selected Ubuntu 24.04. The platform string was
`Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`; the CPU-model field returned
`x86_64`, so the receipt supplies no specific processor model. [2], [6]

## Immutable first-run provenance

The run is [44327994013](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44327994013),
attempt 1. The archived ZIP is the byte-identical original hosted CPU
artifact and contains the original source JSON, CPU JSON and progress
journal. The report and independent audit retain its first score. [1], [3]

| Provenance item | Identifier |
|---|---|
| Registration commit | `363ef19abeba68e2897c593697ad56e2aaf4fefc` |
| First-run trigger commit | `a7edfa6850206fc01a9a67650e09ddd31d198766` |
| Original source artifact | `14438371253` |
| Original source job | `134884681802` |
| Original CPU artifact | `14436529899` |
| Original CPU job | `134884822335` |

| Event | Singapore time, 2026-11-18 | Original UTC receipt field |
|---|---|---|
| Source artifact uploaded | 22:47:17 +08:00 | `2026-11-18T14:47:17Z` |
| Source job completed | 22:47:19 +08:00 | `2026-11-18T14:47:19Z` |
| CPU job started | 22:47:21 +08:00 | `2026-11-18T14:47:21Z` |
| CPU artifact uploaded | 22:50:36 +08:00 | `2026-11-18T14:50:36Z` |
| CPU job completed | 22:50:39 +08:00 | `2026-11-18T14:50:39Z` |

The source upload precedes CPU job start by four seconds. The independent
auditor checks this ordering, the trigger/run bindings and attempt number.
It also checks source-code hashes against every original measured file in
the CPU registry. [1], [3]

| Original evidence | Bytes | SHA-256 |
|---|---:|---|
| Public source JSON | 1,021,038 | `b44655796bdc4c9ed47fade2cd9cd563985d0b40ca1177856bb45f36d17185fd` |
| CPU JSON | 27,720,732 | `fba38d6346d35b371464c1d811360879918aeccc0a8932538b6f3206cc4bc52d` |
| Progress journal | 898,921 | `48828148a4dbce8d53e86002f5029902f4e80857ec6f53eddd25c09b3fe21dea` |
| Original hosted CPU/evidence ZIP | 12,310,423 | `c92e2c4e3bfcfd8e1371d1504c89d6812ac263ef270fc9d62e89d335f23da53c` |
| First receipt | 2,571 | `0bc8cdbe1c50737a12863186620d7ef40faa96509e4347392a66bad58ae471cb` |
| Upstream source-only artifact ZIP | 73,788 | `bea12212a0696ab1b5ad85c3e901f3ed17663e3c9627fd615cbb6df44dd25e03` |

The auditor pins the receipt and hosted ZIP hashes in its own code. It
validates canonical raw JSON, reconstructs the full source grid, authenticates
all five bundles for each source, and compares the exact stored vector with
the original interpreter. Every journal event is linked by SHA-256. All
1,200 timed entries match their saved duration, arm order and signed context.
Eight additional adversarial auditor tests pass in ordinary Python and under
`python -O`. These tests were added after the registered run and validate the
read-only audit; they are separate from its original 43-test preflight. [3], [7]

```sh
python research/nor60_authorized_interpreter/original_archive_audit.py
python -m unittest research.nor60_authorized_interpreter.test_archive_audit -v
```

## Engineering closure and assurance scope

Support compression and invocation-local unary reuse preserve the complete
exact NOR45 output on this registered panel. Combining them with the existing
static geometry root reduces the measured complete verifier workload under
the recorded environment. The implementation accepts a fixed internal policy,
and all original authentication checks precede its execution. Every dynamic
unary value is scoped to one authenticated invocation. [2], [3], [5]

The result assumes an honest external program authority and independently
trusted source. Every signing key used here is a public reproducible test
fixture. The verifier binds signed session nonces within a call; it has no
deployment store that consumes a nonce across calls. The experiment adds no
Lean kernel proof. Its support argument and exact executable checks retain
that distinction. [4], [5]

The evidence supports a finite CPU engineering claim. Zero knowledge,
succinctness, external-world semantic correctness, production key custody,
cross-host performance and learned-model performance remain outside the
measured scope. The complete original result, including subgroup regressions
and timing boundaries, is retained for the paper and reusable gym. [1], [2], [4]

## Evidence references

[1]: results/nor60_first_receipt.json
[2]: results/nor60_first_original_hosted.zip
[3]: original_archive_audit.py
[4]: REGISTRATION.md
[5]: optimized_execution.py
[6]: benchmark_cpu.py
[7]: test_archive_audit.py
