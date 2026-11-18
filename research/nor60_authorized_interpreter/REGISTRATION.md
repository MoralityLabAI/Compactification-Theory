# NOR-60: exact authorized execution with support compression and local unary reuse

## Freeze and scope

This protocol and the complete generator, interpreter, tests and benchmark are
committed before any NOR60 source is generated. `REGISTRATION_SHA.txt` is added
in a later trigger commit and contains the full earlier registration commit.
Generation refuses an absent or malformed marker. The source-only job uploads
its complete public JSON and SHA before the dependent CPU job can execute a
reference output or sign a certificate. A first source, timing or reporting
failure remains in the record. Replacement inputs and threshold changes are
excluded from this registered run.

Issue [280](https://github.com/MoralityLabAI/Compactification-Theory/issues/280)
motivates this final experiment. NOR59's first workload recorded a whole
signed-verifier ratio of sums of source medians of 0.9915548176255947 for its
warm indexed Merkle implementation. Its original source SHA is
`be516ca2149cf6d5970842891280f7f7d3407706b9c08b02c4719d153418b0f9`;
CPU SHA is `c56d65e676e7fd81668d3919eb9cbf29922fbe34cfe350418e3c67ae87c62b2b`.
NOR60 targets repeated unary execution and exact inversion of sparse sources.

The measured system remains the original NOR45 signed executable-program
grammar: observed degree 512 or 1024, full length 769 or 1537, unary operations
I/N/R/S, allowed delays 64/80/96/112 and spacings 24/32, amplitudes -2/-1/1/2,
and optional signed suffix beginning at D+32. Public deterministic research
keys sign every fixture. Honest program authority and independently trusted
source are premises. The verifier binds a nonce in each call; a deployment
ledger that consumes nonces is outside this experiment.

## Exact mechanism and fixed trust boundary

The optional `_interpreter_policy="support_memoized"` dispatches to one fixed
internal implementation after the original program-authority signature,
verifier precommit, provider signature and independent query/session binding
checks. The original policy remains the default. Source type, original full
length and the unchanged NOR45 program grammar are checked again inside the
optimized implementation. Callers cannot supply an interpreter callback,
precomputed result, memo table or claimed support stride.

The source scan chooses s=8 if every off-stride entry is zero, otherwise s=4
under the same condition, otherwise s=1. For F(x)=H(x^s), form H by selecting
the original coefficients at multiples of s. Apply NOR45's existing exact
integer inverse recurrence to H at length floor((m-1)/s)+1 and expand into
the original m positions. For a normalized unit series, uniqueness of the
finite inverse establishes equality modulo x^m. N uses each original index;
S doubles that index with the original truncation. All program shifts and
optional suffix positions retain NOR45 semantics.

Each invocation creates a new local dictionary of unary values keyed by its
needed operation token. A token is computed once for that query. The values
are discarded when the invocation returns. Profile P4 asks for R three times,
so this implementation computes one R. No dynamic result is shared across
sources or signed sessions. NOR59's geometry cache may retain only static
indexed zero/padding hashes and static subtrees. Complete output vectors and
their original full Merkle roots must agree in every arm.

Source-free tests compare finite inverses against independent convolution,
check short and odd lengths, test all base/u/v token combinations, and cover
signed source changes, nonce swaps, forged keys/signatures, false provider
roots, invalid types, wrong full lengths, nonunit inputs and invalid grammar.
Instrumentation that counts calls appears only in toy tests. The benchmark
retains all original function identities and checks source-code hashes.

## Public source panel and content identity

The exact panel has 2 degrees x 3 source strides x 5 profiles x 2 replicates,
giving 60 cases in degree/stride/profile/replicate order. Degrees are 512 and
1024. Strides are 1, 4 and 8. All five profiles are fixed below.

| Profile | Bits | Base | u | v | Delay | Spacing | Amplitude | Hidden beta | Hidden op |
|---|---:|---|---|---|---:|---:|---:|---:|---|
| P0 | 2 | I | R | N | 64 | 24 | 1 | 0 | none |
| P1 | 8 | N | S | I | 80 | 32 | -2 | 1 | R |
| P2 | 16 | R | I | S | 96 | 24 | 2 | -1 | S |
| P3 | 2 | S | N | R | 112 | 32 | -1 | 0 | none |
| P4 | 8 | R | R | R | 80 | 24 | 1 | 2 | N |

The domain is `NOR60/SOURCE/v1`. The random seed is the big-endian integer
SHA-256 of `domain|degree|stride|profile|replicate`. F[0]=1. Every positive
off-stride coefficient is zero. At a supported index, the magnitude is
uniform in [2^(bits-1),2^bits-1]. Replicate 0 draws a seeded random sign;
replicate 1 uses (-1)^(index/stride). All full input coefficients are stored
as canonical signed decimal strings. A new row's source identity hashes its
canonical schema-bearing public core. It contains no output, root, signature,
session nonce, reference value, cache contents or timing.

Every ZIP and public JSON SHA for the 588 first NOR51-NOR59 identities is
verified, including NOR57's original reporting failure and NOR57B. A separate
canonical query-content hash covers only vector length and the full signed
coefficient strings. It ignores experiment schema and descriptive metadata.
This catches vector reuse that a renamed experiment identity cannot catch.
The known 588 historical rows contain 588 distinct vectors. A third hash
binds query content, degree and program for the 60 historical NOR59 rows that
carry a program. Prior reciprocal-only rows have no authorized program;
their complete vectors remain covered by the content audit. Unrecognized or
malformed historical rows cause failure. New IDs, query vectors and program
inputs must each be unique within the 60 cases and disjoint from the
applicable historical sets.

## Four paired arms

The hosted runner is Ubuntu 24.04 with CPython 3.12,
`cryptography==46.0.4`, `gmpy2==2.3.2` and `OMP_NUM_THREADS=1`. This is a
CPU-only engineering experiment. The complete source-free test preflight
runs before measurement. Six geometry templates are built once outside
timed calls after clearing the geometry cache, and their cost is recorded.

| Arm | Interpreter | Root construction |
|---|---|---|
| original_interpreter_full_root | Original NOR45 | Original full indexed Merkle |
| original_interpreter_warm_root | Original NOR45 | NOR59 warm geometry template |
| optimized_interpreter_full_root | Support compression and local unary reuse | Original full indexed Merkle |
| optimized_interpreter_warm_root | Support compression and local unary reuse | NOR59 warm geometry template |

Each source has one original complete reference output and original root,
both constructed outside timing. Five distinct session nonces are derived
from `NOR60/VERIFIER-ISSUED-RESEARCH-NONCE/v1|source_sha|rep`. Each repetition
has a valid program certificate, verifier precommit and signed provider root.
The same bundle is passed to all four arms in that repetition. Signing/setup
time and all signed bundles are retained separately.

Before timing, each arm executes once for exact parity. The five timed
repetitions shuffle arm order from SHA-256 of
`NOR60/ARM-ORDER/v1|source_sha|rep`. Every timed call includes the original
signature and binding checks, full exact execution and full-root validation.
Vector comparisons, durable journal writes and output serialization occur
after the timer stops. Every sample must be finite and positive, return the
original complete vector, and return the original root.

After all primary samples for a source, one separate optimized-interpreter
call uses NOR59's cold root policy. It constructs its template during that
call. This secondary measurement describes cold template construction within
an already running process. Process startup, fresh CPU caches, key generation,
provider computation and signing are separately scoped. That single sample
does not affect any registered threshold. Complete output coefficients are
serialized after primary timing, along with the original indexed output
digest, full root and all five signed bundles.

## Five registered gates

1. For all 40 stride-4/8 cases, the median of sourcewise ratios of optimized
   interpreter/full-root median time to original interpreter/full-root median
   time must be strictly less than 0.70.
2. Across all 60 sources, the sum of optimized-interpreter/warm-root source
   median times divided by the sum of original-interpreter/full-root source
   median times must be strictly less than 0.80.
3. For all 12 P4 cases, the median of sourcewise optimized-interpreter/full-root
   to original-interpreter/full-root median ratios must be strictly less
   than 0.65.
4. All 60 full output vectors and original roots must agree across all four
   signed routes, with checked support closure and all source-free security
   tests passing.
5. Retain the first complete source panel and its upstream SHA, all 588
   historical exclusions and the schema-independent content audit, all five
   samples per arm and five distinct signed sessions, unmodified runtime
   helpers and source hashes, and the original CPU/source/journal bytes.

Engineering PASS requires all five. Any failure is preserved as the first
engineering FAIL. The read-only archive audit independently replays hashes,
the source roster, signed outputs and registered score arithmetic. It also
checks the upstream source upload and first-run provenance. The one-shot
archive publisher is removed after publishing those original bytes.

The P4 threshold includes both dense and sparse cases. Its median alone
cannot separate memoization from compression. Dense P4's four cases and the
source-free call-count test report the memoization effect; source-derived
reduced lengths describe compression. The four arms isolate interpreter and
root changes. They do not separate every individual interpreter subchange.

Secondary reports include degree/stride/profile summaries, dense P4, a
4,096-draw descriptive bootstrap over ten degree/profile blocks of six cases,
geometry setup costs and cold template calls. The fixed panel and one host
limit population claims. Bootstrap intervals are descriptive and do not
replace the engineering thresholds.

## Failure retention and stopping rule

The source job writes with exclusive creation. The CPU job also refuses any
existing final output, progress journal or failure file. An append-only
SHA-chained journal records the attempt start, source start, every completed
timed arm and source outcome; writes are flushed with `fsync` outside timing.
The final CPU JSON binds the journal SHA. A source exception records its
partial samples and failure, then continues through the remaining fixed
inputs. A process-level exception writes a failure receipt and retains the
journal. A hard kill leaves the durable journal for the workflow's unconditional
artifact upload. A later run keeps a distinct follow-up identity.

The first hosted job has a 20-minute hard limit. There are exactly 60 sources,
five timed repetitions and four primary arms. No automatic retry, replacement
source, adaptive threshold or model/GPU work is authorized by this protocol.
Follow-up work must use a separately identified exploratory or registered
experiment, preserving the first record.

## Claims available from completion

The result can establish exact execution parity and measured CPU costs for
this finite signed-program panel under the recorded environment. The support
substitution argument is mathematical prose backed by executable controls.
It has no companion Lean kernel proof in this experiment. The external
authority's semantic correctness, deployment key secrecy, cryptographic
security assumptions and nonce-state management remain explicit premises.
No zero-knowledge, succinctness, Bitcoin consensus, learned-model result or
universal acceleration follows from these timings.
