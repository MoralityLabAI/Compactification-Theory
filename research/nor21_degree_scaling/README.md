# NOR-21: degree-512/1024 exact public-verifier scaling and out-of-grammar failures

**Preregistered 16 November 2026, before generating fresh NOR-21 cases or opening test grades.** Continues [NOR-19 #181](https://github.com/MoralityLabAI/Compactification-Theory/pull/181) and [NOR-20 #183](https://github.com/MoralityLabAI/Compactification-Theory/pull/183), with exact coefficient-streamed semantic equivalence from [MathCan #101](https://github.com/patrickdugan/MathCan/pull/101).

## Questions

1. Does the exact four-unary cached **coefficient-streaming** verifier remain faster and lower in Python temporary allocation than native sixteen-template checking and eager cached sixteen-template checking at **512 and 1,024 public coefficients**?
2. Does public-program certification respond correctly when the true task is outside its complete sixteen-program family? There are two different failures: a **publicly inconsistent demonstration**, which a full public mask should reject; and a **hidden future override**, where all public observations admit a generated program but the withheld future target disagrees. The latter is an indistinguishability counterexample to unconditional future correctness, not a failure of the family-conditional theorem.

The original finite RRC25 grammar is frozen: four unary operators I/N/R/S, delays 64/80/96/112, spacings 24/32, amplitudes ±1/±2, and all sixteen ordered U/V candidate pairs. We change only the number of public coefficients and the source-generating distribution. The original 192-degree RRC25 native checker is **not** used as an oracle for larger degrees, because it deliberately truncates to degree 192. Every proposed scaled checker implements the original exact interpreter semantics at its registered degree.

## Frozen source and evaluation grid

- Public demonstration degrees: **512 and 1024**, inclusive, with exactly two input/output demonstrations per source.
- Public query INPUT degrees: **768 and 1536**, respectively. Withheld query OUTPUT is inaccessible to every verifier and every input fixture.
- Source families: **wide_dense_small**, with nonzero coefficients uniformly sampled from ±{1,2}; **wide_sparse_small**, coefficients sampled from ±{1,2} with 0.12 probability and exact zeros otherwise; **wide_rational_short**, reciprocal of a normalized polynomial with nonzero coefficient ±1 at degree 1 and three additional ±1/±2 coefficients at random degrees 2–12. Even small polynomial coefficients can produce much wider reciprocal-series coefficients; report actual observed widths.
- Cases: **two independent replicate cohorts** indexed r=0,1. For each degree and family in each cohort, generate **three independent source cases**, one of each registered case kind: in_grammar, public_corrupt, future_shift. Total **36 cases**, comprising **18 cases at each degree**, 12 in-scope truthful cases, 12 public-visible corruptions and 12 hidden-future overrides across all degrees. Each case has its own source seed and complete-public input hash. No two cases may share a complete-public input hash, including future-shift cases, to avoid treating public twins as independent timing observations.
- Source seeds: corpus.SEED + **94210000** + 1,000,003*r + 100,009*(0 for degree512 else 1) + 10,007*family_index + 331*casekind_index. Draw a deterministic six-field original grammar program and three fresh normalized sources. Select the first candidate with one public header, exact complete original-program membership and a **unique public-query hypothetical output among all publicly accepted templates** for each *in_grammar* and *future_shift* source. For public_corrupt, alter the second demo's final observed degree coefficient by +1 and require the exact sixteen-template public mask to be empty. Skip unsuitable programs; limit to 80 independent attempts per seed, fail closed on exhaustion. These are predeclared source-filter conditions, not posthoc future-outcome tuning.
- The source generator may know the planted true program to generate cases, but must serialize **only public demonstrations, public query INPUT, the public header, kind and source seed/attempt, and exact public-native accepted mask/unanimity** to the benchmark fixture. The planted program and hidden target never enter the fixture. Source selection may use only full public-demo compatibility and public-query hypothetical candidate predictions.
- Evaluation creates a separate sealed native-generator oracle only **after all public policies have frozen their choices**. For future_shift, the sealed target equals the valid generating program's true future output with its degree-query final coefficient incremented by 1. That produces a valid-world counterfactual with exactly the same complete public observations as the shifted world and a different target. Record the public-input digest identity of the two possible worlds, without exposing private future arrays. For public_corrupt, the public mask must be empty and the verifier must abstain.

This study introduces a fresh source-generating distribution. Its synthetic cases are excluded by new disjoint seed ranges from NOR11–20 and have explicit source hash uniqueness checks, but source-distribution independence alone is not proof of generality to an external theorem prover.

## Fixed public comparison algorithms

Every mode separately computes all sixteen candidate predicted query outputs using four shared exact public unary transforms and separately verifies all sixteen candidate templates against both demonstrations at full registered degree. All return the identical original native 16-bit acceptance mask, output-unanimity predicate, and chosen candidate or abstention decision. A full public mask of zero **must produce an abstention** rather than a Python assertion.

- native16: independently execute each public program via the original RRC25 exact execute implementation applied to sources truncated at the **registered degree**.
- cached16: evaluate four unary transforms per demo and materialize sixteen full candidate outputs at once, using the same finite-series exact formula.
- stream16: four unary transforms per demo, compare candidate predictions coefficient by coefficient with early mismatch exit; at most one demo's unary maps remain live and no full candidate output tuples are materialized.
- chunk4: construct at most four candidate outputs at once, release them after comparing, then proceed.

All four modes use the same query-first order for this comparison, so the changing candidate-buffer strategy is isolated from query/verification execution order. Query continuation simulation is charged to all modes. The experiment also records verifier-only allocation in a separate untimed pass, because query outputs may dominate end-to-end memory.

## Timing, memory, and resource policy

Use the hosted Ubuntu 24.04 CPython 3.12 runner. Each mode gets one warmup and **three measured repetitions per case** in a deterministic shuffled order; collect garbage before each and disable GC during measured segments. Record raw query/verification/certificate/total nanosecond times, per-case medians, and isolate Python-traced peak temporary allocation in separate untimed complete- and verifier-only runs. Record sampled subprocess RSS when available but do not infer temporary allocation equality from equal RSS. Actual Python integer bit widths and quadratic reciprocal recurrence product-count/operand-width proxy must be reported separately from time.

The full scaled native verifier is potentially computationally expensive. Hosted CI fails closed after **60 minutes** for the full benchmark, and never silently lowers sample counts, drops the degree-1024 cohort, or changes seeds. Any resource-driven protocol revision requires a dated amendment **before** new heldout evaluation.

## Preregistered outcome rules

1. **Exactness:** all four modes reproduce the same public acceptance mask and the same output-unanimity/abstention choice on all 36 cases at each degree.
2. **Scope:** all twelve in_grammar cases must certify the correct withheld original program future. All twelve public_corrupt cases must show zero accepted public candidates and abstain. All twelve future_shift cases should yield public unanimity and a nominal certificate, while the sealed native future grader marks them wrong. At least one source from each degree must produce this explicit indistinguishability failure. These illustrate the boundary of the generator-membership premise.
3. **Scaling:** at **each** degree independently, median of case-paired native16/stream16 *complete* CPU ratios must be ≥**1.5×**; median of per-case stream16/cached16 complete traced Python peaks must be ≤**0.80**, with lower streaming memory on ≥**14/18 cases**. The streaming-vs-native speedup and full streaming-vs-cached peak ratio must be computed on the same 18 case identities per degree. All cases in each degree have unique public inputs.
4. Report by degree, source family and case kind: median complete and verifier-only time, complete and verifier-only temporary Python allocation, peak observed reciprocal bit width, native and cached exact R invocation counts, and the count of possible out-of-family false acceptances. A deterministic **4096-draw two-cohort blocked bootstrap** yields conditional intervals for the two paired ratios, clearly labeled as only two cohorts.
5. The degree scaling result is a **CPU engineering observation**, not a formal O(n²) native-time theorem. Lean companion work formalizes exact degree-independent public acceptance and the impossibility of unconditional target correctness when two worlds have identical public observations but incompatible future labels.

The public fixture will be pinned by SHA-256 before being measured, and raw first-run evidence will be source-locked and independently audited. Any failure of degree-1024 runtime, source uniqueness, exactness or the out-of-family controls is recorded as a failure.


## First hosted execution: exact depth scaling, bounded allocation and hidden future counterexamples

The preregistered first run passed all three hosted CI jobs: [Actions #43810513348](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43810513348). The first public-only source fixture and original measured JSON are preserved together in [nor21_original_hosted.zip](results/nor21_original_hosted.zip), with hashes pinned by [the frozen receipt](results/nor21_first_frozen_receipt.json):

- Public fixture: 7,248,513 bytes, SHA-256 c16f4b92db3f61f2753c853c01dc3d3c401c58d8e05138f0a3ca6d58e10d08c2.
- Full measured CPU/memory and sealed-grade JSON: 197,084 bytes, SHA-256 c308a73f8267ff38b78dbbb9d0df909064de0608c5dbfc4ca8f6f84e0853ed29.

The [independent exact receipt auditor](receipt_audit.py) passed in [archival Actions #43812672373](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43812672373). It imports only the original RRC25 exact interpreter, reconstructs the private generating programs and hidden futures from registered seeds after reviewing frozen public policy decisions, and independently checks 36 accepted-template masks and sealed labels, 432 nanosecond timing samples, 288 separate memory records, and 16,384 deterministic bootstrap draws. The one-shot archival publisher was deleted after committing original evidence.

### Observed exact CPU and memory scaling

Measured on Ubuntu 24.04 CPython 3.12. The full benchmark runs one warmup and three measured repetitions per source and verifier mode. Source fixture loading and generation are excluded; temporary Python allocation is measured in separate untimed tracemalloc passes.

| Observation degree | Native 16 median total CPU | Streamed 16 median total CPU | Median paired native/stream speedup | Eager cached median peak Python allocation | Streamed median peak Python allocation | Median paired stream/cache peak ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 136.305 ms | **53.544 ms** | **2.362×** | 1,014.57 KiB | **535.35 KiB** | **0.51892** |
| 1024 | 652.734 ms | **244.897 ms** | **2.421×** | 2,817.33 KiB | **1,679.02 KiB** | **0.49558** |

Streaming used less peak Python-traced memory in all 18 cases at each degree. Both degree-stratified preregistered criteria passed: paired native/stream CPU speedup at least 1.5×, paired stream/cache peak memory ratio at most 0.80, and lower streaming allocation on at least 14/18 cases. The four-output chunked verifier gave median total CPU times 54.493 ms and 245.804 ms at the two degrees with the same exact public decisions.

Verifier-only temporary allocation fell from 580.97 KiB eager to **38.93 KiB streamed** at degree 512, and from 1,491.42 KiB to **119.65 KiB** at degree 1024. Complete-operation peaks remain higher because all methods construct all sixteen public query hypothetical outputs. The largest actual reciprocal inverse coefficient widths were 941 and 1,760 bits respectively; public source coefficients reached 508 and 1,025 bits. Native integer operations grow in operand width as well as coefficient count.

The four-thousand-ninety-six-draw two-replicate conditional percentile intervals were: native/stream CPU ratio 2.291–2.434 at degree 512 and 2.321–2.499 at degree 1024; streaming/cache memory ratio 0.51835–0.51949 and 0.49359–0.53723 respectively. There are only two independent 9-case source replicate blocks at each degree, so these intervals have limited inferential value beyond the measured benchmark.

### Out-of-family native grader test

| Case kind across both degrees | Cases | Public certificates | Sealed future-correct | Out-of-family false accepts |
| --- | ---: | ---: | ---: | ---: |
| Generated in-grammar targets | 12 | **12** | **12** | 0 |
| One corrupted late public demonstration coefficient | 12 | **0** | 0 | 0 |
| Publicly consistent demonstrations, privately shifted query target | 12 | **12** | **0** | **12** |

Every public-corrupt source has zero compatible candidate templates and requires an abstention. The other 24 cases have one accepted template and a unique publicly predicted future. For each future-shift case, the hidden target differs at the last query-output coefficient, while a valid generated world with the **same complete public demonstrations and query INPUT** has the original target. A policy that certifies the valid-world continuation returns the same answer in the hidden-target-shifted world and is therefore incorrect there.

This result exposes the **generator-completeness assumption** behind the earlier certificate theorem. It is a concrete out-of-family counterexample and does not refute the theorem under its stated assumption. [MathCan NOR-21 #106](https://github.com/patrickdugan/MathCan/pull/106) proves exactly the empty-mask abstention, public-world indistinguishability and triangular reciprocal multiplication counts at arbitrary degrees.

### Resource and generalization scope

The permanent CI performs independent immutable evidence replay alongside exact public interpreter tests and a graded complete benchmark. Its hosted benchmark time limit is **60 minutes**, as preregistered; the original first run completed within this limit. The study compares exact synthetic finite-series verifiers and measures CPU time and Python-owned temporary allocation. It does not establish constant total memory, cross-machine performance, or unconditional correctness against unknown future targets.
