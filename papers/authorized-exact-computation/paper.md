---
title: "Exact Computation Under Signed Program Authority"
subtitle: "A study of support compression, verification cost, and preserved experimental evidence"
author: "MoralityLabAI research series"
date: "18 November 2026"
bibliography: references.bib
link-citations: true
---

## Abstract

The Numerical Orbit Research series began with compact representations of recurrent trajectories and reached a practical question: what can an evaluator establish when a proposed continuation, an authorized program, and an external target differ? This paper consolidates the resulting exact-computation studies. An independent verifier executes a signed integer-series program and checks its complete Merkle commitment under separately pinned authority and provider keys. The experiments retain original source records, timing samples, registered decisions, and failed runs. A transparent execution trace first cost 3.28 times direct computation on a historical panel. Later studies identified redundant reciprocal evaluation, coefficient packing, and known-zero hashing as separate costs. Exact support compression passed its registered engineering criteria, while several routing and serialization changes missed theirs. In NOR-59, caching index-specific zero and padding hashes reduced the sparse root-only median ratio to 0.309528; the complete signed workload ratio was 0.991555, with a descriptive interval spanning one. The final NOR-60 factorial experiment reduced the complete matched workload ratio to 0.563961, with all 60 outputs and roots preserved and all five registered gates passed. Its descriptive ten-block interval was [0.369656, 0.797668]. We give finite algebraic arguments for the optimizations, state the narrower scope of the existing Lean results, and derive a reusable evaluation protocol from the failures. The evidence supports computational fidelity for the declared synthetic program grammar. Semantic correctness still requires a justified program authority, and operational security depends on key custody and durable session state.

## 1. Research question and lineage

A compact representation preserves a chosen observable. Its usefulness for another task depends on the information that survives the representation and on the costs needed to recover the target. The NOR line made this dependency concrete through a sequence of finite experiments. Early work studied a small trained GRU under a declared zero-input intervention. Exact rational interval arguments certified distinct attracting states and source-conditioned entry into their neighborhoods. Source-family prediction remained a separate empirical question: the NOR-08 audit found that several nominally different examples shared the same model-visible input. The resulting effective sample count and a majority baseline changed the interpretation of the classification result [NOR08].

Later experiments examined proposed formal-series continuations. NOR-15C enumerated 43,680 fixed four-candidate orders on a finite cohort where distinct full demonstrations shared one neural feature tensor. Its hindsight optimum was a bound for that restricted finite policy class. NOR-16 then used more public information and charged the resulting exact computations. Its four-check consensus criterion failed, and 140 of 192 late-source observations admitted distinct programs with identical public observations and different query outputs [NOR15C], [NOR16]. These constructions exposed the target information that an evaluator would need to resolve.

NOR-42 supplied paired clean and hidden program worlds with identical public observations. The hidden world adds a late term whose first nonzero coefficient occurs after the observed prefix. A source-public predictor must issue the same answer in both worlds. An independently supplied target witness permits direct tests of their disagreement. The subsequent protocol studies introduced authenticated roots, verifier-owned challenge positions, signed proposal commitments, and finally a separate authority for the executable program [NOR42], [NOR43], [NOR44], [NOR45].

The present paper concerns the cost and meaning of that final check. Once a program has been authorized, can exact structural properties reduce the work needed to verify its output while preserving all source and session bindings? The engineering sequence also provides a case study in retaining unfavorable evidence. A speed target can fail despite correct outputs; a reporting exception can leave a first run unscorable. Both outcomes constrain the final account.

The historical evidence through NOR-59 is pinned to repository commit [`714a7f5e1203`](https://github.com/MoralityLabAI/Compactification-Theory/commit/714a7f5e1203289ef053dfa5becaa3537d1b946b). The accompanying evidence ledger identifies each report, original archive, and its first verdict. Comparisons in this paper use matched arms within the relevant experiment. Differences between machines, source panels, and timed boundaries prevent chaining the ratios into a single cumulative speedup.

## 2. Exact model and assurance contract

### 2.1 Finite integer-series programs

Let

$$
F(x)=\sum_{i=0}^{m-1} f_i x^i\quad\text{in }\mathbb Z[x]/(x^m),\qquad f_0=1.
$$

The interpreter implements identity $I(F)=F$, sign substitution $N(F)=F(-x)$, reciprocal $R(F)=F^{-1}\bmod x^m$, and index dilation $S(F)=F(x^2)\bmod x^m$. The reciprocal coefficients satisfy

$$
r_0=1,\qquad r_n=-\sum_{k=1}^{n}f_k r_{n-k},\quad 1\leq n<m.
$$

All arithmetic uses exact signed integers. In the signed NOR-45 grammar, $D\in\{512,1024\}$ and $m=3D/2+1$. A program has the form

$$
P(F)=B(F)+c x^k+c x^{k+t}U(F)+c x^{k+2t}V(F)
       +\beta x^{D+32}T(F)\pmod{x^m}.
$$

Here $B,U,V,T$ are permitted unary operators, $k\in\{64,80,96,112\}$, $t\in\{24,32\}$, and $c\in\{-2,-1,1,2\}$. The optional suffix has $\beta=0$ or $\beta\in\{-2,-1,1,2\}$. Its transform is absent when $\beta=0$. The verifier validates this closed grammar before execution. Several later single-reciprocal experiments also used $m=3073$; that arithmetic workload lies outside the signed NOR-45 program grammar [NOR45], [NOR51], [NOR59].

### 2.2 Authority and chronology

The proposer first commits to a complete output under a source-bound session context. A verifier-signed receipt fixes that proposal. A separately pinned program authority signs the executable specification, query digest, expected output root, and the source/session/precommit fields. The output provider signs its own reference root. The verifier authenticates these messages, executes the authorized program independently, and requires its computed root to agree with both signed roots. A subsequent sparse challenge refers to exactly the positions selected by the verifier [NOR44], [NOR45].

The NOR-59 and NOR-60 timers end when `verify_authorized_execution` returns. The later challenge, opening checks, and session finalization lie outside that measured call.

Ed25519 supplies the signature primitive [@bernstein2012; @rfc8032]. Its use authenticates the holder of a provisioned key. The claim that a particular program describes the intended target comes from the authority policy. This distinction is exercised by an executable control: a correctly pinned authority can authorize a false clean program, and an honest executor can produce its exact root. In the original NOR-45 panel, all 64 such controls passed the cryptographic checks while disagreeing with the designated hidden world [NOR45].

The recorded keys and nonces are public research fixtures. The full NOR-44 session object enforces its in-memory phase order, while the later NOR-45 verification function authenticates each supplied call without maintaining a durable ledger of consumed nonces. A deployment would need its own key management and global session-consumption policy. These operational premises accompany every performance result in the paper.

### 2.3 Complete indexed commitments

For each output position $i<m$, the leaf hash includes a distinct domain tag, the original length $m$, the original index $i$, and a canonical signed-integer encoding. Padding leaves use a separate domain and the same length/index framing. Internal nodes hash an ordered pair of children under another tag. The tree is padded to the next power of two. This construction follows the authenticated-tree approach associated with Merkle commitments [@merkle1988], with explicit encoding choices fixed by NOR-45.

Arithmetic equivalence implies byte-identical roots because each leaf and node receives identical bytes. The inference from an accepted root to a unique committed output additionally assumes the required hash-binding property. Separating these two statements helps locate what an optimization must preserve and what cryptographic assumptions remain external.

## 3. Finite correctness arguments

### 3.1 Information limit of a common observation

Let two worlds $w_0,w_1$ have the same observation $O(w_0)=O(w_1)$ and different targets $Y(w_0)\ne Y(w_1)$. Any deterministic predictor $a(O)$ returns the same value in both worlds, so it can equal at most one of their targets. Under an explicitly equal prior over the pair, a randomized observation-only predictor has average exact success at most one half: its probabilities of returning two distinct answers sum to at most one.

For the NOR-42 pair, the two program outputs differ by $\beta x^{D+32}T(F)$. Every permitted transform satisfies $T(F)_0=1$. Therefore the first changed coefficient is $D+32$ with difference $\beta\ne0$, while the observed coefficients through $D$ agree. This is a finite identifiability argument about the constructed worlds. Its conclusion applies to the declared observation map [NOR42].

If a fixed proposal disagrees with an independently authenticated target on $d$ of $m$ positions, a uniformly drawn set of $k$ distinct checks misses every discrepancy with probability

$$
\Pr(\text{miss})=\frac{\binom{m-d}{k}}{\binom{m}{k}}.
$$

The numerator counts check sets contained in the agreement positions. Thus the detection probability is one minus this ratio, conditional on the stated randomness and chronology. Reproducible public challenge seeds support replay of an experiment; operational unpredictability requires its own source of entropy.

### 3.2 Supported inverse and exact expansion

For an integer $s\geq1$, suppose $f_i=0$ whenever $s$ does not divide $i$. Set

$$
q=\left\lfloor\frac{m-1}{s}\right\rfloor+1,\qquad
H(y)=\sum_{j=0}^{q-1} f_{sj}y^j.
$$

Substitution $y\mapsto x^s$ defines an injective ring homomorphism

$$
\phi_s:\mathbb Z[y]/(y^q)\longrightarrow\mathbb Z[x]/(x^m).
$$

It is well-defined because $sq\ge m$, and it is injective because $s(q-1)<m$. If $G=H^{-1}\bmod y^q$, then

$$
\phi_s(H)\phi_s(G)=\phi_s(1)=1\pmod{x^m}.
$$

The normalized inverse is unique. To see this directly, the coefficient at degree $n$ is determined by the input coefficients and inverse coefficients of degree below $n$ through the displayed recurrence. Induction fixes every coefficient through $m-1$. Consequently

$$
R(F)=\phi_s(G),
$$

so its off-support coefficients are exactly zero. Computing the shorter inverse and expanding to the original length preserves the entire output vector. This argument requires checking the support of the whole source; a declared stride is insufficient evidence.

### 3.3 Closure of the signed grammar

Identity preserves support. Sign substitution changes each supported coefficient by $(-1)^{sj}$, so it also preserves support. When $s$ is even, sign substitution leaves those coefficients unchanged; applying signs according to the compressed index $j$ would give the wrong result. Dilation replaces each supported index $sj$ by $2sj$. The preceding inverse argument supplies reciprocal closure.

Multiplication by $x^a$ preserves stride-$s$ support when $s\mid a$, and sums of supported series remain supported. All signed NOR-45 delays, delay-plus-spacing terms, and permitted suffix onsets are divisible by eight. Hence every program preserves support for $s\in\{1,4,8\}$. The implementation still checks every off-support coefficient of the recomputed output before applying a static Merkle template [NOR59]. This final check protects the root path against an implementation error in an earlier stage.

### 3.4 Memoization within one invocation

For a fixed immutable source, each unary transform is a pure deterministic function of the source and its operator token. Replacing several evaluations of the same token with one local result preserves each occurrence in the program equation. The cache belongs to that invocation and is discarded when it ends. Different sources and sessions obtain fresh computed values.

If an authorized program contains $r$ occurrences of the reciprocal token, a local cache reduces those inverse computations from $r$ to one whenever $r>0$. Combining this with support compression changes the recurrence's scalar multiply-add count from order $r m^2$ to order $q^2$. Expansion and program assembly still touch the full output. Exact-integer bit lengths affect the cost of every multiplication, so this operation-count statement does not prescribe a wall-time speedup.

### 3.5 Reuse of static Merkle subtrees

Fix the original geometry $(m,s)$ after independently checking support. A zero leaf at an off-support position and a padding leaf are determined entirely by the indexed encoding. An internal subtree is static exactly when all its descendants are static. Cache only those hashes.

For an admissible new output, every cached leaf equals the corresponding leaf in a complete fresh tree. Each supported leaf is hashed from the current value. Induction from the leaves to the root then proves equality at every internal node, since both constructions hash the same ordered child pair with the same domain tag. Any node that depends on a supported value is recomputed. This proof concerns deterministic extensional equality; binding an adversarially supplied root is a separate cryptographic property.

### 3.6 What the Lean artifacts establish

The existing MathCan module `NOR45AuthorizedProgram.lean`, pinned to merge commit [`17e9c92dbf48`](https://github.com/patrickdugan/MathCan/commit/17e9c92dbf48724786cdd6a2df2645b9ff7ac111), proves conditional composition statements for authorized execution, valid openings, and proposal-bound sparse rejection. It assumes the supplied execution function, authorization relation, and hash/opening properties. The NOR-46 module at [`9b2e9f5a258b`](https://github.com/patrickdugan/MathCan/commit/9b2e9f5a258b2a0bfa31489333e1e5b9b86b7566) proves uniqueness for a causal integer recurrence and composes a supplied exact-trace equality with the signed-root claims [LEAN45], [LEAN46].

In particular, the concrete trace-to-execution equality is an explicit hypothesis of the composition theorem. The Python convolution packing, support expansion, and static-tree implementation need separate correspondence proofs. Sections 3.2 through 3.5 provide mathematical arguments for those mechanisms; their status here is a written derivation supported by executable tests. The archived NOR-45/46 reports record successful Lean kernel checks for their narrower statements.

## 4. Experimental method

### 4.1 Study sequence and timing boundaries

NOR-45 introduced the signed authority control on 64 historical source pairs. NOR-46 through NOR-48 reused a preselected 16-source historical panel for implementation and timing questions. NOR-49 evaluated 48 previously untimed historical identities. NOR-50 generated a new 64-source panel; the succeeding reciprocal and support studies used prospectively frozen panels of 48 or 60 sources. NOR-59 returned to the full authorized program verifier on a new 60-source grid [NOR45--NOR59].

The arithmetic studies measured exact computation and complete-root construction, with common authentication costs outside selected timed arms. Later compact-witness studies timed their signed envelope checks. NOR-59 measured complete program-authority, verifier-precommit, and provider-reference authentication on every arm before full program execution and root checking. The evidence ledger records these boundaries because the phrase "verification time" alone leaves the comparator underspecified.

For the later studies, a source-only job generated and uploaded its manifest before the dependent CPU job. Registered case identifiers, program profiles, and decision rules were frozen before source-specific timing. Five repetitions per source and arm were interleaved using a deterministic source/repetition schedule. Identical signed bundles were reused across matched arms within a repetition. Reference production and fixture signing were measured or recorded separately from verifier timing. Warm geometry preparation occurred outside the warm call; cold calls charged construction anew.

### 4.2 Two different aggregation questions

Let $t_{a,j}$ be the median of the recorded repetitions for arm $a$ on source $j$. The median sourcewise ratio is

$$
M_{a/b}=\operatorname{median}_{j}\left(t_{a,j}/t_{b,j}\right).
$$

The matched workload ratio is

$$
W_{a/b}=\frac{\sum_j t_{a,j}}{\sum_j t_{b,j}}.
$$

The first gives each source equal position in the ratio distribution. The second weights its contribution by baseline cost. Their answers can diverge sharply when a few expensive sources dominate total time. Each threshold uses its registered endpoint. Post hoc aggregates remain descriptive and preserve the original gate decision.

Reported bootstrap intervals resample the registered source blocks where available. Repetitions within one source measure timing variability; they do not create independent source draws. The NOR-59 interval uses ten degree-by-profile blocks and 4,096 bootstrap resamples. These intervals describe the finite panel under that block scheme. Hardware and runtime replication would be needed for broader platform claims. The emphasis on the level at which variability arises follows established systems-benchmarking methodology [@kalibera2013].

### 4.3 Source identity, failure retention, and resource costs

The later manifests reject previously archived source-record hashes. Their provenance chain makes first-record identity auditable. Different studies use different schema and domain strings, so source-record hash exclusion alone cannot establish uniqueness of the underlying query vector or uniqueness up to program equivalence. A reusable cross-study collection should retain schema-independent query and query/program fingerprints alongside its namespaced record identifiers.

A run's final receipt must preserve the raw samples and exact gate values. NOR-57 demonstrated this requirement when its summarizer failed after source-level progress lines had been printed. Rounded logs omitted enough information to leave the first final gates unscorable. NOR-57B used a fresh registered cohort and retained the earlier failure. It also preserved an inherited metadata error: one descriptive field stated 348 exclusions, while the source fixture and admission check used 408. The archive documents that mismatch without editing the original bytes [NOR57], [NOR57B].

Cost accounting includes whoever produces a witness and the representation used to move it. A precomputed inverse can make a verifier cheap while leaving substantial work with its producer. NOR-54 separately charged local production, JSON serialization/parsing, and verification. A subsequent NOR-55 loopback diagnostic exercised two processes; its scope remained a local transport measurement. None of these observations establishes an off-host service cost [NOR54], [NOR55].

## 5. Preserved results across the sequence

Table 1 lists the first decisions for the computational sequence. Ratios below one indicate less time for the numerator arm on the named endpoint. The gates were engineering targets chosen for each experiment; counting how many passed across the sequence would not form a statistical meta-analysis.

| Study | Cases and principal matched endpoint | First recorded outcome |
|:--|:--|:--|
| NOR-45 | 64 historical program pairs; exact authorized roots | 64/64 parity; dishonest authority control remained semantically wrong |
| NOR-46 | 16 historical pairs; full trace/direct median ratio | 3.28; speed objective failed |
| NOR-47 | Same 16; GMP trace on 12 reciprocal cases | 0.4966 versus direct; frozen speed gates passed |
| NOR-48 | Same 16; dispatch/direct median | 0.818287 versus target <0.75; 4/5 gates |
| NOR-49 | 48 untimed historical sources; selector/direct median | 0.999595 versus target <0.75; 3/5 gates |
| NOR-50 | 64 new sources; GMP/memoized on 19 repeated-R cases | 1.183263 versus target <0.90; 4/5 gates |
| NOR-51 | 48 new series; GMP/native on 16 longest cases | 2.797903 versus target <0.75; 2/5 gates |
| NOR-52 | 60 new series; compressed/full native on 16 longest sparse cases | 0.163421 versus target <0.50; 5/5 gates |
| NOR-53 | 60 new series; fast/original full GMP median | 0.143451 versus target <0.70; 5/5 gates |
| NOR-54 | 60 new series; full local pipeline/compressed native on longest sparse cases | 1.711785 versus target <1.50; 4/5 gates |
| NOR-55 | 60 new series; signed compact/signed JSON on longest sparse cases | 0.851483 versus target <0.85; 4/5 gates |
| NOR-56 | 60 new series; v2/v1 on longest sparse cases | 0.978214 versus target <0.92; 3/5 gates |
| NOR-57 | 60 frozen sources; first summarizer exception | Final CPU receipt absent; registered gates unscored |
| NOR-57B | 60 fresh series; selector/GMP workload ratio | 0.989961 versus target <0.95; 4/5 gates |
| NOR-58 | 60 new series; warm/original signed GMP workload ratio | 0.911459 versus target <0.95; 5/5 gates |
| NOR-59 | 60 new authorized programs; sparse warm/original full-verifier median | 0.959140 versus target <1.02; 5/5 gates |
| NOR-60 | 60 new authorized programs; combined/original complete workload ratio | 0.563961 versus target <0.80; 5/5 gates |

: Selected endpoints and preserved first verdicts. The linked study reports and machine-readable ledger supply all registered gates. Each row retains its own cohort and timed scope [NOR45--NOR59], [NOR60].

### 5.1 Redundant computation and packing costs

The first full trace certified exact computation, yet its 3.28 median ratio exposed the price of Python integer packing and witness handling. GMP acceleration improved the same historical reciprocal cases in NOR-47. A broader routing rule then revealed a confound: the direct program interpreter evaluated a repeated reciprocal more than once, while the trace path deduplicated transforms. NOR-50's memoized direct arm achieved a workload ratio of 0.733371 against the original interpreter. On its repeated-R stratum, GMP cost 1.183263 times the memoized direct arm, causing the registered comparative gate to fail [NOR46], [NOR47], [NOR50].

NOR-51 removed repeated-program structure by comparing single inverses. At $m=3073$, GMP witness verification/native was 2.797903 and Newton production plus verification/native was 7.439232. Both missed their targets. The five-way profile breakdown exposed coefficient-width costs and supported inputs as useful variables. NOR-52 then used the exact substitution argument from Section 3.2: its compressed-native/full-native median was 0.163421 on the longest sparse subset. NOR-53 improved the packing kernel and achieved a fast/original full-GMP median of 0.143451 [NOR51], [NOR52], [NOR53].

These experiments establish a dependence on workload structure and implementation. They provide measured reasons to retain a direct exact baseline after introducing an accelerated witness verifier. They also show why operation counts need the actual coefficient widths and representation costs attached.

### 5.2 Producer costs and limits of routing

NOR-54's fast compressed verifier passed its sparse speed gate at 0.740030 relative to compressed native computation. Including local witness production and JSON handling raised the same endpoint to 1.711785, missing the full-pipeline threshold. NOR-55 reduced the sparse signed wire/full-JSON payload median to 0.412842, while its compact/JSON verification ratio of 0.851483 narrowly missed a strict 0.85 target. NOR-56 reduced the decode phase but missed both speed thresholds; its exact-GMP phase consumed about 81% of the batch-table arm's measured phase time [NOR54], [NOR55], [NOR56].

The NOR-57B routing result made a finite stopping condition available. On its original measured source medians, choosing the cheaper of the native and GMP arms with zero dispatch cost gives

$$
\frac{\sum_j\min(t_{N,j},t_{G,j})}{\sum_j t_{G,j}}=0.964067.
$$

Its registered workload goal was below 0.95. Every sourcewise selection between those two fixed measured arms has at least the sum of their sourcewise minima. Hence the goal lay outside that finite two-backend frontier before routing overhead. This is a retrospective bound on the retained panel, and its value is to identify a mechanism that routing alone cannot supply on those measurements [NOR57B].

### 5.3 Static roots and the complete verifier

NOR-58 changed a common kernel by reusing geometry-only Merkle hashes. Its warm signed-GMP workload ratio was 0.911459, while the cold ratio was 1.022841. NOR-59 then placed the same root idea inside the complete signed program verifier. All 60 source outputs and full indexed roots agreed exactly, and all five registered engineering gates passed [NOR58], [NOR59].

| NOR-59 endpoint | Warm/original ratio | Interpretation |
|:--|--:|:--|
| 40 sparse programs, root-only median | 0.309528 | 69.05% less isolated root time |
| Same sparse programs, complete-verifier median | 0.959140 | 4.09% less median sourcewise verifier time |
| All 60 programs, sum of source medians | 0.991555 | 0.84% less time over this matched workload |
| 20 dense programs, cold/original median | 1.004549 | Cold-template construction charged per call |

: NOR-59 endpoints have different scopes and aggregation rules. The dense row compares the cold arm [NOR59].

The all-source original sum was 7,702.764146 ms and the warm sum was 7,637.712898 ms. A ten-block descriptive 95% bootstrap interval for their ratio was [0.984072, 1.000303]. Thus the observed aggregate gain was small relative to this uncertainty summary. The registered sparse whole-verifier threshold was 1.02, which permits up to 2% overhead; passing that gate alone would have established the declared overhead tolerance. The separate root-only criterion and observed values provide the evidence for the kernel gain.

The cost relationship has the form discussed by Amdahl [@amdahl1967]. If a fraction $p$ of original time receives a local ratio $r$ and additional overhead contributes $h$ in original-time units, the complete ratio is $(1-p)+pr+h$. This model explains why a strong component result can coexist with a small workload result. The paper treats it as an explanatory model; separately measured medians do not uniquely identify $p$.

## 6. Final authorized execution experiment

NOR-60 closes the engineering sequence with a factorial comparison inside the signed NOR-45 verifier. Registration commit [`363ef19abeba`](https://github.com/MoralityLabAI/Compactification-Theory/commit/363ef19abeba68e2897c593697ad56e2aaf4fefc) predates the source-generating trigger commit [`a7edfa685020`](https://github.com/MoralityLabAI/Compactification-Theory/commit/a7edfa6850206fc01a9a67650e09ddd31d198766). The first hosted attempt is run 44327994013. The protocol crosses the original versus support-compressed, locally memoized interpreter with the original versus warm static root. Every arm receives the same authenticated query, program certificate, precommit receipt, and provider reference within a repetition [NOR60REG].

| Arm | Exact interpreter | Original indexed root |
|:--|:--|:--|
| A | Original NOR-45 | Fresh complete tree |
| B | Original NOR-45 | Warm static geometry |
| C | Support-compressed, unary results reused within the call | Fresh complete tree |
| D | Support-compressed, unary results reused within the call | Warm static geometry |

The fixed panel has two observed degrees, three source strides, five program profiles, and two sign replicates, giving 60 cases. Forty have sparse support. Profile P4 requests R in each of the base, left, and right positions and adds an N suffix, giving 12 repeated-reciprocal cases. Three performance gates require the sparse C/A median below 0.70, the complete D/A workload ratio below 0.80, and the P4 C/A median below 0.65. Exact full-vector/root parity and preservation of the first evidence supply the remaining gates [NOR60REG].

Before new source generation, the archive loader checked 588 original NOR-51 through NOR-59 query vectors using a schema-independent content digest. The historical program/input audit additionally covers the 60 NOR-59 rows that contain a program. Earlier reciprocal-only records supply full query content without a program specification. New records must be unique and disjoint at each applicable level. This extends the previous record-hash exclusion to the actual full vectors; algebraic or isomorphism equivalence remains outside that exact-content test.

The first run has five interleaved repetitions for four arms on each source, for 1,200 primary timing samples. Its retained artifacts include all 60 reference vectors and 300 signed session bundles. Reference computation, signing, and the six warm template builds are recorded outside the timed calls. A separate secondary cold-root call charges template construction for each source. Its single sample per source does not enter a registered speed gate.

NOR-57's reporting failure directly shaped the recorder. An exclusively created SHA-chained journal records completed timed calls, then flushes them with `fsync` after the timer stops. A failed source retains its partial samples; a process exception retains the journal and a failure receipt. The final CPU artifact binds the journal's hash. The protocol caps the first job at 20 minutes and fixes its source count and repetitions. These rules retain the first attempt if final summarization fails [NOR60REG].

### 6.1 First registered result

All five original engineering gates passed. Every source returned the same full exact output vector and original indexed root in all four arms. Recomputing medians from the retained 1,200 primary samples reproduces the recorded aggregate values. The independent archive auditor replayed all 60 full original-interpreter vectors, checked the 300 signed bundles, and reconstructed the timing rows and 1,324 journal events. It reproduced all five gates and the descriptive bootstrap summaries [NOR60].

| Frozen gate | First observation | Verdict |
|:--|--:|:--|
| Sparse 40, C/A median <0.70 | 0.204196 | PASS |
| All 60, D/A ratio of sums <0.80 | 0.563961 | PASS |
| P4 twelve, C/A median <0.65 | 0.119726 | PASS |
| Complete signed output/root parity and source-free security controls | 60/60; preflight passed | PASS |
| First-source, historical-content, sample, code, and journal integrity | Complete first evidence retained | PASS |

: First NOR-60 engineering decisions under the frozen thresholds. The preflight recorded 43 tests. Gate status is a property of this registered experiment.

| Full signed verifier arm | Sum of source medians, ms | Ratio to A |
|:--|--:|--:|
| A: original interpreter and full root | 8,286.439526 | 1.000000 |
| B: original interpreter and warm root | 8,166.188945 | 0.985488 |
| C: compressed, locally memoized interpreter and full root | 4,751.423296 | 0.573397 |
| D: compressed, locally memoized interpreter and warm root | 4,673.229136 | 0.563961 |

: Matched NOR-60 workload totals. Every arm includes the original signatures and source/session checks. Preparation outside the call retains its separate recorded cost [NOR60].

The combined arm used 43.60% less time on this matched workload. Changing the interpreter with the original root already produced a ratio of 0.573397, while the root-only change produced 0.985488. This places most of the measured improvement in the interpreter arm for the declared panel. The descriptive ten-block 95% bootstrap interval for D/A was [0.369656, 0.797668]; the corresponding C/A interval was [0.374990, 0.815881]. Both summarize variation across the fixed degree/profile blocks on this host.

![Matched verifier cost for the four frozen arms. Bar lengths divide the sum of 60 source medians by arm A's corresponding sum. Whiskers show descriptive 95% intervals from 4,096 resamples of the ten fixed degree/profile blocks. Each median uses five timed calls. The call ends after signed execution and root verification; later protocol steps remain outside the timer.](figures/nor60_four_arms.png){width=100%}

The complete sourcewise C/A median among the 20 dense cases was 0.978561, and their workload ratio was 0.705272. Expensive repeated-reciprocal cases account for this difference in weighting. The four dense P4 cases had a C/A median of 0.377910, providing a secondary observation of local unary reuse at stride one. On the sparse cases, C/A medians were 0.236668 at stride four and 0.153228 at stride eight. The separate full-vector tests establish exactness across these strata; the measured ratios retain their finite sample scope.

![Sourcewise combined-to-original verifier ratios on the first hosted panel. Each point divides the five-repeat median for arm D by the matching median for arm A. There are 20 sources per stride, including four P4 sources marked with orange diamonds. Short horizontal bars mark each stratum's median ratio. The dashed line denotes equal verifier time on this host.](figures/nor60_source_ratios.png){width=95%}

The run recorded CPython 3.12.16 with four logical CPUs, `cryptography` 46.0.4, and `gmpy2` 2.3.2. Its processor-model field reports `x86_64`, so this artifact identifies the architecture without a processor SKU. The six one-time geometry builds totaled 8.738125 ms. The complete CPU JSON has 27,720,732 bytes, source JSON 1,021,038 bytes, and journal 898,921 bytes. Their SHA-256 digests are retained in the capstone ledger together with the original hosted artifact and run identifiers [NOR60].


The four-arm comparison separates the combined interpreter change from the root change. Support compression and unary memoization share one interpreter arm. Their individual contributions therefore require the registered structural diagnostics: source-derived reduced lengths, the dense P4 cases, and the source-free unary call-count test. The primary ratios quantify the measured combined interpreter mechanism.

## 7. Authorized Exact Gym: AEG-01

AEG-01 packages the completed verifier paths as an executable CPU environment. It supplies 14 public training tasks and 14 public development tasks. Both signed degrees, 512 and 1024, occur in each split. Seven families exercise valid input, an altered authority signature, a query substitution, a session substitution, an incorrect signed provider output, expiration, and a separately authorized program. The final family is accepted according to the program selected by the authority, making the semantic authority premise visible in the task itself [AEG01].

The Python environment has a one-step `reset`/`step` interface. A candidate chooses accept, reject, or abstain. Acceptance requires every output coefficient as a canonical decimal string and the original indexed Merkle root. The grader uses the original NOR-45 interpreter for its reference values and reports incorrect output separately from false acceptance, false rejection, and abstention. An accepted output therefore receives credit only when its full vector and root match the independently retained reference.

Public observations and referee data are exported separately, with a retained manifest that binds their hashes. The public view contains the query, trust context, and signed envelopes. Expected vectors and grading labels stay in the referee view. Strict schemas reject extra fields, repeated case identifiers, missing actions, duplicate JSON keys, and responses bound to another public input. Export and score publication use atomic writes that refuse an existing path. These checks preserve the first artifact when writers collide [AEG01].

Four fixed symbolic policies instantiate the same interpreter/root choices as NOR-60, and all four invoke the complete signed verifier. Each scored 14/14 on the public development panel, with four true acceptances and ten true rejections. Ten boundary tests also passed. These are deterministic implementation smoke results. Diagnostic elapsed time from one sequential pass describes that execution; NOR-60 owns the separately registered repeated timing comparison.

The released package creates public training and development fixtures. Its interface separates the candidate view from the grader's expected outputs, while its public deterministic generator makes the examples reproducible. It contains no sealed test split, and the recorded run evaluated no learned model. A future model study would need new content-disjoint cases, independently verified candidate/checkpoint provenance, a frozen compute budget, and referee isolation. AEG-01 itself requires CPU execution and reserves no GPU work.

## 8. Method transfer implemented in EvalMath and Control-Harness

### 8.1 EvalMath FRE-01

EvalMath PR 54 adds `spiral_ln.evaluation_frozen_evidence` at source commit [`9fc4f39e3bc4`](https://github.com/MoralityLabAI/EvalMath/commit/9fc4f39e3bc4e614a54f674e7ea2281ee0a45725). Its exact query digest and query/program task digest exclude outer experiment labels, schema wrappers, and seed descriptions. Renaming a wrapper therefore preserves the operative content identity. An adapter remains responsible for selecting every operative field, and the digest does not decide algebraic equivalence or common generative ancestry [FRE01].

The auditor receives an anchor from a separately retained record. It binds registration and source bytes, the expected first run identity, and the first artifact bytes. The first artifact digest is captured after those bytes exist; the external register supplies the evidence that registration preceded execution. The anchor also fixes arm names, repeat count, the baseline, context rules, and whether setup is included in each call. Every case/repeat/arm combination must occur exactly once, and matched arms must carry the same task and invocation-context digests.

Timing values are positive integer nanoseconds. Optional phase durations must fit within the whole call. Per-source medians and workload ratios use exact rational arithmetic. A later diagnostic binds its immediate predecessor's identifier and raw digest, while the report retains the original declared outcome and the original timing summary. A benchmark-specific grader still determines whether that declared outcome follows from its gate formulas. FRE-01 audits evidence integrity and pairing under the supplied anchor.

The source-pinned validation includes 32 adversarial tests and an eight-check deterministic receipt. All eight receipt checks passed, including rejection of changed original bytes and wrapper-renamed duplicates. The broader fast receipt replay passed 27/27 receipts. Their fixtures use synthetic timings and synthetic trusted anchors; they establish finite code behavior [FRE01RECEIPT].

### 8.2 Control-Harness adapter

Control-Harness PR 70, source commit [`07d4b3a0a1a9`](https://github.com/MoralityLabAI/Control-Harness/commit/07d4b3a0a1a9749830a4473edc6c34fa4b383437), adds an offline CLI around a byte-identical copy of the EvalMath auditor. A lock records the upstream commit, path, and full SHA-256 digest; the CLI verifies that digest before importing the module. Its report publication is atomic and refuses an existing output path [FRE01HARNESS].

Nine adapter tests cover vendor mutation, altered first evidence, omitted arms, changed contexts, exact content reuse under another wrapper, and retention of a failed original declaration. The new workflow runs the tests on Python 3.11 and 3.13. This integration audits supplied artifacts offline. Existing model controllers and reserved evaluation splits retain their own protocols.

Both patches passed their repository CI and merged through focused pull requests. Their source pins make the transferred contracts reviewable independently of subsequent repository changes. The shared method is to preserve the first evidence and its measured boundary, then report any later diagnosis with explicit lineage. Trust in a registration service, truthful measurement of clock intervals, and independence of the underlying sources require evidence beyond the artifact schema.

## 9. Scope and remaining proof obligations

The evidence covers finite synthetic integer-series programs, selected source profiles, public test signatures, and particular Python/GMP implementations. It supports exact parity and the recorded matched performance comparisons. Platform variability, coefficient growth beyond the tested range, and broader program grammars remain separate empirical questions. Successive studies share design history, so their outcomes are not independent replications of a single preregistered hypothesis.

The complete output vector is materialized and checked. Full witnesses disclose their coefficients. This construction therefore supplies a transparent execution check with explicit costs; succinct proofs and zero-knowledge properties would require a separate protocol and measurement. External semantic correctness continues to depend on how the authorized program is chosen. The dishonest-authority control is retained because it exhibits that boundary with an executable example.

The written support and Merkle arguments can guide future formal correspondence work. A complete machine-checked account would connect the concrete Python representation and signed byte encodings to the algebraic specification, prove the cache dependency rule, and state a durable session transition system. The present capstone retains those obligations in its claim table while preserving the completed experiment sequence and its reusable evaluation artifacts.

## Data and code availability

All first-run sources, receipts, reports, and auditors cited here are available in the Compactification-Theory repository. The historical paper source lock is [`714a7f5e1203`](https://github.com/MoralityLabAI/Compactification-Theory/commit/714a7f5e1203289ef053dfa5becaa3537d1b946b). NOR-60 is separately bound to registration [`363ef19abeba`](https://github.com/MoralityLabAI/Compactification-Theory/commit/363ef19abeba68e2897c593697ad56e2aaf4fefc), trigger [`a7edfa685020`](https://github.com/MoralityLabAI/Compactification-Theory/commit/a7edfa6850206fc01a9a67650e09ddd31d198766), and first run 44327994013. [`evidence_ledger.json`](https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/nor_line_capstone/evidence_ledger.json) records the study-level comparator and claim scope; `claims.json` maps the paper's claims to evidence classes. Archived audits read the original bytes and retain their gate decisions. Running an auditor is distinct from generating a new timing experiment.

## Evidence references

The study labels in the text refer to these pinned repository reports. NOR-45 and NOR-46 formal companions are separately pinned below.

| Label | Evidence |
|:--|:--|
| NOR08 | [Two-attractor and source-diversity audit][NOR08] |
| NOR15C | [Finite shared-feature policy enumeration][NOR15C] |
| NOR16 | [Public-query consensus and observation collisions][NOR16] |
| NOR42 | [Prospective paired hidden programs][NOR42] |
| NOR43 | [Challenge-index security review][NOR43] |
| NOR44 | [Session chronology and first historical receipt][NOR44] |
| NOR45 | [Program authority and exact independent execution][NOR45] |
| NOR46 | [Complete exact-integer trace][NOR46] |
| NOR47 | [GMP convolution][NOR47] |
| NOR48 | [Program-aware dispatch][NOR48] |
| NOR49 | [Reciprocal-count dispatch][NOR49] |
| NOR50 | [Memoized direct execution][NOR50] |
| NOR51 | [Single-reciprocal crossover][NOR51] |
| NOR52 | [Support-compressed inverse][NOR52] |
| NOR53 | [Fast coefficient packing][NOR53] |
| NOR54 | [Matched producer/verifier frontier][NOR54] |
| NOR55 | [Signed compact witness][NOR55] |
| NOR56 | [Batch-table codec][NOR56] |
| NOR57 | [Unscored first-run failure][NOR57] |
| NOR57B | [Fresh registered routing experiment][NOR57B] |
| NOR58 | [Static indexed Merkle templates][NOR58] |
| NOR59 | [Complete authorized program verifier][NOR59] |
| NOR60REG | [Frozen final authorized-interpreter experiment][NOR60REG] |
| NOR60 | [Original final execution run and retained evidence][NOR60] |
| AEG01 | [Authorized Exact Gym and public interface][AEG01] |
| FRE01 | [EvalMath first-evidence auditor][FRE01] |
| FRE01RECEIPT | [Eight deterministic source-pinned checks][FRE01RECEIPT] |
| FRE01HARNESS | [Control-Harness adapter and vendor lock][FRE01HARNESS] |
| LEAN45 | [Conditional authorization and sparse-rejection theorems][LEAN45] |
| LEAN46 | [Causal recurrence uniqueness and trace composition][LEAN46] |

[NOR08]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor01_numerical_orbit_sketch/NOR08.md
[NOR15C]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor15_late_source_ceiling/README.md
[NOR16]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor16_public_query_consensus/README.md
[NOR42]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor42_paired_program_ood/prospective/README.md
[NOR43]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor43_authenticated_witness/SECURITY_REVIEW.md
[NOR44]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor44_session_protocol/RESULTS.md
[NOR45]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor45_authorized_program/RESULTS.md
[NOR46]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor46_exact_integer_trace/RESULTS.md
[NOR47]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor47_gmp_exact_convolution/RESULTS.md
[NOR48]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor48_program_dispatch/RESULTS.md
[NOR49]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor49_reciprocal_count_dispatch/RESULTS.md
[NOR50]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor50_memoized_exact/RESULTS.md
[NOR51]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor51_exact_r_crossover/RESULTS.md
[NOR52]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor52_sparse_inverse/RESULTS.md
[NOR53]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor53_native_pack/RESULTS.md
[NOR54]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor54_matched_frontier/RESULTS.md
[NOR55]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor55_compact_witness/RESULTS.md
[NOR56]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor56_batch_codec/RESULTS.md
[NOR57]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor57_authenticated_router/RESULTS.md
[NOR57B]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor57b_independent/RESULTS.md
[NOR58]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor58_static_merkle/RESULTS.md
[NOR59]: https://github.com/MoralityLabAI/Compactification-Theory/blob/714a7f5e1203289ef053dfa5becaa3537d1b946b/research/nor59_authorized_merkle/RESULTS.md
[NOR45--NOR59]: ../../research/nor_line_capstone/evidence_ledger.json
[LEAN45]: https://github.com/patrickdugan/MathCan/blob/17e9c92dbf48724786cdd6a2df2645b9ff7ac111/MathCan/NOR45AuthorizedProgram.lean
[LEAN46]: https://github.com/patrickdugan/MathCan/blob/9b2e9f5a258b2a0bfa31489333e1e5b9b86b7566/MathCan/NOR46ExactIntegerTrace.lean

[NOR60]: https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44327994013
[NOR60REG]: https://github.com/MoralityLabAI/Compactification-Theory/blob/363ef19abeba68e2897c593697ad56e2aaf4fefc/research/nor60_authorized_interpreter/REGISTRATION.md
[AEG01]: https://github.com/MoralityLabAI/Compactification-Theory/tree/main/research/authorized_exact_gym
[FRE01]: https://github.com/MoralityLabAI/EvalMath/blob/9fc4f39e3bc4e614a54f674e7ea2281ee0a45725/paper/frozen_evidence_audit.md
[FRE01RECEIPT]: https://github.com/MoralityLabAI/EvalMath/blob/9fc4f39e3bc4e614a54f674e7ea2281ee0a45725/output/evaluation_frozen_evidence/fre01_v1.json
[FRE01HARNESS]: https://github.com/MoralityLabAI/Control-Harness/blob/07d4b3a0a1a9749830a4473edc6c34fa4b383437/docs/frozen_evidence_FRE01.md

## References
