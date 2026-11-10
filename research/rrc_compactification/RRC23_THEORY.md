# RRC-23: Degree scaling, staged exact verification and enlarged algebraic syntax

## Finite algebra and model access

Fix an integral unit formal series \(F(q)=1+\sum_{n\ge1}f_nq^n\) and the four public wiring choices from RRC-16: fork, cascade, normalized additive join, and normalized coefficientwise/Hadamard join. The original 64 syntactic programs consist of three unary tokens \(a,b,c\in\{I,N,S,R\}\). The public unary maps are identity, sign substitution \(q\mapsto-q\), degree doubling \(q\mapsto q^2\), and formal reciprocal \(F\mapsto F^{-1}\). All these maps preserve integer unit-series truncations, so exact execution through degree \(T\) is well defined using only coefficients through \(T\).

The frozen RRC-19 GRU, GRU and MLP rankings and RRC-21 source-shift calibration consume **only coefficients through degree 12**. RRC-23 changes the finite checker degree \(T\) to 12, 24, 48, 96 and 192, with a separate eight-case check at 384. The model weights, observed input width, existing degree-12 certified class list, and routing thresholds remain fixed. There are no new gradient updates.

For the larger syntax \((u,a,b,c)\), the new public interpreter evaluates \(c(a(u(F))\star b(u(F)))\), with the appropriate wiring-specific alteration for cascade; equivalently, it first applies \(u\) to the source and then executes the pre-existing three-token graph. This is **256 spellings** over the same known operator vocabulary. It supplies the composition and executable prefix semantics explicitly. The old class representative list is replicated under each \(u\); this is an empirically tested search convenience, without a certified 256-program quotient.

## Why staged verification saves work

A complete graph execution to degree \(T\) computes all coefficients even when the first distinguishing coefficient is small. Ordinary exact convolution uses \((T+1)(T+2)/2\) integer products and triangular reciprocal inversion uses \(T(T+1)/2\). One graph execution has an \(O(T^2)\) coefficient-loop bound under the implemented arithmetic.

Our progressive verifier performs exact two-demonstration checks at distinct degrees \(\{4,12,T\}\) in ascending order. It stops evaluating that candidate after the earliest failed stage. **All repeated executions are charged**. If a candidate reaches the final stage, its full-degree check is identical to the direct verifier. Therefore, for a fixed candidate ordering, progressive and direct verification accept the same first program and execute the same distinct candidate list. This follows by the causality of all permitted operations: equality to degree \(T\) implies equality to every shorter degree, so no truly matching candidate can be discarded at an earlier stage.

With \(K\) candidate attempts and \(p\) the fraction reaching the long stage, the gross coefficient work is bounded schematically by

\[
W_{\rm progressive}=O(K\cdot 4^2+K_4\cdot12^2+pK\cdot T^2),\qquad
W_{\rm direct}=O(K\cdot T^2).
\]

Here \(K_4\le K\) reaches degree 12. The empirical work ledger charges each stage using the actual graph's topology-dependent unary and join complexity rather than assuming a universal unit cost. The profiler times the uninstru­mented Python interpreter, including allocation and dispatch overhead.

## The measured routing break-even condition

Let \(R_{\rm sel}(x)\) be the detector, model inference and program-order cost, \(R_{\rm sym}(x)\) the class-order cost, and \(V_{\rm sel}(x),V_{\rm sym}(x)\) the complete exact verification path times. Selective routing is faster over a declared workload only if

\[
\sum_x [R_{\rm sel}(x)-R_{\rm sym}(x)]
<\sum_x [V_{\rm sym}(x)-V_{\rm sel}(x)].
\]

At degree 192, the 84-case experiment saves 81 *candidate* attempts and about 3.2% of the nominal progressive verifier-work proxy, while adding model/detector execution. Its measured median end-to-end latency is 4.606 ms, compared with 3.302 ms for progressive symbolic class search. On eight stress cases at degree 384, those medians are 21.209 and 17.209 ms. **The measured crossover is absent through the tested range.** A higher-degree or compiled-model crossover remains an open empirical question. Symbolic class compilation, historical model training and calibration creation, cold model loading, and independent scoring are accounted for separately.

## Three kinds of equality and ambiguity

1. **Graph class identity:** RRC-18 proves equality of the relevant sparse integer coefficient polynomials through degree 12 for every integer input. RRC-23 checks 600 additional class-member comparisons per tested horizon using independent input series. Their agreement is evidence about these finite samples, rather than a theorem at arbitrary degree.
2. **Observed-prefix program agreement:** Distinct graph classes can produce identical outputs on a particular pair of demonstrations through degree \(T\). A class ordering or 64-program syntax ordering may select a different valid prefix program. Both are finite certificates at the observed degree, even when their query predictions disagree later.
3. **Query future agreement:** If a hidden query output is altered at a degree \(K>T\), every policy that uses the same demonstrations, query input and deterministic public interpreter feedback takes identical actions for the base and mutant. Any fixed nonabstaining prediction matches at most one of two differing continuations. This is the ordinary finite-prefix information bound; a successful prefix check never proves the complete infinite identity.

The 84-case finite test has complete observed-prefix coverage from the old certified representative ordering at every horizon, and exhaustive 64-syntax coverage too. Their degree-12 *future* correct counts differ (82 versus 81). In the four-token test at degree 12 the larger full syntax list and replicated shorter quotient also accept all 40 examples, with distinct future-correct totals (36 and 38). The difference is selection of distinguishable programs that happen to match the limited public data; there is no evidence here that the original certified classes split universally above degree 12.

## Resources and scope

The 12-check selector uses frozen RRC-19 predictor weights and RRC-21 calibration, including a known-reference nearest-neighbor detector. The class-size list uses an exact degree-12 polynomial compiler and no neural inference. Every candidate check executes *both* public demonstrations, and the progressive test's extra short-prefix executions are metered separately. Classical exact search through all 64 or 256 syntax programs is a higher-resource class-complete reference. Training examples, model weights, calibration references, executable interpreters, class tables, input coefficients, and finite traces are distinct memory/compute resources.

The physical figures come from one Linux x86-64 host using Python 3.13.5 and an AMD EPYC 9V74. They quantify these *Python implementations* and one deterministic set of cases. The byte-for-byte deterministic receipt records semantic outcomes and operation proxies, while physical timing and RSS records carry source hashes without CI latency thresholds. The expanded 40-case grammar result and the 8-case grammar timing subset have separate sample sizes; the degree-384 sample also contains only eight fresh cases.
