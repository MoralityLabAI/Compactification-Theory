# RRC-17: probe-conditioned semantic quotient of finite algebraic graph programs

## Exact task

Fix a coefficient horizon `T=12` and the public operator alphabet `I,N,S,R` from RRC-11. For a unit series `F(q)` and a topology `w` from fork, cascade, normalized additive join, or Hadamard join, a syntax program is `(a,b,c)` with three unary tokens. There are 4³ = 64 programs per topology. The prior experiment RRC-16 supplies the *complete* integer interpreter for all four wirings, including the held-out Hadamard operation. We reuse its weights unchanged.

The RRC-17 search method groups programs by a **randomized finite-field execution signature**, then issues exact integer checks for one cheap representative per group. This is a compression of *candidate programs*, with a shared public interpreter; it is not a learned model, a new formal-series identity, or an offline-free compression method.

## Construction

Draw four independent vectors `F=(1,f₁,…,f₁₂)` whose nonconstant coefficients are uniform in the prime field `Fp`, `p=2³¹−1`. For each wiring and syntax program, compute its first 13 output coefficients using the exact integer interpreter and reduce them mod `p`. The concatenated outputs form a signature `sig_w(P)`. Programs with identical signatures form one candidate class. Within each class choose the interpreter-cost-minimizing syntax as representative, breaking ties by the established DSL order.

The frozen construction produces:

| Wiring | Program spellings | Probe classes | Twelve largest classes' syntax mass / 64 |
|---|---:|---:|---:|
| Fork | 64 | 25 | 40/64 |
| Cascade | 64 | 24 | 49/64 |
| Additive | 64 | 33 | 31/64 |
| Hadamard | 64 | 24 | 46/64 |

The grouping is independently tested with two new batches of finite-field probes and three signed-integer source series per topology. No differing outputs were found within the inferred groups on those controls. These tests are evidence of correct grouping on sampled inputs; the code **does not derive a universal formal equivalence certificate** for each class.

A reason for genuine algebraic aliases is the commutativity of the binary joins, combined with identities such as

`N(A ⊙ B) = N(A) ⊙ B = A ⊙ N(B)`

for Hadamard coefficient multiplication. The original DSL also contains syntactic redundancies from reciprocal involution and sign/substitution interactions. The quotient counts are conditional on the declared finite grammar and wiring semantics.

## Probability and exactness qualification

For normalized input `F=(1,f₁,…,f_T)`, the degree-`n` coefficient of any graph in these four topologies is a polynomial in `f₁,…,f_n` of total degree at most `2n`. This follows inductively from the triangular reciprocal recurrence, multiplication and normalized addition. Hadamard multiplication can multiply two degree-`n` coefficient polynomials, doubling the total-degree bound; subsequent reciprocal preserves the `2n` upper envelope. Consequently any difference polynomial in the observed degree-`T` vectors has total degree at most `2T=24`.

**Conditional bound.** If two programs differ as polynomial maps *over `Fp`* on the declared degree horizon, Schwartz–Zippel bounds the chance of a false match on one independent uniform probe by `24/p`. Four independent probes give at most `(24/p)^4`. A union bound across `4 × C(64,2) = 8,064` syntax pairs gives an upper bound about **1.26 × 10⁻²⁸** for a false merged pair under those finite-field hypotheses. This bound does not cover two distinct integer-polynomial maps that become identical after reduction mod `p`, nor does it prove equality of integer formal series at unbounded degree. The empirical stress probes and exact final verifier serve as separate safeguards.

Every **accepted** graph is independently checked by executing its representative on both public demonstrations through degree 12 with exact integers. This proves finite-prefix agreement for the chosen candidate, even if a probe quotient were erroneous. Discarding a candidate because of false grouping can reduce recall; that risk is expressly outside any unconditional completeness claim. On all RRC-17 registered positive cases, a complete quotient search and full 64-program search each accept the same number of examples.

## Why mass-order is optimal only for a restricted oracle

Assume hidden graph syntax is uniform on the 64 spellings, quotient classes are genuine functional-equivalence classes, and a query reveals **only whether the tested class is correct**. Testing a class `C` has prior success mass `|C|/64`. After a negative outcome the remaining mass is unchanged up to renormalization; any `B` distinct class tests therefore cover at most the sum of the `B` largest class masses. The `mass` policy attains this bound under that restricted Boolean-observation model.

The real interpreter supplies richer residual statistics. A learned adaptive policy could exploit those residuals to rank remaining classes, so the top-`B` prior mass is **not an upper bound** on general active search. Moreover, some graph classes can coincide on a particular observed input despite differing generically, so actual source-specific acceptance can exceed the uniform class-prior estimate. The cost-normalized `mass_per_work` order is a heuristic, not an exact theorem for unequal query costs.

## Resource boundary

Four probes × 64 programs × four topologies cost **1,024 generic-source graph executions**, representing **143,104 nominal integer-loop operations** under the existing RRC-16 cost model. This offline construction occurs once per interpreter revision and is shared across subsequent episodes; on the 368-case primary suite it amortizes to about **389 nominal loop units per case**, in addition to actual exact verification queries. Reusing the same built quotient across the 384-case independent-seed checks lowers the amortized amount over 752 total evaluated cases, but the offline cost never disappears. Each test-time query executes a complete program on both observed demonstrations, requires eight public primitive calls, and is charged according to its graph and topology.

The 12-query quotient method stores the table of 64 graph spellings, the signatures/class partition, the ordering, the interpreter, and source observations, beyond the six syntax bits needed to name one graph. The untrained quotient policy uses zero neural MACs. The frozen RRC-16 recurrent comparator still requires its 7,497 trainable parameters plus the 6,148-parameter initializer and their prior supervised training resources.

Finite pair collisions changing a future coefficient at degree 15 produce identical demonstrations and query for each pair. Every deterministic algorithm, including the complete exact solver, follows the same inference path for both members. A forced single future prediction can match at most one of them. RRC-17 keeps eight such pairs as an explicit future-sufficiency limitation.
