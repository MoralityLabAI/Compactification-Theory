# OVC-01: executed ordered-recurrence certificates

Date: 2026-11-14. Evidence label: EXACT_ORDERED_RECURRENCE_CONTROL.

[Theory](THEORY.md) | [Reference](reference.py) | [Tests](test_reference.py)

Verified source-code run: https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43556811425 . Head 5b186ea81f74e111285880161f27eeef1b93dbbc passed all 18 tests, issued the source-hashed JSON report, and uploaded artifact ovc01-exact-ordered-recurrence.

## Two-paper synthesis

OpenAI Math #116 supplies a claimed complete, bounded-size noncommutative formula test using explicit rational matrices. We implement its printed truncated-integration operators directly and verify their action against independent dense matrices. Source #135 supplies an asymptotic gate lower bound for exact homogeneous depth-five computation of iterated matrix multiplication.

The combined conditional statement is recorded in THEORY.md: for a target ordered matrix-product path program and candidate formula, a polynomial-dimension source probe certifies equality under #116's applicable representation assumptions. Any freely correct homogeneous depth-five presentation scalarizes to IMM, so #135 constrains its gate count. Verification cost depends on the supplied candidate's formula/path representation. This does not assert that every shared circuit can be checked in time polynomial in its DAG gate count.

## Exact compiler-mutation controls

The benchmark compares six valid ordered identities and six invalid rewrites. Every invalid rewrite is a valid identity when its variables are commuting scalars. The six invalid rewrites passed all 450 enumerated scalar assignments from {-2,-1,0,1,2}; the identities also hold for arbitrary scalar assignments by their commutative algebraic form.

A 12-coordinate rational jet probe detects every invalid rewrite. Independent exact word expansion verifies the labels, and full dense matrix evaluation checks the vector implementation.

| Invalid rewrite | First differing Taylor degree | Exact residual coefficient |
|---|---:|---:|
| AB -> BA | 3 | -1/24 |
| ABC -> CBA | 4 | -1/108 |
| A(B+C) -> (B+C)A | 3 | -17/216 |
| (A+B)^2 -> A^2+2AB+B^2 | 3 | 1/24 |
| ABA -> A^2B | 4 | 1/96 |
| AB+BC -> BA+CB | 3 | -5/108 |

All six genuine identities have zero residual and equal exact word polynomials. The positive controls cover multiplication/sum reassociation, left/right distribution, moving a scalar constant, and cancellation. Their equality is independently established by finite symbolic expansion; a zero 12-coordinate probe alone is not presented as a universal identity test.

## Sharp restricted witness for an adjacent order error

For a length-k word w=(i1,...,ik), put c_w=1/(k!*product i_j). The derivation gives

    [z^(k+1)](h_w-h_swapped) = c_w/(k+1)*(1/b-1/a)

when adjacent unequal symbols a,b are swapped. Lower coefficients cancel. Thus dimension k+2 detects every mutation in this restricted class, and the first-column interface at dimension k+1 misses it.

All 3,282 eligible swaps among words over {1,2,3}, lengths two through six, passed the exact formula. These exhaustive finite cases corroborate the supplied induction; they are not a proof of the source's universal bounded-formula theorem.

## Full source-size probes and observation scope

Three examples use the full 2*r*s^2 source formula bound for the difference expression:

| Case | Difference tree size | Variables | Jet dimension | Result |
|---|---:|---:|---:|---|
| x1 compared with itself | 5 | 1 | 50 | exact identity |
| x1*x2 compared with x2*x1 | 9 | 2 | 324 | coefficient -1/24 at degree 3 |
| Multiplication reassociation | 13 | 3 | 1014 | exact identity |

The commutator's full 324-entry residual contains numerators up to 783 bits and denominators up to 792 bits. Output-bit heights are measured separately from the number of rational slots; peak intermediate arithmetic costs can be larger.

For the truncated-integration tuple used here, all matrices are strictly lower triangular. The trace of any polynomial evaluation is its constant coefficient times the dimension. Consequently the commutator trace is zero while its first column is nonzero. Source #116's exceptional r=s=1 tuple is [1]; the trace-blindness statement concerns its nonexceptional jet construction, including every difference expression tested here.

The implementation also evaluates the nonzero word x1^5 on a five-dimensional jet and obtains zero, then detects it with dimension six. That control prevents treating an undersized zero probe as an identity certificate.

## Exact recurrence width and resource accounting

At each internal cut of ordered IMM, an explicit n-by-n identity coefficient minor forces at least n prefix/suffix polynomial summands. Splitting by the intermediate path vertex attains n. The general argument is supplied in THEORY.md; all 15 cuts for n=2,...,6 are checked independently by exact path membership.

For n sequential n-by-n scalar matrices, the dense row recurrence uses n persistent scalar state entries and another n-entry update buffer. Its standard arithmetic upper counts are n^3 multiplications and n^2(n-1) additions. Current input storage and exact scalar precision are separate costs.

| n | Persistent row scalars | Multiplications, upper count | Literal path monomials |
|---:|---:|---:|---:|
| 2 | 2 | 8 | 2 |
| 4 | 4 | 64 | 64 |
| 6 | 6 | 216 | 7,776 |
| 8 | 8 | 512 | 2,097,152 |

At n=8, that literal path expansion uses 33,554,431 binary tree gates. These counts describe a specific expansion baseline. The class-wide lower bound comes from #135, applies only for sufficiently large n, and has not been converted into a small-n numerical claim here.

The cut-width result concerns a separable algebraic interface. The source lower bound requires independent symbolic input matrices at each step. Tied-weight specializations, unrestricted nonlinear state encoders and approximate distribution-specific equivalence require separate analysis.

## Verification and formal companion

The 18-test suite independently checks the O(D) jet action against dense matrices, scalar-probe blindness, exact word expansion, degree-truncation controls, full source-dimension examples, adjacent-swap coefficients, cut minors and streamed numeric products. All arithmetic uses exact rational numbers.

[MathCan PR #37](https://github.com/patrickdugan/MathCan/pull/37), head 53eeea9f155cef0e4eb3536f155561a49792141d, passed its dedicated ordered-jet kernel/axiom workflow 43556967114 and the original mathematical checks 43556967081. Its Init-only module verifies the weighted-position numerator identity, nonzero difference for unequal symbols, and the scaled four-dimensional commutator witness with zero trace. It does not formalize the complete general source theorems or the full cut-rank proof.

No trained neural checkpoint was evaluated. The committed artifacts form a controlled exact-compiler benchmark and a proposed test contract for compact-reasoning work. Both new OVC pull requests remain drafts and unmerged; earlier NHB work remains on its separate branches.
