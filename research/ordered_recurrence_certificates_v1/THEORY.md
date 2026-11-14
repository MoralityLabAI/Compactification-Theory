# OVC-01: ordered verification and recurrence compilation

2026-11-14. Source pair: OpenAI Math #116 and #135. The deductions here are applications and finite refinements of those source statements. Priority or novelty relative to the broader literature has not been established. All general source conclusions remain conditional on their original proofs; our code checks finite constructions independently.

## Sources and proof boundaries

**#116, One Rational Matrix Hitting Point for Noncommutative Formulas**, September 24, 2026. The source introduction states that, for r variables and binary formula-tree size s (including leaves), a deterministic construction supplies rational matrices of dimension at most 2*r*s^2 that detect every nonzero division-free noncommutative formula in that class over a characteristic-zero field. The matrix-generation cost and entry heights are polynomial in r,s. The scope note formally selects the universal hitting property, while explicitly excluding a separate formal assertion of the construction-time and matrix-dimension bounds.

The source's construction.tex also gives an acyclic path-program version: with m states, dimension D=(r-1)*binomial(m,2)+m suffices. Arbitrary shared multiplication circuits are outside that result's stated model. Formula trees, arithmetic branching programs (ABPs) and arbitrary circuit DAGs require distinct size accounting.

Scope: https://github.com/openai/math/blob/main/lean/docs/116.md
Source: https://github.com/openai/math/blob/main/preprints/One-Rational-Matrix-Hitting-Point-for-Noncommutative-Formulas-September-24-2026/build/introduction.tex
Construction: https://github.com/openai/math/blob/main/preprints/One-Rational-Matrix-Hitting-Point-for-Noncommutative-Formulas-September-24-2026/build/construction.tex
Read blob SHAs: scope 1adf3a79084ab4062a5556e0c9bbc7fec5d04897; introduction bbbfbd3ff0bf7ed812cf20d23734ed9dac4bc4d6; construction 0abc841a6a0263f050c93ea926cc6c0cece05e53.

**#135, Homogeneous depth-five lower bounds for iterated matrix multiplication**, September 25, 2026. For the (1,1) entry IMM_(n,n) of a product of n independent n-by-n variable matrices, the source claims at least n^(sqrt(n)/400) gates for every syntactically homogeneous Sigma-Pi-Sigma-Pi-Sigma circuit over characteristic zero, for all sufficiently large n. Its upper bound is n^(sqrt(n)+4). Sharing, arbitrary finite fan-in/fan-out and unrestricted bottom linear-form support are allowed. These are exact symbolic computation bounds under a specified algebraic architecture.

Scope: https://github.com/openai/math/blob/main/lean/docs/135.md . Read blob SHA: 1657acd351d3597af491bfbbaf83b1e9571da761.

Neither source has been fully re-proved by this campaign. We do not infer a numerical lower bound at our small test dimensions, a bound for approximate distillation, or a restriction on general neural-network architectures.

## OVC-1: execute the source hitting tuple without allocating dense matrices

For D Taylor coefficients, the source matrices are

    T_i[r,c] = (-1)^(r-c-1)/(r*i^(r-c))  if 0<=c<r<D,
               0                         otherwise,

with variable index i>=1. They implement J_i(a)=integral_0^z a(t)/(t+i) dt. Let e0 represent the constant series 1. For word w=(i1,...,ik), the first output column is the truncated series

    T_i1 ... T_ik e0 = J_i1 ... J_ik(1).

Multiplication order is essential: the rightmost operator acts first on the column. To apply one T_i to a coefficient vector a, form the quotient series b=a/(z+i):

    b_(-1)=0,
    b_j=(a_j-b_(j-1))/i,
    (T_i a)_0=0,
    (T_i a)_(j+1)=b_j/(j+1).

This requires O(D) rational arithmetic operations and one output vector, rather than allocating D^2 entries. Induction from (z+i)b=a proves the recurrence. Formula evaluation recursively applies the right child before the left child at a multiplication node and adds output vectors at a sum node. It uses O(sD) rational operations for a binary tree of size s; exact numerator/denominator lengths remain an additional cost. Our implementation records output bit heights.

A nonzero computed column supplies an unconditional exact counterexample to a proposed free-algebra identity. A zero column becomes a universal conclusion only with an appropriate degree/size promise and an applicable completeness theorem. The code labels source-dependent completeness separately from the finite word-expansion oracle.

## OVC-2: a verification/compilation consequence of the two sources

Use r=n^3 noncommuting entry variables z_(t,i,j). Define the ordered path polynomial

    P_n = sum_(j1,...,j_(n-1))
          z_(1,0,j1) z_(2,j1,j2) ... z_(n,j_(n-1),0).

It is represented by a layered acyclic path program with m0=n(n-1)+2 states. For scalar entries, its commutativization is IMM_(n,n). A forward row recurrence computes the scalar specialization using n state registers, O(n^3) arithmetic operations and an additional output buffer of n registers. Reading a resident input matrix costs another n^2 scalars unless its entries can be streamed; exact scalar precision is not constant-bit storage.

Let Q be a proposed division-free ordered formula with binary tree size s. The source formula-to-path conversion uses at most 2s states. Direct-summing its strictly upper-triangular representation with the target representation yields P_n-Q on at most

    m=n(n-1)+2+2s

states. Conditional on #116's path-program proposition, a single explicit tuple at

    D=(n^3-1)*binomial(m,2)+m

has zero first-column residual exactly when Q=P_n in the free associative algebra. This polynomial-size verifier avoids explicitly expanding P_n's n^(n-1) paths. The target column can be evaluated backward through the layers using O(nD) rational workspace and O(n^3 D) arithmetic operations. Candidate evaluation adds its own stack and bit costs. The finite code compares this recurrence with independent word expansion.

If the candidate has an exact syntactically homogeneous depth-five presentation with g gates, commutativization preserves that presentation and computes IMM_(n,n). Consequently #135 implies

    g >= n^(sqrt(n)/400)

for sufficiently large n, under its stated characteristic-zero model. Verification can therefore be polynomial in the size of a supplied candidate while that candidate's shallow presentation requires superpolynomial size in n.

The compilation contract here is ordered free-algebra correctness, which is stronger than equality only after commuting scalar entries. We do not apply the noncommutative verifier to arbitrary commutative rewrites. A large shared circuit may require a much larger formula tree; a DAG gate count cannot silently replace s. For arbitrary-fan-in formula presentations, binary expansion is charged in s even though it changes the depth used for verification. The original presentation is the one to which #135 is applied.

## OVC-3: scalar probes and trace summaries miss operator order

Every scalar substitution makes AB and BA equal. Matrix substitutions need not. With A=E12 and B=E21,

    AB=diag(1,0),   BA=diag(0,1).

The (1,1) output changes from 1 to 0 under the swap. Their traces agree, so that scalar summary misses the error as well.

For the actual source tuple the blindness is stronger. Every T_i is strictly lower triangular. Every positive-degree word in the tuple is therefore strictly lower triangular. For any polynomial f,

    trace(f(T)) = D * coefficient_of_empty_word(f).

Thus a trace-only observation discards every positive-degree component of the source probe, even at a dimension large enough for the full matrix/first-column guarantee. The benchmark explicitly checks this mismatch between a discriminating probe and an insufficient observation interface.

## OVC-4: a sharp small witness for one adjacent swap

For positive integer indices w=(i1,...,ik), let h_w=J_i1 ... J_ik(1). Induction on k gives

    h_w(z) = c_w z^k + d_w z^(k+1) + O(z^(k+2)),
    c_w = 1/(k! * product_j i_j),
    d_w = -c_w/(k+1) * sum_(j=1..k) (k-j+1)/i_j.

The leading coefficient depends only on the multiset of symbols. Suppose w' swaps adjacent unequal symbols a,b. The product and c_w stay fixed, and the two position weights differ by one. Therefore

    coefficient_(k+1)(h_w-h_w')
      = c_w/(k+1) * (1/b - 1/a),

which is nonzero. All coefficients through degree k cancel. A jet with coordinates 0 through k+1, hence dimension k+2, detects every single unequal adjacent swap of a length-k word. Dimension k+1 cannot detect these swaps in this first-column jet interface.

This is a restricted-mutation guarantee with an elementary proof, not the universal formula guarantee of #116. Multiple coordinated swaps and linear combinations can cancel this coefficient and require deeper observations. The source-derived full size bound remains a separate fallback.

The commutator x1*x2-x2*x1 has first nonzero coefficient -1/24 at Taylor degree three. The tests enumerate all words over {1,2,3} of lengths two through six and all unequal adjacent swaps: 3,282 exact certificates.

## OVC-5: an independent, complete degree-bounded word oracle

Take a basis e_w indexed by every word of length at most ell over r variables. Define S_i e_w=e_(i,w) when length(w)<ell, and zero otherwise. Then for every polynomial of degree at most ell,

    f(S)e_empty = sum_w coefficient_w(f) e_w.

The representation is faithful on that finite degree class and has dimension 1+r+...+r^ell. The code implements its sparse action independently of both dense matrix multiplication and the source jet recurrence. It is exponential in ell and makes no claim to reproduce #116's polynomial-size construction proof.

Truncation must remain in the certificate: x1^D is nonzero but evaluates to zero on every D-dimensional strictly lower-triangular source matrix. A zero short-probe output never establishes an unqualified identity.

## OVC-6: exact width at a prefix/suffix cut

Fix an internal layer cut k. Separate P_n's variables into prefix layers 1..k and suffix layers k+1..n. For each intermediate vertex j,

    P_n = sum_(j=0..n-1) L_j(prefix) R_j(suffix),

so n separable prefix/suffix summands suffice. Form the coefficient matrix whose rows are prefix words and columns suffix words. Choose one prefix ending at each j and one suffix starting at each j. Their n-by-n coefficient submatrix is the identity, because only matching intermediate vertices form a valid path. Its rank is n.

Any decomposition into m prefix-polynomial times suffix-polynomial summands gives a coefficient matrix of rank at most m. Therefore

    minimum exact separable cut width = n.

The statement holds in this linear/bilinear algebraic interface. It is not a lower bound on arbitrary nonlinear encoders with unlimited real precision. The supplied finite oracle checks all 15 internal cuts for n=2,...,6 and independently checks numeric streaming products against path enumeration.

## Resource comparison and experiment contract

The frozen benchmark has six valid ordered rewrites and six invalid scalar-valid rewrites, including commutation, word reversal, collecting repeated letters, and a commutative square expansion. Scalar probes enumerate {-2,-1,0,1,2} assignments. Twelve-dimensional source jets are compared to an exact word oracle on every case. Three additional examples use the full 2*r*s^2 formula bound, with dimensions 50,324 and 1014. These are finite checks of the construction, not a reproof of its general completeness.

At n=8, the dense row algorithm uses at most 512 scalar multiplications and 448 additions with 8 persistent state scalars plus an 8-scalar update buffer. Literal path expansion has 2,097,152 monomials and 33,554,431 binary formula-tree gates. Those expansion counts describe that construction; they are not a lower bound on every small-n circuit. The asymptotic lower bound imported from #135 has an unspecified sufficiently-large-n threshold.

For looped-transformer or compact-reasoner applications, use this as a controlled algebraic compiler task: distinguish safe reassociation from unsafe operator permutation, charge the verifier's observation interface, and measure the proposed readout's architecture. Nonlinear activations, finite-precision approximation, tied-input restrictions and distribution-specific equivalence require their own specification. No trained neural checkpoint or production BitVM verifier is evaluated in this round. Existing campaigns and pull requests remain unchanged.
