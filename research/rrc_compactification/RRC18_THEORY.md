# RRC-18: Certified finite graph quotient and learned class induction

## Exact algebraic statement

Let \(R=\mathbb Z[x_1,\ldots,x_{12}]\) and let
\(X(q)=1+\sum_{n=1}^{12}x_nq^n\in R[q]/(q^{13})\).
Fix a topology \(\tau\) from fork, cascade, additive, or Hadamard and a
three-operation graph \(p=(a,b,c)\in\{I,N,S,R\}^3\). Here `I` is the
identity, `N` substitutes \(-q\), `S` substitutes \(q^2\), and `R` is the
formal reciprocal of a series with constant term 1. The wiring means

\[
\begin{aligned}
G_{\rm fork}(X)&=c(a(X)b(X)),\\
G_{\rm cascade}(X)&=c(a(X)b(a(X))),\\
G_{\rm additive}(X)&=c(a(X)+b(X)-1),\\
G_{\rm hadamard}(X)&=c(a(X)\odot b(X)),
\end{aligned}
\]
where \((U\odot V)_0=1\) and \((U\odot V)_n=u_nv_n\) for positive degrees.

**Finite-degree equivalence theorem.** Two graph programs \(p,p'\)
produce equal coefficients through degree 12 for every integral unit series
\(F(q)\in1+q\mathbb Z[[q]]\) if and only if the twelve corresponding
coefficient polynomials of \(G_\tau(X)\) are identical in \(R\).

**Proof.** All four unary transformations and all four join implementations
are defined by exact polynomial operations on the first 13 input
coefficients. For the reciprocal, the triangular recurrence
\(r_n=-\sum_{k=1}^nf_kr_{n-k}\), with \(r_0=1\), constructs a polynomial in
\(x_1,\ldots,x_n\). Structural induction on the graph gives a unique
coefficient polynomial for each output degree. Identical polynomials
agree under every integer substitution \(x_i=f_i\), proving sufficiency.
Conversely, a nonzero integer multivariate polynomial cannot vanish on
all \(\mathbb Z^{12}\). Thus different coefficient-polynomial vectors admit
an integral input on which an observed coefficient differs. ∎

The `certify18.py` checker implements this proof by **exact sparse polynomial
normalization**, compares canonical dictionaries, and saves every graph's
class assignment. Its correctness depends on the code implementing the
stated ring operations. This is a machine-checkable finite algebraic audit,
not a Lean/kernel formalization and not a proof of equivalence at every
unbounded degree.

## Packed integer monomials and arithmetic exactness

The polynomial dictionaries store a monomial \(x_1^{e_1}\cdots x_{12}^{e_{12}}\)
as the integer \(\sum_{i=1}^{12}e_i2^{5(i-1)}\). Adding these integers
multiplies monomials. There is no carry in the exponent fields for this
interpreter: an ordinary fork, cascade or additive coefficient at degree
\(n\) has weighted input degree at most \(n\), where \(\deg x_j=j\).
The Hadamard join multiplies two degree-\(n\) polynomials and raises
weighted input degree to at most \(2n\); its outer reciprocal retains
that weighted degree. Hence \(e_i\le24\) for every coefficient in
\(q^{12}\), below the five-bit capacity 32. Coefficients remain unbounded
exact Python integers. The tests also evaluate each symbolic graph on
independently generated integral inputs and compare with the separate
concrete interpreter for all 256 topology/program combinations.

The resulting finite partition has:

| Wiring | Distinct classes | Graph spellings | Accumulated normal-form monomial entries |
|---|---:|---:|---:|
| Fork | 25 | 64 | 8,959 |
| Cascade | 24 | 64 | 5,425 |
| Additive | 33 | 64 | 9,172 |
| Hadamard | 24 | 64 | 25,000 |

The entries column sums the sizes of generated coefficient dictionaries
across all programs and degrees. It is a *representation-size proxy*, not
a count of individual integer additions/multiplications. All exact
partitions agree with the earlier RRC-17 finite-field probe partitions.
RRC-17 needed a probabilistic separation argument; the degree-12
membership assertions here have no random false-collision probability.

## Supervised class learner and independent verifier

A source program and a topology determine one of the exact classes. For
supervision, choose the minimum-work representative of that class, with
canonical syntax order as tie-breaker. The learner receives two observed
input/output examples \((F_i,G_i)\) through degree 12. Its model inputs
are normalized raw integer coefficients and a public topology one-hot;
the exact group calculation supplies its training label but no inference
feature. A GRU reads 13 coefficient positions and emits 64 class-code
logits; the topology mask retains only valid certified representatives.
A feedforward model receives the same 52 normalized coefficient values
and the four topology indicators. Both return a fixed ranked list without
executing candidate graphs.

After ranking, an *independent exact checker* executes each selected
representative on both observed demos and accepts the first matching
program. Hence each accepted graph satisfies exactly the 26 public
coefficient equalities (13 per demo). The strictly later 13–32 coefficients
of a fresh query are graded independently. A top-12 result is a finite
conditional certificate under a public, fixed graph DSL; models are allowed
to abstain by exhausting the 12-query budget. The class predictor's
ability to rank candidates says nothing about equivalence past degree 12.

## Collision impossibility and resource separation

Fix all observed coefficients, query input and topology. Alter one query
target coefficient at degree \(K\in\{13,\ldots,32\}\). Any deterministic
prefix-only learner, including one using deterministic exact graph queries,
produces identical candidates and a single prediction on the two cases.
At most one of their distinct future targets can match it. Eight
base/mutant pairs operationalize this bound.

Each learned proposal pays one dense model inference and every verifier
candidate pays two exact degree-12 graph executions. The recurrence
cost proxy is 53,216 forward MACs, versus 6,120 for feedforward.
For a selected class \(p\), an exact query is charged
\(2\,\mathrm{graph\_cost}(p,\tau)\) nominal coefficient-loop units.
Offline exact polynomial normalization, generating teacher labels,
training passes, class-table bytes, the interpreter, model weights, output
coefficients and memory must be counted in addition to syntax bits.
These resource categories have different computational units.
