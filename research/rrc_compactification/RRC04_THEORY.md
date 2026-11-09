# RRC-04 exact algebra and information limits

These are elementary formal-power-series deductions, written out to delimit a
finite computational benchmark. The independent Rogers–Ramanujan identity is
already established mathematics; finite coefficient checks are not proofs of it.

## 1. Bounded-multiplicity Euler coordinates

Fix a periodic bit mask `A(n) ∈ {0,1}` on the positive integers and a uniform
multiplicity cap `c ∈ {1,2}`, or `c=∞`. Define the partition generating series

\[
P_{c,A}(q)=\prod_{n\ge 1:A(n)=1}\left(1+q^n+\cdots+q^{cn}\right).
\]

The factor is `(1-q^((c+1)n))/(1-q^n)` for finite `c`. For `c=∞` it
is `(1-q^n)^{-1}`. Consequently the unique formal Euler exponents in

\[
P_{c,A}(q)=\prod_{n\ge 1}(1-q^n)^{-e_n}
\]

are

\[
e_n = A(n) - \mathbf 1_{(c+1)\mid n}\,A\!\left(\frac{n}{c+1}\right)
\quad(c<\infty),\qquad e_n=A(n)\quad(c=\infty).
\]

The apparent negative exponents for bounded multiplicity are expected.
There can be cancellation among numerator and denominator factors.

**Proof.** Substitute the finite geometric-series identity and collect
identical factors `(1-q^m)`. The denominator contributes `+A(m)` and the
numerator contributes `−A(m/(c+1))` when divisible. Every coefficient of
a formal product through degree `T` depends only on factors of degree at
most `T`; the calculation is valid coefficientwise. QED.

## 2. Constructive exact synthesis through degree T

Given integer coefficients `f_0=1,f_1,...,f_T`, use their formal logarithmic
derivative to compute the Euler exponents `e_n` uniquely:

\[
 n f_n=\sum_{k=1}^{n} b_k f_{n-k},\qquad
 b_n=\sum_{d\mid n} d e_d.
\]

The RRC-01 integer solver implements the resulting divisor recurrence.
For each proposed cap `c`, recover membership in increasing degree:

\[
A(n)=e_n+\mathbf 1_{(c+1)\mid n} A(n/(c+1))
\]

for finite `c`, with `A(n)=e_n` for the unlimited case. Reject a cap
immediately when any forced membership is outside `{0,1}`.

For each period `m≤M`, a periodic mask exists precisely when the known
membership values agree for all observed indices in every residue class.
If `T≥2M`, each residue occurs at least twice, ruling out a spurious period
that merely allocates one mask bit per observed part. Extract the forced
mask, then independently execute its finite polynomial product through
`T`. Choose the shortest code under the declared fixed grammar encoding.

**Completeness within the declared class.** If an input prefix really is
produced by a period-`m≤M`, cap-`c∈{∞,1,2}` product, its Euler transform
is uniquely fixed through degree `T`. The above recurrence recovers that
same membership. Its period is accepted, so enumeration produces that
program or an equivalent shorter candidate. Conversely, every emitted
program has been directly executed and agrees at each observed degree.
This is a *prefix-class* completeness and soundness statement. It makes
no claim about arbitrary generating functions, the hidden source grammar,
or behavior beyond `T` without the class hypothesis. QED.

A basic implementation needs `O(T²)` integer operations for the naive
Euler inverse and at most `|C| M T` small membership comparisons;
independent verification of each surviving candidate adds polynomial-DP
work. Bit complexity grows with coefficient magnitude. The program
representation has `2+gamma_bits(m)+m` syntax bits with
`gamma_bits(m)=2 floor(log2(m))+1`. Its common interpreter, oracle,
training data and neural parameters are separate expenses.

## 3. Genuine equivalence of different programs

Distinct-part partitions with every part allowed have

\[
\prod_{n\ge1}(1+q^n)
=\prod_{n\ge1}\frac{1-q^{2n}}{1-q^n}
=\prod_{n\ge1,\;n\mathrm{\ odd}}\frac{1}{1-q^n}.
\]

The even factors cancel. Hence `cap=1, period=1, mask={0}` and
`cap=∞, period=2, mask={1}` are *different programs with identical
coefficients at all degrees*. A benchmark that insists on recovering the
historical source-program label would penalize an exact mathematical
solution. RRC-04 instead scores executable forecast equivalence, with
canonical minimum-code labels used exclusively for supervised training.

Likewise the two Rogers–Ramanujan identities translate the gap-two local
families (minimum parts one and two) to period-five unlimited-part masks.
Those facts are used as published positive controls for the held-out
local-gap generator.

## 4. Long-prefix collision impossibility

Let `F(q)` be any generating function with `f_0=1`, fix `K>T`, and set

\[
G(q)=F(q)/(1-q^K).
\]

For every `n<K`, the coefficients of `G` and `F` coincide. At degree
`K`, the coefficient of `G` exceeds that of `F` by `f_0=1`.
Thus a deterministic estimator observing only `f_0,...,f_T` has identical
input under both hypotheses. The same applies to a randomized estimator's
output *distribution*. If the estimator must choose one complete
coefficient vector through `H≥K`, at most one of the two hypotheses can
receive an exact full-vector prediction from each deterministic output.
For a uniform matched pair its maximum expected exact success is `1/2`,
regardless of model size or recurrent depth. For eight disjoint pairs the
forced-prediction ceiling is eight correct out of sixteen; abstention
can reduce false claims at the price of coverage.

An exact prefix certificate attests `P_pred ≡ F (mod q^(T+1))`.
It cannot attest equality of the full infinite generating functions
without an additional structural assumption about the source.
