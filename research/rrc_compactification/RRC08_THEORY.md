# RRC-08: algebraic operator identification and finite certificates

## The question

Can eight real coefficients learned from short exact supervision specify a *reusable
integer recurrence* that produces Euler exponents from previously unseen raw
integer generating functions? The recurrence vocabulary is public and hand-built.
This is operator-coefficient induction, **not** discovery of the vocabulary.

Let \(F(q)=\sum_{n\ge 0}f_nq^n\in\mathbb Z[[q]]\) with \(f_0=1\).
Define \(B(q)=qF'(q)/F(q)=\sum_{n\ge1}b_nq^n\). Coefficient comparison gives

\[
b_n=n f_n-\sum_{k=1}^{n-1}b_k f_{n-k}.\tag{1}
\]

The formal Euler product \(F(q)=\prod_{m\ge1}(1-q^m)^{-a_m}\) gives

\[
b_n=\sum_{d\mid n}d a_d,\qquad
 a_n=\frac{b_n-\sum_{d\mid n,d<n}d a_d}{n}.\tag{2}
\]

Both are triangular identities over formal power series; they require
neither numerical convergence nor a product-type source distribution. The
student's public *operator basis* comprises `n*f[n]`, a prefix convolution,
previous `b`, a constant, `b[n]/n`, a proper-divisor sum, previous `a`, and a
constant. It learns eight real weights. Their fixed snap-to-integer compiler
produces a small program in the bounded signed-integer DSL. The recovered
weights are `(1,-1,0,0; 1,-1,0,0)`, reproducing (1) and (2) with a causal
scan. The exact source coefficients are the only inference-time data.

## Finite exactness proof

By induction on \(n\), (1) computes the unique \(b_n\) from earlier \(b_k\).
Equation (2) computes the unique \(a_n\) from earlier \(a_d\). Thus the
compiled exact-integer serial interpreter computes the formal Euler transform
on every normalized integral input for which the division is integral.
Integral Euler coefficients exist for every \(F\in1+q\mathbb Z[[q]]\):
inductively cancel the degree-\(n\) coefficient using an integer power of
\((1-q^n)\), leaving the lower coefficients intact. The independent exact
checker rejects nonintegral proposals and evaluates the full coefficient
residual \(qF'-F\sum_{n\ge1}(\sum_{d\mid n}d a_d)q^n\).

**The key qualification:** recovering an exact law follows directly from the
supplied basis plus successful coefficient fitting. The interpreter does the
arithmetic. Eight learned weights alone do not encode the divisor graph, the
raw series, the interpreter, the training samples, or the output integers.

## Simultaneous update depth

Let \(\hat b^{(0)}=\hat a^{(0)}=0\), and evaluate both equations in parallel
using only the **previous** pass state, with the new `a[n]` taking its `b[n]`
from the previous pass. Induction gives that \(\hat b_n\) is exact after at
most \(n\) passes. Once it is exact, another pass supplies the correct
\(a_n\), since proper divisors \(d<n\) already have their values. Thus
at most \(T+1\) passes suffice for all degrees \(n\le T\). For \(T=64\),
we test depths `0/1/2/4/8/16/32/65`. This is a deterministic propagation
bound inherited from the *hand-built* recursion graph. It does not measure
a neural network's independently learned reasoning depth.

The serial topological scan reaches the same result in one ascending-degree
execution, with \(O(T^2)\) integer operations for the convolution. It is a
stronger resource baseline than 65 simultaneous sweeps.

## Identification, generalization, and information limits

Four independent training rows per head are sufficient to identify four real
linear weights in the absence of noise. In the actual frozen dataset the
`3072 × 4` matrices both have full rank, with measured condition numbers
4.189 and 4.979. Local teacher traces are exact and supervision is confined
to degrees 1–16. An Adam-trained eight-parameter linear student fits those
rows and snaps to the unique small-integer operator. This is teacher-forced
system identification using a privileged intermediate representation.

The local feedforward control also has eight fitted real weights, and sees
neighboring raw coefficients without convolution or divisor memory. It is
parameter-matched but intentionally **operator-capability-mismatched**. It is
an ablation of algebraic structure, rather than a test of equally expressive
recurrent and feedforward universal approximators.

A late defect changes \(F(q)\) to \(F(q)/(1-q^K)\) for \(K>T\). Every
coefficient through degree \(T\) is unchanged. Therefore every deterministic
prefix-only student issues the same candidate for the base and mutant. If
both complete vectors differ, at most one can match a single proposed future
series. In this evaluation there are six such pairs, and the compiled
operator's valid finite-prefix programs make six future errors. Finite
verification certifies agreement only through the observed degree.

## Accounting

The compiled operator stores eight coefficients in `[-2,2]`. A fixed-width
three-bit code per coefficient uses 24 **syntax-only** bits, excluding the
operator vocabulary and shared decoder. The fitting checkpoint contains
eight FP64 weights (512 bare weight bits) and additional metadata. The
feedforward model also fits eight FP64 weights. Training traces, source
code, coefficient storage, exact verifier, division/convolution routines,
parallel states, CPU time and the independent exact solver are separate
resources. No compression ratio for the entire task is inferred.

The formulas above are classical Euler-transform/Newton-identity algebra.
The experiment's contribution is a tested learning interface and an explicit
information/resource boundary, rather than a new mathematical identity.
