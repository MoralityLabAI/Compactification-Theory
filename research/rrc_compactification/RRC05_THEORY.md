# RRC-05: formal Euler inversion as finite-depth recurrence

## Exact triangular structure

For a formal series \(F(q)=1+\sum_{n\ge1}f_nq^n\) with integral coefficients,
write

\[
qF'(q)/F(q)=\sum_{n\ge1}b_nq^n,\qquad
F(q)=\prod_{n\ge1}(1-q^n)^{-a_n}.
\]

The two formal identities give, without analytical convergence assumptions,

\[
n f_n=\sum_{k=1}^{n}b_kf_{n-k},\qquad
b_n=\sum_{d\mid n}d a_d.
\]

The first equation determines each integer \(b_n\) from the observed prefix
\(f_0,\ldots,f_n\). The second is a triangular system over the divisor order.
For \(n=1,\ldots,N\), define \(c_n=b_n/n\) and
\(P_{nd}=(d/n)\mathbf1\{d\mid n, d<n\}\). Thus \(c=(I+P)a\).
Every nonzero entry of \(P^t\) encodes a strictly increasing divisor chain
\(d_0\mid d_1\mid\cdots\mid d_t\) with each ratio at least two. It follows that
\(P^t=0\) when \(2^t>N\). Starting from zero, the parallel recurrence

\[
a^{(0)}=0,\qquad a^{(t+1)}=c-Pa^{(t)}
\]

therefore satisfies \(a^{(t)}=\sum_{j=0}^{t-1}(-P)^j c\) and reaches the *unique
exact Euler-exponent vector* after at most

\[
T=1+\lfloor\log_2N\rfloor.
\]

For \(N=64\), seven simultaneous iterations suffice. This is a finite
algebraic identity and a tight uniform depth bound for the chosen Jacobi
scheme: a length-six divisor chain \(1,2,4,8,16,32,64\) can propagate a
nonzero correction through all seven layers. A sequential forward substitution
or Möbius inversion solves the same system directly. The recurrence provides
fixed-depth parallel execution with shared rules; it does not lower the
asymptotic cost of a complete exact solver.

## Learned perturbation and exact certification

The trained circuit modifies the known graph update using **two learned
scalars** \(\alpha,\beta\):

\[
x^{(t+1)}=x^{(t)}+\alpha\bigl(c-x^{(t)}-\beta P x^{(t)}\bigr).
\]

The special point \(\alpha=\beta=1\) is the exact finite-step algorithm.
For a general trained pair the linear error matrix is
\((1-\alpha)I-\alpha\beta P\); because it is triangular, its eigenvalues
all equal \(1-\alpha\). Hence \(0<\alpha<2\) gives asymptotic stability,
although finite-depth integer-rounding errors can remain and the value of
\(\beta\) controls the limiting bias. At \(\alpha\ne1\), seven passes
need not produce exact values even if the iteration is stable.

At evaluation the student proposes integer \(\hat a_n=\mathrm{round}(x_n)\).
Two *independent exact-integer* residuals are recorded:

\[
R^{\rm div}_n=b_n-\sum_{d\mid n}d\hat a_d,
\qquad
R^{\rm series}_n=n f_n-\sum_{k=1}^n f_{n-k}
                     \sum_{d\mid k}d\hat a_d.
\]

Both vanish through \(N\) precisely when the proposed Euler exponents
match the observed series through \(N\). The candidate product grammar
\((\mathrm{cap},\mathrm{period},\mathrm{residues})\) is extracted only from
\(\hat a\) via the registered bounded-multiplicity inversion, then **directly
executed** to check the original observed prefix. Any missing certificate
requires abstention. Future coefficients are withheld from this acceptance
check. A valid finite-degree certificate cannot prove an infinite identity.

## Observational collision limit

If two source series have the same coefficients for all \(0\le n\le N\),
then every deterministic procedure using only that prefix, including the
b-transform, neural recurrence, exact solver and finite grammar checks,
has the same output for both. If their future coefficients disagree and
both appear in a paired test, at most one can have the entire future
vector predicted correctly. This bound applies regardless of model size.

## Resource boundaries

The learned model uses two FP32 scalar parameters (64 bare parameter bits),
with a 65-value live exponent vector at \(N=64\). Its total computation
also includes an **exact integer quadratic log-derivative preprocessor**, a
hard-coded divisor graph with all proper-divisor edges, a fixed grammar
interpreter, and exact candidate verification. The feedforward control has
two trained scalars and sees the same \(b_n/n\) but performs a single
coordinatewise pass. The historical RRC-04 models instead see normalized
raw coefficients and emit full programs with a different parameter budget.
Comparing these paths gives a resource decomposition, not an architecture
win under matched information.
