# The cost of a Schelling point: finite counting bound

**Claim grade:** exact finite mathematics under a **synthetic uniform prior**, verified by standard-library exhaustive toy tests. This is a separate layer from the source-coded religion comparison.

Let \(X\) be uniform on a finite set \(\Omega\) of \(M\) complete, distinct label assignments, and let \(f:\Omega\rightarrow Y\) be a summary taking at most \(k\) distinct values. Suppose a decoder \(g:Y\rightarrow\Omega\) guesses the exact label assignment after seeing only \(f(X)\). Then

\[
\Pr[g(f(X))=X] \leq \min(1,k/M).
\]

**Proof.** Every nonempty fiber \(f^{-1}(y)\) has at most one source state \(x\) for which \(g(y)=x\). There are at most \(k\) distinct output fibers, so at most \(k\) of the \(M\) equiprobable source states can be guessed correctly. The bound is sharp when exactly \(k\) nonempty fibers occur. \(\square\)

For a *constant* summary, \(k=1\), the exact recovery probability is \(1/M\); moreover \(H(X\mid f(X))=\log_2 M\). For a general finite deterministic summary with nonempty fiber sizes \(n_y\),

\[
H(X\mid f(X))=\sum_y \frac{n_y}{M}\log_2 n_y.
\]

In a deliberately synthetic model of six independent binary labels, \(M=2^6=64\), and keeping only the common three-stage path erases six bits, leaving optimal exact recovery at \(1/64\). This numerical result depends entirely on the specified prior and toy ontology. The six categorical axes in the real SCH-01 fixture have different alphabets, and their distribution is unknown.

**Decision corollary.** If two states \(x_1,x_2\) have identical summary \(f(x_1)=f(x_2)\) but require different unique valid decisions \(a_1\ne a_2\), any deterministic decision rule based only on \(f\) must choose incorrectly for at least one of the states. This is the same loss-of-decision-relevant-information phenomenon addressed in interface-relative compactification work.

**Tests:** \(2^4=16\) possible two-bucket summaries of four input states are exhaustively checked. The six-binary-axis constant-summary calculation and mixed-fiber entropy identity are checked separately. No model inference and no theological probabilities are involved.
