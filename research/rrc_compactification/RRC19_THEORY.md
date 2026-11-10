# RRC-19: Family-weighted graph identification and finite certificates

## Task and original mathematical quotient

Consider the truncated group of integral unit formal series
\(\mathcal U_T = \{1+\sum_{n=1}^T f_nq^n:f_n\in\mathbb Z\}\), with \(T=12\).
For each public topology \(\tau\in\{\mathrm{fork},\mathrm{cascade},\mathrm{additive},\mathrm{hadamard}\}\), the exact interpreter evaluates a graph \(p=(a,b,c)\in\{I,N,S,R\}^3\). The individual operations are identity, sign substitution, degree doubling, and formal reciprocal. Binary operations are the convolution product, a cascaded convolution, normalized addition, or normalized coefficientwise multiplication.

RRC-18's exact polynomial normalizer identifies the equivalence classes \(p\sim_{\tau,T} p'\) if their output polynomials in the independent variables \(x_1,\ldots,x_T\) agree in every degree through \(T\). This is a universal statement **at degree 12**, including every integral coefficient input. It yields 25, 24, 33 and 24 classes for the four topologies. The exact class construction is a supplied, separately accounted-for backend; RRC-19 trains only candidate rankings.

An input sample comprises two public input/output demonstrations \((F_1,G_1),(F_2,G_2)\) observed through degree 12, plus the public topology. A predictor assigns scores to canonical representative graphs; at most 12 distinct candidates can then be independently executed on *both* demonstrations. Every accepted candidate has a finite, exact certificate of agreement. Degree-13 through degree-32 outputs on a fresh query are used only for grading.

## Training distribution as an explicit variable

Let \(\ell_\theta(X,Y)\) be cross-entropy against the source program's canonical exact class label, with nonrepresentative labels masked. Denote the three source families by \(s\in\{\mathrm{signed},\mathrm{product},\mathrm{sparse}\}\), and the four topologies by \(\tau\). Uniform cell sampling gives the expected objective

\[
L_{\mathrm{balanced}}(\theta)
 = \frac1{12}\sum_{s,\tau}\mathbb E[\ell_\theta(X,Y)\mid s,\tau].
\]

The skewed control has the same training roster, architecture, parameter initialization and minibatch-token budget, with family probabilities \((0.8,0.15,0.05)\) and uniform topology sampling. Its target risk is

\[
L_{\mathrm{skewed}}(\theta)
 =\sum_s \frac{p_s}{4}\sum_\tau\mathbb E[\ell_\theta(X,Y)\mid s,\tau].
\]

The two risks differ by sampling weights, which changes the finite optimization trajectory. Neither has a universal dominance theorem for a genuinely different test distribution. Class canonicalization is teacher-supplied, and the source-family name is never a neural inference feature.

## What counts as generalization

Training sources are independent signed integer series, periodic unrestricted partition-product series, and sparse signed integer series. The new generator families are (a) exact reciprocals of randomly selected finite sparse polynomials and (b) convolution squares of random sparse unit series. Both are generated using public exact arithmetic, and both are excluded from all gradient updates and development selection. The four graph topologies remain familiar.

A signed or product test is a fresh-instance test within a known source family. A rational or squared test changes the *input source generator*, while preserving the target graph language. Neither test withholds the graph interpreter or the symbolic quotient. Positive execution transfer on these groups therefore measures source-distribution robustness of a predeclared finite graph classifier.

The 12-budget ordering can be an exact representative-list prior, a learned neural ordering, or a fixed six-neural/six-class-mass union. A full class search may spend up to 33 queries, and is an **upper reference** for this finite DSL. The mass prior is a prior proportional to the number of syntax spellings in a class. It is not universally optimal, particularly when input selection and source transformations induce nonuniform semantic class frequencies.

## Exact feedback and future impossibility

The checker evaluates each selected graph on both observed prefixes. It stores finite equality decisions and charges each complete two-demo query. The model sees no outputs of untested graph candidates. The finite proof is that the interpreter is a causal, exact polynomial/triangular-series calculation through degree 12, and thus equality of all 13 coefficients on both demos holds when the check accepts. This finite certificate does not imply the hidden degree-32 continuation.

For every degree \(K>12\), a paired future mutant \(\widetilde G(Q)=G(Q)\pm q^K\) can share the same demos, query and topology. A deterministic predictor and every deterministic verifier-feedback policy has an identical trace for both records. One predicted degree-32 series can match at most one member of the pair. The RRC-19 roster has eight such pairs; complete class search necessarily has eight false future acceptances.

## Cost accounting and limits

The recurrent model has 6,240 FP32 trainable parameters, 32 live FP32 recurrent-state coordinates and an estimated 53,216 dense MACs per 13-step inference. The feedforward control has 6,235 FP32 parameters and 6,120 estimated dense MACs per inference. Each model is separately trained for 700 updates of 128 examples (89,600 training-example visits), with identical optimizer budget. Estimated *forward-only* training MACs total 14.85 billion across all four trained arms, excluding backward computation and data loading. The four old topology normalizers cost 256 offline graph normalizations, while exact candidate verifications charge actual graph executions and integer-loop cost proxies.

These counts exclude bigint bit complexity, memory traffic, nonlinearities, the Python interpreter, normal-form storage, source generation, optimizer backpropagation and the prior RRC-18 checkpoints' training costs. Integer-loop operations and floating-point MACs are separate resource dimensions. The result is a controlled **one-seed exploratory model comparison**, with two additional post-primary seed checks; it does not demonstrate an architecture-wide benefit, blind discovery of a new algebra, or an infinite identity proof.
