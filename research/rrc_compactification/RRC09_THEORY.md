# RRC-09: finite operator-program composition and its information limits

## Formal-series program language

Let \(\mathcal F_T=\{f=(f_0,\ldots,f_T)\in\mathbb Z^{T+1}:f_0=1\}\). For any member of \(\mathcal F_T\), define five total exact-integer operators, truncated to degree \(T\):

- `N`: substitute \(q\mapsto -q\), multiplying coefficient \(f_n\) by \((-1)^n\).
- `S`: substitute \(q\mapsto q^2\), placing \(f_j\) at degree \(2j\) and zero at odd degrees.
- `R`: formal reciprocal \(F\mapsto1/F\), with \(r_0=1\) and \(r_n=-\sum_{k=1}^n f_k r_{n-k}\).
- `C`: convolution square \(F\mapsto F^2\), with \(c_n=\sum_{k=0}^n f_k f_{n-k}\).
- `D`: Euler/divisor inverse. Put \(D(F)_0=1\) and \(D(F)_n=a_n\), where \(qF'/F=\sum b_nq^n\), \(b_n=n f_n-\sum_{k=1}^{n-1} b_kf_{n-k}\), and \(a_n=(b_n-\sum_{d\mid n,d<n}d a_d)/n\). Integrality follows from Euler's factorization theorem over \(1+q\mathbb Z[[q]]\).

`C` and `D` are both **excluded from all supervised training programs**. They remain available as *public, hand-written* operations inside the exact bounded solver and execution-guided candidate generator at test time. This matters: their test-time implementations provide semantic information that the direct learned program decoder does not have.

A program has zero, one or two operators selected from the five-token alphabet. Consequently, the unrestricted class has exactly \(1+5+25=31\) syntax programs. Equivalent syntax is permitted. The enumeration order is shortest first, then alphabet order. A two-slot fixed code over a six-symbol alphabet (five operators plus a padding/end token) needs six bits, excluding the interpreter, coefficient data and every model parameter. Program descriptions are distinct from operational costs.

## Complete bounded finite search

Fix observed input \(f_0,\dots,f_T\) and output \(g_0,\dots,g_T\). Evaluate each grammar program \(p\) in order. Its degree-six preview is a *necessary condition* for equality to degree \(T=18\). If the preview matches, evaluate the same program at degree 18 and compare all coordinates. Return the first matching program or abstain after 31 programs. The algorithm is sound and complete **for the declared finite grammar and coefficient horizon**, because each candidate is computed exactly and every legal program is tested. This also establishes why a perfect exact solver is the appropriate in-class comparator. A program matching 18 coefficients is a finite certificate, rather than a theorem of equality for every coefficient.

The execution-guided learner carries out all 31 degree-six previews, ranks candidates by a trained six-weight logistic score on preview residuals, then submits its first \(k\in\{1,4,8,31\}\) candidates to the *same* exact degree-18 verifier. An untrained ranking by first mismatch and prefix residual is a fairer comparison than the direct neural decoder alone: it sees the same 31 interpreter outputs and consumes the same verifier budget. Model ranking may reorder candidate verification but supplies no new executable primitive.

The direct recurrent and feedforward decoders see paired raw coefficient prefixes only. They receive neither `C`/`D` execution results nor the true operator syntax at test time. Training labels are canonical matching programs in the smaller grammar over `N/S/R`.

## Finite-prefix impossibility and absent-primitive uncertainty

Let \(K>T\), and suppose the observed target is \(G(q)\). Construct a mutant \(\tilde G(q)=G(q)+q^K\) (or \(G(q)-q^K\)). Both have the same first \(T+1\) coefficients, so *every* deterministic prefix-only method, exact or learned, must issue the same answer. Its one fixed future prediction can agree with at most one full series when the base and mutant differ. RRC-09 contains eight such pairs. In this roster the enumerator finds a class program for each observed prefix and makes exactly eight incorrect future predictions. This is an information limit, independent of neural architecture.

A second identification limit concerns previously unseen operations. A training corpus containing only programs over `N/S/R` is invariant under any change of the meaning of the unused `C/D` labels. Therefore, training examples alone cannot uniquely specify those two operators. A direct decoder can receive additional program semantics or demonstrations at test time, or use a prior imposed by its designer. RRC-09's execution-guided arm does exactly the former by exposing the full interpreter. There is no assumption that a categorical classifier should discover an unseen operation's meaning from its absence.

## Work accounting

An operator call performs finite exact integer work. `N` and `S` cost \(O(T)\) coefficient operations; `R`, `C`, and `D` each have nominal \(O(T^2)\) work at their current straightforward implementations. The receipt counts each primitive invocation and an explicit *nominal coefficient-loop estimate* at the degree executed. These counts exclude big-integer bit complexity, fitting the models, generating the target samples, Python overhead, candidate sorting and independent benchmark scoring. Measured bytes also exclude the shared compiler and verifier.

The greedy exact baseline can short-circuit after low-degree failure. The guided methods first execute all candidates through degree six, even if they later submit just one or four programs. A lower number of full verifier calls may coexist with higher total integer work. The neural models' FP32 parameters, recurrent hidden state and repeated sweeps remain separate resources.
