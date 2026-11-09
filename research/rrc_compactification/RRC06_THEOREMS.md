# RRC-06: exact series residuals and a finite-prefix information bound

## 1. Coefficient-only algebraic representation

Let \(F(q)=\sum_{n\ge0}f_nq^n\), where \(f_0=1\), and write

\[
F(q)=\prod_{m\ge1}(1-q^m)^{-a_m},\qquad
qF'(q)/F(q)=\sum_{n\ge1}b_nq^n.
\]

Formal logarithmic differentiation gives

\[
b_n=\sum_{d\mid n}d a_d,\qquad
nf_n=\sum_{k=1}^n b_k f_{n-k}.
\]

The first equation is Möbius-invertible, and the second determines \(b_n\) inductively because \(f_0=1\). Consequently the truncated vector \((f_0,\ldots,f_T)\) determines a unique truncated Euler exponent vector \((a_1,\ldots,a_T)\) over the rationals. In the registered partition families it is integral. Every neural candidate is verified against these equations **without modifying the candidate**.

## 2. First-fault certificate theorem

Given an integer candidate \(\hat a\), define its coefficient residual:

\[
R_n^a=n f_n-\sum_{k=1}^n f_{n-k}\sum_{d\mid k}d\hat a_d.
\]

Suppose \(\hat a_j=a_j\) for all \(j<m\), with \(\hat a_m=a_m+\delta\), \(\delta\ne0\). Then \(R_n^a=0\) for all \(n<m\), and

\[
\boxed{R_m^a=-m\delta}.\]

Proof: the first \(m-1\) coefficients of the proposed divisor sums agree with the exact ones. At degree \(m\), the unique changed divisor term is \(d=m\) in \(b_m\); its multiplier is \(f_0=1\). The residual is therefore \(-m\delta\). This is an exact *localization theorem* for the earliest incorrect Euler exponent. A later coefficient cannot hide the first fault.

Similarly define a proposed log-derivative residual:

\[
R_n^b=n f_n-\sum_{k=1}^n\hat b_k f_{n-k}.
\]

If the earliest error occurs at \(m\), with \(\hat b_m=b_m+\delta\), then \(R_n^b=0\) for \(n<m\) and \(R_m^b=-\delta\). Consequently a full vector of vanishing \(R^a\) and \(R^b\) is an exact certificate for the **observed truncation**, independent of how the candidate was proposed.

Both first-fault results are elementary consequences of the triangular formal-series equations. The test suite checks mutations at indices 1, 7, 31 and 64 for Euler exponents, and separately for the \(b\) trace. These are finite executable audits of the written algebra, not formal-assistant kernel proofs.

## 3. Collision bound for future prediction

Fix \(T=64\). For any generating function \(F\) and \(K>T\), let

\[
G(q)=F(q)/(1-q^K).
\]

Then \(G(q)\equiv F(q)\pmod{q^{T+1}}\), whereas \([q^K]G=[q^K]F+f_0=[q^K]F+1\). Hence any deterministic model, exact symbolic algorithm, or stochastic policy depending only on the observed prefix receives precisely the same input on both members. Under a uniform prior over a pair, a single deterministic future-vector forecast is correct for at most one. With five independent equally weighted base/mutant pairs, expected accuracy on those paired future vectors is at most 50%, even if every proposed program verifies on the prefix.

RRC-06 tests \(K\in\{67,79,101,113,127\}\). The bounded exact synthesizer accepts all ten prefixes and correctly predicts precisely the five base cases. This is a tight information bound for the registered paired roster. It does not prohibit abstention or uncertainty estimates.

## 4. Resource distinction

The recurrent model's forward pass receives raw coefficient normalizations and learned weights only. Its head proposes both \(a\) and \(b\) independently. Exact inversion is used **in training targets and in evaluation**; its arithmetic and source code remain outside the inference model. The exact checker runs \(O(T^2)\) integer arithmetic for the two residuals, plus the bounded grammar interpreter and finite product replay. Its work is charged outside neural inference and is also an independent stronger baseline.

With hidden size 23, a single GRU scan uses approximately \(65\cdot[3\cdot23\cdot(3+23)+4\cdot23]=122{,}590\) dense multiply-adds, excluding bias, nonlinearities and feature formation. The two-scan implementation is approximately 245,180 MACs. The flattened feedforward control uses \(195\cdot4+4\cdot260=1{,}820\) dense MACs per example. These are analytic operator counts, not measured hardware throughput. Parameter storage is 2,028 versus 2,084 float32 weights. Shared recurrent weights are used in every scan; weight storage does not increase with depth. Hidden state and intermediate output arrays are counted separately.

## 5. Scope of inference

Exact certificates establish the given 65 coefficients. Recovering a known partition identity requires further mathematical evidence, and a finite test does not establish agreement for all \(n\). Failed exact residuals are stronger counterevidence to the *particular proposed algebraic program* than aggregate coordinate accuracy can provide. A neural model may score well on many individual exponents while still failing at the first low-degree coefficient that matters to the formal product.
