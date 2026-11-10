# RRC-15: graph-topology transfer and algebraic equivalence

## Two unseen interpreter wirings

Let \(\mathcal U=1+q\mathbb Z[[q]]\), under the usual formal-series product. The frozen graph language has 64 syntactic triples \((a,b,c)\in\{I,N,S,R\}^3\), with
\(I(F)=F\), \(N(F)(q)=F(-q)\), \(S(F)(q)=F(q^2)\), and \(R(F)=F^{-1}\).

The previously trained RRC-14 program means

\[\mathrm{fork}_{a,b,c}(F)=c(a(F)\,b(F)).\]

RRC-15 keeps exactly the same six-bit three-token program space and reversible nine-neighbor Hamming edit graph. The fixed *execution graph* changes to either

\[\mathrm{cascade}_{a,b,c}(F)=c(a(F)\,b(a(F))),\]

or, with \(X\oplus Y=X+Y-1\),

\[\mathrm{add}_{a,b,c}(F)=c(a(F)\oplus b(F)).\]

Every expression lies in \(\mathcal U\), as the constant coefficient is one. The integer formal reciprocal is computed by its standard triangular coefficient recurrence. Thus all three programs have exact finite coefficient semantics through the declared horizon, even when their coefficients are negative.

### Why old-graph aliases matter

The maps \(N\) and \(S\) preserve multiplication, while \(R\) sends a product to the product of its reciprocals because \(\mathcal U\) is abelian. Their compositions therefore act as group endomorphisms. Hence

\[\mathrm{cascade}_{a,b,c}(F)=(ca)(F)\cdot(cba)(F),\]

where juxtaposition of operator symbols means composition. The older fork grammar likewise represents \((ca)(F)\cdot(cb)(F)\). Certain cascade triples therefore have equivalent old-fork programs, while combinations needing additional nontrivial compositions escape the three-token fork class. This is the reason to audit *functional equivalence on the observed examples* instead of grading hidden syntax labels.

The additive join has different algebra. For example \(\mathrm{add}_{I,I,I}(F)=2F-1\), which does not preserve products: with \(F=1+uq\), \(G=1+vq\), the coefficient of \(q^2\) in \(2FG-1\) is \(2uv\); in \((2F-1)(2G-1)\) it is \(4uv\). Since the old fork maps are multiplicative group endomorphisms, this additive-join map cannot belong to that old function class. The finite experiment checks equivalence directly on two demonstrations rather than relying solely on this example.

## A bounded search guarantee

The edit neighborhood is the undirected Hamming graph \(K_4\square K_4\square K_4\), with 64 vertices and nine single-token edits per vertex. All edits have legal inverses. A budget \(B\) counts up to \(B\) **unique full graph queries**; each evaluates the selected candidate on both public degree-12 input/output demonstrations, and a vertex cache prevents repeated evaluations. Previously evaluated candidates may contribute residuals to subsequent decisions. The recurrent policy uses its restored parent hidden state when another frontier branch is explored; the reset ablation erases that state.

The exact bounded reference enumerates all 64 candidates against the same two observations. It is complete for the declared six-bit DSL and degree-12 agreement only. The 12-query frozen controllers can use results of previous queries to choose the next candidate. The fraction 12/64 consequently is *not* a general success ceiling for adaptive search. The complete exact oracle exceeds the registered query budget and must be labeled accordingly.

The public demonstration vector \(X\), current graph \(A\), and feedback \(Z=\Phi(X,A)\) satisfy \(H(Z\mid X,A)=0\): an exact evaluation supplies computation, not previously concealed semantic labels. The topology choice is public to the interpreter. Frozen model weights receive its consequences through observations and residuals; no architecture or learned weight is changed to encode the new topology.

## Finite-prefix collisions

For a new query input \(Q\), take a valid target \(Y=G(Q)\). Define a mutant target \(\widetilde Y=Y\pm q^{15}\). These share all public degree-12 demonstrations and the query input. Every deterministic prefix-only policy issues the same sequence of candidate graph evaluations and the same proposal in both worlds. When it predicts a unique future, its output can match at most one member. Twelve RRC-15 pairs (six cascade and six additive) check this. Full exact search accepts both members' observed demonstrations, then makes exactly one future error in each pair.

## Resource accounting and limitations

A candidate query executes its graph on two demonstrations. It charges eight public primitive applications, including the join. The nominal integer coefficient-loop proxy charges each unit-cost unary operation \(I,N,S\) at \(T+1\), reciprocal \(R\) at \(T(T+1)/2\), product at \((T+1)(T+2)/2\), and the additive join at \(T+1\), with \(T=12\). Summing the operations per candidate and multiplying by two charges both demos, including failed candidates. This proxy omits integer bit complexity, Python overhead, memory traffic and independent future grading.

RRC-14's recurrent edit checkpoint retains 7,353 FP32 parameters, and its feedforward control 7,393. All edit arms also use the unchanged 6,148-parameter RRC-13 initializer. Recurrent/FF dense multiply–accumulate estimates, actual candidate queries, nominal integer work and source parameter bytes remain separate metrics; they cannot be summed as a hardware-independent scalar. The published RRC-14 training used 35,631 privileged neighbor queries and 420 optimizer updates. RRC-15 trains **zero** new parameters. The result is a frozen zero-shot topology-transfer probe and a finite verification result, not a new infinite identity or a universal claim about neural graph search.
