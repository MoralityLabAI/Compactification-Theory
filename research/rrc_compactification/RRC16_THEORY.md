# RRC-16: Exact Hadamard joins and limits of topology-conditioned graph search

## Formal-series graph language

Let \(\mathcal U=1+q\mathbb Z[[q]]\), and let \(F\in\mathcal U\) have a coefficient vector truncated through degree \(T=12\). The four unary maps remain the shared `I` (identity), `N` (\(q\mapsto-q\)), `S` (\(q\mapsto q^2\)) and `R` (formal reciprocal). A graph is three two-bit tokens \((a,b,c)\). The old topologies implement

\[
\begin{aligned}
G_{\rm fork}(F)&=c(a(F)\cdot b(F)),\\
G_{\rm cascade}(F)&=c(a(F)\cdot b(a(F))),\\
G_{\rm additive}(F)&=c(a(F)+b(F)-1).
\end{aligned}
\]

The held-out topology is the **Hadamard join**
\[
G_{\rm hadamard}(F)=c\left(a(F)\odot b(F)\right),\quad
(F\odot H)_0=1,\quad(F\odot H)_n=f_nh_n\ (n\ge1).
\]

The coefficientwise operation is bilinear on the positive-degree augmentations \(F-1\) and \(H-1\); it preserves the constant coefficient 1 and integral coefficients. It is distinct from formal-series convolution, where \((F\cdot H)_n=\sum_{k=0}^nf_kh_{n-k}\). It is a deliberately **public, hand-implemented** join during evaluation. The learned policy discovers a path through existing graph labels; it does not synthesize a Hadamard implementation.

## Exact finite interpreter and completeness

Each unary operation is causal on truncations: the first \(T+1\) output coefficients depend only on the first \(T+1\) input coefficients, including the triangular formal reciprocal recurrence. Convolution, addition, cascade and Hadamard join preserve this property. Thus the complete public interpreter produces the correct finite degree-\(T\) output for each of the \(4^3=64\) candidate graph programs. Executing all 64 programs on both demos is sound and complete for the **declared finite DSL at the observed degree**. Equivalence between different syntactic programs is allowed. A successful degree-12 check is a finite certificate; future coefficients are separately graded through degree 32.

For any selected graph, a query computes two exact candidate outputs against two public observed demonstrations, at a cost of eight primitive applications. The learned controller is allowed at most 12 distinct evaluated graphs. The 64-candidate solver is a higher-budget upper control.

## Reversible search and topology conditioning

The program graph is the Hamming graph \(K_4\square K_4\square K_4\), with nine single-token reversibly edited neighbors per vertex and diameter three. Each candidate response is cached. Parent states can be revisited to choose another neighboring graph without a new candidate execution. A priority frontier sees only residuals of **evaluated** parent states. Its best-first choice never receives the output of an unevaluated neighbor.

Every learner receives the same public three-feature topology descriptor:

| Wiring | Cross-degree join | Nonlinear join | Cascade right input |
|---|---:|---:|---:|
| Fork | 1 | 1 | 0 |
| Cascade | 1 | 1 | 1 |
| Additive | 0 | 0 | 0 |
| Hadamard, held out | 0 | 1 | 0 |

The held-out descriptor recombines properties observed separately in training. The exact graph interpreter also switches to the public held-out join for every arm, including the untrained controls. This supplies an important evaluation-time mathematical prior. An advantage here would be conditioned on both the public join definition and the shared operator grammar.

## Information and resource constraints

For public demonstrations \(X\), evaluated candidate history \(A\) and exact residual feedback \(Z=\Phi(X,A)\),
\[
H(Z\mid X,A)=0,\qquad I(G;Z\mid X,A)=0.
\]
The external interpreter adds computation rather than independent hidden information. It may make the visible demonstrations easier for a bounded model to use.

A future-collision pair can share both demos, query, topology and every deterministic feedback trace while differing at coefficient \(K=15>T\). Any deterministic policy makes the same proposal for both cases. If that proposal is accepted, exactly one member of a pair can have the correct future when the two targets differ. RRC-16 includes six such pairs. The exhaustive method accepts 12 observed prefixes and predicts six continuations correctly, while making six false future acceptances.

For degree \(T\), a simple nominal work proxy charges \(T(T+1)/2\) for each reciprocal unary, \(T+1\) for the other unaries, \((T+1)(T+2)/2\) for convolution, and \(T+1\) for additive/Hadamard joins. Every query executes both demos. This proxy excludes bit-complexity of large integers, Python object overhead, neural nonlinearities and memory access. Recurrent and feedforward compute is separately measured in dense MAC estimates. It is inappropriate to add these two units without hardware measurements.

The RRC-16 recurrent edit head has **7,497** FP32 weights, the feedforward head **7,549**, and each shares an independently frozen **6,148**-parameter RRC-13 initializer. The recurrent head carries 16 floating-point state values. Each recurrent failed-node evaluation costs 7,392 nominal dense MACs, while the feedforward costs 7,488; the initializer adds 12,936 MACs per episode. Training additionally used a privileged teacher to evaluate off-path neighbor graphs and copied previously trained RRC-14 weights into the first 132 feature columns. The three new feature columns start at zero.

This result tests a restricted graph-policy transfer, using standard exact algebra and finite search. It does not establish an infinite algebraic identity or general-purpose symbolic discovery.
