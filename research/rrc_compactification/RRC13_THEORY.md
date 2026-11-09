# RRC-13: causal partial-program feedback and finite graph certificates

## The finite algebraic task

Fix \(T=12\), \(H=32\), and formal series \(F\in1+q\mathbb Z[[q]]\).
The candidate graph is

\[
\mathcal G_{a,b,c}(F)=c\bigl(a(F)\cdot b(F)\bigr),
\qquad a,b,c\in\{I,N,S,R\}.
\]

The unary operations are identity, \(N(F)(q)=F(-q)\),
\(S(F)(q)=F(q^2)\), and reciprocal \(R(F)=1/F\).
Multiplication and inversion are exactly computed over truncated integer
power series. A graph is specified by three two-bit tokens (six syntax bits),
excluding the shared interpreter and trained controller. There are 64
syntactic possibilities and 25 equivalence classes on eight generic finite
probes (an empirical grouping, not a general canonical-form theorem).

Each case includes two observed demo pairs \((F_i,G_i)\) through degree
\(T\). The requested answer is a program matching both demos. The query
input and withheld output degrees \(13\ldots32\) are reserved for
independent grading. All candidates receive the same finite verifier.

## The exact feedback interface

After the controller emits its first token \(a\), the public simulator
executes \(\mathcal G_{a,I,I}(F_i)\). After its second token \(b\), it executes
\(\mathcal G_{a,b,I}(F_i)\). Both calls use **only the public demo inputs**
and are compared with the public outputs \(G_i\), degree by degree.
Each step returns 16 real features: on each of two demonstrations, the
first mismatch degree, fraction of matching coordinates, log residual sum,
and five signed-log early residuals. The final token \(c\) is selected without
further simulator feedback; the exact full-prefix verifier runs afterward.

The controller executes one partial program on each demonstration after
each of its first two decisions, giving four exact graph evaluations per
case. The greedy controller executes all four candidate tokens at each
of the three stages on both demonstrations, giving 24 graph evaluations.
The exact and preview baselines inspect a bounded catalogue of 64 graphs,
with all calls included in their separate resource ledger.

### A deterministic-feedback information identity

Let \(X\) collect the observed demo coefficients, \(A_{\le t}\) the
controller's emitted prefix, and \(Z_t\) the deterministic feedback.
Because \(Z_t=\Phi(X,A_{\le t})\) is computed by the public interpreter,

\[
H(Z_t\mid X,A_{\le t})=0,
\qquad I(G;Z_t\mid X,A_{\le t})=0.
\]

Thus feedback adds **computational structure** through an external algebraic
operation, rather than revealing hidden labels or additional raw
information. This distinction matters when comparing feedback against a
model trained on the same demos but unable to call the operator at inference.
The learned advantage, if present, is conditional on the supplied
interpreter, residual encoding, finite grammar and training distribution.

## Certification and impossibility

For each proposed graph \(p\), a standalone verifier executes
\(p(F_i)\) exactly through degree 12 on both public demonstrations.
Acceptance certifies these *finite equalities only*. Correct future
prediction requires equality of its executed output to a separately
held-back degree-32 target on the query input.

For any degree \(K>T\), change a future target \(Y\) into
\(\widetilde Y=Y\pm q^K\) while keeping demos and query input unchanged.
Any deterministic prefix-only constructor then returns an identical
program for both cases. If it outputs one candidate future series,
that output matches at most one member of the pair. RRC-13 includes
8 paired collisions; the complete symbolic solver accepts all 16 visible
prefixes, makes 8 future-correct predictions, and has 8 false future
acceptances. A model abstaining on both members obtains no future credit.

## Mathematical resource boundary

One graph demonstration costs four public primitive applications and is
charged \(3(T+1)^2=507\) nominal coefficient-loop units, preserving the
RRC-11/RRC-12 convention. For two demonstrations, the active learner
pays \(2\) partial queries times \(2\) executions per query, then
\(2\) final verifier executions: \(6\) graph executions, \(24\)
primitive calls and \(3042\) nominal integer-loop units. A passive model
pays only the final certificate, \(1014\) such units.

Each 6,148-parameter controller also performs an estimated 12,936 dense
floating-point multiply-accumulates per case. The estimator omits
activations, data movement and Python runtime. Integer-loop and
floating-point MAC units are **incommensurable** without a measured
hardware-specific conversion; the counts must remain separate.

Training precomputes 20 possible partial-feedback queries for each of
1,200 training cases (4 first-node choices and 16 first/second choices)
and 160 development cases, evaluating both demonstrations in each query.
These privileged candidate executions are a training-time resource.
They are absent from the two-query active inference policy. Teacher-forced
supervision supplies canonical graph tokens, and scheduled sampling
introduces partial traces from incorrect candidate prefixes.
