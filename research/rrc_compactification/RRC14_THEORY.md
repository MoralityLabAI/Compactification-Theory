# RRC-14: reversible graph edits, exact feedback and bounded identification

## Formal task

For `T=12` and unit formal series `F(q) in 1+q Z[[q]]`, an algebraic
program is a graph `G=(a,b,c)` implementing

`G(F) = c(a(F) * b(F))`,

where the four supplied unary operations are `I(F)=F`, `N(F)=F(-q)`,
`S(F)=F(q^2)` and `R(F)=1/F`. Multiplication and reciprocal are truncated
integer coefficient recurrences. Programs receive two input/output
coefficient demonstrations through degree 12, followed by a new query
series whose degree-13–32 output is held out.

The 64 syntactic programs form the Hamming graph `K_4 □ K_4 □ K_4` under
single-slot edits. Every vertex has exactly `3*3=9` neighbors; every edge
has a unique reverse; the diameter is three. The numbers of vertices at
edit distances zero, one, two and three from any graph are respectively
`1, 9, 27, 27`. Thus every program is reachable by at most three
single-slot edits *when the correct changes are known*. Choosing them
from coefficient evidence is the tested search problem.

Several syntax graphs implement the same function, including exchanging
left and right before the commutative product. We grade **any** graph
whose executed coefficients agree with the public demonstrations and
whose forecast equals the held-out continuation. Recovering an arbitrary
latent syntax label is unnecessary. The 25 semantic classes in the
RRC-12 design are empirical classes over generic finite probes, not a
universal algebraic classification.

## Computational feedback and finite exactness

An attempted edit chooses one complete graph. The verifier executes it
on both *public* degree-12 demonstrations, obtaining exactly 16 feedback
features (eight per input/output pair), including first mismatch and
bounded residual summaries. A candidate query uses **two** demonstrations,
with **eight** public primitive calls. One query charges
`2*3*(12+1)^2 = 1,014` nominal integer-loop operations under the
RRC-11 accounting convention. A model may evaluate up to 12 distinct
graphs per episode. The completed graph is accepted only after exact
coefficient equality on both public demonstrations. The search does not
get to evaluate another uncharged full graph after the budget expires.

For public examples `X` and selected edit history `A`, feedback is a
specified deterministic function `Z=Phi(X,A)`. Hence

`H(Z | X,A)=0` and `I(hidden_graph ; Z | X,A)=0`.

Active feedback supplies computational work from an interpreter and can
transform information into an easier representation for a limited model.
It supplies no previously hidden source-program labels. The graph-edit
policy has no access to the held-out series or future coefficients.

When a graph is found, exact equality through degree 12 is a finite
certificate. It neither proves equality of infinite series nor
certifies forecasts through degree 32.

## Reversible frontier invariant

The policy starts at the frozen feedback-disabled RRC-13 proposal, then
maintains a priority queue of **unevaluated** single-slot neighbors of
already evaluated graphs. Each queue element stores its parent graph and
recurrent hidden state. Revisiting a parent is a cache restoration,
charged zero further interpreter executions. A newly evaluated vertex is
charged one exact two-demo query. A visited set prohibits duplicate
candidate executions. Every stored edge is in the undirected Hamming
edit graph, so restoring a parent and trying another child implements
reversible search. Cross-frontier restores are counted separately as
backtracking/nonlocal transitions. The model never previews an unevaluated
neighbor's output when deciding to prioritize it.

At 12 queries the policy certifies no more than 12 distinct attempted
graphs. The observed residuals may help choose subsequent attempts, so
`12/64` is **not** an information-theoretic bound on attainable success.
The exact 64-candidate solver is the complete reference for this bounded
program DSL at the observed horizon. It uses short-circuiting of failed
candidates and charges first- and second-demo executions separately.

## Paired future-collision theorem

Take any two public demonstration pairs and one future query `Q`, with
observed degree `T=12`. For a graph `G`, construct a future mutant by
changing `G(Q)` at a degree `K>T` while leaving the demonstrations and
query unchanged. Every deterministic policy reading only that visible
state, including its adaptive interpreter calls, issues the same graph
sequence for both cases. If the mutant differs from the base, at most one
of the two futures can match the single predicted output. RRC-14 includes
eight such base/mutant pairs, changing degree 15. Exhaustive search
accepts all 16 observed prefixes and forecasts exactly eight correctly.
Policies that abstain on both pairs earn no forecast credit.

## Resource and inference boundaries

One trained recurrent edit head has 7,353 FP32 weights; the
parameter-matched feedforward policy has 7,393. The live recurrent state
contains 16 FP32 values, or 512 bare bits. All edit arms also use the
previously trained 6,148-parameter RRC-13 initializer, whose
feedback-disabled inference costs 12,936 nominal FP multiply-accumulates
and zero interpreter calls. A recurrent failed-candidate update counts
`3*16*(132+16)+9*16 = 7,248` dense MACs, and a feedforward update
counts `132*52+52*9 = 7,332` MACs. Interpreter loop units and model MACs
are kept separate. Float nonlinearities, Python object overhead,
heap operations, full integer bit complexity, training, interpreter code
storage, and stored raw series are excluded from these arithmetic proxies.

A 12-visit frontier can store up to 12 returned 16-coordinate feedback
vectors and up to 108 candidate edges, each with a copy/reference to
its parent recurrent state. The ledger counts source checkpoint weights,
query executions and dense MACs, but does **not** pretend the graph's
six-bit syntax code includes this frontier, weights, or interpreter.

The training teacher executes all nine neighbors of each supervised
nonterminal state on familiar graph classes. The saved checkpoint records
**35,631 teacher neighbor queries** (71,262 two-demo graph executions),
in addition to state evaluations and development-label generation. This
privileged training computation must be considered before any efficiency
comparison against the exact solver.

Classical finite-graph search and formal-series identities explain the
mathematics here. The experiment measures a particular policy under a
fixed budget, not a universal property of recurrent reasoning.
