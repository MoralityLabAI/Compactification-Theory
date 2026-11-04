# QO-03: Multi-action rate–distortion, conflict graphs, and an NP-complete boundary

**Date:** 2026-11-04. **Evidence:** explicit elementary reductions and finite exact-rational witnesses. The NP-completeness conclusion uses the established NP-completeness of graph 3-colorability (Garey, Johnson, Stockmeyer, 1976). This is a theorem about the **declared restricted Markov aggregation problem**, not a complexity theorem for all neural representations or a novelty claim about graph coloring.

**Related results:** [QO-01](GOAL_AWARE_RATE_DISTORTION.md) gives general finite exhaustive search with a goal-readout contract. [QO-02](HIT_RATE_DP.md) gives an \`O(n²k)\` optimum for a **single** hit-probability coordinate. [IDQ-02](MINIMAX_THEOREMS.md) establishes the fibre-wise minimax TV radius of a proposed quotient.

## 1. Model: absorbing-goal actions with independent rate coordinates

Let \(S=\{1,\ldots,n\}\cup\{\dagger\}\). The goal \(\dagger\) is absorbing. For each action \(a\) and transient state \(i\), the controlled transition kernel satisfies

\[
K_a(i,\dagger)=p_i^a,\qquad K_a(i,i)=1-p_i^a,\qquad
K_a(\dagger,\dagger)=1,
\]

where \(p_i^a\in[0,1]\cap\mathbb Q\) and other transition probabilities vanish. An optional \`idle\` action has all \(p_i^{\text{idle}}=0\). Require an **exact Boolean macro goal-readout**: \(\dagger\) has a distinct macrostate from every transient state. Let \(\pi\) partition the transients into \(k\) or fewer classes. The macro kernel \(L_a\) is chosen independently for each action and macrostate, with no restriction to deterministic transitions.

For any transient block \(B\), the projected successor law of member \(i\in B\) is supported on \(B\) and \(\dagger\):

\[
\pi_*K_a(i)=(1-p_i^a)\delta_B+p_i^a\delta_\dagger.
\]

Since total variation between two such laws is \(|p_i^a-p_j^a|\), the Chebyshev-optimal center is their midpoint in hit probability. Thus the exact **uniform one-step** autonomy defect of the partition is

\[
\boxed{\epsilon(\pi)=
\frac12\max_{\substack{B\in\pi\\ a\in A}}
\Bigl(\max_{i\in B}p_i^a-\min_{i\in B}p_i^a\Bigr).}
\tag{1}
\]

This formula is conditional on the absorbing-goal/self-loop kernel structure and an exact goal partition. It is not a general bound for arbitrary controlled Markov chains.

## 2. QO-03/T1: conflict-graph characterization

Define the rate vector \(v_i=(p_i^a)_{a\in A}\) and the \(L^\infty\) pseudometric
\[
d_\infty(i,j)=\max_a|p_i^a-p_j^a|.
\]
For tolerance \(\tau\ge0\), construct a simple *incompatibility graph* \(H_\tau\) on the transient states:
\[
\{i,j\}\in E(H_\tau)\quad\Longleftrightarrow\quad d_\infty(i,j)>2\tau.
\tag{2}
\]

**Theorem T1.** There exists an exact-goal-readout quotient with at most \(k\) transient macrostates and autonomy defect at most \(\tau\) **if and only if** \(H_\tau\) admits a proper \(k\)-coloring.

**Proof.** A macro partition is a coloring of its transient states. Equation (1) shows that \(\epsilon(\pi)\le\tau\) precisely when every pair of states placed in one class has \(d_\infty(i,j)\le2\tau\). By (2), that is exactly the requirement that each class be an independent set in \(H_\tau\), which is equivalent to proper coloring. Conversely, a proper coloring gives pairwise bounded rate ranges in every class, and midpoint macro kernels attain the stated bound. QED.

**Finite candidate principle.** For a fixed instance, the optimum belongs to
\[
\left\{0\right\}\cup
\left\{\frac12d_\infty(i,j):i\ne j\right\}.
\tag{3}
\]
All possible partition scores have this form. Enumerating distinct candidate values and checking graph colorability therefore computes the exact optimum. This gives an alternate certificate algorithm, albeit with exponential graph-coloring worst-case complexity.

**Two-coordinate obstruction.** Four transient rates at the corners of the square,
\[
v_0=(0,0),\ v_1=(0,1),\ v_2=(1,0),\ v_3=(1,1),
\]
admit zero error with two transient classes when either coordinate is evaluated alone. With both actions and only two transient classes, the optimum error is \(1/2\). Every distinct pair differs by 1 in some coordinate, and exact zero defect requires four transient classes. This is an explicit failure of choosing a compression separately for each action and then assuming the result will satisfy the combined interface.

## 3. QO-03/T2: fixed two-bit budget yields NP-completeness

Define the following restricted decision problem **GOAL-HIT-2BIT-QUOTIENT**:

**Input:** A finite set of transient states, an absorbing goal, and explicitly specified hit/hold action kernels with probabilities restricted to \(\{0,1/2,1\}\).

**Question:** Is there a surjective or non-surjective representation using at most **four total macro labels** (two fixed bits), with an exact goal readout and optimal uniform one-step TV defect at most \(1/4\)?

**Theorem T2.** GOAL-HIT-2BIT-QUOTIENT is **NP-complete**, even when every non-idle action is identified with a single edge of a simple graph.

**Proof of hardness.** Reduce simple graph 3-colorability. Given an input graph \(G=(V,E)\), create one transient state \(s_v\) for every \(v\in V\), plus the absorbing goal \(\dagger\). For each undirected edge \(e=\{u,v\}\), choose an orientation \(u<v\) and create action \(a_e\) with

\[
p_{s_x}^{a_e}=
\begin{cases}
0&x=u,\\
1&x=v,\\
1/2&x\notin\{u,v\}.
\end{cases}
\tag{4}
\]

Add an idle action to ensure a nonempty action alphabet when \(E=\varnothing\). All kernel rows have at most two nonzero entries, are rational with denominator at most 2, and can be constructed in polynomial time \(O(|V||E|)\).

For each edge \(e=\{u,v\}\), (4) gives
\[
d_\infty(s_u,s_v)=1.
\]
For any **nonedge** \(\{x,y\}\notin E\), each edge coordinate gives \(|p_{s_x}^{a_e}-p_{s_y}^{a_e}|\le1/2\), because both values cannot be \(0\) and \(1\) simultaneously. Consequently the incompatibility graph \(H_{1/4}\) in (2) is **exactly** the original graph \(G\). Goal visibility consumes one of the four macro labels, leaving at most three transient classes. Theorem T1 therefore gives a feasible quotient iff \(G\) is 3-colorable.

**Proof of membership in NP.** A certificate assigns each transient state one of at most three labels; the goal receives a fourth distinct label. The verifier examines every action and color class, computes the minimum and maximum rational rates, and checks that their difference is at most \(1/2\). This is polynomial-time on the explicit input. The midpoint kernels provide a constructive rational witness. QED.

The reduction uses arbitrarily many edge-indexed actions. It does **not** show NP-completeness when the action alphabet has one or two fixed members, and it does not imply any lower bound for learning from samples rather than from an explicit kernel table. The classical hardness premise is Garey, Johnson & Stockmeyer, *Some simplified NP-complete graph problems* (1976), DOI [10.1016/0304-3975(76)90059-1](https://doi.org/10.1016/0304-3975(76)90059-1).

There is broader prior art on complexity of approximate Markov minimization: Kiefer & Tang, [Approximate Bisimulation Minimisation](https://doi.org/10.4230/LIPIcs.FSTTCS.2021.48), and [On the metric-based approximate minimization of Markov Chains](https://doi.org/10.1016/j.jlamp.2018.05.006). Our mathematical result concerns the specific **one-step worst-case projected TV** objective with an exact goal and two-bit total state encoding; these other distance contracts must be kept distinct.

## 4. Verification and computational scope

The dependency-free [multi_action_coloring.py](multi_action_coloring.py) implements: exact rational per-class TV radii; the graph-to-action construction; independent coloring and macro-partition oracles; a constructive midpoint macro kernel; the general conflict graph for an arbitrary rate matrix; and candidate-threshold search via a graph-colorability oracle.

[Tests](test_multi_action_coloring.py) check:
- all 64 labelled simple graphs on 4 vertices against independent 3-coloring and macrostate-oracle decisions;
- 60 deterministically seeded 5-vertex graphs, plus an explicit complete graph counterexample;
- 2,187 exact comparisons across every two-action, three-state matrix with rates from \(\{0,1/2,1\}\), three class budgets, and multiple error thresholds;
- a constructive exact macro-kernel witness, a two-action interference counterexample, and invalid inputs.

The implementation is intended as an **independent exact witness**, rather than a scalable graph-coloring solver. Theorems T1/T2 have complete written reductions. A later Lean project can formalize the explicit edge-rate conflict identity, which is the key local equivalence in T2, separately from any claim about Lean-certifying NP-completeness.

**Research consequence:** With a single absorbing-goal action, QO-02's sorted-rate dynamic program gives a polynomial optimum. With a variable number of separately controllable hit/hold actions, the state-abstraction problem already encodes graph coloring at a two-bit macro budget. This identifies a real algorithmic boundary and motivates heuristic, structured, or approximate quotient discovery in recurrent AI models while retaining independent held-out evaluations.
