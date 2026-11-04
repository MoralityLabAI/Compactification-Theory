# QO-01: exact goal-aware quotient learning and finite rate–distortion frontiers

**Date:** 2026-11-04. **Evidence class:** elementary finite-state theorems with written proofs, source-hashed exact-rational CPU experiments, and a proposed Lean proof of the coarsest goal-visible future partition. This note extends [IDQ-02 minimax autonomy](MINIMAX_THEOREMS.md) and [IDQ-03 goal visibility](GOAL_RECOVERY.md). It is a finite **offline** optimization method, not a trained neural network, validated biological model, or independent discovery of automaton minimization.

## 1. Coarsest goal-preserving autonomous deterministic quotient

Let \(S\) be a nonempty finite set; \(A\) a nonempty finite menu of fully specified deterministic actions \(F_a:S\to S\); and \(g:S\to\{0,1\}^r\) a *vector of declared goal observations*. An action word \(w=(a_1,\ldots,a_k)\) means actions applied in listed order. Define

\[
 s\sim_g t\quad\Longleftrightarrow\quad
 \forall w\in A^*,\; g(F_w(s))=g(F_w(t)). \tag{1}
\]

**QO-01/T1: minimal future-goal quotient.** Relation (1) is the **coarsest goal-preserving, action-stable equivalence** on \(S\). Its quotient admits a unique induced deterministic action system and an exact goal readout. Every other exact goal-preserving action quotient with \(k\) labels has at least \(|S/\!\sim_g|\) labels. The minimum fixed-length *mutable state label* is

\[
 \boxed{b_{\min}=\lceil\log_2 |S/\!\sim_g|\rceil.} \tag{2}
\]

This bit count excludes storage of the transition table, program/decoder, observations, goal specification, and interfaces; it also assumes every state in \(S\) is within the declared contract. Encoding exact continuous states or arbitrary stochastic posterior histories requires different machinery, e.g. [EvalMath IRC-02](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/interface_relative_compactification.md).

**Proof.** Equality of all future traces is an equivalence. The empty word proves that equivalent states have the same present goal. If \(s\sim_g t\), prepend any action \(a\) to any word \(w\) to obtain \(F_a(s)\sim_g F_a(t)\). Thus the quotient is goal-preserving and action-stable. Conversely, if \(\pi:S\to Z\) has an exact macro update \(G_a\) and a goal readout \(h\) with \(g=h\pi\), then from \(\pi(s)=\pi(t)\), induction on the word length gives \(\pi(F_w(s))=\pi(F_w(t))\), hence equal goal traces. Every fibre of \(\pi\) lies inside a \(\sim_g\)-class, so \(|Z|\ge |S/\!\sim_g|\) for a surjective \(\pi\). Taking \(\pi\) to be the canonical equivalence-class map attains equality. Fixed-length binary strings require at least the stated number of distinct labels, which is attainable. QED.

The theorem is a familiar *Moore/Myhill–Nerode-style deterministic minimization*, applied here to explicitly declared goal predicates. See [Björklund and Cleophas, aggregation-based minimization](https://doi.org/10.1007/s00236-019-00363-5) for related classical automata literature. **No novelty claim** is made for the general minimization theorem.

**QO-01/T2: bounded partition refinement.** Initialize \(P_0\) by grouping states with the same current goal vector. Given \(P_k\), refine by the signature

\[
 \sigma_{k+1}(s)=\bigl([s]_{P_k},([F_a(s)]_{P_k})_{a\in A}\bigr).
\]

Then \(s,t\) share a block of \(P_k\) iff all goal outputs agree after every action word of length at most \(k\). Each strict refinement increases the block count by at least one. If \(m_0\) goal-signature classes exist, stabilization occurs after at most \(|S|-m_0\) strict rounds, and distinct final blocks admit a distinguishing word of length at most \(|S|-m_0\). Induction on \(k\), followed by finite stabilization, proves the claim. Computational cost of this reference implementation is polynomial in \(|S|,|A|\) for fixed representations; the independent oracle explicitly enumerates all action words up to \(|S|-1\) and all candidate partitions on small systems.

**Exact six-state witness.** On \(S=\{0,1,2,3,4,5\}\), let only state 4 satisfy the goal. The `tick` action maps \(0\mapsto2,1\mapsto3,2\mapsto4,3\mapsto4,4\mapsto4,5\mapsto5\); a `reset` action maps the four transient states to 0 and leaves 4 and 5 unchanged. Refinement yields the four classes \(\{0,1\},\{2,3\},\{4\},\{5\}\), requiring exactly two mutable label bits instead of three for six raw states. A 203-partition independent exhaustive oracle agrees with the 4-class minimum; 216 different three-state goal-and-transition systems match the independent future-word oracle.

## 2. Finite stochastic rate–distortion under a goal contract

Let \(K_a(s,\cdot)\) be **fully specified finite stochastic kernels**, \(\pi:S\twoheadrightarrow Z\) a chosen partition, and \(L_a(z,\cdot)\) freely chosen macroscopic kernels. The one-step worst-case autonomy distortion is

\[
 \epsilon(\pi)=\min_L\max_{a,s}
 \operatorname{TV}(\pi_*K_a(s,\cdot),L_a(\pi(s),\cdot)). \tag{3}
\]

IDQ-02 proves that the best \(L\) arises independently from a total-variation Chebyshev center for every fibre/action pair. Define the optimum *mean goal error* given a declared state mass \(\mu\) and Boolean goal vector \(g\), using the optimal deterministic macro goal readout:

\[
 D_{\rm goal}(\pi;\mu)=\sum_{j=1}^{r}\sum_{z\in Z}
 \min\Bigl\{\sum_{s:\pi(s)=z,\,g_j(s)=0}\mu(s),\ 
             \sum_{s:\pi(s)=z,\,g_j(s)=1}\mu(s)\Bigr\}. \tag{4}
\]

**QO-01/T3 (exact finite rate–distortion search).** For fixed integer budget \(b\), accepted goal error \(\delta\), and probability measure \(\mu\), the optimization

\[
 \boxed{R(b,\delta)=
 \min_{\substack{\pi:S\twoheadrightarrow Z,\ |Z|\le2^b\\
                 D_{\rm goal}(\pi;\mu)\le\delta}}
 \epsilon(\pi)} \tag{5}
\]

is attained whenever the feasible set is nonempty. Enumerating every finite set partition and minimizing each fibre's total-variation radius produces an exact global solution. The result is **finite exhaustive optimization**, exponential in \(|S|\); no general computational efficiency or asymptotic novelty is claimed. Equation (4) is exact because each class' independent 0/1 Bayes decision chooses whichever goal label has greater mass. Finite enumeration proves attainment. A constrained worst-case binary goal criterion is also supported: an exact readout is required for any permitted uniform error strictly below \(1/2\); at \(1/2\) a constant quotient becomes admissible.

**Five-state exact witness.** Take transient states \(0,1,2,3\) and an absorbing goal 4. Under the `hit` action, state \(i\) stays put with probability \(1-p_i\) and reaches 4 with \(p_i\in\{1/5,2/5,3/5,4/5\}\); `idle` is identity. A goal-preserving partition must separate goal 4 from every transient. For each transient block \(B\), all its successors project into just its own macro block and the goal block, so the optimal TV radius equals \((\max_{i\in B}p_i-\min_{i\in B}p_i)/2\). This reduces (5), with \(\delta=0\), to an elementary one-dimensional clustering problem. Exhaustive enumeration matches the **independent analytic formula on all 15 goal-visible partitions**.

| Memory bits allowed | Minimum abstract classes (witness) | Optimal uniform one-step TV distortion | Representative transient grouping |
|---|---:|---:|---|
| 0 | infeasible | — | goal must remain distinguishable |
| 1 | 2 | 3/10 | {0,1,2,3} |
| 2 | 3 | 1/10 | {0,1}, {2,3} |
| 3 | 5 | 0 | each transient distinct |

The 1-bit lower bound is half the full 3/5 probability range. At 2 bits, four transients assigned to at most three non-goal blocks force at least one pair into the same block; the closest possible pair has probability gap 1/5 and hence a lower bound of 1/10. Two adjacent pairs attain it. Zero distortion requires separating all four distinct transition rates and the goal state, hence 5 classes and 3 bits. These are exact properties of the stipulated toy kernels, with **no neural/biological extrapolation**.

The complete three-axis Pareto frontier (fixed bits, uniform TV, and mean goal error) also includes the **zero-bit constant quotient** with \((0,0,1/5)\) under uniform \(\mu\). This diagnoses the vacuous solution: it predicts its own single macrostate perfectly while losing the task. The full frontier has four points:

    (bits, TV error, expected goal error)
       (0, 0,    1/5)
       (1, 3/10, 0)
       (2, 1/10, 0)
       (3, 0,    0)

## 3. Frozen-model distribution-shift bound

Suppose \(\pi\) and \(L\) were frozen after fitting the original kernels \(K_a\), with uniform macro TV error \(\epsilon\). A new kernel \(K'_a\) uses the same microstates, actions, goal readout and projection, and obeys

\[
 \sup_{a,s}\operatorname{TV}(K'_a(s,\cdot),K_a(s,\cdot))\le\kappa.
\]

**QO-01/T4 (kernel-shift certificate).** The same frozen macro kernels have distortion at most

\[
 \boxed{\epsilon'(\pi,L;K')\le \min\{1,\epsilon+\kappa\}.} \tag{6}
\]

**Proof.** Total variation cannot increase under the deterministic projection \(\pi\). Apply its contraction and the triangle inequality to \(\pi_*K'_a(s,\cdot),\pi_*K_a(s,\cdot),L_a(\pi(s),\cdot)\), uniformly in \((a,s)\). TV is at most 1. QED.

The bound is **sharp** for the five-state fixture. Freeze the optimal two-bit quotient and its macro law, then change hit probabilities to \((1/5,4/5,2/5,3/5)\), preserving the goal and idle action. The maximum source-kernel shift is \(\kappa=2/5\). The frozen error grows from \(1/10\) to exactly \(1/2=1/10+2/5\). Therefore unbounded transport to unseen interventions is impossible on the basis of a training-only autonomy score. Under a uniform \(\kappa\) certificate, combine (6) with IDQ-02's \(1-(1-\epsilon')^H\) bounded-horizon macro-path disagreement bound.

## 4. Reproduction and source-hashed receipt

- [goal_aware_partition.py](goal_aware_partition.py): reference exact finite partition refinement, canonical restricted-growth set partitions, fibre-wise stochastic minimax kernels and Pareto filter. For >=3 distinct distributions in a fibre it invokes the existing exact rational LP vertex solver, which supports at most 3 macro labels; tests constrain themselves to small finite systems satisfying that condition. Arbitrary larger instances require an actual rational LP solver.
- [test_goal_aware_partition.py](test_goal_aware_partition.py): 10 CPU tests including 216 deterministic instances checked by *independent* all-action-word enumeration and all valid small partitions, 15 goal-preserving stochastic partitions checked by a separate analytic formula, all 52 set partitions checked by an independent Bayes-readout oracle, and held-out shift and hitting-probability checks.
- [scripts/verify_goal_aware_partition.py](../../scripts/verify_goal_aware_partition.py): deterministic source-hashed receipt generator and compare-or-fail audit, pinned to the finite fixture; no API, inference, model or laboratory calls.
- [results/goal_aware_quotient/receipt.json](../../results/goal_aware_quotient/receipt.json): exact computed values and SHA-256 of source files. Reproduce from root with `python scripts/verify_goal_aware_partition.py --check` and `python -m unittest discover -s research/ingression_dynamics_v1 -p 'test_goal_aware_partition.py' -v`.

**Evidence ladder:** (1) mathematical proofs for the finite statements; (2) exact finite tests and committed source hash; (3) conjectural benefits of learned abstractions in actual recurrent models. An optimized toy partition does **not** establish generalization to a new neural substrate. Neural evaluation requires a held-out experimental family, prospective budget, baseline, and no post-selection of favourable tasks.

## 5. Literature and conceptual boundary

This overlaps established deterministic minimization, stochastic strong lumpability, bisimulation pseudometrics, and Markov abstraction. Useful precedents: [Björklund–Cleophas](https://doi.org/10.1007/s00236-019-00363-5), [Bian–Abate on approximate trace and bisimulation](https://research.google/pubs/bisimulation-and-trace-equivalence-in-an-approximate-probabilistic-context/), and [Ferns–Panangaden–Precup on MDP metrics](https://epubs.siam.org/doi/10.1137/10080484X). The repo's distinct next question is to learn an **action-aware goal-visible quotient** from partial observed dynamics with a frozen holdout and a model-description-cost ledger. Those are future experiments, not consequences of the finite optimum.
