# IDQ-02: minimax autonomy defect, stochastic lumpability and multiscale composition

**Status 2026-11-04:** finite-system theorems with proofs and dependency-free exact-rational computational witnesses. These are classical Chebyshev-radius, state aggregation, coupling, and triangle-inequality constructions applied to a common **autonomy-defect** contract. No novelty claim for the underlying generic results. No biological experiments or real-model observations.

**Relation to IDQ-01:** [THEORY.md](THEORY.md) proves exact controlled trajectory transfer and a finite-horizon approximate stochastic coupling bound. Here we compute the **best possible one-step** quotient and separate genuine dynamic closure from visually similar macro trajectories. [MOP-01](https://github.com/patrickdugan/XenobotBench/pull/3) studies finite future-observation equivalence; [QD-02](https://github.com/MoralityLabAI/RSITopology/pull/6) studies spectral closure.

## 1. The exact deterministic minimax theorem

Let S,A,Z be **finite nonempty** sets, pi:S->Z a surjection, (Z,d) a metric space, with microscopic deterministic actions F_a:S->S. A candidate abstract system consists of independent macro action maps G_a:Z->Z. Define

    E(pi,G) = max_{s in S,a in A} d(pi(F_a(s)), G_a(pi(s))).

For each macrostate z and action a, the successor fibre is the **nonempty** subset

    Y_{z,a} = { pi(F_a(s)) : pi(s)=z } subset Z,

with Chebyshev radius relative to allowed **macro centers Z**

    r_Z(Y) = min_{c in Z} max_{y in Y} d(y,c).

**Theorem IDQ-02/T1 (exact optimality, finite).**

    min_{G} E(pi,G) = max_{z,a} r_Z(Y_{z,a}).                 (1)

An optimal G exists and can be chosen independently for each (z,a).

**Proof.** For any G, E(pi,G) is the maximum over (z,a) of max_{y in Y_{z,a}} d(y,G_a(z)). It is at least the maximum of the corresponding independently minimized radii. Conversely, for each (z,a) choose a minimizing center c_{z,a} in the nonempty finite Z and set G_a(z)=c_{z,a}. This attains the maximum of the radii. QED.

**Corollary (exact quotient criterion).** The optimal error vanishes iff every Y_{z,a} is a singleton; equivalently,

    pi(s)=pi(t) => pi(F_a(s))=pi(F_a(t))

for every s,t,a. In that case an exact macro family exists and is unique. This is the familiar fibre-consistency condition.

**Radius bounds.** Writing D(Y)=max_{x,y in Y}d(x,y), triangle inequality and choosing a center in Y give

    D(Y)/2 <= r_Z(Y) <= D(Y).                               (2)

The upper bound can be sharp! With Z={0,1}, discrete metric d(0,1)=1, Y={0,1}, every permitted center has radius 1. In contrast, when allowed centers are **all of R** with Euclidean metric and Y is finite, midpoint c=(min Y+max Y)/2 attains r=(max Y-min Y)/2. Beware changing the center space while comparing statements.

**Computability.** For finite explicit S,A,Z, form all fibres and successors, evaluate each candidate center and take the best. The algorithm uses exact rational distances if d does. An independent brute-force oracle enumerating the entire G table agrees on all 162 surjective binary projections of 3 microstates paired with all 3^3 single-action micro maps.

## 2. Stochastic analogue and the total-variation minimax center

Let K_a(s,.) be a microscopic Markov kernel on finite S. Its observed successor distribution is P_{s,a}=pi_*K_a(s,.) in the probability simplex Delta(Z). For a proposed macro kernel L_a(z,.) define

    E_TV(pi,L)=max_{s,a} TV(P_{s,a}, L_a(pi(s),.)).

Applying T1's independent-center argument in the compact simplex gives

    min_L E_TV(pi,L) = max_{z,a} rad_TV({P_{s,a}:pi(s)=z}).   (3)

A minimizer exists by compactness/continuity. As in (2), half-diameter and diameter bound the radius. The centers are distributions, **not** restricted to deterministic macros.

For two distributions p,q, the midpoint (p+q)/2 attains radius TV(p,q)/2 by norm homogeneity and the triangle inequality. For k>=2 distinct Dirac distributions delta_1,...,delta_k over k states, the uniform distribution is an optimal center and the radius is 1-1/k: TV(delta_i,q)=1-q_i, so max_i TV(delta_i,q)=1-min_i q_i>=1-1/k. In particular three Dirac successors have pairwise TV diameter 1 but minimax radius **2/3**, disproving the tempting universal half-diameter formula.

**Exact finite LP.** With macro alphabet Z={1,...,m}, the TV radius problem is the linear program

    minimize r over q in Delta(Z)
    subject to r + sum_{y in B}q_y >= sum_{y in B}p_i(y)
    for each source successor law p_i and each B subset Z.       (4)

Since TV(p_i,q)=max_B (p_i(B)-q(B)), this LP is equivalent to (3). The reference implementation enumerates vertices with rational Gaussian elimination for m<=3; it is a toy exact oracle, **not a scalable LP solver**.

**Exact strong lumpability criterion.** A macro transition law with zero TV defect exists iff every pair s,t in one fibre has identical aggregated probabilities to **every** macro fibre, for every action. This is the standard finite strong-lumpability condition. It is distinct from convergence of the macro process in one specially chosen initial distribution.

## 3. Multi-step coupling bound, including history policies

Assume for all s,a the observed one-step TV discrepancy against a fixed macro L is at most epsilon in [0,1]. Couple the two processes from z_0=pi(s_0) by taking identical actions **while their observed histories agree**, and maximally coupling their next macro observations. The micro process retains its true hidden state, sampled conditionally on its next observed macro state. At each matched time, the probability of extending agreement is at least 1-epsilon.

**Theorem IDQ-02/T2 (matched-history probability).**

    P(ever unequal observed states by time H)
        <= 1-(1-epsilon)^H.                                    (5)

This is also a bound on the probability of disagreement of an arbitrary measurable event of the common macro-observation trajectory. It holds for any policy driven solely by common macro-observation history and a coupled random tape. A policy with direct access to microstate s is outside this comparison.

If both per-step rewards lie in [0,1] and their absolute difference is at most delta **on matching states under matching actions**, the return difference is bounded by

    sum_{t=0}^{H-1} [delta+(1-delta)*(1-(1-epsilon)^t)].        (6)

**Proof.** Condition on trajectories agreeing up to time t; maximal coupling succeeds for the next step with conditional probability at least 1-epsilon. Multiply the conditional survival bounds. At time t, rewards differ by <=delta on agreement, and by <=1 after a mismatch. Apply the expectation triangle inequality and sum. QED. This sharpens the union-bound factor t*epsilon in IDQ-01 and remains finite for any epsilon.

## 4. Composition of deterministic approximate quotients

Let pi:X->Y and rho:Y->Z, with F_a,G_a,H_a microscopic/intermediate/macroscopic maps and metrics d_Y,d_Z. Suppose

    d_Y(pi F_a(x),G_a pi(x)) <= epsilon_1
    d_Z(rho G_a(y),H_a rho(y)) <= epsilon_2,

and rho is L-Lipschitz from Y to Z on the relevant states. Then

    d_Z(rho pi F_a(x),H_a rho pi(x))
      <= L*epsilon_1 + epsilon_2.                              (7)

**Proof.** Insert rho G_a pi(x), use the triangle inequality and the Lipschitz bound. QED. The constant can be sharp for maps on the real line whose errors align in the same direction.

If H_a is lambda-Lipschitz uniformly and actions are a fixed common schedule, pathwise macro prediction error after t steps is at most

    (L*epsilon_1+epsilon_2) * sum_{j=0}^{t-1}lambda^j.           (8)

This multi-step recurrence requires the common action schedule; a feedback policy whose actions switch discontinuously at close but unequal observations needs additional robustness hypotheses.

## 5. Interpretations, controls and the priority theorem

The rigorous meaning of "autonomous higher-level individual" here is restricted: a specified observation pi has **closed predictive dynamics** under a specified action interface. This definition does not establish agency, phenomenology, personhood, life or metaphysical ingression. Hidden pretraining, evolution, hand-designed state and search must all be charged when making discounted-lunch competence claims.

**Core prospective experiment:** estimate per-fibre successor radii from a *training* intervention panel; freeze pi and fitted G; evaluate calibrated error and goal-recovery transfer on blinded interventions. Include coarsened/relabeled observation maps, microstate reset, matched random partitions, and equal-budget learned baselines.

**Novel research question (open):** can a learned pi simultaneously minimize held-out autonomy defect and retain nontrivial goal-reaching value while using fewer bits than full microstate reconstruction? The vacuous constant projection has zero autonomy defect and must be explicitly penalized by goal observability, predictive value, or distinguishability.

## 6. Prior literature and evidence levels

- Polyanskiy & Wu, *Information Theory: From Coding to Learning*, Section 5.3, Chebyshev radius/diameter and Jung-type bounds: https://people.lids.mit.edu/yp/homepage/data/itbook-2022.pdf
- Standard strong lumpability of Markov chains: https://pmc.ncbi.nlm.nih.gov/articles/PMC7971158/
- Ferns, Panangaden & Precup, *Metrics for Finite Markov Decision Processes*, UAI 2004: https://mlanthology.org/uai/2004/ferns2004uai-metrics/
- Probabilistic-bisimulation approximation survey: https://www.sciencedirect.com/science/article/pii/S1571066113000820

**Algebraic theorem with written proof:** T1, T2, (3), (7), (8). **Finite audited examples:** Python exact rational tests; independent exhaustive G oracle for 162 small cases; LP vertex oracle on at most three macro states. **Open conjecture/application:** how predictive quotient organization relates to experimentally measured biological or AI competence. **Lean formalization:** MOR-02 in OntologicalProofs; only a passing kernel audit may promote its claims from written proof to checked theorem.

Run (from repository root):

    python -m unittest discover -s research/ingression_dynamics_v1 -p 'test_minimax_autonomy.py' -v
