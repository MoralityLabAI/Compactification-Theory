# SMC-01: Synchronized mixing and orbit-locked latent dependence

**Date:** November 15, 2026. Applied research bridge to [OLM-01 pure math](https://github.com/patrickdugan/MathCan/blob/acb543654d8f62ddbf67ff8508b388b803f338ec/research/orbit_lock/ORBIT_LOCK_DUAL_MIXING.md) in MathCan.

## Two OpenAI mathematics inputs

* **#131:** [Polynomial mixing of the switch chain for every graphical degree sequence](https://github.com/openai/math/blob/main/preprints/Polynomial-Mixing-of-the-Switch-Chain-for-Every-Graphical-Degree-Sequence-September-25-2026/main.pdf), September 25, 2026. Selected [Lean scope](https://github.com/openai/math/blob/main/lean/docs/131.md) includes the simple graph switch chain mixing bound of at most \(2n^8\) at TV distance \(1/4\) and switch connectivity.
* **#238:** [Optimal-order mixing of the Thorp shuffle](https://github.com/openai/math/blob/main/preprints/Optimal-order-mixing-of-the-Thorp-shuffle-September-26-2026/paper.pdf), September 26, 2026. Selected [Lean scope](https://github.com/openai/math/blob/main/lean/docs/238.md) includes full-permutation total-variation mixing on \(2^d\) cards after \(O(d)\) physical shuffles.

The synthesis asks whether two individually mixed observables of a looped transformer can retain measurable joint information through shared stochastic controls, hidden gauge frames, or correlated feature paths.

## Implementation and registered discriminators

The executable finite gym uses only Python's standard library and exact rational arithmetic. It enumerates:

* All three perfect matchings of four vertices (degree sequence \(1,1,1,1\)). Its lazy matching switch kernel stays put with probability \(1/2\) and proposes either alternate matching with probability \(1/4\).
* All 24 permutations of four slots. Each binary-cube coordinate sweep independently swaps the edges \((0,1),(2,3)\) and then \((0,2),(1,3)\), each with a fair coin.
* The independent joint transition \(L=P\otimes Q\), and a coupled transition \(K\) which mixes \(L\) with a comonotone shared-uniform-quantile coupling of the two conditional rows. The latter has **exactly the same conditional graph and permutation marginal transition laws**.

This coordinate sweep is a **Thorp-inspired four-slot toy**. Its finite mixing numbers are calculated from the explicit transition matrix, rather than imported from the asymptotic theorem for the original Thorp shuffle. The graph is a restricted four-vertex switch example. There is no trained neural network or SAE checkpoint in this result.

The experiment measures, per step, graph TV, permutation TV, independent joint TV, coupled joint TV, and dependence TV between the coupled and independent joint laws. Four families of regression tests require:

1. Each marginal transition to preserve its own uniform stationary law.
2. The common-randomness coupling to preserve both conditional marginal transition rows.
3. The product joint TV to lie between the maximum marginal TV and \(a+b-ab\).
4. The dependent joint TV to obey both the one-step-defect telescoping inequality and the three-step Dobrushin block certificate.

The formulas and self-contained proofs are in MathCan's OLM-01. In particular, with
\(\eta=\sup_z\operatorname{TV}(K(z,\cdot),L(z,\cdot))\) and independent marginal mixing errors \(a_t,b_t\),

\[
\operatorname{TV}(\delta_z K^t,\pi\otimes\sigma)
\le \min\{1,a_t+b_t-a_tb_t+t\eta\}.
\]

For a block length \(b\) with independent Dobrushin coefficient \(\rho_b<1\), its stationary-error bound is

\[
D_{kb+r}\le \rho_b^kD_0+
b\eta\sum_{j=0}^{k-1}\rho_b^j+r\eta,\qquad 0\le r<b.
\]

## Executed CPU results

Local exact-rational tests pass. For coupling share \(1/4\), the conditional independence defect is \(\eta=5/32\); the independent three-step contraction is \(\rho_3=75/2048\).

| Sweeps | Graph marginal TV | Permutation marginal TV | Independent joint TV | Coupled joint TV |
|---:|---:|---:|---:|---:|
| 0 | 2/3 | 23/24 | 71/72 | 71/72 |
| 1 | 1/6 | 1/3 | 1/3 | 37/96 |
| 3 | 1/96 | 1/48 | 1/48 | 18839/147456 |
| 5 | 1/1536 | 1/768 | 1/768 | 4724999/37748736 |
| 10 | 1/1572864 | 1/786432 | 1/786432 | 4947808878599/39582418599936 |

The final coupled joint TV is about \(0.125\), even though each marginal is below \(0.000002\). The synchronized coupling induces a joint residual that single-coordinate probes cannot certify. The generic perturbative block bound is conservative.

An independent, purely group-theoretic *snapshot adversary* in OLM-01 selects a uniform permutation \(S\) and sets \(G=S G_0\). Its two coordinate marginals are exactly uniform, while

\[
\operatorname{TV}(\mathcal L(G,S),U_G\otimes U_S)=1-\frac{1}{(2k-1)!!}.
\]

At four vertices the joint TV is \(2/3\), and at six vertices it is \(14/15\). The one-bit relational observer \(G=S G_0\) is optimal. Snapshot orbit locking is presented separately from the Markov trajectories because their transition laws are different objects.

## Reproduce

From repository root:

    python -m unittest discover -s research/synchronized_mixing_v1 -p 'test_gym.py' -v
    python research/synchronized_mixing_v1/gym.py

To obtain the ten-step trace without any random sampling:

    python -c "import json; from research.synchronized_mixing_v1.gym import evaluate; print(json.dumps(evaluate(horizon=10),indent=2))"

The scoped GitHub Actions workflow reruns the exact tests on a PR or push and uploads an independently reproduced ten-step JSON trace.

## Next training-gym integration

Instrument a compact looped transformer at a fixed hidden-layer interface. Select an incidence graph of SAE features as the graph-valued coordinate and a token/slot alignment map as the permutation-valued coordinate. Compare interventions that scramble those coordinates independently against interventions sharing an alignment key. Register random-seed and family-disjoint holdouts; measure whether an exact or learned relational monitor succeeds while separately calibrated marginal monitors approach chance. Track the minimum bits, parameters, and recurrent steps needed to preserve or detect the relation.

The trained-model hypothesis remains **unverified**. Feature dictionaries, neural training gains, interpretability claims, and the published asymptotic mixing constants each require their own additional evaluation. The contribution here is a finite observable-dependence failure mode with proofs, exact executable certificates, and a ready-to-port interface.
