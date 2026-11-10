# OIC-21 | Adaptive sensing of a hidden modular generator

**Evidence and scope, 10 November 2026.** Exact finite Bayesian dynamic programming, independent fixed-schedule enumeration, and a CPU-trained oracle-distilled sensor policy. The mathematical statements are specializations of sequential Bayesian experimental design and active automata testing. The learned agent contains a supplied exact enumerated cyclic belief model. Nothing here establishes an intrinsic mind, a neural transformer mechanism, or a result about cosmological dynamics.

## Declared observation contract

Let \(n\ge 2\), hidden shift \(d\in\mathbb Z_n\), and initial phase \(z_0\in\mathbb Z_n\). The controller knows \(n\), and both unknowns carry independent uniform priors. Between successive sensor queries, \(z_t=z_0+td\pmod n\). At each clock step \(t\), the controller chooses exactly one sensor address \(m_t\in\mathbb Z_n\) and receives exactly one labeled bit

\[
y_t=[z_t=m_t].
\]

The controller's next sensor address may depend on previous bits. It has Q time steps and Q returned sensor bits. An adaptive implementation must also transmit addresses (\(Q\lceil\log_2 n\rceil\) additional address bits using a direct binary code), and must store or compute its policy. Time/communication overhead is reported separately. There is no privileged access to the true initial phase or shift during evaluation.

A posterior state is a subset \(S\subseteq\mathbb Z_n\times\mathbb Z_n\) of still-possible \((d,z_0)\) pairs. Define \(S_{m,t}^1=\{(d,z_0)\in S:z_0+td\equiv m\pmod n\}\), and \(S_{m,t}^0=S\setminus S_{m,t}^1\). Conditioning on a deterministic reported bit makes every remaining pair equiprobable, as long as the prior was uniform.

## OIC-T31: optimal adaptive Bayes decision recursion

For shift identification under zero-one loss with a uniform prior, let \(V_Q(S)=\max_{d\in\mathbb Z_n}|S\cap (\{d\}\times\mathbb Z_n)|\). At intermediate times define

\[
\boxed{V_t(S)=\max_{m\in\mathbb Z_n}\{V_{t+1}(S_{m,t}^0)+V_{t+1}(S_{m,t}^1)\}.}
\]

**Proof.** On a leaf history, the optimal MAP decoder guesses the shift having the largest number of surviving initial phases. Each such leaf contributes \(\max_d |S\cap (\{d\}\times\mathbb Z_n)|/n^2\) to the root correct-identification probability. A sensor query partitions S into its two disjoint report sets. The correct-mass contributions of its descendants add, and maximization over the next address yields the recursion by backward induction. Thus the root optimum is \(V_0(\mathbb Z_n^2)/n^2\). QED.

For a **fixed** word \(m_0,\ldots,m_{Q-1}\), the same leaf-MAP score is computed after partitioning by that word. Exhaustive enumeration of the \(n^{Q-1}\) distinct words with \(m_0=0\) finds the fixed optimum; fixing the first marker at zero is valid because jointly translating every marker and the uniform initial phase preserves the outcome law.

Finite exhaustive results for five returned bits:

| n | Best fixed | Optimal adaptive | Correct-mass numerators |
|---|---:|---:|---|
| 4 | 13/16 | 15/16 | 13 and 15 |
| 5 | 17/25 | 19/25 | 17 and 19 |
| 6 | 18/36 | 22/36 | 18 and 22 |

These are exact small-instance witnesses of the benefit of feedback, in agreement with classical sequential optimal experimental design. The general Bellman construction is established; the numerical finite values are new *computed examples*, with no claim of a previously unknown universal theory.

## OIC-T32: a zero-report lower bound

**Theorem.** A policy that uses fewer than n one-bit marker queries cannot guarantee identifying d for every admissible initial phase.

**Proof.** Follow the branch on which every returned bit is zero. The adaptive policy's addresses on this branch are a fixed sequence \(m_0,\ldots,m_{Q-1}\). For any candidate d, observation zero at time t rules out at most one initial phase, namely \(z_0=m_t-td\). After Q<n queries, at least n−Q phases survive for **every** shift d. Thus every shift remains possible along the all-zero branch, and any single final guess must fail for some allowed hidden state. QED.

### OIC-C1 (open): minimum worst-case query horizon

Exact dynamic programs establish that the first fully identifying horizon is Q=2n−2 for n=2,3,4,5,6. The universal conjecture \(Q^*(n)=2n-2\) is **unproved** and may fail at larger composite n. The lower-bound theorem above gives only \(Q^*(n)\ge n\). The full n=6 computation reaches Q=10 and uses over three million memoized subproblems; it is left outside lightweight CI, while its shorter n=6 values and the n≤5 full thresholds have fast replay tests.

## OIC-T33: process-law identification versus single-episode identification

The OIC-20 marker set M=(0,2) on n=6 has distinct *distributions of full observation words* for all six candidate shifts. Yet even after observing both marker bits simultaneously at all six time steps, the Bayes success probability for an unknown shift and a single unknown uniform initial phase is just

\[
\boxed{28/36=77.\overline7\%.}
\]

For example, \((d,z_0)=(0,1)\) and \((3,1)\) both yield a zero vector at every step for M=(0,2), since their state trajectories never visit either mark. The two induced **laws** differ because other initial phases produce different observations. A statistician could distinguish the laws through repeated independent episodes, while a single noiseless trajectory can remain ambiguous. These are distinct identification criteria. QED.

This distinction also clarifies resource comparisons: OIC-20's minimal-marker theorem counts sensor types sufficient for **distribution identifiability**, while OIC-21 measures Bayes correct identification on a single experiment. A fixed multi-marker observation at n steps uses \(n|M|\) output bits, compared with n+2 output bits for our principal one-marker adaptive rollout. Sensor addresses, parallelism, and elapsed time have different cost profiles.

## OIC-T34: interface-relative resource ledger

The exact posterior ranges over \(n^2\) candidate pairs \((d,z_0)\). A lossless posterior subset could be represented as an \(n^2\)-bit membership mask; a noisy posterior requires finite-precision real likelihoods for these \(n^2\) candidates. Query budgets count returned one-bit observations only. A direct-address adaptive interface also uses Q ceil(log₂ n) control bits. If fixed sensor placements are preinstalled, their configuration addresses are paid once rather than on every timestep. The trained query scorer has 145 scalar parameters, each of which also needs an encoded storage precision; its seven per-candidate features require exact enumerated posterior computations.

A simulation that assumes free access to a Bayesian table of all n² hidden hypotheses is a **structure-assisted monitor**. Its learned component is the sensor-address policy. OIC-19's learned generator recovery is a separate project with a different prior.

## OIC-T35: faults can alter the query path

Suppose policy \(m_t=\pi(y_0,\ldots,y_{t-1})\) selects the next marker from previously *received* symbols. A substituted bit at time k can change all future addresses \(m_t,t>k\). Therefore Hamming distances between codewords under a **single fixed marker word** do not alone certify worst-case correction for this adaptive interface. The fault experiment in OIC-21 runs the full policy closed-loop after each corrupted bit and conditions the model posterior on an explicitly assumed 4% independent sensor-flip likelihood. Its one-error panels are adversarial placements, outside that iid likelihood assumption. A robust adaptive guarantee requires an error-aware policy tree with explicit possible receipt sets or a minimax decision game.

## Literature and next experiment

- Shen and Huan, [Bayesian sequential optimal experimental design for nonlinear models using policy gradient reinforcement learning](https://doi.org/10.1016/j.cma.2023.116304), *Computer Methods in Applied Mechanics and Engineering* 416 (2023): finite-horizon Bayesian experimental design and policy learning.
- Chen, Hassani and Krause, [Near-optimal Bayesian Active Learning with Correlated and Noisy Tests](https://proceedings.mlr.press/v54/chen17b.html), AISTATS (2017): limitations of myopic test selection with correlated evidence.
- Soucha and Bogdanov, [Observation Tree Approach: Active Learning Relying on Testing](https://doi.org/10.1093/comjnl/bxz056), *Computer Journal* 63 (2020): adaptive separating sequences for finite-state machines.
- OIC-20 in this repository: [sensor-type identifiability](OIC20_THEORY.md).

The proposed OIC-22 follow-up replaces clean-oracle teacher labels with an explicitly fault-aware decision objective and measures robust identifiability at fixed total communication and computation budgets. No trained-model result in the present note implies consciousness, physical cosmology, or a specific transformer state mechanism.
