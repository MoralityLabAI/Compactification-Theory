# QO-09: Exact finite information floor for noisy quotient identification

**2026-11-04.** This follows [QO-07's anytime active quotient](ANYTIME_ACTIVE_QUOTIENT.md) and [EvalMath QI-01's zero-error identification obstruction](https://github.com/MoralityLabAI/EvalMath/pull/24). The result concerns **two specified Bernoulli worlds** and a *binary decision* about whether one transient macrostate suffices for an exact goal-preserving hit/hold quotient. It is classical statistical binary testing and data-processing mathematics, made explicit as a small CPU oracle with source-linked tests.

## 1. Worlds with different exact quotient capacity

There are two separately addressable transient states and one absorbing goal, with a common action. In World A, both transient hit probabilities are 1/2 and the minimum exact transient quotient uses one class. In World B, one transient remains at 1/2 while the other has hit rate 1/2+delta for 0<delta<1/2; it now needs two transient macro classes for zero one-step error.

An experimental controller can query either cell before each Bernoulli outcome. **Only one cell distinguishes the worlds.** Every probe to the other cell yields iid Bernoulli(1/2) noise independent of the informative stream and of the world. All streams reset between trials and each cell's outcomes are conditionally independent over time. The controller has a hard budget B for queries and may use a random, history-dependent query policy or stop early.

The decision is which world generated the data, with equal prior probability 1/2. We measure the minimum **average** classification error. The result does not automatically give the same formula for minimizing the maximum of the two world-conditional error rates.

## 2. QO-09/T1: an exact Bayes risk formula even for adaptive policies

Write p=1/2, q=1/2+delta. If all B queries address the informative cell, the two hypotheses produce iid Bernoulli bitstrings with distributions P_B and Q_B. Grouping strings by number of successes gives the exact total variation:

\[
\mathrm{TV}_B(p,q)=\frac12\sum_{s=0}^{B}\binom Bs
\left|p^s(1-p)^{B-s}-q^s(1-q)^{B-s}\right|.
\tag{1}
\]

By the elementary equal-prior likelihood-ratio test identity, their minimum mean classification error is

\[
\boxed{r_B^*(p,q)=\frac{1-\mathrm{TV}_B(p,q)}2.}
\tag{2}
\]

**Theorem T1 (no extra information from nuisance actions).** Under the specified conditional independence and world-independent nuisance laws, (2) remains the **optimal** mean decision error among *all* adaptive query policies using at most B experiments.

**Proof.** Pre-generate B independent outcomes for the informative cell and enough independent world-invariant nuisance randomness. Any adaptive controller can be simulated by revealing the next informative pre-generated outcome whenever it requests that cell; nuisance queries are answered from the independent world-invariant random source. The controller's entire history and decision are therefore a randomized measurable function (a Markov kernel) of the B informative outcomes, with the same conditional simulation law under both worlds. Data processing for total variation gives that no adaptive controller can make its transcript more distinguishable than the full B informative sequence. Conversely, always querying the informative cell B times and applying the optimal likelihood-ratio test attains (2). QED.

This is a special, fully identifiable one-cell problem; it does not prove that active experiment selection is always useless. It says only that once the unique informative cell is known, replacing one of its observations with an independent nuisance observation cannot help.

## 3. QO-09/T2: an explicit \(\Omega(\delta^{-2})\) lower bound

For the p=1/2, q=1/2+delta pair, the per-informative-query KL divergence is

\[
 D_{\rm KL}(\mathrm{Ber}(1/2)\|\mathrm{Ber}(1/2+\delta))
 =-\frac12\log(1-4\delta^2).
\tag{3}
\]

The KL chain rule for adaptive experiments gives at most B times this quantity. Pinsker's inequality then yields

\[
\mathrm{TV}_B\le
\sqrt{-\frac B4\log(1-4\delta^2)}.
\tag{4}
\]

If an experiment seeks equal-prior mean error at most \(r<1/2\), then it must have

\[
\boxed{
 B\ge \frac{4(1-2r)^2}{-\log(1-4\delta^2)}.
}\tag{5}
\]

As delta decreases this is \(\Omega(\delta^{-2})\) for fixed \(r<1/2\). The bound applies to the specified binary-world contract and a bounded number of iid addressed observations. A controller with additional prior information or further distinguishing channels requires a different calculation.

**Finite witness:** with delta=1/8 and target error r=1/4, the Pinsker necessary bound is about 15.49 observations, hence at least 16. The exact binomial-overlap oracle yields the **minimum actual budget 29**: at B=28, risk is approximately 0.250807; at B=29, risk is approximately 0.246388. The ratio between these bounds is expected, since Pinsker is conservative.

## 4. Exact independent CPU audits

[identification_sample_floor.py](identification_sample_floor.py) calculates exact Bernoulli-string total variation with Fraction arithmetic, the exact equal-prior Bayes risk and first budget meeting a target. An **independent 2^B word enumerator** in [test_identification_sample_floor.py](test_identification_sample_floor.py) checks the grouped binomial formula for three parameter pairs through B=10.

The tests also enumerate **all 128 deterministic binary-history query policies** with B=3, each choosing between the informative and an independent nuisance cell. Every policy's exact transcript TV is bounded by the all-informative experiment's TV; querying the informative cell at every history attains equality. Additional tests check TV monotonicity, the B=29 threshold, small-gap sample costs, and the numerical Pinsker upper bound.

The formal theorem rests on elementary likelihood testing, data processing and Pinsker. The finite CPU checks are implementation fidelity evidence only; the source does not assert a neural or biological experimental result.

## 5. Consequence for compact recurrent agents

QO-05 supplies a low-dimensional **existence bound** for useful macrostate bits. QO-07 and QI-01 establish what can and cannot be inferred from a specified interface. This result adds a separate constraint: even with reliable state-addressed probes, distinguishing **nearly identical** stochastic macrostates consumes a sample budget proportional to inverse squared separation.

A compact learner must therefore account for two costs independently: the bits needed to encode its learned macrostate and the evidence required to justify that encoding at the chosen tolerance. The evidence acquisition plan is part of the research contract. Any proposed generalization to Qwen/TRM dynamics needs new rate and interface measurements, not just transfer of this algebraic sample bound.
