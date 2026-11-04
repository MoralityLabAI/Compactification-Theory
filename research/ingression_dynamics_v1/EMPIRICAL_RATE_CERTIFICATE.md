# QO-06: Certified quotient learning from sampled hit probabilities

**2026-11-04.** This note supplies an elementary **finite-perturbation theorem** and a classical fixed-sample concentration corollary for the QO-04 hit/hold Markov abstraction contract. The mathematical data are finite rational vectors; the code audits exact toy examples. **No deployed model, wet-lab experiment, or observed neural learning result is claimed.** Statistical confidence statements assume preregistered per-cell sample counts and IID Bernoulli observations within each microscopic state/action pair.

Companions: [QO-04](GEOMETRIC_COMPLEXITY.md) identifies rate-space square covering, [QO-05](INTRINSIC_RATE_ENTROPY.md) bounds state bits given a latent geometric factorization, and [EvalMath MP-01/02](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/metaprune_decision_value.md) studies experiment-budget allocation. The separate EvalMath full-filtration warning matters if an agent adaptively chooses which evidence to publish or when to stop sampling.

## 1. Sup-norm stability of a quotient's minimum TV error

Let \(P=(p_i^a)\in[0,1]^{n\times d}\) denote true hit probabilities, and \(\widehat P=(\widehat p_i^a)\) an estimate on identical microscopic states and action labels. Fix at most \(k\) transient macro classes, with a separate absorbing goal. Set

\[
 \eta=\|P-\widehat P\|_\infty
 =\max_{i,a}|p_i^a-\widehat p_i^a|.
\]

For any partition \(\pi\), define the exact minimax one-step projected TV defect (from QO-03)

\[
 D_P(\pi)=\frac12\max_{\substack{B\in\pi\\a}}
       \bigl(\max_{i\in B}p_i^a-\min_{i\in B}p_i^a\bigr).
\tag{1}
\]

Define the constrained optimum \(D_k^*(P)=\min_{\pi:|\pi|\le k}D_P(\pi)\).

**QO-06/T1 (uniform stability).**

\[
\boxed{\forall\pi:\quad
|D_P(\pi)-D_{\widehat P}(\pi)|\le\eta.}
\tag{2}
\]

**Proof.** For every block \(B\) and action \(a\), the maximum of the true rates differs from the maximum of the estimated rates by at most \(\eta\). The same holds for the minimum. The two block ranges differ by at most \(2\eta\), and their half-ranges by at most \(\eta\). Taking the maximum over every block/action preserves this bound. QED.

**QO-06/T2 (stability of global optimum).**

\[
\boxed{|D_k^*(P)-D_k^*(\widehat P)|\le\eta.}
\tag{3}
\]

**Proof.** Apply (2) to any optimal partition of \(\widehat P\) and obtain \(D_k^*(P)\le D_k^*(\widehat P)+\eta\). Swap \(P\) and \(\widehat P\) for the reverse inequality. QED. A finite optimum exists because there are finitely many partitions.

**QO-06/T3 (frozen quotient regret).** Let \(\widehat\pi\) be an empirically optimal partition under budget \(k\). Then

\[
\boxed{
0\le D_P(\widehat\pi)-D_k^*(P)\le 2\eta.}
\tag{4}
\]

**Proof.** By (2), \(D_P(\widehat\pi)\le D_{\widehat P}(\widehat\pi)+\eta=D_k^*(\widehat P)+\eta\). By (3) the latter is bounded by \(D_k^*(P)+2\eta\). The regret is nonnegative by the definition of the optimum. QED.

There is also an operational bound for the **frozen fitted macro kernels**. In macro class \(B\) at action \(a\), choose the midpoint of the estimated minimum and maximum, \(\widehat q_{B,a}=(\min_{i\in B}\widehat p_i^a+\max_{i\in B}\widehat p_i^a)/2\), and use its Bernoulli hit/hold macro transition. For every microscopic \(i\in B\),

\[
|p_i^a-\widehat q_{B,a}|
\le|p_i^a-\widehat p_i^a|+
    |\widehat p_i^a-\widehat q_{B,a}|
\le\eta+D_k^*(\widehat P).
\]

Thus the frozen model obeys

\[
\boxed{E_{\rm true}(\widehat\pi,\widehat L)
\le D_k^*(\widehat P)+\eta
\le D_k^*(P)+2\eta.}
\tag{5}
\]

This statement is **uniform in microscopic state and action** once the global \(\eta\) bound is valid. It does not assume that the fitted partition was chosen independently of the observations: (2) holds simultaneously for **all** partitions, which is why post-training selection is covered by the deterministic argument.

The constant \(\eta\) in (2) and (5) is sharp: with two transient states in one class, true rates \((0,1)\) and estimated rates \((\eta,1-\eta)\), \(0<\eta<1/2\), the true radius is \(1/2\), the estimated radius is \(1/2-\eta\), and the frozen midpoint error equals \(1/2\).

## 2. Fixed-size Bernoulli sampling certificate

Suppose each pair of microscopic state and action \((i,a)\) has exactly **\(N\) independent Bernoulli hit/hold draws**, with common true success probability \(p_i^a\), before any estimates, partition selection, or evaluation-dependent stopping. Dependencies **between** different pairs do not invalidate the following union bound; within each pair its Bernoulli observations must obey the stated independent fixed-\(N\) model.

For the empirical rate \(\widehat p_i^a\), classical two-sided Hoeffding gives

\[
\Pr(|\widehat p_i^a-p_i^a|>\eta)\le2e^{-2N\eta^2}.
\]

Taking the union bound over \(nd\) pairs yields

\[
\Pr(\|P-\widehat P\|_\infty>\eta)
\le2nd\,e^{-2N\eta^2}.
\tag{6}
\]

Fix confidence level \(1-\alpha\) before sampling. With

\[
\eta_N=\sqrt{\frac{\log(2nd/\alpha)}{2N}},
\tag{7}
\]

equations (2)–(5) hold simultaneously with probability at least \(1-\alpha\). If (7) exceeds 1, the trivial deterministic \(\eta\le1\) may be used instead.

In particular, the **observable, selected macro-model** has

\[
\boxed{
E_{\rm true}(\widehat\pi,\widehat L)
\le D_k^*(\widehat P)+\eta_N
\quad\text{with probability }\ge1-\alpha.}
\tag{8}
\]

If a desired excess over the unknown true optimum is \(\delta>0\), a sufficient **fixed** sample count per microscopic state/action is

\[
\boxed{
N\ge \left\lceil \frac{2}{\delta^2}
    \log\frac{2nd}{\alpha}\right\rceil,}
\tag{9}
\]

which makes \(2\eta_N\le\delta\). Total hit/hold observations required are \(ndN\), an explicit data cost beyond the final compressed label bits.

**Source:** W. Hoeffding, *Probability Inequalities for Sums of Bounded Random Variables*, JASA 58(301), 13–30 (1963), https://doi.org/10.1080/01621459.1963.10500830. The union bound and finite-rate stability deductions are presented here with their assumptions; no novel concentration inequality is claimed.

### Crucial protocol limitation

Sampling until an observed quantity passes a threshold, deleting sources after viewing them, or selecting a favourable subset of microscopic states violates the pre-fixed observation-filtration contract that led to (6). For adaptive sample allocation, use a valid **time-uniform** confidence sequence or other correctly calibrated filtration-aware evidence process, and rederive the simultaneous error event. See [EvalMath validity under actor filtration](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/dynamic_filtration_validity.md).

The bound concerns **one-step projected TV** on a stated set of microstates. To transport to a finite-horizon macro-observation policy, combine it with the previously proved IDQ-02 maximal-coupling formula \(1-(1-\epsilon)^H\), where \(\epsilon\) is the certified frozen macro error and the policy observes only macro history.

## 3. Reproduction and independent oracles

- [rate_sampling.py](rate_sampling.py) implements the exact finite sup error, per-partition radius, empirical midpoint macro kernel, frozen-model error, true/ex-post regret audit, and closed-form Hoeffding union-bound calculator. It uses the QO-04 exact finite geometric optimizer, so the fitted partition is globally optimal for this small explicitly enumerated model.
- [test_rate_sampling.py](test_rate_sampling.py) checks one example attaining the \(\eta\) bound exactly, a partition-switching example with positive regret, 270 seeded cases independently reoptimized by *all* finite class assignments, Bernoulli-tail probability checks from exact binomial arithmetic, and sample-budget formula checks.

The exact tests can verify the algebra on the stipulated finite data. They do **not** verify that a physical sampler produces IID Bernoulli outcomes. The confidence theorem relies on the cited probabilistic inequality and the stated sampling model.

Run:

    python -m unittest discover -s research/ingression_dynamics_v1 -p 'test_rate_sampling.py' -v

The repository's established read-only-token morphospace GitHub workflow runs this alongside the earlier QO-01–05 CPU audits. Any future learned-controller report should retain the frozen action/state universe, sample-count protocol, full rate estimates or sufficient-statistic receipts, policy/evaluation lineage, and exact model-and-state code budget.
