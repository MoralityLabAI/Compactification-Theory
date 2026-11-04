# QO-07: Anytime-valid active experiments for goal-preserving macrostate quotients

**Date:** 2026-11-04. This continues [QO-06](EMPIRICAL_RATE_CERTIFICATE.md) from fixed-sample experiments to optional stopping and adaptive probe selection. All mathematics is conditional on a **known, separately addressable microscopic state/action interface**: an experimenter resets to named transient state i, chooses action a, and observes one Bernoulli hit bit. The unknown rates are only partially observed. General hidden-state POMDP identification is a separate problem.

The core constructions are classical beta-binomial mixture-martingale confidence sequences and elementary finite robust minimax optimization. There are no claims of new general martingale, POMDP, or biological theorems.

## 1. A single evidence contract across adaptive observations

Let n be the number of transient states and d the number of actions. For every cell c=(i,a) the unknown Bernoulli hit probability is p_c in [0,1]. Before collecting evidence, posit an iid Bernoulli(p_c) stream X_(c,1),X_(c,2),... for that cell. The agent can choose the next cell using past observations and public randomization, but it sees each stream's next draw **only after making that choice**. Every cell can be reset and addressed by its known identity.

For a cell with t observed hits/holds and s successes, use the uniform-Beta mixture of Bernoulli likelihood ratios:

\[
 M_t(p)=\int_0^1
 \frac{q^s(1-q)^{t-s}}{p^s(1-p)^{t-s}}\,dq
 =\frac{1}{(t+1)\binom{t}{s}\,p^s(1-p)^{t-s}}.
\tag{1}
\]

For p at the probability boundaries 0 or 1, use the consistent limiting value on null-possible paths. Impossible paths carry probability zero under that null.

**QO-07/T1, time-uniform evidence:** For fixed candidate true p, (1) is a nonnegative mean-one martingale in the filtration of its cell's i.i.d. draws, with M_0=1. Fix alpha in (0,1) and m=nd cells *before* collecting data. Define the closed superlevel interval

\[
 C_{c,t}=\left\{
 p\in[0,1]:\,
 p^s(1-p)^{t-s}\ge
 \frac{\alpha}{m(t+1)\binom{t}{s}}
 \right\}.
\tag{2}
\]

Ville's maximal inequality and a union bound over m cells imply

\[
 \boxed{\Pr\left(\forall c,\forall t\ge0,\;p_c\in C_{c,t}\right)
 \ge 1-\alpha.}
\tag{3}
\]

*Proof.* Each likelihood ratio relative to a fixed q is a nonnegative martingale; integrating over q preserves the martingale property. Ville gives Pr(sup_t M_t(p)>=m/alpha)<=alpha/m. Failure of (2) implies crossing. Union-bound the predeclared m cell streams. The event holds for *all local sample counts simultaneously*, so it remains valid after arbitrarily interleaved predictable probes and optional stopping. QED.

The implementation solves the monotone likelihood tests by **exact-rational bisection** around the binomial maximum-likelihood point s/t. The returned endpoints are deliberately rounded **outward**. Intersecting the interval with all earlier intervals yields a nested confidence set without sacrificing coverage. Finite bisection makes intervals wider than the exact analytic solution.

Primary statistical precedent: Howard, Ramdas, McAuliffe and Sekhon, [Time-uniform, nonparametric, nonasymptotic confidence sequences](https://doi.org/10.1214/20-AOS1991), *Annals of Statistics*, 2021. The Bernoulli uniform-beta mixture is an elementary, narrower specialization.

## 2. Robust one-step quotient optimization

At a data-dependent stopping time retain a rectangular uncertainty region

\[
\mathcal C=\prod_{i,a}[\ell_{ia},u_{ia}].
\]

The goal state occupies its own macrostate. For a proposed partition pi of at most k transient classes, B denotes a nonempty quotient fibre. Each action has one fitted macro hit probability q_(B,a).

**QO-07/T2, exact robust Chebyshev optimum:**

\[
 \boxed{
 R_{\mathcal C}(\pi)=
 \min_q\sup_{P\in\mathcal C}\max_{i,a}
 |p_{ia}-q_{\pi(i),a}|
 =\frac12\max_{B,a}
 \left(\max_{i\in B}u_{ia}-\min_{i\in B}\ell_{ia}\right).
 }\tag{4}
\]

The minimizing fitted macro hit probability is

\[
\widehat q_{B,a}=
\frac{\min_{i\in B}\ell_{ia}+\max_{i\in B}u_{ia}}2.
\tag{5}
\]

*Proof.* Within each class/action, the worst possible rate ranges over the union of its constituent intervals. Any center must be within half the span of both extremes. The midpoint attains half the span; all class/action centers are independently chosen. QED.

For each partition also define an optimistic radius bound

\[
 L_{\mathcal C}(\pi)=\frac12\max_{B,a}
 \left[\max_{i\in B}\ell_{ia}-\min_{i\in B}u_{ia}\right]_+.
\tag{6}
\]

For every actual table P contained in the intervals,

\[
 L_{\mathcal C}(\pi)\le D_P(\pi)\le R_{\mathcal C}(\pi),
\]

where D_P(pi) denotes the minimum exact-rate one-step macro error of that fixed partition.

Let \(R^*=\min_{|\pi|\le k}R_{\mathcal C}(\pi)\) and \(L^*=\min_{|\pi|\le k}L_{\mathcal C}(\pi)\). If \(\widehat\pi\) attains R* and its fitted macro kernels are frozen at (5), then:

**QO-07/T3, selected-model certificate:**

\[
\boxed{
 L^*\le D_k^*(P)\le R^*,
 \qquad
 E_{\rm true}(\widehat\pi,\widehat q)\le R^*,
 \qquad
 D_P(\widehat\pi)-D_k^*(P)\le R^*-L^*.
}\tag{7}
\]

The bracket width is bounded by \(w=\max_{i,a}(u_{ia}-\ell_{ia})\): \(R^*-L^*\le w\).

*Proof.* For each partition, the true minimum macro error lies between (6) and (4); the fitted midpoint rates incur no more than (4). Apply the inequalities to both the robust-selected and the true-optimal partitions. For each class/action the optimistic and robust half-ranges differ by at most w, because replacing both extrema by values inside width-w intervals changes the range by at most 2w. Thus R_C(pi)<=L_C(pi)+w. Minimize over partitions. QED.

**Decision rule at tolerance tau:** If R*<=tau then a feasible quotient is certified; if L*>tau then every allowed quotient is infeasible for any table consistent with the confidence intervals. Other outcomes remain unresolved. All three decisions are valid on the same probability event (3), even when the probe sequence and stopping time depend on the evidence.

An unprobed cell retains the full interval [0,1] and therefore forces R*>=1/2 for **any** quotient, when the macro model must predict that action at that state. This exposes a real exploration floor.

## 3. A hard limit of unaddressable hidden states

**QO-07/T4, observational alias:** Suppose each experiment instead draws one of two hidden transient states with equal probability, independently anew before every trial, and reveals only the hit bit. Compare:

    World A rates: (0, 1).
    World B rates: (1/2, 1/2).

Under either world, every observed bit is Bernoulli(1/2) independently. For every t and bitstring y_1,...,y_t,

\[
\Pr_A(y_1,\ldots,y_t)=2^{-t}=\Pr_B(y_1,\ldots,y_t).
\]

However the minimum **exact** goal-visible quotient of World A has two distinct transient classes plus the absorbing goal, requiring two label bits; World B has one transient class plus the absorbing goal, requiring one bit.

No procedure using only this non-addressable redraw interface can infer the correct exact quotient cardinality with guaranteed success across both worlds, regardless of its evidence budget. The example relies on independent hidden-state redraws; temporal persistence, trusted state tags or state-specific interventions change its identification power. This is a strictly limited, explicit two-world identifiability result.

## 4. Exact implementation and verification scope

[anytime_active_quotient.py](anytime_active_quotient.py) provides:
- beta-mixture e-values and outward-rational anytime confidence bounds;
- a finite addressable-state query ledger with immutable receipts and nested intervals;
- independent worst-case and optimistic quotient bounds, midpoint macro kernels, and a three-valued feasibility certificate;
- round-robin, largest-width and robust-bottleneck probe policies which only inspect past data;
- a seeded, resettable, digital hit/hold simulator and a separate hidden-alias trace oracle.

[test_anytime_active_quotient.py](test_anytime_active_quotient.py) checks exact fixed-time mean-one martingale identities, 20-step first-crossing probabilities by independent binomial recursion, interval outer rounding against rational grid points, robust minimax midpoints against a brute-force rational-grid oracle, Bell-partition global optima, optional-stopping decisions, incomplete-evidence rejection, reproducible active-query receipts and all hidden-alias bitstrings through length eight.

The executable exact-rational tests verify software identities. The \(1-\alpha\) guarantee is a theorem under the specified infinite iid per-cell Bernoulli stream model and Ville's inequality; the tests themselves do not establish that any real hardware or neural state source meets the assumptions. The dynamic quotient's model-code and decoder costs, identity/reset interface, sampling budget, drift, and objective selection must all be charged before a compact-AGI claim.

**Reproduce:** `python -m unittest discover -s research/ingression_dynamics_v1 -p 'test_anytime_active_quotient.py' -v`. The existing GitHub Actions morphospace workflow runs the entire established test directory.

A next-stage evaluation would train a compact recurrent experiment selector to beat the fixed policies on *sealed* new rate-table families under matched query/token budgets. Experimental selection remains a separate performance question: all three current policies enjoy the same coverage guarantee.
