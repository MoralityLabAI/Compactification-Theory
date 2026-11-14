# KSR-01: executed finite reconstruction and compact-monitor results

Date: 2026-11-14.
Evidence class: **EXACT_FINITE_TREE_CONTROL**.

[Mathematical proofs](THEORY.md) |
[Exact reference](reference.py) |
[Independent tests](test_reference.py) |
[GitHub Actions](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43544854449).

The source results are OpenAI Mathematics #229 (three-state
symmetric-channel reconstruction) and #236 (factor-of-IID
threshold for free zero-field Ising spins). Their published
infinite-tree claims are imported research statements,
with different formalization scopes.

## Exact nonasymptotic root reconstruction

On a b=2 rooted broadcast tree with edge channel
eigenvalue lambda=4/5, every leaf is observed.
The table is computed by enumerating all possible
leaf-state patterns and their exact rational joint
probabilities; percentages are rounded for display.

| q | Depth | KSR-2 provable count floor | Exact count-only MAP | Exact full-leaf MAP |
|---:|---:|---:|---:|---:|
| 2 | 1 | 89.02% | 90.00% | 90.00% |
| 2 | 2 | 83.31% | 87.76% | 87.76% |
| 2 | 3 | 79.89% | 86.37% | 86.37% |
| 3 | 1 | 85.37% | 86.67% | 86.67% |
| 3 | 2 | 77.75% | 84.32% | 84.32% |
| 3 | 3 | 73.19% | 82.35% | 82.48% |

For q=3, depth 3, the complete leaf pattern
has a modest advantage of about 0.123 percentage
points over the q-histogram. The count-only MAP
is not generally identical to full-pattern MAP.

The exact second-moment limit for b=2, lambda=4/5
is V_infty=16/7, with asymptotic guaranteed
count reconstruction of 23/32 (q=2) and 5/8 (q=3).
The mutable count-memory budget grows only linearly
with depth. The original likelihood decoder's
description and its offline computational cost are
not included in that counter bit count.

## Below-critical and critical controls

For q=3, b=2 and lambda=1/2 (t=1/2),
the exact count MAP probabilities are 57.18%
at L=2 and 49.46% at L=3, each exceeding
random 33.33% on these finite depths. The
source #229 negative asymptotic theorem is
not inferred from two shallow finite observations.

At b=4, lambda=1/2, t=1,
KSR-1 yields V_L=1+3L/4 exactly.
At L=8 the three-state count reconstruction
lower bound is 3/7. The bound tends toward
random guessing as L grows. This degeneration
alone does not prove optimal non-reconstruction.
Source #236 states that free Ising on the
infinite d=5 regular tree is FIID at equality;
our finite rooted b=4 broadcast model is a
separate, forward-tree inference problem.

## Independent retention controls

| q, b | L | r | KSR-3 floor | Exact count MAP | Exact full-pattern MAP |
|---|---:|---:|---:|---:|---:|
| 2, 2 | 3 | 1/2 | 73.26% | 82.01% | 82.34% |
| 3, 2 | 2 | 1/2 | 64.91% | 75.63% | 75.70% |
| 3, 2 | 2 | 1/4 | 53.34% | 62.86% | 62.88% |

These probabilities use all possible observed
leaf patterns, including explicitly missing
symbols and their independent Bernoulli
retention weights. For every row, the
independent posterior enumeration verifies
the exact moment correction

\[
V_L(r)=V_L+\frac{1-r}{r(b\lambda^2)^L}.
\]

## KSR-6: sparse budget with constant information floor

Take r_L=(b lambda^2)^(-L) for t=b lambda^2>1.
The expected number of retained leaves is
lambda^(-2L), compared with b^L total leaves,
and the normalized variance tends to V_infty+1.
Thus a Bayes count decoder retains a strictly
positive asymptotic root-recovery advantage.

At b=2, lambda=4/5 the finite exact formulas
give r_L=(25/32)^L and expected observations
(25/16)^L, with the sparse variance cap
V_L(r_L)<=23/7. The recovery floor is at
least 15/23 for q=2 and 37/69 for q=3
at **every finite depth**, as well as in
the asymptotic lower limit.

The integer gap
7*B_L+16*25^L=23*32^L for
B_L=32^L V_L(r_L) is checked in
[MathCan KSR-01 PR #35](https://github.com/patrickdugan/MathCan/pull/35).

This theorem concerns independent leaf
sampling with a known original broadcast
channel. The guarantee counts observed
symbols and mutable counters, with the
static posterior-decoder cost left separate.

## Verification and promotion boundaries

The exact suite passed 15 standard-library
Python tests in the recorded GitHub Actions
run. The posterior probabilities come from
bottom-up exact rational belief propagation,
while the moment formula is also checked
by independent leaf-pair lowest-common-
ancestor covariances.

Neither source paper's infinite-tree proof,
the q=3 critical non-reconstruction direction,
nor the unrooted Ising FIID construction
is reproduced by the finite reference.
The newly trained synthetic count-monitor
pilot is reported separately in [RESULTS_B.md](RESULTS_B.md).
