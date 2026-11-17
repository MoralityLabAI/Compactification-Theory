# SCT-01: executed SAT core covers, hitting-time and streaming proof results

Date: 2026-11-14.
Evidence: **EXACT_FINITE_SAT_CORE_CONTROL** and
**FROZEN_SYNTHETIC_SAT_STREAM_CONTROL**.

[Full mathematical derivation](THEORY.md) |
[64-assignment SAT oracle and exact probabilities](reference.py) |
[Bounded stream gym](arena.py) |
[Recorded protocol](PROTOCOL.md).

Verified source run:
https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43562983433

The preceding independent source-code runs and the latest branch
are subject to separate CI after documentation commits. The source
run above passed **29 Python tests**, checking both the
full SAT oracle and the bounded streaming gym.

## Source pair and limits

OpenAI Math #175 formalization scope reports integral/fractional
expectation thresholds within a universal factor, with the same
cover budget 1/2. OpenAI Math #235 covers full uniform proper
random k-SAT, with replacement, at increasing n and
large-n limiting thresholds/hitting-time properties.

This experiment uses a **restricted two-gadget six-variable
3-SAT clause pool**: eight clauses on variables 0,1,2 and eight
on variables 3,4,5. Each group contains all eight signed proper
clauses, one excluding each local assignment. Its exact finite
probabilities are independent results and must not be interpreted
as finite-n estimates of #235's full uniform six-variable model.

## Exhaustive minimal contradiction structure

All 65,536 distinct clause subsets are evaluated by two
independent routes: an explicit 64-assignment SAT bitset,
and the analytical criterion that at least one complete
eight-clause gadget is present. They agree on every subset.

| Formula subset property | Exact count |
|---|---:|
| Satisfiable subsets | 65,025 |
| Unsatisfiable subsets | 511 |
| Inclusion-minimal UNSAT cores | 2 |
| Clauses per minimal UNSAT core | 8 |

For r disjoint complete eight-clause gadgets the only
minimal UNSAT cores are those r gadgets.

## Exact cover and probability thresholds

The p-cost of the optimal integral or fractional
containment cover, including allowable empty-set weights, is

\[
\boxed{C(p)=\min(1,2p^8).}
\]

The cover-budget-1/2 threshold is

\[
q_{\rm cover}=4^{-1/8}\approx0.8408964153.
\]

Independent Bernoulli p inclusion of the sixteen
clause types gives

\[
\Pr_p(\mathrm{UNSAT})=2p^8-p^{16}.
\]

Its median is

\[
p_{1/2}=(1-1/\sqrt2)^{1/8}\approx0.8577064275.
\]

The integral/fractional cover optima coincide while
the probability-median threshold is higher.
At p=1/2, exactly 511 of 65,536 subsets are UNSAT,
so the event probability is 511/65536 and the
two-witness cover cost is 1/128.

## Exact with-replacement hitting times

For t independent draws from the fixed sixteen-clause
universe, the first UNSAT stopping time H has
an exact inclusion-exclusion CDF. An independent
Stirling-surjection expression computed from the
full 65,536-set winning histogram matches it.

| Draw count t | Exact Pr(H≤t), decimal |
|---:|---:|
| 8 | 0.0000187755 |
| 16 | 0.0274303881 |
| 24 | 0.2266936863 |
| 30 | 0.4680035533 |
| 31 | 0.5077259498 |
| 32 | 0.5462160039 |
| 40 | 0.7885357221 |
| 48 | 0.9140676960 |
| 64 | 0.9883053228 |

Thus the median is exactly **31 draws**. The mean
and variance are exact rational values:

\[
\boxed{
\mathbb EH=\frac{1481069}{45045}\approx32.879765,
\qquad
\mathrm{Var}(H)=\frac{80211997256}{676350675}
\approx118.595279.
}
\]

A separate backwards Markov dynamic program over all
65,536 presence sets reproduces both moments within
floating-point tolerance.

As a calibration matching the source full proper
random 3-SAT model at n=3, the eight-clause universe
has ordinary coupon-collector hitting-time mean
761/35 and variance 838034/11025.

## Exact deterministic monitor states

For r disjoint complete gadgets with k clauses each,
the Myhill–Nerode continuation proof gives

\[
\boxed{
N_{\mathrm{decision}}=(2^k-1)^r+1,
\qquad
N_{\mathrm{certifying}}=(2^k-1)^r+r.
}
\]

The source constructs deterministic automata that attain
both bounds. Every incomplete clause-presence tuple has
a unique base-(2^k-1) encoding, and a completed gadget
enters an absorbing state. For the certifier there is one
accepting state for each valid public gadget index.

At r=2,k=8 the exact state counts are 65,026 and
65,027, requiring **16 bits of mutable memory** each.
The completed eight-clause contradiction can be
referenced by a one-bit gadget index given the public
static clause catalog and trusted verifier.

Tests cover every incomplete presence state and all
sixteen possible next clause arrivals, plus arbitrary
repeated streams. An exhaustive small Myhill–Nerode
check separately tests every pair of incomplete
states for two three-clause gadgets.

## RefutationBench-style streaming gym

The scripted gym samples 1,024 independent streams
per horizon using a frozen seed and the exact
six-variable restricted clause distribution.
Each monitor sees identical clause arrivals.

| Draws | Exact UNSAT probability | Both 16-bit exact policies | First-gadget 8-bit accuracy | Arrival-count 8-bit accuracy |
|---:|---:|---:|---:|---:|
| 8 | 0.0019% | 100% | 100% | 99.02% |
| 16 | 2.74% | 100% | 98.34% | 3.22% |
| 24 | 22.67% | 100% | 89.45% | 21.58% |
| 31 | 50.77% | 100% | 77.25% | 50.49% |
| 32 | 54.62% | 100% | 78.42% | 52.05% |
| 40 | 78.85% | 100% | 74.51% | 79.10% |
| 48 | 91.41% | 100% | 77.34% | 90.23% |
| 64 | 98.83% | 100% | 89.94% | 98.73% |

The accuracy columns refer to the **sampled 1,024
streams**, while the probability column is the exact
source-law population CDF.

At t=31 the first-gadget monitor incurred 233
false negatives and no false positives.
The arrival-count monitor incurred 507 false
positives and no false negatives, because eight
arrivals can include repeats of the same clause.

The exact proof-producing monitor returned
valid public gadget identifiers on every positive
case. Its host verifier checked the claimed
complete clause set, and no proof was accepted
without all eight distinct clauses having
appeared. A sequence repeating one clause
eight times is an explicit adversarial false
positive for raw counting.

## Formal counterpart and scope

[MathCan PR #39](https://github.com/patrickdugan/MathCan/pull/39)
kernel-checks the eight-clause local SAT semantics,
a disjoint-gadget composition predicate, the
two-gadget state-count arithmetic and a unique
base-255 encoding of incomplete pairs.
Its full Lean axiom audit and the prior rational
regressions are separately tracked in that PR.

The complete general fractional-cover proof,
hitting-time probability derivations and
unbounded-stream Myhill–Nerode proof reside
in THEORY.md with independent finite tests.
The original OpenAI #175 universal comparison
and #235 large-n random-SAT theorems are
not reconstructed here.

No trained Lean reasoner or language-model
checkpoint was evaluated. The in-process
Python gym is a deterministic evaluation
control and provides no isolation guarantee
against arbitrary malicious Python code.
