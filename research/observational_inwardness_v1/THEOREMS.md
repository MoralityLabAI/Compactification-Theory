# OIC-01: hidden transition topology and an exact intervention horizon

**Status:** elementary written proofs, exact integer arithmetic, exhaustive
finite test receipts. This file has no Lean kernel certificate. The result is
a specific construction from classical nonminimal realization and
observability theory, with a restricted intervention identification bound.

## 1. Declared state and observation interface

For integer n >= 3, let

\[
S_n = \mathbb Z_2 \times \mathbb Z_n,\qquad h(x,z)=x.
\]

The first coordinate is the entire externally readable output; the second is
a hidden phase. Let a=0 represent fixed hidden state and a=1 represent
rotation. For any input u in {0,1}, use the deterministic transition

\[
 F_{a,u}(x,z)
 =
 \bigl(x\oplus1\oplus(u\land[z=0]),\ z+a\bmod n\bigr).
\]

Both model families have all 2n states initially available, with the
hidden initial phase unknown to the experimenter. Observations occur at
time zero and after each action. The external controller may choose u
adaptively from already observed outputs. The **passive interface** fixes
u=0 at every transition.

Topological invariants are computed from the directed transition
**multigraph** for the passive action. Geometric realization treats each
directed edge as a distinct 1-cell, including each of the opposite arrows
in a 2-cycle. A directed 2-cycle therefore realizes a circle with two
edges. This convention describes transition-graph topology, not the
topology of the finite discrete state space or a physical manifold.

## OIC-T01. Identical visible futures with distinct graph homology

For u=0, visible output obeys x_(t+1) = x_t xor 1. Therefore, for
every t >= 0 and every state (x,z),

\[
 h(F_{0,0}^{t}(x,z))
  = x\oplus(t\bmod2)
  = h(F_{1,0}^{t}(x,z)).
\]

The full passive future-trace quotient has exactly two classes, the
classes x=0 and x=1, and both quotient models toggle these classes.

In the fixed model each hidden phase determines one 2-cycle, so the
graph contains n disjoint 2-cycles. In the rotating model, return
requires both the visible parity and the hidden modulo-n phase to return.
Its orbit length is

\[
 L = \operatorname{lcm}(2,n).
\]

Since there are 2n states, the rotating model has 2n/L cycles: two cycles
of length n for even n, and one cycle of length 2n for odd n.
A cycle contributes one connected component and one independent
integer H_1 generator to the CW graph realization. Hence

\[
(\beta_0,\beta_1)_{\rm fixed}=(n,n),\qquad
(\beta_0,\beta_1)_{\rm rotating}=
\begin{cases}(2,2)&n\text{ even},\\(1,1)&n\text{ odd}.\end{cases}
\]

For n >= 3 these invariants differ. Conjugacy of finite permutation
systems preserves orbit lengths, so the two passive state dynamics are
inequivalent. The quotient loses their distinction completely. QED.

**Interpretation:** this is an exact hidden-mode witness relative to a
given sensor. It is an instance of classical unobservable state, and
supplies no evidence of experience or consciousness.

## OIC-T02. Sharp worst-case intervention identification cost

For every integer n >= 2, when the experimenter can choose the
registered input u, but begins ignorant of the hidden initial phase,
the minimum worst-case number of action opportunities that guarantees
identification of the model family is

\[
\boxed{T^*(n)=n.}
\]

**Upper bound.** Choose u=1 for n successive transitions. For this
probe, the visible bit remains unchanged precisely when z=0. In the
fixed model, the number of unchanged transitions is either zero or n:
if z starts nonzero it stays nonzero, and if z starts at zero it stays
at zero. In the rotating model, z visits zero exactly once over n
successive transitions. Its unchanged-transition count is one.
For n >= 2 the possible counts are disjoint, so T*(n) <= n.

**Lower bound, including adaptive policies.** Consider any T<n.
Choose hidden phase z=1 initially in both models, with the same
initial visible bit. In the fixed model z remains nonzero indefinitely.
In the rotating model the phase sequence before the first n
transitions is 1,2,...,n-1. Consequently the hidden-zero indicator
vanishes in both models throughout the first T transitions. Their
visible output histories therefore agree for every action word.
Under an adaptive controller, identical observed histories force
identical decisions along this indistinguishable pair. Thus every
policy using fewer than n opportunities fails on an admissible
pair of hidden initial phases, proving T*(n) >= n. QED.

The budget counts action opportunities (including choosing u=0);
the upper-bound strategy uses n actual active probes. Different
sensors, a phase-reset capability, a known hidden initial state,
or alternative coupling actions may alter this result. There
is no neural-network complexity lower bound here.

## OIC-T03. Separate contextual sheaf obstruction

Consider three binary measurements a,b,c with allowable pair
contexts constrained by

\[
a\oplus b=0,\quad b\oplus c=0,\quad c\oplus a=1.
\]

Each pair has two allowed local assignments. Give each context
the uniform distribution over its allowed assignments. Every
single-variable marginal is uniform, and therefore the local
empirical models agree on overlaps.

A global assignment would have to satisfy all three equations.
XOR of the left sides is zero, whereas XOR of the right sides is
one, making the global assignment impossible.

For the cycle graph C3 with F2-valued parity edge labels, the
XOR around the cycle is one. Every coboundary of vertex values
has zero cycle sum. The parity labels consequently represent a
nonzero cohomology class in H^1(C3;F2). This small example gives a
transparent local-to-global obstruction of the kind studied in
the sheaf account of contextuality by Abramsky and Brandenburger.

This **contextuality** obstruction and the **hidden-state** example
in T01 are different mathematical mechanisms. In particular, the
contextual example does not imply a hidden dynamical cycle or
intrinsic mentality.

## Verification contract

The reference model enumerates all 2n states and checks output-trace
equivalence, transition cycle lengths and graph Betti numbers. For
n=3,...,8, it exhausts every binary **fixed** action schedule up to
length n to find the first universally separating schedule. It also
checks the explicit pair used for the adaptive lower bound over
every shorter action word. Both tests are independent of the analytic
argument for arbitrary n.

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
python research/observational_inwardness_v1/run.py --check
~~~

The pinned data in results/reference.json contains finite reference
results and the separate contextual parity witness.

## Existing work and forward experiment

- [6GW-T01](../../6GW/THEORY.md) already defines exact
  intervention-preserving finite quotients and implements partition refinement.
- [LSV-10](../levin_sieve_v1/pretrained_trm/GOAL_HORIZON.md) proves
  linear goal-history rank and switched affine observability statements.
- [RC-01](../recurrent_circuits_v1/THEORY.md) treats exact
  intervention-respecting equivalence for binary recurrent circuits.
- [Intervention-preserving compactification](../../note/intervention-preserving-compactification.md)
  gives the approximate near-fibre framework and resource accounting.

OIC-01 adds transition-graph topology on *discarded fibres*
and an exact worst-case intervention horizon. A neural follow-up
would register two hidden mechanisms with matched ordinary readouts,
declare all recurrent state and caches, and test an explicit probe
family at matched compute and monitor budget. A proof of this toy
system makes no claim about trained transformer dynamics.

## References

M. Alan Kazlev, The Lost Insight of Cosmology, November 10, 2026:
https://cosmicevolution.substack.com/p/the-lost-insight-of-cosmology

S. Abramsky and A. Brandenburger, The Sheaf-Theoretic Structure
of Non-Locality and Contextuality (2011):
https://arxiv.org/abs/1102.0264

Classical system identification and minimal realization:
see the repo's LSV-10 and RC-01 for assumptions and context.


## OIC-T04. Exact Bayesian discrimination curve and adaptive optimality

Assume the two model families have equal prior probability 1/2, while
the initial hidden phase is independently uniform on Z_n in either model.
The initial visible bit is observed. Let R_T be the optimal probability
of choosing the wrong family after T opportunities, using the declared
binary probes and allowing adaptive policies.

The exact Bayes risk is

\[
\boxed{
 R_T=
 \begin{cases}
  \tfrac12,&T=0,1,\\
  \tfrac{n-T}{2n},&2\leq T\leq n,\\
  0,&T\geq n.
 \end{cases}
}
\]

**Proof.** Continuously set u=1, and extract from consecutive visible
readouts the indicator b_t=[z_t=0], since x remains unchanged exactly
when b_t=1. For T between 2 and n, the fixed model gives the all-zero
indicator word with probability (n-1)/n and the all-one word with
probability 1/n. The rotating model gives the all-zero word with
probability (n-T)/n and gives each of the T one-hot indicator words
with probability 1/n. Their support overlaps only on the all-zero
word. Thus their total-variation distance is T/n, and with equal
model priors the minimum classification error is (1-TV)/2 =
(n-T)/(2n). For T=1, both models yield b_0=1 with probability
1/n, hence their distributions coincide and the risk remains 1/2.
At T=n, the supports are disjoint.

**Adaptive optimality.** Hidden-phase evolution does not depend on
the action. The all-one probe schedule reveals the entire binary
indicator sequence b_0,...,b_(T-1). Given that sequence, an observer
can *simulate* any other admissible adaptive strategy: reproduce its
chosen actions, and recursively compute the visible outputs that
strategy would have seen. Every alternative observation transcript
is therefore a function of the full-probe transcript, and can carry
no more information about the model family. The displayed Bayes
risk is optimal among adaptive and randomized strategies. QED.

**Consequence.** A first active probe yields exactly zero statistical
advantage under this prior, even though the two internal graphs differ.
From the second probe onward the total-variation distance grows
linearly until complete discrimination at n. This illustrates how
prior assumptions affect average-case detection while OIC-T02 gives
a prior-free worst-case guarantee.

The exact Fraction-based implementation and exhaustive fixed-schedule
search are in bayes.py and test_bayes.py. No stochastic numerical
approximation or machine-learning evaluation is involved.

## OIC-T05. Fixed visible quotient with unbounded detection depth

For every integer K, choose n > max(K,2). OIC-T01 yields the same
two-class passive observational quotient for both systems, while
OIC-T02 requires more than K action opportunities for worst-case
identification. Therefore no finite universal bound on the required
probe horizon can depend solely on the cardinality of that passive
quotient. This follows directly from the constructed family.
