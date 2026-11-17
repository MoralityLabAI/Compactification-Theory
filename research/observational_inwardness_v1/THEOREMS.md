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


## OIC-T06. Exact error/erasure robustness law

The preceding results assume perfect per-step measurements. We now
**change the measurement interface** to a single audit channel. At
each transition, the evaluator receives a bit reporting whether the
visible state remained unchanged. In the clean system this bit equals

\[
b_t=u_t\,[z_t=0].
\]

The evaluator does **not** receive an additional, uncorrupted copy of
the raw visible state x_t. Otherwise the observer could reconstruct b_t
and bypass corruption. Before the verdict, an adversary may replace at
most e audit bits by their opposites, and independently erase at most r
audit bits with known erased positions (symbol ⊥). There is one
global budget (e,r) per observation transcript. The model and hidden
initial phase remain unknown; the intervention sequence can depend on
previous reported audit symbols.

**Theorem.** For all integers n>=2 and e,r>=0, the minimum worst-case
number of transition opportunities that allows guaranteed
identification of the fixed and rotating model families under this
audit channel is

\[
\boxed{T^*_{e,r}(n)=n(2e+r+1).}
\]

**Proof, step 1 (coding separation).** For a fixed intervention
schedule, regard the two model families as sets C_A and C_B of
binary length-T words, one codeword per possible hidden initial
phase. A common corrupted audit transcript can come from codewords
a∈C_A and b∈C_B if and only if their Hamming distance satisfies

\[
d_H(a,b)\leq 2e+r.
\]

For sufficiency, erase up to r of their disagreement positions and
split the remaining at most 2e differences between the two
possible corruption patterns, using at most e substitutions for
each model. For necessity, each unexplained disagreement requires
at least one substitution on one of the two clean words; erasures
can remove at most r such disagreements. Thus zero-error
identification is possible exactly when the minimum interfamily
distance is at least 2e+r+1.

**Step 2 (distance of the fully probed interface).** Choose u_t=1
on every transition. The fixed family's codebook contains
0^T (all initial phases z≠0) and 1^T (initial phase z=0).
For the rotating family each initial phase produces a pulse
once per n transitions, so each rotating codeword has either
floor(T/n) or ceil(T/n) one-bits. Distances to the fixed
all-zero codeword are the pulse counts, and distances to the
fixed all-one codeword are T minus the pulse counts. Since
n>=2,

\[
T-\lceil T/n\rceil\geq\lfloor T/n\rfloor.
\]

Consequently,

\[
\boxed{d_{\min}(C_A,C_B)=\lfloor T/n\rfloor.}
\]

The coding separation condition is therefore equivalent to
floor(T/n)>=2e+r+1, or T>=n(2e+r+1).

**Step 3 (adaptive optimality).** Hidden-phase evolution z_t
does not depend on u_t. Under continuous probing, the clean
signal is the entire hidden-zero indicator stream. For any
alternative adaptive policy, reconstruct its decisions from a
fully probed *possibly corrupted* transcript recursively:
at a step where its decision is u_t=1, pass the actual audit
symbol (including any erasure); if it chooses u_t=0, pass a
perfect zero. The resulting transcript is feasible for that
adaptive policy under an error/erasure budget no greater than
the original (e,r), as only reports at u_t=1 retain possible
corruptions. Any flawless decoder for the alternative policy
would thus yield a flawless decoder for continuous probing.
Since the continuous-probe codebooks are inseparable below
the bound, adaptivity cannot improve the guaranteed
identification horizon. Continuous probing attains it. QED.

The proof requires that the defender's action choice has no
effect on the hidden-phase update, that corruption applies
to the specified **audit symbols**, and that missing positions
are explicitly marked. A different sensor, raw x_t corruption,
unreported erasures, or a phase-reset action requires a new
distance calculation.

## OIC-T07. Sharp nonadaptive trusted-probe budget

Suppose a fixed observation schedule contains q active probes
over any number T>=q of time steps. Group its active probe
indices into the n residue classes modulo n, with counts
h_0,...,h_(n-1), whose sum is q. Every rotating hidden phase
produces a codeword with ones precisely on one residue class.
Comparing that codeword with fixed model's all-zero audit
word gives distance h_j for the corresponding phase. Thus

\[
d_{\min}\leq\min_j h_j\leq\lfloor q/n\rfloor.
\]

To correct e substitutions and r known-position erasures for
all hidden phases, the necessary active-probe budget is

\[
\boxed{q\geq n(2e+r+1).}
\]

This is sharp: use exactly q=n(2e+r+1) consecutive active
probes. Each residue class is sampled 2e+r+1 times. The
distance to the fixed all-one codeword is at least as large,
and OIC-T06 applies. This particular *resource* claim concerns
fixed (nonadaptive) schedules; OIC-T06 separately covers
adaptive policies under a bounded time horizon.

## Executable evidence, lineage, and remaining work

The standard-library implementation is robust.py, with
test_robust.py and robust_run.py. Tests verify exact codebook
distances, construct valid ambiguous receipts when the
distance condition fails, and exhaustively compare all
binary nonadaptive probe schedules for small n and T.
The pinned JSON receipt records 25 (n,e,r) configurations,
including the exact boundary T-1 and T. Tests also run
with Python optimization enabled.

The mathematics uses classical coding-theoretic minimum
distance ideas in a narrow finite dynamical interface.
Related work on sensor attacks and observer reconstruction:

- Chong, Wakaiki and Hespanha, *Observability of Linear
  Systems Under Adversarial Attacks*, IEEE ACC (2015):
  https://web.ece.ucsb.edu/~hespanha/published/19_observability-adversarial-acc-submit-revised.pdf
- Dean, Angluin et al., *Inferring Finite Automata with Stochastic
  Output Functions and an Application to Map Learning*,
  Machine Learning 18 (1995):
  https://doi.org/10.1023/A:1022874607797

A next empirical step is to test whether recurrent AI
monitors exhibit independently validated phase-recurrence
signatures and how expensive it is to audit them. These
finite examples establish no current neural architecture's
hidden-state topology or monitor error budget.


## OIC-T08. Sharp robustness law for corrupted raw state observations (even n)

**New sensor contract.** The evaluator now receives the *raw visible
state* x_t at every time t=0,...,T, with a total budget of e adversarial
bit substitutions and r marked erasures of individual state samples.
The interventions u_t are selected in advance (nonadaptive fixed
schedule). The hidden initial phase remains unknown and the initial
visible state x_0 is fixed and common to both models. This differs from
OIC-T06's transition-level fault model.

For a binary action word w=(u_0,...,u_(T-1)), define the raw codebooks

\[
C_A(w)=\{(x_0,\ldots,x_T):z_0\in Z_n,\ a=0\},\quad
C_B(w)=\{(x_0,\ldots,x_T):z_0\in Z_n,\ a=1\}.
\]

Let d(w)=min_(a∈C_A,b∈C_B) d_H(a,b).

**Theorem (even n).** For each even integer n>=2 and T>=0,

\[
\boxed{
\max_{w\in\{0,1\}^T}d(w)=\max(0,T-n+1).
}
\]

Consequently, the minimum length of a nonadaptive fixed schedule
guaranteeing identification with <=e corruptions and <=r erasures is

\[
\boxed{T^*_{\rm raw}(n,e,r)=n+2e+r.}
\]

**Upper bound.** For any schedule and n>=2, compare the fixed
model with nonzero hidden phase z_0=1 to the rotating model with
z_0=1 and the same x_0. In the fixed model, the phase never reaches
zero; in the rotating model, z_t=1+t (mod n), whose first zero
occurs at t=n-1. Until transition t=n-1 the visible trajectories
are identical for *any* action word. Therefore the first n raw
samples x_0,...,x_(n-1) are equal in these two admissible
initial conditions, giving d(w)<=max(0,T-n+1) for T>=n-1.
When T<n there are no differences at all, so d(w)=0.

**Attainment for even n.** For T>=n take w=1^n 0^(T-n).
In the active prefix each rotating initial phase visits z=0
exactly once, whereas the fixed model's hidden phase visits
zero either zero or n times. Since n is even, those two
pulse counts have opposite parity for every possible pair
of hidden initial phases. The visible state at time n
therefore differs across *every* fixed/rotating pair.
After time n, u=0 makes both models toggle x at each
transition, preserving their visible disagreement at
times n,...,T. Thus d(w)>=T-n+1. The upper bound
shows equality.

A common corrupted raw transcript from two codewords is
possible exactly when their Hamming distance is <=2e+r,
using the same elementary error/erasure collision
construction as OIC-T06. Thus guaranteed decoding
requires d(w)>=2e+r+1. The maximum-distance formula
gives T>=n+2e+r, attained by the stated word. QED.

## OIC-T09. Exact active-probe budget and the odd parity obstruction

**Minimal active probes for even n.** Any fixed action word
with q<n active interventions misses at least one phase
residue class modulo n. The rotating initial phase which
passes through zero only on that residue class then produces
an all-zero hidden pulse signal for the complete word.
That visible trajectory equals the fixed model with any
nonzero constant phase. Consequently d(w)=0 when q<n,
for any T. The OIC-T08 attaining word contains exactly n
active probes regardless of e and r. The minimum
nonadaptive active-probe count is therefore

\[
\boxed{q^*_{\rm raw}(n,e,r)=n,\quad n\text{ even},}
\]

when the observation horizon may grow. Here the passive
state samples are still collected and their sensor costs
must be charged; q counts active interventions alone.

**Terminal parity obstruction for odd n.** Let q be the
number of active interventions in an arbitrary fixed
word w. The fixed model's final-state parity relative
to the baseline x_0 xor (T mod 2) is either 0 (nonzero
initial hidden phase) or q mod 2 (zero initial phase).
Let h_j be the number of active probes in phase residue
j mod n, with sum_j h_j=q. Each rotating initial phase
has final-state parity h_j mod 2.

For final raw state alone to separate the model families,
the fixed model must have only parity 0, hence q must be
even. All n rotating phase counts h_j would then need
to be odd, so their sum would have parity n mod 2.
For odd n this contradicts q even. If q is odd, the
fixed model already produces both terminal parities,
which must overlap the nonempty rotating family.
Therefore, for every odd n>=3 and every fixed action word,

\[
\boxed{
\{x_T^A\}\cap\{x_T^B\}\neq\varnothing.
}
\]

This obstruction concerns the *final-state readout*.
Entire histories may still distinguish the families,
and repeated nonadaptive probes can overcome finite
fault budgets. The odd-cycle result is a parity
constraint on this particular interface, rather than a
claim about general topology or phenomenal inwardness.

**Finite odd examples.** Independent exhaustive fixed
schedule searches give the maximum interfamily raw
distance for n=3 at horizons T=3,4,5,6,7 as
1,1,2,2,3. Thus correcting one corrupted raw-state
sample first becomes possible at T=7 rather than
the even-n formula's value 5. For n=5 with one
corrupted sample, the optimum distance reaches 3
already at T=7. For n=3, two substitutions require
distance 5 and are first possible at T=10 in the
exhaustive range. The results depend on the model's
exact phase count and available probe schedules.

## OIC-T10. Correlated audit errors induced by raw-state noise

For any clean raw state sequence x_0,...,x_T,
the noiseless inferred pulse is

\[
b_t=1\oplus x_t\oplus x_{t+1}.
\]

Suppose the raw channel substitutes state bits
according to an error indicator f_t, so
\(\widetilde x_t=x_t\oplus f_t\). The derived audit
error is

\[
\boxed{
\widetilde b_t\oplus b_t=f_t\oplus f_{t+1}.
}
\]

A single interior raw bit substitution affects the two
neighboring derived audit symbols. A corrupted boundary
sample affects one. Multiple adjacent raw substitutions
can cancel in their shared inferred audit symbol.
A marked raw-state erasure removes at most its two
neighboring derived audit symbols.

Therefore at most e raw substitutions and r erased
state samples induce at most 2e audit substitutions and
2r audit erasures. Applying OIC-T06 to these upper
bounds yields the conservative sufficient horizon

\[
T\geq n(4e+2r+1).
\]

OIC-T08 performs direct raw-trace decoding and proves
the sharper exact horizon n+2e+r for even n. Both
budgets concern a single declared measurement channel;
monitor architecture, costs and independence assumptions
must be specified before comparing fault rates.

## Reproduction and evidence

- raw_sensor.py: raw-trace codebook, exact cross-family
  distance, odd parity final-state test, raw/audit
  conversion, adversarial common-receipt generator and
  exhaustive fixed-schedule optimizer.
- test_raw_sensor.py: independent finite checks for
  the even-n optimality, odd-n obstruction and raw
  fault-correlation formula.
- raw_run.py --check: deterministic exact JSON
  replay for 20 even-phase robustness configurations
  and three odd-phase counterexample witnesses.

These are finite deterministic calculations plus
handwritten mathematical proofs. Generalizing to
learned recurrence would require identifying a
realistic sensor interface and a phase-type hidden
mode with independently measured dynamics. A
realistic floating-point or probabilistic corruption
model needs a separate analysis.
