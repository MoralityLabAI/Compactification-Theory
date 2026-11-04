# QO-05: Intrinsic metric entropy bounds for goal-aware macrostate capacity

**2026-11-04. Mathematical status:** two elementary covering/packing inequalities and a constructive certificate for affine rate maps, backed by exact rational CPU witnesses. The inequalities are standard metric-entropy mathematics applied to the narrowly defined **absorbing-goal hit/hold** quotient contract. There is no novelty claim for covering-number inequalities, and no neural-model training or biological measurement.

See [QO-04](GEOMETRIC_COMPLEXITY.md) for the full known-kernel geometric equivalence, [QO-03](MULTIACTION_COLORING.md) for variable-action graph-coloring hardness, and [QO-02](HIT_RATE_DP.md) for the single-action dynamic program.

## 1. Metric covering numbers become exact mutable-state bit budgets

Let \(P=\{v_i\}_{i=1}^n\subseteq[0,1]^d\) be the **d-action hit-probability vectors** of the microscopic transient states. All non-hit probability remains at the same transient state; a separate absorbing goal \(\dagger\) has a distinct Boolean goal label.

At tolerance \(\tau\ge0\), set \(r=2\tau\). Define the **closed-cube covering number**

\[
N_\square(P,r)=\min\bigl\{k:\text{some }k\text{ closed axis-parallel }d
\text{-cubes of side }r\text{ cover }P\bigr\}.
\]

The QO-04 geometry theorem gives a *literal exact equality* between \(N_\square(P,r)\) and the smallest number of transient macrostate classes achieving worst-case one-step projected TV defect at most \(\tau\), with exact goal readout. Therefore the minimal fixed-length **mutable macrostate label** is

\[
\boxed{b_{\min}(\tau)
=\left\lceil\log_2\bigl(1+N_\square(P,2\tau)\bigr)\right\rceil.}
\tag{1}
\]

The additional 1 counts the absorbing goal. This bound charges no bits for the quotient decoder, rate table, model selection, latent map, inference machinery, or interface; those must be measured separately in compact-AI applications.

## 2. Packing lower bound

Let \(M(P,r)\) be the maximum cardinality of a subset \(Q\subseteq P\) such that \(\|u-v\|_\infty>r\) for every two distinct points in \(Q\). The inequality is **strict** because a pair at distance exactly \(r\) can occupy one closed box of side \(r\).

**QO-05/T1.**

\[
\boxed{M(P,r)\le N_\square(P,r).}\tag{2}
\]

**Proof.** Any side-\(r\) cube contains at most one point from a pairwise \(>r\)-separated subset, by comparing coordinate ranges. Consequently at least \(M(P,r)\) covering cubes are needed. QED.

The exact packing number is a computationally demanding clique-like quantity in general. The supplied exponential oracle is explicitly limited to at most 16 rate vectors and is **verification instrumentation**, not a scalable packing algorithm.

## 3. Latent Lipschitz upper bound and intrinsic dimension

Assume the \(n\) observed rate vectors factor through an \(m\)-dimensional latent space:

\[
v_i=\phi(u_i),\quad u_i\in[0,1]^m,\quad
\|\phi(u)-\phi(v)\|_\infty\le L\|u-v\|_\infty
\quad\forall u,v\in[0,1]^m,
\tag{3}
\]

where \(L\ge0\) is a **global certified** Lipschitz constant. Here the number \(m\) of latent coordinates can be much smaller than the number \(d\) of actions.

For \(\tau>0\) let

\[
q=\max\left(1,\left\lceil\frac{L}{2\tau}\right\rceil\right).
\]

Partition \([0,1]^m\) into \(q^m\) coordinate grid cells of side \(1/q\), assigning grid boundaries consistently. For points \(u_i,u_j\) in one grid cell, \(\|u_i-u_j\|_\infty\le1/q\). Condition (3) implies their rate vectors have diameter at most \(L/q\le2\tau\). Their images therefore form one admissible macrostate cluster.

**QO-05/T2 (constructive intrinsic-cover bound).**

\[
\boxed{
N_\square(P,2\tau)\le q^m,\qquad
b_{\min}(\tau)\le
\left\lceil\log_2(1+q^m)\right\rceil.}
\tag{4}
\]

**Proof.** The latent-grid cells partition the finite input states into at most \(q^m\) groups. Each group's image fits inside a rate-space axis-aligned cube of side \(2\tau\) by the Lipschitz inequality. Apply the QO-04 geometric quotient equivalence and (1). QED.

If \(L=0\), every rate vector is identical, \(q=1\), and exactly one transient macrostate suffices. For fixed positive \(\tau,L\), the mutable-state capacity bound grows as \(O(m\log(1+L/\tau))\) and is independent of the number of microscopic states \(n\) and number of action coordinates \(d\). This is an **existence bound** given a certified latent factorization, rather than an algorithm that discovers the factorization from examples.

## 4. Tight finite witness

Let \(m=d=2\), \(\phi\) be the identity map, and let

\[
U=\{0,1/2,1\}^2,\qquad \tau=1/5.
\]

Then \(L=1\), \(q=\lceil 1/(2/5)\rceil=3\), so T2 gives \(N_\square\le3^2=9\). Every pair of distinct grid points differs by at least \(1/2\) in one coordinate, strictly more than \(2\tau=2/5\). T1 gives \(N_\square\ge9\). Therefore

\[
\boxed{N_\square(U,2/5)=9,\qquad b_{\min}=4.}
\]

This proves that the exponential dependence on intrinsic dimension \(m\) in (4) is **attainable** for a finite family, even though many other latent configurations compress much further.

## 5. A high-action, one-latent-dimensional witness

Use \(n=41\) latent parameters \(u_i=i/40\), and **12** distinct action-rate coordinates

\[
p_i^{(a)}=\frac{a}{12}u_i,\qquad a=1,\ldots,12.
\]

The global affine Lipschitz constant is \(L=1\) in \(L^\infty\), because the largest slope is 1. At tolerance \(\tau=1/8\), \(q=\lceil 1/(1/4)\rceil=4\), so four transient macrostate classes and **three total label bits** suffice for 41 microscopic states and 12 independently observable actions. The exact arithmetic certificate checks every class' actual rate-space diameter and one-step TV radius.

For a general affine factorization \(\phi(u)=Au+c\), the code certifies the global Lipschitz bound \(L=\max_a\sum_j|A_{aj}|\), and it checks the full cube range \([0,1]^m\) by exact affine extrema before issuing a certificate. Checking only sampled pairs is separately labelled as a **necessary finite-sample condition** that cannot establish global Lipschitz continuity.

## 6. Evidence ledger and controls

- [intrinsic_entropy.py](intrinsic_entropy.py): exact rational latent-grid quotient, global affine rate certificate, one-step TV audit, and an exponential finite packing oracle.
- [test_intrinsic_entropy.py](test_intrinsic_entropy.py): the tight nine-state two-dimensional grid, the 41-state/12-action compression, a mixed affine map, a constant map, strict packing boundaries, monotonicity, invalid contracts, and a case whose observed sample rates are valid while the affine map leaves the probability cube elsewhere.
- CI: the existing \`ingression-autonomy.yml\` runs all unit tests for this research directory; no specialized dependencies or model calls are required.

A research application would freeze the latent map and its Lipschitz certificate before evaluating new intervention families. A large compression gain obtained by fitting the map on the test outcomes is a post-selection artifact. The rate-factorization and its decoding cost, including potential table memorization, must be measured alongside mutable state bits.

The same mathematical framing is **compatible** with Levin-inspired questions about reusable organization across different substrates. It does not establish an underlying Platonic realm or a biological free lunch.
