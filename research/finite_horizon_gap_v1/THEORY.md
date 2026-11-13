# FHC-01: Finite-horizon compactification gap

Date: 2026-11-13. **Status:** conditional mathematical derivations, exact
finite rational reference, and a proposed recurrent-semantic experiment.
The proofs here have not been Lean-checked. The OpenAI manuscripts are research
claims at different stages of external review. We import only the stated
results, and label the new elementary deductions separately.

## 0. Two source statements and their scopes

**OpenAI Math family #094**, *Subpolynomial dimension reduction in Lp*
(2026-09-23). For fixed 1<p<infinity, p!=2, D>1, every N-point subset of
real Lp has an embedding into finite-dimensional l_p^d of distortion D with

\[
d\le\exp(C_{p,D}(\log N)^{\gamma(p)}),\qquad
\gamma(p)=\begin{cases}2-p,&p<2\\1-2/p,&p>2.\end{cases}
\]

At p=4 the exponent is 1/2. Scope: https://github.com/openai/math/blob/main/lean/docs/094.md .
The comparator challenge has admitted statements; it is not itself a
finished kernel proof.

**OpenAI Math family #098**, *Compact counterexamples to bi-Lipschitz
dimension reduction* (2026-09-25). The released scope includes a compact
doubling subset K of an infinite-dimensional real Banach space (hence
one may choose Hilbert space) with no bi-Lipschitz embedding into any
finite-dimensional normed space at any finite distortion.
Scope: https://github.com/openai/math/blob/main/lean/docs/098.md .
Use the CompactBanach companion for compactness; the separate
DoublingHilbert comparator statement omits that hypothesis.

## 1. Gaussian transfer into L4 (FHC-G)

Let H=l_2(N), with independent standard Gaussian random variables
G_j on a common probability space. Define, initially for finitely
supported x,

\[
Jx=3^{-1/4}\sum_jx_jG_j.
\]

Fourth moments give

\[
E|\sum_j(x_j-y_j)G_j|^4
=3(\sum_j|x_j-y_j|^2)^2,\quad
\boxed{\|Jx-Jy\|_{L4}=\|x-y\|_2}.\tag{FHC-G}
\]

The series extends by completion in L4 to all H. We can therefore
apply the p=4 case of #094 to finite subsets of the Hilbert-space
witness in #098. This explicitly uses #094's p != 2 scope.

## 2. Finite witness compactness lemma (FHC-C)

Let K be compact metric, Y a finite-dimensional normed space, and D>=1.
If every finite F subset K admits some map f_F:F->Y and scale a_F>0 with

\[
a_Fd(x,y)\le\|f_F(x)-f_F(y)\|\le Da_Fd(x,y),
\]

then all K admits a D-bi-Lipschitz embedding into Y.

**Proof.** Pick dense x_0,x_1,... . On each finite prefix scale the
embedding by 1/a_F and translate its image of x_0 to zero. Every
image is bounded by D diam(K). Finite-dimensional closed bounded
balls are compact, so diagonalize to get convergent images for every
fixed x_j. Both pairwise distance inequalities survive limits.
The resulting D-Lipschitz map on the dense sequence extends to K,
and its lower inequality gives injectivity. QED.

**Finite obstruction:** If K admits no D embedding into Y, some
finite A subset K already has no D embedding. No cardinality
bound follows without further quantitative geometry.

## 3. Main combined theorem (FHC-1)

Assume #094 and #098. There exists a compact doubling
K subset l_2 such that, for every fixed D>1 and any dense
sequence x_1,x_2,... in K, the least dimension d_D(N) needed
to embed the first N points into l_4^{d_D(N)} at distortion D obeys

\[
\boxed{d_D(N)\to\infty,\qquad
d_D(N)\le\exp(C_D\sqrt{\log N})\quad(N\ge2).}\tag{FHC-1}
\]

Apply FHC-G and #094 for the upper bound. Prefix containment makes
d_D(N) nondecreasing. If it stayed bounded by d, zero-pad the
coordinate embeddings to l_4^d; FHC-C would give a D embedding of K,
contradicting #098. The lower result is qualitative. Do not copy
#094's universal worst-case lower bound to this specific K. QED.

## 4. Robust net obstruction with explicit tolerance (FHC-2)

Fix d, D>=1 and alpha>1. FHC-C gives a finite
A subset K lacking any (alpha D)-embedding into l_4^d.
Let delta_A=min_{a!=a' in A}d(a,a')>0.

Suppose Q is an epsilon-net in K and f:Q->l_4^d has distortion D.
Choose q(a)∈Q within epsilon of each a∈A. Normalize the embedding
so its lower Lipschitz bound is one. Then

\[
d(a,a')-2\epsilon\le\|f(q(a))-f(q(a'))\|
\le D(d(a,a')+2\epsilon).
\]

The induced map on A has distortion at most

\[
D_{\rm transfer}
=D\frac{\delta_A+2\epsilon}{\delta_A-2\epsilon}
\quad (2\epsilon<\delta_A).
\]

Solving D_transfer<=alpha D yields

\[
\boxed{
\epsilon<\frac{\delta_A(\alpha-1)}{2(\alpha+1)}
\implies\text{every finite epsilon-net of K fails D-embedding into l_4^d}.
}\tag{FHC-2}
\]

For alpha=2 the sufficient threshold is epsilon<delta_A/6.
The separation of the obstruction is existential until its witness
is constructed. QED.

## 5. Finite-resolution positive statement (FHC-3)

If Q is an epsilon-net of K with a normalized D-embedding
f:Q->l_4^d, select q(x)∈Q with d(x,q(x))<=epsilon
and let E(x)=f(q(x)). The triangle inequality gives

\[
\boxed{
\max(0,d(x,y)-2\epsilon)
\le\|E(x)-E(y)\|_4
\le D(d(x,y)+2\epsilon).
}\tag{FHC-3}
\]

This is an additive-error representation of all K at scale epsilon.
It requires a charged nearest-net lookup or a learned selection map.
The fixed-d obstruction concerns a uniform multiplicative guarantee
as epsilon goes to zero. QED.

## 6. Doubling scale frontier (FHC-4)

Let Lambda be an intrinsic doubling constant of K and
R=diam(K)>0. A ball of radius 2R covers K. Repeatedly
halving its radius L=max(0,ceil(log_2(4R/epsilon))) times
gives an epsilon-net Q_epsilon with

\[
|Q_\epsilon|\le\Lambda^L.
\]

Since log|Q_epsilon|=O(log(1/epsilon)), #094 and
FHC-G produce a choice of finite net satisfying

\[
\boxed{
d_D(Q_\epsilon)
\le\exp(O_D(\sqrt{\log(1/\epsilon)}))
\quad(\epsilon\downarrow0).
}\tag{FHC-4}
\]

Every sufficiently fine net nevertheless needs more than any
preassigned finite d, by FHC-2. **Covering entropy can stay
logarithmic in resolution while bounded-distortion dimension
diverges.** Embedding existence does not bound model parameter bits
or the work needed to compute an encoder.

## 7. Explicit non-doubling control: equilateral packing (FHC-P)

Take N Hilbert vectors e_i/sqrt(2): all distinct pair distances
are one. An arbitrary d-dimensional normed embedding with distances
normalized into [1,D] places N disjoint radius-1/2 balls inside
a radius-(D+1/2) ball. Volume comparison gives

\[
\boxed{N\le(1+2D)^d,\quad
d\ge\log N/\log(1+2D).}\tag{FHC-P}
\]

For a one-dimensional target the optimal distortion is exactly N-1:
order real image points and add consecutive gaps of at least one.
The set of N equilateral points has doubling constant N, so this is
an **explicit non-doubling calibration**, not a recreation of #098.

## 8. Interface and neural evaluation contract

Apply to RC-01 and IC-01, with optional SEM-ECHO transfer. Build a
task metric from future observations, separate from raw activation
Euclidean distances, e.g.

\[
\Phi_T(h)=(w_1 G(F(h)),\ldots,w_T G(F^T(h))).
\]

For an infinite-horizon Hilbert metric require square-summable
weights, bounded observables, and declared state support.

At horizons 1,2,4,8,16,... measure pairwise relative distances,
maximum distortion, near-fiber failures and held-out interventions.
Compare PCA, SAE, random projections and interface-relative encoders
at matched d, numeric precision, trained parameter budget and
inference cost. Every claimed failure must retain actual state pairs
and the declared observation metric.

A fixed finite neural checkpoint lives in finite-dimensional hidden
space under its raw Euclidean metric, which trivially embeds at that
same dimension. The #098 obstruction is only relevant to an
infinite-dimensional limit geometry or separately defined future
semantic metric. No property of current Qwen, Q5 or other checkpoints
is assumed.

Status ledger:
- FHC-G, FHC-C, FHC-2, FHC-3, FHC-P: elementary supplied proofs.
- FHC-1 and FHC-4: conditional on source results #094 and #098.
- Synthetic rational checks: finite validation, not the #098 witness.
- Checkpoint experiments: protocol only until actual assets and
  preregistered run receipts exist.


## 9. Constructive recurrent control: horizon observability rank (FHC-5)

Let the latent state h belong to R^m. Define the reversible cyclic
shift F(h_0,...,h_{m-1})=(h_1,...,h_{m-1},h_0) and a scalar
readout G(h)=h_0. At horizon T, retain the observation tuple

\[
O_T(h)=(G(h),G(Fh),\ldots,G(F^{T-1}h)).
\]

Its matrix has row t equal to standard basis vector e_(t mod m).
The first min(T,m) rows are independent and every later row repeats
one of the first m. Therefore

\[
\boxed{\mathrm{rank}(O_T)=\min(T,m).}\tag{FHC-5}
\]

Suppose a linear d-coordinate encoder E and a linear future decoder
R exactly factor the observation map on all R^m, i.e.
O_T=R E. Then rank(O_T)<=rank(E)<=d, hence

\[
\boxed{d\ge\min(T,m).}
\]

Equality is attained by retaining exactly the first min(T,m)
coordinates, and for T>=m by the full state. At fixed width m=8,
the required linear ranks at T=1,2,4,8,16 are exactly 1,2,4,8,8.

This is an exact **finite-dimensional linear** lower bound.
It supplies an executable toy regression for an observation
metric that refines with horizon. Unlike FHC-1, the rank
saturates at the actual state width. Nonlinear encoders and
finite-sample approximate reconstruction have separate
constraints; FHC-5 states a linear factorization theorem.


## 10. FHC-6: Binary predictive sufficiency versus exact metric dimension

We now instantiate the recurrent-observation problem on RC-01's
six-bit affine circuit class. This is a finite result independent of
the deep #094/#098 source claims; it tests the *distinction* those
claims encourage us to measure.

Let x be uniform over GF(2)^6, let F be the reversible companion
transition

\[
F(x)_i=x_{i+1}\quad(0\le i<5),\qquad
F(x)_5=x_0\oplus x_1,
\quad G(x)=x_0.
\]

The underlying polynomial \(z^6+z+1\) is primitive over GF(2).
Equivalently, direct exact enumeration of the dual recurrence
\(r_{t+1}=r_t A\), \(r_0=e_0\), visits all 63 nonzero six-bit row
masks before repeating. Our reference enumerates and verifies this
claim without importing a finite-field library.

Write the length-T future trace as

\[
\Phi_T(x)=\bigl(G(x),G(Fx),\ldots,G(F^{T-1}x)\bigr)
\in\{0,1\}^T.
\]

**FHC-6a (minimal predictive bits, any deterministic binary code).**
For \(1\le T\le63\), the GF(2) rank of the first T row masks
is \(r_T=\min(T,6)\). Therefore \(\Phi_T\) takes exactly
\(2^{r_T}\) distinct values. Any exact future decoder receiving only
a fixed B-bit code E(x) requires

\[
\boxed{B\ge r_T=\min(T,6).}\tag{FHC-6a}
\]

A basis encoding of those row masks attains this bound and the
reference verifies its decoding on every one of the 64 states.
The cardinality argument covers nonlinear encoders too, provided
they are limited to B finite binary bits. It does not constrain a
single arbitrary-precision real coordinate.

**FHC-6b (exact Euclidean metric dimension).**
For each nonzero mask r let \(\chi_r(x)=(-1)^{r\cdot x}\).
Under uniform x the characters obey

\[
\mathbb E\,\chi_r=0,\qquad
\mathbb E\,\chi_r\chi_s=\mathbf1[r=s].
\]

The centered real-valued future coordinates satisfy
\(\Phi_T(x)_t-\tfrac12=-\tfrac12\chi_{r_t}(x)\).
Because the row masks are distinct until period 63,

\[
\boxed{\mathrm{Cov}(\Phi_T(X))=\frac14 I_T.}\tag{FHC-6b}
\]

Hence the full 64-point trace configuration has affine dimension T.
Pairwise squared Euclidean distances determine its centered Gram
matrix, so any *exact Euclidean isometry* of these 64 points into
\(\mathbb R^d\) requires

\[
\boxed{d\ge T,\quad 1\le T\le63.}\tag{FHC-6c}
\]

In particular the exact predictive quotient has **six binary bits**
at T=63, while exact Euclidean isometry of its 64 future traces
requires **63 real coordinates**. These are different resources and
different fidelity contracts.

**FHC-6d (equilateral terminal horizon).**
When T=63, every nonzero six-bit mask is present. For any x≠y,
exactly half of the 64 masks have odd dot product with x xor y, and
the zero mask is among those with even dot product. Thus exactly 32
of the 63 future bits differ:

\[
\boxed{\|\Phi_{63}(x)-\Phi_{63}(y)\|_2^2=32,\quad x\ne y.}
\tag{FHC-6d}
\]

These 64 points are the vertices of a regular simplex in its
63-dimensional affine span. The family is **not uniformly doubling**
as m increases; its finite packing obstruction is distinct from
the compact doubling witness in #098.

At distortion D>1, approximate Euclidean embedding may take far
fewer coordinates. Standard finite-set random projection estimates
depend logarithmically on N=64 rather than linearly on T, with
constants depending on D. This is the exact-isometry versus
bounded-distortion phenomenon being measured by FHC-01B.

## 11. FHC-7: Intervention alphabet expands the predictive quotient

At a one-step observation horizon, if the only permitted readout is
\(G(x)=x_0\), the equivalence partition has two classes and needs
one binary bit. If the observer may apply one of six distinct cyclic
coordinate rotations before reading x0, the available readouts are

\[
\{x_0,x_1,\ldots,x_5\}.
\]

Their GF(2) span has rank six. The minimal exact binary code for
all allowed one-step interventions therefore has six bits:

\[
\boxed{B_{\mathrm{single\ read}}=1,\qquad
B_{\mathrm{six\ reads}}=6.}\tag{FHC-7}
\]

The pair x=0, y=2 is observationally equivalent under the original
bit-zero readout and distinguishable by the bit-one probe.
This witnesses a concrete interface-dependent quotient refinement.
The reference checks the pair and all action masks, reusing the RC-01
GF(2) algebra.

## 12. Empirical FHC-01B finite pilot

The new executable benchmark fits a PCA projector from 48
preselected calibration states and compares Gaussian random
projectors and the exact minimal binary quotient. It evaluates
every pair among 16 disjoint held-out states, across the frozen
horizons 1,2,4,6,8,16,32,63. All target future traces come from
the exact RC-01 source recurrence.

The metric is Euclidean distance between finite future output
traces. Each embedding uses one scale chosen from calibration
pairs. Pairwise test ratios are retained, along with a concrete
worst-pair witness, calibrated failures at D=1.5 and D=2.0,
and the descriptive optimum rescaling distortion of the held-out
pairs. The optimum scale derived from held-out data is never used
for the primary calibrated gate.

The **exact semantic quotient code** can decode all T future
bits on every state, although its Euclidean geometry need not
preserve the future-output metric. Computing the PCA or Gaussian
code from the full future traces requires T oracle operations
per state. Its projection matrix and mean vector are charged to
the resource ledger; these baselines do not establish a cheap
neural implementation from the original recurrent state.

PCA is trained on synthetic oracle data. The pilot is neither a
trained language-model experiment nor a replication of the source
OpenAI papers. Its exact results distinguish predictive
sufficiency, exact metric isometry and approximate sampled
distortion. Additional real-checkpoint studies remain a distinct
gate.


## 13. FHC-8: Disjoint held-out simplex vertices are invisible to PCA

Let \(v_1,\ldots,v_N\) be the vertices of a regular Euclidean simplex,
with the same positive distance between every distinct pair. Choose a
proper training subset \(A\subset\{1,\ldots,N\}\) of at least two
vertices, and suppose there are at least two vertices outside A.
Write

\[
\bar v_A=|A|^{-1}\sum_{a\in A}v_a,\qquad
U_A=\operatorname{span}\{v_a-\bar v_A:a\in A\}.
\]

For any i,j outside A and any a inside A, the simplex identity gives

\[
\begin{aligned}
\langle v_i-v_j, v_a-\bar v_A\rangle
&=\langle v_i,v_a\rangle-\langle v_j,v_a\rangle\\
&\quad-\frac1{|A|}\sum_{b\in A}
(\langle v_i,v_b\rangle-\langle v_j,v_b\rangle)=0.
\end{aligned}
\]

Each parenthesized difference vanishes because i and j are both
distinct from every a,b in A, and all pairwise distinct-simplex
inner products are the same after translation to the global center.
Thus

\[
\boxed{P_{U_A}v_i=P_{U_A}v_j
\quad\text{for every }i,j\notin A.}\tag{FHC-8}
\]

Every positive-variance PCA component fitted to the centered training
vertices belongs to U_A. Consequently *any PCA code restricted to
positive-variance training components collapses all held-out vertices
to a single point*, however many such components are retained.
If its true future metric distinguishes any two unseen vertices,
its worst-case test distortion is infinite.

**Concrete RC-01 result.** At T=63 the 64 future traces in FHC-6 form
a regular simplex. Fit PCA to 48 calibration vertices and evaluate
the remaining 16 vertices. Its nonzero training span has dimension
47. For every d<=47, PCA projects all 16 held-out vertices to one
point up to floating-point roundoff. The frozen empirical pilot
tests d=2,4,6,8,16,32; all 120 nontrivial held-out pairs collapse
at every tested PCA width. The exact finite proof is independent
of random seed and applies to any disjoint training subset with
at least two unseen vertices.

This is a negative control for *training-subset PCA on maximally
symmetric data*. It does not assert a general weakness of PCA on
ordinary hidden states. A random projection selected independently
of the training subset has a different inductive bias and can
preserve many unseen distances. Source result #094 allows
nonlinear embeddings and does not require them to be trained using
PCA. The comparison motivates finite-set, held-out geometry
checks before interpreting a compact representation as sufficient
for new semantic states.
