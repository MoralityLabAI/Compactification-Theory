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
