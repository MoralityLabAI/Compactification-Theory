# SUD-01: Semantic uncertainty dimension and reverse-trace memory

**Status (2026-11-13):** original conditional toy-model derivation; numerical
finite checks. The results here have not been Lean-checked. The hypotheses
matter, especially independence, balanced outputs, and geometric separation.

## The two external results

**OpenAI Math #119 (2026-09-24), sharp binary information contraction.**
For uniform binary vector X, independent crossover-p bit noise Y=X xor N,
and Boolean f, Courtade–Kumar gives
\[
I(f(X);Y)\le 1-h_2(p).
\]
A single coordinate or its complement attains equality. The Lean scope is
recorded at https://github.com/openai/math/blob/main/lean/docs/119.md .

**OpenAI Math #148 (2026-09-24), entropy-rate dimension.**
For a stationary self-similar probability measure, the manuscript and Lean
scope record
\[
\dim_H \mu = \min(1,h_{\rm RW}/\chi),
\]
including unequal contractions and exact overlaps:
https://github.com/openai/math/blob/main/lean/docs/148.md .

Our conditional posterior has observation-dependent digit weights and is
generally nonstationary. The #148 theorem cannot simply be applied to that
posterior. Its entropy/Lyapunov structure motivates the conditional theorem
below, which has its own proof under strong separation. Extending this result
to exact overlaps or coupled dynamics is subsequent research.

## Setup

Fix an integer \(n\ge1\), \(0\le p\le1/2\), \(0<r<1/2\), and a balanced
Boolean function \(f:\{0,1\}^n\to\{0,1\}\). For each \(t\ge1\), draw \(X_t\)
uniformly and independently, together with independent Bernoulli(p) bit flips
\(N_t\). Let

\[
S_t=f(X_t),\qquad Y_t=X_t\oplus N_t.
\]

Define affine contractions \(F_0(z)=rz\), \(F_1(z)=rz+1-r\) on [0,1] and
the Cantor-coded semantic trace

\[
Z=(1-r)\sum_{t\ge1}r^{t-1}S_t.
\]

Write \(\mu^Y=\mathcal L(Z\mid Y_1,Y_2,\ldots)\) for the quenched posterior
measure and \(\chi=\log_2(1/r)\). Define \(h_2(x)=-x\log_2x-(1-x)\log_2(1-x)\)
at interior probabilities, extended continuously at the endpoints.

A neural model only enters through an experimentally validated map into this
source, observation and contraction model. Neural state dimensions and a
Cantor-coded posterior are different mathematical objects.

## SUD-1: Conditional entropy gives an uncertainty dimension floor

For almost every full observation sequence Y, the posterior measure \(\mu^Y\)
is exact-dimensional, with

\[
\boxed{\dim_H\mu^Y
=\frac{H(S_1\mid Y_1)}{\chi}
\ge \frac{h_2(p)}{\chi}.}\tag{SUD-1}
\]

Equality in the inequality holds when f selects one input coordinate.

**Proof.** Set \(q(y)=\Pr(S_1=1\mid Y_1=y)\).
Conditionally on all observations, digits are independent. If
\(s_1,\dots,s_T\) is the sampled prefix, the posterior mass of its
symbolic cylinder is

\[
\prod_{t=1}^T\Pr(S_t=s_t\mid Y_t).
\]

The iid pair law and strong law of large numbers give

\[
-\frac1T\log_2\mu^Y([S_1\cdots S_T])
\longrightarrow H(S_1\mid Y_1)
\]

joint-almost surely. Every level-T geometric cylinder has diameter at
most \(r^T\). Distinct level-T cylinders are separated by at least
\((1-2r)r^{T-1}\). Hence the cylinder-mass exponent equals the
Euclidean local dimension; disintegration over Y gives the stated
quenched exact-dimensional identity. Since f is balanced,
\(H(S_1)=1\), and #119 yields

\[
H(S_1\mid Y_1)=1-I(S_1;Y_1)\ge h_2(p).
\]

Coordinates attain equality. Because \(\chi>1\), this dimension is below one.
QED.

## SUD-2: Temporal reconstruction and geometric accuracy

Let \(e_f\) be the optimal bit-error probability in reconstructing S from Y.
Binary Fano and SUD-1 yield

\[
h_2(e_f)\ge H(S\mid Y)\ge h_2(p),
\qquad
\boxed{e_f\ge p.}\tag{SUD-2}
\]

The entropy function increases on [0,1/2]. Because independent time steps
have product posteriors, even a decoder seeing the *whole* observation
sequence satisfies

\[
\boxed{\Pr(\widehat S_{1:T}=S_{1:T})\le(1-p)^T.}\tag{SUD-3}
\]

These bounds are attained by a coordinate function. Different level-T
Cantor cylinders are separated by at least \((1-2r)r^{T-1}\).
An estimate with absolute error less than half that gap uniquely
determines the true prefix. For every estimator \(\widehat Z(Y)\),

\[
\boxed{\Pr\bigl(|\widehat Z-Z|<
\tfrac12(1-2r)r^{T-1}\bigr)\le(1-p)^T.}\tag{SUD-4}
\]

This converts a symbolic decoding limit into a geometric
finite-precision reconstruction limit.

## SUD-3: Lower bound for auxiliary semantic memory

Suppose a decoder additionally receives a message M whose alphabet has
size at most \(2^B\). The message may be computed from the true source
trace. Let its block-error probability be at most \(\delta\le1/2\).
Then

\[
\boxed{B\ge T h_2(p)-h_2(\delta)-\delta T.}\tag{SUD-5}
\]

**Proof.** Independence gives
\(H(S_{1:T}\mid Y_{1:T})=T H(S_1\mid Y_1)\ge Th_2(p)\).
Conditioning on M can reduce entropy by at most
\(H(M)\le B\). Fano applied to the \(2^T\)-word source gives

\[
H(S_{1:T}\mid Y_{1:T},M)
\le h_2(\delta)+\delta\log_2(2^T-1)
\le h_2(\delta)+\delta T.
\]

Rearranging proves SUD-5. For independent steps with individual crossover
rates \(p_t\), the same argument yields

\[
B\ge \sum_{t=1}^T h_2(p_t)-h_2(\delta)-\delta T.\tag{SUD-6}
\]

At strictly positive p, zero-error fixed-length coding requires B at
least T because every source word remains possible for every observation;
SUD-5 is a useful entropy lower bound for small nonzero error.
Asymptotically, conditional entropy becomes the natural lossless coding
rate with vanishing error under standard Slepian–Wolf assumptions. QED.

## Parity example and order-two dimension

For \(S=\bigoplus_{i=1}^nX_i\), the entire uncertainty is parity noise
with effective crossover \(p_n=(1-(1-2p)^n)/2\). Therefore

\[
e_f=p_n,\qquad
\dim_H\mu^Y=\frac{h_2(p_n)}{\chi}.
\]

At fixed \(0<p<1/2\), this dimension tends to \(1/\chi\) as n grows.
A single-coordinate readout has the smallest conditional dimension.

A secondary diagnostic is the posterior order-two Rényi dimension. Let
\(q(y)=\Pr(S=1\mid Y=y)\). Under the same separation and iid assumptions,

\[
D_2(\mu^Y)=
-\frac{\mathbb E_Y\log_2(q(Y)^2+(1-q(Y))^2)}{\chi}.
\tag{SUD-7}
\]

For a fixed Y, the level-T sum of squared cylinder probabilities
factors into the corresponding per-step sums. Taking logarithms and
using the strong law proves SUD-7. The #119 Shannon inequality alone
does not assert an analogous Rényi lower bound.

## Practical experimental contract

For Compactification Theory RC-01, charge the side-message B against
its live or auxiliary resource ledger. In a LoopedTransformers Q5
follow-up, freeze the loop clock, model checkpoint, semantic labels,
source distribution, intervention location, and train/test split.
Use dictator, XOR, and random-balanced synthetic function controls.

Apply independently seeded input-bit crossover interventions to a
verified semantic interface. Calibrate the conditional probabilities
q(y) on training trajectories, then evaluate held-out information
and Bayes error with uncertainty intervals. Explicitly test the
effects of correlated temporal noise and unobserved auxiliary channels.

Comparisons to trained recurrent networks require an independently
validated encoding metric. SUD-1 concerns the posterior geometry of a
particular separated two-branch code, not a generic latent manifold.
The theory has not been tested on Qwen, Q5, or Silico checkpoints.

Kernel-verification target: formalize finite entropy identities,
Bayes error and Fano, and the finite-prefix memory bound before
attempting the local-dimension theorem. The source #119/#148
Comparator statements are useful dependencies for a separate audit.
