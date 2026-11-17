# MTB-01: Exact drift and geometry controls

Date: November 14, 2026.
Evidence: **EXACT_AFFINE_TRANSPORT_MARKOV_CONTROL**
and **EXACT_STATIONARY_NONREVERSIBLE_HILBERT_CONTROL**.

[Source pairing and mathematical deductions](THEORY.md) |
[Registered original protocol](PROTOCOL.md) |
[Exact reversible reference](reference.py) |
[Directed nonreversible control](nonreversible.py) |
[Independent test suites](test_reference.py).

Verified GitHub Actions run:
https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43592117700

The CI checked **23 tests** and emitted two source-hashed JSON
artifacts, one for the affine/Markov geometry and one for
the separately versioned directed-square control.

## OpenAI source pair

#327 claims a Banach-space characterization of superreflexivity
by nontrivial Markov type. #374 claims sharp uniform 1/3 Hölder
stability of Brenier maps in L2 when their target laws move in
Wasserstein-2 distance, for a fixed compact convex uniform
source and targets in one fixed compact set.

MTB-1 composes those statements: for a reversible stationary
source process with uniform law, an L-Lipschitz feature map into
a Markov-type-p space (1<p<=2), and transported target laws μ,ν,

\[
\|A(T_\nu(X_t))-A(T_\mu(X_0))\|_{L^p}
\le LCW_2(\mu,\nu)^{1/3}
+M_p t^{1/p} E_1^{1/p}.
\]

The first term is target shift; the second is recurrent
movement under its representation norm. The statement
assumes the type inequality applies to the declared
reversible process, including needed integrability.

This general theorem remains conditional on source
#374's target stability claim and source #327's
geometric Markov-type interpretation. The finite
results below have independent exact-rational proofs.

## Affine Brenier/refresh chain

The source is uniform on [0,1]^2. For positive rational
a,b,c,d, the two Brenier maps are diagonal scalings
(u,v)↦(au,bv) and (u,v)↦(cu,dv) to uniform rectangles.

At every step each source coordinate independently
refreshes to a new uniform variable with probability s.
This stationary process is reversible. With q=(1-s)^t,

\[
\begin{aligned}
W_2^2 &=\frac{(c-a)^2+(d-b)^2}{3},\\
D_{\mathrm{cross}}(t)^2
&=W_2^2+\frac{ac+bd}{6}(1-q),\\
D_{\mathrm{same}}(t)^2
&=\frac{a^2+b^2}{6}(1-q)
\le t\frac{a^2+b^2}{6}s.
\end{aligned}
\]

Every identity is checked as an exact Fraction,
with the cross moment independently integrated
using E[U²]=1/3 and E[UV]=1/4. Parameter cells
are frozen in the original protocol.

| Source/target example | t | Target W2² | Exact cross drift² | Decision flip at threshold 1/2 |
|---|---:|---:|---:|---:|
| (1,1)→(101/100,1), s=1/100 | 0 | 1/30000 | 1/30000 | 1/202 |
| Same | 1 | 1/30000 | 203/60000 | 1/101 |
| (1,1)→(2,1), s=1/4 | 0 | 1/3 | 1/3 | 1/4 |
| Same | 2 | 1/3 | 53/96 | 23/64 |
| (3/2,2)→(2,5/2), s=1/4 | 0 | 1/6 | 1/6 | 1/12 |

The affine family has exact L2 transport-map
distance equal to W2. This is **stronger
regularity within this restricted family**.
It does not recreate #374's source three-atom
counterexample establishing sharp 1/3 stability
in the general supported-target class.

### Decision margin

For an initial threshold θ and margin m>0,
a decision flip requires movement at least m
except when the source point lies within m
of the original decision boundary. Therefore

\[
P_{\rm flip}\le
\min(1,\varepsilon_m+D_{\rm cross}^2/m^2).
\]

The code tests the exact flip probability
against this conservative finite bound.
For the small target shift 1→101/100,
refresh s=1/100 and t=1, the actual flip
probability is exactly 1/101, with the
source's local margin probability separately
included in the certified upper bound.

## ℓ1 versus Euclidean hypercube geometry

Start a uniformly distributed reversible
walk on the n-bit cube. Every step flips one
independent uniformly chosen coordinate index.

For Hamming displacement H_t, set
a_t=(1-2/n)^t and b_t=(1-4/n)^t.
The code verifies independently

\[
E H_t=\frac n2(1-a_t),\quad
E H_t^2=\frac n2(1-a_t)
+\frac{n(n-1)}4(1-2a_t+b_t).
\]

Three oracles agree: the parity-moment
formula, a birth-death Markov chain on
Hamming weight, and direct propagation
over all bitmask states for small n.

At t=n, the Markov-type-2 constant
for ℓ1^n must be at least:

| n | Certified lower bound on M₂ |
|---:|---:|
| 4 | 1.060660 |
| 8 | 1.362492 |
| 16 | 1.828400 |
| 32 | 2.515832 |
| 64 | 3.508347 |
| 128 | 4.926438 |
| 256 | 6.942184 |

More generally Jensen gives, for every p>1,

\[
M_p(\ell_1^n)\ge
\frac{1-e^{-2}}2 n^{1-1/p}.
\]

This obstructs a **dimension-independent**
nontrivial Markov-type bound for the ℓ1^n
family. Each fixed finite-dimensional ℓ1^n
is superreflexive, with potentially
dimension-dependent constants.

In a Euclidean embedding of the exact
same binary states, squared Euclidean
distance equals Hamming. Thus E H_t<=t,
consistent with Markov type 2 constant one.

The Lean companion checks the n=4,t=4
integer path-count distribution
(40,0,192,0,24) out of 256 paths,
yielding EH_4=15/8 and EH_4²=9/2.
It separately validates the affine
two-step drift and threshold flip
denominators.

## MTB-01B: stationarity without reversibility

After the original protocol and first
source-code pass, a separate negative
control was added to distinguish
stationarity from reversibility.

Take 4m lattice vertices on the boundary
of a square of side m and advance one
vertex clockwise each step, beginning
with a uniform vertex. The chain stays
stationary, but detailed balance fails.

Its squared Euclidean movement is one
per step. At quarter-period t=m,

\[
\boxed{E\|X_m-X_0\|_2^2=(2m^2+1)/3.}
\]

The ratio to an **inapplicable**
reversible Hilbert-type temporal bound m
grows without bound:

| Square side m | Mean squared drift at t=m | Ratio to reversible bound |
|---:|---:|---:|
| 2 | 3 | 1.5 |
| 4 | 11 | 2.75 |
| 8 | 43 | 5.375 |
| 16 | 171 | 10.6875 |
| 32 | 683 | 21.34375 |
| 64 | 2731 | 42.671875 |
| 128 | 10923 | 85.3359375 |

The exact source enumerates all perimeter
states, each transition and detailed-balance
flux. This example does not contradict
Hilbert Markov type 2: the required
reversibility assumption fails.

The MathCan extension checks the m=4
total one-step displacement 16 across
16 states and four-step displacement
176. Since 176>4*16, the failure of
the *misapplied* bound is kernel-checked.

## Scientific limits

The model experiments are exact and
synthetic. Markov-type drift estimates
require a reversible chain, with a
suitable equivalent norm and explicit
type constant for general Banach targets.
The original #374 stability result
requires true Brenier-optimal maps from
a uniform convex source and fixed
compact target support.

A trained recurrent transformer
typically has neither the declared
refresh kernel nor an automatically
valid transport map. Checkpoint and
activation evaluation will require
a separate registered experiment.

No trained Qwen/LoopedTransformers
checkpoint, SAE, or production
control system has been tested here.
The general source #327 equivalence
and source #374 sharp counterexample
remain outside this finite exercise.
