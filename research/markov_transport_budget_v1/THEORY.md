# MTB-01: Markov transport stability and the geometry of monitoring drift

Date: 2026-11-14. Source pairing: OpenAI Mathematics #327 × #374.

The general source statements are imported research claims, subject to
independent mathematical review. The elementary composition theorems,
closed-form transport examples, and hypercube obstructions below have
separate derivations and executable exact-rational tests. The campaign
makes no claim to establish mathematical priority over prior literature.

## 0. Source claims and distinct scopes

**OpenAI Math #327: Markov type characterizes superreflexivity.**
[Formalization scope](https://github.com/openai/math/blob/main/lean/docs/327.md).
For a Banach space Y, possessing Markov type p for some p>1 is
equivalent to admitting an equivalent uniformly convex norm, hence to
superreflexivity. Markov type is a uniform inequality for every
finite stationary reversible chain and every t>=1:

\[
\mathbb E \|f(Z_t)-f(Z_0)\|_Y^p
\le M_p(Y)^p\,t\,
\mathbb E\|f(Z_1)-f(Z_0)\|_Y^p.
\tag{MT-definition}
\]

The exponent p, constant and equivalent norm can depend on Y. In
particular, this source theorem does not give a universal M_p=1
for arbitrary superreflexive representations.

**OpenAI Math #374: Sharp one-third stability of Brenier maps.**
[Formalization scope](https://github.com/openai/math/blob/main/lean/docs/374.md).
Fix a compact convex body Ω⊂R^d with nonempty interior, d>=2,
source ρ uniform on Ω, and compact target support set K.
For any targets μ,ν supported in K, their unique quadratic-optimal
Brenier maps satisfy

\[
\boxed{\|T_\mu-T_\nu\|_{L^2(\rho)}
\le C_{\Omega,K}\,W_2(\mu,\nu)^{1/3}.}
\tag{source-374}
\]

The exponent 1/3 is optimal for the general target class; the
source produces three-atom targets ruling out a universal exponent
greater than 1/3. No numerical C_{Ω,K} is assumed in this campaign.

These source results concern different aspects of geometry:
the first bounds temporal movement under reversible transitions in a
chosen norm, the second bounds how an optimal transport map changes
when its target law changes.

## MTB-1: A combined time-and-distribution-shift drift theorem

Let (X_t) be a stationary reversible Markov process with one-time
marginal ρ, uniform on Ω. Let A:R^d→Y be an L-Lipschitz
representation/readout, where Y has Markov type p for some 1<p<=2,
with constant M_p. Assume its Markov-type inequality applies to
the particular reversible process (for general compact-state kernels
this can be obtained from finite-kernel approximation under the
usual integrability hypotheses).

Write F_\mu=A∘T_\mu, F_\nu=A∘T_\nu and

\[
E_1=\mathbb E\|F_\mu(X_1)-F_\mu(X_0)\|_Y^p.
\]

Then

\[
\boxed{
\left(\mathbb E\|F_\nu(X_t)-F_\mu(X_0)\|_Y^p\right)^{1/p}
\le
L C_{\Omega,K}W_2(\mu,\nu)^{1/3}
+
M_p(Y)t^{1/p}E_1^{1/p}.
}\tag{MTB-1}
\]

**Proof.** Add and subtract F_\mu(X_t). Minkowski's inequality in
L^p of Y-valued random variables gives the sum of their Lp norms.
The first term is

\[
\|A(T_\nu(X_t))-A(T_\mu(X_t))\|_{L^p}
\le L\|T_\nu-T_\mu\|_{L^p(\rho)}
\le L\|T_\nu-T_\mu\|_{L^2(\rho)}
\le L C_{\Omega,K}W_2^{1/3},
\]

using stationarity, p<=2, and source #374. The second is bounded
by M_p t^{1/p} E_1^{1/p} from Markov type. QED.

For Y=R^d with its Euclidean norm and A=identity, Hilbert
Markov type 2 has M_2=1. A direct proof needs no source #327:
for a reversible Markov operator P with spectral support in
[-1,1], \(1-\lambda^t\le t(1-\lambda)\), giving

\[
\mathbb E\|F(X_t)-F(X_0)\|_2^2
\le t\,\mathbb E\|F(X_1)-F(X_0)\|_2^2.
\tag{Hilbert}
\]

The source #327 supplies the geometric criterion for extending
a nontrivial exponent to more general Banach representations,
under a suitable norm and process-level Markov-type bound.
It does not assert that arbitrary ℓ1 feature maps receive
a dimension-independent bound.

## MTB-2: A decision-flip certificate with independently charged margin

Suppose a binary decision map g:Y→{0,1} is locally constant
throughout the radius-m ball around F_\mu(X_0), except on an
event of probability at most ε, where m>0. Any decision flip
outside this exceptional event then requires norm displacement
of at least m. Markov's inequality and MTB-1 yield

\[
\boxed{
\Pr\!\left[g(F_\nu(X_t))\ne g(F_\mu(X_0))\right]
\le
\min\left\{1,\;
\varepsilon+
\left[
\frac{L C_{\Omega,K}W_2^{1/3}
+M_p t^{1/p}E_1^{1/p}}{m}
\right]^p\right\}.
}\tag{MTB-2}
\]

The margin event and target transport distance must be independently
measured. A high accuracy on repeated labels, without an auditable
margin, is insufficient to invoke this certificate.

For the explicit affine rectangles below we have an *exact*
drift second moment, so the margin inequality can use that measured
number instead of the looser general bound.

## MTB-3: Exact affine Brenier maps under reversible refresh

Let Ω=[0,1]^2 and source ρ be its continuous uniform law. Choose
positive a,b,c,d. The target μ is uniform on [0,a]×[0,b],
and ν is uniform on [0,c]×[0,d].

Their Brenier maps are

\[
T_\mu(u,v)=(au,bv),\qquad T_\nu(u,v)=(cu,dv).
\]

Both are gradients of convex diagonal quadratic potentials. They
push ρ forward to the respective uniform product targets. By
coordinatewise monotone coupling and one-dimensional optimality,

\[
\boxed{W_2^2(\mu,\nu)
=\frac{(a-c)^2+(b-d)^2}{3}
=\|T_\mu-T_\nu\|_{L^2(\rho)}^2.}
\tag{MTB-3a}
\]

The affine family has a stronger *linear* W2 map stability
relation than the universal 1/3 bound. It is not a witness to
the source #374 sharp exponent.

Define a stationary reversible source process as follows:
at every step each coordinate independently keeps its value
with probability 1-s, or resamples a fresh independent U[0,1]
value with probability s. This is a convex combination of
identity and fresh-uniform kernels, each reversible w.r.t. ρ.
After t steps let q_t=(1-s)^t.

For a single coordinate U, conditioned on the initial U_0,
U_t equals U_0 with probability q_t and is independent uniform
otherwise. Thus

\[
\mathbb E U_t^2=\mathbb E U_0^2=1/3,\qquad
\mathbb E(U_0U_t)=\frac{q_t}{3}+\frac{1-q_t}{4}.
\]

Expand the squared transported distance for both coordinates:

\[
\boxed{
\begin{aligned}
D_{\mu\to\nu}(t)^2
&=\mathbb E\|T_\nu(X_t)-T_\mu(X_0)\|_2^2\\
&=\frac{(a-c)^2+(b-d)^2}{3}
  +\frac{ac+bd}{6}\bigl[1-(1-s)^t\bigr].
\end{aligned}}
\tag{MTB-3b}
\]

For an unchanged target,

\[
\boxed{
D_{\mu\to\mu}(t)^2
=\frac{a^2+b^2}{6}[1-(1-s)^t]
\le t \frac{a^2+b^2}{6}s.
}
\tag{MTB-3c}
\]

The last inequality follows from Bernoulli's inequality
\(1-(1-s)^t\le ts\). It gives a direct exact model
for the Markov-type-2 temporal bound.

**Numerical certificate with exact rationals.**
Choose a=b=d=1, c=2, s=1/4, t=2. Then
q_t=9/16, W2²=1/3, cross drift²=53/96,
same-target drift²=7/48, and the one-step energy is 1/12.
The temporal Hilbert bound is 7/48≤2/12.

## MTB-4: Exact decision flips in the affine family

For a first-coordinate threshold θ, define the initial decision
\(g_\mu=\mathbf 1\{aU_0\ge\theta\}\) and the final decision
\(g_\nu=\mathbf 1\{cU_t\ge\theta\}\). Put

\[
p_0=\bigl(1-\theta/a\bigr)_{[0,1]},\qquad
p_1=\bigl(1-\theta/c\bigr)_{[0,1]},
\]

where \((\cdot)_{[0,1]}\) clamps to the unit interval.
The two decisions disagree with probability \(|p_1-p_0|\)
when the source coordinate persists and with probability
\(p_0+p_1-2p_0p_1\) after an independent refresh.
Consequently

\[
\boxed{
P_{\mathrm{flip}}(t)
=q_t|p_1-p_0|
+(1-q_t)(p_0+p_1-2p_0p_1).
}\tag{MTB-4}
\]

For θ=1/2, a=1,c=2,s=1/4,t=2,
the flip probability is exactly 23/64.

An initial threshold margin m>0 fails precisely when
\(|aU_0-\theta|<m\). Its probability is

\[
\varepsilon_m=
\frac{\bigl[\min(a,\theta+m)-\max(0,\theta-m)\bigr]_+}{a}.
\]

A decision flip outside that event requires displacement at
least m. An exact elementary certificate therefore reads

\[
\boxed{
P_{\mathrm{flip}}(t)\le
\min\{1,\varepsilon_m+D_{\mu\to\nu}(t)^2/m^2\}.
}\tag{MTB-4b}
\]

The finite reference computes both sides using only Fraction.
This risk certificate is a property of the declared threshold
and continuous source distribution, not a neural model's
empirically measured safety margin.

## MTB-5: Exact ℓ1 hypercube obstruction to dimension-free Markov type

For n>=4 let Z_t be the stationary uniform simple random walk
on {0,1}^n: each step flips exactly one uniformly chosen coordinate.
The chain is reversible. Give its states the ℓ1 metric under the
identity embedding, i.e. Hamming distance. Define
\(H_t=\|Z_t-Z_0\|_1\).

The parity of coordinate i's number of flips has expectation

\[
a_t=\mathbb E(-1)^{N_i(t)}=(1-2/n)^t.
\]

For distinct i,j, the joint parity expectation is
\(b_t=(1-4/n)^t\). Elementary expansions give

\[
\boxed{\mathbb E H_t=
\frac n2(1-a_t)}
\tag{MTB-5a}
\]

and

\[
\boxed{
\mathbb E H_t^2=
\frac n2(1-a_t)+
\frac{n(n-1)}4(1-2a_t+b_t).
}
\tag{MTB-5b}
\]

The original one-step distance equals one deterministically,
so a Markov-type-2 constant for ℓ1^n must satisfy

\[
\boxed{M_2(\ell_1^n)^2
\ge\frac{\mathbb E H_t^2}{t}.}
\tag{MTB-5c}
\]

For t=n and arbitrary p>1, Jensen's inequality gives

\[
M_p(\ell_1^n)
\ge \frac{\mathbb E H_n}{n^{1/p}}
=\frac{1-(1-2/n)^n}{2}n^{1-1/p}
\ge \boxed{\frac{1-e^{-2}}2 n^{1-1/p}.}
\tag{MTB-5d}
\]

Thus no constant uniform in n can give Markov type p>1 for
this ℓ1^n family. This is a concrete finite obstruction
consistent with source #327's statement about the infinite
ℓ1 space lacking superreflexivity. Each fixed finite
dimensional ℓ1^n *is* superreflexive; it is the uniform
constant across dimensions that fails.

An alternative Hilbert representation sends each cube vertex
to its {0,1}^n coordinate vector in Euclidean ℓ2.
Its **squared** Euclidean displacement is Hamming distance
H_t, and E H_t≤t. Hilbert Markov type 2 with constant one
therefore holds uniformly. The two representations encode
the same exact discrete state information and differ in
norm geometry and the resulting drift certificate.

The closed moments are independently checked by:
1. a birth-death chain on Hamming counts 0,...,n;
2. direct transition probability propagation on all 2^n
   bitmasks for n<=8.

At n=4,t=4, EH_t=15/8 and E H_t²=9/2, giving
\(M_2(\ell_1^4)^2\ge9/8\).

## MTB-6: Resource implications for compact monitoring

The combined theorem has three separately controlled inputs:
the target-law shift W2, the one-step recurrent energy E1, and
the norm geometry (p,M_p). The exponent 1/3 derives from the
target-map stability theorem. The time exponent 1/p derives
from the chosen representation's Markov type.

For a semantic-state monitor, a proposed sparse/ℓ1 readout
cannot be assigned a dimension-independent nontrivial
Markov-type budget by analogy with Euclidean latent features.
The hypercube controls show this failure can occur even for
a stationary reversible process with a complete, exact
known transition law.

A Hilbert readout permits type-2 temporal control in the
declared reversible regime, but the target shift remains
subject to source #374's worst-case one-third stability.
The affine example has stronger linear stability and thus
must not be advertised as exhibiting the source's
sharp three-atom counterexample.

**Prospective experiment.** On frozen LoopedTransformers/Qwen
activations, first test whether the projected process is
stationary reversible, then fix the representation norm,
a target distribution map/approximation, one-step energy,
and a held-out margin event. A neural activation stream is
generally not an independent refresh chain. A learned
transport map need not be Brenier optimal, and a general
language-model rollout need not satisfy a reversible
Markov law. Those gates are prerequisites to any
model-level extrapolation.

No original model checkpoint, native SAE, medical data,
or production financial system was used in this exact
synthetic campaign. The source #327 superreflexivity
equivalence and #374 sharp one-third construction have
not been reproved in Lean or in the finite Python code.


## MTB-7: Stationarity without reversibility permits ballistic drift

The Markov-type inequality in source #327 concerns **stationary
reversible** processes. The condition matters even for Hilbert
target space \(\mathbb R^2\).

Fix integer m>=2. Enumerate the 4m distinct integer-lattice
vertices of a square perimeter with side length m, each edge
of Euclidean length one. Let X_0 be uniform over these vertices,
and deterministically advance one vertex clockwise at each
step.

The process is stationary: the update is a permutation of the
uniform distribution. It is nonreversible for 4m>=8:
the stationary flow from vertex 0 to vertex 1 has mass
1/(4m), whereas the reverse edge has mass zero.

Every step has squared Euclidean movement one:

\[
\boxed{\mathbb E\|X_1-X_0\|_2^2=1.}\tag{MTB-7a}
\]

After m steps each point is displaced a quarter of the
perimeter. If the initial point lies on the first side at
position x∈{0,...,m−1}, its squared displacement is

\[
(m-x)^2+x^2.
\]

The remaining three sides have the same distribution by
rotational symmetry. Averaging over the m positions gives

\[
\boxed{
\mathbb E\|X_m-X_0\|_2^2
=\frac1m\sum_{x=0}^{m-1}
[(m-x)^2+x^2]
=\frac{2m^2+1}{3}.
}\tag{MTB-7b}
\]

If the source Hilbert Markov-type-2 inequality with constant
one were applied to this *nonreversible* stationary chain,
it would assert a bound of m on the preceding quadratic drift.
The actual ratio is

\[
\boxed{
\frac{\mathbb E\|X_m-X_0\|_2^2}
{m\mathbb E\|X_1-X_0\|_2^2}
=\frac{2m^2+1}{3m}\to\infty.
}\tag{MTB-7c}
\]

This is a source-independent exact finite counterexample
to the **misapplication** of a reversible Markov-type
estimate to a directed stationary evolution. It does
not contradict Markov type two of Hilbert space.

The independent [nonreversible.py](nonreversible.py) oracle
enumerates every point of each directed perimeter for
m=2,4,8,16,32,64,128, verifies period and unit step
length, computes exact rational quarter-drift and
reports a concrete detailed-balance violation.
This control was added after the original MTB-01
protocol and initial finite experiments; it is
versioned separately as MTB-01B.

For recurrent AI monitoring, the combination of
MTB-5 and MTB-7 establishes two independent
ways a drift guarantee can fail: the chosen
representation norm may have unbounded type
constants across width, and the state-transition
process may be directional even if its marginal
distribution is stationary. Both are explicit
assumption gates for source-conditioned claims.
