# KSR-01: Branching reconstruction, locality and compact monitoring

Date: November 14, 2026. Results KSR-1 through KSR-4 are elementary
finite/probability deductions with exact-rational verification. KSR-5
is a **conditional cross-paper comparison** using two OpenAI Mathematics
Release claims. Neither original manuscript is treated as an
independently community-validated proof in this campaign.

## Source claims and distinct mathematical objects

**OpenAI Math #229**: *The exact reconstruction threshold for the
three-state symmetric channel* (September 25, 2026).
Source scope:
https://github.com/openai/math/blob/main/lean/docs/229.md .
The manuscript reports regular b-ary three-state root reconstruction
precisely in the phase b lambda^2 > 1, including a non-reconstruction
claim at equality. The published Lean **scope note** identifies its
formalized portion as supercritical reconstruction for regular and
Poisson trees; the non-reconstruction half lies outside that selected
scope.

**OpenAI Math #236**: *The sharp factor-of-IID threshold for the free
Ising model on regular trees* (September 26, 2026).
Source scope:
https://github.com/openai/math/blob/main/lean/docs/236.md .
For the free zero-field ferromagnetic Ising measure on an infinite
d-regular tree, the paper reports an automorphism-equivariant factor
of independent vertex labels exactly when
tanh(beta) <= 1/sqrt(d-1), **including equality**. The Lean scope
includes that threshold. It excludes finitary coding and coding
radius. A factor of IID *generates* the full unrooted infinite-tree
random field. A reconstruction estimator *infers* a distinguished
root spin from noisy descendants. They are different tasks.

The common parameter in their regular-tree settings is
t=b*lambda^2, with forward branching b=d-1 for Ising.
A b-ary rooted broadcast tree is a forward branch of a
d-regular tree selected by an oriented edge, rather than the
full d-regular ball around a distinguished root. Only the
threshold matches automatically; root-degree constants differ.

## 1. Symmetric q-state broadcast and exact eigenfunctions

Let q>=2 and b>=2. Root X_empty is uniform over the q labels.
Every edge independently transmits the parent's state using

\[
P_\lambda(a,s)=\frac{1-\lambda}{q}
+\lambda\,\mathbf1_{\{a=s\}},
\qquad
-\frac1{q-1}\le\lambda\le1 .
\tag{KSR-setup}
\]

Fix a label j and put g_j(s)=q 1{s=j}-1. Then

\[
E g_j(X)=0,\quad E g_j(X)^2=q-1,\quad
E[g_j(X_{\mathrm{child}})\mid X_{\mathrm{parent}}]
=\lambda g_j(X_{\mathrm{parent}}).
\]

These identities are exact by summing the q transition
probabilities. Positive lambda is the ferromagnetic regime
used for the cross-paper comparison.

## 2. KSR-1: Normalized leaf-count martingale and exact energy

At depth L the forward tree has N=b^L leaves. For lambda != 0 define

\[
M_{j,L} =
\frac{\sum_{|v|=L}g_j(X_v)}{(b\lambda)^L}
=\frac{qN_{j,L}-b^L}{(b\lambda)^L},
\tag{KSR-1a}
\]

where N_{j,L} is the number of leaves bearing label j.
The conditional eigenfunction equation yields

\[
E[M_{j,L}\mid X_\empty]=g_j(X_\empty),\qquad
E[g_j(X_\empty)M_{j,L}]=q-1.
\tag{KSR-1b}
\]

Write V_L=E[M_{j,L}^2]/(q-1); V_0=1.
Conditional on the root, the b descendant subtrees are
independent and identically distributed. Their normalized
subtree scores have conditional mean lambda*g_j(root).
Expanding the square of their sum yields

\[
\boxed{
V_{L+1}=\frac{V_L}{b\lambda^2}+\frac{b-1}{b}.
}\tag{KSR-1c}
\]

Writing t=b lambda^2 gives the exact closed form

\[
\boxed{
V_L=t^{-L}+\frac{b-1}{b}\sum_{k=0}^{L-1}t^{-k}.
}\tag{KSR-1d}
\]

This formula is q-independent after dividing by the source
eigenfunction variance. It follows from the finite second-moment
calculation, independently of source #229 and #236.

For t>1 it tends to
\[
\boxed{
V_\infty=\frac{(b-1)t}{b(t-1)}.
}\tag{KSR-1e}
\]
At t=1 it is exactly
\[
\boxed{V_L=1+\frac{b-1}{b}L.}\tag{KSR-1f}
\]
For 0<t<1 it grows exponentially. These are statements
about THIS normalized linear leaf-count statistic; a diverging
variance alone does not rule out all nonlinear reconstruction.

## 3. KSR-2: Root decoding from a compact histogram

Let C_L=(N_{0,L},...,N_{q-2,L}) be the first q-1
leaf-label counts. The last count follows from their sum and
N=b^L. Let pi_j=P(X_empty=j|C_L), eta_j=q pi_j-1.
The score M_{j,L} is C_L-measurable. By conditional expectation
and Cauchy-Schwarz,

\[
q-1=E[\eta_j M_{j,L}]
\le\sqrt{E\eta_j^2\,(q-1)V_L},
\quad
E\eta_j^2\ge\frac{q-1}{V_L}.
\]

Summing over j, and using sum_j pi_j=1,

\[
\sum_j\eta_j^2=q^2\sum_j\pi_j^2-q.
\]

For every posterior vector pi,
\(\max_j\pi_j\ge\sum_j\pi_j^2\).
The Bayes-optimal root decoder using C_L therefore obeys

\[
\boxed{
P(\hat X_\empty=X_\empty \mid
  \text{optimal decoding of leaf counts})
\ge
\frac1q+\frac{q-1}{qV_L}.
}\tag{KSR-2}
\]

The bound is nonasymptotic and holds for every q>=2,
b>=2, finite L, and nonzero valid lambda. The full leaf
pattern has at least as much information as the counts.
The bound is thus also a certified full-observation
reconstruction lower bound.

For fixed q,b and t>1, letting L grow gives

\[
\boxed{
\liminf_L P_{\mathrm{MAP,count}}
\ge\frac1q+\frac{q-1}{q}
\frac{b(t-1)}{(b-1)t} >\frac1q .
}\tag{KSR-2∞}
\]

For b=2 and lambda=4/5, t=32/25 and V_infty=16/7:
q=2 gives accuracy >=23/32=71.875%;
q=3 gives accuracy >=5/8=62.5%.

**Memory accounting.** A one-pass stream monitor can maintain
the first q-1 label counts as exact integers in
[0,b^L]. Its mutable storage is at most

\[
\boxed{
B_L=(q-1)\lceil\log_2(b^L+1)\rceil
=O_q(L\log b)\ \text{bits}.
}\tag{KSR-2B}
\]

It observes exponentially many leaf symbols but retains
linearly many bits in the depth. This is an upper bound
on mutable *counter memory*, not a bound on the static
description or computation of the Bayes-optimal decoder.
The readout may require a table or a costly likelihood
calculation. Charge it in the RC-01 ledger.

## 4. KSR-3: Sparse leaf observation and the retention law

Retain each leaf independently with probability r in (0,1],
and reveal which leaves are missing, independently of spins.
Let I_v be Bernoulli(r) and define

\[
M_{j,L}^{(r)}=
\frac{\sum_{|v|=L}I_v g_j(X_v)}
{r(b\lambda)^L}.
\]

The root covariance in KSR-1b remains q-1.
For distinct leaves, E I_v I_w=r^2, while
E I_v^2=r. Consequently the second moment has
the exact **diagonal dropout correction**

\[
\boxed{
V_L(r)=
\frac{E[M_{j,L}^{(r)2}]}{q-1}
=V_L+\frac{1-r}{r(b\lambda^2)^L}.
}\tag{KSR-3a}
\]

The Cauchy/conditional-posterior argument of KSR-2
applies to the q retained label counts (now the total
number retained is random), giving

\[
\boxed{
P_{\mathrm{MAP,retained-count}}
\ge\frac1q+\frac{q-1}{q V_L(r)}.
}\tag{KSR-3b}
\]

Exact streaming storage now uses q counts, each
in [0,b^L], so B_L(r)<=q ceil(log2(b^L+1))
bits; the decoder never needs the identities of
missing leaf positions for this guarantee.

For fixed 0<r<=1 and t>1, the additional variance
decays like t^{-L}. More generally, if r=r_L depends
on the depth, the sufficient condition for a
uniformly positive lower bound is

\[
\boxed{\inf_L r_L t^L>0,\quad t>1.}\tag{KSR-3c}
\]

The observation's expected leaf budget is
r_L b^L. For r_L=t^{-L}, this becomes lambda^{-2L}.
The conclusion concerns a lower bound obtained by a
particular count statistic. It supplies neither a
necessity theorem nor a bound for arbitrary learned
monitors with limited observation access.

## 5. KSR-4: Two independent finite exact oracles

**Pairwise LCA covariance.** For leaves u,v at depth L,
whose youngest common ancestor has depth k,

\[
E[g_j(X_u)g_j(X_v)]
=(q-1)\lambda^{2(L-k)}.
\]

The source reference independently sums this over all
leaf pairs and checks that dividing by
(b lambda)^(2L) yields exactly V_L from KSR-1.

**Full posterior enumeration.** For small q,b,L and
optional missing leaves, the exact evaluator enumerates
all possible observed leaf strings. A bottom-up
sum-product message computes

\[
m_v(a)=\prod_{c\in\mathrm{children}(v)}
\sum_s P_\lambda(a,s)m_c(s),
\]

with indicator-vector leaf messages for observed
labels and all-ones messages for missing labels.
Root has a uniform q-state prior. The evaluator
independently computes full-leaf MAP success and
histogram-only MAP success by summing the exact
joint probabilities over every count-equivalence cell.

It checks, with exact rational arithmetic,

\[
1/q\le\text{KSR-2/3 floor}
\le \mathrm{MAP(counts)}
\le\mathrm{MAP(full\ pattern)}\le1
\]

and verifies both the root-score covariance and
second-moment dropout formula from those same
enumerated joint distributions. This tests the
analytic proof against a separate statistical route.

## 6. KSR-5: Algebraic coincidence with the FIID boundary

An infinite d-regular tree has d-1 forward children
after choosing an oriented edge. Put b=d-1.
For the free zero-field ferromagnetic Ising law,
lambda=tanh beta >=0. The conditional source
#236 theorem states

\[
\mu_{d,\beta}\text{ is an equivariant factor of IID}
\iff t=(d-1)\tanh^2\beta\le1.
\tag{source-236}
\]

By the independent KSR-2 derivation, if t>1,
even a forward b-ary branch permits a finite-depth
leaf-count decoder with a root advantage bounded
uniformly above zero as depth grows. By #236,
the *unrooted infinite-tree free Ising law*
cannot then be generated as an automorphism-
equivariant factor of IID vertex labels.

At critical t=1, #236 says the infinite free
Ising law **does** admit such a factor.
The count-estimator variance V_L grows linearly
and our certified count-recovery advantage
lower bound tends to zero. The lower bound's
degeneration is not itself a proof of optimal
non-reconstruction.

For the q=3 symmetric broadcast model, source
#229 reports positive root reconstruction
precisely for t>1, including non-reconstruction
at equality. Our KSR-2 calculation directly
establishes the positive regular-tree direction
by an elementary linear-score argument; the
exact negative direction remains an imported
source claim rather than a reproduced proof.

**This is a threshold correspondence across
different tasks and spin alphabets.** Factor
of IID does not mean short finite-radius coding,
and the rooted leaf-count decoder is not an
equivariant generative rule for an unrooted tree.
The comparison yields an experimental control
for branching latent-state monitorability,
not a universal neural model phase transition.

## 7. Applications to current compact-AI research

**LoopedTransformers semantic observability.**
Use a fixed branching proof-search or storyworld
state-transition generator with q semantic states.
Estimate b and the leading channel eigenvalue
lambda from a frozen transition corpus, then
plot the count-monitor accuracy, second-moment
ratio and trained-probe accuracy against t.
A decoder trained on private root labels must
be evaluated on independent tree families.

**Compactification Theory / RC-01.**
Charge each observed leaf symbol, the
O(L)-bit counter state, the decoder/codebook
description and its arithmetic. The mathematical
count monitor uses small live memory with
potentially exponential observation cost,
which makes the tradeoff explicit.

**Sparse autoencoders and monitor dropout.**
The exact KSR-3 law suggests an intervention:
remove independently sampled terminal
semantic observations at varying retention r.
If the source is a genuine symmetric broadcast
process, the extra normalized second moment
must follow (1-r)/(r t^L).
Real neural trajectories may violate the
Markov, tree, symmetry and independent-mask
assumptions; such deviations are the signal
to diagnose rather than ignore.

**Proof/evaluation integrity.**
The experiment measures synthetic root
reconstruction. It is not evidence of a
transformer's hidden chain of thought
or a model's self-ascribed beliefs, and
it does not instantiate a finite algorithm
for the #236 infinite factor-of-IID proof.
