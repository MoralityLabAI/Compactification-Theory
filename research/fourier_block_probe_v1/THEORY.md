# FPB-01: Hidden coordinated modes behind first-order Fourier salience

Date: 2026-11-13. Research status: new conditional construction, elementary
Fourier/Boolean proofs and a finite, executable interpretability benchmark.
Mathematical claims relying on the source manuscripts are explicitly conditional.
There is no trained language-model intervention experiment in this phase.

## Source pair and precise scope

**OpenAI Mathematics #132** (*A superquadratic separation of sensitivity
and block sensitivity*, September 25, 2026). The released
[Lean scope](https://github.com/openai/math/blob/main/lean/docs/132.md)
reports total Boolean functions h with unbounded bs(h)/s(h)^2.
The paper's explicit family has
bs(h,0)/s(h)^2 >= 2^d / [4(d+2)^2], for arbitrarily large d,
and produces functions with bs(h,0) growing faster than s(h)^alpha
for some alpha>2.

**OpenAI Mathematics #192** (*Unbounded violations of the square-root
degree bound*, September 26, 2026). The released
[Lean scope](https://github.com/openai/math/blob/main/lean/docs/192.md)
reports Boolean sign functions f with
sum_i hat f({i}) / sqrt(deg f) arbitrarily large.
Degree here is the REAL multilinear Fourier degree. The comparator
challenge file contains admitted theorem statements; do not present a
comparator as an independent proof of the original release result.

The new campaign studies a mechanistic monitor that uses singleton
Fourier correlations of binary internal variables to rank features.
It must distinguish global Fourier salience from the behavior of a
conditional computation mode under coordinated interventions.

## Definitions and convention

Work over the uniform sign cube {+1,-1}^n, with character
chi_S(x)=product_{i in S} x_i and
hat f(S)=E[f(X) chi_S(X)]. Boolean values of f are signs.
The Fourier degree is max{|S|:hat f(S)!=0}.
Sensitivity s(h,x) of a bit-output function h counts individual
coordinates whose flip changes the output, and s(h)=max_x s(h,x).
Block sensitivity bs(h,x) is the largest number of pairwise
disjoint nonempty coordinate sets, each of whose simultaneous
flip changes h(x). A block is allowed to contain multiple bits.

## FPB-1: Parity lift erases every singleton coefficient

For arbitrary h:{0,1}^n->{0,1}, define its sign version
H(z)=(-1)^{h(z)} and the parity lift

\[
G(x,y)=H(x\oplus y),\quad x,y\in\{0,1\}^n.
\]

Here xor operates coordinatewise. For every mask A,B subset [n],

\[
\boxed{
\widehat G(A_x\cup B_y)
=\begin{cases}
\widehat H(A),&A=B,\\
0,&A\ne B.
\end{cases}}\tag{FPB-1a}
\]

**Proof.** Let Z=X xor Y for independent uniform X,Y. The pair
(Z,X) is independent uniform. The Fourier character of selected
X coordinates and Y coordinates becomes
chi_A(X)chi_B(Y)=chi_{A triangle B}(X)chi_B(Z).
Averaging over X gives zero unless A=B; in that case the
remaining expectation is hat H(B). QED.

In particular all 2n singleton Fourier coefficients of G vanish,
and its entire Fourier spectrum lies on paired x/y supports.
For nonconstant H, the maximum Fourier degree doubles:

\[
\boxed{\deg G=2\deg H.}\tag{FPB-1b}
\]

Every nonzero Fourier character indexed by A in H becomes a
character supported on the 2|A| paired input coordinates.

## FPB-2: Sensitivity and block sensitivity survive the lift

Flipping x_i or y_i toggles precisely coordinate i of Z=x xor y.
Therefore for every x,y,

\[
s(G,x,y)=2s(h,x\oplus y),
\qquad
\boxed{s(G)=2s(h).}\tag{FPB-2a}
\]

If A_1,...,A_b are disjoint blocks sensitive for h at z=0,
the corresponding disjoint blocks of x coordinates are sensitive
for G at (x,y)=(0,0). Thus

\[
\boxed{bs(G,(0,0))\ge bs(h,0).}\tag{FPB-2b}
\]

Combining with the claimed #132 source family yields an
unbounded sensitivity gap for the lifted function, while its
entire singleton Fourier spectrum remains exactly zero:

\[
\frac{bs(G,(0,0))}{s(G)^2}
\ge \frac14\frac{bs(h,0)}{s(h)^2}
\longrightarrow\infty.\tag{FPB-2c}
\]

If the source family obeys bs(h,0)>=s(h)^alpha for alpha>2,
then bs(G,(0,0))>=2^{-alpha}s(G)^alpha.

These are conditional consequences of #132, with elementary
and independently checkable transport identities.

## FPB-3: One function with global Fourier salience and hidden
## coordinated block sensitivity

The second source result enters essentially in the following
quantified combination theorem.

**Conditional theorem.** Assuming the source statements #132
and #192, for any C,K>0, there exists a finite Boolean sign
function F(q,x,y), with a single selector q and disjoint
surface/hidden input registers, such that:

1. Its global singleton Fourier sum exceeds
   C sqrt(deg(F)).
2. On the hidden selector slice q=-1, F agrees with a
   function G whose singleton Fourier coefficients are all zero.
3. This hidden slice obeys bs(G,0)/s(G)^2>K.

**Construction and proof.** Choose h from #132 so that
bs(h,0)/s(h)^2>4K and form the parity lift G of FPB-1.
Set D=deg(G)>=1. Choose a sign f from #192 with degree d>=1 and

\[
S_f:=\sum_i\widehat f(\{i\})>
R\sqrt d,\qquad
R=2(C+2)\sqrt{D+1}.
\]

Set F(q,x,y)=f(x) when q=+1, and F(q,x,y)=G(y)
when q=-1, with x and y disjoint and q uniform.
Then the full singleton Fourier sum is

\[
\boxed{
S_F=\frac12S_f+\frac12(E f-E G).
}\tag{FPB-3a}
\]

The contribution from the hidden x/y variables is zero
by FPB-1. The final selector coefficient has absolute
value <=1. Furthermore,

\[
\boxed{\deg(F)=1+\max(d,D).}\tag{FPB-3b}
\]

The two highest-degree terms live on different independent
registers, so cancellation is impossible. Since d>=1,

\[
1+\max(d,D)\le d(D+1)
\]

and hence, after normalizing,

\[
\frac{S_F}{\sqrt{\deg(F)}}
>\frac{R}{2\sqrt{D+1}}-1=C+1>C.
\]

The hidden slice has the required zero singleton spectrum and
block-sensitivity gap by FPB-1/2. QED.

**Interpretation.** A global singleton Fourier summary can be
arbitrarily salient even when an embedded computation mode has
no singleton Fourier signal and contains many disjoint
coordinated perturbations that change its output. The theorem
is about a MODE-CONDITIONAL slice. It makes no assertion that
the full selector function F itself has a superquadratic
global block-sensitivity ratio.

## FPB-4: Exact regression-probe distinction

Suppose the hidden sign target is G(X,Y), with uniform independent
inputs. The best population mean-squared affine prediction from
the 2n one-coordinate sign features is the constant E[G],
because every singleton feature has zero covariance with G.
Its minimum population MSE equals Var(G).

Add the paired features
Z_i=X_iY_i=(-1)^{x_i xor y_i}. They are orthonormal
under the uniform distribution. The optimal affine
regression on these features has MSE

\[
\boxed{
\mathrm{MSE}_{\mathrm{paired}}=
\mathrm{Var}(G)-\sum_i\widehat H(\{i\})^2.
}\tag{FPB-4}
\]

This follows from orthogonal L2 projection onto the constant
and the paired characters. Features built from paired masks of
higher degree similarly recover the corresponding
higher-degree H coefficients.

Zero singleton Fourier coefficients obstruct a first-order
**least-squares/Fourier** monitor. They are not an impossibility
theorem for arbitrary nonlinear probes or linear-threshold
classification. An adaptive probe that sees a bit of X and
then its partner in Y can obtain nontrivial information.

## Finite executable witness and resource budget

For a compact exact control, define Rubinstein-type h16 on
four consecutive four-bit blocks. A block accepts precisely
1100 or 0011; h16 is the OR of block acceptances.
There are eight disjoint adjacent-pair flips sensitive
at the all-zero input, and exact enumeration gives
s(h16)=4. This elementary finite witness has a quadratic
type of coordinated gap, not the asymptotic #132 separation.

Add one sentinel bit w and set
h17(z,w)=h16(z) OR w. The 17-bit source has 9 disjoint
zero-sensitive blocks and s(h17)=5. Its 34-bit parity
lift has all singleton Fourier coefficients zero,
s(G)=10 and bs(G,0)>=9.

The h16 rejection probability is exactly 2401/4096.
Consequently the sign h17 has

\[
\widehat H(\varnothing)=-1695/4096,\quad
\widehat H(\{w\})=2401/4096.
\]

A paired sentinel Fourier feature captures
2401/4096 correlation with the hidden sign label. The
best raw-singleton affine probe has MSE
1-(1695/4096)^2, whereas including the paired sentinel
reduces this by (2401/4096)^2.

A simple sign-majority f5 provides a SMALL explicit surface
mode for the selector construction. The finite f5 does not
witness #192's unbounded-ratio statement. The constructed
finite F has degree 35 and can be exhaustively diagnosed
through the analytic selector and parity-lift Fourier
transport identities without enumerating all 2^40 inputs.

## Neural extension

Next apply the dual probe to recurrence-monitorability Q5
and any available frozen LoopedTransformers activation trace.
Select binary features from a fixed, independently defined
basis, with their paired relationships registered before
scoring. Test singleton correlation, pair-parity probes,
and coordinated input interventions on held-out graph
families and checkpoint seeds. Charge feature-extractor
weights, probe readout, precision, intervention search and
the recurring hidden-state budget separately.

A positive paired-probe result on synthetic XOR data is a
diagnostic of the evaluation pipeline. It cannot establish
a hidden reasoning mode, model belief or sentience in a
trained language model. No original Q5 checkpoint is
claimed available by this finite campaign.


## FPB-5: Adaptive monitors learn nothing without completing a pair

Let X,Y be independent uniform n-bit strings and let Z=X xor Y.
Consider an adaptive monitor with its own randomness independent
of (X,Y). It may query individual input bits of X or Y, stopping
at a time determined by previous observations. Suppose that along
**every possible execution path** it queries at most one side
of each paired coordinate i: observing X_i precludes observing
Y_i, and vice versa. Repeated queries to the same side
are harmless.

Then its entire observation transcript T, **including which
coordinates it chose, observed bit values, and stopping decisions**,
has a distribution independent of Z:

\[
\boxed{I(Z;T)=0,\qquad I(H(Z);T)=0}\tag{FPB-5}
\]

for every deterministic or randomized hidden target H.

**Proof.** Condition on an arbitrary fixed source value Z=z.
Generate X uniformly and set Y=X xor z. For a coordinate i
whose pair has not yet been queried, both X_i and Y_i are
unbiased and independent of all observations from previous
coordinates; this remains true conditional on any fixed z.
The monitor's next queried side is a function of its current
transcript and independent random coins. Consequently,
conditional on any possible transcript so far, the law of the
next newly observed bit is a fair coin for every z. A repeat
query to the same side reveals its already-known value.
The conditional laws of future query choices and stopping
decisions therefore remain the same for all z. Induction over
the query sequence, including stopping, proves the same law
for the full transcript under every z. QED.

Querying both X_i and Y_i for one index instead reveals the
corresponding hidden bit exactly, since

\[
\boxed{Z_i=X_i\oplus Y_i.}
\]

An adaptive monitor can read as many as n distinct raw input
bits while remaining in the independent regime; the key
resource is the number of **completed matched pairs**.
The rule is structural, so it applies to arbitrary nonlinear
decoders of the unpaired transcript. It does not prohibit a
monitor from intentionally querying both sides of a pair.

The exact finite reference tests one genuinely adaptive
unpaired policy: it queries X_0, then chooses X_i or Y_i for
the next index depending on the previously observed bit.
The conditional transcript laws are enumerated for every
hidden z through n=5; all agree exactly. A completed-pair
oracle is checked independently and reveals one hidden
parity bit with disjoint conditional transcript supports.

This gives a falsifiable feature-acquisition budget for a
mechanistic monitor: a collection of individually salient
observations can be useless until the correct interaction
grammar permits two-sided access to the same hidden factor.
A real model's feature pairing and latent source law would
need separate evidence before importing the theorem.
