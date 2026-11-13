# DMB-01: deletion channels, finite monitor memory, and precision

Date: 2026-11-13.
Evidence: self-contained finite mathematical derivations, exact Fraction checks,
and conditional consequences of two OpenAI mathematics release statements.
No kernel proof of the external theorems or trained model evaluation is claimed.

## 0. Source pair and status

**#122, Quantitative trace-reconstruction bounds with a uniform decoder**
(September 24 / October 5, 2026). The source's Lean scope reports
superpolynomial lower bounds on the number of independent ordinary deletion
traces needed for exact worst-case reconstruction of a binary word at every
fixed deletion probability in (0,1), even with unrestricted computation.
It also reports single-trace hard-pair TV tending to zero faster than
every fixed inverse polynomial. Scope:
https://github.com/openai/math/blob/main/lean/docs/122.md

**#140, Memory-sample lower bounds for noiseless Gaussian regression**
(September 27, 2026). Under a uniform unit-sphere prior and exact Gaussian
inner-product observations, a one-pass learner with at most Ad^2 persistent
bits needs c_A d log(1/epsilon) observations for 2/3 angular-recovery
success in sufficiently large d, for epsilon in (0,1/10]. Scope:
https://github.com/openai/math/blob/main/lean/docs/140.md

The model may have unrestricted computation, but only its terminal
finite state, stopping index and independent randomness are available
at output. The statements are claims of a developing mathematics
release subject to independent review. The comparator challenge
declarations alone contain admitted theorem bodies and do not serve
as standalone proof certificates.

Our finite experimental controls are independent of those source
claims. The new results below are small deductions or conditional
consequences, with their hypotheses written explicitly.

## 1. Monitoring contract

An unknown length-n binary source X produces independent ordinary
deletion traces T_1,...,T_m. Each input position is independently
deleted with probability q; retained bits are emitted in their
original order. The monitor sees complete trace words including
their lengths, and only stores a mutable B-bit state M.

The final decoder receives M and a random seed R independent of X.
An explicitly optional stopping index tau in {0,...,m} may also
be supplied; it has a separate log2(m+1)-bit information capacity.
It receives no earlier raw trace, deletion mask, position alignment,
unmetered cache, prompt-conditioned program, or raw observation at
decoding time. If these resources exist, add them to the ledger.

The unknown-word prior is uniform when claiming average-success
bounds. Worst-case distinguishing bounds are stated separately for
particular source words.

## 2. DMB-1: universal terminal-state success ceiling

For X uniform on the 2^n binary words, with any possibly randomized
streaming processing and final decoder,

\[
\boxed{
P(\widehat X=X)\le \min(1,2^{B-n}).
}\tag{DMB-1a}
\]

If the terminal decoder may also access the stop index tau, then

\[
\boxed{
P(\widehat X=X)\le
\min(1,(m+1)\,2^{B-n}).
}\tag{DMB-1b}
\]

**Proof.** Condition on all independent random coins R. For a
deterministic encoder/decoder with at most K terminal outcomes,
at most one source word per terminal outcome can be exactly decoded.
The number of correctly decoded words is at most K. Uniform X
therefore gives success at most K/2^n, and averaging over
independent coins preserves the inequality. For randomized decoder
outputs, use convexity or condition on its coins. K=2^B without
tau, or K≤(m+1)2^B with tau. QED.

This is an information-theoretic ceiling for a uniformly random
full trace. A structured low-entropy source can be reconstructed
with far fewer bits; see the explicit repetition control below.

## 3. DMB-2: length-deck collision and deletion TV

For a word x and length k, let Deck_k(x)(t) count occurrences of t
as a subsequence of x, with distinct retained index sets counted
separately. At deletion probability q, the probability of complete
trace t of length k is exactly

\[
\boxed{
P_q(t\mid x)=\mathrm{Deck}_k(x)(t)\,
(1-q)^k q^{n-k}.}\tag{DMB-2a}
\]

Suppose two distinct words x,y have identical decks at every
length ≤L. Their deletion trace distributions are equal on all
outputs of length at most L. Thus

\[
\boxed{
\mathrm{TV}(P_x,P_y)
\le \Pr[\mathrm{Bin}(n,1-q)>L].
}\tag{DMB-2b}
\]

**Proof.** Partition the trace output space by output length.
The total probability on length > L is the same for both sources,
since the retained count is Bin(n,1-q), independently of the word.
The short-length distributions cancel pointwise; TV of the two
remaining subprobability measures is at most their shared mass.
QED.

### Exact witness

For x=0110 and y=1001, the decks of lengths 0,1,2 agree.
Their length-3 subsequences are disjoint:

- x yields {110, 010, 010, 011}.
- y yields {001, 101, 101, 100}.

Their length-4 traces are also distinct. Therefore equality
holds in DMB-2b when L=2. At q=4/5 and retention 1/5,

\[
\boxed{\delta=\mathrm{TV}(P_x,P_y)
=\Pr[\mathrm{Bin}(4,1/5)\ge3]
=\frac{17}{625}=0.0272.}\tag{DMB-2c}
\]

The exact-rational reference evaluates all deletion masks and
checks the equality and the underlying deck counts.

## 4. DMB-3: hard-pair monitorability from m deleted traces

Let x≠y be two fixed words and δ=TV(P_x,P_y). For m independent
traces, coupling each observation pair optimally and independently
gives

\[
\mathrm{TV}(P_x^{\otimes m},P_y^{\otimes m})
\le 1-(1-\delta)^m.\tag{DMB-3a}
\]

If a monitor successfully reconstructs each of x and y with
probability at least s>1/2, its output distinguishes the two
hypotheses with average probability at least s. Le Cam's
two-point testing inequality and data processing imply

\[
2s-1
\le\mathrm{TV}(P_x^{\otimes m},P_y^{\otimes m})
\le 1-(1-\delta)^m.\tag{DMB-3b}
\]

For 0<δ<1, necessarily

\[
\boxed{
m\ge
\frac{\ln(2(1-s))}{\ln(1-\delta)}.
}\tag{DMB-3c}
\]

If δ=0, no finite number of samples achieves s>1/2 on both
words. The ceiling concerns even unrestricted storage/computation
and therefore also any finite-state monitor.

At q=4/5, x=0110 and y=1001 have δ=17/625. For s=2/3,
\((608/625)^{14}>2/3\), so the necessary number of independent
traces is at least 15. This is a *necessary* budget for this
finite pair, not a decoder construction achieving that success.

The #122 paper claims existence of much harder length-n pairs.
If its reported δ_n is superpolynomially small, DMB-3b directly
implies a superpolynomial necessary trace count. We do not
reproduce those asymptotic pairs in our finite tests.

## 5. DMB-4: simultaneous observation and memory bottlenecks

Let X be a uniform n-bit word. Write p=1-q for the retention
probability, and let all m traces be independently generated
conditional on X. The length L of a trace is independent of X
and has expectation np. Consequently

\[
I(X;T_j)\le I(X;T_j,L_j)\le H(T_j\mid L_j)
\le\mathbb E[L_j]=np.
\]

Conditional independence of the traces given X and the chain
rule imply

\[
I(X;T_1,\ldots,T_m)\le mnp.
\]

Data processing through a monitor with B-bit state and an
optional stop index tau gives the new two-resource bound

\[
\boxed{
I(X;M,\tau\mid R)\le
\min\{B+\log_2(m+1),\ mn(1-q)\}.
}\tag{DMB-4}
\]

Without a stop index, replace the first term with B. R denotes
public/decoder randomness independent of X; all source-dependent
communication must already be included in M,tau.

If exact recovery succeeds with probability at least 1-e for
e in [0,1-2^{-n}], finite-alphabet Fano also requires

\[
\boxed{
n-h_2(e)-e\log_2(2^n-1)
\le \min\{B+\log_2(m+1),mn(1-q)\}.
}\tag{DMB-4F}
\]

The uniform exact-success ceiling DMB-1 is tighter for small
terminal alphabets. DMB-4F is convenient when the source is only
partially reconstructed.

## 6. DMB-5: continuous spherical precision floor

Consider a separate problem, Gaussian regression: an unknown
uniform unit vector θ in S^{d-1} is estimated from observations,
but only a B-bit terminal state and optional stop index are
available to the final decoder. Let epsilon be angular accuracy
in radians, 0<epsilon≤pi/2, d≥2. The normalized spherical cap
volume is

\[
v_d(\epsilon)=
\frac{\int_0^\epsilon \sin^{d-2}t\,dt}
{\int_0^\pi\sin^{d-2}t\,dt}.
\]

Using sin t≤t and sin t≥2t/pi on [0,pi/2], we obtain

\[
\boxed{
v_d(\epsilon)\le \frac2\pi\epsilon^{d-1}.
}\tag{DMB-5a}
\]

The union bound over K possible terminal messages yields the
model-independent continuous-precision ceiling

\[
\boxed{
P(\angle(\widehat\theta,\theta)\le\epsilon)
\le\min\bigl(1,\frac2\pi\,2^B(m+1)\epsilon^{d-1}\bigr).
}\tag{DMB-5b}
\]

The stop-index factor is omitted when no index is revealed.
For target success s, this implies

\[
B+\log_2(m+1)\ge
(d-1)\log_2(1/\epsilon)+\log_2(\pi s/2).
\tag{DMB-5c}
\]

This is an elementary covering/packing bound. Source #140
provides an additional **sample** restriction when B≤Ad²;
it is stronger in regimes where terminal-state capacity alone
already allows the target accuracy.

## 7. DMB-6: transferring the Gaussian sample lower bound
## to independently erased measurement opportunities

Suppose a Gaussian learner is offered m opportunities to observe
an exact iid Gaussian row and its noiseless response, each
independently retained with probability r∈(0,1]. The learner
cannot see a removed observation; it may see independent
erasure flags. Let K~Bin(m,r) be its number of retained rows.

Assume the erased-observation learner achieves angular error
epsilon with success probability at least 5/6. Choose

\[
k=\lceil 6mr\rceil.
\]

Markov's inequality gives P(K>k)≤1/6. Simulate the erased stream
using at most k genuine full Gaussian samples. Pre-generate
independent erasure flags, feed the next fresh Gaussian sample
whenever a retained opportunity occurs, and abort if >k samples
are needed. The emulator's success probability is at least
\(5/6-1/6=2/3\).

The simulator must account for the original B-bit state and
its clock/sample counters, say \(B'=B+2\lceil\log_2(m+1)\rceil+3\)
bits. Under #140's hypotheses, if B'≤Ad², d sufficiently large,
epsilon≤1/10, and the source lower bound has constant c_A,
then its required number of Gaussian samples is at least
\(c_A d\ln(1/\epsilon)\). Therefore

\[
\boxed{
m\ge\frac{c_A d\ln(1/\epsilon)-1}{6r}.
}\tag{DMB-6}
\]

This is a **conditional corollary of #140**, with a deliberately
simple constant factor from Markov's inequality, and it remains
valid for dimension-dependent retention r under the hypotheses
above. No numerical c_A has been inferred from the paper.

There is no assumption that deleted string traces equal erased
Gaussian rows. They are separate observation models sharing
the same finite-state resource accounting.

## 8. Structured-source control and applications

Uniform n-bit strings contain n independent source bits. A
highly structured source need not. If X is equally likely to
be 0^n or 1^n, a one-bit streaming monitor can set its bit
when it sees any retained 1, and otherwise output 0. Its exact
success is

\[
\boxed{1-\tfrac12 q^{nm}.}\tag{DMB-7}
\]

At n=8, q=4/5, m=1, it already exceeds 0.9. This does not
contradict DMB-1 because the source is supported on two words.

Applications:
- **Compactification Theory RC-01:** charge monitor state,
  stopping-index disclosure, prompt-conditioned lookup tables
  and decoder parameters, rather than comparing raw model sizes.
- **LoopedTransformers / SEM-ECHO:** apply controlled deletion
  to *observable* semantic event streams. If the source is a
  structured trajectory rather than uniform words, estimate its
  actual prior and avoid assigning uniform-source theorems.
- **RefutationBench:** compare certificate fragments under
  deletion and a fixed evidence-memory budget. Exact replay
  of the original source/verification oracle is necessary.
- **StegoBench / Shaft:** bound high-entropy hidden-payload
  recovery from a bounded terminal state. Distributed storage,
  common model weights and generated structure count as
  independent side resources and can change the bound.

The code implements exact deletion laws, short-deck checks,
full-transcript MAP for small n, bounded-state MAP with a
deterministic transition, trace-product TV, and the elementary
Gaussian cap/erasure checks. It neither reproduces #122's
asymptotic hard-pair construction nor simulates high-dimensional
Gaussian regression training.

An experimental prediction to test: under the same storage
budget, the fixed-state monitor's exact uniform-word success
plateaus as extra deletion traces arrive, while a decoder
retaining full traces continues improving. Deletion-deck
collisions identify cases where even unlimited memory requires
many observations.
