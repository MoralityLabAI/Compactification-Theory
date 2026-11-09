# RRC-03: exact active-query capacity theorem and its scope

This note proves an elementary information bound for the finite RRC-03 family.
It is a proof about the **specified binary coefficient-comparison interface**, not
an infinite Rogers–Ramanujan identity, a new lower bound for arbitrary
mathematical discovery, or a theorem about recurrent neural networks.

## Definition: delayed Euler-product defects

Write

\[
F(q)=\prod_{n\ge 1,\ n\equiv 1,4\pmod5}(1-q^n)^{-1}
    =\sum_{n\ge 0} f_n q^n.
\]

Because part `1` is permitted, `f_n >= 1` for every nonnegative integer `n`.
Fix `T=64`, choose `L` distinct defect degrees `K_j=T+1+j` for
`j=0,...,L-1`, and consider the `H=L+1` equally likely hypotheses

\[
G_j(q)=F(q)/(1-q^{K_j})\quad (0\le j<L),\qquad G_L(q)=F(q).
\]

The extra factor is a new unrestricted part type of weight `K_j`. Every
hypothesis has the same coefficients through degree `T`, since its first
additional nonconstant term has degree at least `T+1`. All are proper
nonnegative-integer generating functions. The finite baseline identity is
independently checked by both the gap-two local recurrence and product DP;
the defect convolution is independently checked by an Euler-transform
reconstruction for every candidate.

## Lemma 1: binary threshold response

The agent may query one degree `n=T+1+t`, with `0 <= t < L`. The oracle
returns only the bit

\[
Y_t(j)=\mathbf{1}[\,[q^n]G_j(q)>[q^n]F(q)\,].
\]

Then `Y_t(j)=1` precisely when `j<=t` for `j<L`, and `Y_t(L)=0`.
Indeed,

\[
[q^n](G_j-F)=\sum_{r\ge 1:r K_j\le n}f_{n-rK_j}.
\]

If `n<K_j`, the sum is empty; if `n>=K_j`, its first summand is positive.
Thus a query divides the ordered candidate set into a left and a right
interval. A sequence of observations maintains the exact interval
`[lo,hi]` of consistent candidates. This interval is a sufficient public
history state for every subsequent binary query under the declared uniform
prior and noise-free oracle.

## Theorem 2: sharp adaptive and fixed-query capacities

Under the uniform prior on the `H` hypotheses, the optimal expected exact
identification accuracy after *at most* `B` binary oracle queries is

\[
\boxed{\quad \frac{\min(H,2^B)}{H} \quad}
\]

if queries may depend on previous responses. With a fixed query schedule
chosen in advance, it is

\[
\boxed{\quad \frac{\min(H,B+1)}{H}. \quad}
\]

**Proof (upper bounds).** A deterministic depth-`B` binary decision tree
has at most `2^B` nonempty leaf transcripts. At each leaf, a uniform-prior
MAP decoder correctly identifies one hypothesis; each leaf contributes
exactly `1/H` to expected success. For a fixed schedule, sort its queried
thresholds. As the hidden `j` varies in order, the binary answer vector
changes at most once at each of the `B` thresholds; there are at most
`B+1` distinct vectors and hence at most `B+1` nonempty transcripts.
Randomizing the policy is a convex combination of deterministic policies,
so these upper bounds apply to randomized schedules as well.

**Proof (attainment).** Adaptive midpoint queries recursively partition a
remaining interval as equally as integer cardinality permits. Their
decision tree has exactly `min(H,2^B)` nonempty leaves after `B` queries.
The fixed schedule of `min(B,H-1)` evenly spaced thresholds makes
`min(H,B+1)` nonempty intervals. Both choose the smallest surviving
candidate as the fixed deterministic MAP tie-break. Their exact finite
Python implementations are checked against an independently coded Bellman
recursion for arbitrary positive integer priors. QED.

At `H=33`, four adaptive queries achieve `16/33` accuracy while four
fixed queries achieve `5/33`. Six adaptive queries exactly identify all
33 candidates; a fixed schedule requires 32 threshold queries for that
same guarantee. This is a query-complexity separation **for this specific
binary threshold interface**. Access to unrestricted numeric coefficient
values, other experimental operations or different priors may change the
optimal query complexity.

## Posterior memory and architecture accounting

The public history quotient is the integer interval `[lo,hi]` with endpoints
in `0..H-1`. A direct exact implementation uses at most
`2*ceil(log2 H)` bits for the endpoints, apart from the known hypothesis
count, query budget, decoder and computation. The 33-hypothesis case uses
at most 12 endpoint bits. This is an upper bound for an algorithmic
controller, not a neural-model parameter count or a minimum-state proof.

The frozen neural student instead has 16 floating-point hidden values
(512 live FP32 bits) and 353 trainable parameters (11,296 FP32 parameter
bits). Its feedforward control sees the *entire public query history*,
and has exactly the same number of parameters. Both pass proposed queries
to the same exact external MAP decoder. The observed gap over a fixed
query schedule tests use of adaptive feedback. The comparison between
the two trained architectures tests whether a learned state is beneficial
under these particular inputs and budgets.

## Identifiability boundary

Before the first paid query, the family has a shared 65-coefficient prefix
and the posterior remains uniform. The RRC-02 prefix impossibility result
therefore holds. This experiment makes extra evidence available through
queries at degrees 65 and above; each answer has a counted one-bit
information budget. Its finite adaptive algorithm can exploit those bits
without deciding whether any hypothetical identity holds at infinity.
