# PEF-02S | Nonautonomous loop schedules and order-sensitive memory

The preceding PEF-02 theorem used the same contraction ratio at every
iteration. A looped transformer may instead have a variable per-step local
Jacobian contraction schedule, which changes effective precision.

Let k_0,...,k_(L-1) be positive integer contraction exponents in chronological
order and let b_j be a binary input at step j:

    x_(j+1)=(x_j+(2^(k_j)-1)b_j)/2^(k_j), x_0=0.

The final exact state is a binary fraction whose leading blocks consist
of k_(L-1) copies of the newest bit b_(L-1), followed by k_(L-2) copies
of b_(L-2), and so on until k_0 copies of the oldest input.

Define K_r=sum_{j=1}^r k_(L-j), the cumulative widths of the r newest
blocks, and K=K_L. For B>0, the B-bit fixed-point readout
C_B=floor(2^B x_L) exposes exactly the t most recent inputs, where

    t = min{r in [1,L] : K_r >= B}, with t=L if B>K.

The exact fibre count remains 2^t outputs, each with 2^(L-t) histories,
and uniform-source full-history success is 2^(t-L).
The sharp full-history threshold is now

    B_min = 1 + sum_{j=1}^{L-1} k_j = K-k_0+1.

**New schedule gap theorem.** For t>=1, among exact orbit points
belonging to different t-most-recent-bit classes, the minimum gap is

    g = 2^(-K) + min_{1<=r<=t}
                    [(2^(k_(L-r))-2) 2^(-K_r)].      (PEF-02S)

Proof: the suffix-start values are sums of 0 or all-one binary
blocks. If two suffixes first differ at the r-th latest block,
their initial nonzero separation is (1-2^(-k_r))2^(-K_(r-1)).
All later retained blocks can offset at most
2^(-K_r)-2^(-K_t), so their smallest possible start gap is
(2^(k_r)-2)2^(-K_r)+2^(-K_t). The maximum unobserved older-history
tail is 2^(-K_t)-2^(-K); subtract it and minimize over r.
The minimum is attained by choosing the first differing block
oppositely and all subsequent retained blocks against it.

The sharp adversarial analog-noise threshold remains 2ε<g. With an
independent bounded additive disturbance of size η after every
step, the end error is at most

    η S, S=sum_{j=0}^{L-1} product_{r=j+1}^{L-1} 2^(-k_r).

For a final B-bit fixed-point floor quantizer, the sufficient
robustness guard is 2η S + 2^-B <= g. The per-step floor rounding
also commutes exactly with endpoint rounding for these integer-base
binary-block recurrences, by the same floor identity as PEF-02.

**Order-sensitive witness.** At L=3, B=6, these schedules share
the same multiset of contraction widths {2,2,4}, total exponent K=8,
and dimension-surrogate mean K/L. Yet chronological widths

    A = (2,2,4): t=2, g=9/256
    B = (2,4,2): t=2, g=57/256
    C = (4,2,2): t=3, B_min=5

At equal retained history count, schedule B has **19/3 times** the
analog suffix separation margin of schedule A. Rearranging the same
contractions can also change how many bits remain visible: C retains
all three bits at B=6, while A and B retain two.

The result shows the average contraction or entropy-rate/dimension
proxy alone is insufficient to predict a finite-depth recurrent
history-recovery and robustness frontier. The ordered partial
products and observable coordinate normalization are material.

## Evidence

Exact integer and Fraction code enumerates every mixed schedule with
k_j in {1,2,3}, depth at most five and all B from 1 to K+1.
It checks the closed gap theorem, uniform fibre counts, and repeated
floor-versus-end-only identity. Separate tests verify the exact
order-sensitive witnesses, error accumulation, and sharp precision
threshold. CI archives the suite.

Run:

    python -m unittest discover -s research/precision_entropy_frontier -p test_pef02_schedule.py -v
    python research/precision_entropy_frontier/pef02_schedule.py --suite

The Lean finite witnesses and precision threshold results are in
MathCan/PrecisionEntropySchedule.lean.

Extension to learned loop layers requires measuring ordered
input-conditioned Jacobians and nonlinear error propagation. Replacing
these dyadic contraction factors by a single measured Lyapunov
exponent is a hypothesis to test, not a theorem about neural weights.
