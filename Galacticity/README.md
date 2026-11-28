# GAL-01: Irreducible Galacticity
Status: initial restricted theorem and exact synthetic campaign, 2026-11-28.
Companion MathCan branch: research/gal01-nonperturbative-20261128.

## A. Finite-scale profiles
Fix a task P, size map, admissible programs (uniform, exact unless error budget
explicit), resource envelope R, cold/warm start convention, and fixed baseline B.
For each program A define W_A(n,m) as the supremum charged cost over legal
size-n workloads of m requests, including instance-dependent setup, dispatch,
query, verification, and output. Count peak memory and advice separately;
charge external preprocessing or declare it as a different warm-start model.
A cost vector may be compared by Pareto order or a declared scalar price.
Fix q>1 and a workload axis. Define S_A(q)={m: q W_A(n,m)<=W_B(n,m)}.
A first hit is min S_A; a stable tail threshold is min{M: every m>=M is in S_A};
empty sets have threshold infinity. For a single uniform program achieving
target asymptotic exponent alpha, take infimum over its stable threshold
subject to R. This differs from the pointwise optimum (program can vary with m).
A finite sweep only gives observed hits or censored intervals, not a tail proof.
Admissibility includes the same answers, input access and risk; approximate
arms must use a registered tolerance. Larger admissible families can only
decrease the infimum. No machine-invariance claim: a simulator with a
multiplicative factor c changes the speedup requirement by c; additive costs
matter at finite scale.

## B/C. Algebra and escape routes
If B(m)=m u and A(m)=C+m v with nonnegative C, q>1, then q A<=B iff
q C<=m(u-qv). For positive delta=u-qv the threshold is max(1,ceil(q C/delta)).
If delta=0 and C=0 all positive m qualify; otherwise delta<=0 yields no hit
(except the delta<0 case at m=0, outside the workload domain).
Finite-prefix specialization and fallback can remove slowdowns but cannot
supply q-fold speedup at a scale where every admissible algorithm has cost
greater than B/q. Caching, training and shared preprocessing must be charged
once over the workload boundary; output/verification costs cannot be silently
dropped. If an exact quotient has build cost C and saves d>0 per repeat,
its ordinary break-even is ceil(C/d) (ties allowed). For workload sequences
with changing gains, use prefix cumulative savings, never a stationary fit
without a residual check. Ideal portfolio cost min(A,B) is a lower envelope,
not a free implementation; a sequential dispatcher must pay decision work,
parallel racing pays aggregate resources and may need cancellation.
For uniform prefix budgets an inclusion of admissible sets yields monotonicity;
no inference about a universal fastest program follows (cf. Blum speedup).

## D. Exact oracle theorem
K independent blocks of b arbitrary bits, K,b positive; requests reveal a
block address and require its parity. Online deterministic exact answers,
cold start, arbitrary charged bit probes, one unit per emitted answer, no
free data-dependent advice; storage adequate for K cached parities. All other
operations are free in this deliberately restricted query model.
Direct scan cost D=m(b+1). For m<=K an adversary asks distinct blocks. Each
requested bit must be read at least once by answer time: otherwise flip that
bit, keep the transcript of all probes and addresses identical, and change
the correct parity. For m>K request all K blocks first. Thus worst-case
cost >=b min(m,K)+m including all preprocessing. Lazy memoization attains
this on every workload. An optimal single algorithm therefore achieves
O(m+Kb) (fixed K,b) and the exact lower envelope for all m simultaneously.
For integer 1<q<b+1, no m<=K qualifies and the inequality for m>=K is
m(b+1-q)>=qKb. Exact threshold ceil(qKb/(b+1-q)); q>=b+1 has no finite
threshold; q=1 gives 1. With K=2^k, the barrier is exponential in address
width and linear in explicit database size. It is a genuine model-relative
lower bound even with arbitrary caching and preprocessing, not a lower bound
on RAM complexity of explicit inputs or on an unrestricted problem.
Stronger setup tradeoff: if preprocessing reads p bits and every possible
first query can inspect at most t<b additional bits, every block must have
at most t unread bits on every preprocessing path. Therefore p>=K(b-t);
otherwise flipping an unread bit contradicts correctness. Cache the parity
of the first b-t bits per block to attain the probe bound (storage charged
separately). Free warm-start parity advice removes the cold barrier.
Randomized bounded-error algorithms, succinct correlated blocks and word
probes are separate models, with no transfer asserted.

## E. Experiment protocol and DISTILL adapter
Run node Galacticity/gal01.mjs; the checked-in receipt is deterministic.
The campaign checks boundaries, exhaustive tiny databases and query streams,
tensor recurrences and exact affine calibration. It records logical counts,
not machine timings. distill_profile.py reads archived DISTILL-11 row JSON
from a separate checkout of PR #314, pinned at
7fa19c90372b13d71676e6fef13819d2a18b9be3.
The native quotient is policy-relative; do not reuse it for arbitrary actors.
Compare literal vs flags/marginal with the same tail/broad mode, cap, root and
risk contract; record finite observed cap hits, never label them amortization
thresholds. Actual live rows were not available in this session.
For subsequent native runs record independently metered build, dispatch,
execution and verify costs and identical workload semantics. BPD: cached
calibration/residual build, repeated forwards, held-out intervention error.
LoopedTransformers: matched correctness, recurrent visits, routing overhead,
prefill and decode separately. DataGoblin: matched verified decisions,
training and verifier calls, cold and warm deployment. Nova: setup/folding/
final verification, proof size. TradeLayer: matched settlement and capital
feasibility, graph construction and clearing. Each requires source-pinned
receipts and crossover intervals with right censoring. No cross-gym speedups
are claimed from synthetic rows.

## F. Tensor finite powers
For n=2^j, multiplication weight w and addition weight 1, conventional
scalar arithmetic costs (w+1)8^j-4^j. Full recursion of Strassen (7 child
products, 18 half-size matrix additions per node) costs
(w+6)7^j-6*4^j. The campaign enumerates first 2x hits; these are exact for
this arithmetic schedule, not wall-clock measurements or lower bounds over
all algorithms. Practical hybrids and memory layout are independent arms.
More generally a tensor rank sequence r_j is submultiplicative; Fekete
gives asymptotic rank inf_j r_j^(1/j). A finite witness r_j can upper-bound
the exponent and certify repeated powers. Asymptotic exponent alone gives
no quantitative entry-scale guarantee; decomposition construction, coefficient
precision and realization overhead need a finite witness. The algebraic
Galactic Method is a named prior construction class and should not be
confused with this workload-scale notion.

## G. Research gym
The public affine calibration packets use two observations of A(m)=C+mv
and known per-use savings d, with disjoint construction cost ranges:
128 development cases C=1..16, 128 heldout C=17..32, d=1..8.
Predict the first nonnegative net savings horizon; provide identified C,v,
exact answer, evidence and a censor/abstain state. Two-point affine fit
solves 128/128; development median solves 0/128. A nonintegral slope is
rejected. This is an analytic baseline, not trained DataGoblin. Real gym
packets must hide oracle and hold out task family, source and root. Score
correctness and proof of boundary, false early claims, interval coverage,
and cost of investigation. Add nonlinear and nonidentifiable profiles:
finite observations cannot determine tail thresholds without constraints.
Require exact verifier and independent replay before admitting distillation.
Use local Inky Mini for future rollouts; no paid calls made here.

## Literature / scope
- Spielman & Teng, Smoothed Analysis of Algorithms and Heuristics:
  Progress and Open Questions (random perturbation framework).
  https://www.cs.yale.edu/homes/spielman/PAPERS/focmSmoothed.pdf
- Roughgarden (ed.), Beyond the Worst-Case Analysis of Algorithms (2021),
  particularly parameterized and instance-optimal chapters.
  https://www.cambridge.org/core/books/beyond-the-worst-case-analysis-of-algorithms/8A8128BBF7FC2857471E9CA52E69AC21
- Blum, A Machine-Independent Theory of the Complexity of Recursive Functions
  (1967), speedup theorem.
- Patrascu & Thorup, Time-Space Trade-Offs for Predecessor Search (2006).
  https://arxiv.org/abs/cs/0603043
- Alman & Vassilevska Williams, Limits on All Known (and Some Unknown)
  Approaches to Matrix Multiplication, https://arxiv.org/abs/1810.08671
- Alman, Limits on the Universal Method for Matrix Multiplication,
  https://arxiv.org/abs/1812.08731
The oracle argument is elementary; no novelty or unrestricted lower bound
is claimed. The proposed research agenda is resource-accounted finite-scale
profiles and structural barriers across computational compactification.
