# GAL-03: Access-priced rank and discovery barriers

Status: deterministic oracle lower bounds, randomized verification argument,
exact synthetic operation-count pilots, and standalone core-Lean candidates.
No native neural-model or wall-clock benchmark is claimed.

## 1. Rank is not a discovery certificate

Let A be an unknown n-by-p binary matrix. Entry probes reveal individual
bits; computation, adaptivity and storage are unrestricted. There is no
free instance-dependent advice. A factorization must specify A exactly,
not merely its rank. Promise rank(A)<=1.

On the zero input, any unread entry can be changed to one. The resulting
matrix is a single-entry spike, still in the promised class, and the entire
adaptive transcript is unchanged. Thus an exact factorizer must probe
all np entries on zero. This also holds for a verifier accepting a fixed
claim A=0 and rejecting every spike. If rank exactly one is promised,
follow an all-zero transcript until only one candidate spike coordinate
remains; while two remain the transcript cannot identify the matrix.
Hence some promised input requires at least np-1 probes, tight for the
single-spike family. The same argument in a hidden complementary block
with a known (r-1)-by-(r-1) identity gives (n-r+1)(p-r+1)-1
for rank exactly r. These are entry-oracle results, not lower bounds on
factorization given a trusted factor representation or a stronger
matrix-vector oracle.

For a randomized verifier or factorizer with error <= e on each input,
couple its random seed on zero and on a spike at j. Outputs can differ
only if the zero-input path probes j. The probability gap for the
event "accept zero" (or "output zero") is at least 1-2e. Summing
over j proves expected distinct probes on zero >= (1-2e)np.
This applies to two-sided bounded error with 0<=e<1/2; it is a paper
argument, not a Lean probability theorem. Verification by random
matrix-vector products has a different access model and does not
contradict this entry-oracle bound.

## 2. A finite-scale barrier even for rank at most one

Consider h online requests for entire columns of A; output every bit.
One entry read and one emitted bit each cost one unit. Direct scan costs
2nh. On zero, each previously unrequested column needs all n reads
before its output, since any unread entry in it could be a spike.
An adversary asks distinct columns until the p columns are exhausted.
A literal column cache achieves the matching worst-case cost
n min(h,p)+nh for the full promised class. Arbitrary deterministic
exact preprocessing and hybrid policies cannot improve this worst-case
lower envelope when all entry reads count. Memory sufficient to cache
the requested columns is allowed and separately priced if desired.
For 1<q<2, the exact stable q-fold threshold is ceil(q p/(2-q)).
In particular a 1.5x gain requires 3p requests. At q>=2 no finite
horizon suffices for p>0. Low rank alone does not imply accessible
compression in this interface. On a log-dimension scale this is an
exponential horizon, with explicit input size np; it is not an
unrestricted time-complexity breakthrough.

## 3. Representation and conditioning

A nonzero GF(2) rank-one matrix uniquely determines its two nonzero
factors, since the only nonzero field scalar is 1. Thus the class has
(2^n-1)(2^p-1) elements. Distinct exact representations need at least
ceil(log2((2^n-1)(2^p-1))) bits in a fixed-length encoding; a dense
factor pair uses n+p bits. A spike alone has a much shorter position
code; the information-access bound concerns discovering that position.
For the GAL-02 linear oracle with a known r-dimensional basis,
a cache that can subsequently answer every basis query without probes
must distinguish 2^r independent parity vectors, requiring >=r bits
of state. Dynamic basis geometry, workspace, dispatch and allocation
costs are additional.

A separate fixed-basis real-valued example illustrates conditioning.
Measure y1=x1 and y2=x1+delta*x2 where delta=2^-k, with each stored
measurement independently perturbed within [-2^-p,2^-p]. Recover x2
as (y2-y1)/delta; worst-case error is exactly 2^(1-p+k).
This is also a minimax lower bound for recovering x2 from these two
perturbed measurements alone: true measurement pairs (+eta,-eta) and
(-eta,+eta), eta=2^-p, both permit the observed pair (0,0) and their
x2 values differ by 4eta/delta. No additional restrictions on x are
assumed. To guarantee error <=2^-s requires p>=k+s+1. This is a
precision requirement on a fixed ill-conditioned observation interface;
direct measurement or a better basis can evade it. The receipt's
conditioning rows merely check the dyadic exponent formula, not an
empirical numerical-conditioning test.

## 4. Priced factorization pilot

Run node Galacticity/gal03.mjs. Discover exact GF(2) column factors by
incremental elimination with pivot tests and full-vector XORs. The
source-read count, tests and XORs comprise C_build. Independently
reconstruct and compare all entries, with cost C_verify=2np(r+1),
charging one AND and XOR per rank term, a source read and a comparison
per entry. Dense matvec charges u=2np+n, factorized matvec
v=2r(n+p)+n. Intermediate allocation, resident access, loop control
and writes are free; emitted output costs one per bit. The factors'
binary payload is r(n+p), not an RSS or peak allocator measurement.
Discovery may keep the original matrix and intermediate copies, and
the receipt does not assert a physical memory bound.

For q>=1, first stable horizon is max(1,ceil(q C/(u-qv))) if
u-qv>0, where C is the selected setup charge. At u=qv it exists iff
C=0, and at u<qv it cannot exist. These costs are for the specified
algorithms, not optimal exact verification or factorization costs.
For square matrices and q=2, the marginal gate is
2n^2+n>8nr+2n, hence r<(2n-1)/8. Even trusted factors at r>=n/4
do not provide 2x under this model. The full build+verify horizon
depends on the actual elimination path and verification algorithm.
A discovery lower bound of np alone does not imply a huge matvec
horizon because dense execution already costs Theta(np) per call.

The campaign verifies all 512 3-by-3 matrices against all eight
input vectors, 204 single-spike cases, 20 generated positive-rank
matrices with identity anchors and eight matvec replays each.
Total 5,306 checks. Example n=128,r=16: trusted 1, discover 18,
discover+verify 86 matvecs for 2x. At n=128,r=32 no 2x horizon.
This is a synthetic exact binary analogue, not BPD floating-point
timing or a claim that ordinary BPD weights have these ranks.

## 5. Priced rank cache

Run node Galacticity/gal03_basis.mjs. The GAL-02 oracle is retained,
with price w per slice-local linear probe. Direct pays b probes and
b parity XORs per nonzero request. The literal cache pays r+1 ideal
key/lookup units per request (r for bit-key processing, one lookup),
then direct work on misses. The basis cache charges one pivot test
per retained vector, r bit XORs per elimination, one parity XOR for
each elimination, r units to locate a new pivot and one parity XOR
to store its corrected value. Packed zero test, dispatch, memory
access, writes, and allocation are free. Output costs one unit in
all arms. Payload cap: r+1 bits per literal entry and
r+1+ceil(log2 r) bits per basis entry (values, keys/pivots only).
No evictions; cap rejection falls back to direct on misses. This
is a declared logical-cost model, not a hardware model.

The fixed 81-cell grid varies r in {8,16,24}, b in {2,8,32}, probe
price in {1,16,128}, and 256 requests of each of three types:
four repeated one-hot labels, a 4-dimensional subspace, and dense
full-dimensional queries. Winner by priced total: direct 23,
literal cache 37, basis 21. Ties use the fixed order direct, literal,
basis. All replay comparisons preserve answers. Small exhaustive
controls use all two-slice 3-bit worlds, every length-three sequence
of nonzero labels, and four payload caps, across both caches.
175,616 bounded-payload replay comparisons and 351,394 checks.

For r=24,b=8,dense workload the basis uses 192 probes vs 2048 direct.
At w=1 its cost is 79154 vs 4352; at w=128, 103538 vs 264448.
For any two fixed traces with cost wP+A, a positive probe saving
P_old-P_new pays its extra non-probe work only above the ratio
(A_new-A_old)/(P_old-P_new); if that ratio is negative the saving
already pays. This is a trace comparison, not an optimized online
selector. A real adaptive router must meter its own estimation,
decision, construction and fallback costs.

## 6. BPD transfer: reuse within invalidation epochs

Source pinned FUSE-09 report:
https://github.com/MoralityLabAI/BPD/blob/1ee5a870c25e88ba6cafa07f349297498daf945f/mechinterp/bpd/experiments/fused_residual/FUSE09_RESULTS.md

At fixed W,V,U, a residual D=W-(VU)^T is constant under source-only
PGD differentiation; sharing detached D is valid only within the
registered source-only scope. The report excludes retained joint
factor gradients from this reuse. With T eligible closed-loop passes
and 2 PGD iterations, N=3T calls per selected site. Its reconstruction
counts are N->1; with four selected sites at T=12, 144->4. Those counts
exclude native final adversarial, calibration and other losses.
The best completed wide PGD timing ratio reported there is 1.3900x,
width 512, 12 passes, 8 MiB payload cap, four paired updates; production
was held and two wider configurations remained incomplete at that
report snapshot. No new native timing was collected in GAL-03.

Model one cache epoch with reconstruction cost R, additional setup C,
per-call cache overhead H, and common per-call work L. Baseline
N(R+L), cached R+C+N(H+L). A q-fold gain requires
N[R-qH-(q-1)L] >= q(R+C).
If the bracket is nonpositive and R+C>0, the requested speedup is
inaccessible. For ordinary cost improvement the shared L cancels.
Across weight-version invalidations, charge each new fill and
reconstruction; global reuse count is not an epoch horizon.
FUSE-10 calibration transfer should isolate original materialized
residual reuse under no_grad, preserve native numerical and lifecycle
gates, log eligible calls/version epochs and costs, and compare full
trainer durations on matched roots. This is a design consequence,
not a change to BPD or a validated prediction.

## 7. Next gate

GAL-04: priced adaptive routing across changing resource regimes,
with cold/warm epochs, metered structure discovery, explicit memory
limits, held-out workloads, and independent certificates of safe
fallback. No online policy may use future query labels or hidden
worlds to choose a cache. Compare against direct, literal, basis and
an offline oracle clearly labeled as a non-deployable upper bound.
