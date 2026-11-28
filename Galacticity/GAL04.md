# GAL-04: Adaptive compactification markets

Status: exact synthetic online-cost campaign, mathematical derivations,
core-Lean arithmetic certificates. No native model update or timing claim.

## 1. Fixed-price representation investment

One epoch begins cold. Exact representation construction costs B>0 once.
Request t reveals its incremental direct-versus-compact execution cost
d_t>=0 before service. Compact service and any mandatory output work are
common to both arms and excluded from the incremental comparison.
Construction is sufficient to serve this and every remaining request.
There is no holding cost, shared capacity conflict, future information,
or controller cost in this theorem. The horizon and total savings D are
unknown; the fixed B and current d_t are known.

The offline optimum is min(B,D): if buying at all it can buy before the
first request, as all rents are nonnegative. Rent until the first request
for which prior rent plus current d_t is >=B, then buy before service.
If no such request occurs, cost=D<B and is optimal. Otherwise D>=B and
paid rent P<B, so cost=P+B<2B. If all amounts are integers, paid rent
<=B-1 and ratio <=2-1/B. Unit rent achieves that bound at horizon B.
For any deterministic unit-rent policy, let t be its purchase day if
the sequence continues. For t<=B, stop on day t: ratio=(B+t-1)/t
>=2-1/B. For t>B, stop on day t: ratio=(B+t-1)/B>=2. Never-buy has
unbounded ratio. Thus the discrete unit-rent bound is sharp. For B=1
the optimal threshold buys at the first positive rent, attaining 1.
For variable integer rents, the same upper bound holds; unit rents
are a subfamily. Adding nonnegative common service work preserves
an upper competitive ratio >=1. Independent cold epochs can be summed.
Known invalidations end each epoch; construction cannot cross them.

This is classical ski rental (see Online Algorithms, Cambridge,
https://doi.org/10.1017/9781009349178.007 for related task systems;
https://www.cambridge.org/core/books/abs/online-algorithms/skirental/E2C1D7248E39A4BBD055004818FDBB21
for ski rental). Related variable-size caching: Young, On-Line File
Caching, https://digitalcommons.dartmouth.edu/cs_tr/154/.
The contribution is the explicit mapping of representation setup,
admission and invalidation to the cost contract. No novel competitive
constant for the classical problem is claimed.

## 2. Prediction-augmented threshold

Let 0<lambda<=1. A prediction says whether D>=B. If predicted long,
purchase before crossing lambda B; otherwise use threshold B/lambda.
For a correct long prediction, prior rent <lambda B and OPT=B, so
ratio <=1+lambda. A correct short prediction never buys and is optimal.
For an incorrect long prediction, if purchase occurs D>=lambda B
and cost <(1+lambda)B, giving ratio <=1+1/lambda; if no purchase,
it is optimal. For an incorrect short prediction, if purchase occurs
prior rent <B/lambda and OPT=B; otherwise D<B/lambda, giving ratio
<=1/lambda. Thus the uniform worst-case bound is 1+1/lambda.
The binary prediction can come from a research agent, but this
campaign uses hypothetical predictions, not a trained LLM.

Replay gal04_rent_buy.mjs: 749,774 checks, all rent sequences of
length 0..6 over {0,1,2,3} for B=1..16, including lambda=1/2,1/4
and the unit-rent sharpness checks for B=1..64.
This validates arithmetic instances, not an unrestricted proof.

## 3. Coupled market environment

Five sites, each with a known logical size, direct work coefficient,
cache-hit work and construction overhead. Public request event:
(site, execution price, holding price, invalidation mask). Current
event is revealed before the decision; future events are hidden from
online policies. Private semantic state is a 32-bit word per site,
regenerated on invalidation. A cache represents the exact parity
of the current word. Independent reference parity is computed by a
fixed 32-bit scan; every served output is checked. The parity
representation is deliberately a semantic fixture; declared cost
coefficients and payload sizes are synthetic, not actual bit probes,
allocation size or machine timing.

After invalidation, the controller may retain a subset of valid
entries and optionally build only the requested entry. No prefetch.
Eviction is free. Retained payload <= capacity. Build cost is current
execution price times direct work plus extra construction overhead.
Service pays direct or valid hit work plus one output unit. Holding
pays current holding price times post-action retained payload every
request. All costs nonnegative. The oracle minimizes this market cost
by dynamic programming over the 32 subsets, including invalidation.
Its planning cost is not charged; it is an offline lower bound, not
a deployable policy. Exhaustive independent recursion checks 1,364
small instances, and reconstructed DP actions are replayed through
the independent semantic/cost ledger.

Online policies: direct; always-admit LRU with capacity evictions;
credit threshold with LRU and idle expiration under positive holding
price. The threshold accumulates observed nonnegative net savings
on misses, admits when credit plus current saving reaches a
multiple of current build price, resets on invalidation/eviction,
and drops a requested cache with zero immediate net saving.
It charges one work unit per site scanned at each decision, one
additional current-site decision unit, and one per site scanned
during each victim choice. Other bookkeeping and exact arithmetic
are free under this declared meter. Direct pays no routing work.
No guarantee against the offline optimum is asserted for these
coupled threshold or LRU policies.

Calibration uses 6 seeds per family (stable, bursts, churn, prices),
three capacities 5,8,12, 72 episodes of 72 requests each. Evaluate
11 prespecified configurations (direct, LRU, 3 multipliers x 3 TTLs)
by total charged cost. Select once before heldout evaluation:
threshold 1 with TTL 12. The test has eight disjoint seeds per family
and three capacities, including the entirely new shock family,
120 episodes / 8,640 requests. Calibration is hyperparameter
selection, not neural or DataGoblin training. Different capacities
on one seed are paired and dependent. No statistical generalization
or independently blind external benchmark claim.

Heldout totals:
- direct 589227
- always admit LRU 556798 (includes 67180 controller units)
- fixed threshold/TTL4 400847 (includes 53260)
- calibrated threshold/TTL12 383489 (includes 53770)
- offline optimum 211309 (planning uncharged)

Calibrated saves 34.9% over direct overall, loses on 27/120 episodes.
All 24 churn episodes lose: 64118 vs 44172 direct (45.2% more).
Shock improves to 85791 vs 204810 direct. Worst test total/offline
ratio 2.6344. These are observed ratios, not guarantees.

## 4. Prefix-budget guard and capital friction

Let cumulative direct cost be D_t, spent cost A_t, and premium
delta=a/b>0. Require b A_t <= (a+b) D_t at every prefix.
Assume exact pre-action accounting and a gate cost g with
b*g <= a*d_t for each direct step d_t. The fallback of doing the
current request directly, evicting everything, and paying g is
affordable because prior prefix safety plus b*g<=a*d_t imply
b(A_{t-1}+d_t+g)<=(a+b)D_t.

A proposed action has a known worst-case controller bound M, and
proposal accounting costs E. Run it only when the remaining
integer budget floor((a+b)D_t/b)-A_{t-1} can cover
d_t+g+E+M. After computing the proposal, admit if its exact market
cost plus incurred work fits the residual budget. Otherwise execute
direct, pay all incurred work, clear caches, reset the proposal
policy. The reserve guarantees a rejected proposal remains affordable.
Induction establishes prefix safety for the specified cost model.
There is no comparison guarantee to offline OPT. Unknown runtime
costs, non-free eviction, or inaccurate controller bounds require
upper certificates or a larger reserve.

The implementation uses g=1, E=number of sites, M=n+1+n^2;
the latter bounds the declared threshold scan and at most n
eviction passes. It evaluates delta in {1/10,1/4,1/2} on the same
heldout episodes as post-hoc safety sensitivity, not a fresh policy
selection or test. All 25,920 heldout prefix assertions pass,
plus 1,364 tiny replays with delta=1. Totals: 582510,489103,434704.
Worst observed final/direct ratios 1.09691,1.24851,1.49330.
The unguarded policy is cheaper in aggregate; safety and controller
friction consume returns. The 10% guard nearly eliminates savings.
There is no actual wall-time budget bound.

A zero-premium prefix guarantee has a fundamental obstruction.
If the first acquisition costs strictly more than current direct
execution (including all overhead), then no earlier acquisition
could have produced savings. Assuming previous work cost at least
direct, that purchase violates zero-premium safety immediately.
This proves impossibility of the first purchase; induction shows
a controller with this guarantee cannot ever make such an
investment, even for indefinitely reusable representations.
For positive premium delta and constant direct cost u per request,
if first acquisition plus service costs u+K with K>0, and no past
savings, safety requires delta*t*u>=K at its purchase request t.
Any earlier controller overhead only increases the required budget.
Hence first admissible horizon >=ceil(K/(delta*u)).
This is a safety-induced access delay, distinct from profitability.
A zero-premium controller can acquire if its immediate cost is
already no greater than direct; that is outside this obstruction.

## 5. Research integration

DataGoblin gym contract: expose only current public event, past
ledger, known costs and cap; ask for retention/admission choice,
claimed upper cost and safety certificate. Exact verifier checks
legal mask, current version, output, capacity and ledger. Keep
future events and offline DP costs private until grading. Score
total charged cost, per-prefix excess, fallback/abstention,
semantic errors and memory violations. Train on development
episodes only, freeze before new family evaluation; no training
admission from this synthetic campaign. For BPD use versioned
fixed-weight residual epochs and measured bounded reconstruction
costs, do not share source-only detached residuals across joint
gradient scopes. For DISTILL, cache state must be policy-relative
and verifier cost included. No native code changed.

Next: finite-state market policies with online prefix work
functions, resource augmentation, learned forecasting behind
verified budget admission, and native receipts. Classical task
systems require metric switching costs; the free-eviction,
paid-construction, invalidation model here is directed and time
varying, so importing a metric guarantee without new hypotheses
would be invalid.
