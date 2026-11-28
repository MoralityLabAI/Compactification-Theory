# GAL-06: Stochastic computational capital
Status: exact finite hidden-world planning, reserve quantile derivations,
synthetic experiment, standalone research gym and core-Lean candidates.
No native training, economic gain, timing or model-call result.

## 1. Contract

Fix a finite set of hidden worlds with positive integer weights w_omega;
sample one world at the start. The entire scenario table and weights are
public, but the sampled world is hidden. Each persistent project i has
upfront cost c_i>=0, positive integer settlement delay, payload size,
prerequisite mask and a signed settlement payoff X_i(omega). One project
or idle per tick. Pay cost before acquiring, without borrowing. The
project occupies payload immediately; prerequisites must already have
settled. Commitments are mandatory and cannot be canceled, and a project
can be bought only if it settles by deadline H. No more acquisitions
after the goal is bought. At each time, simultaneously due payoffs settle
as an atomic net batch; their individual values become observable.
Negative post-settlement cash is ruin and terminates the episode.
At the deadline, success means the goal has been acquired and all
commitments settled without ruin. A policy may stop making investments,
but cannot evade already committed losses. No controller costs, recurring
holding prices or eviction in this model. The signed gross settlement
payoff is separate from upfront cost, so net gain is X_i-c_i. Payoffs
model independently accounted savings or losses in synthetic work units.

Observed outcomes partition the worlds. Given a deterministic policy
and history, the compatible mask B plus the original weights exactly
specifies the posterior: normalize the weights on B. State
(t,K,M,P,B) comprises time, current cash, acquired mask, pending
project/due-time pairs and compatible worlds. It is sufficient for the
future contract. Actions depend on observable state, not hidden world.
Correlation arises from the common latent world; the scenario table
is assumed correct. No guarantee under distribution shift outside
that table or under omitted liabilities.

## 2. Delayed cash-flow reserves

For an existing schedule of pending commitments and no further purchases,
let D_omega = max(0, max_t -sum_{i: due_i<=t} X_i(omega)).
The maximum ranges over atomic settlement times. For nonnegative
current cash K, ruin occurs on world omega iff K<D_omega.
The least robust reserve is max_{omega in B} D_omega. More generally
for allowed conditional ruin mass b, the least cash reserve is
min{K>=0: sum_{omega in B, D_omega>K} w_omega<=b}.
For an allowed conditional probability epsilon, b is floor(epsilon
times the compatible mass) when working with integer weights. Thus
the weighted loss quantile is exact, including atoms and strict ruin
at negative rather than zero cash. For no further actions this is
necessary and sufficient; it does not itself guarantee goal success.

For a new acquisition, require K>=cost+reserve(updated schedule,B).
If the prior reserve condition held, settling a batch leaves enough
cash for the remaining commitments in every still compatible world:
its future prefix sums are the original prefix sums minus the observed
batch. Restricting the compatible set cannot increase the required
reserve for a fixed schedule. Idle is always safe under this invariant;
induction gives pathwise zero ruin for every admitted sequence.
A marginal upper reserve sum_i max(0,-min_{omega in B} X_i(omega))
is safe but can discard beneficial correlations. It need not be tight.

Example two worlds, equally weighted: two payoffs individually distributed
as {-5,8}. With correlated pairs (-5,-5) and (8,8), simultaneous
settlement needs reserve 10. With anticorrelated (-5,8) and (8,-5),
simultaneous settlement needs 0, staggered settlement needs 5.
These are reserves of existing commitments, not claims that one can
acquire the hedge at zero initial capital under one-buy-per-tick rules.
A robust sequential controller may reject an intermediate exposure
that would be safe if jointly acquired. Atomic netting is essential;
settlement order inside a batch changes the result if netting is absent.

Repeatedly checking a positive local conditional risk allowance does
not imply the same global risk allowance. If conditional failure
hazards on surviving paths are bounded by epsilon_t, survival is at
least product_t(1-epsilon_t) by the chain rule. The union bound
sum_t epsilon_t also bounds failure. To enforce one global budget,
track actual branch mass or a sound allocation of residual risk.
Zero-risk pathwise reserves avoid this compounding under the model.

## 3. Exact deterministic probability frontier

Define attainable pairs (S,R) of unnormalized integer success and
ruin mass. The residual outcomes are solvent failures to reach the
goal. At the terminal observation node the unique pair is (0,mass(B))
on ruin, or (mass(B),0) on success, or (0,0) on solvent failure.
At a decision node take the union over legal acquisitions and idle.
At a settlement observation node partition B by the vector of observed
payoffs, then take the Minkowski sum of the branch frontiers. A policy
may choose a different continuation for each observable branch. The
unconditional masses of disjoint branches add, without rounding
posterior probabilities. Prune any pair dominated by another with
no smaller S and no larger R. Dominance is preserved by addition,
and replacing a dominated continuation cannot worsen an ancestor
constraint. Backward induction proves the frontier is exact for all
deterministic history-dependent policies in this finite contract.
Memoization over sufficient state shares the possible continuation
frontiers but does not force distinct histories to choose the same
pair. Witness trees preserve the chosen branch decisions.

For seed K and root mass W, maximize S/W subject to R/W<=epsilon.
Minimum seed for success probability p is the least K admitting
a point with S>=pW and R<=epsilon W. Larger capital can emulate a
lower-capital policy by keeping an internal balance offset; paths
on which the lower-capital policy survives have the same decisions,
and paths where it ruins can instead idle. Success cannot decrease
and ruin cannot increase, assuming idle never adds new liabilities.
Thus the constrained optimum is monotone in K. The planner is a
finite exact exponential oracle; world masks, pending schedules,
capital and frontiers can all cause state explosion. No free planning
or wall-clock advantage is asserted.

For randomized policies with private random seed independent of the
world, fixing the seed gives a deterministic policy. The attainable
probability set is the convex hull of deterministic pairs. With one
linear ruin constraint the maximizing mixture uses at most two
deterministic policies, by elementary planar convex geometry / linear
programming. If risk cap B0 lies strictly between r1 and r2, interpolate
s=(s1*(r2-B0)+s2*(B0-r1))/(r2-r1). Compare all such pairs and feasible
pure policies. This is exact rational arithmetic at the frontier
level. Randomization may deliberately spread risk across episodes;
it is a different contract from per-seed deterministic or pathwise
guarantees. Example seed 1, gamble cost 1 returns 10 or -4 equally,
goal cost 5. Frontier [(0,0),(1,1)] in mass units W=2. With zero risk
only waiting is feasible. A half-probability mixture gives success
1/4 and ruin 1/4 under an unconditional quarter-risk contract.

## 4. Experiments

Run:
    node Galacticity/gal06_stochastic.mjs
    node Galacticity/gal06_risk.mjs

The first campaign checks 224 tiny independently enumerated
unpruned policy-frontier instances, with unequal world weights.
Every retained frontier witness is replayed using separate per-world
cash/observation accounting, including nonanticipative tree branches.
Main grid: eight seeds in each of five scenario families (stable,
correlated, hedged, delayed, shift), 40 total environments, four equally
weighted worlds, capital 0..24 inclusive. The initial pilot at 0..16
failed an overly strong assumption that all thresholds were attained;
the final grid is wider and treats unattained thresholds as null (could
be above the sweep or structurally impossible). All final targets
were attained. The registered policy comparison uses capital values
2,4,6,8,10: 200 cases / total world mass 800. These are related
synthetic cases, not independent sampled external episodes. No
calibration, neural training or blind external evaluation occurred.
Heuristic policies: wait/direct goal; posterior-mean payoff per delay
with mandatory-project priority; same heuristic plus joint robust reserve;
same heuristic plus sum-of-marginal robust reserve. All enforce
remaining mandatory capacity, settled prerequisites and affordability.
The mean heuristic may accept signed liabilities that later cause ruin.

Outcomes (success mass, ruin mass), aggregated over 200 cases:
- direct/wait (40,0)
- posterior-mean (388,159), violates quarter-risk in 67 cases
- joint reserve heuristic (313,0)
- marginal reserve heuristic (313,0)
- deterministic zero-risk oracle: success 375
- deterministic quarter-risk oracle: success 399
- randomized quarter-risk envelope: success 418

Each oracle uses its per-case risk bound, not just aggregate risk.
Allowing quarter risk improved deterministic success in 19/200
capital cases. Randomized mixture improved over the deterministic
quarter-risk oracle in 27/200 cases. These are model-oracle bounds,
not deployable policy speedups. Joint reserve is less restrictive than
marginal reserve, but the greedy joint heuristic can take a bad action:
joint beats marginal in 4 cases, loses in 5, and ties in aggregate.
The safe oracle gap remains 62 success mass. A larger feasible action
set improves optimum, not necessarily a heuristic's policy.

Average minimum capital over 40 environments:
- certain success, zero ruin: 11.05
- 75% success, zero ruin: 10.4
- 75% success, <=25% ruin: 10.4
- 50% success, <=25% ruin: 4.4

No sampled 75%-success minimum fell when 25% ruin was permitted.
This is a negative result specific to these scenarios. It does not
contradict the observed improvements at fixed capital. No across-family
causal claim because several scenario parameters vary together.

Stochastic campaign: 4,138 assertions, 1,310 retained witness
replays, 321,298 main-grid memoized oracle nodes. Risk campaign:
26,852 assertions, 17,500 weighted reserve minimality cases,
3,400 small mixture-frontier controls, exact two-policy interpolation
versus exhaustive rational mixtures. Only the displayed aggregate
mixture sum uses floating-point; case values are rational pairs.
Both are finite tests supplementing the derivations.

## 5. Research gym

    node Galacticity/gal06_gym.mjs emit 71001 hedged 6 1 > packet.json
    node Galacticity/gal06_gym.mjs grade packet.json response.json
    node Galacticity/gal06_gym.mjs selftest

Public packet has schema gal06.packet.v1, capital, riskMass (integer
mass, not percent), and env containing the known scenario table,
weights, goal, H, cap, projects. It omits the actual world and oracle
answers. The deterministic response schema gal06.response.v1 has
successMass, ruinMass and policy. A policy starts with {kind:"chance",
by:{...}}; keys are comma-separated signed integer settlement
values in ascending project index order, or the empty string when
nothing settles. A live child is {kind:"act",i:projectIndexOrMinusOne,
next:chanceNode}; terminal child is {kind:"end"}. The grader validates
shape and resource limits, independently replays every hidden world
against the submitted contingent policy, compares claimed masses,
checks risk, and recomputes optimum under the declared bound. 363
self-tests passed, including wrong mass, invalid tree and unsafe
gamble. No LLM or native DataGoblin run; checked-in generator and
fixtures are public and not a blind training set. The grader computes
the exact optimum, so a witness alone is not an independently compact
optimality certificate.

## 6. Transfer and next gate

For DataGoblin, commitments can be costly experiments with delayed
verified savings. Forecast correlations through shared hidden
hypotheses, reserve obligations before experiments, and treat
unverified estimates as non-spendable. For BPD or recurrent-loop
savings, model version invalidation and resource charges; overlapping
savings must not be booked twice. No actual BPD/DataGoblin code or
weights changed here.

GAL-07: model uncertainty itself. Robust reserve depends on compatible
scenario coverage and chance frontier depends on calibrated weights.
Introduce a certified ambiguity set, adversarial missing worlds,
information purchase and value of verification, exact robust bounds,
and held-out shifted scenario controls. The key question is what
capital is needed to safely discover that a profitable transformation
really is profitable.
