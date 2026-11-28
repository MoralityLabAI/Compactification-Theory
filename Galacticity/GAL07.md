# GAL-07: Epistemic galacticity

Status: exact restricted audit-cap theorem, finite ambiguity-aware
planning, information-purchase and distribution-shift controls.
No native model training, timing or unrestricted complexity claim.

## 1. Validation access and hard capital

An unknown proposed representation is either correct on all N validation
cases or has exactly one failing case at j. A validation probe of j
returns pass/fail and costs c>0; there is no stronger oracle, external
certificate, free instance-dependent advice, or verified income before
acquisition. Admission requires paying construction price B>=0 after
testing. Rejected proposals need not be acquired. The policy has private
randomness independent of the hidden case.

Deterministic exact admission on the clean world forces all N probes:
any unread coordinate can be the single failing case with the same
transcript. For random policies, couple coins on the clean world and
the j-failure world. On clean admitted paths that never probe j, the
same decision admits the faulty world. If clean admission probability
is p0>=p>0 and each false admission probability is <=e, then
Pr(clean admission AND probe j)>=p0-e. Summing over j gives expected
distinct probes >=N*max(0,p-e). This expectation bound is attained by
full audit with probability max(0,p-e), unconditional admission with
probability min(p,e), and rejection otherwise, for 0<p<=1 and e>=0.

Stronger hard-budget statement: on every clean admitted execution, if
initial capital is K>=B, the number of distinct probes is at most
m=floor((K-B)/c). Count accepted probe incidences two ways:
N*(p0-e)<=p0*m. Therefore m>=N*(1-e/p0)>=N*(1-e/p)
when e<=p. Conversely, with probability p choose a uniform m-subset
of N, inspect it (abort on failure), and admit on all-pass; otherwise
reject. It admits clean with probability p and any spike with
probability p*(1-m/N). Thus exact minimum capital is
B+c*max(0,ceil(N*(1-e/p))). This assumes free random sampling,
bookkeeping and tests besides their declared costs; any added
nonnegative cost cannot weaken the lower bound. For e>=p, no audit
is needed and B suffices. The construction does not grant free
acquisition; the hard budget applies to every branch. For N=2^k
the barrier grows exponentially in the address width and linearly
in explicit validation cases, not an unrestricted runtime lower bound.
The audit argument is elementary query indistinguishability; no novelty
claim for classical testing or lower-bound techniques.

## 2. Exact reserve versus verification

One candidate costs c0 to acquire. In a good world its payoff makes
it successful and solvent; in a bad world its signed settlement is -L
with L>=0. Both worlds possible, only one terminal purchase or abstention.
To achieve purchase on the good world with pathwise zero ruin, one
may either fund c0+L and buy regardless, or pay v for a perfect
distinguishing test and then buy only on good. The exact minimum is
c0+min(L,v) in this two-action model (including v=0). If the contract
also requires never buying a bad representation, only the tested
strategy is valid; its capital is c0+v. Verification can release a
reserve but costs capital itself. With unobserved bad worlds sharing
all observed signals with a good world, deterministic admission cannot
separate them. A proof or stronger test may change the interface.

## 3. Fixed-prior ambiguity and exact planner

Environment: finite known worlds; public project costs and signed gross
terminal payoffs; public deterministic test outcome table and test
prices; hidden actual world. Test once per identity, at most a known
number of tests. A test is paid upfront and only reveals its own signal.
Then choose another legal test, abstain, or buy one terminal project.
Purchase must be affordable. Final cash is K-c+payoff; ruin means
negative final cash, success means solvent and gross payoff >= c+gain.
Abstain is solvent failure. No overlapping projects or borrowing;
no controller time, holding charge or native performance measurement.

The ambiguity set is the convex hull of finitely many prior vertices
(all same total integer weight W). Nature chooses a single prior at
the start; its identity is not observed. One policy must serve all
priors. For any fixed policy, success and ruin are world subsets
(S,R). Worst success = min_v mass_v(S), worst ruin = max_v mass_v(R).
Linearity means checking vertices suffices for all convex mixtures.
Maximize worst success subject to every vertex's ruin <= budget.

At each state (remaining capital, compatible world mask, used test
mask, remaining test count), enumerate attainable event-mask pairs.
At a test observation node, combine all branch choices by set union
on disjoint compatible-world masks. At decisions take the union of
available policy sets. Discard a pair only if another has a superset
success event and subset ruin event; this is dominance for every
nonnegative prior, and survives disjoint branch union. Retain one
policy witness per identical pair. A constant-signal test cannot help:
skip it and emulate the same continuation at greater capital with
at least as many tests remaining. Backward induction establishes
exactness for deterministic history-dependent policies. Do not
prune based on one prior's scalar score. The algorithm is exponential
in worlds/tests and not a free planner; witnesses replay against every
world using separate capital accounting. The objective ignores test
cost beyond affordability (cost reduces remaining capital), so tie
selection may spend more than the cheapest equally successful policy.
Randomized ambiguity policies are outside this experiment.

Rectangularity matters: prior vertices on four worlds are
(9,1,1,9)/20 and (1,9,9,1)/20. Observe which pair of worlds occurred:
A={0,1}, B={2,3}. A fixed policy ruins on {1,3}. Each prior gives
probability 1/2, but separate branchwise worst conditional probabilities
are 9/10 in both branches. Replacing the shared prior by a separate
adversarial choice per branch gives an excessive 9/10 bound for
the original contract. Preserve full world-event masks or vectors
of prior-specific masses; taking scalar max/min before branch
combination changes the uncertainty set. See Iyengar, Robust Dynamic
Programming (2005), https://doi.org/10.1287/moor.1040.0129
for the rectangularity condition in robust DP. Information comparison
and costly experiments are established fields; see Pomatto, Strack,
Tamuz, The Cost of Information (2023),
https://cowles.yale.edu/sites/default/files/2024-02/p1854.pdf.
Our scope is capital-constrained finite verification, not a new
general robust-DP or Blackwell theorem.

## 4. Total variation robustness

For finite known universe and event E, TV(P,Q)=sup_A |P(A)-Q(A)|.
Over the radius rho ball around P, the maximum event probability is
min(1,P(E)+rho) if E is nonempty, else 0; the minimum is
max(0,P(E)-rho) if its complement is nonempty, else 1.
Proof: move up to rho mass between E and its complement; event
distance gives the matching upper/lower bounds. The same exact
integer formulas apply to mass-W distributions and integer radius.
Evaluate all prior vertices enlarged by a TV ball; the same event
extrema hold over the convex hull of those balls. This is not a
claim about the unknown true scenario distribution. Support matters:
a zero nominal mass event on a nonempty part of the declared universe
can carry up to rho mass under TV shift. An omitted world outside
the universe is not covered. If an additional bad world shares the
full accepted transcript, it can cause ruin regardless of nominal
validation; explicit coverage or an independently justified
contamination model is required. Under epsilon contamination
Q=(1-epsilon)P+epsilon H, a nominal-zero ruin event can carry epsilon
mass if H concentrates on such a world. A robust policy on the
declared world set can still fail on a missing world.

## 5. Exact campaigns

Run:
    node Galacticity/gal07_ambiguity.mjs
    node Galacticity/gal07_risk.mjs

40 synthetic environments: 8 roots in each of screenable, aliased,
alternative, hazard, expensive families. Six worlds, three prior
vertices of mass 24, a midpoint of two vertices for shift evaluation,
and a distinct out-of-hull prior heavily weighting worlds 4,5.
Same universe across all priors; no unseen scenario rows in this
campaign. Capital sweep 0..20 (840 configurations); 200 comparison
cases at capital 4,6,8,10,12. Compare exact nominal optimization,
exact robust optimization, robust optimization without information,
and pathwise safe optimization. 4,805 assertions, 280 tiny brute
frontiers, 1,645 policy-witness replays, 20,922 memoized main nodes.
The 12.5% risk cap (3/24) permits no nonempty deterministic ruin
event over the hull: each individual world has max vertex weight
>=4. Thus the robust quarter-of-half cap coincides with zero-risk
in this grid; do not mistake this for a general equivalence.

At 3/24 budget across 200 cases (total mass 4,800):
nominal worst-success mass 2,804, hull violations 35;
robust worst-success mass 2,682, hull violations 0;
robust no-tests worst-success mass 1,300, hull violations 0.
Robust information purchase improves worst success in 113/200
cases, reduces the minimum capital for 50% worst success at zero
risk in 39/40 environments. Both thresholds attained by capital 20.
Nominal policy violates the midpoint cap in 34 cases. Robust
zero-risk policies have no ruin on any declared world, so any
prior over these same worlds remains safe. This does not verify
new omitted worlds.

Looser cap 6/24 and TV mass radius 2 sensitivity (same cases,
post-hoc control; not independent heldout):
nominal worst-success 3,138, hull violations 50, TV violations 72,
outside violations 72; hull robust worst-success 2,872,
hull violations 0, TV violations 34, outside violations 28;
TV robust worst-success 2,682, TV-adjusted worst-success 2,556,
hull/TV/outside violations all 0. Hull robust takes nonzero risk
in 34 cases; TV robust selected zero-risk policies in all cases.
This is a genuine tradeoff on the same known scenario universe.
There is no outside-ambiguity guarantee from the hull objective;
the TV arm's outside success is incidental to its selected zero
ruin events. Aggregate mass sums are case summaries, not one
common worst prior across all cases or independent population estimates.
No calibration or learned policy training.

Risk campaign: 13,690 assertions, 1,568 integer TV event envelope
cases (all mass-6 distributions on three worlds), 4,608 hard-cap
audit parameter cases, exact BigInt scaling profiles, and the
fixed-prior counterexample. These tests check finite arithmetic
instances; the coupling and optimality arguments are paper proofs.

## 6. Research gym and next step

Expose costs, test menu, signal partitions, possible payoffs, prior
vertices, risk cap and capital. Keep the actual world and oracle
answer private. A submitted policy must be nonanticipative; replay
every world, assess all priors with a common policy and verify
optimality against the exact oracle. Treat unknown support as
a model failure, not a calibrated prior. Distinguish cost of
discovering/validating scenario coverage from cost of solving a
known finite model. No DataGoblin or BPD code or parameters changed.

GAL-08: budgeted discovery of verification interfaces. Study
how a structural certificate can replace exhaustive audits, charge
the cost of constructing and verifying that certificate, and
characterize when its advantage survives adversarial missing
structure. Use a paired promised-structure versus unstructured
control, not an assumed transfer to real model weights.
