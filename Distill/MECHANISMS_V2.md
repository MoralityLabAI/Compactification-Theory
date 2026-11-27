# DISTILL-12 / independently specified synthetic mechanism curriculum v2

Preserve the original 69-row v1 and its report. Do not promote those labels to
training. V2 replaces two of its toy mechanisms. The v1 overhead task remains
reference-only; a matched-resource overhead benchmark is separate work.

## Continuation equivalence

A state has an experiment budget and two fixed receipt flags. All states in
a pair share a schedule of remaining projects. On each project, one action
is chosen: defer, recommend either truth, buy a direct certificate, or use a
scope-eligible transport certificate. Rewards are expected values under the
stipulated marginal posterior. No observations, stochastic transitions or
new receipt acquisition occur. Costs deplete the same budget.

Define the terminal signature as the empty tuple. At each stage construct the
sorted tuple of (legal action, cost, exact expected immediate reward, successor
signature). Equality is a sufficient all-action congruence by induction:
every common action has matching immediate cost and reward and successor
class. Thus all legal action traces have equal costs and expected rewards,
and optimal continuation values match. Conversely, every legal prefix has
a defer extension; equality of full labeled traces with per-step costs and
rewards implies equality of these recursive signatures. An independent
enumerator checks this equivalence and the maximum total reward.

Equal scalar optimum alone does not certify congruence. For two projects with
certificate reward 11, direct cost 3, transport cost 2, valid transport receipts,
posterior 1/2 and recommendation rewards +6/-12, budgets 2 and 4 have
one-project value 9 but two-project values 9 and 18. A second witness has
equal direct and transport costs: dropping transport eligibility can preserve
optimal value while changing the legal action set.

A cheap sufficient key records the remaining per-project transport eligibility
vector and budget saturated at the sum of maximum eligible project costs.
Any feasible action string spends at most this sum, so budgets at or above
the cap cannot change feasibility. This key may miss legal merges; exact
recursive signatures decide the toy labels. This result does not establish
arbitrary-policy congruence in the stochastic native DISTILL-11 program.

## Multi-context quantization

A finite scalar belief distribution has weighted atoms and multiple weighted
decision contexts with distinct false-positive and false-negative costs.
One contiguous k-bin partition is shared across contexts. The representative
of a bin is its weighted mean. Each context makes a Bayes decision using
that representative. Structural evaluation adds an explicit abstention
action with constant cost, additional contexts and variable bin count.

The objective is expected decision loss minus unquantized Bayes loss.
Brier excess is reported separately. Both metrics are additive by segment.
An exact partition DP minimizes (decision regret, Brier excess, cut tuple).
An exhaustive independent cut enumerator instead computes bin losses
directly and uses the weighted pairwise variance identity for Brier excess.
Both verify every candidate and optimum. Brier-optimal DP and balanced-bin
partitions are separate comparators. A single decision threshold is no longer
a universally optimal partition. The exact DP is a strong analytic oracle;
any learned-controller claim requires a measured resource/utility tradeoff,
not a claim of improving on its mathematical optimum.

## Export

    python -m Distill.mechanism_curriculum_v2 --out runs/mechanism-curriculum-v2
    python -m unittest Distill.tests.test_mechanism_curriculum_v2 -v

216 rows: 96 train, 24 calibration, 48 fresh ordinary roots and 48 structural
shift. Normalized content IDs exclude split, seed, ordinal and regime
metadata. Normalization covers pair swaps, common atom/context weight scaling,
rational representations, and atom/context order and duplicate aggregation.
It is not a general isomorphism test. Reject any duplicate normalized content
across all splits. Every referee row binds its exact public row digest.
Referee files remain evaluator-side. The structural split introduces longer,
heterogeneous project schedules with scope changes and additional decision
actions in quantization; it is public development, not sealed confirmation.

All material is synthetic and reference-only. No provider output enters this
generator, no native DataGoblin training is authorized, and no provider call
is made here. Keep v1 and live teacher scores immutable.
