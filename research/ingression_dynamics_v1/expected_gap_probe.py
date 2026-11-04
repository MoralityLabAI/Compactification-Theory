"""QO-11: expected one-step certification-gap acquisition for QO-07.

State-action choices use only past public Bernoulli observations, fixed
Beta(1,1) *planning* predictive probabilities, and hypothetical QO-07
confidence interval updates. Every realized QO-07 confidence rectangle
is nested, so its robust/optimistic certification gap is nonincreasing.

The beta posterior predictive is a HEURISTIC acquisition model, never a
frequentist confidence guarantee. QO-07's separate e-process authorizes
all subsequent decisions for any predictable query schedule.

The one-step objective can be identically zero on a wide plateau;
the policy has explicit baseline fallbacks and counts those events.
"""
from fractions import Fraction as Q

from research.ingression_dynamics_v1.anytime_active_quotient import (
    anytime_outer_interval, robust_optimum, choose_probe
)


def gap_bounds(bounds, max_classes):
    """Compute R*, L*, G* for the current rectangle, exactly.

    The class-cap=1 case has a closed form, avoiding Bell enumeration.
    The general finite case calls the original exact QO-07 optimizer.
    """
    bounds = tuple(tuple((Q(lo),Q(hi)) for lo,hi in row) for row in bounds)
    if not bounds or not bounds[0] or any(len(row)!=len(bounds[0])
            or any(not 0<=lo<=hi<=1 for lo,hi in row)
            for row in bounds):
        raise ValueError("nonempty rectangular probability intervals required")
    if type(max_classes) is not int or max_classes < 1:
        raise ValueError("positive transient class budget required")
    if max_classes == 1:
        n_actions = len(bounds[0])
        robust = max((max(bounds[i][a][1] for i in range(len(bounds)))
                       - min(bounds[i][a][0] for i in range(len(bounds))))/2
                     for a in range(n_actions))
        optimistic = max(max(Q(0),
                             max(bounds[i][a][0] for i in range(len(bounds)))
                             - min(bounds[i][a][1] for i in range(len(bounds))))/2
                         for a in range(n_actions))
    else:
        robust, optimistic, _, _ = robust_optimum(bounds, max_classes)
    if not Q(0) <= optimistic <= robust <= Q(1,2):
        raise AssertionError("invalid QO-07 upper/lower extrema")
    return robust, optimistic, robust - optimistic


def future_bounds(ledger, state, action, outcome):
    """QO-07's hypothetical next interval, with exact outward rounding.

    Returns None if a rare hypothetical observation makes the current
    confidence set empty. The planner will fall back rather than
    treat an empty statistical set as a valid gain.
    """
    if (type(state) is not int or not 0<=state<ledger.n
            or type(action) is not int or not 0<=action<ledger.d
            or type(outcome) is not int or outcome not in (0,1)):
        raise ValueError("invalid prospective probe or Bernoulli outcome")
    n = ledger.counts[state][action]+1
    successes = ledger.hits[state][action]+outcome
    next_lo, next_hi = anytime_outer_interval(
        n, successes, ledger.alpha, ledger.n*ledger.d, ledger.bits)
    old_lo, old_hi = ledger.bounds[state][action]
    narrowed = (max(old_lo, next_lo), min(old_hi, next_hi))
    if narrowed[0] > narrowed[1]:
        return None
    updated = [[entry for entry in row] for row in ledger.bounds]
    updated[state][action] = narrowed
    return tuple(tuple(row) for row in updated)


def beta_planning_probability(ledger, state, action,
                              prior_success=Q(1),
                              prior_failure=Q(1)):
    """Beta predictive is used for planning only, not coverage claims."""
    a, b = Q(prior_success), Q(prior_failure)
    if a<=0 or b<=0:
        raise ValueError("positive Beta pseudo-counts required")
    return ((a + ledger.hits[state][action]) /
            (a + b + ledger.counts[state][action]))


def exact_next_gap_scores(ledger, max_classes,
                          prior_success=Q(1),prior_failure=Q(1)):
    """All 2nd-branch hypothetical QO-07 gaps, in exact rational arithmetic.

    Each cell's score is current G minus the Beta predictive
    expected next gap. The posterior predictive plays no part in the
    independent QO-07 e-process coverage guarantee.

    Output includes the number of completed hypothetical optimizer
    calls for honest planning-cost accounting.
    """
    current = gap_bounds(ledger.snapshot(),max_classes)[2]
    scores={}
    hypothetical_evals=0
    for state in range(ledger.n):
        for action in range(ledger.d):
            p=beta_planning_probability(
                ledger,state,action,prior_success,prior_failure)
            by_outcome=[]
            for outcome in (0,1):
                hypothetical=future_bounds(ledger,state,action,outcome)
                if hypothetical is None:
                    by_outcome.append(None)
                else:
                    by_outcome.append(gap_bounds(hypothetical,max_classes)[2])
                    hypothetical_evals += 1
            if any(x is None for x in by_outcome):
                scores[(state,action)] = None
                continue
            expected = (1-p)*by_outcome[0] + p*by_outcome[1]
            gain = current-expected
            if gain < 0:
                raise AssertionError("nested intervals caused gap to grow")
            scores[(state,action)] = {
                "beta_predictive_hit":p,
                "current_gap":current,
                "hit_gap":by_outcome[1],
                "miss_gap":by_outcome[0],
                "expected_next_gap":expected,
                "expected_reduction":gain,
            }
    return scores,hypothetical_evals


def choose_expected_gap_probe(ledger, max_classes,
                              fallback="bottleneck",
                              prior_success=Q(1),prior_failure=Q(1)):
    """Select a predictable probe and return an auditable planning receipt.

    If all one-step acquisitions are zero, the policy uses an unchanged
    QO-07 bottleneck/width baseline. This addresses the multi-cell
    plateau of myopic information acquisition. A rare empty
    hypothetical intersection is never rewarded as informative.
    """
    if fallback not in ("bottleneck","width"):
        raise ValueError("fallback must be bottleneck or width")
    scores, evals = exact_next_gap_scores(
        ledger,max_classes,prior_success,prior_failure)
    valid=[(key,value["expected_reduction"]) for key,value
           in scores.items() if value is not None]
    best=max((gain for key,gain in valid),default=Q(0))
    if best>0:
        # Highest score; equal scores break by interval width, then cell ID.
        def ordering(entry):
            (i,a),score=entry
            lo,hi=ledger.bounds[i][a]
            return (-score,-(hi-lo),ledger.counts[i][a],i,a)
        chosen=min((x for x in valid if x[1]==best),key=ordering)[0]
        mode="positive_acquisition"
    else:
        chosen=choose_probe(ledger,max_classes,
                            policy=("bottleneck" if fallback=="bottleneck" else "width"))
        mode="fallback_"+fallback
    return chosen,{
        "mode":mode,
        "candidate_cells":ledger.n*ledger.d,
        "hypothetical_optimizer_calls":evals,
        "unavailable_branches":sum(x is None for x in scores.values()),
        "best_expected_gap_reduction":str(best),
        "observed_gap":str(gap_bounds(ledger.snapshot(),max_classes)[2]),
    }


def plateau_witness():
    """At least two wholly unobserved cells in one action hide early progress.

    With one class, two cells initially [0,1], probing only one cell
    leaves the other full. Both R and L retain 1/2 and 0 respectively.
    This is a minimal exact one-step lookahead plateau.
    """
    from research.ingression_dynamics_v1.anytime_active_quotient import ProbeLedger
    ledger=ProbeLedger(2,1)
    scores, calls=exact_next_gap_scores(ledger,1)
    assert calls==4
    assert all(s["expected_reduction"]==0 for s in scores.values())
    return {
        "state_action_cells":2,
        "starting_gap":str(gap_bounds(ledger.snapshot(),1)[2]),
        "all_one_step_gains":tuple(str(v["expected_reduction"])
                                   for v in scores.values()),
    }
