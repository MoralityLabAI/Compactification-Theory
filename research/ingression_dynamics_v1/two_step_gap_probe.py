"""QO-12: exact two-step expected certification-gap acquisition.

Finite-horizon Bellman lookahead over the two Bernoulli outcomes of each
addressable state/action cell. Every counterfactual updates the EXACT
QO-07 outer-rational anytime confidence interval, with a Beta(1,1)
posterior predictive used ONLY as an acquisition heuristic.

The practical 'on_plateau' controller requests the more costly two-step
lookahead only when QO-11's best one-step acquisition is zero. It
records every hypothetical gap evaluation, cache hit and fallback.

Theorem: with a common, nonempty nested confidence contract and a
normalized predictive, V_2 <= V_1 <= G. A no-lookahead plateau may
therefore have strictly positive two-step value.
"""
from fractions import Fraction as Q

from research.ingression_dynamics_v1.anytime_active_quotient import (
    anytime_outer_interval, choose_probe
)
from research.ingression_dynamics_v1.expected_gap_probe import (
    gap_bounds, exact_next_gap_scores, choose_expected_gap_probe
)


def planning_state(ledger):
    """Immutable exact evidence: (bounds, counts, hits); no private rates."""
    bounds=ledger.snapshot()
    counts=tuple(tuple(row) for row in ledger.counts)
    hits=tuple(tuple(row) for row in ledger.hits)
    return bounds, counts, hits


def _validate_plan(ledger, classes, prior_success, prior_failure):
    if type(classes) is not int or classes < 1:
        raise ValueError("positive macro class count required")
    a,b=Q(prior_success),Q(prior_failure)
    if a<=0 or b<=0:
        raise ValueError("positive Beta planning pseudo-counts required")
    if ledger.n*ledger.d > 12:
        raise ValueError("two-step exact research oracle limited to twelve cells")
    return a,b


def _predict(state, cell, a, b):
    _,counts,hits=state
    i,j=cell
    return (a+hits[i][j])/(a+b+counts[i][j])


def _child(state, cell, outcome, alpha, bits):
    """Pure hypothetical ledger update, identical to QO-07 observe()."""
    bounds,counts,hits=state
    i,j=cell
    n=counts[i][j]+1
    s=hits[i][j]+outcome
    lower,upper=anytime_outer_interval(
        n,s,alpha,len(bounds)*len(bounds[0]),bits)
    old_lower,old_upper=bounds[i][j]
    lower=max(old_lower,lower)
    upper=min(old_upper,upper)
    if lower>upper:
        return None
    updated_bounds=list(bounds)
    row=list(updated_bounds[i])
    row[j]=(lower,upper)
    updated_bounds[i]=tuple(row)
    updated_counts=list(counts)
    count_row=list(updated_counts[i])
    count_row[j]=n
    updated_counts[i]=tuple(count_row)
    updated_hits=list(hits)
    hit_row=list(updated_hits[i])
    hit_row[j]=s
    updated_hits[i]=tuple(hit_row)
    return tuple(updated_bounds),tuple(updated_counts),tuple(updated_hits)


def optimal_horizon_value(ledger, classes, horizon=2,
                          prior_success=Q(1),prior_failure=Q(1)):
    """Return the minimal predicted gap and a policy tree under exact recourse.

    Search states/observations causally: after the first outcome, the
    optimal second cell can depend on that outcome. Hypothetical invalid
    confidence intersections are excluded from candidate policies.
    The output includes unique gap-optimizer calls and Bellman cache hits.
    """
    a,b=_validate_plan(ledger,classes,prior_success,prior_failure)
    if type(horizon) is not int or horizon not in (0,1,2):
        raise ValueError("implemented research horizons are 0,1,2")
    state=planning_state(ledger)
    cells=tuple((i,j) for i in range(ledger.n) for j in range(ledger.d))
    gap_cache={}
    value_cache={}
    counter={
        "leaf_gap_optimizer_calls":0,
        "leaf_cache_hits":0,
        "bellman_cache_hits":0,
        "hypothetical_outcome_branches":0,
        "invalid_intersections":0,
        "bellman_candidates":0,
    }

    def gap(s):
        bounds=s[0]
        if bounds in gap_cache:
            counter["leaf_cache_hits"]+=1
            return gap_cache[bounds]
        out=gap_bounds(bounds,classes)[2]
        gap_cache[bounds]=out
        counter["leaf_gap_optimizer_calls"]+=1
        return out

    def value(s, steps):
        key=(s,steps)
        if key in value_cache:
            counter["bellman_cache_hits"]+=1
            return value_cache[key]
        if steps==0:
            result=(gap(s),None)
            value_cache[key]=result
            return result
        best=None
        for cell in cells:
            counter["bellman_candidates"]+=1
            hit_prob=_predict(s,cell,a,b)
            next_values=[]
            for outcome in (0,1):
                counter["hypothetical_outcome_branches"]+=1
                child=_child(s,cell,outcome,ledger.alpha,ledger.bits)
                if child is None:
                    counter["invalid_intersections"]+=1
                    next_values.append(None)
                else:
                    next_values.append(value(child,steps-1))
            if any(x is None for x in next_values):
                continue
            expected=(1-hit_prob)*next_values[0][0]+hit_prob*next_values[1][0]
            # Deterministic tie: score, current interval width (descending),
            # observed count (ascending), cell ID.
            lo,hi=s[0][cell[0]][cell[1]]
            tie=(expected,-(hi-lo),s[1][cell[0]][cell[1]],cell)
            if best is None or tie<best[0]:
                best=(tie,{
                    "first_cell":cell,
                    "first_hit_prob":hit_prob,
                    "expected_remaining_gap":expected,
                    "second_if_hold":next_values[0][1],
                    "second_if_hit":next_values[1][1],
                })
        if best is None:
            value_cache[key]=None
            return None
        result=(best[0][0],best[1])
        value_cache[key]=result
        return result

    present=gap(state)
    plan=value(state,horizon)
    if plan is None:
        return {
            "horizon":horizon,"present_gap":present,
            "expected_remaining_gap":None,
            "expected_reduction":None,
            "first_cell":None,"second_if_hold":None,"second_if_hit":None,
            **counter,
        }
    val,tree=plan
    if val>present:
        raise AssertionError("monotone nested evidence had negative Bellman gain")
    return {
        "horizon":horizon,
        "present_gap":present,
        "expected_remaining_gap":val,
        "expected_reduction":present-val,
        "first_cell":None if tree is None else tree["first_cell"],
        "second_if_hold":None if tree is None else tree["second_if_hold"],
        "second_if_hit":None if tree is None else tree["second_if_hit"],
        **counter,
    }


def choose_two_step_on_plateau(ledger,classes,
                               fallback="bottleneck",
                               prior_success=Q(1),prior_failure=Q(1)):
    """Practical QO-12 controller: QO-11 when positive, two-step on plateau.

    Return selected cell with a complete deterministic planning receipt.
    The two-step selector only acts after verifying QO-11 has NO positive
    one-step expected reduction. Thus the extra cost is charged to
    states where lookahead has a reason to matter.
    """
    if fallback not in ("bottleneck","width"):
        raise ValueError("fallback must be bottleneck or width")
    _validate_plan(ledger,classes,prior_success,prior_failure)
    scores,one_calls=exact_next_gap_scores(
        ledger,classes,prior_success,prior_failure)
    valid=[(c,v["expected_reduction"]) for c,v in scores.items()
           if v is not None]
    largest=max((v for c,v in valid),default=Q(0))
    if largest>0:
        def tie(item):
            (i,j),gain=item
            low,high=ledger.bounds[i][j]
            return (-gain,-(high-low),ledger.counts[i][j],i,j)
        choice=min((entry for entry in valid if entry[1]==largest),
                   key=tie)[0]
        return choice,{
            "mode":"one_step_positive",
            "one_step_gap_calls":one_calls,
            "two_step_gap_calls":0,
            "two_step_bellman_candidates":0,
            "two_step_branches":0,
            "two_step_invalid_branches":0,
            "certification_gap_before":str(gap_bounds(ledger.snapshot(),classes)[2]),
            "two_step_expected_gain":"0",
        }
    deep=optimal_horizon_value(
        ledger,classes,2,prior_success,prior_failure)
    if deep["first_cell"] is not None and deep["expected_reduction"]>0:
        choice=deep["first_cell"]
        mode="two_step_positive"
    else:
        choice=choose_probe(ledger,classes,policy=fallback)
        mode="fallback_"+fallback
    return choice,{
        "mode":mode,
        "one_step_gap_calls":one_calls,
        "two_step_gap_calls":deep["leaf_gap_optimizer_calls"],
        "two_step_bellman_candidates":deep["bellman_candidates"],
        "two_step_branches":deep["hypothetical_outcome_branches"],
        "two_step_invalid_branches":deep["invalid_intersections"],
        "certification_gap_before":str(deep["present_gap"]),
        "two_step_expected_gain":str(
            deep["expected_reduction"] if deep["expected_reduction"] is not None
            else Q(0)),
    }


def two_cell_exact_plateau():
    """Exact unrounded analytic two-probe witness, alpha=1/20 and 2 cells.

    A first hit/hold gives a one-trial likelihood lower/upper endpoint
    1/80 or 79/80. If both queried cells yield equal bits, the final
    gap drops from 1/2 to 79/160. Otherwise it stays 1/2. With a
    Beta(1,1) planning rate 1/2 for each cell, equal bits occur with
    probability 1/2, so expected two-step improvement is exactly 1/320.
    The executable QO-07 interval bisection rounds these endpoints
    outwards, producing a slightly smaller positive certificate.
    """
    initial=Q(1,2)
    same=Q(79,160)
    different=Q(1,2)
    expectation=(same+different)/2
    assert initial-expectation==Q(1,320)
    return {
        "unrounded_initial_gap":str(initial),
        "unrounded_same_bit_gap":str(same),
        "unrounded_mixed_bit_gap":str(different),
        "unrounded_expected_two_step_reduction":str(initial-expectation),
        "one_step_reduction":"0",
    }
