"""QO-13: finite-horizon probability of attaining a QO-07 certificate.

For an addressable Bernoulli hit/hold microstate/action table, optimize
the probability that QO-07 declares FEASIBLE or INFEASIBLE within h
additional probes. Every probability is a rational Beta(1,1) POSTERIOR
PREDICTIVE used only for planning. The confidence-sequence certificate
and its time-uniform frequentist coverage remain QO-07's separate law.

Bellman objective (success is absorbing):
  V_0(E) = 1{QO-07 already certified E}
  V_h(E) = 1 if certified else
       max_cell [ (1-p_cell) V_(h-1)(E[cell,0])
                    +p_cell V_(h-1)(E[cell,1]) ].

Hypothetical empty confidence intersections are marked as FAILED paths
with utility zero. They are never rewarded as certificates and remain
in the predictive probability mass; the model has full support.

Exact Fraction results and cache/branch counts are reported. Horizon
0..3 is supported for small public test systems. The practical policy
caps depth-three search at two addressed cells, and skips Bellman
search until enough evidence is present for a near-threshold decision.
This is a cost-gated heuristic, not a globally optimal controller.
"""
from fractions import Fraction as Q

from research.ingression_dynamics_v1.expected_gap_probe import gap_bounds
from research.ingression_dynamics_v1.two_step_gap_probe import (
    planning_state, _child
)
from research.ingression_dynamics_v1.anytime_active_quotient import choose_probe


def certification_status(bounds, max_classes, tolerance):
    """QO-07-compatible, mutually exclusive, three-valued decision."""
    tol=Q(tolerance)
    if not 0<=tol<=1:
        raise ValueError("TV tolerance must lie within [0,1]")
    upper,lower,gap=gap_bounds(bounds,max_classes)
    if upper<=tol:
        state="certified-feasible"
    elif lower>tol:
        state="certified-infeasible"
    else:
        state="undetermined"
    return state,upper,lower,gap


def _planning_prob(state, cell, prior_success, prior_failure):
    _,counts,hits=state
    i,a=cell
    return (prior_success+hits[i][a]) / (
        prior_success+prior_failure+counts[i][a])


def optimal_certificate_reachability(
        ledger, max_classes, tolerance, horizon=2,
        prior_success=Q(1),prior_failure=Q(1),
        max_cells=8):
    """Exact finite Beta-predictive Bellman value, with full recourse.

    At each observation, the next cell can depend on that outcome.
    The whole observed ledger is the state, including *nested* QO-07
    confidence intervals, so each branch is a valid simulation of
    the original e-process algorithm.

    Returns value, first-cell, conditional future actions and exact
    work counts. A source tree represents one optimal adaptive policy,
    not a claim about unconditional real-world success probability.
    """
    if type(horizon) is not int or not 0<=horizon<=3:
        raise ValueError("supported finite planning horizon is 0..3")
    if type(max_classes) is not int or max_classes<1:
        raise ValueError("positive transient class count required")
    if type(max_cells) is not int or max_cells<1 or ledger.n*ledger.d>max_cells:
        raise ValueError("addressed cell count exceeds declared exact solver cap")
    ps,pf=Q(prior_success),Q(prior_failure)
    if ps<=0 or pf<=0:
        raise ValueError("positive Beta planning pseudo-counts required")
    tol=Q(tolerance)
    if not 0<=tol<=1:
        raise ValueError("invalid one-step TV tolerance")

    start=planning_state(ledger)
    cells=tuple((i,a) for i in range(ledger.n) for a in range(ledger.d))
    cache={}
    certificate_cache={}
    root_action_values={}
    counters={
        "decision_gap_evaluations":0,
        "decision_gap_cache_hits":0,
        "bellman_states_evaluated":0,
        "bellman_state_cache_hits":0,
        "bellman_candidates":0,
        "hypothetical_outcome_branches":0,
        "hypothetical_invalid_branches":0,
        "terminal_certified_nodes":0,
    }

    def status(state):
        bounds=state[0]
        if bounds in certificate_cache:
            counters["decision_gap_cache_hits"]+=1
            return certificate_cache[bounds]
        result=certification_status(bounds,max_classes,tol)
        certificate_cache[bounds]=result
        counters["decision_gap_evaluations"]+=1
        return result

    def recurse(state,remaining):
        key=(state,remaining)
        if key in cache:
            counters["bellman_state_cache_hits"]+=1
            return cache[key]
        counters["bellman_states_evaluated"]+=1
        dec=status(state)[0]
        if dec!="undetermined":
            counters["terminal_certified_nodes"]+=1
            result=(Q(1),None)
            cache[key]=result
            return result
        if remaining==0:
            result=(Q(0),None)
            cache[key]=result
            return result

        best=None
        for cell in cells:
            counters["bellman_candidates"]+=1
            p=_planning_prob(state,cell,ps,pf)
            children=[]
            for bit in (0,1):
                counters["hypothetical_outcome_branches"]+=1
                successor=_child(state,cell,bit,ledger.alpha,ledger.bits)
                if successor is None:
                    counters["hypothetical_invalid_branches"]+=1
                    children.append((Q(0),None))
                else:
                    children.append(recurse(successor,remaining-1))
            prob=(1-p)*children[0][0]+p*children[1][0]
            if state==start and remaining==horizon:
                root_action_values[cell]=prob
            low,high=state[0][cell[0]][cell[1]]
            tie=(-prob,-(high-low),state[1][cell[0]][cell[1]],cell)
            if best is None or tie<best[0]:
                best=(tie,{
                    "cell":cell,
                    "hit_predictive":p,
                    "hold_probability":children[0][0],
                    "hit_probability":children[1][0],
                    "after_hold":children[0][1],
                    "after_hit":children[1][1],
                })
        if best is None:
            raise AssertionError("nonempty addressable cell menu required")
        result=(-best[0][0],best[1])
        if not 0<=result[0]<=1:
            raise AssertionError("reachability probability outside unit interval")
        cache[key]=result
        return result

    before=status(start)[0]
    chance,tree=recurse(start,horizon)
    return {
        "horizon":horizon,
        "current_decision":before,
        "root_action_values":root_action_values,
        "predicted_certificate_probability":chance,
        "first_cell":None if tree is None else tree["cell"],
        "after_hold_cell":None if tree is None or tree["after_hold"] is None
                          else tree["after_hold"]["cell"],
        "after_hit_cell":None if tree is None or tree["after_hit"] is None
                         else tree["after_hit"]["cell"],
        "first_hold_certificate_probability":None if tree is None
                         else tree["hold_probability"],
        "first_hit_certificate_probability":None if tree is None
                         else tree["hit_probability"],
        **counters,
    }


def select_certificate_probe(
        ledger,max_classes,tolerance,horizon=2,
        min_cell_samples=8,near_threshold=Q(1,20),
        fallback="bottleneck"):
    """Predictable cost-gated policy, using the same QO-07 certificate.

    The exact solver is used when every addressed cell has at least the
    predeclared number of observations and an optimized bound is near
    the decision threshold. Otherwise it uses the cheap baseline.
    A depth-three request is clipped to depth two for >2 cells.

    Return a fixed-schema planning receipt, enabling compute-aware
    comparison with ordinary bottleneck and QO-11/QO-12.
    """
    if fallback not in ("bottleneck","width"):
        raise ValueError("fallback must be bottleneck or width")
    if type(horizon) is not int or horizon not in (1,2,3):
        raise ValueError("pilot horizons are 1,2,3")
    if type(min_cell_samples) is not int or min_cell_samples<0:
        raise ValueError("nonnegative readiness sample count required")
    threshold=Q(near_threshold)
    if threshold<0:
        raise ValueError("nonnegative decision-boundary distance required")
    state,upper,lower,gap=certification_status(
        ledger.snapshot(),max_classes,tolerance)
    min_samples=min(v for row in ledger.counts for v in row)
    ready=min_samples>=min_cell_samples
    near=(min(abs(upper-Q(tolerance)),abs(lower-Q(tolerance)))
          <=threshold)
    effective=horizon if (horizon<=2 or ledger.n*ledger.d<=2) else 2
    reason=None
    if state!="undetermined":
        reason="already_certified"
    elif not ready:
        reason="insufficient_addressed_coverage"
    elif not near:
        reason="distant_from_threshold"
    if reason is None:
        plan=optimal_certificate_reachability(
            ledger,max_classes,tolerance,effective)
        if plan["predicted_certificate_probability"]>0 and (
                plan["first_cell"] is not None):
            cell=plan["first_cell"]
            mode="positive_certificate_probability"
        else:
            cell=choose_probe(ledger,max_classes,policy=fallback)
            mode="zero_reachability_fallback"
        metrics={key:plan[key] for key in (
            "decision_gap_evaluations","decision_gap_cache_hits",
            "bellman_states_evaluated","bellman_state_cache_hits",
            "bellman_candidates","hypothetical_outcome_branches",
            "hypothetical_invalid_branches","terminal_certified_nodes")}
        prob=plan["predicted_certificate_probability"]
    else:
        cell=choose_probe(ledger,max_classes,policy=fallback)
        mode="gated_"+reason
        metrics={key:0 for key in (
            "decision_gap_evaluations","decision_gap_cache_hits",
            "bellman_states_evaluated","bellman_state_cache_hits",
            "bellman_candidates","hypothetical_outcome_branches",
            "hypothetical_invalid_branches","terminal_certified_nodes")}
        prob=Q(1) if state!="undetermined" else Q(0)
    return cell,{
        "mode":mode,
        "requested_horizon":horizon,
        "effective_horizon":effective,
        "current_decision":state,
        "predicted_certificate_probability":str(prob),
        "current_certification_gap":str(gap),
        "near_decision_boundary":near,
        "min_cell_samples":min_samples,
        **metrics,
    }
