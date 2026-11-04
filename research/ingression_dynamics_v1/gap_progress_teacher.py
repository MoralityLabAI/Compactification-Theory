"""QO-16: exact QO-12 per-first-action two-probe expected-gap progress.

Every intermediate QO-07 confidence update is an exact rational
hypothetical, and each second action can depend on the first outcome.
Invalid hypothetical confidence-set intersections disqualify a candidate
action. The teacher is solely a predictable planning heuristic; only
the unchanged QO-07 e-process can issue valid certificates.
"""
from fractions import Fraction as F

from research.ingression_dynamics_v1.two_step_gap_probe import (
    planning_state, _child, optimal_horizon_value,
)
from research.ingression_dynamics_v1.expected_gap_probe import gap_bounds
from research.ingression_dynamics_v1.certificate_student_data import (
    ACTION_CELLS, TOLERANCE,
)
from research.ingression_dynamics_v1.certificate_reachability import (
    certification_status,
)

GAIN_SCALE = F(1,64)
NORMALIZED_GAIN_GATE = F(1,20)
READINESS_MIN_PER_CELL = 8


def _predict(state, cell):
    """Beta(1,1) posterior predictive, used only for planning."""
    _, counts, hits = state
    i, a = cell
    return F(hits[i][a]+1, counts[i][a]+2)


def gain_to_target(gain):
    """Strictly increasing bounded label transform preserves action order."""
    gain = F(gain)
    if gain < 0 or gain > F(1,2):
        raise ValueError("expected gap reduction outside [0,1/2]")
    return gain / (gain + GAIN_SCALE) if gain else F(0)


def target_to_gain(score):
    """A diagnostic inverse, never needed for greedy action selection."""
    s=F(score)
    if s<0 or s>=1:
        raise ValueError("normalized target must lie in [0,1)")
    return GAIN_SCALE*s/(1-s)


def gap_action_values(ledger):
    """Exact progress for each first probe with conditional second-step recourse.

    A first candidate is valid only if each first outcome has at least
    one second candidate with both valid hypothetical outcomes. This
    reproduces the QO-12 solver's invalid-branch exclusion rule.
    An unavailable action is assigned zero progress and disclosed.
    """
    if (ledger.n,ledger.d)!=(2,1):
        raise ValueError("QO-16 first prototype requires two addressable cells")
    start=planning_state(ledger)
    current=gap_bounds(start[0],1)[2]
    decision=certification_status(start[0],1,TOLERANCE)[0]
    if decision!="undetermined":
        return (F(0),F(0)), {
            "current_gap":str(current),"gap_evaluations":1,
            "candidate_action_evaluations":0,
            "invalid_branches":0,"unavailable_first_actions":2,
            "already_certified":True,
        }
    leaves={}
    count=1
    invalid=0
    candidates=0
    first_values=[]
    for first in ACTION_CELLS:
        p1=_predict(start,first)
        first_children=[]
        viable=True
        for y in (0,1):
            s1=_child(start,first,y,ledger.alpha,ledger.bits)
            if s1 is None:
                invalid+=1
                viable=False
                break
            best=None
            for second in ACTION_CELLS:
                candidates+=1
                p2=_predict(s1,second)
                next_gaps=[]
                for z in (0,1):
                    s2=_child(s1,second,z,ledger.alpha,ledger.bits)
                    if s2 is None:
                        invalid+=1
                        next_gaps.append(None)
                    else:
                        bound=s2[0]
                        if bound not in leaves:
                            leaves[bound]=gap_bounds(bound,1)[2]
                            count+=1
                        next_gaps.append(leaves[bound])
                if all(g is not None for g in next_gaps):
                    v=(1-p2)*next_gaps[0]+p2*next_gaps[1]
                    if best is None or v<best:
                        best=v
            if best is None:
                viable=False
                break
            first_children.append(best)
        if viable and len(first_children)==2:
            expected=(1-p1)*first_children[0]+p1*first_children[1]
            gain=current-expected
            if gain<0 or gain>current:
                raise AssertionError("nested QO-07 evidence increased gap")
            first_values.append(gain)
        else:
            first_values.append(F(0))
    return tuple(first_values),{
        "current_gap":str(current),
        "gap_evaluations":count,
        "candidate_action_evaluations":candidates,
        "invalid_branches":invalid,
        "unavailable_first_actions":None,
        "already_certified":False,
    }


def assert_qo12_reference_parity(ledger):
    """Independent existing QO-12 optimizer must equal max(first-action gain)."""
    q, stats=gap_action_values(ledger)
    ref=optimal_horizon_value(ledger,1,horizon=2)
    expected=ref["expected_reduction"]
    if expected is None:
        # All invalid branches: exact solver has no valid first probe.
        if max(q)!=0:
            raise AssertionError("unavailable QO-12 teacher had positive gain")
    elif max(q)!=expected:
        raise AssertionError(
            "QO-16 first-action values differ from QO-12 exact optimizer: "+
            repr((tuple(map(str,q)),str(expected))))
    return q, stats, ref


def greedy_gain_cell(ledger, scores):
    """QO-12 tie-breaking on public width/count; no hidden rate access."""
    if len(scores)!=2:
        raise ValueError("exactly two action scores required")
    return min((0,1),key=lambda i:(
        -scores[i],-(ledger.bounds[i][0][1]-ledger.bounds[i][0][0]),
        ledger.counts[i][0],i))
