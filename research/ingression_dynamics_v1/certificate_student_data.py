"""QO-14: exact Bellman Q-targets and split-safe Bernoulli evidence examples.

Training labels come from the existing QO-13 finite certificate-reachability
objective with horizon=2. Each action's Q-value uses full conditional
recourse at the second probe. All probability arithmetic in teacher
targets is exact Fraction arithmetic. The student sees ONLY public QO-07
counts, hit totals, confidence intervals, class budget and tolerance.

Train, validation and test splits have disjoint declared underlying
rate families and independent Bernoulli stream seeds. Test outcomes
are never used during parameter training or model selection.
"""
from fractions import Fraction as Q
from itertools import product
from random import Random

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, certificate
)
from research.ingression_dynamics_v1.two_step_gap_probe import (
    planning_state, _child
)
from research.ingression_dynamics_v1.certificate_reachability import (
    certification_status
)

TOLERANCE=Q(1,5)
ALPHA=Q(1,20)
BISECTION_BITS=10
ACTION_CELLS=((0,0),(1,0))
TRAIN_RATES=(
    (Q(1,25),Q(24,25)),(Q(2,25),Q(23,25)),
    (Q(4,25),Q(21,25)),(Q(6,25),Q(19,25)),
    (Q(9,25),Q(16,25)),(Q(12,25),Q(13,25)),
    (Q(21,25),Q(4,25)),(Q(24,25),Q(1,25)),
)
VALIDATION_RATES=(
    (Q(3,25),Q(22,25)),(Q(8,25),Q(17,25)),
    (Q(22,25),Q(3,25))
)
TEST_RATES=(
    (Q(1,20),Q(19,20)),(Q(3,20),Q(17,20)),
    (Q(1,5),Q(4,5)),(Q(7,20),Q(13,20)),
    (Q(19,20),Q(1,20)),(Q(4,5),Q(1,5)),
)
TRAIN_SEEDS=(101,103,107)
VALIDATION_SEEDS=(401,409)
TEST_SEEDS=(701,709,719)
SCHEDULES=("balanced","skew_low","skew_high")
CHECKPOINTS=(20,24,28,32,36,40,44,48,52,56)

assert not(set(TRAIN_RATES)&set(VALIDATION_RATES))
assert not(set(TRAIN_RATES)&set(TEST_RATES))
assert not(set(VALIDATION_RATES)&set(TEST_RATES))
assert not(set(TRAIN_SEEDS)&set(VALIDATION_SEEDS))
assert not(set(TRAIN_SEEDS)&set(TEST_SEEDS))


def probe_choice(schedule, t):
    """Source-frozen, outcome-independent coverage patterns."""
    if schedule=="balanced": return (t%2,0)
    if schedule=="skew_low": return (0 if t%3!=2 else 1,0)
    if schedule=="skew_high":return (1 if t%3!=2 else 0,0)
    raise ValueError("invalid registered sampling schedule")


def make_ledger(n0,success0,n1,success1):
    """Reconstruct synthetic grouped evidence using source-order events.

    Used by independent tests. Outcome order affects nested QO-07
    confidence regions, so actual training uses full sampled receipts.
    """
    values=((n0,success0),(n1,success1))
    if any(type(n) is not int or type(s) is not int or
           not(0<=s<=n) for n,s in values):
        raise ValueError("invalid source history")
    ledger=ProbeLedger(2,1,ALPHA,BISECTION_BITS)
    for i,(n,s) in enumerate(values):
        for t in range(n):
            ledger.observe(i,0,int(t<s))
    return ledger


def evidence_features(ledger):
    """Two token vectors (6 floats each), with no access to true rates."""
    if (ledger.n,ledger.d)!=(2,1):
        raise ValueError("QO-14 first study is exactly two states, one action")
    tokens=[]
    for i in range(2):
        n=ledger.counts[i][0]
        h=ledger.hits[i][0]
        lo,hi=ledger.bounds[i][0]
        tokens.append((
            n/(n+24.0),
            (h+1)/(n+2.0),
            float(lo),
            float(hi),
            float(hi-lo),
            float((Q(lo+hi,2)-Q(1,2))),
        ))
    return tuple(tokens)


def _success(bounds, tolerance=TOLERANCE):
    return int(certification_status(bounds,1,tolerance)[0]!="undetermined")


def teacher_q_values(ledger,tolerance=TOLERANCE):
    """Exact horizon-two Q-values for BOTH first cells, independent of policy.

    For each prospective first bit, optimize the second probe after
    seeing that bit, and average using the first cell's Beta(1,1)
    planning predictive. Invalid nested intersections score zero.
    """
    if (ledger.n,ledger.d)!=(2,1):
        raise ValueError("two addressable one-action cells required")
    tol=Q(tolerance)
    if _success(ledger.snapshot(),tol):
        return (Q(1),Q(1))
    start=planning_state(ledger)
    values=[]
    for first in ACTION_CELLS:
        n=ledger.counts[first[0]][0]
        s=ledger.hits[first[0]][0]
        p=Q(s+1,n+2)
        first_branches=[]
        for outcome in (0,1):
            after=_child(start,first,outcome,ledger.alpha,ledger.bits)
            if after is None:
                first_branches.append(Q(0))
                continue
            if _success(after[0],tol):
                first_branches.append(Q(1))
                continue
            best=Q(0)
            for second in ACTION_CELLS:
                n2=after[1][second[0]][0]
                s2=after[2][second[0]][0]
                p2=Q(s2+1,n2+2)
                branches=[]
                for bit in (0,1):
                    final=_child(after,second,bit,ledger.alpha,ledger.bits)
                    branches.append(Q(0) if final is None
                                    else Q(_success(final[0],tol)))
                best=max(best,(1-p2)*branches[0]+p2*branches[1])
            first_branches.append(best)
        values.append((1-p)*first_branches[0]+p*first_branches[1])
    return tuple(values)


def training_example(ledger,split,rate_index,seed,schedule,t):
    q=teacher_q_values(ledger)
    return {
        "tokens":evidence_features(ledger),
        "q":q,
        "teacher_action":0 if q[0]>=q[1] else 1,
        "teacher_value":max(q),
        "split":split,
        "rate_index":rate_index,
        "seed":seed,
        "schedule":schedule,
        "time":t,
    }


def generate_split(split):
    """Deterministic simulated trajectories; true rates used ONLY for draws."""
    if split=="train":rates,seeds=TRAIN_RATES,TRAIN_SEEDS
    elif split=="validation":rates,seeds=VALIDATION_RATES,VALIDATION_SEEDS
    elif split=="test":rates,seeds=TEST_RATES,TEST_SEEDS
    else:raise ValueError("unknown split")
    results=[]
    failures=[]
    for idx,rate in enumerate(rates):
        for seed in seeds:
            for sched in SCHEDULES:
                ledger=ProbeLedger(2,1,ALPHA,BISECTION_BITS)
                rngs=[Random(1000003*seed+100019*idx+9791*i+
                             8599*SCHEDULES.index(sched)) for i in range(2)]
                for t in range(1,max(CHECKPOINTS)+1):
                    i,a=probe_choice(sched,t-1)
                    outcome=int(rngs[i].random()<float(rate[i]))
                    try:
                        ledger.observe(i,a,outcome)
                    except ValueError:
                        failures.append((split,idx,seed,sched,t))
                        break
                    if t in CHECKPOINTS:
                        results.append(training_example(
                            ledger,split,idx,seed,sched,t))
    return tuple(results),tuple(failures)


def split_signature(example):
    """Every model input is a function of observation history only."""
    return tuple(round(x,12) for token in example["tokens"] for x in token)
