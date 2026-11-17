"""NOR-30 independently implemented schedule and greedy attack replay.

Does not import NOR30 experiment, optimizer, or policy scoring.
Reuses only the previously merged NOR25 public auditor implementation.
"""
from __future__ import annotations
from hashlib import sha256
import random

from research.nor25_robust_audits import experiment as old25

ARMS=("uniform","static_prior","posterior","shuffled_prior",
      "hybrid_25","hybrid_50","hybrid_75")
ATTACKERS=ARMS+("mixed_best_response",)
CAL=256
EVAL=512
R=4
NREPS=8
QUANTA=16


def keyed(*args):
    return int.from_bytes(
        sha256("|".join(str(x) for x in args).encode("ascii")).digest(),
        "big")


def generate_schedules(regions,digest,stage):
    stream={"calibration":202611300091,"evaluation":202611300093}[stage]
    phase=sha256(f"{stream}|{stage}|{digest}".encode("ascii")).hexdigest()
    n=CAL if stage=="calibration" else EVAL
    schedules={a:[] for a in ARMS}
    for trial in range(n):
        original=old25.policy_schedules(regions,phase,trial)
        for arm in ARMS:
            checked=tuple(original[(arm,16)])
            assert len(checked)==len(set(checked))==16
            schedules[arm].append(checked)
    return schedules


def attack_greedy(plans,weights,digest,name,restart):
    """Alternative set-union objective, separate from production bitsets."""
    assert name in ATTACKERS
    m=max((max(v) for plan in plans.values() for checked in plan
           for v in [checked]), default=-1)+1
    # The true coefficient universe may exceed the largest observed
    # schedule index. Use the public support universe supplied as m.
    raise AssertionError("call attack_greedy_known_m, not inferred length")


def attack_greedy_known_m(plans,m,weights,digest,name,restart):
    assert name in ATTACKERS and 0<=restart<NREPS
    cover={a:[set() for _ in range(m)] for a in ARMS}
    for arm in ARMS:
        for trial,checked in enumerate(plans[arm]):
            for i in checked:
                cover[arm][i].add(trial)
    rng=random.Random(keyed(202611300092,digest,name,restart))
    order=list(range(m))
    rng.shuffle(order)
    selected=[]
    exposed={a:set() for a in ARMS}
    for _ in range(R):
        best=None
        best_value=None
        for i in order:
            if i in selected:
                continue
            if name=="mixed_best_response":
                value=sum(weights[j]*len(exposed[a]|cover[a][i])
                          for j,a in enumerate(ARMS))
            else:
                value=len(exposed[name]|cover[name][i])
            if best_value is None or value<best_value:
                best=i
                best_value=value
        assert best is not None
        selected.append(best)
        for arm in ARMS:
            if name=="mixed_best_response" or arm==name:
                exposed[arm].update(cover[arm][best])
    assert len(set(selected))==R
    return tuple(sorted(selected))


def independent_blind(digest,m,restart):
    r=random.Random(keyed(202611300092,digest,"fresh",restart))
    return tuple(sorted(r.sample(range(m),R)))


def template(digest,kind,support,restart):
    r=random.Random(keyed(202611300095,digest,kind,restart))
    return tuple(sorted(r.sample(sorted(support[kind]),R)))


def chosen_mixed_arm(weights,digest,trial):
    pick=random.Random(keyed(202611300094,digest,trial)).randrange(QUANTA)
    cumulative=0
    for i,weight in enumerate(weights):
        cumulative+=weight
        if pick<cumulative:return ARMS[i]
    raise ValueError("bad mixed defense")


def canonical(data):
    import json
    return (json.dumps(data,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()
