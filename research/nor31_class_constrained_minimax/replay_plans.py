"""NOR-31 independent trial-ID attack/schedule replay.

Separate from the applied NOR31 bitset/categorical evaluator. Uses only
the original merged NOR25 schedule generator and pinned public seed recipe.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import random

from research.nor25_robust_audits import experiment as original25

ARMS=("uniform","static_prior","posterior",
      "hybrid_25","hybrid_50","hybrid_75")
LEGACY_ATTACKS=("uniform","static_prior","posterior","shuffled_prior",
                "hybrid_25","hybrid_50","hybrid_75")
ATTACKERS=LEGACY_ATTACKS+("mixed_best_response",)
KINDS=("tail","magnitude","lag","uniform","off_template")
K=16
R=4
N_RESTARTS=8
N_CAL=256
N_EVAL=512


def seed(*fields):
    return int.from_bytes(
        sha256("|".join(map(str,fields)).encode("ascii")).digest(),"big")


def canonical(x):
    import json
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()


def plans(regions,public_digest,phase):
    phase_num={"calibration":202611310091,"evaluation":202611310093}[phase]
    tag=sha256(f"{phase_num}|{phase}|{public_digest}".encode("ascii")).hexdigest()
    n=N_CAL if phase=="calibration" else N_EVAL
    out={a:[] for a in LEGACY_ATTACKS}
    for trial in range(n):
        p=original25.policy_schedules(regions,tag,trial)
        for a in LEGACY_ATTACKS:
            coords=tuple(p[(a,K)])
            assert len(set(coords))==K
            out[a].append(coords)
    return out


def trial_cover(plans,m):
    groups={a:[set() for _ in range(m)] for a in LEGACY_ATTACKS}
    for a in LEGACY_ATTACKS:
        for ordinal,checked in enumerate(plans[a]):
            for i in checked:groups[a][i].add(ordinal)
    return groups


def approximate_attack(cover,weights,source_digest,class_name,rep):
    """Greedy union of trial-index SETS, independent of bitset evaluator."""
    assert class_name in ATTACKERS
    m=len(cover["uniform"])
    rng=random.Random(seed(202611310092,source_digest,class_name,rep))
    order=list(range(m))
    rng.shuffle(order)
    used=set()
    unions={arm:set() for arm in LEGACY_ATTACKS}
    for _ in range(R):
        best=None
        best_score=None
        for i in order:
            if i in used:continue
            if class_name=="mixed_best_response":
                score=sum(weights[j]*len(unions[a]|cover[a][i])
                          for j,a in enumerate(ARMS))
            else:
                score=len(unions[class_name]|cover[class_name][i])
            if best_score is None or score<best_score:
                best=i;best_score=score
        assert best is not None
        used.add(best)
        for a in LEGACY_ATTACKS:
            if class_name=="mixed_best_response" or a==class_name:
                unions[a].update(cover[a][best])
    return tuple(sorted(used))


def fresh_attack(digest,m,rep):
    r=random.Random(seed(202611310092,digest,"fresh",rep))
    return tuple(sorted(r.sample(range(m),R)))


def support_attack(digest,kind,regions,rep):
    r=random.Random(seed(202611310095,digest,kind,rep))
    return tuple(sorted(r.sample(sorted(regions[kind]),R)))


def realize_mixed(weights,digest,trial):
    n=random.Random(seed(202611310094,digest,trial)).randrange(16)
    cumul=0
    for a,w in zip(ARMS,weights):
        cumul+=w
        if n<cumul:return a
    raise AssertionError("bad quantization")
