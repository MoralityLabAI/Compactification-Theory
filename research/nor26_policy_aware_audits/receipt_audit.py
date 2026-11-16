"""NOR-26 independent schedule-blind attack-game evidence replay.

This replayer does not import research.nor26_policy_aware_audits.experiment.
It rebuilds all source-only proposals, RRC25 native acceptance, private
targets, calibration exposure lists, greedy attack sets and heldout 7x7
cross-play detections from the documented frozen streams.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
from math import comb, isclose
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.nor24_adaptive_audits import experiment as prev
from research.nor25_robust_audits import experiment as robust
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

ARMS=("uniform","static_prior","posterior","shuffled_prior",
      "hybrid_25","hybrid_50","hybrid_75")
REP=(6,7)
D=(512,1024)
CAL=256
EVAL=512
N_ATTACKS=8
R=4


def seed(*fields):
    return int.from_bytes(sha256("|".join(str(x) for x in fields).encode("ascii")).digest(),"big")


def source_seed(rep,d,fam):
    return (corpus.SEED+109326000+1000003*rep+
            100009*D.index(d)+10007*nor21.FAMILIES.index(fam))


def restore(row,public_proposal):
    d=row["observation_degree"]
    rng=random.Random(source_seed(row["replicate"],d,row["family"]))
    for _ in range(row["selected_attempt"]+1):
        p=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,row["family"],3*d//2),
                     corpus.execute(x,p)) for _ in range(2))
        query=nor21.public_source(rng,row["family"],3*d//2)
    assert row["source_seed"]==source_seed(row["replicate"],d,row["family"])
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    t=corpus.execute(query,p)
    assert t==public_proposal
    return t


def plans(regions,digest,phase):
    stream=202611260074 if phase=="calibration" else 202611260077
    key=sha256(f"{stream}|{phase}|{digest}".encode("ascii")).hexdigest()
    n=CAL if phase=="calibration" else EVAL
    p={arm:[] for arm in ARMS}
    for trial in range(n):
        original=robust.policy_schedules(regions,key,trial)
        for arm in ARMS:
            p[arm].append(tuple(original[(arm,16)]))
    return p


def attack_greedy(plans,digest,arm,rep,m):
    # Independent implementation: trial-index sets, not integer masks.
    coverage=[set() for _ in range(m)]
    for trial,checked in enumerate(plans):
        for i in checked:
            coverage[i].add(trial)
    indices=list(range(m))
    random.Random(seed(202611260075,digest,arm,rep)).shuffle(indices)
    chosen=set()
    exposed=set()
    for _ in range(R):
        j=min((i for i in indices if i not in chosen),
              key=lambda i:len(exposed|coverage[i]))
        chosen.add(j)
        exposed|=coverage[j]
    return tuple(sorted(chosen)),len(exposed)


def replay(src_path,out_path):
    srcraw=Path(src_path).read_bytes()
    outraw=Path(out_path).read_bytes()
    src=json.loads(srcraw)
    out=json.loads(outraw)
    assert src["schema"]=="nor26.fresh_public_only_12_sources.v1"
    assert src["source_only"] and src["hidden_truth_absent"] and src["attack_sets_absent"]
    assert src["replicates"]==[6,7]
    assert len(src["cases"])==12
    assert out["schema"]=="nor26.policy_aware_blind_attack_crossplay.v1"
    assert out["fixture_sha256"]==sha256(srcraw).hexdigest()
    assert out["source_count"]==12
    assert out["calibration_schedule_count"]==CAL
    assert out["evaluation_schedule_count"]==EVAL
    assert out["attack_replica_count"]==N_ATTACKS
    assert out["audit_budget"]==16 and out["attack_radius"]==R
    assert out["frozen_prior_numerators"]==[295,246,153,78]
    assert out["frozen_prior_denominator"]==772
    assert out["policy_aware_attacks_precede_evaluation_schedules"]
    assert out["private_truth_only_after_public_proposals"]
    assert {(row["replicate"],row["observation_degree"],row["family"])
            for row in src["cases"]}=={
            (rep,d,f) for rep in REP for d in D for f in nor21.FAMILIES}
    outputs={r["public_sha256"]:r for r in out["source_records"]}
    assert len(outputs)==12
    all_primary=[]
    actions=0
    for row in src["cases"]:
        digest=row["full_public_input_sha256"]
        a=outputs[digest]
        c,h=prev.public_only_case(row)
        mask=nor21.native_mask(c,h,row["observation_degree"])
        assert mask==row["original_public_accept_mask"]
        proposal=prev.public_proposal(row)
        assert proposal["mask"]==a["public_mask"]==mask
        truth=restore(row,proposal["predicted"])
        assert a["public_prediction_sha256"]==proposal["sha256"]
        assert a["sealed_truth_postchoice_sha256"]==prev.sha_target(truth)
        regions=prev.supports(proposal["predicted"],h)
        assert a["support_sizes"]=={k:len(regions[k]) for k in prev.ALL_KINDS}
        m=len(proposal["predicted"])
        assert a["m"]==m
        calibration=plans(regions,digest,"calibration")
        attacks={}
        for arm in ARMS:
            attempted=[attack_greedy(calibration[arm],digest,arm,rep,m)
                       for rep in range(N_ATTACKS)]
            attacks[arm]=[set(p) for p,_ in attempted]
            assert a["policy_aware_attacks"][arm]==[list(p) for p,_ in attempted]
            assert a["attack_unique_counts"][arm]==len({p for p,_ in attempted})
            assert isclose(a["calibration_diagonal_rates"][arm],
                           sum(n for _,n in attempted)/(N_ATTACKS*CAL),abs_tol=1e-14)
        fresh=[tuple(sorted(random.Random(seed(202611260076,digest,rep)).sample(range(m),R)))
               for rep in range(N_ATTACKS)]
        assert a["fresh_blind_attacks"]==[list(v) for v in fresh]
        evals=plans(regions,digest,"evaluation")
        counts=Counter()
        controls=Counter()
        for arm in ARMS:
            for schedule in evals[arm]:
                s=set(schedule)
                assert len(s)==16
                assert len(set(range(m))-s)>=R
                for atk in ARMS:
                    for changed in attacks[atk]:
                        counts[(atk,arm)]+=bool(s&changed)
                        actions+=1
                for f in fresh:
                    controls[arm]+=bool(s&set(f))
        assert len(a["crossplay"])==49
        for obs in a["crossplay"]:
            key=obs["attacker_policy"],obs["auditor_policy"]
            hits=counts[key]
            assert obs["detected"]==hits
            assert obs["total"]==EVAL*N_ATTACKS
            assert isclose(obs["detection_rate"],hits/(EVAL*N_ATTACKS),abs_tol=1e-15)
        assert a["fresh_control_detected"]==dict(controls)
        assert a["legitimate_failures"]==0
        assert a["informed_detected"]==0
        assert a["delta_disagreements"]==0
        floor=Fraction(1)-Fraction(comb(m-12,8),comb(m-8,8))
        assert a["blind_floor_fraction"]==[floor.numerator,floor.denominator]
        diagonal={arm:counts[(arm,arm)]/(EVAL*N_ATTACKS) for arm in ARMS}
        fresh_rate={arm:controls[arm]/(EVAL*N_ATTACKS) for arm in ARMS}
        found={p["public_sha256"]:p for p in out["primary_by_source"]}[digest]
        for arm in ARMS:
            assert isclose(found["diagonal"][arm],diagonal[arm],abs_tol=1e-15)
            assert isclose(found["fresh_blind"][arm],fresh_rate[arm],abs_tol=1e-15)
        assert isclose(found["hybrid50_minus_static"],
                       diagonal["hybrid_50"]-diagonal["static_prior"],abs_tol=1e-15)
        assert isclose(found["hybrid50_minus_uniform"],
                       diagonal["hybrid_50"]-diagonal["uniform"],abs_tol=1e-15)
        all_primary.append(diagonal)
    wins=sum(a["hybrid_50"]>a["static_prior"] for a in all_primary)
    mean=sum(a["hybrid_50"] for a in all_primary)/12
    assert wins==out["primary_hybrid50_vs_static_wins"]
    assert isclose(mean,out["primary_hybrid50_selfplay_mean"],abs_tol=1e-15)
    assert out["primary_exact_floor_verified"]
    assert out["primary_informed_attack_evasion_verified"]
    assert out["registered_primary_pass"]==(wins>=10 and mean>=.02)
    print("NOR26 INDEPENDENT REPLAY PASS",len(all_primary),"sources",
          actions,"cross-play decisions","SCIENTIFIC PASS",out["registered_primary_pass"])
    print("NOR26 RAW SOURCE",len(srcraw),sha256(srcraw).hexdigest())
    print("NOR26 RAW OUTCOME",len(outraw),sha256(outraw).hexdigest())


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--full",type=Path,required=True)
    z=p.parse_args()
    replay(z.fixture,z.full)
