"""NOR-27 independent second-process source and leakage-decision replayer.

This script never imports NOR27 experiment.py. The attacker computes greedy
coverage with explicit sets of trial indices, rather than the evaluator's
integer bitsets, and recomputes raw seed phases, exposed prefixes, source
witnesses, exact rational floors and all prospective gates independently.
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
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.nor24_adaptive_audits import experiment as base
from research.nor25_robust_audits import experiment as robust
from research.nor26_policy_aware_audits import experiment as source_prior
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

REPS=(8,9)
DEGREES=(512,1024)
LEAKS=(0,2,4,6,8)
BITS=(0,2,4)
ARMS=("hybrid_50","uniform16")
CAL=128
EVAL=256
NREP=4
PREVIEW=1024
RADIUS=4


def seed(*parts):
    return int.from_bytes(sha256("|".join(str(x) for x in parts).encode("ascii")).digest(),"big")


def src_seed(rep,d,family):
    return (corpus.SEED+119327000+1000003*rep+
            100009*DEGREES.index(d)+10007*nor21.FAMILIES.index(family))


def private_source_target(row,pred):
    rep,d,f=row["replicate"],row["observation_degree"],row["family"]
    rng=random.Random(src_seed(rep,d,f))
    for _ in range(row["selected_attempt"]+1):
        p=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,f,3*d//2),
                     corpus.execute(x,p)) for _ in range(2))
        query=nor21.public_source(rng,f,3*d//2)
    assert row["source_seed"]==src_seed(rep,d,f)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    truth=corpus.execute(query,p)
    assert truth==pred
    return truth


def risk_prefix(regions,digest):
    return tuple(base.static_schedule(regions,robust.PRIOR,digest,0)[:8])


def sample(digest,s,m,arm,T):
    rand=random.Random(seed("NOR27|audit",digest,arm,s))
    if arm=="hybrid_50":
        unseen=tuple(i for i in range(m) if i not in set(T))
        return T+tuple(rand.sample(unseen,8))
    return tuple(rand.sample(range(m),16))


def leak_attack(digest,ell,revealed,T,m):
    forbidden=set(T)|set(revealed)
    permitted=[i for i in range(m) if i not in forbidden]
    attack_seed=seed(202611270082,digest,ell,*revealed)
    return tuple(sorted(random.Random(attack_seed).sample(permitted,RADIUS)))


def conditional_seed(digest,b,bucket,phase,j):
    stream=202611270083 if phase=="cal" else 202611270084
    lo=64-b
    return (bucket<<lo)|(seed(stream,digest,b,bucket,j)&((1<<lo)-1))


def calibration_greedy(plans,digest,arm,b,bucket,rep,m):
    cov=[set() for _ in range(m)]
    for j,plan in enumerate(plans):
        for i in plan:
            cov[i].add(j)
    tie=list(range(m))
    random.Random(seed(202611270085,digest,arm,b,bucket,rep)).shuffle(tie)
    used=set()
    hit=set()
    for _ in range(RADIUS):
        # Explicit greedy schedule-set union; distinct from integer bitset evaluator.
        idx=min((i for i in tie if i not in used),key=lambda i:len(hit|cov[i]))
        used.add(idx)
        hit|=cov[idx]
    return tuple(sorted(used)),len(hit)


def full_replay(fixture,result):
    fr=Path(fixture).read_bytes()
    rr=Path(result).read_bytes()
    src=json.loads(fr)
    z=json.loads(rr)
    assert src["schema"]=="nor27.fresh_public_only_12_sources.v1"
    assert src["source_only"] and src["hidden_truth_absent"]
    assert src["audit_seeds_absent"] and src["leakage_and_attack_absent"]
    assert src["replicates"]==list(REPS)
    assert z["schema"]=="nor27.seed_prefix_and_reserve_preview_leakage.v1"
    assert z["fixture_sha256"]==sha256(fr).hexdigest()
    assert z["source_count"]==12
    assert z["frozen_training_prior_numerators"]==[295,246,153,78]
    assert z["frozen_training_prior_denominator"]==772
    assert z["public_targeted_size"]==8 and z["uniform_reserve_size"]==8
    assert z["altered_radius"]==RADIUS and z["preview_trials"]==PREVIEW
    assert z["calibration_per_seed_bucket"]==CAL
    assert z["evaluation_per_seed_bucket"]==EVAL
    assert len(src["cases"])==12
    assert len({c["full_public_input_sha256"] for c in src["cases"]})==12
    assert {(c["replicate"],c["observation_degree"],c["family"]) for c in src["cases"]}=={
        (rep,d,f) for rep in REPS for d in DEGREES for f in nor21.FAMILIES}
    by_id={r["public_sha256"]:r for r in z["source_records"]}
    assert len(by_id)==12
    all_means={ell:[] for ell in LEAKS}
    degree_means={m:{ell:[] for ell in LEAKS} for m in (769,1537)}
    seed_means={f"{arm}|{b}":[] for arm in ARMS for b in BITS}
    decision_count=0
    for row in src["cases"]:
        d=row["observation_degree"]
        digest=row["full_public_input_sha256"]
        out=by_id[digest]
        case,h=base.public_only_case(row)
        native=nor21.native_mask(case,h,d)
        prop=base.public_proposal(row)
        assert native==row["original_public_accept_mask"]==prop["mask"]==out["public_mask"]
        pred=prop["predicted"]
        truth=private_source_target(row,pred)
        assert out["sealed_target_sha256_after_public_choice"]==base.sha_target(truth)
        assert out["public_prediction_sha256"]==prop["sha256"]
        m=len(pred)
        assert out["m"]==m and m in (769,1537)
        T=risk_prefix(base.supports(pred,h),digest)
        assert out["targeted_public_prefix"]==list(T)
        # Reproduce Game A using ONLY public leaked values in the attack selector.
        counts=Counter()
        receipt=sha256()
        for trial in range(PREVIEW):
            S=seed(202611270081,digest,trial)&((1<<64)-1)
            checked=sample(digest,S,m,"hybrid_50",T)
            revealed=checked[8:]
            for ell in LEAKS:
                A=leak_attack(digest,ell,revealed[:ell],T,m)
                hit=bool(set(A)&set(checked))
                assert not set(A)&(set(T)|set(revealed[:ell]))
                counts[ell]+=hit
                receipt.update((str((ell,A,hit))+"\n").encode("ascii"))
                decision_count+=1
        assert out["preview_decision_sha256"]==receipt.hexdigest()
        for observation in out["preview"]:
            ell=observation["revealed_reserve_coordinates"]
            assert ell in LEAKS
            expect=Fraction(1)-Fraction(comb(m-8-ell-RADIUS,8-ell),
                                        comb(m-8-ell,8-ell))
            assert observation["theoretical_floor_fraction"]==[
                expect.numerator,expect.denominator]
            assert observation["detected"]==counts[ell]
            assert observation["total"]==PREVIEW
            assert observation["rate"]==counts[ell]/PREVIEW
            all_means[ell].append(observation["rate"])
            degree_means[m][ell].append(observation["rate"])
        assert counts[8]==0
        per_bucket={}
        for b in BITS:
            for label in range(1<<b):
                cal_s=[conditional_seed(digest,b,label,"cal",j) for j in range(CAL)]
                ev_s=[conditional_seed(digest,b,label,"eval",j) for j in range(EVAL)]
                assert len(set(cal_s))==CAL and len(set(ev_s))==EVAL
                assert set(cal_s).isdisjoint(ev_s)
                assert all(s>>(64-b)==label for s in cal_s+ev_s)
                for arm in ARMS:
                    calibration=[sample(digest,s,m,arm,T) for s in cal_s]
                    attacks=[calibration_greedy(calibration,digest,arm,b,label,j,m)
                             for j in range(NREP)]
                    eval_schedules=[sample(digest,s,m,arm,T) for s in ev_s]
                    hits=[sum(bool(set(plan)&set(A)) for plan in eval_schedules)
                          for A,_ in attacks]
                    key=(b,label,arm)
                    per_bucket[key]=(attacks,hits)
                    decision_count+=NREP*EVAL
        assert len(out["seed_prefix_buckets"])==sum(1<<b for b in BITS)*len(ARMS)
        for br in out["seed_prefix_buckets"]:
            key=br["bits"],br["bucket"],br["policy"]
            attacks,hits=per_bucket[key]
            assert br["attacks"]==[list(A) for A,_ in attacks]
            assert br["calibration_hit_counts"]==[n for _,n in attacks]
            assert br["evaluation_hits_by_attacker"]==hits
            assert br["detected"]==sum(hits)
            assert br["total"]==NREP*EVAL
            assert br["distinct_attacks"]==len({A for A,_ in attacks})
        for summary in out["seed_summary"]:
            b,arm=summary["bits"],summary["policy"]
            values=[per_bucket[(b,bucket,arm)] for bucket in range(1<<b)]
            count=sum(sum(hits) for _,hits in values)
            denom=len(values)*EVAL*NREP
            assert summary["detected"]==count
            assert summary["total"]==denom
            assert summary["rate"]==count/denom
            seed_means[f"{arm}|{b}"].append(count/denom)
        assert out["honest_target_failures"]==0
        assert out["delta_disagreements"]==0
        assert out["fully_informed_preview_detections"]==0
    means={str(ell):statistics.mean(all_means[ell]) for ell in LEAKS}
    assert all(isclose(means[ell],z["preview_source_means"][ell],abs_tol=1e-15)
               for ell in means)
    assert isclose(z["preview_zero_minus_four"],means["0"]-means["4"],abs_tol=1e-15)
    empirical={str(m):{str(ell):statistics.mean(degree_means[m][ell])
                       for ell in LEAKS} for m in (769,1537)}
    for m in (769,1537):
        for ell in LEAKS:
            assert isclose(empirical[str(m)][str(ell)],
                           z["preview_empirical_by_degree"][str(m)][str(ell)],abs_tol=1e-15)
            exact=float(Fraction(1)-Fraction(comb(m-8-ell-RADIUS,8-ell),
                                            comb(m-8-ell,8-ell)))
            assert z["preview_theoretical_by_degree"][str(m)][str(ell)]==exact
    preview_gate=(means["0"]-means["4"]>=.008 and means["4"]>means["8"] and means["8"]==0)
    agreement=all(
        abs(empirical[str(m)][str(ell)]-
            z["preview_theoretical_by_degree"][str(m)][str(ell)])<=.015
        for m in (769,1537) for ell in LEAKS)
    s_mean={k:statistics.mean(v) for k,v in seed_means.items()}
    for k,v in s_mean.items():
        assert isclose(z["seed_prefix_source_means"][k],v,abs_tol=1e-15)
    seed_gate=all(abs(s_mean[f"{arm}|{b}"]-s_mean[f"{arm}|0"])<=.015
                  for arm in ARMS for b in (2,4))
    assert z["primary_preview_degradation_pass"]==preview_gate
    assert z["primary_preview_theory_agreement_pass"]==agreement
    assert z["primary_seed_prefix_sensitivity_pass"]==seed_gate
    assert z["primary_exact_witness_controls_pass"]
    assert z["registered_primary_pass"]==(preview_gate and agreement and seed_gate)
    print("NOR27 INDEPENDENT SOURCE-LOCKED DECISION REPLAY",len(by_id),
          "sources",decision_count,"decisions","SCIENTIFIC PASS",z["registered_primary_pass"])
    print("SOURCE BYTES SHA256",len(fr),sha256(fr).hexdigest())
    print("RESULT BYTES SHA256",len(rr),sha256(rr).hexdigest())


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--full",type=Path,required=True)
    args=p.parse_args()
    full_replay(args.fixture,args.full)
