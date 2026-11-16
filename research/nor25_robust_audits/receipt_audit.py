"""NOR-25 independent second-process deterministic source and decision replay.

This auditor deliberately never imports research.nor25_robust_audits.experiment.
It independently reproduces NOR-25's hybrid allocation, fixed blind attacks,
combinatorial detection floor and primary source-level scores.
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

from research.nor24_adaptive_audits import experiment as n
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

REPS=(4,5)
DEGREES=(512,1024)
FAMILIES=n.FAMILIES
KINDS=n.KINDS
ALL_KINDS=n.ALL_KINDS
RADII=(1,4,16)
BUDGETS=(4,16,64)
ARMS=("uniform","static_prior","posterior","shuffled_prior",
      "hybrid_25","hybrid_50","hybrid_75","oracle_support")
FRACTIONS={"hybrid_25":1,"hybrid_50":2,"hybrid_75":3}
PRIOR=dict(zip(KINDS,(295/772,246/772,153/772,78/772)))
TRIALS=1024


def seed(*fields):
    return int.from_bytes(sha256("|".join(str(x) for x in fields).encode("ascii")).digest(),"big")


def source_seed(rep,d,family):
    return (corpus.SEED + 99325000 + 1000003*rep +
            100009*DEGREES.index(d) + 10007*FAMILIES.index(family))


def restore_sealed_truth(row, expected):
    d=row["observation_degree"]
    rep=row["replicate"]
    family=row["family"]
    s=source_seed(rep,d,family)
    assert row["source_seed"]==s
    rng=random.Random(s)
    for _ in range(row["selected_attempt"]+1):
        prog=nor21.new_header_program(rng)
        demos=tuple((x := nor21.public_source(rng,family,3*d//2),
                     corpus.execute(x,prog)) for _ in range(2))
        query=nor21.public_source(rng,family,3*d//2)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    truth=corpus.execute(query,prog)
    assert tuple(expected)==tuple(truth)
    return truth


def independent_hybrid(static,m,digest,arm,k,trial):
    u=k*FRACTIONS[arm]//4
    t=k-u
    targeted=tuple(static[:t])
    remaining=sorted(set(range(m))-set(targeted))
    rng=random.Random(seed(202611250073,digest,arm,k,trial))
    return targeted+tuple(rng.sample(remaining,u))


def replay(fixture_path,result_path):
    fixture_bytes=Path(fixture_path).read_bytes()
    result_bytes=Path(result_path).read_bytes()
    source=json.loads(fixture_bytes)
    record=json.loads(result_bytes)
    assert source["schema"]=="nor25.source_only_12_fresh_test.v1"
    assert source["source_only"] and source["private_truth_absent"]
    assert source["attack_sets_absent"] and len(source["cases"])==12
    assert record["schema"]=="nor25.frozen_prior_robust_blind_coordinate_audit.v1"
    assert record["fixture_sha256"]==sha256(fixture_bytes).hexdigest()
    assert record["frozen_prior_numerators"]==[295,246,153,78]
    assert record["frozen_prior_denominator"]==772
    assert record["source_count"]==12
    assert record["challenge_trials_per_source_policy"]==1024
    assert record["blind_attacks_fixed_before_policy_challenges"]
    assert record["public_proposals_frozen_before_private_truth"]
    assert {(r["replicate"],r["observation_degree"],r["family"])
            for r in source["cases"]} == {
            (rep,d,f) for rep in REPS for d in DEGREES for f in FAMILIES}
    mapped={r["public_sha256"]:r for r in record["source_records"]}
    assert len(mapped)==12
    primary=[]
    total_checked=0
    for row in source["cases"]:
        digest=row["full_public_input_sha256"]
        observed=mapped[digest]
        public,header=n.public_only_case(row)
        assert nor21.native_mask(public,header,row["observation_degree"]) == \
               row["original_public_accept_mask"]
        proposal=n.public_proposal(row)
        truth=restore_sealed_truth(row,proposal["predicted"])
        assert observed["public_mask"]==proposal["mask"]
        assert observed["chosen_class"]==proposal["chosen_class"]
        assert observed["public_prediction_sha256"]==proposal["sha256"]
        assert observed["sealed_true_future_sha256_postchoice"]==n.sha_target(truth)
        regions=n.supports(proposal["predicted"],header)
        m=len(proposal["predicted"])
        assert observed["support_sizes"]=={kind:len(regions[kind]) for kind in ALL_KINDS}
        attacks={}
        for kind in ALL_KINDS:
            for r in RADII:
                rng=random.Random(seed(202611250072,digest,kind,r))
                attacks[(kind,r)]=set(rng.sample(sorted(regions[kind]),r))
                assert observed["blind_attack_positions"][f"{kind}|{r}"] == \
                       sorted(attacks[(kind,r)])
        for arm in FRACTIONS:
            for k in BUDGETS:
                for r in RADII:
                    u=k*FRACTIONS[arm]//4
                    t=k-u
                    p=Fraction(1)-Fraction(comb(m-t-r,u),comb(m-t,u))
                    assert observed["theoretical_blind_floors"][f"{arm}|{k}|{r}"] == \
                           [p.numerator,p.denominator]
        counts=Counter()
        for trial in range(TRIALS):
            baselines=n.policy_schedules(regions,PRIOR,digest,trial)
            hybrids={(arm,k):independent_hybrid(
                baselines["static_prior"],m,digest,arm,k,trial)
                for arm in FRACTIONS for k in BUDGETS}
            for kind in ALL_KINDS:
                for k in BUDGETS:
                    for arm in ARMS:
                        if arm in FRACTIONS:
                            checked=hybrids[(arm,k)]
                        elif arm=="oracle_support":
                            checked=baselines[(arm,kind)][:k]
                        else:
                            checked=baselines[arm][:k]
                        assert len(checked)==len(set(checked))==k
                        assert all(i>=0 and i<m for i in checked)
                        checked_set=set(checked)
                        assert len(regions[kind]-checked_set)>=max(RADII)
                        for r in RADII:
                            # Independent exact rejection oracle: an authentic
                            # target that equals the public proposal differs
                            # after nonzero tampering precisely on attack set.
                            counts[(kind,r,k,arm)] += bool(
                                checked_set & attacks[(kind,r)])
                            total_checked+=1
        assert len(observed["observations"])==len(ALL_KINDS)*len(RADII)*len(BUDGETS)*len(ARMS)
        for cell in observed["observations"]:
            key=(cell["kind"],cell["radius"],cell["budget"],cell["policy"])
            assert cell["detected"]==counts[key],(digest,key,cell["detected"],counts[key])
            assert cell["missed"]==TRIALS-counts[key]
        assert observed["valid_target_failures"]==0
        assert observed["informed_attack_detected"]==0
        assert observed["delta_invariance_failures"]==0
        weighted={
            arm:sum(w*counts[(kind,4,16,arm)]/TRIALS
                    for kind,w in zip(KINDS,(.4,.3,.2,.1)))
            for arm in ARMS}
        off={arm:counts[(n.STRESS,4,16,arm)]/TRIALS for arm in ARMS}
        primary.append({
            "public_sha256":digest,"degree":row["observation_degree"],
            "family":row["family"],"weighted_in_template_rates":weighted,
            "off_template_rates":off,
            "hybrid50_minus_uniform":weighted["hybrid_50"]-weighted["uniform"],
            "hybrid50_minus_static":weighted["hybrid_50"]-weighted["static_prior"],
            "hybrid50_off_minus_static":off["hybrid_50"]-off["static_prior"],
        })
    assert len(primary)==12
    observed_primary={x["public_sha256"]:x for x in record["primary_by_source"]}
    for x in primary:
        y=observed_primary[x["public_sha256"]]
        assert x.keys()==y.keys()
        for key in x:
            if isinstance(x[key],dict):
                for arm in x[key]:
                    assert isclose(x[key][arm],y[key][arm],abs_tol=1e-13)
            elif isinstance(x[key],float):
                assert isclose(x[key],y[key],abs_tol=1e-13)
            else:
                assert x[key]==y[key]
    win=sum(x["hybrid50_minus_uniform"]>0 for x in primary)
    offwin=sum(x["hybrid50_off_minus_static"]>0 for x in primary)
    gain=sum(x["hybrid50_minus_uniform"] for x in primary)/12
    off=sum(x["off_template_rates"]["hybrid_50"] for x in primary)/12
    assert record["wins_hybrid50_vs_uniform"]==win
    assert record["offtemplate_wins_hybrid50_vs_static"]==offwin
    assert isclose(record["source_mean_weighted_gain_vs_uniform"],gain,abs_tol=1e-13)
    assert isclose(record["source_mean_offtemplate_hybrid50"],off,abs_tol=1e-13)
    assert record["registered_primary_pass"]==(win>=9 and gain>=.10 and offwin>=10 and off>=.02)
    print("NOR25 independent replay OK",len(primary),"sources",
          total_checked,"decisions","PASS",record["registered_primary_pass"])
    print("fixture",len(fixture_bytes),sha256(fixture_bytes).hexdigest())
    print("outcome",len(result_bytes),sha256(result_bytes).hexdigest())


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--fixture",type=Path,required=True)
    ap.add_argument("--full",type=Path,required=True)
    args=ap.parse_args()
    replay(args.fixture,args.full)
