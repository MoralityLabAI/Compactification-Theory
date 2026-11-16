"""NOR-26: source-separated policy-aware, evaluation-schedule-blind attack game.

Registered before data in REGISTRATION.md at commit 51ee625.
Python standard library only; source fixtures contain no generating program or
hidden target. Replays use deterministic SHA-derived streams, not production
cryptographic secrets.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from math import comb
from pathlib import Path
import json
import random
import statistics
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.nor24_adaptive_audits import experiment as n24
from research.nor25_robust_audits import experiment as n25
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

SCHEMA="nor26.policy_aware_blind_attack_crossplay.v1"
FIXTURE_SCHEMA="nor26.fresh_public_only_12_sources.v1"
OFFSET=109326000
REPS=(6,7)
DEGREES=(512,1024)
FAMILIES=nor21.FAMILIES
TRIALS_CAL=256
TRIALS_EVAL=512
RADIUS=4
BUDGET=16
ATTACK_REPS=8
DELTAS=(1,4294049777)
CAL_STREAM=202611260074
ATTACK_STREAM=202611260075
FRESH_STREAM=202611260076
EVAL_STREAM=202611260077
AUDITORS=("uniform","static_prior","posterior","shuffled_prior",
          "hybrid_25","hybrid_50","hybrid_75")
PRIOR_NUMERATORS=(295,246,153,78)
PRIOR_DENOMINATOR=772
assert (n25.PRIOR_NUMERATORS==PRIOR_NUMERATORS and
        n25.PRIOR_DENOMINATOR==PRIOR_DENOMINATOR)


def seed(*parts):
    return int.from_bytes(sha256("|".join(map(str,parts)).encode("ascii")).digest(),"big")


def source_seed(rep,degree,family):
    if rep not in REPS or degree not in DEGREES or family not in FAMILIES:
        raise ValueError("NOR26 source not registered")
    return (corpus.SEED+OFFSET+1000003*rep+
            100009*DEGREES.index(degree)+10007*FAMILIES.index(family))


def new_public_source(rep,degree,family,seen):
    q=3*degree//2
    rng=random.Random(source_seed(rep,degree,family))
    for attempt in range(100):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,family,q),
                     corpus.execute(x,planted)) for _ in range(2))
        query=nor21.public_source(rng,family,q)
        case=corpus.Case("nor26-public-only",family,demos,query,(),None)
        headers=corpus.possible_headers(case)
        if len(headers)!=1:
            continue
        header=headers[0]
        if (header.base,header.delay,header.spacing,header.amplitude)!=planted[:4]:
            continue
        digest=nor21.public_digest(case,degree)
        if digest in seen:
            continue
        mask=nor21.native_mask(case,header,degree)
        if mask==0:
            continue
        preds=nor21.candidate_predictions(query,header,q)
        if len({preds[i] for i in range(16) if mask&(1<<i)})!=1:
            continue
        planted_j=corpus.OPS.index(planted[4])*4+corpus.OPS.index(planted[5])
        assert mask&(1<<planted_j)
        seen.add(digest)
        return {
            "split":"TEST","replicate":rep,"observation_degree":degree,
            "query_degree":q,"family":family,
            "source_seed":source_seed(rep,degree,family),
            "selected_attempt":attempt,
            "full_public_input_sha256":digest,
            "public_header":{"base":header.base,"delay":header.delay,
                             "spacing":header.spacing,"amplitude":header.amplitude},
            "demos":[[list(x[:degree+1]),list(y[:degree+1])] for x,y in demos],
            "public_query_input":list(query),
            "original_public_accept_mask":mask,
        }
    raise RuntimeError("NOR26 source admissibility exhausted 100 attempts")


def make_fixture():
    seen=set()
    rows=[new_public_source(rep,d,fam,seen)
          for rep in REPS for d in DEGREES for fam in FAMILIES]
    assert len(rows)==len(seen)==12
    return {
        "schema":FIXTURE_SCHEMA,"source_offset":OFFSET,
        "source_only":True,"hidden_truth_absent":True,
        "attack_sets_absent":True,"replicates":list(REPS),
        "calibration_schedules":TRIALS_CAL,"evaluation_schedules":TRIALS_EVAL,
        "attack_replicates":ATTACK_REPS,"cases":rows,
    }


def read_fixture(path):
    raw=Path(path).read_bytes()
    z=json.loads(raw)
    assert z["schema"]==FIXTURE_SCHEMA and z["source_offset"]==OFFSET
    assert z["source_only"] and z["hidden_truth_absent"] and z["attack_sets_absent"]
    assert z["replicates"]==list(REPS)
    assert z["calibration_schedules"]==TRIALS_CAL
    assert z["evaluation_schedules"]==TRIALS_EVAL
    assert z["attack_replicates"]==ATTACK_REPS
    assert len(z["cases"])==12
    assert len({c["full_public_input_sha256"] for c in z["cases"]})==12
    assert {(c["replicate"],c["observation_degree"],c["family"]) for c in z["cases"]}=={
        (rep,d,f) for rep in REPS for d in DEGREES for f in FAMILIES}
    for c in z["cases"]:
        assert c["split"]=="TEST"
        assert all(bad not in c for bad in ("program","future","witness","attack"))
    return z,sha256(raw).hexdigest()


def restore_private_after_choice(row,predicted):
    rep,d,fam=row["replicate"],row["observation_degree"],row["family"]
    rng=random.Random(source_seed(rep,d,fam))
    for _ in range(row["selected_attempt"]+1):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,fam,3*d//2),
                     corpus.execute(x,planted)) for _ in range(2))
        query=nor21.public_source(rng,fam,3*d//2)
    assert row["source_seed"]==source_seed(rep,d,fam)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    target=corpus.execute(query,planted)
    assert tuple(predicted)==tuple(target)
    return target


def phase_digest(digest,phase):
    if phase=="calibration":
        stream=CAL_STREAM
    elif phase=="evaluation":
        stream=EVAL_STREAM
    else:
        raise ValueError("unknown prospective phase")
    return sha256(f"{stream}|{phase}|{digest}".encode("ascii")).hexdigest()


def schedules(regions,digest,phase):
    stream_digest=phase_digest(digest,phase)
    n=TRIALS_CAL if phase=="calibration" else TRIALS_EVAL
    by_policy={arm:[] for arm in AUDITORS}
    for trial in range(n):
        plans=n25.policy_schedules(regions,stream_digest,trial)
        for arm in AUDITORS:
            plan=plans[(arm,BUDGET)]
            assert len(plan)==len(set(plan))==BUDGET
            by_policy[arm].append(tuple(plan))
    return by_policy


def exposure_bitsets(clean_schedules,m):
    masks=[0]*m
    for trial,checked in enumerate(clean_schedules):
        for i in checked:
            masks[i]|=(1<<trial)
    return masks


def greedy_attack(exposure,digest,policy,rep):
    """Greedily minimize empirical schedule union coverage (not global optimal)."""
    if policy not in AUDITORS or rep not in range(ATTACK_REPS):
        raise ValueError("unregistered greedy best response")
    indices=list(range(len(exposure)))
    random.Random(seed(ATTACK_STREAM,digest,policy,rep)).shuffle(indices)
    used=set()
    union=0
    picks=[]
    for _ in range(RADIUS):
        best=None
        score=None
        for j in indices:
            if j in used:
                continue
            new=(union|exposure[j]).bit_count()
            if score is None or new<score:
                best,score=j,new
        assert best is not None
        picks.append(best)
        used.add(best)
        union|=exposure[best]
    assert len(set(picks))==RADIUS
    return tuple(sorted(picks)),union.bit_count()


def fresh_attack(digest,m,rep):
    rng=random.Random(seed(FRESH_STREAM,digest,rep))
    return tuple(sorted(rng.sample(range(m),RADIUS)))


def theoretical_floor(m):
    t=BUDGET-BUDGET//2
    u=BUDGET//2
    return Fraction(1)-Fraction(comb(m-t-RADIUS,u),comb(m-t,u))


def run(path):
    fixture,fixture_hash=read_fixture(path)
    # Public proposals across ALL sources precede any private truth.
    proposals={c["full_public_input_sha256"]:n24.public_proposal(c)
               for c in fixture["cases"]}
    assert len(proposals)==12
    records=[]
    for c in fixture["cases"]:
        digest=c["full_public_input_sha256"]
        proposal=proposals[digest]
        y=proposal["predicted"]
        m=len(y)
        regions=n24.supports(y,proposal["header"])
        cal=schedules(regions,digest,"calibration")
        cov={arm:exposure_bitsets(cal[arm],m) for arm in AUDITORS}
        selected={arm:[greedy_attack(cov[arm],digest,arm,rep)
                       for rep in range(ATTACK_REPS)] for arm in AUDITORS}
        fresh=[fresh_attack(digest,m,rep) for rep in range(ATTACK_REPS)]
        assert all(len(f)==RADIUS for f in fresh)
        # Freeze sampled evaluators and all attacker choices before touching truth.
        eval_plans=schedules(regions,digest,"evaluation")
        true=restore_private_after_choice(c,y)
        attack_sets={arm:[frozenset(p) for p,_ in selected[arm]]
                     for arm in AUDITORS}
        fresh_sets=[frozenset(p) for p in fresh]
        counts={(atk,aud):0 for atk in AUDITORS for aud in AUDITORS}
        fresh_counts={aud:0 for aud in AUDITORS}
        informed_count=0
        legitimate_failure=0
        shift_disagreement=0
        for aud in AUDITORS:
            for trial,checked in enumerate(eval_plans[aud]):
                checked_set=frozenset(checked)
                if not n24.exact_witness_accepts(y,true,checked,frozenset(),1):
                    legitimate_failure+=1
                informed=n24.informed_attack(checked_set,regions["uniform"],RADIUS)
                informed_count+=bool(checked_set&informed)
                for atk in AUDITORS:
                    for changed in attack_sets[atk]:
                        counts[(atk,aud)]+=bool(checked_set&changed)
                for changed in fresh_sets:
                    fresh_counts[aud]+=bool(checked_set&changed)
                if trial<8:
                    for atk in AUDITORS:
                        for changed in attack_sets[atk]:
                            d1=n25.exact_detection(y,true,checked,changed,DELTAS[0])
                            d2=n25.exact_detection(y,true,checked,changed,DELTAS[1])
                            assert d1==bool(checked_set&changed)
                            shift_disagreement+=d1!=d2
        assert informed_count==legitimate_failure==shift_disagreement==0
        cal_diag={arm:sum(cov_rate for _,cov_rate in selected[arm])/
                  (ATTACK_REPS*TRIALS_CAL) for arm in AUDITORS}
        observations=[
            {"attacker_policy":atk,"auditor_policy":aud,
             "detected":counts[(atk,aud)],"total":TRIALS_EVAL*ATTACK_REPS,
             "detection_rate":counts[(atk,aud)]/(TRIALS_EVAL*ATTACK_REPS)}
            for atk in AUDITORS for aud in AUDITORS
        ]
        floor=theoretical_floor(m)
        records.append({
            "public_sha256":digest,"replicate":c["replicate"],
            "family":c["family"],"degree":c["observation_degree"],
            "m":m,"public_mask":proposal["mask"],
            "public_prediction_sha256":proposal["sha256"],
            "sealed_truth_postchoice_sha256":n24.sha_target(true),
            "support_sizes":{kind:len(regions[kind]) for kind in n24.ALL_KINDS},
            "policy_aware_attacks":{arm:[list(p) for p,_ in selected[arm]]
                                    for arm in AUDITORS},
            "attack_unique_counts":{arm:len(set(tuple(p) for p,_ in selected[arm]))
                                    for arm in AUDITORS},
            "calibration_diagonal_rates":cal_diag,
            "fresh_blind_attacks":[list(f) for f in fresh],
            "fresh_control_detected":fresh_counts,
            "blind_floor_fraction":[floor.numerator,floor.denominator],
            "informed_detected":informed_count,
            "legitimate_failures":legitimate_failure,
            "delta_disagreements":shift_disagreement,
            "crossplay":observations,
        })
    primary=[]
    for rec in records:
        mat={(o["attacker_policy"],o["auditor_policy"]):o["detection_rate"]
             for o in rec["crossplay"]}
        diagonal={arm:mat[(arm,arm)] for arm in AUDITORS}
        fresh={arm:rec["fresh_control_detected"][arm]/
               (TRIALS_EVAL*ATTACK_REPS) for arm in AUDITORS}
        primary.append({
            "public_sha256":rec["public_sha256"],
            "replicate":rec["replicate"],"degree":rec["degree"],
            "family":rec["family"],"diagonal":diagonal,
            "fresh_blind":fresh,
            "hybrid50_minus_static":diagonal["hybrid_50"]-diagonal["static_prior"],
            "hybrid50_minus_uniform":diagonal["hybrid_50"]-diagonal["uniform"],
        })
    wins=sum(row["hybrid50_minus_static"]>0 for row in primary)
    mean=statistics.mean(row["diagonal"]["hybrid_50"] for row in primary)
    floor_ok=all(Fraction(*rec["blind_floor_fraction"])==
                 Fraction(1)-Fraction(comb(rec["m"]-12,8),comb(rec["m"]-8,8))>0
                 for rec in records)
    valid=all(rec["informed_detected"]==0 for rec in records)
    passed=(wins>=10 and mean>=.020 and floor_ok and valid)
    return {
        "schema":SCHEMA,"fixture_sha256":fixture_hash,"source_count":12,
        "calibration_schedule_count":TRIALS_CAL,
        "evaluation_schedule_count":TRIALS_EVAL,
        "attack_replica_count":ATTACK_REPS,"audit_budget":BUDGET,
        "attack_radius":RADIUS,"frozen_prior_numerators":list(PRIOR_NUMERATORS),
        "frozen_prior_denominator":PRIOR_DENOMINATOR,
        "policy_aware_attacks_precede_evaluation_schedules":True,
        "private_truth_only_after_public_proposals":True,
        "source_records":records,"primary_by_source":primary,
        "primary_hybrid50_vs_static_wins":wins,
        "primary_hybrid50_selfplay_mean":mean,
        "primary_exact_floor_verified":floor_ok,
        "primary_informed_attack_evasion_verified":valid,
        "registered_primary_pass":passed,
        "interpretation":("Policy-aware greedy attacks fitted only to calibration "
                          "challenge distributions, no realized evaluation schedules. "
                          "Eight attacks and 512 tests per source are correlated. "
                          "External exact target witness and reproducible seeds remain assumptions."),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--make-fixture",type=Path)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    if args.make_fixture:
        if args.fixture or args.out:
            p.error("source generation and private outcomes must run in separate stages")
        raw=n24.canonical(make_fixture())
        args.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        args.make_fixture.write_bytes(raw)
        print("NOR26 FROZEN NEW SOURCE-ONLY",len(raw),sha256(raw).hexdigest())
        return
    if not args.fixture:
        p.error("--fixture is required for prospective evaluation")
    result=run(args.fixture)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(n24.canonical(result))
    print("NOR26 SELFPLAY SOURCE PAIRED",
          [(x["degree"],x["family"],round(x["hybrid50_minus_static"],5),
            round(x["hybrid50_minus_uniform"],5))
           for x in result["primary_by_source"]])
    print("NOR26 PRIMARY",result["registered_primary_pass"],
          "wins",result["primary_hybrid50_vs_static_wins"],
          "selfplay",round(result["primary_hybrid50_selfplay_mean"],6))
    print("NOR26 FLOOR",[(rec["m"],rec["blind_floor_fraction"])
                         for rec in result["source_records"][:2]])


if __name__=="__main__":
    main()
