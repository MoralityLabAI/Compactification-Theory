"""NOR-28 independent cross-process replay of PREP, CHALLENGE and grade.

Never imports research.nor28_commit_entropy code. Greedy calibration uses sets
of trial IDs instead of the evaluator's bitsets; HMAC sampling is independently
implemented with a sparse swap map rather than the sampler's list-swap routine.
The OS root is revealed only after the prior commitment and schedule artifacts
were frozen by separate GitHub Actions jobs.
"""
from __future__ import annotations
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from math import comb, isclose
import argparse
import hashlib
import hmac
import json
from pathlib import Path
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor24_adaptive_audits import experiment as old
from research.nor25_robust_audits import experiment as robust
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

ARMS=("hybrid_50","uniform16","static_prior")
KINDS=("calibrated","fresh_uniform")
TRIALS=512
K=16
R=4
M=1<<256


def canonical(obj):
    return old.canonical(obj)


def source_seed(rep,d,family):
    return (corpus.SEED+129328000+1000003*rep+
            100009*(0 if d==512 else 1)+10007*nor21.FAMILIES.index(family))


def restore(row,pred):
    rep,d,f=row["replicate"],row["observation_degree"],row["family"]
    rng=random.Random(source_seed(rep,d,f))
    for _ in range(row["selected_attempt"]+1):
        program=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,f,3*d//2),
                     corpus.execute(x,program)) for _ in range(2))
        query=nor21.public_source(rng,f,3*d//2)
    assert row["source_seed"]==source_seed(rep,d,f)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    actual=corpus.execute(query,program)
    assert actual==pred
    return actual


def sha_preimage(p):
    return sha256(b"NOR28-attack-commit-v1\x00"+canonical(p)).hexdigest()


def rank_order(pred,h,digest):
    return old.static_schedule(old.supports(pred,h),
                               robust.PRIOR,digest,0)


def cal_greedy(digest,order,m,rep):
    digest_cal=sha256(("nor28-calibration|"+digest).encode("ascii")).hexdigest()
    targeted=tuple(order[:8])
    remaining=[i for i in range(m) if i not in set(targeted)]
    exposure=[set() for _ in range(m)]
    for trial in range(256):
        rng=random.Random(old.stream_seed(
            202611250073,digest_cal,"hybrid_50",16,trial))
        plan=targeted+tuple(rng.sample(remaining,8))
        for i in plan:exposure[i].add(trial)
    ties=list(range(m))
    random.Random(old.stream_seed(202611280091,digest,rep)).shuffle(ties)
    used=set()
    hit=set()
    for _ in range(4):
        best=min((i for i in ties if i not in used),
                 key=lambda i:len(hit|exposure[i]))
        used.add(best)
        hit|=exposure[best]
    return tuple(sorted(used))


def cal_fresh(digest,m,rep):
    rng=random.Random(old.stream_seed(202611280092,digest,rep))
    return tuple(sorted(rng.sample(range(m),4)))


def derive_schedule(root,digest,arm,trial,m,order):
    if arm=="static_prior": return tuple(order[:16])
    if arm=="hybrid_50":
        targeted=tuple(order[:8])
        eligible=[i for i in range(m) if i not in set(targeted)]
        k=8
    else:
        targeted=()
        eligible=list(range(m))
        k=16
    remaining=len(eligible)
    swaps={}
    selected=[]
    for draw in range(k):
        bound=remaining-draw
        limit=M-(M%bound)
        rejection=0
        while True:
            msg=(f"NOR28-sampling-v1|{digest}|{arm}|{trial}|"
                 f"{draw}|{rejection}").encode("ascii")
            v=int.from_bytes(hmac.new(root,msg,hashlib.sha256).digest(),"big")
            if v<limit:
                j=v%bound
                break
            rejection+=1
            assert rejection<100
        selected.append(swaps.get(j,eligible[j]))
        swaps[j]=swaps.get(bound-1,eligible[bound-1])
    return targeted+tuple(selected)


def replay(fixture_path,manifest_path,sealed_path,challenge_path,out_path):
    fr=Path(fixture_path).read_bytes()
    mr=Path(manifest_path).read_bytes()
    sr=Path(sealed_path).read_bytes()
    cr=Path(challenge_path).read_bytes()
    rr=Path(out_path).read_bytes()
    fixture=json.loads(fr)
    manifest=json.loads(mr)
    sealed=json.loads(sr)
    challenge=json.loads(cr)
    result=json.loads(rr)
    assert fixture["schema"]=="nor28.public_only_12_sources.v1"
    assert fixture["source_offset"]==129328000
    assert fixture["replicates"]==[10,11]
    assert fixture["source_only"] and fixture["private_truth_absent"]
    assert fixture["attacks_absent"] and fixture["challenge_entropy_absent"]
    assert len(fixture["cases"])==12
    assert len({x["full_public_input_sha256"] for x in fixture["cases"]})==12
    assert manifest["schema"]=="nor28.public_commitments.v1"
    assert manifest["fixture_sha256"]==sha256(fr).hexdigest()
    assert manifest["sealed_sha256"]==sha256(sr).hexdigest()
    assert manifest["attack_count"]==96 and manifest["sources"]==12
    assert manifest["challenge_entropy_absent"] is True
    assert sealed["schema"]=="nor28.attack_openings.v1"
    assert sealed["fixture_sha256"]==sha256(fr).hexdigest()
    entries={(e["source_sha256"],e["attack_class"],e["replica"]):
             e["commitment_sha256"] for e in manifest["commitments"]}
    assert len(entries)==len(sealed["openings"])==96
    preimages={}
    for op in sealed["openings"]:
        key=(op["source_sha256"],op["attack_class"],op["replica"])
        assert key in entries
        assert sha_preimage(op)==entries[key]
        assert len(op["nonce_hex"])==64
        assert len(op["changed_indices"])==len(set(op["changed_indices"]))==4
        amended=dict(op)
        amended["nonce_hex"]=("0" if op["nonce_hex"][0]!="0" else "1")+op["nonce_hex"][1:]
        assert sha_preimage(amended)!=entries[key]
        preimages[key]=op
    assert len(preimages)==96
    assert challenge["schema"]=="nor28.fresh_os_entropy_512_audits_per_source.v1"
    assert challenge["fixture_sha256"]==sha256(fr).hexdigest()
    assert challenge["manifest_sha256"]==sha256(mr).hexdigest()
    assert challenge["source_count"]==12 and challenge["trial_count"]==512
    assert challenge["root_origin"]=="os_secrets_token_bytes_32"
    root=bytes.fromhex(challenge["root_hex_disclosed_after_commit"])
    assert len(root)==32
    assert hashlib.sha256(b"NOR28-root-v1\x00"+root).hexdigest()==\
           challenge["root_commitment_sha256"]
    assert result["schema"]=="nor28.commit_before_entropy_exact_audit.v1"
    assert result["fixture_sha256"]==sha256(fr).hexdigest()
    assert result["manifest_sha256"]==sha256(mr).hexdigest()
    assert result["sealed_attack_sha256"]==sha256(sr).hexdigest()
    assert result["original_challenge_sha256"]==sha256(cr).hexdigest()
    assert result["os_root_commitment"]==challenge["root_commitment_sha256"]
    assert result["challenge_entropy_origin"]==challenge["root_origin"]
    assert result["source_count"]==12
    assert result["attack_commitment_count"]==96
    assert result["independent_decisions"]==147456
    assert result["disclosure_bit_views"]==[0,8,16,256]
    assert result["commitment_openings_accepted"]==96
    assert result["deliberate_mutations_rejected"]==96
    assert result["root_disclosed_after_challenge_freeze"]
    assert result["attack_commitment_precedes_root"]
    assert result["postcommit_leakage_decision_changes"]==0
    assert result["nonzero_shift_disagreements"]==0
    assert result["fully_informed_after_schedule_detected"]==0
    assert result["forged_prior_commitment_accepted"]==0
    assert result["honest_target_rejections"]==0
    assert len(result["source_records"])==12
    detected_total=0
    for source,record_challenge,receipt in zip(
            fixture["cases"],challenge["records"],result["source_records"]):
        digest=source["full_public_input_sha256"]
        assert digest==record_challenge["public_sha256"]==receipt["public_sha256"]
        d=source["observation_degree"]
        assert source["source_seed"]==source_seed(source["replicate"],d,source["family"])
        case,h=old.public_only_case(source)
        mask=nor21.native_mask(case,h,d)
        p=old.public_proposal(source)
        assert p["mask"]==mask==source["original_public_accept_mask"]
        assert receipt["original_public_mask"]==mask
        assert receipt["public_prediction_sha256"]==p["sha256"]
        truth=restore(source,p["predicted"])
        assert receipt["original_target_postchoice_sha256"]==old.sha_target(truth)
        m=len(p["predicted"])
        assert record_challenge["m"]==receipt["m"]==m
        order=rank_order(p["predicted"],h,digest)
        assert record_challenge["targeted_prefix"]==receipt["targeted_prefix"]==list(order[:8])
        expected_attack={}
        for rep in range(4):
            expected_attack[("calibrated",rep)]=cal_greedy(digest,order,m,rep)
            expected_attack[("fresh_uniform",rep)]=cal_fresh(digest,m,rep)
        for key,a in expected_attack.items():
            assert tuple(preimages[(digest,*key)]["changed_indices"])==a
        obs={(o["auditor"],o["attack_class"],o["replica"]):o
             for o in receipt["observations"]}
        assert len(obs)==24
        each_counts={}
        for arm in ARMS:
            plans=record_challenge["schedules"][arm]
            assert len(plans)==TRIALS
            for trial,plan in enumerate(plans):
                expected=derive_schedule(root,digest,arm,trial,m,order)
                assert tuple(plan)==expected,(digest,arm,trial)
            for kind in KINDS:
                for rep in range(4):
                    A=set(expected_attack[(kind,rep)])
                    hits=0
                    bits=0
                    for trial,plan in enumerate(plans):
                        ishit=bool(A&set(plan))
                        if ishit:
                            hits+=1
                            bits|=1<<trial
                    record=obs[(arm,kind,rep)]
                    assert record["total"]==TRIALS
                    assert record["detected"]==hits
                    assert record["bitset_hex"]==format(bits,"0128x")
                    assert isclose(record["rate"],hits/TRIALS,abs_tol=1e-15)
                    detected_total+=hits
                    each_counts[(arm,kind,rep)]=hits
        for arm in ARMS:
            for kind in KINDS:
                calc=sum(each_counts[(arm,kind,rep)] for rep in range(4))/(TRIALS*4)
                assert isclose(
                    receipt["rates_by_auditor_attack_class"][arm][kind],
                    calc,abs_tol=1e-15)
        floor=Fraction(1)-Fraction(comb(m-12,8),comb(m-8,8))
        assert receipt["fixed_blind_exact_floor"]==[floor.numerator,floor.denominator]
        assert receipt["attack_unique_calibrated"]==len({
            expected_attack[("calibrated",rep)] for rep in range(4)})
        pair=(receipt["rates_by_auditor_attack_class"]["hybrid_50"]["calibrated"]-
              receipt["rates_by_auditor_attack_class"]["static_prior"]["calibrated"])
        assert isclose(pair,receipt["calibrated_hybrid_minus_static"],abs_tol=1e-15)
        # A retrospective full-schedule attacker can always pick four unseen
        # positions, producing zero detection while failing prior commitment.
        for arm in ARMS:
            for plan in record_challenge["schedules"][arm]:
                unseen=[i for i in range(m) if i not in set(plan)][:4]
                assert len(unseen)==4
                assert not set(unseen)&set(plan)
                opening=preimages[(digest,"calibrated",0)]
                forged=dict(opening)
                forged["changed_indices"]=unseen
                forged["nonce_hex"]=("0" if opening["nonce_hex"][0]!="0" else "1")+opening["nonce_hex"][1:]
                assert sha_preimage(forged)!=entries[(digest,"calibrated",0)]
    assert result["fully_informed_after_schedule_evasion_count"]==12*3*512
    wins=sum(x["calibrated_hybrid_minus_static"]>0
             for x in result["source_records"])
    hybrid=statistics.mean(
        x["rates_by_auditor_attack_class"]["hybrid_50"]["calibrated"]
        for x in result["source_records"])
    uniform=statistics.mean(
        x["rates_by_auditor_attack_class"]["uniform16"]["calibrated"]
        for x in result["source_records"])
    static=statistics.mean(
        x["rates_by_auditor_attack_class"]["static_prior"]["calibrated"]
        for x in result["source_records"])
    assert wins==result["primary_hybrid_vs_static_source_wins"]
    assert isclose(hybrid,result["primary_hybrid_calibrated_source_mean"],abs_tol=1e-15)
    assert isclose(static,result["static_calibrated_source_mean"],abs_tol=1e-15)
    assert isclose(uniform,result["uniform_calibrated_source_mean"],abs_tol=1e-15)
    assert result["registered_primary_pass"]==(wins>=10 and hybrid>=.020)
    print("NOR28 INDEPENDENT 12-SOURCE COMMITMENT, HMAC, ATTACK AND EXACT DECISION REPLAY PASS")
    print("NOR28 147456 source/policy/attack/trial decisions; detected",detected_total,
          "scientific PASS",result["registered_primary_pass"])
    for label,raw in (("fixture",fr),("sealed",sr),("manifest",mr),
                      ("challenges",cr),("outcome",rr)):
        print("NOR28 ORIGINAL",label,len(raw),sha256(raw).hexdigest())


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("fixture","manifest","sealed","challenge","out"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    replay(a.fixture,a.manifest,a.sealed,a.challenge,a.out)
