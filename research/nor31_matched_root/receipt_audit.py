"""NOR-31 independent original-byte and 2,359,296-decision replay.

DOES NOT IMPORT NOR31 evaluator, source, sampler, or attack code. Uses
RRC25 original semantics and a separately written sparse-map Fisher-Yates
HMAC sampler and schedule-ID-set greedy attack implementation. Reports
scientific FAIL exactly as registered if original rates do not pass.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
from hashlib import sha256
from math import comb,isclose
from pathlib import Path
import hashlib
import hmac
import json
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor24_adaptive_audits import experiment as old
from research.nor25_robust_audits import experiment as prev
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

REPS=(22,23,24,25)
DEGREES=(512,1024)
ARMS=("hybrid_50","uniform16","static_prior")
PHASES=("calibration","evaluation")
REGIMES=("independent_calibration","leaked_evaluation_root")
N=24
K=16
R=4
TRIALS=512
RESTARTS=32
EXPECTED=N*len(ARMS)*len(REGIMES)*RESTARTS*TRIALS
MOD=1<<256


def source_seed(rep,degree,family):
    return (corpus.SEED+150031000+1000003*rep+
            100009*DEGREES.index(degree)+10007*nor21.FAMILIES.index(family))


def restore_native_truth(row,predicted):
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
    truth=corpus.execute(query,program)
    assert truth==predicted
    return truth


def hash_root(root):
    return sha256(b"NOR31-root-v1\x00"+root).hexdigest()


def schedule(root,phase,digest,arm,trial,m,order):
    """Independent sparse swap-map HMAC generator for both phase roots."""
    if arm=="static_prior":
        return tuple(order[:16])
    key=hmac.new(root,("NOR31-v1-"+phase+"-subkey").encode("ascii"),
                 hashlib.sha256).digest()
    if arm=="hybrid_50":
        top=tuple(order[:8])
        candidates=[i for i in range(m) if i not in set(top)]
        ndraw=8
    else:
        top=()
        candidates=list(range(m))
        ndraw=16
    overrides={}
    chosen=[]
    for draw in range(ndraw):
        remaining=len(candidates)-draw
        limit=MOD-(MOD%remaining)
        retry=0
        while True:
            message=(f"NOR28-sampling-v1|{digest}|{arm}|{trial}|"
                     f"{draw}|{retry}").encode("ascii")
            num=int.from_bytes(hmac.new(key,message,hashlib.sha256).digest(),"big")
            if num<limit:
                at=num%remaining
                break
            retry+=1
            assert retry<100
        chosen.append(overrides.get(at,candidates[at]))
        overrides[at]=overrides.get(remaining-1,candidates[remaining-1])
    return top+tuple(chosen)


def exposure(plans,m):
    """Schedule-ID sets, independent of primary integer exposure masks."""
    out=[set() for _ in range(m)]
    for trial,checked in enumerate(plans):
        for i in checked:
            out[i].add(trial)
    return out


def greedy(cov,digest,arm,rep):
    order=list(range(len(cov)))
    random.Random(old.stream_seed(202611310111,digest,arm,rep)).shuffle(order)
    taken=set()
    covered=set()
    for _ in range(R):
        best=min((i for i in order if i not in taken),
                 key=lambda i:len(covered|cov[i]))
        taken.add(best)
        covered|=cov[best]
    return tuple(sorted(taken)),len(covered)


def commitment(preimage):
    return sha256(b"NOR31-attack-commit-v1\x00"+old.canonical(preimage)).hexdigest()


def replay(paths):
    names=("fixture","calibration","pre_sealed","pre_manifest","evaluation",
           "late_sealed","late_manifest","outcome")
    raw={k:Path(paths[k]).read_bytes() for k in names}
    digests={k:sha256(raw[k]).hexdigest() for k in names}
    z={k:json.loads(raw[k]) for k in names}
    f=z["fixture"]
    assert f["schema"]=="nor31.public_only_24_fresh_sources.v1"
    assert f["source_offset"]==150031000 and f["replicates"]==list(REPS)
    assert f["source_only"] and f["hidden_truth_absent"]
    assert f["attack_sets_absent"] and f["challenge_entropy_absent"]
    assert len(f["cases"])==N
    assert len({x["full_public_input_sha256"] for x in f["cases"]})==N
    assert {(x["replicate"],x["observation_degree"],x["family"]) for x in f["cases"]}=={
        (rep,d,fam) for rep in REPS for d in DEGREES for fam in nor21.FAMILIES}
    for phase in PHASES:
        x=z[phase]
        assert x["schema"]=="nor31.independent_fresh_os_root_24x3x512.v1"
        assert x["phase"]==phase
        assert x["fixture_sha256"]==digests["fixture"]
        assert x["root_origin"]=="os_secrets_token_bytes_32"
        assert x["source_count"]==N and x["trial_count"]==TRIALS
        assert x["sampler"]=="HMAC-SHA256(NOR31 phase subkey) then NOR28 unbiased Fisher-Yates"
        assert len(x["records"])==N
        root=bytes.fromhex(x["root_hex_disclosed_after_freeze"])
        assert len(root)==32 and hash_root(root)==x["root_commitment_sha256"]
    assert z["calibration"]["root_hex_disclosed_after_freeze"]!=\
           z["evaluation"]["root_hex_disclosed_after_freeze"]
    assert z["calibration"]["precommit_manifest_sha256"] is None
    assert z["evaluation"]["precommit_manifest_sha256"]==digests["pre_manifest"]
    assert z["evaluation"]["calibration_sha256"]==digests["calibration"]
    for regime,prefix in (("independent_calibration","pre"),
                          ("leaked_evaluation_root","late")):
        sealed=z[prefix+"_sealed"]
        manifest=z[prefix+"_manifest"]
        phase="calibration" if prefix=="pre" else "evaluation"
        assert sealed["schema"]==f"nor31.{('pre_root' if prefix=='pre' else 'leaked_root')}_2304_salted_openings.v1"
        assert manifest["schema"]==f"nor31.{('pre_root' if prefix=='pre' else 'leaked_root')}_2304_commitments.v1"
        assert sealed["fixture_sha256"]==manifest["fixture_sha256"]==digests["fixture"]
        assert sealed["phase_sha256"]==manifest["phase_sha256"]==digests[phase]
        assert manifest["sealed_sha256"]==digests[prefix+"_sealed"]
        assert sealed["calibration_sha256"]==manifest["calibration_sha256"]==digests["calibration"]
        assert sealed["precommit_manifest_sha256"]==manifest["precommit_manifest_sha256"]==(
            None if prefix=="pre" else digests["pre_manifest"])
        assert manifest["attacker_class"]==regime
        assert manifest["root_of_phase_known_before_committing"]==(prefix=="late")
        assert len(sealed["openings"])==len(manifest["commitments"])==2304
    cal_root=bytes.fromhex(z["calibration"]["root_hex_disclosed_after_freeze"])
    eval_root=bytes.fromhex(z["evaluation"]["root_hex_disclosed_after_freeze"])
    pri={(p["source_sha256"],p["auditor"],p["replica"]):p
         for p in z["pre_sealed"]["openings"]}
    lat={(p["source_sha256"],p["auditor"],p["replica"]):p
         for p in z["late_sealed"]["openings"]}
    assert len(pri)==len(lat)==2304
    original_manifest={(m["source_sha256"],m["auditor"],m["replica"]):
                       m["commitment_sha256"]
                       for m in z["pre_manifest"]["commitments"]}
    late_manifest={(m["source_sha256"],m["auditor"],m["replica"]):
                   m["commitment_sha256"]
                   for m in z["late_manifest"]["commitments"]}
    assert len(original_manifest)==len(late_manifest)==2304
    for lookup,commits,phase in ((pri,original_manifest,"independent_calibration"),
                                 (lat,late_manifest,"leaked_evaluation_root")):
        for key,x in lookup.items():
            assert x["attack_class"]==phase
            assert x["source_sha256"]==key[0] and x["auditor"]==key[1] and x["replica"]==key[2]
            assert x["changed_indices"]==sorted(x["changed_indices"])
            assert len(set(x["changed_indices"]))==len(x["changed_indices"])==R
            assert len(x["nonce_hex"])==64
            assert commitment(x)==commits[key]
            bad=dict(x,nonce_hex=("0" if x["nonce_hex"][0]!="0" else "1")+x["nonce_hex"][1:])
            assert commitment(bad)!=commits[key]
        if phase=="leaked_evaluation_root":
            for key,x in lookup.items():
                old_open=pri[key]
                attempted=dict(old_open,changed_indices=x["changed_indices"],
                               nonce_hex=x["nonce_hex"])
                assert commitment(attempted)!=original_manifest[key]
    result=z["outcome"]
    assert result["schema"]=="nor31.prospective_independent_calibration_vs_leaked_eval_root.v1"
    for key,field in (("fixture","fixture_sha256"),("calibration","calibration_sha256"),
                      ("pre_manifest","pre_root_manifest_sha256"),
                      ("pre_sealed","pre_root_sealed_sha256"),
                      ("evaluation","evaluation_sha256"),
                      ("late_manifest","late_manifest_sha256"),
                      ("late_sealed","late_sealed_sha256")):
        assert result[field]==digests[key],field
    assert result["calibration_root_commitment_sha256"]==z["calibration"]["root_commitment_sha256"]
    assert result["evaluation_root_commitment_sha256"]==z["evaluation"]["root_commitment_sha256"]
    assert result["original_early_commitments"]==result["post_eval_root_commitments"]==2304
    assert result["source_count"]==N and len(result["source_records"])==N
    assert result["fixed_batch_decisions"]==EXPECTED
    assert result["attack_restarts_per_source_arm"]==RESTARTS
    assert result["training_schedules_per_source_arm"]==TRIALS
    assert result["evaluation_schedules_per_source_arm"]==TRIALS
    assert result["root_phase_independence_checked"]
    assert result["original_attacks_unchanged_by_eval_root_disclosure"]
    assert result["late_or_informed_accepted_old_commitments"]==0
    assert result["delta_disagreements"]==0
    assert result["legitimate_target_rejections"]==0
    assert result["per_schedule_informed_detections"]==0
    assert result["per_schedule_informed_tests"]==N*len(ARMS)*TRIALS
    metrics={arm:{reg:[] for reg in REGIMES} for arm in ARMS}
    paired=[]
    total_decisions=0
    changed_supports=0
    informed_checks=0
    for row,cal_row,eval_row,receipt in zip(
            f["cases"],z["calibration"]["records"],z["evaluation"]["records"],
            result["source_records"]):
        digest=row["full_public_input_sha256"]
        assert digest==cal_row["source_sha256"]==eval_row["source_sha256"]==receipt["source_sha256"]
        public,h=old.public_only_case(row)
        assert public.program is None and public.future==()
        mask=nor21.native_mask(public,h,row["observation_degree"])
        proposal=old.public_proposal(row)
        assert proposal["mask"]==mask==row["original_public_accept_mask"]
        assert cal_row["public_mask"]==eval_row["public_mask"]==receipt["public_mask"]==mask
        assert proposal["sha256"]==cal_row["public_prediction_sha256"]==eval_row["public_prediction_sha256"]
        truth=restore_native_truth(row,proposal["predicted"])
        assert receipt["true_future_sha256_postfreeze"]==old.sha_target(truth)
        m=len(truth)
        assert cal_row["m"]==eval_row["m"]==receipt["m"]==m
        assert receipt["degree"]==row["observation_degree"]
        order=old.static_schedule(old.supports(proposal["predicted"],h),
                                  prev.PRIOR,digest,0)
        assert receipt["risk_prefix"]==eval_row["risk_prefix"]==cal_row["risk_prefix"]==list(order[:8])
        for phase,root,phase_record in (("calibration",cal_root,cal_row),
                                         ("evaluation",eval_root,eval_row)):
            for arm in ARMS:
                plans=phase_record["schedules"][arm]
                assert len(plans)==TRIALS
                for j,checked in enumerate(plans):
                    expected=schedule(root,phase,digest,arm,j,m,order)
                    assert tuple(checked)==expected,(digest,phase,arm,j)
        observations={(q["auditor"],q["regime"],q["replica"]):q
                      for q in receipt["observations"]}
        assert len(observations)==len(ARMS)*len(REGIMES)*RESTARTS
        for arm in ARMS:
            test_plans=eval_row["schedules"][arm]
            coverage_by_regime={
                "independent_calibration":exposure(cal_row["schedules"][arm],m),
                "leaked_evaluation_root":exposure(test_plans,m),
            }
            for regime in REGIMES:
                hitcount=0
                unique=set()
                for rep in range(RESTARTS):
                    key=(digest,arm,rep)
                    op=pri[key] if regime=="independent_calibration" else lat[key]
                    expected,train_hits=greedy(coverage_by_regime[regime],digest,arm,rep)
                    assert tuple(op["changed_indices"])==expected
                    assert op["training_covered_schedules"]==train_hits
                    unique.add(expected)
                    if regime=="leaked_evaluation_root":
                        if expected!=tuple(pri[key]["changed_indices"]):
                            changed_supports+=1
                    A=set(expected)
                    hits=0
                    mask_bits=0
                    for trial,plan in enumerate(test_plans):
                        if A.intersection(plan):
                            hits+=1
                            mask_bits|=1<<trial
                    obs=observations[(arm,regime,rep)]
                    assert obs["detected"]==hits and obs["total"]==TRIALS
                    assert obs["bitset_hex"]==format(mask_bits,"0128x")
                    hitcount+=hits
                    total_decisions+=TRIALS
                assert receipt["unique_attacks"][arm][regime]==len(unique)
                rate=hitcount/(RESTARTS*TRIALS)
                assert isclose(receipt["detection"][arm][regime],rate,abs_tol=1e-15)
                metrics[arm][regime].append(rate)
            # Counterfactual full-schedule attacker can pick four
            # still-uninspected indices and will fail to open old digest.
            for checked in test_plans:
                unseen=[i for i in range(m) if i not in set(checked)][:4]
                assert len(unseen)==4 and set(unseen).isdisjoint(checked)
                original=pri[(digest,arm,0)]
                forged=dict(original,changed_indices=unseen,
                            nonce_hex=("0" if original["nonce_hex"][0]!="0"
                                       else "1")+original["nonce_hex"][1:])
                assert commitment(forged)!=original_manifest[(digest,arm,0)]
                informed_checks+=1
        floor=Fraction(1)-Fraction(comb(m-12,8),comb(m-8,8))
        assert receipt["exact_fixed_blind_reserve_floor"]==[floor.numerator,floor.denominator]
        d=(receipt["detection"]["hybrid_50"]["independent_calibration"]-
           receipt["detection"]["hybrid_50"]["leaked_evaluation_root"])
        assert isclose(receipt["paired_hybrid_calibration_minus_leak"],d,abs_tol=1e-15)
        paired.append(d)
    assert total_decisions==EXPECTED
    assert informed_checks==N*len(ARMS)*TRIALS
    assert result["late_changed_attack_supports"]==changed_supports
    source_wins=sum(x>0 for x in paired)
    source_mean=statistics.mean(paired)
    assert source_wins==result["primary_positive_source_pairs"]
    assert isclose(source_mean,result["primary_paired_hybrid_mean_decrease"],abs_tol=1e-15)
    for arm in ARMS:
        for regime in REGIMES:
            mean=statistics.mean(metrics[arm][regime])
            assert isclose(result["equal_source_policy_means"][arm][regime],
                           mean,abs_tol=1e-15)
    expected_gates={
        "all_original_and_late_commitments_valid":True,
        "hybrid_matched_source_win_count_and_mean":source_wins>=20 and source_mean>=.015,
        "independent_source_schedules_attack_and_decision_replay":True,
        "informed_early_recommitment_evasion":True,
        "distinct_os_roots_and_stage_order":True,
    }
    assert result["registered_gates"]==expected_gates
    assert result["registered_primary_pass"]==all(expected_gates.values())
    print("NOR31 INDEPENDENT SOURCE AND HMAC REPLAY",N,"SOURCES",EXPECTED,"DECISIONS",
          "PAIRED WINS",source_wins,"MEAN",source_mean,
          "SCIENTIFIC PASS",result["registered_primary_pass"])
    for key in names:
        print("NOR31 FIRST ORIGINAL",key,len(raw[key]),digests[key])


def main():
    p=argparse.ArgumentParser()
    for key in ("fixture","calibration","pre-sealed","pre-manifest",
                "evaluation","late-sealed","late-manifest","outcome"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    replay({k:getattr(a,k) for k in ("fixture","calibration","pre_sealed","pre_manifest",
                                     "evaluation","late_sealed","late_manifest","outcome")})


if __name__=="__main__":main()
