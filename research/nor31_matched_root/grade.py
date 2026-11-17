"""NOR-31 exact prospective matched-root audit grader.

All commitments and 512 calibration/evaluation schedules must have been
frozen before this separate process reconstructs original true futures.
Root-aware attacks optimize on the EVALUATION schedules in sample.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
from hashlib import sha256
from math import comb
from pathlib import Path
import json
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor31_matched_root import source, sampler, attacks
from research.nor24_adaptive_audits import experiment as n24
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

SCHEMA="nor31.prospective_independent_calibration_vs_leaked_eval_root.v1"
REGIMES=attacks.LABELS
NONZERO_DELTAS=(1,4294049777)
DECISIONS=24*3*2*32*512


def restore_truth_after_freeze(row,predicted):
    rep,d,f=row["replicate"],row["observation_degree"],row["family"]
    rng=random.Random(source.source_seed(rep,d,f))
    for _ in range(row["selected_attempt"]+1):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,f,3*d//2),
                     corpus.execute(x,planted)) for _ in range(2))
        query=nor21.public_source(rng,f,3*d//2)
    assert row["source_seed"]==source.source_seed(rep,d,f)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    honest=corpus.execute(query,planted)
    assert honest==predicted
    return honest


def exact_detect(candidate,real,plan,changed,delta):
    return not n24.exact_witness_accepts(candidate,real,plan,
                                         frozenset(changed),delta)


def run(fixture_path,calibration_path,pre_sealed_path,pre_manifest_path,
        evaluation_path,late_sealed_path,late_manifest_path):
    fixture,fhash=source.load_fixture(fixture_path)
    cal,cal_hash=sampler.verify_phase(fixture_path,calibration_path,exact=False)
    e,eval_hash=sampler.verify_phase(fixture_path,evaluation_path,exact=False)
    assert cal["phase"]=="calibration" and e["phase"]=="evaluation"
    assert cal["root_origin"]==e["root_origin"]=="os_secrets_token_bytes_32"
    assert cal["root_hex_disclosed_after_freeze"]!=e["root_hex_disclosed_after_freeze"]
    pre,pre_manifest_hash=attacks.verify_openings(
        fixture_path,calibration_path,pre_sealed_path,pre_manifest_path,
        "independent_calibration")
    late,late_manifest_hash=attacks.verify_openings(
        fixture_path,evaluation_path,late_sealed_path,late_manifest_path,
        "leaked_evaluation_root",pre_manifest_path,calibration_path)
    assert e["precommit_manifest_sha256"]==pre_manifest_hash
    assert e["calibration_sha256"]==cal_hash
    pre_open={(x["source_sha256"],x["auditor"],x["replica"]):x
              for x in pre["openings"]}
    late_open={(x["source_sha256"],x["auditor"],x["replica"]):x
               for x in late["openings"]}
    original_commits={(x["source_sha256"],x["auditor"],x["replica"]):
                       x["commitment_sha256"]
                      for x in attacks.verify_manifest(
                          fixture_path,pre_manifest_path,"independent_calibration")[0]
                          ["commitments"]}
    assert len(pre_open)==len(late_open)==len(original_commits)==2304
    proposals={r["full_public_input_sha256"]:n24.public_proposal(r)
               for r in fixture["cases"]}
    n_decisions=0
    nonzero_disagreements=0
    legitimate_rejections=0
    informed_checks=0
    informed_detections=0
    forged_openings=0
    late_changed_supports=0
    rows=[]
    for original,calrow,evalrow in zip(fixture["cases"],cal["records"],e["records"]):
        digest=original["full_public_input_sha256"]
        assert digest==calrow["source_sha256"]==evalrow["source_sha256"]
        p=proposals[digest]
        m=len(p["predicted"])
        assert m==calrow["m"]==evalrow["m"]
        public,h=n24.public_only_case(original)
        native=nor21.native_mask(public,h,original["observation_degree"])
        assert native==original["original_public_accept_mask"]
        assert p["mask"]==native==calrow["public_mask"]==evalrow["public_mask"]
        assert p["sha256"]==calrow["public_prediction_sha256"]==evalrow["public_prediction_sha256"]
        true=restore_truth_after_freeze(original,p["predicted"])
        rates={}
        observations=[]
        uniques={}
        for arm in sampler.ARMS:
            # Grading uses only frozen evaluation schedules.
            plans=evalrow["schedules"][arm]
            assert len(plans)==sampler.TRIALS
            checked=[frozenset(plan) for plan in plans]
            rates[arm]={}
            uniques[arm]={}
            for regime in REGIMES:
                hits_total=0
                distinct=set()
                for rep in range(attacks.REPLICAS):
                    key=(digest,arm,rep)
                    entry=(pre_open[key] if regime=="independent_calibration"
                           else late_open[key])
                    A=frozenset(entry["changed_indices"])
                    assert len(A)==attacks.ATTACK_RADIUS
                    distinct.add(tuple(entry["changed_indices"]))
                    if regime=="leaked_evaluation_root":
                        prior=pre_open[key]
                        if A!=frozenset(prior["changed_indices"]):
                            late_changed_supports+=1
                        # A postroot opening with a changed nonce/label MUST fail
                        # the old commitment, including where support happens to match.
                        attempted=dict(prior,changed_indices=entry["changed_indices"],
                                       nonce_hex=entry["nonce_hex"])
                        if attacks.commitment(attempted)==original_commits[key]:
                            forged_openings+=1
                    hits=0
                    bitmask=0
                    for trial,(plan,seen) in enumerate(zip(plans,checked)):
                        if rep==0 and regime=="independent_calibration":
                            legitimate_rejections+=int(not n24.exact_witness_accepts(
                                p["predicted"],true,plan,frozenset(),1))
                        actual_hit=bool(A&seen)
                        d1=exact_detect(p["predicted"],true,plan,A,NONZERO_DELTAS[0])
                        d2=exact_detect(p["predicted"],true,plan,A,NONZERO_DELTAS[1])
                        assert actual_hit==d1==d2
                        nonzero_disagreements+=int(d1!=d2)
                        if d1:
                            hits+=1
                            bitmask|=1<<trial
                    hits_total+=hits
                    n_decisions+=sampler.TRIALS
                    observations.append({
                        "auditor":arm,"regime":regime,"replica":rep,
                        "detected":hits,"total":sampler.TRIALS,
                        "bitset_hex":format(bitmask,"0128x"),
                    })
                rates[arm][regime]=hits_total/(attacks.REPLICAS*sampler.TRIALS)
                uniques[arm][regime]=len(distinct)
            for seen,plan in zip(checked,plans):
                unseen=tuple(i for i in range(m) if i not in seen)[:4]
                assert len(unseen)==4 and set(unseen).isdisjoint(seen)
                informed_checks+=1
                informed_detections+=int(exact_detect(p["predicted"],true,plan,unseen,1))
                # The informed per-challenge alternative cannot open
                # the earlier, independently-calibrated salted commitment.
                early=pre_open[(digest,arm,0)]
                bogus=dict(early,changed_indices=list(unseen),
                           nonce_hex=("0" if early["nonce_hex"][0]!="0" else "1")
                                      +early["nonce_hex"][1:])
                forged_openings+=int(attacks.commitment(bogus)==original_commits[(digest,arm,0)])
        floor=Fraction(1)-Fraction(comb(m-12,8),comb(m-8,8))
        rows.append({
            "source_sha256":digest,
            "replicate":original["replicate"],
            "degree":original["observation_degree"],
            "family":original["family"],"m":m,
            "public_mask":p["mask"],
            "public_prediction_sha256":p["sha256"],
            "true_future_sha256_postfreeze":n24.sha_target(true),
            "risk_prefix":evalrow["risk_prefix"],
            "exact_fixed_blind_reserve_floor":[floor.numerator,floor.denominator],
            "unique_attacks":uniques,
            "detection":rates,"observations":observations,
            "paired_hybrid_calibration_minus_leak":
                rates["hybrid_50"]["independent_calibration"]-
                rates["hybrid_50"]["leaked_evaluation_root"],
        })
    assert n_decisions==DECISIONS
    assert informed_checks==24*3*512
    assert informed_detections==legitimate_rejections==nonzero_disagreements==forged_openings==0
    paired=[r["paired_hybrid_calibration_minus_leak"] for r in rows]
    wins=sum(p>0 for p in paired)
    diff=statistics.mean(paired)
    policy_means={
        arm:{reg:statistics.mean(row["detection"][arm][reg] for row in rows)
             for reg in REGIMES}
        for arm in sampler.ARMS}
    gates={
        "all_original_and_late_commitments_valid":True,
        "hybrid_matched_source_win_count_and_mean":wins>=20 and diff>=.015,
        "independent_source_schedules_attack_and_decision_replay":(
            n_decisions==DECISIONS and nonzero_disagreements==0),
        "informed_early_recommitment_evasion":(
            informed_checks==24*3*512 and informed_detections==0),
        "distinct_os_roots_and_stage_order":(
            cal["root_origin"]==e["root_origin"]=="os_secrets_token_bytes_32"
            and cal["root_hex_disclosed_after_freeze"]!=e["root_hex_disclosed_after_freeze"]),
    }
    return {
        "schema":SCHEMA,
        "source_count":source.N_SOURCES,
        "fixture_sha256":fhash,
        "calibration_sha256":cal_hash,
        "pre_root_manifest_sha256":pre_manifest_hash,
        "pre_root_sealed_sha256":sha256(Path(pre_sealed_path).read_bytes()).hexdigest(),
        "evaluation_sha256":eval_hash,
        "late_manifest_sha256":late_manifest_hash,
        "late_sealed_sha256":sha256(Path(late_sealed_path).read_bytes()).hexdigest(),
        "calibration_root_commitment_sha256":cal["root_commitment_sha256"],
        "evaluation_root_commitment_sha256":e["root_commitment_sha256"],
        "fixed_batch_decisions":n_decisions,
        "attack_restarts_per_source_arm":attacks.REPLICAS,
        "training_schedules_per_source_arm":sampler.TRIALS,
        "evaluation_schedules_per_source_arm":sampler.TRIALS,
        "original_early_commitments":len(pre_open),
        "post_eval_root_commitments":len(late_open),
        "late_changed_attack_supports":late_changed_supports,
        "late_or_informed_accepted_old_commitments":forged_openings,
        "delta_disagreements":nonzero_disagreements,
        "legitimate_target_rejections":legitimate_rejections,
        "per_schedule_informed_tests":informed_checks,
        "per_schedule_informed_detections":informed_detections,
        "original_attacks_unchanged_by_eval_root_disclosure":True,
        "root_phase_independence_checked":True,
        "source_records":rows,
        "primary_paired_hybrid_mean_decrease":diff,
        "primary_positive_source_pairs":wins,
        "equal_source_policy_means":policy_means,
        "registered_gates":gates,
        "registered_primary_pass":all(gates.values()),
        "scope":("24 independent sources; 32 tie-matched greedy attack restarts and "
                 "512 schedule trials per source/policy. Leaked root attacks train "
                 "IN SAMPLE on the exact evaluation schedules. Computational "
                 "SHA binding, HMAC security, OS entropy, runner-stage integrity "
                 "and authentic exact target witness are external assumptions."),
    }


def main():
    p=argparse.ArgumentParser()
    for key in ("fixture","calibration","pre-sealed","pre-manifest",
                "evaluation","late-sealed","late-manifest","out"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    result=run(a.fixture,a.calibration,a.pre_sealed,a.pre_manifest,
               a.evaluation,a.late_sealed,a.late_manifest)
    raw=source.canonical(result)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_bytes(raw)
    print("NOR31 FIRST PROSPECTIVE SCIENTIFIC PASS",result["registered_primary_pass"])
    print("NOR31 MATCHED ROOT WINS",result["primary_positive_source_pairs"],
          "MEAN DECREASE",result["primary_paired_hybrid_mean_decrease"])
    print("NOR31 POLICY SOURCE MEANS",result["equal_source_policy_means"])
    print("NOR31 FIRST FULL OUTCOME",len(raw),sha256(raw).hexdigest())


if __name__=="__main__":main()
