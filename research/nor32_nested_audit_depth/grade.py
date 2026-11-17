"""NOR-32 source-locked exact witness grader after attacks and both root batches freeze.

The 2048 evaluation schedules are sampled exactly once per source/policy.
Depth n uses their first n entries. Each attacker fixes four positions
over its corresponding n-audit batch. Changes are checked against the
sealed original RRC25 program, reconstructed only after all commitments.
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
from research.nor32_nested_audit_depth import source,sampler,attacks
from research.nor24_adaptive_audits import experiment as n24
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

SCHEMA="nor32.nested_depth_attack_coverage_and_detection.v1"
DELTAS=(1,4294049777)
REGIMES=attacks.LABELS
N_DECISIONS=source.N_SOURCES*len(sampler.ARMS)*len(sampler.DEPTHS)*len(REGIMES)*attacks.REPLICAS*sum(sampler.DEPTHS)/len(sampler.DEPTHS)
N_DECISIONS=int(N_DECISIONS)
assert N_DECISIONS==23592960


def restore_truth(row,pred):
    rep,d,f=row["replicate"],row["observation_degree"],row["family"]
    rng=random.Random(source.source_seed(rep,d,f))
    for _ in range(row["selected_attempt"]+1):
        prog=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,f,3*d//2),
                     corpus.execute(x,prog)) for _ in range(2))
        query=nor21.public_source(rng,f,3*d//2)
    assert row["source_seed"]==source.source_seed(rep,d,f)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    true=corpus.execute(query,prog)
    assert pred==true
    return true


def exact_hit(pred,true,plan,changed,delta):
    return not n24.exact_witness_accepts(pred,true,plan,frozenset(changed),delta)


def run(fixture_path,calibration_path,pre_sealed_path,pre_manifest_path,
        eval_path,late_sealed_path,late_manifest_path):
    fixture,fixture_hash=source.load_fixture(fixture_path)
    calibration,cal_hash=sampler.verify_phase(fixture_path,calibration_path,exact=False)
    evaluation,eval_hash=sampler.verify_phase(fixture_path,eval_path,exact=False)
    assert calibration["phase"]=="calibration"
    assert evaluation["phase"]=="evaluation"
    assert calibration["root_origin"]==evaluation["root_origin"]=="os_secrets_token_bytes_32"
    assert calibration["root_hex_disclosed_after_freeze"]!=evaluation["root_hex_disclosed_after_freeze"]
    before,pre_sha=attacks.verify_openings(fixture_path,calibration_path,
         pre_sealed_path,pre_manifest_path,REGIMES[0])
    after,late_sha=attacks.verify_openings(fixture_path,eval_path,
         late_sealed_path,late_manifest_path,REGIMES[1],pre_manifest_path,calibration_path)
    assert evaluation["precommit_manifest_sha256"]==pre_sha
    assert evaluation["calibration_sha256"]==cal_hash
    pre={(x["source_sha256"],x["auditor"],x["depth"],x["replica"]):x
         for x in before["openings"]}
    late={(x["source_sha256"],x["auditor"],x["depth"],x["replica"]):x
          for x in after["openings"]}
    old={(x["source_sha256"],x["auditor"],x["depth"],x["replica"]):
         x["commitment_sha256"]
         for x in attacks.verify_manifest(fixture_path,pre_manifest_path,REGIMES[0])[0]
                   ["commitments"]}
    assert len(pre)==len(late)==len(old)==attacks.TOTAL
    proposals={row["full_public_input_sha256"]:n24.public_proposal(row)
               for row in fixture["cases"]}
    total=0
    delta_violations=0
    honest_rejections=0
    forged_accept=0
    informed_total=0
    informed_hit=0
    support_changed=0
    records=[]
    for row,cal_row,e_row in zip(fixture["cases"],calibration["records"],evaluation["records"]):
        digest=row["full_public_input_sha256"]
        assert cal_row["source_sha256"]==e_row["source_sha256"]==digest
        p=proposals[digest]
        pred=p["predicted"]
        assert cal_row["public_prediction_sha256"]==e_row["public_prediction_sha256"]==p["sha256"]
        assert cal_row["public_mask"]==e_row["public_mask"]==p["mask"]
        native_case,header=n24.public_only_case(row)
        assert nor21.native_mask(native_case,header,row["observation_degree"])==p["mask"]==row["original_public_accept_mask"]
        true=restore_truth(row,pred)
        m=len(pred)
        assert m==cal_row["m"]==e_row["m"]
        assert cal_row["risk_prefix"]==e_row["risk_prefix"]
        rates={}
        unique={}
        unseen={}
        attack_observations=[]
        for arm in sampler.ARMS:
            plans=e_row["schedules"][arm]
            assert len(plans)==sampler.TRIALS
            checked=[frozenset(plan) for plan in plans]
            rates[arm]={}
            unique[arm]={}
            unseen[arm]={}
            for depth in sampler.DEPTHS:
                str_depth=str(depth)
                flattened=frozenset().union(*checked[:depth])
                never=m-len(flattened)
                unseen[arm][str_depth]=never
                assert never>=0
                rates[arm][str_depth]={}
                unique[arm][str_depth]={}
                for label in REGIMES:
                    total_hits=0
                    supports=set()
                    for rep in range(attacks.REPLICAS):
                        key=(digest,arm,depth,rep)
                        opening=pre[key] if label==REGIMES[0] else late[key]
                        A=frozenset(opening["changed_indices"])
                        supports.add(tuple(opening["changed_indices"]))
                        if label==REGIMES[1]:
                            original=pre[key]
                            if A!=frozenset(original["changed_indices"]):
                                support_changed+=1
                            forged=dict(original,changed_indices=list(A),
                                        nonce_hex=opening["nonce_hex"])
                            forged_accept+=int(attacks.commitment(forged)==old[key])
                        hits=0
                        # Store each 512-2048-bit outcome mask, allowing
                        # independent exact reconstruction for every pair.
                        bits=0
                        for trial,(plan,indices) in enumerate(zip(plans[:depth],checked[:depth])):
                            if rep==0 and label==REGIMES[0]:
                                honest_rejections+=int(not n24.exact_witness_accepts(
                                    pred,true,plan,frozenset(),1))
                            actual=bool(A&indices)
                            d1=exact_hit(pred,true,plan,A,DELTAS[0])
                            d2=exact_hit(pred,true,plan,A,DELTAS[1])
                            assert actual==d1==d2
                            delta_violations+=int(d1!=d2)
                            if actual:
                                hits+=1
                                bits|=1<<trial
                        total+=depth
                        total_hits+=hits
                        attack_observations.append({
                            "auditor":arm,"depth":depth,
                            "attack_class":label,"replica":rep,
                            "hits":hits,"total":depth,
                            "detected_bits_hex":format(bits,f"0{depth//4}x"),
                        })
                    rates[arm][str_depth][label]=total_hits/(attacks.REPLICAS*depth)
                    unique[arm][str_depth][label]=len(supports)
            # Each of the 2048 original schedules is a counterfactual
            # independent inspection; adaptive per-schedule supports
            # cannot open the earlier precommitment.
            for plan,indices in zip(plans,checked):
                never_selected=[i for i in range(m) if i not in indices][:4]
                assert len(never_selected)==4
                assert not set(never_selected)&indices
                informed_total+=1
                informed_hit+=int(exact_hit(pred,true,plan,never_selected,1))
                op=pre[(digest,arm,2048,0)]
                forged=dict(op,changed_indices=never_selected,
                            nonce_hex=("0" if op["nonce_hex"][0]!="0" else "1")+op["nonce_hex"][1:])
                forged_accept+=int(attacks.commitment(forged)==old[(digest,arm,2048,0)])
        floor=Fraction(1)-Fraction(comb(m-12,8),comb(m-8,8))
        records.append({
            "source_sha256":digest,"replicate":row["replicate"],
            "degree":row["observation_degree"],"family":row["family"],
            "m":m,"public_mask":p["mask"],"prediction_sha256":p["sha256"],
            "true_target_sha256_after_freeze":n24.sha_target(true),
            "risk_prefix":e_row["risk_prefix"],
            "fixed_blind_one_trial_floor":[floor.numerator,floor.denominator],
            "never_inspected":unseen,
            "unique_attacks":unique,"detection_rates":rates,
            "observations":attack_observations,
            "hybrid2048_independent_minus_leaked":
                 rates["hybrid_50"]["2048"][REGIMES[0]]-
                 rates["hybrid_50"]["2048"][REGIMES[1]],
        })
    assert total==N_DECISIONS
    assert informed_total==source.N_SOURCES*len(sampler.ARMS)*sampler.TRIALS==147456
    assert delta_violations==honest_rejections==informed_hit==forged_accept==0
    # Coverage is literally a nested prefix: unseen counts must decline.
    monotone=all(
        all(all(rec["never_inspected"][arm][str(a)]>=rec["never_inspected"][arm][str(b)]
                for a,b in zip(sampler.DEPTHS,sampler.DEPTHS[1:]))
            for arm in sampler.ARMS)
        for rec in records)
    long=[r for r in records if r["m"]==1537]
    assert len(long)==12
    at512=sum(r["never_inspected"]["hybrid_50"]["512"]>=4 for r in long)
    at2048=sum(r["never_inspected"]["hybrid_50"]["2048"]>=4 for r in long)
    primary_source_wins=sum(r["hybrid2048_independent_minus_leaked"]>0 for r in records)
    primary_mean=statistics.mean(r["hybrid2048_independent_minus_leaked"] for r in records)
    policy_rates={
        arm:{str(d):{reg:statistics.mean(r["detection_rates"][arm][str(d)][reg] for r in records)
                     for reg in REGIMES} for d in sampler.DEPTHS}
        for arm in sampler.ARMS}
    feasibility={
        arm:{str(depth):{str(m):sum(r["never_inspected"][arm][str(depth)]>=4
                                   for r in records if r["m"]==m)
                         for m in (769,1537)}
             for depth in sampler.DEPTHS}
        for arm in sampler.ARMS}
    exact_ideal_expectation={
        str(m):{arm:{str(depth):float((m-8 if arm=="hybrid_50" else m)
                        * ((m-16 if arm=="hybrid_50" else m-16)/
                           (m-8 if arm=="hybrid_50" else m))**depth)
                       if arm!="static_prior" else float(m-16)
                       for depth in sampler.DEPTHS} for arm in sampler.ARMS}
        for m in (769,1537)}
    gates={
        "long_output_hybrid_four_unseen_512_vs_2048":
            at512>=10 and at2048<=2 and at512-at2048>=8,
        "2048_hybrid_calibrated_detection_beats_known_root":
            primary_source_wins>=20 and primary_mean>=.005,
        "coverage_monotone_and_commitments_precede_eval_root":monotone,
        "exact_two_root_replay_23592960_witness_decisions":
            total==N_DECISIONS and delta_violations==0 and honest_rejections==0,
        "fully_informed_evasion_and_commitment_binding":
            informed_total==147456 and informed_hit==forged_accept==0,
    }
    return {
        "schema":SCHEMA,
        "source_count":source.N_SOURCES,
        "fixture_sha256":fixture_hash,
        "calibration_sha256":cal_hash,
        "pre_sealed_sha256":sha256(Path(pre_sealed_path).read_bytes()).hexdigest(),
        "pre_manifest_sha256":pre_sha,
        "evaluation_sha256":eval_hash,
        "late_sealed_sha256":sha256(Path(late_sealed_path).read_bytes()).hexdigest(),
        "late_manifest_sha256":late_sha,
        "calibration_root_commitment":calibration["root_commitment_sha256"],
        "evaluation_root_commitment":evaluation["root_commitment_sha256"],
        "calibration_root_os_origin":calibration["root_origin"],
        "evaluation_root_os_origin":evaluation["root_origin"],
        "depths":list(sampler.DEPTHS),
        "restarts":attacks.REPLICAS,
        "early_commitment_count":len(pre),
        "late_commitment_count":len(late),
        "fixed_batch_decisions":total,
        "informed_per_trial_controls":informed_total,
        "informed_per_trial_detected":informed_hit,
        "source_native_mask_mismatches":0,
        "honest_target_rejections":honest_rejections,
        "delta_disagreements":delta_violations,
        "forged_prior_commitments_accepted":forged_accept,
        "late_changed_attack_support_count":support_changed,
        "postcommit_root_disclosure_changes":0,
        "monotone_nested_coverage":monotone,
        "long_hybrid_sources_with_four_unseen_at_512":at512,
        "long_hybrid_sources_with_four_unseen_at_2048":at2048,
        "hybrid2048_positive_source_pairs":primary_source_wins,
        "hybrid2048_source_mean_detection_gain":primary_mean,
        "equal_source_detection":policy_rates,
        "batch_zero_hit_feasibility":feasibility,
        "ideal_expected_never_inspected":exact_ideal_expectation,
        "source_records":records,
        "registered_gates":gates,
        "registered_primary_pass":all(gates.values()),
        "scope":("Source unit n=24; nested 512-2048 audit prefixes, equal greedy 32 "
                 "tie restarts, two independent OS roots. Root-leak arms optimize "
                 "IN SAMPLE. Cryptographic PRF/commitment guarantees and authentic "
                 "target-coefficient oracle are external assumptions."),
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
    print("NOR32 REGISTERED SCIENTIFIC PASS",result["registered_primary_pass"])
    print("NOR32 COVERAGE LONG 512/2048",
          result["long_hybrid_sources_with_four_unseen_at_512"],
          result["long_hybrid_sources_with_four_unseen_at_2048"])
    print("NOR32 HYBRID2048 PAIRED",
          result["hybrid2048_positive_source_pairs"],
          result["hybrid2048_source_mean_detection_gain"])
    print("NOR32 DETECTION GRID",result["equal_source_detection"])
    print("NOR32 FIRST OUTCOME",len(raw),sha256(raw).hexdigest())


if __name__=="__main__":main()
