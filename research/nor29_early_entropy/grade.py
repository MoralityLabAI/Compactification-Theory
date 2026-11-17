"""NOR-29 exact witness grading after both commitment stages have frozen.

Counts 24*3*2*32*512=2,359,296 fixed-batch decisions. Late-root attacks
are deliberately optimized *in sample* on the actual evaluation schedules.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from fractions import Fraction
from math import comb
from pathlib import Path
import json
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor29_early_entropy import prepare
from research.nor29_early_entropy import challenge
from research.nor29_early_entropy import leaked_root_attacker as late
from research.nor24_adaptive_audits import experiment as n24
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

SCHEMA="nor29.batch_fixed_early_root_leakage_v1"
DELTAS=(1,4294049777)
REGIMES=("blind_uniform","early_root_aware")
EXPECTED_DECISIONS=prepare.N_SOURCES*len(prepare.ARMS)*len(REGIMES)*prepare.REPLICAS*prepare.TRIALS


def restore_true_after_freeze(row,predicted):
    rep,d,f=row["replicate"],row["observation_degree"],row["family"]
    rng=random.Random(prepare.source_seed(rep,d,f))
    for _ in range(row["selected_attempt"]+1):
        program=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,f,3*d//2),
                     corpus.execute(x,program)) for _ in range(2))
        query=nor21.public_source(rng,f,3*d//2)
    assert row["source_seed"]==prepare.source_seed(rep,d,f)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    true=corpus.execute(query,program)
    assert true==predicted
    return true


def exact_detect(pred,true,plan,changed,delta):
    return not n24.exact_witness_accepts(pred,true,plan,frozenset(changed),delta)


def evaluate(fixture_path,early_sealed,early_manifest,challenge_path,
             late_sealed,late_manifest):
    fixture,fixture_hash=prepare.load_fixture(fixture_path)
    original,early_manifest_hash=prepare.verify_openings(
        fixture_path,early_sealed,early_manifest)
    sched,challenge_hash=challenge.verify_schedules(
        fixture_path,early_manifest,challenge_path,exact=False)
    assert sched["root_origin"]=="os_secrets_token_bytes_32"
    after,late_manifest_hash=late.verify_late(
        fixture_path,early_manifest,challenge_path,late_sealed,late_manifest)
    assert after["challenge_sha256"]==challenge_hash
    blind={(x["source_sha256"],x["replica"]):x for x in original["openings"]}
    compromised={(x["source_sha256"],x["auditor"],x["replica"]):x
                 for x in after["openings"]}
    old_commitments={(x["source_sha256"],x["replica"]):x["commitment_sha256"]
                     for x in prepare.verify_manifest(
                         fixture_path,early_manifest)[0]["commitments"]}
    assert len(blind)==768 and len(compromised)==2304
    proposals={row["full_public_input_sha256"]:n24.public_proposal(row)
               for row in fixture["cases"]}
    assert len(proposals)==prepare.N_SOURCES
    source_rows=[]
    total=0
    delta_fail=0
    legitimate_fail=0
    forged_old_commit_accepted=0
    informed_detected=0
    informed_total=0
    late_commit_accepted=0
    for row,source in zip(fixture["cases"],sched["records"]):
        digest=row["full_public_input_sha256"]
        assert digest==source["public_sha256"]
        pred=proposals[digest]["predicted"]
        true=restore_true_after_freeze(row,pred)
        assert source["public_prediction_sha256"]==proposals[digest]["sha256"]
        assert source["original_public_mask"]==proposals[digest]["mask"]
        obs=[]
        rates={}
        late_unique={}
        for arm in prepare.ARMS:
            checked=source["schedules"][arm]
            assert len(checked)==prepare.TRIALS
            checked_sets=[set(p) for p in checked]
            rates[arm]={}
            late_unique[arm]=len({
                tuple(compromised[(digest,arm,rep)]["changed_indices"])
                for rep in range(prepare.REPLICAS)})
            for regime in REGIMES:
                count=0
                for rep in range(prepare.REPLICAS):
                    opening=(blind[(digest,rep)] if regime=="blind_uniform"
                             else compromised[(digest,arm,rep)])
                    attack=frozenset(opening["changed_indices"])
                    hits=0
                    bits=0
                    if regime=="early_root_aware":
                        # A root-aware attempt cannot reopen the earlier commitment.
                        old=blind[(digest,rep)]
                        attempted=dict(old,changed_indices=list(attack),
                                       nonce_hex=opening["nonce_hex"])
                        accepted=prepare.commit(attempted)==old_commitments[(digest,rep)]
                        forged_old_commit_accepted+=int(accepted)
                        assert not accepted
                        late_commit_accepted+=1
                    for j,(plan,seen) in enumerate(zip(checked,checked_sets)):
                        # Authentic unchanged target must always pass.
                        if rep==0 and regime=="blind_uniform":
                            if not n24.exact_witness_accepts(pred,true,plan,frozenset(),1):
                                legitimate_fail+=1
                        actual_hit=bool(attack&seen)
                        d1=exact_detect(pred,true,plan,attack,DELTAS[0])
                        d2=exact_detect(pred,true,plan,attack,DELTAS[1])
                        delta_fail+=int(d1!=d2)
                        assert actual_hit==d1==d2
                        if actual_hit:
                            hits+=1
                            bits|=1<<j
                    count+=hits
                    total+=prepare.TRIALS
                    obs.append({"auditor":arm,"regime":regime,"replica":rep,
                                "detected":hits,"total":prepare.TRIALS,
                                "bitset_hex":format(bits,"0128x")})
                rates[arm][regime]=count/(prepare.REPLICAS*prepare.TRIALS)
            # Post-root, per-schedule adaptive attacker may change A on each trial.
            for plan in checked:
                uninspected=[i for i in range(len(pred)) if i not in set(plan)][:4]
                assert len(uninspected)==4 and set(uninspected).isdisjoint(plan)
                informed_total+=1
                informed_detected+=int(exact_detect(pred,true,plan,uninspected,1))
                first=blind[(digest,0)]
                fake=dict(first,changed_indices=uninspected)
                fake["nonce_hex"]=("0" if first["nonce_hex"][0]!="0" else "1")+first["nonce_hex"][1:]
                assert prepare.commit(fake)!=old_commitments[(digest,0)]
        m=len(pred)
        floor=Fraction(1)-Fraction(comb(m-12,8),comb(m-8,8))
        source_rows.append({
            "public_sha256":digest,"replicate":row["replicate"],
            "observation_degree":row["observation_degree"],"family":row["family"],
            "m":m,"public_mask":proposals[digest]["mask"],
            "public_prediction_sha256":proposals[digest]["sha256"],
            "true_future_sha256_after_choices":n24.sha_target(true),
            "risk_prefix":source["risk_prefix"],
            "blind_attack_positions":[blind[(digest,rep)]["changed_indices"]
                                      for rep in range(prepare.REPLICAS)],
            "late_attack_unique_counts":late_unique,
            "rates":rates,"observations":obs,
            "hybrid_blind_minus_late":
                rates["hybrid_50"]["blind_uniform"]-
                rates["hybrid_50"]["early_root_aware"],
            "theoretical_fixed_blind_floor":[floor.numerator,floor.denominator],
        })
    assert total==EXPECTED_DECISIONS
    assert informed_total==prepare.N_SOURCES*len(prepare.ARMS)*prepare.TRIALS
    assert delta_fail==legitimate_fail==informed_detected==forged_old_commit_accepted==0
    assert late_commit_accepted==2304
    paired=[x["hybrid_blind_minus_late"] for x in source_rows]
    paired_wins=sum(x>0 for x in paired)
    mean_gain=statistics.mean(paired)
    group_means={
        arm:{regime:statistics.mean(x["rates"][arm][regime] for x in source_rows)
             for regime in REGIMES}
        for arm in prepare.ARMS}
    passes={
        "valid_early_commitment_and_late_rejection":True,
        "hybrid_source_wins_and_mean_decrease":paired_wins>=20 and mean_gain>=.010,
        "all_independent_decisions_and_postbinding_invariance":(
            total==EXPECTED_DECISIONS and delta_fail==0),
        "informed_evasion_and_native_parity":(
            informed_detected==0 and legitimate_fail==0),
        "os_seed_and_stage_order_recorded":(
            sched["root_origin"]=="os_secrets_token_bytes_32" and
            after["challenge_sha256"]==challenge_hash),
    }
    return {
        "schema":SCHEMA,
        "source_count":prepare.N_SOURCES,
        "fixture_sha256":fixture_hash,
        "original_early_manifest_sha256":early_manifest_hash,
        "original_sealed_early_sha256":sha256(Path(early_sealed).read_bytes()).hexdigest(),
        "original_challenge_sha256":challenge_hash,
        "late_attack_manifest_sha256":late_manifest_hash,
        "late_sealed_attack_sha256":sha256(Path(late_sealed).read_bytes()).hexdigest(),
        "root_commitment_sha256":sched["root_commitment_sha256"],
        "os_root_origin":sched["root_origin"],
        "original_early_attack_commitments":768,
        "postroot_new_attack_commitments":2304,
        "fixed_batch_decisions":total,
        "first_run_replicates_per_source":prepare.REPLICAS,
        "challenges_per_source_policy":prepare.TRIALS,
        "shift_magnitude_disagreements":delta_fail,
        "unchanged_target_rejections":legitimate_fail,
        "late_attack_opens_old_commitment":forged_old_commit_accepted,
        "fully_informed_per_schedule_decisions":informed_total,
        "fully_informed_per_schedule_detected":informed_detected,
        "original_blind_attack_decisions_unchanged_after_root_disclosure":True,
        "source_records":source_rows,
        "source_hybrid_blind_minus_late_mean":mean_gain,
        "source_hybrid_blind_minus_late_wins":paired_wins,
        "equal_source_policy_means":group_means,
        "registered_gates":passes,
        "registered_primary_pass":all(passes.values()),
        "interpretation":("24 independent public source units. 32 attack cohorts and "
                          "512 schedule trials share each source; late attacks optimize "
                          "the exact 512-schedule evaluation batch IN SAMPLE. "
                          "Fresh OS root provenance, HMAC pseudorandomness, SHA binding, "
                          "trusted CI stage ordering and authentic exact target witness "
                          "are external assumptions."),
    }


def main():
    p=argparse.ArgumentParser()
    for arg in ("fixture","early-sealed","early-manifest","challenge",
                "late-sealed","late-manifest","out"):
        p.add_argument("--"+arg,type=Path,required=True)
    a=p.parse_args()
    result=evaluate(a.fixture,a.early_sealed,a.early_manifest,
                    a.challenge,a.late_sealed,a.late_manifest)
    raw=prepare.canonical(result)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_bytes(raw)
    print("NOR29 REGISTERED SCIENTIFIC PASS",result["registered_primary_pass"])
    print("NOR29 PAIRED SOURCE WINS",result["source_hybrid_blind_minus_late_wins"],
          "mean decrease",result["source_hybrid_blind_minus_late_mean"])
    print("NOR29 MEANS",result["equal_source_policy_means"])
    print("NOR29 FIRST FULL RESULT",len(raw),sha256(raw).hexdigest())


if __name__=="__main__":main()
