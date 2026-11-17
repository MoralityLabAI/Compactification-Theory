"""NOR-29 separate-process independent replay; NEVER imports NOR-29 evaluator.

Recreates 24 RRC25 native public masks, 768 early SHA commitments,
independent root-keyed HMAC schedules using a sparse Fisher-Yates sampler,
2304 root-aware greedy attacks using trial-id SETS, and 2,359,296 paired
detection decisions. The attacker used exactly the evaluation batch, so this
is a deliberately in-sample timing-compromise measurement.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
import hashlib
import hmac
import json
from math import comb,isclose
from fractions import Fraction
from pathlib import Path
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor24_adaptive_audits import experiment as original
from research.nor25_robust_audits import experiment as prior
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

ARMS=("hybrid_50","uniform16","static_prior")
REGIMES=("blind_uniform","early_root_aware")
REPS=(12,13,14,15)
DEGREES=(512,1024)
N=24
TRIALS=512
ATTACKS=32
R=4
TOTAL=N*len(ARMS)*len(REGIMES)*ATTACKS*TRIALS
MAX=1<<256


def source_seed(rep,d,family):
    return (corpus.SEED+139329000+1000003*rep+
            100009*DEGREES.index(d)+10007*nor21.FAMILIES.index(family))


def commitment(opening):
    return sha256(b"NOR29-attack-commit-v1\x00"+
                  original.canonical(opening)).hexdigest()


def true_after_public_choice(row,pred):
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
    assert truth==pred
    return truth


def independent_schedule(root,digest,arm,trial,m,order):
    """Sparse-map implementation, independent from NOR28 and NOR29 samplers."""
    if arm=="static_prior":
        return tuple(order[:16])
    key=hmac.new(root,b"NOR29-v1-schedule-subkey",sha256).digest()
    if arm=="hybrid_50":
        top=tuple(order[:8])
        eligible=[i for i in range(m) if i not in set(top)]
        draw_count=8
    else:
        top=()
        eligible=list(range(m))
        draw_count=16
    swap={}
    chosen=[]
    for draw in range(draw_count):
        bound=len(eligible)-draw
        limit=MAX-(MAX%bound)
        rejection=0
        while True:
            msg=(f"NOR28-sampling-v1|{digest}|{arm}|{trial}|"
                 f"{draw}|{rejection}").encode("ascii")
            val=int.from_bytes(hmac.new(key,msg,hashlib.sha256).digest(),"big")
            if val<limit:
                index=val%bound
                break
            rejection+=1
            assert rejection<100
        chosen.append(swap.get(index,eligible[index]))
        swap[index]=swap.get(bound-1,eligible[bound-1])
    return top+tuple(chosen)


def independent_late_attack(digest,arm,rep,plans,m):
    # Explicit schedule-id sets give an implementation independent from
    # the stored integer bitmask calibration optimizer.
    cov=[set() for _ in range(m)]
    for j,plan in enumerate(plans):
        for i in plan:
            cov[i].add(j)
    tie=list(range(m))
    random.Random(original.stream_seed(
        202611290102,digest,arm,rep)).shuffle(tie)
    selected=set()
    exposed=set()
    for _ in range(R):
        idx=min((i for i in tie if i not in selected),
                key=lambda i:len(exposed|cov[i]))
        selected.add(idx)
        exposed|=cov[idx]
    return tuple(sorted(selected)),len(exposed)


def replay(fixture_path,early_sealed_path,early_manifest_path,
           challenge_path,late_sealed_path,late_manifest_path,result_path):
    files={
        "fixture":Path(fixture_path).read_bytes(),
        "early_sealed":Path(early_sealed_path).read_bytes(),
        "early_manifest":Path(early_manifest_path).read_bytes(),
        "challenge":Path(challenge_path).read_bytes(),
        "late_sealed":Path(late_sealed_path).read_bytes(),
        "late_manifest":Path(late_manifest_path).read_bytes(),
        "outcome":Path(result_path).read_bytes(),
    }
    f,early,em,ch,late,lm,out=(
        json.loads(files[k]) for k in
        ("fixture","early_sealed","early_manifest","challenge",
         "late_sealed","late_manifest","outcome"))
    sha={k:sha256(v).hexdigest() for k,v in files.items()}
    assert f["schema"]=="nor29.public_only_24_fresh_sources.v1"
    assert f["source_only"] and f["hidden_truth_absent"]
    assert f["attack_sets_absent"] and f["challenge_entropy_absent"]
    assert f["source_offset"]==139329000 and f["replicates"]==list(REPS)
    assert len(f["cases"])==N
    assert {(r["replicate"],r["observation_degree"],r["family"])
            for r in f["cases"]}=={
            (rep,d,family) for rep in REPS for d in DEGREES for family in nor21.FAMILIES}
    assert early["schema"]=="nor29.original_768_salted_commitments.v1"
    assert em["schema"]=="nor29.public_768_precommitments.v1"
    assert early["fixture_sha256"]==em["fixture_sha256"]==sha["fixture"]
    assert em["sealed_sha256"]==sha["early_sealed"]
    assert em["attack_count"]==len(early["openings"])==768
    assert em["challenge_entropy_absent"]
    orig_commit={(x["source_sha256"],x["replica"]):x["commitment_sha256"]
                 for x in em["commitments"]}
    blind={(x["source_sha256"],x["replica"]):x for x in early["openings"]}
    assert len(orig_commit)==len(blind)==768
    for key,opening in blind.items():
        assert opening["attack_class"]=="blind_uniform"
        assert commitment(opening)==orig_commit[key]
        assert len(opening["nonce_hex"])==64
        alt=dict(opening)
        alt["nonce_hex"]=("0" if opening["nonce_hex"][0]!="0" else "1")+opening["nonce_hex"][1:]
        assert commitment(alt)!=orig_commit[key]
    assert ch["schema"]=="nor29.single_fresh_os_root_24x3x512_schedules.v1"
    assert ch["fixture_sha256"]==sha["fixture"]
    assert ch["early_manifest_sha256"]==sha["early_manifest"]
    assert ch["root_origin"]=="os_secrets_token_bytes_32"
    assert ch["source_count"]==N and ch["trial_count"]==TRIALS
    root=bytes.fromhex(ch["root_hex_disclosed_after_freeze"])
    assert len(root)==32
    assert sha256(b"NOR29-root-v1\x00"+root).hexdigest()==ch["root_commitment_sha256"]
    assert lm["schema"]=="nor29.early_root_leaked_2304_new_commitments.v1"
    assert late["schema"]=="nor29.early_root_leaked_2304_attack_openings.v1"
    assert lm["sealed_sha256"]==sha["late_sealed"]
    assert lm["challenge_sha256"]==late["challenge_sha256"]==sha["challenge"]
    assert lm["early_manifest_sha256"]==late["early_manifest_sha256"]==sha["early_manifest"]
    assert lm["attack_count"]==len(late["openings"])==2304
    assert lm["root_disclosed_to_attacker_before_new_commitments"] is True
    new_commit={(x["source_sha256"],x["auditor"],x["replica"]):x["commitment_sha256"]
                for x in lm["commitments"]}
    changed={(x["source_sha256"],x["auditor"],x["replica"]):x
             for x in late["openings"]}
    assert len(new_commit)==len(changed)==2304
    for key,opening in changed.items():
        assert opening["attack_class"]=="early_root_aware"
        assert commitment(opening)==new_commit[key]
        early_open=blind[(key[0],key[2])]
        fake=dict(early_open)
        fake["changed_indices"]=opening["changed_indices"]
        fake["nonce_hex"]=opening["nonce_hex"]
        assert commitment(fake)!=orig_commit[(key[0],key[2])]
    assert out["schema"]=="nor29.batch_fixed_early_root_leakage_v1"
    assert out["fixture_sha256"]==sha["fixture"]
    assert out["original_early_manifest_sha256"]==sha["early_manifest"]
    assert out["original_sealed_early_sha256"]==sha["early_sealed"]
    assert out["original_challenge_sha256"]==sha["challenge"]
    assert out["late_attack_manifest_sha256"]==sha["late_manifest"]
    assert out["late_sealed_attack_sha256"]==sha["late_sealed"]
    assert out["root_commitment_sha256"]==ch["root_commitment_sha256"]
    assert out["os_root_origin"]==ch["root_origin"]
    assert out["original_early_attack_commitments"]==768
    assert out["postroot_new_attack_commitments"]==2304
    assert out["source_count"]==N
    assert out["fixed_batch_decisions"]==TOTAL
    assert out["first_run_replicates_per_source"]==ATTACKS
    assert out["challenges_per_source_policy"]==TRIALS
    assert out["shift_magnitude_disagreements"]==0
    assert out["unchanged_target_rejections"]==0
    assert out["late_attack_opens_old_commitment"]==0
    assert out["fully_informed_per_schedule_detected"]==0
    assert out["fully_informed_per_schedule_decisions"]==N*len(ARMS)*TRIALS
    assert out["original_blind_attack_decisions_unchanged_after_root_disclosure"]
    assert len(ch["records"])==len(out["source_records"])==N
    paired=[]
    group={arm:{typ:[] for typ in REGIMES} for arm in ARMS}
    replayed=0
    informed=0
    for row,source,record in zip(f["cases"],ch["records"],out["source_records"]):
        digest=row["full_public_input_sha256"]
        assert digest==source["public_sha256"]==record["public_sha256"]
        d=row["observation_degree"]
        assert row["source_seed"]==source_seed(row["replicate"],d,row["family"])
        public,h=original.public_only_case(row)
        assert public.program is None and public.future==()
        mask=nor21.native_mask(public,h,d)
        p=original.public_proposal(row)
        assert mask==p["mask"]==row["original_public_accept_mask"]
        assert record["public_mask"]==source["original_public_mask"]==mask
        actual=true_after_public_choice(row,p["predicted"])
        assert record["true_future_sha256_after_choices"]==original.sha_target(actual)
        assert record["public_prediction_sha256"]==source["public_prediction_sha256"]==p["sha256"]
        m=len(actual)
        assert record["m"]==source["m"]==m
        assert record["observation_degree"]==d
        order=original.static_schedule(
            original.supports(p["predicted"],h),prior.PRIOR,digest,0)
        assert source["risk_prefix"]==record["risk_prefix"]==list(order[:8])
        for rep in range(ATTACKS):
            orig=blind[(digest,rep)]
            rng=random.Random(original.stream_seed(202611290101,digest,rep))
            expected=tuple(sorted(rng.sample(range(m),R)))
            assert tuple(orig["changed_indices"])==expected
        assert record["blind_attack_positions"]==[
            blind[(digest,rep)]["changed_indices"] for rep in range(ATTACKS)]
        obs={(x["auditor"],x["regime"],x["replica"]):x
             for x in record["observations"]}
        assert len(obs)==len(ARMS)*len(REGIMES)*ATTACKS
        rate_by_arm={}
        for arm in ARMS:
            raw_plans=source["schedules"][arm]
            assert len(raw_plans)==TRIALS
            for trial,checked in enumerate(raw_plans):
                expected=independent_schedule(root,digest,arm,trial,m,order)
                assert tuple(checked)==expected,(digest,arm,trial)
                seen=set(checked)
                assert len(seen)==16
                # Per-schedule informed attacker picks four untouched outputs.
                unseen=[i for i in range(m) if i not in seen][:R]
                assert len(unseen)==R and not seen&set(unseen)
                assert commitment(dict(blind[(digest,0)],
                                       changed_indices=unseen,
                                       nonce_hex="0"*64))!=orig_commit[(digest,0)]
                informed+=1
            rate_by_arm[arm]={}
            for typ in REGIMES:
                hits_all=0
                for rep in range(ATTACKS):
                    if typ=="blind_uniform":
                        change=tuple(blind[(digest,rep)]["changed_indices"])
                    else:
                        op=changed[(digest,arm,rep)]
                        # Independent greedy oracle uses trial-ID sets.
                        expected,batch_hits=independent_late_attack(
                            digest,arm,rep,raw_plans,m)
                        assert tuple(op["changed_indices"])==expected
                        assert op["calibration_batch_hits"]==batch_hits
                        change=expected
                    A=set(change)
                    hits=0
                    hitmask=0
                    for trial,plan in enumerate(raw_plans):
                        detected=bool(A&set(plan))
                        if detected:
                            hits+=1
                            hitmask|=1<<trial
                    observed=obs[(arm,typ,rep)]
                    assert observed["detected"]==hits
                    assert observed["total"]==TRIALS
                    assert observed["bitset_hex"]==format(hitmask,"0128x")
                    hits_all+=hits
                    replayed+=TRIALS
                rate=hits_all/(ATTACKS*TRIALS)
                assert isclose(record["rates"][arm][typ],rate,abs_tol=1e-15)
                rate_by_arm[arm][typ]=rate
                group[arm][typ].append(rate)
            unique=len({tuple(changed[(digest,arm,rep)]["changed_indices"])
                        for rep in range(ATTACKS)})
            assert record["late_attack_unique_counts"][arm]==unique
        floor=Fraction(1)-Fraction(comb(m-12,8),comb(m-8,8))
        assert record["theoretical_fixed_blind_floor"]==[
            floor.numerator,floor.denominator]
        delta=(rate_by_arm["hybrid_50"]["blind_uniform"]-
               rate_by_arm["hybrid_50"]["early_root_aware"])
        assert isclose(delta,record["hybrid_blind_minus_late"],abs_tol=1e-15)
        paired.append(delta)
    assert replayed==TOTAL
    assert informed==N*len(ARMS)*TRIALS
    mean=statistics.mean(paired)
    wins=sum(v>0 for v in paired)
    assert wins==out["source_hybrid_blind_minus_late_wins"]
    assert isclose(mean,out["source_hybrid_blind_minus_late_mean"],abs_tol=1e-15)
    for arm in ARMS:
        for typ in REGIMES:
            value=statistics.mean(group[arm][typ])
            assert isclose(value,out["equal_source_policy_means"][arm][typ],abs_tol=1e-15)
    gates={
        "valid_early_commitment_and_late_rejection":True,
        "hybrid_source_wins_and_mean_decrease":wins>=20 and mean>=.010,
        "all_independent_decisions_and_postbinding_invariance":True,
        "informed_evasion_and_native_parity":True,
        "os_seed_and_stage_order_recorded":True,
    }
    assert out["registered_gates"]==gates
    assert out["registered_primary_pass"]==all(gates.values())
    print("NOR29 ORIGINAL SEPARATE-PROCESS REPLAY",N,"SOURCES",
          replayed,"FIXED-BATCH DECISIONS",informed,"INFORMED CONTROLS",
          "REGISTERED PASS",out["registered_primary_pass"])
    for name,raw in files.items():
        print("NOR29 ORIGINAL",name,len(raw),sha[name])


def main():
    p=argparse.ArgumentParser()
    for name in ("fixture","early-sealed","early-manifest","challenge",
                 "late-sealed","late-manifest","outcome"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    replay(a.fixture,a.early_sealed,a.early_manifest,a.challenge,
           a.late_sealed,a.late_manifest,a.outcome)


if __name__=="__main__":main()
