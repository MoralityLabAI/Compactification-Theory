"""NOR-33 prospective exact-overlap-matched transfer campaign.

Seven complete PUBLIC rankings, five forced support-overlap levels
(0,24,48,72,96)/96, and 8 source-bound four-coordinate attacks with
exactly q=0..4 changes inside the historic risk union. All public
candidate outputs and decisions freeze before private native truths
are reconstructed. Original NOR-25 auditors are unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from math import comb
from pathlib import Path
from zipfile import ZipFile
import json
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import experiment as nor21
from research.nor24_adaptive_audits import experiment as nor24
from research.nor25_robust_audits import experiment as nor25
from research.nor32_ood_audits import experiment as nor32

SCHEMA="nor33.first_matched_overlap_attack_transfer.v1"
FIXTURE_SCHEMA="nor33.source_only_reps40to43.v1"
REPS=(40,41,42,43)
DEGREES=(512,1024)
FAMILIES=nor21.FAMILIES
OFFSET=130033000
ARMS=("uniform","static_prior","posterior",
      "hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
SUPPORT_FAMILIES=("central","stride17","low_abs","curvature",
                  "anti_prior","hash","stratified_random")
STRATA=(0,1,2,3,4)
SUPPORT_SIZE=96
K=16
R=4
ATTACKS=8
TRIALS=512
ATTACK_STREAM=202611330071
AUDIT_STREAM=202611330072
MIXED_STREAM=202611330073
DELTAS=(1,4294049777)
N32_PUBLIC_SHA="129dba8ef9b1ae64edf58f9b71d6de761c3f4ca4fe5be4224e2495e3353518b7"


def canonical(x):
    return (json.dumps(x,indent=2,sort_keys=True,ensure_ascii=True)+"\n").encode()


def sha(x):
    return sha256(x).hexdigest()


def seed(*parts):
    return int.from_bytes(sha256("|".join(map(str,parts)).encode("ascii")).digest(),"big")


def past_source_digests():
    original36=nor32.load_historical()
    archive=ROOT/"research/nor32_ood_audits/results/nor32_original_hosted.zip"
    with ZipFile(archive) as z:
        assert z.testzip() is None
        source=z.read("nor32_fixture.json")
    assert len(source)==3892735 and sha(source)==N32_PUBLIC_SHA
    prior=json.loads(source)
    assert prior["schema"]=="nor32.fresh_public_only_reps32to35.v1"
    prior24=[r["full_public_input_sha256"] for r in prior["cases"]]
    assert len(prior24)==24
    all_ids=sorted(original36+prior24)
    assert len(set(all_ids))==60
    return all_ids


def source_seed(rep,d,family):
    if rep not in REPS or d not in DEGREES or family not in FAMILIES:
        raise ValueError("source outside registered roster")
    return (corpus.SEED+OFFSET+1000003*rep+
            100009*DEGREES.index(d)+10007*FAMILIES.index(family))


def generate_source(rep,d,family,seen):
    q=3*d//2
    rng=random.Random(source_seed(rep,d,family))
    for attempt in range(100):
        planted=nor21.new_header_program(rng)
        demos=tuple((src:=nor21.public_source(rng,family,q),
                     corpus.execute(src,planted)) for _ in range(2))
        query=nor21.public_source(rng,family,q)
        c=corpus.Case("nor33-source",family,demos,query,(),None)
        headers=corpus.possible_headers(c)
        if len(headers)!=1:continue
        h=headers[0]
        if (h.base,h.delay,h.spacing,h.amplitude)!=planted[:4]:continue
        digest=nor21.public_digest(c,d)
        if digest in seen:continue
        mask=nor21.native_mask(c,h,d)
        if mask==0:continue
        predicted=nor21.candidate_predictions(query,h,q)
        if len({predicted[i] for i in range(16) if mask&(1<<i)})!=1:continue
        planted_idx=4*corpus.OPS.index(planted[4])+corpus.OPS.index(planted[5])
        assert mask&(1<<planted_idx)
        seen.add(digest)
        return {
            "split":"TEST","replicate":rep,"degree":d,"query_degree":q,
            "family":family,"source_seed":source_seed(rep,d,family),
            "selected_attempt":attempt,"full_public_input_sha256":digest,
            "public_header":{"base":h.base,"delay":h.delay,
                "spacing":h.spacing,"amplitude":h.amplitude},
            "demos":[[list(src[:d+1]),list(y[:d+1])] for src,y in demos],
            "public_query_input":list(query),
            "native_public_mask":mask}
    raise RuntimeError("NOR33 100 admissible source attempts exhausted")


def make_fixture():
    before=past_source_digests()
    seen=set(before)
    cases=[generate_source(rep,d,f,seen)
           for rep in REPS for d in DEGREES for f in FAMILIES]
    assert len(cases)==24 and len(seen)==84
    return {"schema":FIXTURE_SCHEMA,"registration_first_commit":
            "9e14d1132744272c6c674e3799cea47cdb70b031",
            "offset":OFFSET,"replicates":list(REPS),
            "degrees":list(DEGREES),"families":list(FAMILIES),
            "public_only":True,"planted_and_true_future_absent":True,
            "attack_and_external_witness_absent":True,
            "prior_source_count":60,
            "prior_source_ids_sha256":sha(canonical(before)),
            "cases":cases}


def read_fixture(path):
    raw=Path(path).read_bytes()
    obj=json.loads(raw)
    past=past_source_digests()
    assert obj["schema"]==FIXTURE_SCHEMA
    assert obj["registration_first_commit"]==(
        "9e14d1132744272c6c674e3799cea47cdb70b031")
    assert obj["offset"]==OFFSET and obj["replicates"]==list(REPS)
    assert obj["public_only"] and obj["planted_and_true_future_absent"]
    assert obj["attack_and_external_witness_absent"]
    assert obj["prior_source_count"]==len(past)==60
    assert obj["prior_source_ids_sha256"]==sha(canonical(past))
    assert len(obj["cases"])==24
    ids=[c["full_public_input_sha256"] for c in obj["cases"]]
    assert len(set(ids))==24 and set(ids).isdisjoint(past)
    assert {(c["replicate"],c["degree"],c["family"]) for c in obj["cases"]}=={
        (rep,d,f) for rep in REPS for d in DEGREES for f in FAMILIES}
    assert all(c["split"]=="TEST" and
               not any(x in c for x in
                       ("program","future","target","attack","witness"))
               for c in obj["cases"])
    return obj,sha(raw)


def public_proposal(row):
    d=row["degree"]
    h=row["public_header"]
    header=corpus.Header(h["base"],h["delay"],h["spacing"],h["amplitude"])
    c=corpus.Case("nor33-public",row["family"],
                  tuple((tuple(x),tuple(y)) for x,y in row["demos"]),
                  tuple(row["public_query_input"]),(),None)
    assert c.program is None and c.future==()
    assert nor21.public_digest(c,d)==row["full_public_input_sha256"]
    choice=nor21.certify(c,header,d,row["query_degree"],"stream16")
    assert choice["mask"]==row["native_public_mask"]>0
    assert choice["unanimous"] and choice["chosen_class"] is not None
    predicted=corpus.execute(c.query,nor21.program(header,choice["chosen_class"]))
    return {"pred":predicted,"header":header,"mask":choice["mask"],
            "chosen_class":choice["chosen_class"],
            "future_sha256":nor24.sha_target(predicted)}


def private_truth_postchoice(row,predicted):
    rep,d,fam=row["replicate"],row["degree"],row["family"]
    rng=random.Random(source_seed(rep,d,fam))
    for _ in range(row["selected_attempt"]+1):
        planted=nor21.new_header_program(rng)
        demos=tuple((src:=nor21.public_source(rng,fam,3*d//2),
                     corpus.execute(src,planted)) for _ in range(2))
        query=nor21.public_source(rng,fam,3*d//2)
    assert row["demos"]==[[list(src[:d+1]),list(y[:d+1])] for src,y in demos]
    assert row["public_query_input"]==list(query)
    truth=corpus.execute(query,planted)
    assert truth==predicted
    return truth


def risk_union(old):
    u=set(old["tail"])|set(old["magnitude"])|set(old["lag"])
    m=len(old["uniform"])
    assert 96<=len(u)<=m-96
    return frozenset(u)


def full_rankings(pred,header,digest,old):
    m=len(pred)
    assert m in (769,1537)
    risk=nor24.risk_scores(old,nor25.PRIOR,m)
    offset=int(digest[:16],16)%m
    curvature=lambda i:abs(pred[(i-1)%m]-2*pred[i]+pred[(i+1)%m])
    rankings={
        "central":sorted(range(m),key=lambda i:(abs(2*i-(m-1)),i)),
        "stride17":[(offset+17*i)%m for i in range(m)],
        "low_abs":sorted(range(m),key=lambda i:(abs(pred[i]),i)),
        "curvature":sorted(range(m),key=lambda i:(-curvature(i),i)),
        "anti_prior":sorted(range(m),key=lambda i:(risk[i],i)),
        "hash":sorted(range(m),key=lambda i:(
            sha256(f"NOR33|{digest}|{i}".encode("ascii")).digest(),i)),
        "stratified_random":sorted(range(m),key=lambda i:(
            sha256(f"NOR33-MATCHED|{digest}|{i}".encode("ascii")).digest(),i)),
    }
    assert tuple(rankings)==SUPPORT_FAMILIES
    assert all(len(v)==m and len(set(v))==m for v in rankings.values())
    return rankings


def forced_support(ranking,u,m,stratum):
    in_needed=24*stratum
    out_needed=96-in_needed
    inside=[]
    outside=[]
    for i in ranking:
        if i in u:
            if len(inside)<in_needed:inside.append(i)
        elif len(outside)<out_needed:
            outside.append(i)
        if len(inside)==in_needed and len(outside)==out_needed:break
    assert len(inside)==in_needed and len(outside)==out_needed
    assert len(set(inside+outside))==96
    return frozenset(inside+outside)


def blind_attack(digest,family,z,support,u,rep):
    assert family in SUPPORT_FAMILIES and z in STRATA
    assert 0<=rep<ATTACKS
    inside=sorted(support&u)
    outside=sorted(support-u)
    assert len(inside)==24*z and len(outside)==96-24*z
    rng=random.Random(seed(ATTACK_STREAM,digest,family,z,rep))
    attack=frozenset(rng.sample(inside,z)+rng.sample(outside,R-z))
    assert len(attack)==R and len(attack&u)==z
    return tuple(sorted(attack))


def audit_plans(old,digest):
    tag=sha256(f"{AUDIT_STREAM}|nor33|{digest}".encode("ascii")).hexdigest()
    result={arm:[] for arm in ARMS}
    for trial in range(TRIALS):
        original=nor25.policy_schedules(old,tag,trial)
        for arm in ARMS:
            checked=tuple(original[(arm,K)])
            assert len(checked)==len(set(checked))==K
            result[arm].append(checked)
    return result


def mixed_choice(digest,trial):
    idx=random.Random(seed(MIXED_STREAM,digest,trial)).randrange(16)
    acc=0
    for arm,w in zip(ARMS,WEIGHTS):
        acc+=w
        if idx<acc:return arm
    raise AssertionError("bad frozen auditor weights")


def ideal_floor(m,arm):
    if arm=="uniform":
        return Fraction(1)-Fraction(comb(m-R,K),comb(m,K))
    if arm.startswith("hybrid_"):
        reserve={"hybrid_25":4,"hybrid_50":8,"hybrid_75":12}[arm]
        targeted=K-reserve
        return Fraction(1)-Fraction(
            comb(m-targeted-R,reserve),comb(m-targeted,reserve))
    return Fraction(0)


def conditional_prob_lookup():
    probabilities={}
    for z in STRATA:
        a=24*z
        b=96-a
        dena=comb(a,z)
        denb=comb(b,R-z)
        for ti in range(17):
            for to in range(17):
                # These entries are used only when ti<=a and to<=b.
                if ti<=a and to<=b:
                    probabilities[(z,ti,to)]=1-(
                        comb(a-ti,z)/dena)*(comb(b-to,R-z)/denb)
    return probabilities


def evaluate(fixture_path):
    fixture,digest=read_fixture(fixture_path)
    # Freeze public choices for every new source before reconstructing truth.
    proposed={row["full_public_input_sha256"]:public_proposal(row)
              for row in fixture["cases"]}
    assert len(proposed)==24
    table=conditional_prob_lookup()
    records=[]
    for row in fixture["cases"]:
        pub=row["full_public_input_sha256"]
        p=proposed[pub]
        pred=p["pred"]
        old=nor24.supports(pred,p["header"])
        u=risk_union(old)
        ranking=full_rankings(pred,p["header"],pub,old)
        supports={(f,z):forced_support(ranking[f],u,len(pred),z)
                  for f in SUPPORT_FAMILIES for z in STRATA}
        attacks={(f,z):[blind_attack(pub,f,z,supports[(f,z)],u,j)
                        for j in range(ATTACKS)]
                 for f in SUPPORT_FAMILIES for z in STRATA}
        audits=audit_plans(old,pub)
        chosen=[mixed_choice(pub,t) for t in range(TRIALS)]
        # Private target access starts here; no policy depends on it.
        truth=private_truth_postchoice(row,pred)
        assert nor24.sha_target(truth)==p["future_sha256"]
        masks={arm:[sum(1<<i for i in plan) for plan in plans]
               for arm,plans in audits.items()}
        rows=[]
        valid_errors=informed_hits=direct_checks=0
        for arm in ARMS:
            for t,plan in enumerate(audits[arm]):
                if not nor24.exact_witness_accepts(pred,truth,plan,frozenset(),1):
                    valid_errors+=1
                informed=nor24.informed_attack(plan,old["uniform"],R)
                informed_hits+=int(bool(set(plan)&informed))
        for family in SUPPORT_FAMILIES:
            for z in STRATA:
                S=supports[(family,z)]
                U_part=S&u
                O_part=S-u
                asets=attacks[(family,z)]
                assert all(len(set(a))==R and len(set(a)&u)==z for a in asets)
                atk_masks=[sum(1<<i for i in a) for a in asets]
                observed={}
                conditional={}
                for arm in ARMS:
                    plans=audits[arm]
                    count=sum(bool(planmask&attackmask)
                              for planmask in masks[arm]
                              for attackmask in atk_masks)
                    observed[arm]=int(count)
                    probs=[table[(z,len(set(plan)&U_part),
                                     len(set(plan)&O_part))] for plan in plans]
                    conditional[arm]=statistics.mean(probs)
                    # Direct honest target exact integer comparison checks on
                    # four preselected trial/attack pairs, both delta values.
                    for t in range(2):
                        for idx in range(2):
                            checked=plans[t]
                            a=frozenset(asets[idx])
                            detected=bool(masks[arm][t]&atk_masks[idx])
                            yes1=not nor24.exact_witness_accepts(
                                pred,truth,checked,a,DELTAS[0])
                            yes2=not nor24.exact_witness_accepts(
                                pred,truth,checked,a,DELTAS[1])
                            assert yes1==yes2==detected
                            direct_checks+=1
                mix_count=sum(w*observed[arm]
                              for arm,w in zip(ARMS,WEIGHTS))
                mixrate=mix_count/(16*ATTACKS*TRIALS)
                hybridrate=observed["hybrid_50"]/(ATTACKS*TRIALS)
                baseline=observed["uniform"]/(ATTACKS*TRIALS)
                realized=0
                for t,arm in enumerate(chosen):
                    pm=masks[arm][t]
                    realized+=sum(bool(pm&am) for am in atk_masks)
                rows.append({
                    "family":family,"stratum":z,
                    "support_public_sha256":sha(canonical(sorted(S))),
                    "positions_in_historical_union":len(U_part),
                    "positions_outside_historical_union":len(O_part),
                    "attack_positions":[list(a) for a in asets],
                    "pure_arm_detected_out_of_4096":observed,
                    "conditional_exact_mean_detection":conditional,
                    "mix_expected_numerator":mix_count,
                    "mix_expected_denominator":16*ATTACKS*TRIALS,
                    "mixed_expected_rate":mixrate,
                    "hybrid50_rate":hybridrate,
                    "uniform_rate":baseline,
                    "mixed_realized_detected_out_of_4096":realized,
                    "mix_minus_hybrid50":mixrate-hybridrate,
                })
        assert len(rows)==len(SUPPORT_FAMILIES)*len(STRATA)==35
        floor=sum(Fraction(w,16)*ideal_floor(len(pred),arm)
                  for arm,w in zip(ARMS,WEIGHTS))
        assert floor>0 and valid_errors==informed_hits==0
        records.append({
            "public_sha256":pub,"replicate":row["replicate"],
            "degree":row["degree"],"family":row["family"],
            "future_length":len(pred),"native_mask":p["mask"],
            "chosen_class":p["chosen_class"],
            "public_prediction_sha256":p["future_sha256"],
            "private_true_future_sha256_postchoice":nor24.sha_target(truth),
            "historical_risk_union_size":len(u),
            "historical_risk_union_sha256":sha(canonical(sorted(u))),
            "evaluation_plan_sha256":sha(canonical(audits)),
            "mixture_realized_arm_counts":dict(Counter(chosen)),
            "cell_records":rows,
            "honest_false_rejections":valid_errors,
            "fully_informed_attack_detections":informed_hits,
            "direct_exact_witness_pairs":direct_checks,
            "delta_decision_disagreements":0,
            "ideal_mixture_floor_fraction":[floor.numerator,floor.denominator]})
    pairs=[]
    for rec in records:
        by={(r["family"],r["stratum"]):r for r in rec["cell_records"]}
        diffs={}
        for z in STRATA:
            diffs[str(z)]=statistics.mean(
                by[(f,z)]["mix_minus_hybrid50"] for f in SUPPORT_FAMILIES)
        residual=(
            -by[("curvature",2)]["mix_minus_hybrid50"]
            +by[("stratified_random",2)]["mix_minus_hybrid50"])
        pairs.append({
            "public_sha256":rec["public_sha256"],
            "degree":rec["degree"],"family":rec["family"],
            "mean_family_paired_diff_by_stratum":diffs,
            "low_overlap_mix_advantage":diffs["0"],
            "high_overlap_mix_advantage":diffs["4"],
            "signed_gradient":diffs["0"]-diffs["4"],
            "curvature_minus_random_hybrid_advantage_at_z2":residual,
        })
    wgradient=sum(x["signed_gradient"]>0 for x in pairs)
    wlow=sum(x["low_overlap_mix_advantage"]>0 for x in pairs)
    whigh=sum(x["high_overlap_mix_advantage"]<0 for x in pairs)
    wresidual=sum(x["curvature_minus_random_hybrid_advantage_at_z2"]>0 for x in pairs)
    means={
        "gradient":statistics.mean(x["signed_gradient"] for x in pairs),
        "low":statistics.mean(x["low_overlap_mix_advantage"] for x in pairs),
        "high":statistics.mean(x["high_overlap_mix_advantage"] for x in pairs),
        "residual":statistics.mean(
            x["curvature_minus_random_hybrid_advantage_at_z2"] for x in pairs)}
    exact=all(
        len(rec["cell_records"])==35 and
        rec["public_prediction_sha256"]==rec["private_true_future_sha256_postchoice"] and
        rec["honest_false_rejections"]==0 and
        rec["fully_informed_attack_detections"]==0 and
        rec["delta_decision_disagreements"]==0 and
        rec["direct_exact_witness_pairs"]==35*6*4 and
        Fraction(*rec["ideal_mixture_floor_fraction"])>0
        for rec in records)
    gates={
        "positive_overlap_gradient":wgradient>=18 and means["gradient"]>=.015,
        "mix_advantage_at_zero_overlap":wlow>=18 and means["low"]>=.002,
        "hybrid_advantage_at_full_overlap":whigh>=18 and means["high"]<=-.005,
        "curvature_residual_after_exact_matching":wresidual>=16 and means["residual"]>=.005,
        "frozen_sources_witness_and_independent_replay":exact}
    family_means={}
    for f in SUPPORT_FAMILIES:
        family_means[f]={str(z):{
            "mix":statistics.mean(next(c for c in rec["cell_records"]
                if c["family"]==f and c["stratum"]==z)["mixed_expected_rate"]
                for rec in records),
            "hybrid50":statistics.mean(next(c for c in rec["cell_records"]
                if c["family"]==f and c["stratum"]==z)["hybrid50_rate"]
                for rec in records),
            "uniform":statistics.mean(next(c for c in rec["cell_records"]
                if c["family"]==f and c["stratum"]==z)["uniform_rate"]
                for rec in records),
        } for z in STRATA}
    return {
        "schema":SCHEMA,"public_fixture_sha256":digest,
        "registered_first_commit":"9e14d1132744272c6c674e3799cea47cdb70b031",
        "fresh_public_source_count":len(records),
        "frozen_nor31_weights":list(WEIGHTS),
        "support_families":list(SUPPORT_FAMILIES),
        "overlap_strata_z":list(STRATA),
        "support_size":SUPPORT_SIZE,
        "changed_coordinates":R,"inspected_coordinates":K,
        "attack_restarts_per_cell":ATTACKS,
        "challenges_per_source_arm":TRIALS,
        "nonzero_shifts":list(DELTAS),
        "public_proposals_frozen_before_private_target":True,
        "attack_sets_fixed_before_audit_plans":True,
        "honest_external_exact_value_witness_assumed":True,
        "source_records":records,
        "source_level_primary":pairs,
        "source_wins":{"gradient":wgradient,"low":wlow,
                       "high":whigh,"curvature_residual":wresidual},
        "equal_source_means":means,
        "by_family_stratum":family_means,
        "registered_five_gates":gates,
        "registered_primary_pass":all(gates.values()),
        "scope":("Source-public forced size-96 supports with exactly z of 4 "
                 "altered coordinates inside the historical risk union. "
                 "The attack distribution is conditional and differs from "
                 "NOR32; output witness is an assumed extra honest oracle. "
                 "Public PRNG streams are reproducible, not cryptographic secrecy.")}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--make-fixture",type=Path)
    parser.add_argument("--fixture",type=Path)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.make_fixture:
        if args.fixture or args.out:parser.error("source creation and grading are separate")
        raw=canonical(make_fixture())
        args.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        args.make_fixture.write_bytes(raw)
        print("NOR33 FIRST PUBLIC-ONLY FIXTURE",len(raw),sha(raw))
        return
    if not args.fixture:parser.error("--fixture required")
    data=evaluate(args.fixture)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(canonical(data))
    print("NOR33 EXACT-OVERLAP PRIMARY",data["registered_primary_pass"],
          data["registered_five_gates"])
    print("NOR33 SOURCE WINS",data["source_wins"])
    print("NOR33 SOURCE MEANS",data["equal_source_means"])
    print("NOR33 STRATIFIED FAMILY RATES",data["by_family_stratum"])


if __name__=="__main__":
    main()
