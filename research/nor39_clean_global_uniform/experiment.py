"""NOR-38: prospectively registered whole-output uniform reserve vs policy-aware greed.

First scientific registration commit 4a5f3750d240d46c031e0cb432fb7b1a884cfbe9
precedes ALL new NOR-38 public inputs, attacks, CAL/EVAL audit plans and grades.
The source-only fixture must be persisted before the grade is created.
"""
from __future__ import annotations
import argparse
import json
import random
import statistics
import sys
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from math import comb
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import experiment as nor21
from research.nor24_adaptive_audits import experiment as nor24
from research.nor25_robust_audits import experiment as nor25
from research.nor34_eight_atom_curvature import experiment as nor34
from research.nor35_intraatom_curvature import experiment as nor35
from research.nor37_stochastic_curvature import experiment as nor37
from research.nor38_global_uniform import experiment as nor38

REGISTERED="4a5f3750d240d46c031e0cb432fb7b1a884cfbe9"
FIXTURE_SCHEMA="nor39.clean_global_uniform_public_only_reps68to75.v1"
RESULT_SCHEMA="nor39.clean_global_uniform_policyaware.v1"
PREV_SOURCE_SHA="37fa2aeedeb5ab54bc1082ec2d50f8a084c4f4a7d6279acd5d122396d760e504"
PREV_RESULT_SHA="9c69d663d4d1511e664e930c0f8ed42d8d6f59c4d10557323fb5ee54478c036d"
REPS=tuple(range(68,76))
DEGREES=(512,1024)
FAMILIES=nor21.FAMILIES
OFFSET=130039000
ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
MODES=("original","atom_random","atom_curvature50","stochastic50")
POLICIES=("uniform16","original_mix","deterministic_mix","stochastic50_mix",
          "atom_random_h50","global7_mix","global8_mix")
CONTROLS=("top_curvature","bottom_curvature","eight_atom_matched",
          "full_uniform","low_cal_exposure")
K,R,CAL,EVAL,ATTACKS=16,4,256,512,8
S_CAL=202611390071
S_EVAL=202611390072
S_ATTACK=202611390073
S_BLIND=202611390074
S_BRANCH=202611390075
S_TRANSFORM=202611390076
DELTAS=(1,4294049777)
DENOM=256


def canonical(obj):
    return (json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")


def digest(buf):
    return sha256(buf).hexdigest()


def seed(*args):
    return int.from_bytes(sha256("|".join(map(str,args)).encode("ascii")).digest(),"big")


def history():
    historic=nor38.history()
    archive=ROOT/"research/nor38_global_uniform/results/nor38_original_first_hosted.zip"
    with ZipFile(archive) as z:
        fixture=z.read("nor38_fixture.json")
        result=z.read("nor38_full.json")
    assert len(fixture)==9319507 and digest(fixture)==PREV_SOURCE_SHA
    assert len(result)==929756 and digest(result)==PREV_RESULT_SHA
    saved=json.loads(fixture)
    assert saved["schema"]=="nor38.global_uniform_public_only_reps60to67.v1"
    assert len(saved["cases"])==48
    all_ids=sorted(historic+[r["full_public_input_sha256"] for r in saved["cases"]])
    assert len(all_ids)==len(set(all_ids))==228
    return all_ids

def source_seed(rep,d,fam):
    assert rep in REPS and d in DEGREES and fam in FAMILIES
    return (corpus.SEED+OFFSET+1000003*rep+
            100009*DEGREES.index(d)+10007*FAMILIES.index(fam))


def new_source(rep,d,fam,seen):
    q=3*d//2
    rng=random.Random(source_seed(rep,d,fam))
    for attempt in range(100):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,fam,q),
                     corpus.execute(x,planted)) for _ in range(2))
        query=nor21.public_source(rng,fam,q)
        case=corpus.Case("nor39-source",fam,demos,query,(),None)
        headers=corpus.possible_headers(case)
        if len(headers)!=1:continue
        h=headers[0]
        if (h.base,h.delay,h.spacing,h.amplitude)!=planted[:4]:continue
        public_sha=nor21.public_digest(case,d)
        if public_sha in seen:continue
        mask=nor21.native_mask(case,h,d)
        if not mask:continue
        candidates=nor21.candidate_predictions(query,h,q)
        if len({candidates[i] for i in range(16) if mask&(1<<i)})!=1:continue
        true_class=4*corpus.OPS.index(planted[4])+corpus.OPS.index(planted[5])
        assert mask&(1<<true_class)
        seen.add(public_sha)
        return {"split":"TEST","replicate":rep,"degree":d,"query_degree":q,
                "family":fam,"source_seed":source_seed(rep,d,fam),
                "selected_attempt":attempt,"full_public_input_sha256":public_sha,
                "public_header":{"base":h.base,"delay":h.delay,
                                 "spacing":h.spacing,"amplitude":h.amplitude},
                "demos":[[list(x[:d+1]),list(y[:d+1])] for x,y in demos],
                "public_query_input":list(query),
                "original_native_public_mask":mask}
    raise RuntimeError("NOR39 source exhausted 100 admissible attempts")


def make_fixture():
    old=history()
    seen=set(old)
    rows=[new_source(rep,d,fam,seen)
          for rep in REPS for d in DEGREES for fam in FAMILIES]
    assert len(rows)==48 and len(seen)==276
    return {"schema":FIXTURE_SCHEMA,"registered_first_commit":REGISTERED,
            "replicates":list(REPS),"degrees":list(DEGREES),
            "source_offset":OFFSET,"families":list(FAMILIES),
            "public_only":True,"planted_and_private_target_absent":True,
            "attack_witness_and_audit_absent":True,"excluded_historical_sources":228,
            "prior_source_ids_sha256":digest(canonical(old)),"cases":rows}


def read_fixture(path):
    raw=Path(path).read_bytes()
    x=json.loads(raw)
    hist=history()
    assert x["schema"]==FIXTURE_SCHEMA and x["registered_first_commit"]==REGISTERED
    assert x["replicates"]==list(REPS) and x["degrees"]==list(DEGREES)
    assert x["source_offset"]==OFFSET and x["families"]==list(FAMILIES)
    assert x["public_only"] and x["planted_and_private_target_absent"]
    assert x["attack_witness_and_audit_absent"] and x["excluded_historical_sources"]==len(hist)==228
    assert x["prior_source_ids_sha256"]==digest(canonical(hist))
    rows=x["cases"]
    sources=[row["full_public_input_sha256"] for row in rows]
    assert len(sources)==len(set(sources))==48 and set(sources).isdisjoint(hist)
    assert {(r["replicate"],r["degree"],r["family"]) for r in rows}=={
        (rep,d,fam) for rep in REPS for d in DEGREES for fam in FAMILIES}
    assert all(r["split"]=="TEST" and
               not any(v in r for v in ("program","future","target","attack","witness","audit"))
               for r in rows)
    return x,digest(raw)


def restore_native_target(row,pred):
    rep,d,fam=row["replicate"],row["degree"],row["family"]
    rng=random.Random(source_seed(rep,d,fam))
    for _ in range(row["selected_attempt"]+1):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,fam,3*d//2),
                     corpus.execute(x,planted)) for _ in range(2))
        query=nor21.public_source(rng,fam,3*d//2)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    truth=corpus.execute(query,planted)
    assert truth==pred
    return truth


def phase_tag(src,phase):
    tok={"cal":S_CAL,"eval":S_EVAL}[phase]
    return sha256(f"{tok}|nor39|{phase}|{src}".encode("ascii")).hexdigest()


def components(policy):
    assert policy in POLICIES
    singles={"uniform16":("original","uniform"),
             "atom_random_h50":("atom_random","hybrid_50")}
    if policy in singles:
        mode,arm=singles[policy]
        return ((mode,arm,DENOM),)
    if policy in ("original_mix","deterministic_mix","stochastic50_mix"):
        mode={"original_mix":"original","deterministic_mix":"atom_curvature50",
              "stochastic50_mix":"stochastic50"}[policy]
        return tuple((mode,arm,16*w) for arm,w in zip(ARMS,WEIGHTS) if w)
    assert policy in ("global7_mix","global8_mix")
    q=7 if policy=="global7_mix" else 8
    return (("original","uniform",16*q),)+tuple(
        ("atom_curvature50",arm,(16-q)*w) for arm,w in zip(ARMS,WEIGHTS) if w)


def schedules(old,cats,scores,src,phase):
    n=CAL if phase=="cal" else EVAL
    root=phase_tag(src,phase)
    transform_key=sha256(f"{S_TRANSFORM}|{root}".encode("ascii")).hexdigest()
    result={mode:{arm:[] for arm in ARMS} for mode in MODES}
    behavior=Counter()
    for t in range(n):
        raw=nor25.policy_schedules(old,root,t)
        for arm in ARMS:
            base=tuple(raw[(arm,K)])
            assert len(base)==len(set(base))==K
            result["original"][arm].append(base)
            for mode in ("atom_random","atom_curvature50"):
                p=nor35.transformed_plan(base,cats,scores,transform_key,arm,t,mode)
                assert len(p)==len(set(p))==K
                result[mode][arm].append(p)
            p,diagnostic=nor37.new_stochastic_plan(base,cats,scores,root,src,arm,t,"stochastic50")
            behavior.update(diagnostic)
            result["stochastic50"][arm].append(p)
    return result,dict(behavior)


def trial_exposure(plans,m):
    exposures={}
    for mode,arms in plans.items():
        for arm,allplans in arms.items():
            bits=[0]*m
            for t,checked in enumerate(allplans):
                mark=1<<t
                for i in checked:bits[i]|=mark
            exposures[(mode,arm)]=bits
    return exposures


def greedy_attack(cov,policy,src,restart):
    parts=components(policy)
    m=len(cov[parts[0][:2]])
    candidates=list(range(m))
    random.Random(seed(S_ATTACK,src,policy,restart)).shuffle(candidates)
    chosen=set()
    seen={p[:2]:0 for p in parts}
    for _ in range(R):
        best=None
        value=None
        for i in candidates:
            if i in chosen:continue
            cost=sum(w*(seen[(mode,arm)]|cov[(mode,arm)][i]).bit_count()
                     for mode,arm,w in parts)
            if value is None or cost<value:
                best,value=i,cost
        assert best is not None
        chosen.add(best)
        for mode,arm,_ in parts:seen[(mode,arm)]|=cov[(mode,arm)][best]
    assert len(chosen)==R
    return tuple(sorted(chosen))


def blind_attacks(src,cats,u,cu,cv,bu,bv,m,cal_exposure):
    out={name:[] for name in CONTROLS}
    low_score=[sum(w*cal_exposure[(mode,arm)][i].bit_count()
                   for mode,arm,w in components("deterministic_mix"))
               for i in range(m)]
    low_pool=sorted(range(m),key=lambda i:(low_score[i],i))[:min(96,m)]
    assert len(low_pool)>=R
    for rep in range(ATTACKS):
        rng=random.Random(seed(S_BLIND,src,"top_curvature",rep))
        top=tuple(sorted(rng.sample(sorted(cu),2)+rng.sample(sorted(cv),2)))
        rng=random.Random(seed(S_BLIND,src,"bottom_curvature",rep))
        bottom=tuple(sorted(rng.sample(sorted(bu),2)+rng.sample(sorted(bv),2)))
        quota=[sum(i in c for i in top) for c in cats]
        assert sum(quota)==R
        rng=random.Random(seed(S_BLIND,src,"eight_atom_matched",rep))
        matched=tuple(sorted(i for j,q in enumerate(quota)
                             for i in rng.sample(sorted(cats[j]),q)))
        rng=random.Random(seed(S_BLIND,src,"full_uniform",rep))
        uniform=tuple(sorted(rng.sample(range(m),R)))
        rng=random.Random(seed(S_BLIND,src,"low_cal_exposure",rep))
        low=tuple(sorted(rng.sample(low_pool,R)))
        for name,attack in zip(CONTROLS,(top,bottom,matched,uniform,low)):
            assert len(attack)==len(set(attack))==R
            out[name].append(attack)
        assert [sum(i in c for i in matched) for c in cats]==quota
    return out


def realized_component(src,t,policy):
    assert policy in ("original_mix","deterministic_mix","stochastic50_mix",
                      "global7_mix","global8_mix")
    rng=random.Random(seed(S_BRANCH,"eval",src,t,policy))
    if policy in ("global7_mix","global8_mix"):
        q=7 if policy=="global7_mix" else 8
        if rng.randrange(16)<q:return ("original","uniform")
        mode="atom_curvature50"
    else:
        mode={"original_mix":"original","deterministic_mix":"atom_curvature50",
              "stochastic50_mix":"stochastic50"}[policy]
    pos=rng.randrange(16)
    for arm,w in zip(ARMS,WEIGHTS):
        if pos<w:return (mode,arm)
        pos-=w
    raise AssertionError("mixture quantization")


def exact_top(plans,cu,cv):
    out={}
    for policy in POLICIES:
        value=Fraction(0)
        for mode,arm,w in components(policy):
            for checked in plans[mode][arm]:
                value+=Fraction(w,DENOM)*nor35.two_block_detect(cu,cv,checked)
        out[policy]=value/EVAL
    return out


def exact_checks(plans,cats,cu,cv,m):
    uniform=nor35.ideal_uniform_attack_detect(m)
    assert uniform==Fraction(1)-Fraction(comb(m-K,R),comb(m,R))
    assert all(sum(w for _,_,w in components(p))==DENOM for p in POLICIES)
    fine=nor34.exact_probabilities(cats,cu,cv)
    bad_quota=bad_matched=bad_duplicate=0
    for arm in ARMS:
        for t in range(EVAL):
            base=plans["original"][arm][t]
            old_quota=tuple(len(set(base)&a) for a in cats)
            expected=fine(old_quota,len(set(base)&cu),len(set(base)&cv))["eight_atom"]
            for mode in ("atom_random","atom_curvature50","stochastic50"):
                p=plans[mode][arm][t]
                quotas=tuple(len(set(p)&a) for a in cats)
                bad_quota+=int(quotas!=old_quota)
                bad_matched+=int(fine(quotas,len(set(p)&cu),
                                      len(set(p)&cv))["eight_atom"]!=expected)
                bad_duplicate+=int(len(p)!=K or len(set(p))!=K)
            bad_duplicate+=int(len(base)!=K or len(set(base))!=K)
    # Global uniform support is the full m-domain, even when old union is smaller.
    assert 0<len(cu)<m and 0<len(cv)<m
    assert K<m
    assert all(Fraction(q,16)*uniform>0 for q in (7,8))
    assert bad_quota==bad_matched==bad_duplicate==0
    return {"quota_mismatches":bad_quota,"eight_atom_exact_mismatches":bad_matched,
            "duplicate_or_non16_plans":bad_duplicate,
            "uniform_exact_mismatch":0}


def measures(records):
    rows=[]
    for rec in records:
        d=rec["crossplay_weighted_numerators"]
        den=rec["crossplay_denominator"]
        top={key:Fraction(*val) for key,val in rec["exact_top_support_detection"].items()}
        w7=min(d[name]["global7_mix"]/den for name in POLICIES)
        wo=min(d[name]["original_mix"]/den for name in POLICIES)
        rows.append({
            "public_sha256":rec["public_sha256"],"replicate":rec["replicate"],
            "degree":rec["degree"],"family":rec["family"],
            "global7_self_minus_deterministic_self":
                (d["global7_mix"]["global7_mix"]-
                 d["deterministic_mix"]["deterministic_mix"])/den,
            "global7_worst_minus_original_worst":w7-wo,
            "global7_worst":w7,"original_worst":wo,
            "global7_top_minus_atom_random":float(
                top["global7_mix"]-top["atom_random_h50"]),
            "deterministic_top_minus_atom_random":float(
                top["deterministic_mix"]-top["atom_random_h50"]),
            "global8_self":d["global8_mix"]["global8_mix"]/den,
            "uniform_self":d["uniform16"]["uniform16"]/den,
            "global7_self":d["global7_mix"]["global7_mix"]/den,
            "original_mix_self":d["original_mix"]["original_mix"]/den,
        })
    return rows


def primary_gates(records,rows):
    keys=("global7_self_minus_deterministic_self",
          "global7_worst_minus_original_worst",
          "global7_top_minus_atom_random",
          "deterministic_top_minus_atom_random")
    means={k:statistics.mean(r[k] for r in rows) for k in keys}
    wins={
        "global7_self_minus_deterministic_self":sum(
            r["global7_self_minus_deterministic_self"]>0 for r in rows),
        "global7_worst_minus_original_worst_tolerance":sum(
            r["global7_worst_minus_original_worst"]>=-.010 for r in rows),
        "global7_top_minus_atom_random":sum(
            r["global7_top_minus_atom_random"]>0 for r in rows)}
    ratio=(means["global7_top_minus_atom_random"]/
           means["deterministic_top_minus_atom_random"]
           if means["deterministic_top_minus_atom_random"]>0 else None)
    gates={
        "global7_improves_curvature_mix_selfplay":(
            wins["global7_self_minus_deterministic_self"]>=36 and
            means["global7_self_minus_deterministic_self"]>=.010),
        "global7_competitive_vs_original_mix_worst":(
            wins["global7_worst_minus_original_worst_tolerance"]>=36 and
            means["global7_worst_minus_original_worst"]>=-.010 and
            statistics.mean(r["global7_worst"] for r in rows)>=.025),
        "global7_retains_top_curvature_signal":(
            wins["global7_top_minus_atom_random"]>=36 and
            means["global7_top_minus_atom_random"]>=.020 and
            ratio is not None and ratio>=.20)}
    full_good=all(
        r["native_mask"]>0 and r["public_prediction_sha256"]==
        r["private_native_truth_sha256_after_freeze"] and
        r["honest_target_false_rejections"]==0 and
        r["fully_schedule_informed_attacker_detections"]==0 and
        r["delta_decision_disagreements"]==0 and
        r["direct_authentic_witness_checks"]>0 and
        all(v==0 for v in r["exact_quota_and_control_checks"].values())
        and Fraction(*r["exact_full_uniform_attack_detection_fraction"])>0
        and Fraction(*r["global7_exact_theoretical_floor"])>0
        for r in records)
    gates["exact_global_floor_and_witness_integrity"]=full_good
    gates["source_archive_and_replay_integrity_contract"]=(
        full_good and len(records)==48 and
        len({r["public_sha256"] for r in records})==48)
    # Separate receipt_audit.py is the mandatory second implementation
    # for the final reproducibility claim. This initial grade retains
    # the frozen gate, then CI independently verifies it.
    return means,wins,ratio,gates


def evaluate(fixture_path):
    fixture,fixture_hash=read_fixture(fixture_path)
    proposed={r["full_public_input_sha256"]:nor37.public_proposal(r)
              for r in fixture["cases"]}
    assert len(proposed)==48
    records=[]
    for ix,row in enumerate(fixture["cases"]):
        src=row["full_public_input_sha256"]
        p=proposed[src]
        pred=p["pred"]
        old,cats,u,cu,cv,bu,bv,scores=nor37.source_geometry(
            pred,p["header"],src)
        cal,cal_diag=schedules(old,cats,scores,src,"cal")
        exposure=trial_exposure(cal,len(pred))
        learned={name:[greedy_attack(exposure,name,src,i)
                       for i in range(ATTACKS)] for name in POLICIES}
        blind=blind_attacks(src,cats,u,cu,cv,bu,bv,len(pred),exposure)
        attacks={**learned,**blind}
        evaluations,eval_diag=schedules(old,cats,scores,src,"eval")
        realized={name:[realized_component(src,t,name) for t in range(EVAL)]
                  for name in ("original_mix","deterministic_mix",
                               "stochastic50_mix","global7_mix","global8_mix")}
        # Attack sets and ALL plans frozen before reconstructing private truth.
        truth=restore_native_target(row,pred)
        assert p["sha"]==nor24.sha_target(truth)
        eval_exp=trial_exposure(evaluations,len(pred))
        cross={}
        for att,all_sets in attacks.items():
            cross[att]={}
            for policy in POLICIES:
                total=0
                for mode,arm,w in components(policy):
                    count=0
                    for attack in all_sets:
                        union=0
                        for i in attack:union|=eval_exp[(mode,arm)][i]
                        count+=union.bit_count()
                    total+=w*count
                cross[att][policy]=total
        realhits={}
        branchcounts={}
        for policy,which in realized.items():
            branchcounts[policy]={"uniform":sum(m=="original" and a=="uniform"
                                                for m,a in which),
                                  "curvature":sum(m=="atom_curvature50"
                                                  for m,a in which)}
        for att,attack_sets in attacks.items():
            realhits[att]={}
            for policy,choice in realized.items():
                realhits[att][policy]=sum(
                    bool(set(evaluations[m][arm][t])&set(a))
                    for t,(m,arm) in enumerate(choice) for a in attack_sets)
        bad_honest=bad_informed=direct_checks=0
        for mode,arms in evaluations.items():
            for arm,plans in arms.items():
                for plan in plans:
                    bad_honest+=int(not nor24.exact_witness_accepts(
                        pred,truth,plan,frozenset(),1))
                    bad_informed+=int(bool(set(plan)&nor24.informed_attack(
                        plan,old["uniform"],R)))
        assert bad_honest==bad_informed==0
        for att,attacksets in attacks.items():
            for policy in POLICIES:
                for attack in attacksets[:2]:
                    for mode,arm,w in components(policy):
                        for t in range(2):
                            plan=evaluations[mode][arm][t]
                            hit=bool(set(plan)&set(attack))
                            check1=not nor24.exact_witness_accepts(
                                pred,truth,plan,frozenset(attack),DELTAS[0])
                            check2=not nor24.exact_witness_accepts(
                                pred,truth,plan,frozenset(attack),DELTAS[1])
                            assert check1==check2==hit
                            direct_checks+=1
        checks=exact_checks(evaluations,cats,cu,cv,len(pred))
        uniform=nor35.ideal_uniform_attack_detect(len(pred))
        top=exact_top(evaluations,cu,cv)
        records.append({
            "public_sha256":src,"replicate":row["replicate"],
            "degree":row["degree"],"family":row["family"],
            "future_length":len(pred),"native_mask":p["mask"],
            "chosen_class":p["class"],
            "public_prediction_sha256":p["sha"],
            "private_native_truth_sha256_after_freeze":nor24.sha_target(truth),
            "original_atom_sizes":[len(c) for c in cats],
            "top_curvature_support_sha256":digest(canonical(sorted(cu|cv))),
            "bottom_curvature_support_sha256":digest(canonical(sorted(bu|bv))),
            "calibration_plans_sha256":digest(canonical(cal)),
            "evaluation_plans_sha256":digest(canonical(evaluations)),
            "calibration_stochastic_branch_counts":cal_diag,
            "evaluation_stochastic_branch_counts":eval_diag,
            "greedy_attack_sets":{key:[list(a) for a in vals]
                                  for key,vals in learned.items()},
            "blind_attack_sets":{key:[list(a) for a in vals]
                                 for key,vals in blind.items()},
            "greedy_attack_unique_sets":{key:len(set(vals))
                                         for key,vals in learned.items()},
            "crossplay_weighted_numerators":cross,
            "crossplay_denominator":DENOM*ATTACKS*EVAL,
            "realized_mixture_branch_counts":branchcounts,
            "realized_mixture_attack_hits":realhits,
            "exact_top_support_detection":{
                name:[v.numerator,v.denominator] for name,v in top.items()},
            "exact_full_uniform_attack_detection_fraction":[
                uniform.numerator,uniform.denominator],
            "global7_exact_theoretical_floor":[
                (Fraction(7,16)*uniform).numerator,
                (Fraction(7,16)*uniform).denominator],
            "global8_exact_theoretical_floor":[
                (Fraction(1,2)*uniform).numerator,
                (Fraction(1,2)*uniform).denominator],
            "honest_target_false_rejections":bad_honest,
            "fully_schedule_informed_attacker_detections":bad_informed,
            "delta_decision_disagreements":0,
            "exact_quota_and_control_checks":checks,
            "direct_authentic_witness_checks":direct_checks})
        print(f"NOR39 source {ix+1}/48 rep={row['replicate']} D={row['degree']} "
              f"family={row['family']}",flush=True)
    rows=measures(records)
    means,wins,ratio,gates=primary_gates(records,rows)
    replicate_means={str(rep):{
        key:statistics.mean(r[key] for r in rows if r["replicate"]==rep)
        for key in ("global7_self_minus_deterministic_self",
                    "global7_worst_minus_original_worst",
                    "global7_top_minus_atom_random")}
        for rep in REPS}
    return {
        "schema":RESULT_SCHEMA,"registration_first_commit":REGISTERED,
        "public_fixture_sha256":fixture_hash,
        "fresh_source_units":48,"historical_sources_excluded":228,
        "auditor_policies":list(POLICIES),"component_auditor_arms":list(ARMS),
        "original_mix_weights":list(WEIGHTS),"blind_control_families":list(CONTROLS),
        "transform_modes":list(MODES),"global_uniform_probabilities":{
            "global7_mix":[7,16],"global8_mix":[8,16]},
        "calibration_schedules_per_source":CAL,
        "heldout_schedules_per_source":EVAL,
        "greedy_restarts_per_policy":ATTACKS,
        "audit_budget":K,"attack_radius":R,"nonzero_deltas":list(DELTAS),
        "public_proposals_frozen_before_private_truth":True,
        "calibrated_attacks_frozen_before_independent_heldout_schedules":True,
        "authentic_external_target_coefficient_witness_assumed":True,
        "source_records":records,"source_level_primary":rows,
        "source_means":means,"source_direction_wins":wins,
        "retained_ratio_global7_to_deterministic_top_gain":ratio,
        "replicate_block_means":replicate_means,
        "registered_five_gates":gates,
        "registered_primary_pass":all(gates.values()),
        "scope":"Finite synthetic RRC25, public research SHA coins (not deployed entropy), "
                "authentic external target coefficient witness assumed; "
                "unrestricted four-coordinate maximin is uniform, greedy "
                "calibration is approximate; independent replay is mandatory."
    }


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--make-fixture",type=Path)
    a.add_argument("--fixture",type=Path)
    a.add_argument("--out",type=Path)
    z=a.parse_args()
    if z.make_fixture:
        if z.fixture or z.out:a.error("fixture creation and grading are separate")
        raw=canonical(make_fixture())
        z.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        z.make_fixture.write_bytes(raw)
        print("NOR39 FIRST PUBLIC-ONLY 48 SOURCES",len(raw),digest(raw),flush=True)
        return
    if not z.fixture:a.error("--fixture required")
    result=evaluate(z.fixture)
    if z.out:
        z.out.parent.mkdir(parents=True,exist_ok=True)
        z.out.write_bytes(canonical(result))
    print("NOR39 REGISTERED PRIMARY",result["registered_primary_pass"])
    print("NOR39 FIVE GATES",result["registered_five_gates"])
    print("NOR39 WINS",result["source_direction_wins"])
    print("NOR39 MEANS",result["source_means"])
    print("NOR39 TOP RETENTION",result["retained_ratio_global7_to_deterministic_top_gain"])


if __name__=="__main__":
    main()
