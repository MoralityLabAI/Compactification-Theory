"""NOR-41: prospective input-family OOD and CAL-budget transfer experiment.

First registration commit 87dd3de03f43e79d83d45ed7a5c3a3bb1f20746f
precedes ALL new NOR-41 public inputs, attacks, CAL/EVAL audit plans and grades.
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
from research.nor40_strong_adversaries import experiment as nor40

REGISTERED="87dd3de03f43e79d83d45ed7a5c3a3bb1f20746f"
AMENDMENT_01="e5a95cad1ffdcd019f153cd09dd4d48b40848ef9"
FIXTURE_SCHEMA="nor41.scale_ood_public_only_reps84to99.v1"
RESULT_SCHEMA="nor41.input_ood_nested_cal_transfer.v1"
PREV_SOURCE_SHA="b51e8031715754c36661b74cff0c26e672fdc21355ff0dc9ab48efded3901de5"
PREV_RESULT_SHA="42adeebcbf92626c080a6f0f5cb906eed491d6b5ed67250ba7b70bd6d0559092"
REPS=tuple(range(84,100))
DEGREES=(512,1024)
FAMILIES=("wide_dense_small","wide_regime_blocks")
OFFSET=130041000
ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
MODES=("original","atom_random","atom_curvature50")
POLICIES=("uniform16","original_mix","deterministic_mix","global7_mix")
DIAGNOSTIC="atom_random_h50"
CONTROLS=("top_curvature","bottom_curvature","eight_atom_matched",
          "full_uniform","low_cal_exposure","zero_quota_stress")
K,R,CAL,EVAL,ATTACKS=16,4,4096,1024,8
BUDGETS=(256,512,2048,4096)
GREEDY_STARTS=16
POOL_SIZE=16
SWAP_PASSES=4
S_CAL=202611410071
S_EVAL=202611410072
S_ATTACK=202611410073
S_BLIND=202611410074
S_BRANCH=202611410075
S_TRANSFORM=202611410076
DELTAS=(1,4294049777)
DENOM=256

def canonical(obj):
    return (json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")


def digest(buf):
    return sha256(buf).hexdigest()


def seed(*args):
    return int.from_bytes(sha256("|".join(map(str,args)).encode("ascii")).digest(),"big")


def history():
    old=nor40.history()
    with ZipFile(ROOT/"research/nor40_strong_adversaries/results/nor40_original_first_hosted.zip") as z:
        assert z.testzip() is None
        fixture=z.read("nor40_fixture.json")
        grade=z.read("nor40_full.json")
    assert len(fixture)==9387973 and digest(fixture)==PREV_SOURCE_SHA
    assert len(grade)==3492789 and digest(grade)==PREV_RESULT_SHA
    saved=json.loads(fixture)
    assert saved["schema"]=="nor40.strong_adversary_public_only_reps76to83.v1"
    assert len(saved["cases"])==48
    all_ids=sorted(old+[r["full_public_input_sha256"] for r in saved["cases"]])
    assert len(all_ids)==len(set(all_ids))==324
    return all_ids


def public_input(rng,fam,q):
    """Frozen new family grammar; no cohort identity is generated here."""
    if fam=="wide_dense_small":
        return nor21.public_source(rng,fam,q)
    assert fam=="wide_regime_blocks" and q>=512
    result=[1]
    coeffs=(-2,-1,1,2)
    for i in range(1,q+1):
        regime=((i-1)//64)%4
        draw=rng.random()
        if regime==0: use=draw<.90
        elif regime==1: use=draw<.04
        elif regime==2: use=draw<.16 and i%16<4
        else: use=draw<.25
        result.append(rng.choice(coeffs) if use else 0)
    return tuple(result)

def source_seed(rep,d,fam):
    assert rep in REPS and d in DEGREES and fam in FAMILIES
    return (corpus.SEED+OFFSET+1000003*rep+
            100009*DEGREES.index(d)+10007*FAMILIES.index(fam))


def new_source(rep,d,fam,seen):
    q=3*d//2
    rng=random.Random(source_seed(rep,d,fam))
    for attempt in range(100):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=public_input(rng,fam,q),
                     corpus.execute(x,planted)) for _ in range(2))
        query=public_input(rng,fam,q)
        case=corpus.Case("nor41-source",fam,demos,query,(),None)
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
    raise RuntimeError("NOR41 source exhausted 100 admissible attempts")


def make_fixture():
    old=history()
    seen=set(old)
    rows=[new_source(rep,d,fam,seen)
          for rep in REPS for d in DEGREES for fam in FAMILIES]
    assert len(rows)==64 and len(seen)==388
    return {"schema":FIXTURE_SCHEMA,"registered_first_commit":REGISTERED,
            "predata_amendment_01_commit":AMENDMENT_01,
            "replicates":list(REPS),"degrees":list(DEGREES),
            "source_offset":OFFSET,"families":list(FAMILIES),
            "public_only":True,"planted_and_private_target_absent":True,
            "attack_witness_and_audit_absent":True,"excluded_historical_sources":324,
            "prior_source_ids_sha256":digest(canonical(old)),"cases":rows}


def read_fixture(path):
    raw=Path(path).read_bytes()
    x=json.loads(raw)
    hist=history()
    assert x["schema"]==FIXTURE_SCHEMA and x["registered_first_commit"]==REGISTERED
    assert x["predata_amendment_01_commit"]==AMENDMENT_01
    assert x["replicates"]==list(REPS) and x["degrees"]==list(DEGREES)
    assert x["source_offset"]==OFFSET and x["families"]==list(FAMILIES)
    assert x["public_only"] and x["planted_and_private_target_absent"]
    assert x["attack_witness_and_audit_absent"] and x["excluded_historical_sources"]==len(hist)==324
    assert x["prior_source_ids_sha256"]==digest(canonical(hist))
    rows=x["cases"]
    sources=[row["full_public_input_sha256"] for row in rows]
    assert len(sources)==len(set(sources))==64 and set(sources).isdisjoint(hist)
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
        demos=tuple((x:=public_input(rng,fam,3*d//2),
                     corpus.execute(x,planted)) for _ in range(2))
        query=public_input(rng,fam,3*d//2)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    truth=corpus.execute(query,planted)
    assert truth==pred
    return truth


def phase_tag(src,phase):
    tok={"cal":S_CAL,"eval":S_EVAL}[phase]
    return sha256(f"{tok}|nor41|{phase}|{src}".encode("ascii")).hexdigest()


def components(policy):
    assert policy in POLICIES or policy==DIAGNOSTIC
    if policy=="uniform16":return (("original","uniform",256),)
    if policy==DIAGNOSTIC:return (("atom_random","hybrid_50",256),)
    if policy in ("original_mix","deterministic_mix"):
        mode="original" if policy=="original_mix" else "atom_curvature50"
        return tuple((mode,arm,16*w) for arm,w in zip(ARMS,WEIGHTS) if w)
    return (("original","uniform",112),)+tuple(
        ("atom_curvature50",arm,9*w) for arm,w in zip(ARMS,WEIGHTS) if w)


def schedules(old,cats,scores,src,phase):
    n=CAL if phase=="cal" else EVAL
    root=phase_tag(src,phase)
    key=sha256(f"{S_TRANSFORM}|{root}".encode("ascii")).hexdigest()
    plans={mode:{arm:[] for arm in ARMS} for mode in MODES}
    for t in range(n):
        raw=nor25.policy_schedules(old,root,t)
        for arm in ARMS:
            base=tuple(raw[(arm,K)])
            assert len(base)==len(set(base))==K
            plans["original"][arm].append(base)
            for mode in ("atom_random","atom_curvature50"):
                p=nor35.transformed_plan(base,cats,scores,key,arm,t,mode)
                assert len(p)==len(set(p))==K
                plans[mode][arm].append(p)
    return plans

def trial_exposure(plans,m,n=None):
    exposures={}
    for mode,arms in plans.items():
        for arm,allplans in arms.items():
            bits=[0]*m
            for t,checked in enumerate(allplans if n is None else allplans[:n]):
                mark=1<<t
                for i in checked:bits[i]|=mark
            exposures[(mode,arm)]=bits
    return exposures


def greedy_attack(cov,policy,src,restart,budget):
    parts=components(policy)
    m=len(cov[parts[0][:2]])
    candidates=list(range(m))
    random.Random(seed(S_ATTACK,src,policy,budget,restart)).shuffle(candidates)
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



def cal_objective(exposure,policy,attack):
    """Exact integer 256-weighted union count on CAL trial-ID bitsets."""
    assert len(attack)==R and len(set(attack))==R
    score=0
    for mode,arm,w in components(policy):
        bits=0
        for i in attack:
            bits|=exposure[(mode,arm)][i]
        score+=w*bits.bit_count()
    return score


def singletons(exposure,policy):
    return [sum(w*bits.bit_count() for mode,arm,w in components(policy)
                for bits in (exposure[(mode,arm)][i],))
            for i in range(len(exposure[components(policy)[0][:2]]))]


def refine_swap(exposure,policy,attack):
    """CAL-only 4-coordinate steepest descent, deterministic tie order."""
    best=tuple(sorted(attack))
    bestscore=cal_objective(exposure,policy,best)
    m=len(exposure[components(policy)[0][:2]])
    for _ in range(SWAP_PASSES):
        choice=best
        value=bestscore
        for drop in best:
            retained=tuple(j for j in best if j!=drop)
            retained_set=set(retained)
            # Hoist the three-coordinate union outside the replacement loop.
            branches=[]
            for mode,arm,w in components(policy):
                fixed=0
                for j in retained:fixed|=exposure[(mode,arm)][j]
                branches.append((fixed,exposure[(mode,arm)],w))
            for i in range(m):
                if i in retained_set:continue
                candidate=tuple(sorted((*retained,i)))
                score=sum(w*(fixed|arr[i]).bit_count() for fixed,arr,w in branches)
                if score<value or (score==value and score<bestscore
                                    and candidate<choice):
                    choice,value=candidate,score
        if value>=bestscore:break
        best,bestscore=choice,value
    return best


def nested_attacks(exposure,policy,src,budget):
    """Each budget sees a strictly nested prefix of the same CAL 4096 draws."""
    assert budget in BUDGETS
    subset={k:[v&((1<<budget)-1) for v in arr] for k,arr in exposure.items()}
    greedy=[greedy_attack(subset,policy,src,j,budget)
            for j in range(GREEDY_STARTS)]
    order=lambda a:(cal_objective(subset,policy,a),a)
    ranked=sorted(greedy,key=order)
    refined=[refine_swap(subset,policy,a) for a in ranked[:2]]
    scores=singletons(subset,policy)
    pool=list(min(greedy,key=order))
    for index in sorted(range(len(scores)),key=lambda k:(scores[k],k)):
        if index not in pool:pool.append(index)
        if len(pool)==POOL_SIZE:break
    pool=tuple(sorted(pool))
    assert len(pool)==POOL_SIZE and len(set(pool))==POOL_SIZE
    from itertools import combinations
    choices=[(cal_objective(subset,policy,a),a)
             for a in combinations(pool,R)]
    assert len(choices)==1820
    exact_score,exact_attack=min(choices)
    ranking=sorted(set(greedy+refined+[exact_attack]),key=order)
    greedy_sum=sum(cal_objective(subset,policy,a) for a in ranked[:ATTACKS])
    selected=None
    selected_sum=None
    chosen_unique=0
    for q in range(min(ATTACKS,len(ranking)),0,-1):
        proposal=(ranking[:q]+[ranking[0]]*(ATTACKS-q))
        val=sum(cal_objective(subset,policy,a) for a in proposal)
        if val<=greedy_sum:
            selected,selected_sum,chosen_unique=proposal,val,q
            break
    assert selected is not None and len(selected)==ATTACKS
    assert chosen_unique>=1 and selected_sum<=greedy_sum
    lower=sorted(scores)[R-1]
    upper=cal_objective(subset,policy,selected[0])
    assert 0<=lower<=upper<=exact_score
    return selected,{
        "budget":budget,
        "greedy16_sets":[list(a) for a in greedy],
        "swap_refined_top2_sets":[list(a) for a in refined],
        "restricted_candidate_pool":list(pool),
        "restricted_exact_optimum":list(exact_attack),
        "restricted_exact_value":exact_score,
        "restricted_sets_enumerated":1820,
        "global_singleton_lower_bound":lower,
        "global_best_found_upper_bound":upper,
        "global_optimality_gap":upper-lower,
        "greedy_best8_cal_sum":greedy_sum,
        "selected_best8_cal_sum":selected_sum,
        "selected_attack_uniques":len(set(selected)),
        "max_distinct_subject_to_cal_dominance":chosen_unique,
        "solver_scope":"exact_pool16; lower is global fourth singleton"
    }


def blind_attacks(src,cats,u,cu,cv,bu,bv,m,cal_exposure):
    out={name:[] for name in CONTROLS}
    low_score=[sum(w*cal_exposure[(mode,arm)][i].bit_count()
                   for mode,arm,w in components("deterministic_mix"))
               for i in range(m)]
    low_pool=sorted(range(m),key=lambda i:(low_score[i],i))[:min(96,m)]
    assert len(low_pool)>=R
    outside=sorted((i for i in range(m) if i not in u),
                   key=lambda i:(low_score[i],i))
    stress_pool=(outside if len(outside)>=R else
                 sorted(range(m),key=lambda i:(low_score[i],i)))[:min(96,m)]
    assert len(stress_pool)>=R
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
        rng=random.Random(seed(S_BLIND,src,"zero_quota_stress",rep))
        stress=tuple(sorted(rng.sample(stress_pool,R)))
        for name,attack in zip(CONTROLS,(top,bottom,matched,uniform,low,stress)):
            assert len(attack)==len(set(attack))==R
            out[name].append(attack)
        assert [sum(i in c for i in matched) for c in cats]==quota
    return out


def realized_component(src,t,policy):
    assert policy in ("original_mix","deterministic_mix","global7_mix")
    rng=random.Random(seed(S_BRANCH,"eval",src,t,policy))
    if policy=="global7_mix":
        q=7
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
    for policy in (*POLICIES,DIAGNOSTIC):
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
            for mode in ("atom_random","atom_curvature50"):
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
    for r in records:
        x=r["crossplay_weighted_numerators"]
        den=r["crossplay_denominator"]
        top={k:Fraction(*v) for k,v in r["exact_top_support_detection"].items()}
        t={p:x[f"cal4096__{p}"] for p in POLICIES}
        strong_g=t["global7_mix"]["global7_mix"]/den
        strong_det=t["deterministic_mix"]["deterministic_mix"]/den
        w7=min(t[p]["global7_mix"]/den for p in POLICIES)
        wo=min(t[p]["original_mix"]/den for p in POLICIES)
        gain=x["cal256__global7_mix"]["global7_mix"]/den-strong_g
        rows.append({
            "public_sha256":r["public_sha256"],
            "replicate":r["replicate"],"family":r["family"],
            "degree":r["degree"],"future_length":r["future_length"],
            "global7_256_minus_4096_eval":gain,
            "global7_strong_self_minus_deterministic":strong_g-strong_det,
            "global7_strong_worst_minus_original":w7-wo,
            "global7_strong_worst":w7,"original_strong_worst":wo,
            "global7_top_minus_atom_random":float(top["global7_mix"]-top[DIAGNOSTIC]),
            "deterministic_top_minus_atom_random":float(
                top["deterministic_mix"]-top[DIAGNOSTIC]),
            "uniform_strong_worst":min(t[p]["uniform16"]/den for p in POLICIES),
            "global7_budget_self":{
                str(n):x[f"cal{n}__global7_mix"]["global7_mix"]/den for n in BUDGETS},
            "original_native_mask_cardinality":r["native_mask"].bit_count()
        })
    return rows

def primary_gates(records,rows):
    def m(key,subset=rows):return statistics.mean(r[key] for r in subset)
    anchor=[r for r in rows if r["family"]=="wide_dense_small"]
    ood=[r for r in rows if r["family"]=="wide_regime_blocks"]
    assert len(rows)==64 and len(anchor)==len(ood)==32
    low=[r for r in rows if r["degree"]==512]
    high=[r for r in rows if r["degree"]==1024]
    assert len(low)==len(high)==32
    blocks={
        rep:statistics.mean(
            r["global7_256_minus_4096_eval"] for r in rows
            if r["replicate"]==rep and r["degree"]==512
        )-statistics.mean(
            r["global7_256_minus_4096_eval"] for r in rows
            if r["replicate"]==rep and r["degree"]==1024
        ) for rep in REPS}
    contrast=m("global7_256_minus_4096_eval",low)-m(
        "global7_256_minus_4096_eval",high)
    ood_base=m("deterministic_top_minus_atom_random",ood)
    ood_top=m("global7_top_minus_atom_random",ood)
    ratio=ood_top/ood_base if ood_base>0 else None
    means={
        "global7_budget_transfer":m("global7_256_minus_4096_eval"),
        "degree512_minus_degree1024_transfer":contrast,
        "ood_top_gain":ood_top,
        "ood_deterministic_top_gain":ood_base,
        "global7_strong_self_minus_deterministic":m(
            "global7_strong_self_minus_deterministic"),
        "global7_strong_worst_minus_original":m(
            "global7_strong_worst_minus_original"),
        "global7_strong_worst_absolute":m("global7_strong_worst")
    }
    wins={
        "positive_degree_scale_blocks":sum(v>0 for v in blocks.values()),
        "positive_ood_top_sources":sum(
            r["global7_top_minus_atom_random"]>0 for r in ood),
        "strong_global7_transfer_source_wins":sum(
            r["global7_256_minus_4096_eval"]>0 for r in rows),
        "strong_global7_self_advantage_source_wins":sum(
            r["global7_strong_self_minus_deterministic"]>0 for r in rows),
        "global7_worst_within_1pp_original":sum(
            r["global7_strong_worst_minus_original"]>=-.010 for r in rows)
    }
    cal_cert=all(
        all(
            rec["attack_certificates"][str(n)][p]["selected_best8_cal_sum"]<=
            rec["attack_certificates"][str(n)][p]["greedy_best8_cal_sum"]
            for n in BUDGETS for p in POLICIES)
        for rec in records)
    core=all(
        r["native_mask"]>0 and
        r["public_prediction_sha256"]==r["private_native_truth_sha256_after_freeze"]
        and r["honest_target_false_rejections"]==0
        and r["fully_schedule_informed_attacker_detections"]==0
        and r["delta_decision_disagreements"]==0
        and r["direct_authentic_witness_checks"]>0
        and all(value==0 for value in r["exact_quota_and_control_checks"].values())
        and Fraction(*r["exact_full_uniform_attack_detection_fraction"])>0
        and Fraction(*r["global7_exact_theoretical_floor"])>0
        and all(
            len(r["calibrated_attacks"][str(n)][p])==ATTACKS
            and all(len(set(a))==R for a in r["calibrated_attacks"][str(n)][p])
            and r["attack_certificates"][str(n)][p]["restricted_sets_enumerated"]==1820
            and r["attack_certificates"][str(n)][p]["global_singleton_lower_bound"]<=
                r["attack_certificates"][str(n)][p]["global_best_found_upper_bound"]
            and r["attack_certificates"][str(n)][p]["global_best_found_upper_bound"]<=
                r["attack_certificates"][str(n)][p]["restricted_exact_value"]
            and r["attack_certificates"][str(n)][p]["global_optimality_gap"]==
                r["attack_certificates"][str(n)][p]["global_best_found_upper_bound"]-
                r["attack_certificates"][str(n)][p]["global_singleton_lower_bound"]
            for n in BUDGETS for p in POLICIES)
        for r in records)
    gates={
        "scale_sensitive_cal_to_eval_transfer":(
            wins["positive_degree_scale_blocks"]>=11 and contrast>=.001),
        "novel_input_family_top_gain_retained":(
            wins["positive_ood_top_sources"]>=24 and ood_top>=.020
            and ratio is not None and ratio>=.20),
        "stronger_cal4096_evaluation_transfer":(
            wins["strong_global7_transfer_source_wins"]>=32 and
            means["global7_budget_transfer"]>=.001 and cal_cert),
        "global7_frontier_survives_4096_cal_attack":(
            wins["strong_global7_self_advantage_source_wins"]>=48 and
            means["global7_strong_self_minus_deterministic"]>=.010
            and wins["global7_worst_within_1pp_original"]>=48
            and means["global7_strong_worst_minus_original"]>=-.010
            and means["global7_strong_worst_absolute"]>=.020),
        "source_first_native_witness_and_full_replay_contract":(
            core and cal_cert and len(records)==64 and
            len({r["public_sha256"] for r in records})==64)
    }
    return means,wins,ratio,blocks,gates

def evaluate(fixture_path):
    fixture,fixture_sha=read_fixture(fixture_path)
    # Public proposals are computed from the committed source-only fixture.
    proposed={r["full_public_input_sha256"]:nor37.public_proposal(r)
              for r in fixture["cases"]}
    assert len(proposed)==64
    records=[]
    for index,row in enumerate(fixture["cases"]):
        src=row["full_public_input_sha256"]
        p=proposed[src]
        pred=p["pred"]
        old,cats,u,cu,cv,bu,bv,scores=nor37.source_geometry(
            pred,p["header"],src)
        cal=schedules(old,cats,scores,src,"cal")
        exp=trial_exposure(cal,len(pred))
        learned={}
        certs={}
        for budget in BUDGETS:
            attacks={}
            proofs={}
            for policy in POLICIES:
                a,c=nested_attacks(exp,policy,src,budget)
                attacks[policy]=a
                proofs[policy]=c
            learned[str(budget)]=attacks
            certs[str(budget)]=proofs
        blind=blind_attacks(src,cats,u,cu,cv,bu,bv,len(pred),exp)
        all_attacks={
            **{f"cal{n}__{p}":learned[str(n)][p]
               for n in BUDGETS for p in POLICIES},
            **blind
        }
        assert len(all_attacks)==22
        ev=schedules(old,cats,scores,src,"eval")
        # ALL CAL-only support and EVAL schedules now frozen.
        truth=restore_native_target(row,pred)
        assert nor24.sha_target(truth)==p["sha"]
        evexp=trial_exposure(ev,len(pred))
        cross={}
        for attacker,variants in all_attacks.items():
            cross[attacker]={}
            for policy in POLICIES:
                total=0
                for mode,arm,w in components(policy):
                    count=0
                    for A in variants:
                        union=0
                        for i in A:union|=evexp[(mode,arm)][i]
                        count+=union.bit_count()
                    total+=w*count
                cross[attacker][policy]=total
        realized={}
        for policy in ("original_mix","deterministic_mix","global7_mix"):
            selected=[realized_component(src,t,policy) for t in range(EVAL)]
            realized[policy]={
                "uniform":sum(mode=="original" and arm=="uniform"
                              for mode,arm in selected),
                "curvature":sum(mode=="atom_curvature50"
                               for mode,arm in selected)
            }
        bad_honest=bad_informed=direct=0
        for mode,arms in ev.items():
            for arm,plans in arms.items():
                for plan in plans:
                    bad_honest+=int(not nor24.exact_witness_accepts(
                        pred,truth,plan,frozenset(),1))
                    bad_informed+=int(bool(set(plan)&nor24.informed_attack(
                        plan,old["uniform"],R)))
        assert bad_honest==bad_informed==0
        for attacker,variants in all_attacks.items():
            for policy in POLICIES:
                for A in variants[:2]:
                    for mode,arm,w in components(policy):
                        for t in range(2):
                            plan=ev[mode][arm][t]
                            hit=bool(set(plan)&set(A))
                            check1=not nor24.exact_witness_accepts(
                                pred,truth,plan,frozenset(A),DELTAS[0])
                            check2=not nor24.exact_witness_accepts(
                                pred,truth,plan,frozenset(A),DELTAS[1])
                            assert check1==check2==hit
                            direct+=1
        check=exact_checks(ev,cats,cu,cv,len(pred))
        top=exact_top(ev,cu,cv)
        uniform=nor35.ideal_uniform_attack_detect(len(pred))
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
            "evaluation_plans_sha256":digest(canonical(ev)),
            "calibrated_attacks":{
                n:{p:[list(a) for a in aa] for p,aa in policies.items()}
                for n,policies in learned.items()},
            "attack_certificates":certs,
            "blind_attack_sets":{k:[list(a) for a in v]
                                 for k,v in blind.items()},
            "crossplay_weighted_numerators":cross,
            "crossplay_denominator":DENOM*ATTACKS*EVAL,
            "realized_mixture_branch_counts":realized,
            "exact_top_support_detection":{
                k:[v.numerator,v.denominator] for k,v in top.items()},
            "exact_full_uniform_attack_detection_fraction":[
                uniform.numerator,uniform.denominator],
            "global7_exact_theoretical_floor":[
                (Fraction(7,16)*uniform).numerator,
                (Fraction(7,16)*uniform).denominator],
            "honest_target_false_rejections":bad_honest,
            "fully_schedule_informed_attacker_detections":bad_informed,
            "delta_decision_disagreements":0,
            "exact_quota_and_control_checks":check,
            "direct_authentic_witness_checks":direct
        })
        print(f"NOR41 first source {index+1}/64 rep={row['replicate']} "
              f"D={row['degree']} family={row['family']}",flush=True)
    rows=measures(records)
    means,wins,ratio,blocks,gates=primary_gates(records,rows)
    return {
        "schema":RESULT_SCHEMA,"registration_first_commit":REGISTERED,
        "predata_amendment_01_commit":AMENDMENT_01,
        "public_fixture_sha256":fixture_sha,
        "fresh_source_units":64,"historical_sources_excluded":324,
        "replicates":list(REPS),"degrees":list(DEGREES),"families":list(FAMILIES),
        "auditor_policies":list(POLICIES),"diagnostic_auditor":DIAGNOSTIC,
        "component_arms":list(ARMS),"transform_modes":list(MODES),
        "original_mix_weights":list(WEIGHTS),"blind_control_families":list(CONTROLS),
        "global_uniform_probability":[7,16],"audit_budget":K,"attack_radius":R,
        "calibration_schedules_per_source":CAL,
        "calibration_prefix_budgets":list(BUDGETS),
        "heldout_schedules_per_source":EVAL,
        "greedy_starts_per_policy_budget":GREEDY_STARTS,
        "selected_attack_sets_per_policy_budget":ATTACKS,
        "exact_candidate_pool_size":POOL_SIZE,
        "exact_pool_combination_count":1820,
        "nonzero_deltas":list(DELTAS),
        "public_proposals_frozen_before_private_truth":True,
        "calibrated_attacks_frozen_before_independent_heldout_schedules":True,
        "authentic_external_target_coefficient_witness_assumed":True,
        "source_records":records,"source_level_primary":rows,
        "source_means":means,"source_direction_wins":wins,
        "retained_ood_top_gain_ratio":ratio,
        "degree_scale_replica_block_contrasts":{str(k):v for k,v in blocks.items()},
        "registered_five_gates":gates,
        "registered_primary_pass":all(gates.values()),
        "scope":"Input-family OOD within fixed RRC25 six-tuple program class; "
                "hidden audit entropy and target-output witness external; "
                "source-level synthetic scale test, no program-grammar OOD."
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
        print("NOR41 FIRST PUBLIC-ONLY 64 SOURCES",len(raw),digest(raw),flush=True)
        return
    if not z.fixture:a.error("--fixture required")
    result=evaluate(z.fixture)
    if z.out:
        z.out.parent.mkdir(parents=True,exist_ok=True)
        z.out.write_bytes(canonical(result))
    print("NOR41 REGISTERED PRIMARY",result["registered_primary_pass"])
    print("NOR41 FIVE GATES",result["registered_five_gates"])
    print("NOR41 WINS",result["source_direction_wins"])
    print("NOR41 MEANS",result["source_means"])
    print("NOR41 OOD TOP RETENTION",result["retained_ood_top_gain_ratio"])


if __name__=="__main__":
    main()
