"""NOR-37 independent source/greedy/stochastic-auditor cross-play replay.

Does NOT import the NOR37 experiment module. Reconstructs 24 original
native RRC25 programs with independent NOR21 source interpreter; uses
separate per-atom stochastic sampler, CAL trial-ID SET union attacks,
and EVAL position-SET intersections. Verifies original integer counts,
source-level gates, original positive/negative controls and SHA evidence.
"""
from __future__ import annotations

import argparse,json,random,statistics,sys
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from math import comb,isclose
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import receipt_audit as original21
from research.nor25_robust_audits import experiment as original25
from research.nor36_policyaware_curvature import receipt_audit as original36
from research.nor34_eight_atom_curvature import receipt_audit as original34

REGISTERED="a87401971de2cdefd7f36110fc6beae009c43bef"
ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
MODES=("original","atom_random","atom_curvature50",
       "stochastic25","stochastic50","stochastic75")
POLICIES=("original_h50","atom_random_h50","deterministic_h50",
          "stochastic25_h50","stochastic50_h50","stochastic75_h50",
          "uniform16","original_mix","deterministic_mix","stochastic50_mix")
CONTROLS=("top_curvature","bottom_curvature","eight_atom_matched","full_uniform")
REPS=(56,57,58,59)
DEGREES=(512,1024)
FAMILIES=("wide_dense_small","wide_sparse_small","wide_rational_short")
OFFSET=130037000
K=16
R=4
CAL=256
EVAL=512
ATTACKS=8
EXPLORE={"stochastic25":1,"stochastic50":2,"stochastic75":3}
TRANSFORM_STREAM={"atom_random":202611350076,
                  "atom_curvature50":202611350077}


def canonical(a):
    return (json.dumps(a,indent=2,sort_keys=True,ensure_ascii=True)+"\n").encode()


def seed(*args):
    return int.from_bytes(sha256("|".join(map(str,args)).encode("ascii")).digest(),"big")


def public_future_sha(y):
    return sha256(json.dumps(y,separators=(",",":")).encode()).hexdigest()


def historical_source_ids():
    old=original36.historical_roster()
    with ZipFile(ROOT/"research/nor36_policyaware_curvature/results/nor36_original_hosted.zip") as z:
        a=z.read("nor36_fixture.json")
        b=z.read("nor36_full.json")
    assert len(a)==4224502 and sha256(a).hexdigest()==(
        "881bf776e8e77eff6b8a5406789d07a4ced2eed9394a62b27d02d835c99355a4")
    assert len(b)==341232 and sha256(b).hexdigest()==(
        "ffc9ebaf5b8e16c93e01c2c71cc387e4c8a455505f5af8cb3c9cad42ea9a44b5")
    source=json.loads(a)
    assert source["schema"]=="nor36.new_public_only_reps52to55.v1"
    old.extend(row["full_public_input_sha256"] for row in source["cases"])
    assert len(old)==len(set(old))==156
    return sorted(old)


def regenerate_native(row):
    rep,d,fam=row["replicate"],row["degree"],row["family"]
    assert rep in REPS and d in DEGREES and fam in FAMILIES
    expected=corpus.SEED+OFFSET+1000003*rep+100009*DEGREES.index(d)+10007*FAMILIES.index(fam)
    assert row["source_seed"]==expected
    assert row["query_degree"]==3*d//2 and 0<=row["selected_attempt"]<100
    rng=random.Random(expected)
    planted=demos=query=None
    for _ in range(row["selected_attempt"]+1):
        planted=(rng.choice(corpus.OPS),rng.choice(corpus.DELAYS),
                 rng.choice(corpus.SPACINGS),rng.choice(corpus.AMPLITUDES),
                 rng.choice(corpus.OPS),rng.choice(corpus.OPS))
        demos=tuple((x:=original21.independent_source(rng,fam,3*d//2),
                     corpus.execute(x,planted)) for _ in range(2))
        query=original21.independent_source(rng,fam,3*d//2)
    assert planted is not None
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    hr=row["public_header"]
    header=corpus.Header(hr["base"],hr["delay"],hr["spacing"],hr["amplitude"])
    assert planted[:4]==(header.base,header.delay,header.spacing,header.amplitude)
    c=corpus.Case("nor37-independent",fam,
        tuple((tuple(x),tuple(y)) for x,y in row["demos"]),tuple(query),(),None)
    assert c.program is None and c.future==()
    assert original21.public_source_hash(c,d)==row["full_public_input_sha256"]
    mask=original21.independent_native_mask(c,header,d)
    assert mask==row["original_native_public_mask"] and mask>0
    j=next(i for i in range(16) if mask&(1<<i))
    pred=corpus.execute(query,original21.candidate_program(header,j))
    for i in range(16):
        if mask&(1<<i):
            assert corpus.execute(query,original21.candidate_program(header,i))==pred
    assert corpus.execute(query,planted)==pred
    return mask,j,pred,header


def geometry(pred,header):
    m=len(pred)
    assert m in (769,1537)
    tail=set(range(m-96,m))
    magnitude=set(sorted(range(m),key=lambda i:(-abs(pred[i]),i))[:96])
    lag=set(range(header.delay+header.spacing,header.delay+header.spacing+96))
    old={"tail":tail,"magnitude":magnitude,"lag":lag,"uniform":set(range(m))}
    union=tail|magnitude|lag
    old["off_template"]=set(range(m))-union
    atoms=[set() for _ in range(8)]
    scores=[abs(pred[(i-1)%m]-2*pred[i]+pred[(i+1)%m]) for i in range(m)]
    for i in range(m):
        atoms[int(i in tail)+2*int(i in magnitude)+4*int(i in lag)].add(i)
    assert sum(map(len,atoms))==m
    top=sorted(range(m),key=lambda i:(-scores[i],i))
    bottom=sorted(range(m),key=lambda i:(scores[i],i))
    cu=set([i for i in top if i in union][:48])
    cv=set([i for i in top if i not in union][:48])
    bu=set([i for i in bottom if i in union][:48])
    bv=set([i for i in bottom if i not in union][:48])
    assert len(cu)==len(cv)==len(bu)==len(bv)==48
    return old,atoms,union,cu,cv,bu,bv,scores


def component_parts(policy):
    single={
        "original_h50":("original","hybrid_50"),
        "atom_random_h50":("atom_random","hybrid_50"),
        "deterministic_h50":("atom_curvature50","hybrid_50"),
        "stochastic25_h50":("stochastic25","hybrid_50"),
        "stochastic50_h50":("stochastic50","hybrid_50"),
        "stochastic75_h50":("stochastic75","hybrid_50"),
        "uniform16":("original","uniform"),
    }
    if policy in single:
        m,a=single[policy]
        return ((m,a,16),)
    mode={"original_mix":"original","deterministic_mix":"atom_curvature50",
          "stochastic50_mix":"stochastic50"}[policy]
    return tuple((mode,arm,w) for arm,w in zip(ARMS,WEIGHTS) if w)


def original_replan(base,atoms,scores,key,arm,t,mode):
    if mode=="original":return tuple(base)
    quota=[len(set(base)&a) for a in atoms]
    result=[]
    for j,(atom,needed) in enumerate(zip(atoms,quota)):
        if not needed:continue
        rng=random.Random(seed(TRANSFORM_STREAM[mode],key,arm,t,j))
        ordered=sorted(atom)
        if mode=="atom_random":
            result.extend(rng.sample(ordered,needed))
        else:
            high=sorted(ordered,key=lambda i:(-scores[i],i))
            count=(needed+1)//2
            prefix=high[:count]
            rest=sorted(atom-set(prefix))
            result.extend(prefix+rng.sample(rest,needed-count))
    assert len(result)==len(set(result))==K
    return tuple(result)


def stochastic_replan(base,atoms,scores,phase_key,src,arm,t,mode):
    assert mode in EXPLORE
    base_set=set(base)
    result=[]
    branches={"explore_atoms":0,"targeted_atoms":0,"targeted_prefix_draws":0}
    for j,atom in enumerate(atoms):
        quota=len(base_set&atom)
        if quota==0:continue
        ordered=sorted(atom,key=lambda i:(-scores[i],i))
        n=len(ordered)
        rng=random.Random(seed(202611370076,phase_key,src,arm,t,j,mode))
        if rng.randrange(4)<EXPLORE[mode]:
            result.extend(rng.sample(sorted(atom),quota))
            branches["explore_atoms"]+=1
            continue
        c=(quota+1)//2
        window=min(n,max(2*c,(n+7)//8))
        selected=rng.sample(ordered[:window],c)
        remain=sorted(atom-set(selected))
        result.extend(selected+rng.sample(remain,quota-c))
        branches["targeted_atoms"]+=1
        branches["targeted_prefix_draws"]+=c
    assert len(result)==len(set(result))==K
    assert all(len(set(result)&a)==len(base_set&a) for a in atoms)
    return tuple(result),branches


def policy_schedules(old,atoms,scores,src,phase):
    n=CAL if phase=="cal" else EVAL
    stream={"cal":202611370071,"eval":202611370073}[phase]
    phase_key=sha256(f"{stream}|{phase}|{src}".encode("ascii")).hexdigest()
    key=sha256(f"{phase_key}|nor37-original-transform".encode("ascii")).hexdigest()
    out={mode:{arm:[] for arm in ARMS} for mode in MODES}
    branch_counts={mode:Counter() for mode in EXPLORE}
    for t in range(n):
        base=original25.policy_schedules(old,phase_key,t)
        for arm in ARMS:
            ref=tuple(base[(arm,K)])
            for mode in MODES:
                if mode in EXPLORE:
                    plan,counts=stochastic_replan(ref,atoms,scores,phase_key,src,arm,t,mode)
                    branch_counts[mode].update(counts)
                else:
                    plan=original_replan(ref,atoms,scores,key,arm,t,mode)
                out[mode][arm].append(plan)
    return out,{mode:dict(n) for mode,n in branch_counts.items()}


def trial_id_sets(plans,m):
    out={}
    for mode,arms in plans.items():
        for arm,rows in arms.items():
            group=[set() for _ in range(m)]
            for t,checked in enumerate(rows):
                for i in checked:group[i].add(t)
            out[(mode,arm)]=group
    return out


def independent_greedy_attacks(plans,src,m):
    coverage=trial_id_sets(plans,m)
    result={}
    for policy in POLICIES:
        parts=component_parts(policy)
        found=[]
        for restart in range(ATTACKS):
            order=list(range(m))
            random.Random(seed(202611370072,src,policy,restart)).shuffle(order)
            chosen=set()
            seen={p[:2]:set() for p in parts}
            for _ in range(R):
                def cost(i):
                    return sum(w*len(seen[(mode,arm)]|coverage[(mode,arm)][i])
                               for mode,arm,w in parts)
                best=min((i for i in order if i not in chosen),key=cost)
                chosen.add(best)
                for mode,arm,w in parts:
                    seen[(mode,arm)].update(coverage[(mode,arm)][best])
            assert len(chosen)==R
            found.append(tuple(sorted(chosen)))
        result[policy]=found
    return result


def independent_blind(src,atoms,u,cu,cv,bu,bv,m):
    out={name:[] for name in CONTROLS}
    for t in range(ATTACKS):
        rng=random.Random(seed(202611370074,src,"top_curvature",t))
        top=tuple(sorted(rng.sample(sorted(cu),2)+rng.sample(sorted(cv),2)))
        rng=random.Random(seed(202611370074,src,"bottom_curvature",t))
        bottom=tuple(sorted(rng.sample(sorted(bu),2)+rng.sample(sorted(bv),2)))
        quota=[sum(i in a for i in top) for a in atoms]
        assert quota[0]==2 and sum(quota)==R
        rng=random.Random(seed(202611370074,src,"eight_atom_matched",t))
        matched=tuple(sorted(i for j,q in enumerate(quota)
                             for i in rng.sample(sorted(atoms[j]),q)))
        rng=random.Random(seed(202611370074,src,"full_uniform",t))
        uniform=tuple(sorted(rng.sample(range(m),R)))
        assert all(len(set(a))==4 for a in (top,bottom,matched,uniform))
        assert [sum(i in a for i in matched) for a in atoms]==quota
        for name,a in zip(CONTROLS,(top,bottom,matched,uniform)):
            out[name].append(a)
    return out


def mixture_choice(src,t):
    i=random.Random(seed(202611370075,src,t)).randrange(16)
    s=0
    for arm,w in zip(ARMS,WEIGHTS):
        s+=w
        if i<s:return arm
    raise AssertionError("unexpected policy mixture")


def top_prob(cu,cv,checked):
    setp=set(checked)
    return Fraction(1)-Fraction(comb(48-len(setp&cu),2),comb(48,2))*Fraction(
        comb(48-len(setp&cv),2),comb(48,2))


def exact_top(plans,cu,cv):
    out={}
    for policy in POLICIES:
        total=Fraction(0)
        for mode,arm,w in component_parts(policy):
            total+=Fraction(w,16)*sum(
                (top_prob(cu,cv,plan) for plan in plans[mode][arm]),
                Fraction(0))/EVAL
        out[policy]=total
    return out


def scored_rows(records):
    rows=[]
    for rec in records:
        cross=rec["crossplay_weighted_numerators"]
        den=rec["crossplay_denominator"]
        top={name:Fraction(*v) for name,v in rec["exact_top_support_detection"].items()}
        self_s=cross["stochastic50_h50"]["stochastic50_h50"]/den
        self_d=cross["deterministic_h50"]["deterministic_h50"]/den
        worst_s=min(cross[name]["stochastic50_h50"]/den for name in POLICIES)
        worst_d=min(cross[name]["deterministic_h50"]/den for name in POLICIES)
        self_sm=cross["stochastic50_mix"]["stochastic50_mix"]/den
        self_dm=cross["deterministic_mix"]["deterministic_mix"]/den
        rows.append({
            "public_sha256":rec["public_sha256"],"degree":rec["degree"],
            "family":rec["family"],
            "self_stoch50_minus_deterministic":self_s-self_d,
            "worst_class_stoch50_minus_deterministic":worst_s-worst_d,
            "top_stoch50_minus_atom_random":float(
                top["stochastic50_h50"]-top["atom_random_h50"]),
            "top_deterministic_minus_atom_random":float(
                top["deterministic_h50"]-top["atom_random_h50"]),
            "mix_self_stoch50_minus_deterministic":self_sm-self_dm,
            "mix_top_stoch50_minus_original":float(
                top["stochastic50_mix"]-top["original_mix"]),
            "self_stoch25_minus_deterministic":
                cross["stochastic25_h50"]["stochastic25_h50"]/den-self_d,
            "self_stoch75_minus_deterministic":
                cross["stochastic75_h50"]["stochastic75_h50"]/den-self_d,
            "worst_stoch50":worst_s,
            "worst_deterministic":worst_d,
            "self_uniform":cross["uniform16"]["uniform16"]/den,
        })
    return rows


def replay(fixture_raw,result_raw):
    fixture=json.loads(fixture_raw)
    outcome=json.loads(result_raw)
    past=historical_source_ids()
    assert fixture["schema"]=="nor37.new_public_only_reps56to59.v1"
    assert outcome["schema"]=="nor37.stochastic_intraatom_curvature_policyaware.v1"
    assert fixture["registered_first_commit"]==outcome["registration_first_commit"]==REGISTERED
    assert fixture["public_only"] and fixture["planted_and_private_target_absent"]
    assert fixture["attack_witness_and_audit_absent"]
    assert fixture["excluded_historical_sources"]==len(past)==156
    assert fixture["prior_source_ids_sha256"]==sha256(canonical(past)).hexdigest()
    assert outcome["public_fixture_sha256"]==sha256(fixture_raw).hexdigest()
    assert outcome["fresh_source_units"]==len(fixture["cases"])==len(outcome["source_records"])==24
    assert outcome["historical_sources_excluded"]==156
    assert outcome["auditor_policies"]==list(POLICIES)
    assert outcome["component_auditor_arms"]==list(ARMS)
    assert outcome["original_mix_weights"]==list(WEIGHTS)
    assert outcome["transform_modes"]==list(MODES)
    assert outcome["blind_control_families"]==list(CONTROLS)
    assert outcome["calibration_schedules_per_source"]==CAL
    assert outcome["heldout_schedules_per_source"]==EVAL
    assert outcome["greedy_restarts_per_policy"]==ATTACKS
    assert outcome["audit_budget"]==K and outcome["attack_radius"]==R
    assert outcome["public_proposals_frozen_before_private_truth"]
    assert outcome["calibrated_attacks_frozen_before_independent_heldout_schedules"]
    assert outcome["authentic_external_target_coefficient_witness_assumed"]
    ids=[r["full_public_input_sha256"] for r in fixture["cases"]]
    assert len(set(ids))==24 and set(ids).isdisjoint(past)
    cross_decisions=0
    replayed=[]
    for row,record in zip(fixture["cases"],outcome["source_records"],strict=True):
        src=row["full_public_input_sha256"]
        mask,j,pred,header=regenerate_native(row)
        assert record["public_sha256"]==src
        assert record["replicate"]==row["replicate"]
        assert record["degree"]==row["degree"] and record["family"]==row["family"]
        assert record["future_length"]==len(pred)
        assert record["native_mask"]==mask and record["chosen_class"]==j
        assert record["public_prediction_sha256"]==public_future_sha(pred)
        assert record["private_native_truth_sha256_after_freeze"]==public_future_sha(pred)
        old,atoms,u,cu,cv,bu,bv,scores=geometry(pred,header)
        assert record["original_atom_sizes"]==list(map(len,atoms))
        assert record["top_curvature_support_sha256"]==sha256(canonical(sorted(cu|cv))).hexdigest()
        assert record["bottom_curvature_support_sha256"]==sha256(canonical(sorted(bu|bv))).hexdigest()
        cal,cal_behavior=policy_schedules(old,atoms,scores,src,"cal")
        assert record["calibration_plans_sha256"]==sha256(canonical(cal)).hexdigest()
        assert record["calibration_stochastic_branch_counts"]==cal_behavior
        greedy=independent_greedy_attacks(cal,src,len(pred))
        assert record["greedy_attack_sets"]=={
            k:[list(a) for a in vals] for k,vals in greedy.items()}
        assert record["greedy_attack_unique_sets"]=={
            k:len(set(v)) for k,v in greedy.items()}
        blind=independent_blind(src,atoms,u,cu,cv,bu,bv,len(pred))
        assert record["blind_attack_sets"]=={
            k:[list(a) for a in vals] for k,vals in blind.items()}
        attacks={**greedy,**blind}
        evals,eval_behavior=policy_schedules(old,atoms,scores,src,"eval")
        assert record["evaluation_plans_sha256"]==sha256(canonical(evals)).hexdigest()
        assert record["evaluation_stochastic_branch_counts"]==eval_behavior
        selected=[mixture_choice(src,t) for t in range(EVAL)]
        assert record["realized_mixture_arm_counts"]==dict(Counter(selected))
        plan_sets={(mode,arm):[set(p) for p in evals[mode][arm]]
                   for mode in MODES for arm in ARMS}
        for arm in ARMS:
            for t in range(EVAL):
                baseline=plan_sets[("original",arm)][t]
                quota=[len(baseline&atom) for atom in atoms]
                for mode in MODES:
                    p=plan_sets[(mode,arm)][t]
                    assert len(p)==K
                    assert [len(p&a) for a in atoms]==quota
        observed={}
        realized={}
        for attacker,variants in attacks.items():
            observed[attacker]={}
            for policy in POLICIES:
                total=0
                for mode,arm,w in component_parts(policy):
                    for a in variants:
                        changes=set(a)
                        total+=w*sum(not checked.isdisjoint(changes)
                                     for checked in plan_sets[(mode,arm)])
                        cross_decisions+=len(variants)*0 + EVAL
                observed[attacker][policy]=total
            realized[attacker]={}
            for policy,mode in (
                ("original_mix","original"),
                ("deterministic_mix","atom_curvature50"),
                ("stochastic50_mix","stochastic50")):
                realized[attacker][policy]=sum(
                    int(not plan_sets[(mode,selected[t])][t].isdisjoint(set(a)))
                    for t in range(EVAL) for a in variants)
        assert record["crossplay_weighted_numerators"]==observed
        assert record["crossplay_denominator"]==16*ATTACKS*EVAL
        assert record["realized_mixture_attack_hits"]==realized
        expected_top=exact_top(evals,cu,cv)
        assert record["exact_top_support_detection"]=={
            name:[v.numerator,v.denominator] for name,v in expected_top.items()}
        uniform=Fraction(1)-Fraction(comb(len(pred)-K,R),comb(len(pred),R))
        assert record["exact_full_uniform_attack_detection_fraction"]==[
            uniform.numerator,uniform.denominator]
        assert record["honest_target_false_rejections"]==0
        assert record["fully_schedule_informed_attacker_detections"]==0
        assert record["delta_decision_disagreements"]==0
        assert all(v==0 for v in record["exact_quota_and_control_checks"].values())
        assert record["direct_authentic_witness_checks"]>0
        assert Fraction(*record["original_mixture_conditional_blind_floor_fraction"])>0
        replayed.append(record)
    rows=scored_rows(replayed)
    assert len(rows)==len(outcome["source_level_primary"])==24
    for got,expected in zip(outcome["source_level_primary"],rows,strict=True):
        assert got.keys()==expected.keys()
        for key,value in expected.items():
            if isinstance(value,float):
                assert isclose(got[key],value,rel_tol=0,abs_tol=1e-14),(key,got[key],value)
            else:assert got[key]==value
    primary_cols=("self_stoch50_minus_deterministic",
                  "worst_class_stoch50_minus_deterministic",
                  "top_stoch50_minus_atom_random",
                  "top_deterministic_minus_atom_random",
                  "mix_self_stoch50_minus_deterministic",
                  "mix_top_stoch50_minus_original")
    means={k:statistics.mean(row[k] for row in rows) for k in primary_cols}
    wins={k:sum(row[k]>0 for row in rows) for k in primary_cols}
    ratio=means["top_stoch50_minus_atom_random"]/(
        means["top_deterministic_minus_atom_random"]
        if means["top_deterministic_minus_atom_random"]>0 else 1)
    gates={
        "stochastic50_selfplay_improves":wins["self_stoch50_minus_deterministic"]>=18
            and means["self_stoch50_minus_deterministic"]>=.005,
        "stochastic50_worst_calibrated_class_improves":
            wins["worst_class_stoch50_minus_deterministic"]>=18
            and means["worst_class_stoch50_minus_deterministic"]>=.005,
        "feature_aligned_curvature_signal_retained":
            wins["top_stoch50_minus_atom_random"]>=18
            and means["top_stoch50_minus_atom_random"]>=.020
            and means["top_deterministic_minus_atom_random"]>0
            and ratio>=.25,
        "stochastic50_mixture_resilience_and_top_gain":
            wins["mix_self_stoch50_minus_deterministic"]>=18
            and means["mix_self_stoch50_minus_deterministic"]>=.005
            and wins["mix_top_stoch50_minus_original"]>=18
            and means["mix_top_stoch50_minus_original"]>=.020,
        "original_source_native_witness_and_full_replay":True
    }
    assert outcome["source_means"].keys()==means.keys()
    for k in means:assert isclose(outcome["source_means"][k],means[k],abs_tol=1e-14)
    assert outcome["source_direction_wins"]==wins
    assert isclose(outcome["retained_ratio_stochastic_to_deterministic_top_gain"],
                   ratio,abs_tol=1e-14)
    assert outcome["registered_five_gates"]==gates
    assert outcome["registered_primary_pass"]==all(gates.values())
    return {
        "historical_source_units_excluded":156,
        "new_source_units_reconstructed":24,
        "calibrated_greedy_attack_sets_replayed":24*len(POLICIES)*ATTACKS,
        "fixed_blind_attack_sets_replayed":24*len(CONTROLS)*ATTACKS,
        "component_calibration_and_evaluation_plans_independently_built":24*len(MODES)*len(ARMS)*(CAL+EVAL),
        "source_policy_attack_world_decisions":cross_decisions,
        "source_means":means,"source_wins":wins,
        "registered_primary_pass":all(gates.values()),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--full",type=Path,required=True)
    p.add_argument("--check",action="store_true")
    a=p.parse_args()
    result=replay(a.fixture.read_bytes(),a.full.read_bytes())
    print("NOR37 INDEPENDENT SOURCE GREEDY STOCHASTIC AUDITOR REPLAY",result)
    if a.check:
        assert result["new_source_units_reconstructed"]==24
        assert result["source_policy_attack_world_decisions"]==(
            24*14*ATTACKS*EVAL*(7+3*5))


if __name__=="__main__":
    main()
