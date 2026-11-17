"""NOR-38 wholly separate interpreter, greedy trial-ID-SET, and heldout scorer.

Independent of research.nor38_global_uniform.experiment, and never calls its
evaluator. Reuses original NOR21/NOR25 historical input grammar and the
NOR37 independent-replayer geometry, with new standalone NOR38 policy logic.
Checks every pair of selected altered support and held-out audit schedule,
then recomputes the five registered gates.
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

REGISTERED="37f9ed96b122ea16fb681dd5df638cbc4285d1e4"
ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
MODES=("original","atom_random","atom_curvature50","stochastic50")
POLICIES=("uniform16","original_mix","deterministic_mix","stochastic50_mix",
          "atom_random_h50","global7_mix","global8_mix")
CONTROLS=("top_curvature","bottom_curvature","eight_atom_matched",
          "full_uniform","low_cal_exposure")
REPS=tuple(range(60,68))
DEGREES=(512,1024)
FAMILIES=("wide_dense_small","wide_sparse_small","wide_rational_short")
OFFSET=130038000
K,R,CAL,EVAL,ATTACKS=16,4,256,512,8
EXPLORE={"stochastic50":2}
TRANSFORM_STREAM={"atom_random":202611350076,
                  "atom_curvature50":202611350077}
DENOM=256


def canonical(obj):
    return (json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")


def seed(*args):
    return int.from_bytes(sha256("|".join(map(str,args)).encode("ascii")).digest(),"big")


def public_future_sha(y):
    return sha256(json.dumps(y,separators=(",",":")).encode()).hexdigest()


def historical_source_ids():
    past=original36.historical_roster()
    with ZipFile(ROOT/"research/nor36_policyaware_curvature/results/nor36_original_hosted.zip") as z:
        a=z.read("nor36_fixture.json")
        b=z.read("nor36_full.json")
    assert len(a)==4224502 and sha256(a).hexdigest()==(
        "881bf776e8e77eff6b8a5406789d07a4ced2eed9394a62b27d02d835c99355a4")
    assert len(b)==341232 and sha256(b).hexdigest()==(
        "ffc9ebaf5b8e16c93e01c2c71cc387e4c8a455505f5af8cb3c9cad42ea9a44b5")
    fixture=json.loads(a)
    past.extend(row["full_public_input_sha256"] for row in fixture["cases"])
    assert len(past)==len(set(past))==156
    with ZipFile(ROOT/"research/nor37_stochastic_curvature/results/nor37_original_hosted.zip") as z:
        a=z.read("nor37_fixture.json")
        b=z.read("nor37_full.json")
    assert len(a)==4857464 and sha256(a).hexdigest()==(
        "f969925a159aa3e36d7061e2e65e0a6b9e6e0f82634e2a7e52b4dac874df6710")
    assert len(b)==559101 and sha256(b).hexdigest()==(
        "4852bd7de8012484db34404519553ae4d1e9f388203f2fdbf8b7516a534b684c")
    fixture=json.loads(a)
    past.extend(row["full_public_input_sha256"] for row in fixture["cases"])
    assert len(past)==len(set(past))==180
    return sorted(past)


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
    c=corpus.Case("nor38-independent",fam,
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

def trial_id_sets(plans,m):
    out={}
    for mode,arms in plans.items():
        for arm,rows in arms.items():
            group=[set() for _ in range(m)]
            for t,checked in enumerate(rows):
                for i in checked:group[i].add(t)
            out[(mode,arm)]=group
    return out

def top_prob(cu,cv,checked):
    setp=set(checked)
    return Fraction(1)-Fraction(comb(48-len(setp&cu),2),comb(48,2))*Fraction(
        comb(48-len(setp&cv),2),comb(48,2))


def component_parts(policy):
    if policy=="uniform16":return (("original","uniform",256),)
    if policy=="atom_random_h50":return (("atom_random","hybrid_50",256),)
    if policy in ("original_mix","deterministic_mix","stochastic50_mix"):
        mode={"original_mix":"original","deterministic_mix":"atom_curvature50",
              "stochastic50_mix":"stochastic50"}[policy]
        return tuple((mode,arm,w*16) for arm,w in zip(ARMS,WEIGHTS) if w)
    q={"global7_mix":7,"global8_mix":8}[policy]
    return (("original","uniform",q*16),)+tuple(
        ("atom_curvature50",arm,(16-q)*w) for arm,w in zip(ARMS,WEIGHTS) if w)


def policy_schedules(old,atoms,scores,src,phase):
    n={"cal":CAL,"eval":EVAL}[phase]
    stream={"cal":202611380071,"eval":202611380072}[phase]
    phase_key=sha256(f"{stream}|nor38|{phase}|{src}".encode("ascii")).hexdigest()
    key=sha256(f"{202611380076}|{phase_key}".encode("ascii")).hexdigest()
    out={mode:{arm:[] for arm in ARMS} for mode in MODES}
    branches=Counter()
    for t in range(n):
        base=original25.policy_schedules(old,phase_key,t)
        for arm in ARMS:
            ref=tuple(base[(arm,K)])
            out["original"][arm].append(ref)
            for mode in ("atom_random","atom_curvature50"):
                plan=original_replan(ref,atoms,scores,key,arm,t,mode)
                out[mode][arm].append(plan)
            plan,count=stochastic_replan(ref,atoms,scores,phase_key,src,arm,t,"stochastic50")
            out["stochastic50"][arm].append(plan)
            branches.update(count)
    return out,dict(branches)


def independent_greedy_attacks(plans,src,m):
    visited=trial_id_sets(plans,m)
    out={}
    for policy in POLICIES:
        parts=component_parts(policy)
        variants=[]
        for restart in range(ATTACKS):
            order=list(range(m))
            random.Random(seed(202611380073,src,policy,restart)).shuffle(order)
            used=set()
            exposed={p[:2]:set() for p in parts}
            for _ in range(R):
                best=None
                best_cost=None
                for i in order:
                    if i in used:continue
                    cost=sum(w*len(exposed[(mode,arm)]|visited[(mode,arm)][i])
                             for mode,arm,w in parts)
                    if best_cost is None or cost<best_cost:
                        best,best_cost=i,cost
                assert best is not None
                used.add(best)
                for mode,arm,_ in parts:
                    exposed[(mode,arm)].update(visited[(mode,arm)][best])
            variants.append(tuple(sorted(used)))
        out[policy]=variants
    return out,visited


def independent_blind(src,atoms,u,cu,cv,bu,bv,m,visited):
    exposure=[sum(w*len(visited[(mode,arm)][i])
                  for mode,arm,w in component_parts("deterministic_mix"))
              for i in range(m)]
    pool=sorted(range(m),key=lambda i:(exposure[i],i))[:min(m,96)]
    out={name:[] for name in CONTROLS}
    for t in range(ATTACKS):
        rng=random.Random(seed(202611380074,src,"top_curvature",t))
        top=tuple(sorted(rng.sample(sorted(cu),2)+rng.sample(sorted(cv),2)))
        rng=random.Random(seed(202611380074,src,"bottom_curvature",t))
        bottom=tuple(sorted(rng.sample(sorted(bu),2)+rng.sample(sorted(bv),2)))
        quotas=[sum(i in cat for i in top) for cat in atoms]
        rng=random.Random(seed(202611380074,src,"eight_atom_matched",t))
        matched=tuple(sorted(i for j,q in enumerate(quotas)
                             for i in rng.sample(sorted(atoms[j]),q)))
        rng=random.Random(seed(202611380074,src,"full_uniform",t))
        uniform=tuple(sorted(rng.sample(range(m),R)))
        rng=random.Random(seed(202611380074,src,"low_cal_exposure",t))
        low=tuple(sorted(rng.sample(pool,R)))
        for name,attack in zip(CONTROLS,(top,bottom,matched,uniform,low)):
            assert len(attack)==len(set(attack))==R
            out[name].append(attack)
        assert [sum(i in atom for i in matched) for atom in atoms]==quotas
    return out


def realized_choice(src,t,policy):
    rng=random.Random(seed(202611380075,"eval",src,t,policy))
    if policy in ("global7_mix","global8_mix"):
        q={"global7_mix":7,"global8_mix":8}[policy]
        if rng.randrange(16)<q:return ("original","uniform")
        mode="atom_curvature50"
    else:
        mode={"original_mix":"original","deterministic_mix":"atom_curvature50",
              "stochastic50_mix":"stochastic50"}[policy]
    idx=rng.randrange(16)
    for arm,w in zip(ARMS,WEIGHTS):
        if idx<w:return (mode,arm)
        idx-=w
    raise AssertionError("invalid mixture integer quantization")


def exact_top(plans,cu,cv):
    out={}
    for policy in POLICIES:
        total=Fraction(0)
        for mode,arm,w in component_parts(policy):
            total+=Fraction(w,DENOM)*sum(
                (top_prob(cu,cv,checked) for checked in plans[mode][arm]),
                Fraction(0))/EVAL
        out[policy]=total
    return out


def scored_rows(records):
    out=[]
    for rec in records:
        cross=rec["crossplay_weighted_numerators"]
        den=rec["crossplay_denominator"]
        top={k:Fraction(*v) for k,v in rec["exact_top_support_detection"].items()}
        gw=min(cross[att]["global7_mix"]/den for att in POLICIES)
        ow=min(cross[att]["original_mix"]/den for att in POLICIES)
        out.append({
            "public_sha256":rec["public_sha256"],"replicate":rec["replicate"],
            "degree":rec["degree"],"family":rec["family"],
            "global7_self_minus_deterministic_self":
                (cross["global7_mix"]["global7_mix"]-
                 cross["deterministic_mix"]["deterministic_mix"])/den,
            "global7_worst_minus_original_worst":gw-ow,
            "global7_worst":gw,"original_worst":ow,
            "global7_top_minus_atom_random":float(
                top["global7_mix"]-top["atom_random_h50"]),
            "deterministic_top_minus_atom_random":float(
                top["deterministic_mix"]-top["atom_random_h50"]),
            "global8_self":cross["global8_mix"]["global8_mix"]/den,
            "uniform_self":cross["uniform16"]["uniform16"]/den,
            "global7_self":cross["global7_mix"]["global7_mix"]/den,
            "original_mix_self":cross["original_mix"]["original_mix"]/den})
    return out


def replay(fixture_raw,graded_raw):
    fixture=json.loads(fixture_raw)
    grade=json.loads(graded_raw)
    historical=historical_source_ids()
    assert fixture["schema"]=="nor38.global_uniform_public_only_reps60to67.v1"
    assert grade["schema"]=="nor38.global_uniform_policyaware.v1"
    assert fixture["registered_first_commit"]==grade["registration_first_commit"]==REGISTERED
    assert fixture["public_only"] and fixture["planted_and_private_target_absent"]
    assert fixture["attack_witness_and_audit_absent"]
    assert fixture["replicates"]==list(REPS) and fixture["degrees"]==list(DEGREES)
    assert fixture["excluded_historical_sources"]==len(historical)==180
    assert fixture["prior_source_ids_sha256"]==sha256(canonical(historical)).hexdigest()
    assert grade["public_fixture_sha256"]==sha256(fixture_raw).hexdigest()
    assert grade["fresh_source_units"]==48
    assert grade["historical_sources_excluded"]==180
    assert grade["auditor_policies"]==list(POLICIES)
    assert grade["blind_control_families"]==list(CONTROLS)
    assert grade["transform_modes"]==list(MODES)
    assert grade["original_mix_weights"]==list(WEIGHTS)
    assert grade["global_uniform_probabilities"]=={"global7_mix":[7,16],"global8_mix":[8,16]}
    assert grade["calibration_schedules_per_source"]==CAL
    assert grade["heldout_schedules_per_source"]==EVAL
    assert grade["greedy_restarts_per_policy"]==ATTACKS
    assert grade["audit_budget"]==K and grade["attack_radius"]==R
    assert grade["public_proposals_frozen_before_private_truth"]
    assert grade["calibrated_attacks_frozen_before_independent_heldout_schedules"]
    assert grade["authentic_external_target_coefficient_witness_assumed"]
    ids=[r["full_public_input_sha256"] for r in fixture["cases"]]
    assert len(ids)==len(set(ids))==len(fixture["cases"])==len(grade["source_records"])==48
    assert set(ids).isdisjoint(historical)
    assert {(r["replicate"],r["degree"],r["family"]) for r in fixture["cases"]}=={
        (rep,d,fam) for rep in REPS for d in DEGREES for fam in FAMILIES}
    decision_count=0
    for row,rec in zip(fixture["cases"],grade["source_records"],strict=True):
        src=row["full_public_input_sha256"]
        assert row["split"]=="TEST"
        assert all(key not in row for key in
                   ("program","future","target","attack","witness","audit"))
        mask,j,pred,header=regenerate_native(row)
        assert rec["public_sha256"]==src
        assert rec["replicate"]==row["replicate"]
        assert rec["degree"]==row["degree"] and rec["family"]==row["family"]
        assert rec["future_length"]==len(pred)
        assert rec["native_mask"]==mask and rec["chosen_class"]==j
        assert rec["public_prediction_sha256"]==public_future_sha(pred)
        assert rec["private_native_truth_sha256_after_freeze"]==public_future_sha(pred)
        old,atoms,u,cu,cv,bu,bv,scores=geometry(pred,header)
        assert rec["original_atom_sizes"]==list(map(len,atoms))
        assert rec["top_curvature_support_sha256"]==sha256(canonical(sorted(cu|cv))).hexdigest()
        assert rec["bottom_curvature_support_sha256"]==sha256(canonical(sorted(bu|bv))).hexdigest()
        cal,cal_counts=policy_schedules(old,atoms,scores,src,"cal")
        assert rec["calibration_plans_sha256"]==sha256(canonical(cal)).hexdigest()
        assert rec["calibration_stochastic_branch_counts"]==cal_counts
        greedy,visits=independent_greedy_attacks(cal,src,len(pred))
        assert rec["greedy_attack_sets"]=={
            k:[list(v) for v in variants] for k,variants in greedy.items()}
        assert rec["greedy_attack_unique_sets"]=={
            k:len(set(variants)) for k,variants in greedy.items()}
        blind=independent_blind(src,atoms,u,cu,cv,bu,bv,len(pred),visits)
        assert rec["blind_attack_sets"]=={
            k:[list(v) for v in variants] for k,variants in blind.items()}
        attacks={**greedy,**blind}
        ev,eval_counts=policy_schedules(old,atoms,scores,src,"eval")
        assert rec["evaluation_plans_sha256"]==sha256(canonical(ev)).hexdigest()
        assert rec["evaluation_stochastic_branch_counts"]==eval_counts
        selections={pol:[realized_choice(src,t,pol) for t in range(EVAL)]
                    for pol in ("original_mix","deterministic_mix","stochastic50_mix",
                                "global7_mix","global8_mix")}
        diag={pol:{"uniform":sum(mode=="original" and arm=="uniform"
                                   for mode,arm in sel),
                   "curvature":sum(mode=="atom_curvature50" for mode,arm in sel)}
              for pol,sel in selections.items()}
        assert rec["realized_mixture_branch_counts"]==diag
        plan_sets={(mode,arm):[set(p) for p in ev[mode][arm]]
                   for mode in MODES for arm in ARMS}
        for arm in ARMS:
            for t in range(EVAL):
                oldp=plan_sets[("original",arm)][t]
                quota=[len(oldp&cat) for cat in atoms]
                for mode in MODES:
                    now=plan_sets[(mode,arm)][t]
                    assert len(now)==K and [len(now&cat) for cat in atoms]==quota
        observed={}
        realized={}
        for attacker,variants in attacks.items():
            observed[attacker]={}
            for policy in POLICIES:
                total=0
                for mode,arm,w in component_parts(policy):
                    for altered in variants:
                        changed=set(altered)
                        total+=w*sum(not plan.isdisjoint(changed)
                                     for plan in plan_sets[(mode,arm)])
                        decision_count+=EVAL
                observed[attacker][policy]=total
            realized[attacker]={}
            for policy,selected in selections.items():
                realized[attacker][policy]=sum(
                    int(not plan_sets[(mode,arm)][t].isdisjoint(set(attack)))
                    for t,(mode,arm) in enumerate(selected) for attack in variants)
        assert rec["crossplay_weighted_numerators"]==observed
        assert rec["crossplay_denominator"]==DENOM*ATTACKS*EVAL
        assert rec["realized_mixture_attack_hits"]==realized
        top=exact_top(ev,cu,cv)
        assert rec["exact_top_support_detection"]=={
            k:[v.numerator,v.denominator] for k,v in top.items()}
        uval=Fraction(1)-Fraction(comb(len(pred)-K,R),comb(len(pred),R))
        assert rec["exact_full_uniform_attack_detection_fraction"]==[
            uval.numerator,uval.denominator]
        assert rec["global7_exact_theoretical_floor"]==[
            (Fraction(7,16)*uval).numerator,(Fraction(7,16)*uval).denominator]
        assert rec["global8_exact_theoretical_floor"]==[
            (Fraction(1,2)*uval).numerator,(Fraction(1,2)*uval).denominator]
        assert rec["honest_target_false_rejections"]==0
        assert rec["fully_schedule_informed_attacker_detections"]==0
        assert rec["delta_decision_disagreements"]==0
        assert rec["direct_authentic_witness_checks"]>0
        assert all(v==0 for v in rec["exact_quota_and_control_checks"].values())
    rows=scored_rows(grade["source_records"])
    assert len(rows)==len(grade["source_level_primary"])==48
    for got,expected in zip(grade["source_level_primary"],rows,strict=True):
        assert got.keys()==expected.keys()
        for k,v in expected.items():
            if isinstance(v,float):assert isclose(got[k],v,rel_tol=0,abs_tol=1e-14),(k,got[k],v)
            else:assert got[k]==v,(k,got[k],v)
    keys=("global7_self_minus_deterministic_self",
          "global7_worst_minus_original_worst",
          "global7_top_minus_atom_random",
          "deterministic_top_minus_atom_random")
    means={k:statistics.mean(r[k] for r in rows) for k in keys}
    wins={
        "global7_self_minus_deterministic_self":sum(
            x["global7_self_minus_deterministic_self"]>0 for x in rows),
        "global7_worst_minus_original_worst_tolerance":sum(
            x["global7_worst_minus_original_worst"]>=-.010 for x in rows),
        "global7_top_minus_atom_random":sum(
            x["global7_top_minus_atom_random"]>0 for x in rows)}
    denominator=means["deterministic_top_minus_atom_random"]
    ratio=means["global7_top_minus_atom_random"]/denominator if denominator>0 else None
    g={
        "global7_improves_curvature_mix_selfplay":
            wins["global7_self_minus_deterministic_self"]>=36 and
            means["global7_self_minus_deterministic_self"]>=.010,
        "global7_competitive_vs_original_mix_worst":
            wins["global7_worst_minus_original_worst_tolerance"]>=36 and
            means["global7_worst_minus_original_worst"]>=-.010 and
            statistics.mean(r["global7_worst"] for r in rows)>=.025,
        "global7_retains_top_curvature_signal":
            wins["global7_top_minus_atom_random"]>=36 and
            means["global7_top_minus_atom_random"]>=.020 and
            ratio is not None and ratio>=.20,
        "exact_global_floor_and_witness_integrity":True,
        "source_archive_and_replay_integrity_contract":True}
    for key,v in means.items():
        assert isclose(grade["source_means"][key],v,rel_tol=0,abs_tol=1e-13)
    assert grade["source_direction_wins"]==wins
    if ratio is None:assert grade["retained_ratio_global7_to_deterministic_top_gain"] is None
    else:assert isclose(grade["retained_ratio_global7_to_deterministic_top_gain"],
                        ratio,rel_tol=0,abs_tol=1e-13)
    assert grade["registered_five_gates"]==g
    assert grade["registered_primary_pass"]==all(g.values())
    replicates={str(rep):{
        key:statistics.mean(r[key] for r in rows if r["replicate"]==rep)
        for key in ("global7_self_minus_deterministic_self",
                    "global7_worst_minus_original_worst",
                    "global7_top_minus_atom_random")} for rep in REPS}
    assert grade["replicate_block_means"]==replicates
    assert decision_count==48*len(attacks)*ATTACKS*EVAL*sum(
        len(component_parts(p)) for p in POLICIES)
    return {"new_sources":48,"historical_sources":180,
            "independent_greedy_attack_sets":48*len(POLICIES)*ATTACKS,
            "blind_attack_sets":48*len(CONTROLS)*ATTACKS,
            "replayed_cal_eval_component_plans":48*len(MODES)*len(ARMS)*(CAL+EVAL),
            "crossplay_world_decisions":decision_count,
            "source_means":means,"source_wins":wins,
            "science_pass":all(g.values())}


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--fixture",type=Path,required=True)
    a.add_argument("--full",type=Path,required=True)
    a.add_argument("--check",action="store_true")
    args=a.parse_args()
    results=replay(args.fixture.read_bytes(),args.full.read_bytes())
    print("NOR38 INDEPENDENT EXACT SOURCE / GREEDY / CROSSPLAY REPLAY",results)
    if args.check:
        assert results["new_sources"]==48
        assert results["crossplay_world_decisions"]==48*12*ATTACKS*EVAL*29


if __name__=="__main__":
    main()
