"""NOR-37 prospective stochastic intra-atom curvature policy-aware cross-play.

Source-only fixtures freeze before CAL attacks. Calibrated greedy attacks
freeze before independent heldout EVAL audit schedules. Honest target
output is restored only after every public choice. First preregistration
commit a874019 preceded this implementation and any new source data.
"""
from __future__ import annotations
import argparse,json,random,statistics,sys
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
from research.nor36_policyaware_curvature import experiment as nor36

REGISTERED="a87401971de2cdefd7f36110fc6beae009c43bef"
FIXTURE_SCHEMA="nor37.new_public_only_reps56to59.v1"
RESULT_SCHEMA="nor37.stochastic_intraatom_curvature_policyaware.v1"
PREV_SOURCE_SHA="881bf776e8e77eff6b8a5406789d07a4ced2eed9394a62b27d02d835c99355a4"
PREV_RESULT_SHA="ffc9ebaf5b8e16c93e01c2c71cc387e4c8a455505f5af8cb3c9cad42ea9a44b5"
REPS=(56,57,58,59)
DEGREES=(512,1024)
FAMILIES=nor21.FAMILIES
OFFSET=130037000
ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
MODES=("original","atom_random","atom_curvature50",
       "stochastic25","stochastic50","stochastic75")
POLICIES=("original_h50","atom_random_h50","deterministic_h50",
          "stochastic25_h50","stochastic50_h50","stochastic75_h50",
          "uniform16","original_mix","deterministic_mix","stochastic50_mix")
CONTROLS=("top_curvature","bottom_curvature","eight_atom_matched","full_uniform")
K=16
R=4
CAL=256
EVAL=512
ATTACKS=8
S_CAL=202611370071
S_ATTACK=202611370072
S_EVAL=202611370073
S_BLIND=202611370074
S_MIX=202611370075
S_TRANSFORM=202611370076
DELTAS=(1,4294049777)
EXPLORE_QUANTA={"stochastic25":1,"stochastic50":2,"stochastic75":3}


def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()


def digest(b):
    return sha256(b).hexdigest()


def seed(*args):
    return int.from_bytes(sha256("|".join(map(str,args)).encode("ascii")).digest(),"big")


def history():
    past=nor36.history()
    p=ROOT/"research/nor36_policyaware_curvature/results/nor36_original_hosted.zip"
    with ZipFile(p) as z:
        assert z.testzip() is None
        source=z.read("nor36_fixture.json")
        grade=z.read("nor36_full.json")
    assert len(source)==4224502 and digest(source)==PREV_SOURCE_SHA
    assert len(grade)==341232 and digest(grade)==PREV_RESULT_SHA
    f=json.loads(source)
    assert f["schema"]=="nor36.new_public_only_reps52to55.v1"
    all_ids=sorted(past+[r["full_public_input_sha256"] for r in f["cases"]])
    assert len(all_ids)==len(set(all_ids))==156
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
        case=corpus.Case("nor37-source",fam,demos,query,(),None)
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
    raise RuntimeError("NOR37 source exhausted 100 admissible attempts")


def make_fixture():
    old=history()
    seen=set(old)
    rows=[new_source(rep,d,fam,seen)
          for rep in REPS for d in DEGREES for fam in FAMILIES]
    assert len(rows)==24 and len(seen)==180
    return {"schema":FIXTURE_SCHEMA,"registered_first_commit":REGISTERED,
            "replicates":list(REPS),"degrees":list(DEGREES),
            "source_offset":OFFSET,"families":list(FAMILIES),
            "public_only":True,"planted_and_private_target_absent":True,
            "attack_witness_and_audit_absent":True,
            "excluded_historical_sources":156,
            "prior_source_ids_sha256":digest(canonical(old)),
            "cases":rows}


def read_fixture(path):
    raw=Path(path).read_bytes()
    f=json.loads(raw)
    old=history()
    assert f["schema"]==FIXTURE_SCHEMA and f["registered_first_commit"]==REGISTERED
    assert f["replicates"]==list(REPS) and f["source_offset"]==OFFSET
    assert f["public_only"] and f["planted_and_private_target_absent"]
    assert f["attack_witness_and_audit_absent"]
    assert f["excluded_historical_sources"]==len(old)==156
    assert f["prior_source_ids_sha256"]==digest(canonical(old))
    ids=[r["full_public_input_sha256"] for r in f["cases"]]
    assert len(ids)==len(set(ids))==24 and set(ids).isdisjoint(old)
    assert {(r["replicate"],r["degree"],r["family"]) for r in f["cases"]}=={
        (rep,d,fam) for rep in REPS for d in DEGREES for fam in FAMILIES}
    assert all(r["split"]=="TEST" and not any(name in r for name in
        ("program","future","target","attack","witness","audit"))
        for r in f["cases"])
    return f,digest(raw)


def public_proposal(row):
    return nor35.proposal(row)


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


def source_geometry(pred,header,public_sha):
    return nor35.source_atoms(pred,header,public_sha)


def phase_tag(src,phase):
    token={"cal":S_CAL,"eval":S_EVAL}[phase]
    return sha256(f"{token}|{phase}|{src}".encode("ascii")).hexdigest()


def new_stochastic_plan(original,cats,scores,phase_key,src,arm,trial,mode):
    """Exactly preserve all original eight atom quotas at k=16."""
    assert mode in EXPLORE_QUANTA
    quotas=[len(set(original)&atom) for atom in cats]
    plan=[]
    behavior={"explore_atoms":0,"targeted_atoms":0,"targeted_prefix_draws":0}
    for j,(atom,t) in enumerate(zip(cats,quotas)):
        if t==0:continue
        ranked=sorted(atom,key=lambda i:(-scores[i],i))
        n=len(ranked)
        assert 1<=t<=n
        rng=random.Random(seed(S_TRANSFORM,phase_key,src,arm,trial,j,mode))
        uniform_branch=rng.randrange(4)<EXPLORE_QUANTA[mode]
        if uniform_branch:
            plan.extend(rng.sample(sorted(atom),t))
            behavior["explore_atoms"]+=1
            continue
        c=(t+1)//2
        window=min(n,max(2*c,(n+7)//8))
        assert c<=window<=n
        selected=rng.sample(ranked[:window],c)
        selected_set=set(selected)
        remaining=sorted(set(atom)-selected_set)
        plan.extend(selected)
        plan.extend(rng.sample(remaining,t-c))
        behavior["targeted_atoms"]+=1
        behavior["targeted_prefix_draws"]+=c
    assert len(plan)==len(set(plan))==K
    assert all(len(set(plan)&atom)==t for plan_atom,t,atom in
               ((j,q,cats[j]) for j,q in enumerate(quotas)))
    return tuple(plan),behavior


def schedules(old,cats,scores,src,phase):
    n=CAL if phase=="cal" else EVAL
    root=phase_tag(src,phase)
    transform_key=sha256(f"{root}|nor37-original-transform".encode("ascii")).hexdigest()
    results={mode:{a:[] for a in ARMS} for mode in MODES}
    behaviors={mode:Counter() for mode in EXPLORE_QUANTA}
    for t in range(n):
        raw=nor25.policy_schedules(old,root,t)
        for arm in ARMS:
            base=tuple(raw[(arm,K)])
            assert len(set(base))==K
            for mode in MODES:
                if mode in EXPLORE_QUANTA:
                    proposed,b=new_stochastic_plan(base,cats,scores,root,src,arm,t,mode)
                    behaviors[mode].update(b)
                else:
                    proposed=nor35.transformed_plan(
                        base,cats,scores,transform_key,arm,t,mode)
                results[mode][arm].append(proposed)
    return results,{mode:dict(c) for mode,c in behaviors.items()}


def components(policy):
    assert policy in POLICIES
    singles={
        "original_h50":("original","hybrid_50"),
        "atom_random_h50":("atom_random","hybrid_50"),
        "deterministic_h50":("atom_curvature50","hybrid_50"),
        "stochastic25_h50":("stochastic25","hybrid_50"),
        "stochastic50_h50":("stochastic50","hybrid_50"),
        "stochastic75_h50":("stochastic75","hybrid_50"),
        "uniform16":("original","uniform"),
    }
    if policy in singles:
        m,a=singles[policy]
        return ((m,a,16),)
    mode={"original_mix":"original","deterministic_mix":"atom_curvature50",
          "stochastic50_mix":"stochastic50"}[policy]
    return tuple((mode,arm,w) for arm,w in zip(ARMS,WEIGHTS) if w)


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
    hits={p[:2]:0 for p in parts}
    for _ in range(R):
        best=None;value=None
        for i in candidates:
            if i in chosen:continue
            cost=sum(w*(hits[(mode,arm)]|cov[(mode,arm)][i]).bit_count()
                     for mode,arm,w in parts)
            if value is None or cost<value:
                best=i;value=cost
        assert best is not None
        chosen.add(best)
        for mode,arm,w in parts:
            hits[(mode,arm)]|=cov[(mode,arm)][best]
    assert len(chosen)==R
    return tuple(sorted(chosen))


def blind_attacks(src,cats,u,cu,cv,bu,bv,m):
    out={a:[] for a in CONTROLS}
    for rep in range(ATTACKS):
        r=random.Random(seed(S_BLIND,src,"top_curvature",rep))
        top=tuple(sorted(r.sample(sorted(cu),2)+r.sample(sorted(cv),2)))
        r=random.Random(seed(S_BLIND,src,"bottom_curvature",rep))
        bottom=tuple(sorted(r.sample(sorted(bu),2)+r.sample(sorted(bv),2)))
        quota=[sum(i in c for i in top) for c in cats]
        assert quota[0]==2 and sum(quota)==R
        r=random.Random(seed(S_BLIND,src,"eight_atom_matched",rep))
        atom=tuple(sorted(i for j,q in enumerate(quota)
                          for i in r.sample(sorted(cats[j]),q)))
        r=random.Random(seed(S_BLIND,src,"full_uniform",rep))
        uniform=tuple(sorted(r.sample(range(m),R)))
        assert len(set(top))==len(set(bottom))==len(set(atom))==len(set(uniform))==R
        assert tuple(sum(i in c for i in atom) for c in cats)==tuple(quota)
        for fam,a in zip(CONTROLS,(top,bottom,atom,uniform)):
            out[fam].append(a)
    return out


def mixed_choice(src,t):
    rng=random.Random(seed(S_MIX,src,t))
    pos=rng.randrange(16)
    a=0
    for arm,w in zip(ARMS,WEIGHTS):
        a+=w
        if pos<a:return arm
    raise AssertionError("quantization")


def exact_top(plans,cu,cv):
    out={}
    for policy in POLICIES:
        total=Fraction(0)
        for mode,arm,w in components(policy):
            for checked in plans[mode][arm]:
                total+=Fraction(w,16)*nor35.two_block_detect(cu,cv,checked)
        out[policy]=total/EVAL
    return out


def exact_control_checks(plans,cats,cu,cv,m):
    """Every k16 plan has uniform-attack equal expectation; quota law
    exactly equal for transforms of the SAME original source/arm/trial."""
    exact_uniform=nor35.ideal_uniform_attack_detect(m)
    fine=nor34.exact_probabilities(cats,cu,cv)
    bad_quota=bad_fine=bad_uniform=0
    for arm in ARMS:
        for t in range(EVAL):
            base=plans["original"][arm][t]
            base_quotas=tuple(len(set(base)&cat) for cat in cats)
            bs=set(base)
            baseline=fine(base_quotas,
                          len(bs&cu),len(bs&cv))["eight_atom"]
            for mode in MODES:
                checked=plans[mode][arm][t]
                atom_quotas=tuple(len(set(checked)&cat) for cat in cats)
                bad_quota+=int(atom_quotas!=base_quotas)
                # Quota induced fine law depends only on atom counts:
                if mode!="original":
                    got=fine(atom_quotas,len(set(checked)&cu),
                             len(set(checked)&cv))["eight_atom"]
                    bad_fine+=int(got!=baseline)
                bad_uniform+=int(nor35.ideal_uniform_attack_detect(m)!=exact_uniform)
    assert bad_quota==bad_fine==bad_uniform==0
    return {"quota_mismatches":bad_quota,"eight_atom_exact_mismatches":bad_fine,
            "uniform_attack_exact_mismatches":bad_uniform}


def source_measures(records):
    pairs=[]
    for rec in records:
        d=rec["crossplay_weighted_numerators"]
        den=rec["crossplay_denominator"]
        top={name:Fraction(*v) for name,v in rec["exact_top_support_detection"].items()}
        self_stoch=d["stochastic50_h50"]["stochastic50_h50"]/den
        self_det=d["deterministic_h50"]["deterministic_h50"]/den
        worst_stoch=min(d[att]["stochastic50_h50"]/den for att in POLICIES)
        worst_det=min(d[att]["deterministic_h50"]/den for att in POLICIES)
        mix_stoch=d["stochastic50_mix"]["stochastic50_mix"]/den
        mix_det=d["deterministic_mix"]["deterministic_mix"]/den
        pairs.append({
            "public_sha256":rec["public_sha256"],"degree":rec["degree"],
            "family":rec["family"],
            "self_stoch50_minus_deterministic":self_stoch-self_det,
            "worst_class_stoch50_minus_deterministic":worst_stoch-worst_det,
            "top_stoch50_minus_atom_random":float(
                top["stochastic50_h50"]-top["atom_random_h50"]),
            "top_deterministic_minus_atom_random":float(
                top["deterministic_h50"]-top["atom_random_h50"]),
            "mix_self_stoch50_minus_deterministic":mix_stoch-mix_det,
            "mix_top_stoch50_minus_original":float(
                top["stochastic50_mix"]-top["original_mix"]),
            "self_stoch25_minus_deterministic":
                d["stochastic25_h50"]["stochastic25_h50"]/den-self_det,
            "self_stoch75_minus_deterministic":
                d["stochastic75_h50"]["stochastic75_h50"]/den-self_det,
            "worst_stoch50":worst_stoch,
            "worst_deterministic":worst_det,
            "self_uniform":d["uniform16"]["uniform16"]/den,
        })
    return pairs


def primary_measures(pairs):
    cols=("self_stoch50_minus_deterministic",
          "worst_class_stoch50_minus_deterministic",
          "top_stoch50_minus_atom_random",
          "top_deterministic_minus_atom_random",
          "mix_self_stoch50_minus_deterministic",
          "mix_top_stoch50_minus_original")
    means={k:statistics.mean(r[k] for r in pairs) for k in cols}
    wins={k:sum(r[k]>0 for r in pairs) for k in cols}
    ratio=(means["top_stoch50_minus_atom_random"]/
           means["top_deterministic_minus_atom_random"]
           if means["top_deterministic_minus_atom_random"]>0 else None)
    gates={
        "stochastic50_selfplay_improves":(
            wins["self_stoch50_minus_deterministic"]>=18 and
            means["self_stoch50_minus_deterministic"]>=.005),
        "stochastic50_worst_calibrated_class_improves":(
            wins["worst_class_stoch50_minus_deterministic"]>=18 and
            means["worst_class_stoch50_minus_deterministic"]>=.005),
        "feature_aligned_curvature_signal_retained":(
            wins["top_stoch50_minus_atom_random"]>=18 and
            means["top_stoch50_minus_atom_random"]>=.020 and
            ratio is not None and ratio>=.25),
        "stochastic50_mixture_resilience_and_top_gain":(
            wins["mix_self_stoch50_minus_deterministic"]>=18 and
            means["mix_self_stoch50_minus_deterministic"]>=.005 and
            wins["mix_top_stoch50_minus_original"]>=18 and
            means["mix_top_stoch50_minus_original"]>=.020)
    }
    return means,wins,ratio,gates


def evaluate(fixture_path):
    fixture,fixture_hash=read_fixture(fixture_path)
    proposed={r["full_public_input_sha256"]:public_proposal(r)
              for r in fixture["cases"]}
    assert len(proposed)==24
    records=[]
    for row in fixture["cases"]:
        src=row["full_public_input_sha256"]
        p=proposed[src]
        pred=p["pred"]
        old,cats,u,cu,cv,bu,bv,scores=source_geometry(pred,p["header"],src)
        cal,cal_behavior=schedules(old,cats,scores,src,"cal")
        exposure=trial_exposure(cal,len(pred))
        learned={policy:[greedy_attack(exposure,policy,src,i) for i in range(ATTACKS)]
                 for policy in POLICIES}
        blind=blind_attacks(src,cats,u,cu,cv,bu,bv,len(pred))
        all_attacks={**learned,**blind}
        evals,eval_behavior=schedules(old,cats,scores,src,"eval")
        chosen=[mixed_choice(src,t) for t in range(EVAL)]
        # All public decisions/attack locations/actual challenge schedules
        # are now frozen. Only here reconstruct original native true future.
        truth=restore_native_target(row,pred)
        assert p["sha"]==nor24.sha_target(truth)
        eval_exposure=trial_exposure(evals,len(pred))
        cross={}
        for name,attacks in all_attacks.items():
            cross[name]={}
            for policy in POLICIES:
                score=0
                for mode,arm,w in components(policy):
                    count=0
                    for a in attacks:
                        union=0
                        for i in a:union|=eval_exposure[(mode,arm)][i]
                        count+=union.bit_count()
                    score+=w*count
                cross[name][policy]=score
        realized={}
        for name,attacks in all_attacks.items():
            realized[name]={}
            for policy in ("original_mix","deterministic_mix","stochastic50_mix"):
                mode={"original_mix":"original","deterministic_mix":"atom_curvature50",
                      "stochastic50_mix":"stochastic50"}[policy]
                realized[name][policy]=sum(
                    int(bool(set(evals[mode][chosen[t]][t])&set(a)))
                    for t in range(EVAL) for a in attacks)
        bad_witness=bad_informed=direct_checks=0
        for mode,arms in evals.items():
            for arm,plans in arms.items():
                for plan in plans:
                    if not nor24.exact_witness_accepts(pred,truth,plan,frozenset(),1):
                        bad_witness+=1
                    informed=nor24.informed_attack(plan,old["uniform"],R)
                    bad_informed+=int(bool(set(plan)&informed))
        assert bad_witness==bad_informed==0
        for name,attacksets in all_attacks.items():
            for policy in POLICIES:
                for attack in attacksets[:2]:
                    for mode,arm,w in components(policy):
                        for t in range(2):
                            plan=evals[mode][arm][t]
                            hit=bool(set(plan)&set(attack))
                            check1=not nor24.exact_witness_accepts(
                                pred,truth,plan,frozenset(attack),DELTAS[0])
                            check2=not nor24.exact_witness_accepts(
                                pred,truth,plan,frozenset(attack),DELTAS[1])
                            assert check1==check2==hit
                            direct_checks+=1
        exact_checks=exact_control_checks(evals,cats,cu,cv,len(pred))
        floor=sum((Fraction(w,16)*nor34.ideal_floor(len(pred),a)
                   for a,w in zip(ARMS,WEIGHTS)),Fraction(0))
        assert floor>0
        top=exact_top(evals,cu,cv)
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
            "evaluation_plans_sha256":digest(canonical(evals)),
            "calibration_stochastic_branch_counts":cal_behavior,
            "evaluation_stochastic_branch_counts":eval_behavior,
            "greedy_attack_sets":{k:[list(a) for a in v] for k,v in learned.items()},
            "blind_attack_sets":{k:[list(a) for a in v] for k,v in blind.items()},
            "greedy_attack_unique_sets":{k:len(set(v)) for k,v in learned.items()},
            "crossplay_weighted_numerators":cross,
            "crossplay_denominator":16*ATTACKS*EVAL,
            "realized_mixture_arm_counts":dict(Counter(chosen)),
            "realized_mixture_attack_hits":realized,
            "exact_top_support_detection":{
                name:[v.numerator,v.denominator] for name,v in top.items()},
            "exact_full_uniform_attack_detection_fraction":[
                nor35.ideal_uniform_attack_detect(len(pred)).numerator,
                nor35.ideal_uniform_attack_detect(len(pred)).denominator],
            "original_mixture_conditional_blind_floor_fraction":[
                floor.numerator,floor.denominator],
            "honest_target_false_rejections":bad_witness,
            "fully_schedule_informed_attacker_detections":bad_informed,
            "delta_decision_disagreements":0,
            "exact_quota_and_control_checks":exact_checks,
            "direct_authentic_witness_checks":direct_checks,
        })
    pairs=source_measures(records)
    means,wins,ratio,gates=primary_measures(pairs)
    good=all(r["native_mask"]>0 and
             r["public_prediction_sha256"]==r["private_native_truth_sha256_after_freeze"]
             and r["honest_target_false_rejections"]==0
             and r["fully_schedule_informed_attacker_detections"]==0
             and r["delta_decision_disagreements"]==0
             and all(v==0 for v in r["exact_quota_and_control_checks"].values())
             and r["direct_authentic_witness_checks"]>0
             and Fraction(*r["original_mixture_conditional_blind_floor_fraction"])>0
             for r in records)
    gates["original_source_native_witness_and_full_replay"]=good
    return {
        "schema":RESULT_SCHEMA,"registration_first_commit":REGISTERED,
        "public_fixture_sha256":fixture_hash,"fresh_source_units":len(records),
        "historical_sources_excluded":156,"auditor_policies":list(POLICIES),
        "component_auditor_arms":list(ARMS),"original_mix_weights":list(WEIGHTS),
        "transform_modes":list(MODES),"blind_control_families":list(CONTROLS),
        "calibration_schedules_per_source":CAL,
        "heldout_schedules_per_source":EVAL,
        "greedy_restarts_per_policy":ATTACKS,"audit_budget":K,"attack_radius":R,
        "nonzero_deltas":list(DELTAS),
        "public_proposals_frozen_before_private_truth":True,
        "calibrated_attacks_frozen_before_independent_heldout_schedules":True,
        "authentic_external_target_coefficient_witness_assumed":True,
        "source_records":records,"source_level_primary":pairs,
        "source_means":means,"source_direction_wins":wins,
        "retained_ratio_stochastic_to_deterministic_top_gain":ratio,
        "registered_five_gates":gates,"registered_primary_pass":all(gates.values()),
        "scope":("Source-conditional synthetic RRC25, new random curvature top-window "
                 "and per-atom full-uniform exploration. Calibrated 256/8 greedy "
                 "attacks freeze before 512 evaluation schedules. True target "
                 "coefficient witness and actual independent hidden coins assumed; "
                 "public SHA seeded research streams provide deterministic replay."),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--make-fixture",type=Path)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.make_fixture:
        if a.fixture or a.out:p.error("fixture generation and grading are separate")
        raw=canonical(make_fixture())
        a.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        a.make_fixture.write_bytes(raw)
        print("NOR37 FIRST FROZEN NEW PUBLIC SOURCE FIXTURE",len(raw),digest(raw))
        return
    if not a.fixture:p.error("--fixture required")
    result=evaluate(a.fixture)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_bytes(canonical(result))
    print("NOR37 FIRST REGISTERED PRIMARY",result["registered_primary_pass"])
    print("NOR37 FIVE GATES",result["registered_five_gates"])
    print("NOR37 SOURCE WINS",result["source_direction_wins"])
    print("NOR37 SOURCE MEANS",result["source_means"])
    print("NOR37 RETAINED TOP GAIN RATIO",result["retained_ratio_stochastic_to_deterministic_top_gain"])


if __name__=="__main__":
    main()
