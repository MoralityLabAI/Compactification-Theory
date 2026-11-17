"""NOR-41 independent native interpreter and heldout graded crossplay replayer.

Independent of research.nor41_scale_ood.experiment, and never calls its
evaluator. Reuses original NOR21/NOR25 historical input grammar and the
NOR37 independent-replayer geometry, with new standalone NOR40 policy logic and stronger CAL search.
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

REGISTERED="87dd3de03f43e79d83d45ed7a5c3a3bb1f20746f"
ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
MODES=("original","atom_random","atom_curvature50")
POLICIES=("uniform16","original_mix","deterministic_mix","global7_mix")
DIAGNOSTIC="atom_random_h50"
CONTROLS=("top_curvature","bottom_curvature","eight_atom_matched",
          "full_uniform","low_cal_exposure","zero_quota_stress")
REPS=tuple(range(84,100))
DEGREES=(512,1024)
FAMILIES=("wide_dense_small","wide_regime_blocks")
OFFSET=130041000
K,R,CAL,EVAL,ATTACKS=16,4,4096,1024,8
BUDGETS=(256,512,2048,4096)
GREEDY_STARTS=16
POOL_SIZE=16
SWAP_PASSES=4
TRANSFORM_STREAM={"atom_random":202611350076,"atom_curvature50":202611350077}
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
    with ZipFile(ROOT/"research/nor38_global_uniform/results/nor38_original_first_hosted.zip") as z:
        a=z.read("nor38_fixture.json")
        b=z.read("nor38_full.json")
    assert len(a)==9319507 and sha256(a).hexdigest()==(
        "37fa2aeedeb5ab54bc1082ec2d50f8a084c4f4a7d6279acd5d122396d760e504")
    assert len(b)==929756 and sha256(b).hexdigest()==(
        "9c69d663d4d1511e664e930c0f8ed42d8d6f59c4d10557323fb5ee54478c036d")
    source=json.loads(a)
    assert source["schema"]=="nor38.global_uniform_public_only_reps60to67.v1"
    past.extend(row["full_public_input_sha256"] for row in source["cases"])
    assert len(past)==len(set(past))==228
    with ZipFile(ROOT/"research/nor39_clean_global_uniform/results/nor39_original_first_hosted.zip") as z:
        a=z.read("nor39_fixture.json")
        b=z.read("nor39_full.json")
    assert len(a)==8457812 and sha256(a).hexdigest()==(
        "3dd23e71663d3c179ecacd437892c650469a6980230e71c353840f3b8f5a290c")
    assert len(b)==930059 and sha256(b).hexdigest()==(
        "995b6e110c829fd9a8ebb036e4e08ef64cfccee8298bff0def255c53f35c17c6")
    fixture=json.loads(a)
    assert fixture["schema"]=="nor39.clean_global_uniform_public_only_reps68to75.v1"
    past.extend(row["full_public_input_sha256"] for row in fixture["cases"])
    assert len(past)==len(set(past))==276
    with ZipFile(ROOT/"research/nor40_strong_adversaries/results/nor40_original_first_hosted.zip") as z:
        original=z.read("nor40_fixture.json")
        graded=z.read("nor40_full.json")
    assert len(original)==9387973 and sha256(original).hexdigest()==(
        "b51e8031715754c36661b74cff0c26e672fdc21355ff0dc9ab48efded3901de5")
    assert len(graded)==3492789 and sha256(graded).hexdigest()==(
        "42adeebcbf92626c080a6f0f5cb906eed491d6b5ed67250ba7b70bd6d0559092")
    old=json.loads(original)
    assert old["schema"]=="nor40.strong_adversary_public_only_reps76to83.v1"
    past.extend(a["full_public_input_sha256"] for a in old["cases"])
    assert len(past)==len(set(past))==324
    return sorted(past)


def independent_input(rng,family,degree):
    if family=="wide_dense_small":
        return (1,)+tuple(rng.choice((-2,-1,1,2)) for _ in range(degree))
    if family!="wide_regime_blocks":raise ValueError("unexpected NOR41 family")
    output=[1]
    for index in range(1,degree+1):
        kind=((index-1)//64)%4
        draw=rng.random()
        if kind==0:emit=draw<.90
        elif kind==1:emit=draw<.04
        elif kind==2:emit=draw<.16 and index%16<4
        else:emit=draw<.25
        output.append(rng.choice((-2,-1,1,2)) if emit else 0)
    return tuple(output)


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
        demos=tuple((x:=independent_input(rng,fam,3*d//2),
                     corpus.execute(x,planted)) for _ in range(2))
        query=independent_input(rng,fam,3*d//2)
    assert planted is not None
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    hr=row["public_header"]
    header=corpus.Header(hr["base"],hr["delay"],hr["spacing"],hr["amplitude"])
    assert planted[:4]==(header.base,header.delay,header.spacing,header.amplitude)
    c=corpus.Case("nor41-independent",fam,
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
    assert policy in POLICIES or policy==DIAGNOSTIC
    if policy=="uniform16":return (("original","uniform",256),)
    if policy==DIAGNOSTIC:return (("atom_random","hybrid_50",256),)
    if policy in ("original_mix","deterministic_mix"):
        m="original" if policy=="original_mix" else "atom_curvature50"
        return tuple((m,arm,16*w) for arm,w in zip(ARMS,WEIGHTS) if w)
    return (("original","uniform",112),)+tuple(
        ("atom_curvature50",arm,9*w) for arm,w in zip(ARMS,WEIGHTS) if w)


def policy_schedules(old,atoms,scores,src,phase):
    n={"cal":CAL,"eval":EVAL}[phase]
    stream={"cal":202611410071,"eval":202611410072}[phase]
    root=sha256(f"{stream}|nor41|{phase}|{src}".encode("ascii")).hexdigest()
    key=sha256(f"{202611410076}|{root}".encode("ascii")).hexdigest()
    out={mode:{arm:[] for arm in ARMS} for mode in MODES}
    for t in range(n):
        original=original25.policy_schedules(old,root,t)
        for arm in ARMS:
            ref=tuple(original[(arm,K)])
            out["original"][arm].append(ref)
            for mode in ("atom_random","atom_curvature50"):
                out[mode][arm].append(original_replan(ref,atoms,scores,key,arm,t,mode))
    return out

def independent_cal_vectors(plans,m):
    """Build a fresh bitset per coordinate from source-public CAL schedule records."""
    rows={}
    for mode,arms in plans.items():
        for arm,schedules in arms.items():
            arr=[0 for _ in range(m)]
            for t,plan in enumerate(schedules):
                bit=1<<t
                for index in plan:arr[index]|=bit
            rows[(mode,arm)]=arr
    return rows


def independent_cost(visited,policy,attack):
    total=0
    for mode,arm,weight in component_parts(policy):
        coverage=0
        for i in attack:coverage|=visited[(mode,arm)][i]
        total+=weight*coverage.bit_count()
    return total


def independent_greedy(visited,policy,src,budget,restart):
    order=list(range(len(visited[component_parts(policy)[0][:2]])))
    random.Random(seed(202611410073,src,policy,budget,restart)).shuffle(order)
    selected=set()
    seen={part[:2]:0 for part in component_parts(policy)}
    for _ in range(R):
        best=None;cost=None
        for idx in order:
            if idx in selected:continue
            value=sum(w*(seen[(mode,arm)]|visited[(mode,arm)][idx]).bit_count()
                      for mode,arm,w in component_parts(policy))
            if cost is None or value<cost:
                best,cost=idx,value
        assert best is not None
        selected.add(best)
        for mode,arm,_ in component_parts(policy):
            seen[(mode,arm)]|=visited[(mode,arm)][best]
    return tuple(sorted(selected))


def independent_refinement(visited,policy,start):
    state=tuple(sorted(start))
    score=independent_cost(visited,policy,state)
    m=len(visited[component_parts(policy)[0][:2]])
    for _ in range(SWAP_PASSES):
        candidate=state;value=score
        for drop in state:
            prefix=tuple(i for i in state if i!=drop)
            old=set(prefix)
            accum={}
            for mode,arm,_ in component_parts(policy):
                visited_ids=0
                for i in prefix:visited_ids|=visited[(mode,arm)][i]
                accum[(mode,arm)]=visited_ids
            for i in range(m):
                if i in old:continue
                A=tuple(sorted((*prefix,i)))
                cost=sum(w*(accum[(mode,arm)]|visited[(mode,arm)][i]).bit_count()
                         for mode,arm,w in component_parts(policy))
                if cost<value or (cost==value and cost<score and A<candidate):
                    candidate,value=A,cost
        if value>=score:break
        state,score=candidate,value
    return state


def independent_nested_games(plans,src,m):
    full=independent_cal_vectors(plans,m)
    trained={}
    certificates={}
    from itertools import combinations
    for budget in BUDGETS:
        exposure={p:[v&((1<<budget)-1) for v in seq] for p,seq in full.items()}
        groups={};audit={}
        for policy in POLICIES:
            greedy=[independent_greedy(exposure,policy,src,budget,rep)
                    for rep in range(GREEDY_STARTS)]
            rating=lambda A:(independent_cost(exposure,policy,A),A)
            ranked=sorted(greedy,key=rating)
            swaps=[independent_refinement(exposure,policy,A) for A in ranked[:2]]
            singles=[sum(w*exposure[(mode,arm)][i].bit_count()
                         for mode,arm,w in component_parts(policy)) for i in range(m)]
            pool=list(min(greedy,key=rating))
            for i in sorted(range(m),key=lambda j:(singles[j],j)):
                if i not in pool:pool.append(i)
                if len(pool)==16:break
            pool=tuple(sorted(pool))
            optimum=min((independent_cost(exposure,policy,A),A)
                        for A in combinations(pool,R))
            candidates=sorted(set(greedy+swaps+[optimum[1]]),key=rating)
            chosen=(candidates[:8]+[candidates[0]]*8)[:8]
            selected_sum=sum(independent_cost(exposure,policy,a) for a in chosen)
            greedy_sum=sum(independent_cost(exposure,policy,a) for a in ranked[:8])
            lower=sorted(singles)[3]
            upper=independent_cost(exposure,policy,chosen[0])
            assert 0<=lower<=upper<=optimum[0] and selected_sum<=greedy_sum
            groups[policy]=chosen
            audit[policy]={
                "budget":budget,
                "greedy16_sets":[list(a) for a in greedy],
                "swap_refined_top2_sets":[list(a) for a in swaps],
                "restricted_candidate_pool":list(pool),
                "restricted_exact_optimum":list(optimum[1]),
                "restricted_exact_value":optimum[0],
                "restricted_sets_enumerated":1820,
                "global_singleton_lower_bound":lower,
                "global_best_found_upper_bound":upper,
                "global_optimality_gap":upper-lower,
                "greedy_best8_cal_sum":greedy_sum,
                "selected_best8_cal_sum":selected_sum,
                "selected_attack_uniques":len(set(chosen)),
                "solver_scope":"exact_pool16; lower is global fourth singleton"
            }
        trained[str(budget)]=groups
        certificates[str(budget)]=audit
    return trained,certificates,full

def independent_blind(src,atoms,u,cu,cv,bu,bv,m,visited):
    exposure=[sum(w*visited[(mode,arm)][i].bit_count()
                  for mode,arm,w in component_parts("deterministic_mix"))
              for i in range(m)]
    pool=sorted(range(m),key=lambda i:(exposure[i],i))[:min(m,96)]
    outside=sorted((i for i in range(m) if i not in u),
                   key=lambda i:(exposure[i],i))
    stress_pool=(outside if len(outside)>=R else
                 sorted(range(m),key=lambda i:(exposure[i],i)))[:min(m,96)]
    out={name:[] for name in CONTROLS}
    for t in range(ATTACKS):
        rng=random.Random(seed(202611410074,src,"top_curvature",t))
        top=tuple(sorted(rng.sample(sorted(cu),2)+rng.sample(sorted(cv),2)))
        rng=random.Random(seed(202611410074,src,"bottom_curvature",t))
        bottom=tuple(sorted(rng.sample(sorted(bu),2)+rng.sample(sorted(bv),2)))
        quotas=[sum(i in cat for i in top) for cat in atoms]
        rng=random.Random(seed(202611410074,src,"eight_atom_matched",t))
        matched=tuple(sorted(i for j,q in enumerate(quotas)
                             for i in rng.sample(sorted(atoms[j]),q)))
        rng=random.Random(seed(202611410074,src,"full_uniform",t))
        uniform=tuple(sorted(rng.sample(range(m),R)))
        rng=random.Random(seed(202611410074,src,"low_cal_exposure",t))
        low=tuple(sorted(rng.sample(pool,R)))
        rng=random.Random(seed(202611410074,src,"zero_quota_stress",t))
        stress=tuple(sorted(rng.sample(stress_pool,R)))
        for name,attack in zip(CONTROLS,(top,bottom,matched,uniform,low,stress)):
            assert len(attack)==len(set(attack))==R
            out[name].append(attack)
        assert [sum(i in atom for i in matched) for atom in atoms]==quotas
    return out


def realized_choice(src,t,policy):
    assert policy in ("original_mix","deterministic_mix","global7_mix")
    rng=random.Random(seed(202611410075,"eval",src,t,policy))
    if policy=="global7_mix":
        if rng.randrange(16)<7:return ("original","uniform")
        mode="atom_curvature50"
    else:
        mode="original" if policy=="original_mix" else "atom_curvature50"
    position=rng.randrange(16)
    for arm,w in zip(ARMS,WEIGHTS):
        if position<w:return (mode,arm)
        position-=w
    raise AssertionError("invalid policy mix")


def exact_top(plans,cu,cv):
    output={}
    for p in (*POLICIES,DIAGNOSTIC):
        total=Fraction(0)
        for mode,arm,w in component_parts(p):
            total+=Fraction(w,DENOM)*sum(
                (top_prob(cu,cv,Q) for Q in plans[mode][arm]),Fraction(0))/EVAL
        output[p]=total
    return output

def scored_rows(records):
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



def replay(fixture_raw,graded_raw):
    fixture=json.loads(fixture_raw)
    grade=json.loads(graded_raw)
    historical=historical_source_ids()
    assert fixture["schema"]=="nor40.strong_adversary_public_only_reps76to83.v1"
    assert grade["schema"]=="nor40.strong_calibration_policyaware.v1"
    assert fixture["registered_first_commit"]==grade["registration_first_commit"]==REGISTERED
    assert fixture["public_only"] and fixture["planted_and_private_target_absent"]
    assert fixture["attack_witness_and_audit_absent"]
    assert fixture["replicates"]==list(REPS) and fixture["degrees"]==list(DEGREES)
    assert fixture["excluded_historical_sources"]==len(historical)==276
    assert fixture["prior_source_ids_sha256"]==sha256(canonical(historical)).hexdigest()
    assert grade["public_fixture_sha256"]==sha256(fixture_raw).hexdigest()
    assert grade["fresh_source_units"]==48 and grade["historical_sources_excluded"]==276
    assert grade["auditor_policies"]==list(POLICIES)
    assert grade["blind_control_families"]==list(CONTROLS)
    assert grade["transform_modes"]==list(MODES)
    assert grade["original_mix_weights"]==list(WEIGHTS)
    assert grade["global_uniform_probabilities"]=={"global7_mix":[7,16],"global8_mix":[8,16]}
    assert grade["calibration_schedules_per_source"]==CAL
    assert grade["baseline_calibration_schedules_per_source"]==BASE_CAL
    assert grade["heldout_schedules_per_source"]==EVAL
    assert grade["greedy_restarts_per_policy"]==GREEDY_STARTS
    assert grade["attack_sets_per_policy"]==ATTACKS
    assert grade["restricted_pool_size"]==POOL_SIZE
    assert grade["restricted_sets_per_policy"]==1820
    assert grade["swap_passes_cap"]==SWAP_PASSES
    assert grade["audit_budget"]==K and grade["attack_radius"]==R
    assert grade["public_proposals_frozen_before_private_truth"]
    assert grade["calibrated_attacks_frozen_before_independent_heldout_schedules"]
    assert grade["authentic_external_target_coefficient_witness_assumed"]
    ids=[row["full_public_input_sha256"] for row in fixture["cases"]]
    assert len(ids)==len(set(ids))==len(fixture["cases"])==len(grade["source_records"])==48
    assert set(ids).isdisjoint(historical)
    assert {(r["replicate"],r["degree"],r["family"]) for r in fixture["cases"]}=={
        (rep,d,fam) for rep in REPS for d in DEGREES for fam in FAMILIES}
    decision_count=0
    all_witness=0
    for idx,(row,rec) in enumerate(zip(fixture["cases"],grade["source_records"],strict=True)):
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
        cal,cal_count=policy_schedules(old,atoms,scores,src,"cal")
        assert rec["calibration_plans_sha256"]==sha256(canonical(cal)).hexdigest()
        assert rec["calibration_stochastic_branch_counts"]==cal_count
        baseline,strong,certificate,visited=independent_strong_games(cal,src,len(pred))
        assert rec["baseline_attack_sets"]=={
            p:[list(a) for a in variants] for p,variants in baseline.items()}
        assert rec["greedy_attack_sets"]=={
            p:[list(a) for a in variants] for p,variants in strong.items()}
        assert rec["greedy_attack_unique_sets"]=={
            p:len(set(variants)) for p,variants in strong.items()}
        assert rec["attack_certificates"]==certificate
        blind=independent_blind(src,atoms,u,cu,cv,bu,bv,len(pred),visited)
        assert rec["blind_attack_sets"]=={
            p:[list(a) for a in variants] for p,variants in blind.items()}
        attacks={**strong,**{"baseline__"+p:v for p,v in baseline.items()},**blind}
        assert len(attacks)==20
        ev,eval_counts=policy_schedules(old,atoms,scores,src,"eval")
        assert rec["evaluation_plans_sha256"]==sha256(canonical(ev)).hexdigest()
        assert rec["evaluation_stochastic_branch_counts"]==eval_counts
        choices={p:[realized_choice(src,t,p) for t in range(EVAL)]
                 for p in ("original_mix","deterministic_mix","stochastic50_mix",
                           "global7_mix","global8_mix")}
        realbranches={p:{"uniform":sum(mode=="original" and arm=="uniform"
                                        for mode,arm in selected),
                         "curvature":sum(mode=="atom_curvature50"
                                        for mode,arm in selected)}
                      for p,selected in choices.items()}
        assert rec["realized_mixture_branch_counts"]==realbranches
        ev_sets={(mode,arm):[set(plan) for plan in ev[mode][arm]]
                 for mode in MODES for arm in ARMS}
        for arm in ARMS:
            for t in range(EVAL):
                reference=ev_sets[("original",arm)][t]
                quotas=[len(reference&cat) for cat in atoms]
                for mode in MODES:
                    inspected=ev_sets[(mode,arm)][t]
                    assert len(inspected)==K and [len(inspected&cat) for cat in atoms]==quotas
        cross={}
        realized={}
        for name,variants in attacks.items():
            cross[name]={}
            for policy in POLICIES:
                total=0
                for mode,arm,weight in component_parts(policy):
                    for altered in variants:
                        A=set(altered)
                        total+=weight*sum(not read.isdisjoint(A)
                                          for read in ev_sets[(mode,arm)])
                        decision_count+=EVAL
                cross[name][policy]=total
            realized[name]={}
            for policy,selected in choices.items():
                realized[name][policy]=sum(
                    int(not ev_sets[(mode,arm)][t].isdisjoint(set(altered)))
                    for t,(mode,arm) in enumerate(selected) for altered in variants)
        assert cross==rec["crossplay_weighted_numerators"]
        assert rec["crossplay_denominator"]==DENOM*ATTACKS*EVAL
        assert realized==rec["realized_mixture_attack_hits"]
        top=exact_top(ev,cu,cv)
        assert rec["exact_top_support_detection"]=={
            p:[v.numerator,v.denominator] for p,v in top.items()}
        uniform=Fraction(1)-Fraction(comb(len(pred)-K,R),comb(len(pred),R))
        assert rec["exact_full_uniform_attack_detection_fraction"]==[
            uniform.numerator,uniform.denominator]
        assert rec["global7_exact_theoretical_floor"]==[
            (Fraction(7,16)*uniform).numerator,(Fraction(7,16)*uniform).denominator]
        assert rec["global8_exact_theoretical_floor"]==[
            (Fraction(1,2)*uniform).numerator,(Fraction(1,2)*uniform).denominator]
        assert rec["honest_target_false_rejections"]==0
        assert rec["fully_schedule_informed_attacker_detections"]==0
        assert rec["delta_decision_disagreements"]==0
        assert all(v==0 for v in rec["exact_quota_and_control_checks"].values())
        direct_checks=0
        for name,variants in attacks.items():
            for policy in POLICIES:
                for changed in variants[:2]:
                    A=set(changed)
                    for mode,arm,w in component_parts(policy):
                        for t in range(2):
                            checked=ev_sets[(mode,arm)][t]
                            # Reconstruct authentic output integers explicitly.
                            for delta in (1,4294049777):
                                proposed=list(pred)
                                for i in A:proposed[i]+=delta
                                rejected=any(proposed[i]!=pred[i] for i in checked)
                                assert rejected==bool(checked&A)
                            direct_checks+=1
        assert rec["direct_authentic_witness_checks"]==direct_checks
        all_witness+=direct_checks
        print("NOR40 independently replayed source",idx+1,"/48",flush=True)
    rows=scored_rows(grade["source_records"])
    assert len(rows)==len(grade["source_level_primary"])==48
    for got,expected in zip(grade["source_level_primary"],rows,strict=True):
        assert got.keys()==expected.keys()
        for k,v in expected.items():
            if isinstance(v,float):
                assert isclose(got[k],v,rel_tol=0,abs_tol=1e-13),(k,got[k],v)
            else:assert got[k]==v,(k,got[k],v)
    columns=("global7_baseline_minus_strong_selfplay",
             "global7_strong_self_minus_deterministic_strong",
             "global7_strong_worst_minus_original_strong_worst",
             "global7_top_minus_atom_random",
             "deterministic_top_minus_atom_random")
    means={k:statistics.mean(row[k] for row in rows) for k in columns}
    wins={
        "strong_global7_beats_baseline_eval":sum(
            row["global7_baseline_minus_strong_selfplay"]>0 for row in rows),
        "global7_strong_self_over_deterministic":sum(
            row["global7_strong_self_minus_deterministic_strong"]>0 for row in rows),
        "global7_strong_worst_within_1pp_original":sum(
            row["global7_strong_worst_minus_original_strong_worst"]>=-.010
            for row in rows),
        "global7_top_positive":sum(
            row["global7_top_minus_atom_random"]>0 for row in rows)}
    topden=means["deterministic_top_minus_atom_random"]
    ratio=means["global7_top_minus_atom_random"]/topden if topden>0 else None
    certificate_good=all(
        all(rec["attack_certificates"][p]["strong_eight_full_cal_sum"]<=
            rec["attack_certificates"][p]["baseline_eight_full_cal_sum"]
            for p in POLICIES) for rec in grade["source_records"])
    gate={
        "stronger_attacker_heldout_transfer":(
            wins["strong_global7_beats_baseline_eval"]>=24 and
            means["global7_baseline_minus_strong_selfplay"]>=.001 and certificate_good),
        "global7_robustness_vs_strong_attack_classes":(
            wins["global7_strong_self_over_deterministic"]>=36 and
            means["global7_strong_self_minus_deterministic_strong"]>=.010 and
            wins["global7_strong_worst_within_1pp_original"]>=36 and
            means["global7_strong_worst_minus_original_strong_worst"]>=-.010 and
            statistics.mean(r["global7_strong_worst"] for r in rows)>=.020),
        "global7_retains_frozen_top_curvature_signal":(
            wins["global7_top_positive"]>=36 and
            means["global7_top_minus_atom_random"]>=.020 and
            ratio is not None and ratio>=.20),
        "exact_calibration_brackets_and_native_witness":True,
        "original_source_first_and_independent_replay_contract":True
    }
    for k,v in means.items():
        assert isclose(v,grade["source_means"][k],rel_tol=0,abs_tol=1e-13)
    assert wins==grade["source_direction_wins"]
    if ratio is None:
        assert grade["retained_ratio_global7_to_deterministic_top_gain"] is None
    else:
        assert isclose(ratio,grade["retained_ratio_global7_to_deterministic_top_gain"],
                       abs_tol=1e-13)
    assert grade["registered_five_gates"]==gate
    assert grade["registered_primary_pass"]==all(gate.values())
    blocks={str(rep):{
        k:statistics.mean(row[k] for row in rows if row["replicate"]==rep)
        for k in ("global7_baseline_minus_strong_selfplay",
                  "global7_strong_self_minus_deterministic_strong",
                  "global7_strong_worst_minus_original_strong_worst",
                  "global7_top_minus_atom_random")}
        for rep in REPS}
    assert grade["replicate_block_means"]==blocks
    expected_count=48*20*ATTACKS*EVAL*sum(len(component_parts(p)) for p in POLICIES)
    assert decision_count==expected_count
    return {"replayed_new_sources":48,"excluded_historical_sources":276,
            "baseline_attack_sets":48*len(POLICIES)*ATTACKS,
            "full2048_greedy_starts":48*len(POLICIES)*GREEDY_STARTS,
            "strong_attack_sets":48*len(POLICIES)*ATTACKS,
            "blind_attack_sets":48*len(CONTROLS)*ATTACKS,
            "component_cal_eval_plans":48*len(MODES)*len(ARMS)*(CAL+EVAL),
            "crossplay_world_decisions":decision_count,
            "native_integer_witness_pairs":all_witness,
            "source_means":means,"source_wins":wins,
            "science_pass":all(gate.values())}


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--fixture",type=Path,required=True)
    a.add_argument("--full",type=Path,required=True)
    a.add_argument("--check",action="store_true")
    args=a.parse_args()
    results=replay(args.fixture.read_bytes(),args.full.read_bytes())
    print("NOR40 INDEPENDENT EXACT SOURCE / GREEDY / CROSSPLAY REPLAY",results)
    if args.check:
        assert results["replayed_new_sources"]==48
        assert results["crossplay_world_decisions"]==48*20*ATTACKS*EVAL*sum(len(component_parts(p)) for p in POLICIES)


if __name__=="__main__":
    main()
