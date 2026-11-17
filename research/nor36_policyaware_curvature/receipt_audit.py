"""NOR-36 independent source, atom, CAL greedy attack and EVAL cross-play replay.

Does NOT import the NOR36 applied evaluator. Regenerates every original
RRC25 planted program with the independent NOR21 source interpreter,
rebuilds policy transforms and calibration attack proposals in separate
functions, then grades all heldout coordinate intersections from sets
rather than the applied scorer's trial-ID exposure bitsets. It also
recomputes the exact conditional top-support Fraction controls.
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
from research.nor21_degree_scaling import receipt_audit as original
from research.nor25_robust_audits import experiment as old25
from research.nor35_intraatom_curvature import receipt_audit as original35

ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
POLICIES=("original_h50","atom_random_h50","atom_curv_h50",
          "uniform16","original_mix","atom_curv_mix")
CONTROLS=("top_curvature","bottom_curvature","eight_atom_matched","full_uniform")
REPS=(52,53,54,55)
DEGREES=(512,1024)
FAMILIES=("wide_dense_small","wide_sparse_small","wide_rational_short")
CAL=256
EVAL=512
R=4
K=16
ATTACKS=8
FIRST="ede05db28b814d5000ac855fecb3b762876c6f9e"


def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()


def seed(*parts):
    return int.from_bytes(sha256("|".join(map(str,parts)).encode("ascii")).digest(),"big")


def future_sha(pred):
    return sha256(json.dumps(pred,separators=(",",":")).encode()).hexdigest()


def historical_roster():
    previous=original35.old_history()
    p=ROOT/"research/nor35_intraatom_curvature/results/nor35_original_hosted.zip"
    with ZipFile(p) as z:
        a=z.read("nor35_fixture.json")
        b=z.read("nor35_full.json")
    assert len(a)==4539616 and sha256(a).hexdigest()==(
        "7a517c8a9f87b15e12362da2dd99d188933a018a233fe2d043ea4aca9c4b02ef")
    assert len(b)==706150 and sha256(b).hexdigest()==(
        "194a0709ed5672dbb8b6706b4d40d6f8f3a28e28164dc3275005d8421ccd05a6")
    ids=sorted(previous+[c["full_public_input_sha256"]
                         for c in json.loads(a)["cases"]])
    assert len(ids)==len(set(ids))==132
    return ids


def source(row):
    rep,d,fam=row["replicate"],row["degree"],row["family"]
    assert rep in REPS and d in DEGREES and fam in FAMILIES
    expected=(corpus.SEED+130036000+1000003*rep+
              100009*DEGREES.index(d)+10007*FAMILIES.index(fam))
    assert row["source_seed"]==expected
    assert row["query_degree"]==3*d//2
    assert 0<=row["selected_attempt"]<100
    rng=random.Random(expected)
    planted=demos=query=None
    for _ in range(row["selected_attempt"]+1):
        planted=(rng.choice(corpus.OPS),rng.choice(corpus.DELAYS),
                 rng.choice(corpus.SPACINGS),rng.choice(corpus.AMPLITUDES),
                 rng.choice(corpus.OPS),rng.choice(corpus.OPS))
        demos=tuple((x:=original.independent_source(rng,fam,3*d//2),
                     corpus.execute(x,planted)) for _ in range(2))
        query=original.independent_source(rng,fam,3*d//2)
    assert planted is not None
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    hh=row["public_header"]
    header=corpus.Header(hh["base"],hh["delay"],hh["spacing"],hh["amplitude"])
    assert planted[:4]==(header.base,header.delay,header.spacing,header.amplitude)
    c=corpus.Case("nor36-independent-public",fam,
        tuple((tuple(x),tuple(y)) for x,y in row["demos"]),
        tuple(query),(),None)
    assert c.program is None and c.future==()
    assert original.public_source_hash(c,d)==row["full_public_input_sha256"]
    mask=original.independent_native_mask(c,header,d)
    assert mask==row["original_native_public_mask"] and mask>0
    j=next(i for i in range(16) if mask&(1<<i))
    pred=corpus.execute(query,original.candidate_program(header,j))
    for i in range(16):
        if mask&(1<<i):
            assert corpus.execute(query,original.candidate_program(header,i))==pred
    assert corpus.execute(query,planted)==pred
    return mask,j,pred,header


def public_atom_geometry(pred,header):
    m=len(pred)
    assert m in (769,1537)
    tail=set(range(m-96,m))
    magnitude=set(sorted(range(m),key=lambda i:(-abs(pred[i]),i))[:96])
    lag=set(range(header.delay+header.spacing,header.delay+header.spacing+96))
    old={"tail":tail,"magnitude":magnitude,"lag":lag,
         "uniform":set(range(m))}
    u=tail|magnitude|lag
    old["off_template"]=set(range(m))-u
    atoms=[set() for _ in range(8)]
    score=[abs(pred[(i-1)%m]-2*pred[i]+pred[(i+1)%m])
           for i in range(m)]
    for i in range(m):
        j=int(i in tail)+2*int(i in magnitude)+4*int(i in lag)
        atoms[j].add(i)
    assert len(atoms)==8 and sum(map(len,atoms))==m
    ranktop=sorted(range(m),key=lambda i:(-score[i],i))
    rankbottom=sorted(range(m),key=lambda i:(score[i],i))
    cu=set([i for i in ranktop if i in u][:48])
    cv=set([i for i in ranktop if i not in u][:48])
    bu=set([i for i in rankbottom if i in u][:48])
    bv=set([i for i in rankbottom if i not in u][:48])
    assert len(cu)==len(cv)==len(bu)==len(bv)==48
    return old,atoms,u,cu,cv,bu,bv,score


def schedule_source_key(src,phase):
    label={"cal":202611360071,"eval":202611360073}[phase]
    return sha256(f"{label}|{phase}|{src}".encode("ascii")).hexdigest()


def transformed(original,atoms,scores,transform_src,arm,trial,kind):
    if kind=="original":return tuple(original)
    assert kind in ("atom_random","atom_curvature50")
    stream={"atom_random":202611350076,
            "atom_curvature50":202611350077}[kind]
    checked=set(original)
    result=[]
    for j,atom in enumerate(atoms):
        quota=len(checked&atom)
        if quota==0:continue
        candidates=sorted(atom)
        rng=random.Random(seed(stream,transform_src,arm,trial,j))
        if kind=="atom_random":
            result.extend(rng.sample(candidates,quota))
            continue
        fixed_count=(quota+1)//2
        ranked=sorted(candidates,key=lambda i:(-scores[i],i))
        prefix=ranked[:fixed_count]
        rest=[i for i in candidates if i not in set(prefix)]
        result.extend(prefix+rng.sample(rest,quota-fixed_count))
    assert len(result)==len(set(result))==16
    assert all(len(set(result)&a)==len(checked&a) for a in atoms)
    return tuple(result)


def policy_schedules(old,atoms,scores,src,phase):
    root=schedule_source_key(src,phase)
    transform_src=sha256(f"{root}|nor35-transform".encode("ascii")).hexdigest()
    number=CAL if phase=="cal" else EVAL
    plans={mode:{arm:[] for arm in ARMS}
           for mode in ("original","atom_random","atom_curvature50")}
    for t in range(number):
        oldplans=old25.policy_schedules(old,root,t)
        for arm in ARMS:
            base=tuple(oldplans[(arm,16)])
            assert len(set(base))==16
            for mode in plans:
                plans[mode][arm].append(
                    transformed(base,atoms,scores,transform_src,arm,t,mode))
    return plans


def components(name):
    if name=="original_h50":return (("original","hybrid_50",16),)
    if name=="atom_random_h50":return (("atom_random","hybrid_50",16),)
    if name=="atom_curv_h50":return (("atom_curvature50","hybrid_50",16),)
    if name=="uniform16":return (("original","uniform",16),)
    if name=="original_mix":mode="original"
    elif name=="atom_curv_mix":mode="atom_curvature50"
    else:raise ValueError(name)
    return tuple((mode,a,w) for a,w in zip(ARMS,WEIGHTS) if w)


def trial_ids(plans,m):
    out={}
    for mode,arms in plans.items():
        for arm,rows in arms.items():
            masks=[0]*m
            for t,checked in enumerate(rows):
                bit=1<<t
                for i in checked:masks[i]|=bit
            out[(mode,arm)]=masks
    return out


def choose_attacks(cal,src,m):
    seen=trial_ids(cal,m)
    proposed={}
    for name in POLICIES:
        parts=components(name)
        variants=[]
        for replica in range(ATTACKS):
            order=list(range(m))
            random.Random(seed(202611360072,src,name,replica)).shuffle(order)
            chosen=set()
            unions={p[:2]:0 for p in parts}
            for _ in range(R):
                # The seed-fixed shuffled ORDER encodes randomized ties.
                def objective(i):
                    return sum(w*(unions[(mode,a)]|seen[(mode,a)][i]).bit_count()
                               for mode,a,w in parts)
                candidate=min((i for i in order if i not in chosen),
                              key=objective)
                chosen.add(candidate)
                for mode,a,w in parts:
                    unions[(mode,a)]|=seen[(mode,a)][candidate]
            assert len(chosen)==R
            variants.append(tuple(sorted(chosen)))
        proposed[name]=variants
    return proposed


def source_blind_attacks(src,atoms,u,cu,cv,bu,bv,m):
    groups={a:[] for a in CONTROLS}
    for replica in range(ATTACKS):
        rng=random.Random(seed(202611360074,src,"top_curvature",replica))
        top=tuple(sorted(rng.sample(sorted(cu),2)+rng.sample(sorted(cv),2)))
        rng=random.Random(seed(202611360074,src,"bottom_curvature",replica))
        bottom=tuple(sorted(rng.sample(sorted(bu),2)+rng.sample(sorted(bv),2)))
        q=tuple(sum(i in c for i in top) for c in atoms)
        rng=random.Random(seed(202611360074,src,"eight_atom_matched",replica))
        matched=tuple(sorted(i for j,n in enumerate(q)
                             for i in rng.sample(sorted(atoms[j]),n)))
        rng=random.Random(seed(202611360074,src,"full_uniform",replica))
        randomset=tuple(sorted(rng.sample(range(m),4)))
        assert q[0]==2 and sum(q)==R
        assert tuple(sum(i in c for i in matched) for c in atoms)==q
        for family,positions in zip(CONTROLS,(top,bottom,matched,randomset)):
            assert len(set(positions))==R
            groups[family].append(positions)
    return groups


def mixed_draw(src,trial):
    r=random.Random(seed(202611360075,src,trial)).randrange(16)
    q=0
    for a,w in zip(ARMS,WEIGHTS):
        q+=w
        if r<q:return a
    raise AssertionError("bad policy")


def exact_top(supportU,supportV,checked):
    p=set(checked)
    t1=len(p&supportU)
    t2=len(p&supportV)
    return Fraction(1)-Fraction(comb(48-t1,2),comb(48,2))*Fraction(
        comb(48-t2,2),comb(48,2))


def pure_rate(policies,cu,cv):
    result={}
    for target in POLICIES:
        acc=Fraction(0)
        for mode,arm,w in components(target):
            val=sum((exact_top(cu,cv,plan)
                     for plan in policies[mode][arm]),Fraction(0))/EVAL
            acc+=Fraction(w,16)*val
        result[target]=acc
    return result


def replay(fixture_raw,grade_raw):
    fixture=json.loads(fixture_raw)
    grade=json.loads(grade_raw)
    previous=historical_roster()
    assert fixture["schema"]=="nor36.new_public_only_reps52to55.v1"
    assert grade["schema"]=="nor36.policy_aware_curvature_frozen_greedy_attack.v1"
    assert fixture["registration_first_commit"]==grade["registration_first_commit"]==FIRST
    assert fixture["public_only"] and fixture["planted_program_and_private_target_absent"]
    assert fixture["attacks_and_challenge_schedules_absent"]
    assert fixture["excluded_historical_source_count"]==len(previous)==132
    assert fixture["historical_source_roster_sha256"]==sha256(canonical(previous)).hexdigest()
    assert grade["public_only_fixture_sha256"]==sha256(fixture_raw).hexdigest()
    assert grade["fresh_source_count"]==len(fixture["cases"])==len(grade["source_records"])==24
    assert grade["excluded_historical_sources"]==132
    assert grade["frozen_mixture_weights"]==list(WEIGHTS)
    assert grade["attacker_and_auditor_policies"]==list(POLICIES)
    assert grade["blind_control_families"]==list(CONTROLS)
    assert grade["calibration_trials"]==CAL and grade["heldout_trials"]==EVAL
    assert grade["greedy_attack_restarts"]==ATTACKS
    assert grade["source_proposals_frozen_before_private_truth"]
    assert grade["attack_sets_committed_before_heldout_schedules"]
    assert grade["external_authenticated_exact_target_witness_assumed"]
    ids=[row["full_public_input_sha256"] for row in fixture["cases"]]
    assert len(set(ids))==24 and set(ids).isdisjoint(previous)
    rows=[]
    decisions=0
    for row,record in zip(fixture["cases"],grade["source_records"],strict=True):
        src=row["full_public_input_sha256"]
        mask,j,pred,header=source(row)
        assert record["public_sha256"]==src
        assert record["replicate"]==row["replicate"]
        assert record["degree"]==row["degree"] and record["family"]==row["family"]
        assert record["future_length"]==len(pred)
        assert record["mask"]==mask and record["chosen_class"]==j
        assert record["public_prediction_sha256"]==record["private_true_future_sha256_postchoice"]==future_sha(pred)
        old,atoms,u,cu,cv,bu,bv,scores=public_atom_geometry(pred,header)
        assert record["historical_atom_sizes"]==list(map(len,atoms))
        assert record["top_curvature_support_sha256"]==sha256(
            canonical(sorted(cu|cv))).hexdigest()
        assert record["bottom_curvature_support_sha256"]==sha256(
            canonical(sorted(bu|bv))).hexdigest()
        cal=policy_schedules(old,atoms,scores,src,"cal")
        assert record["calibration_plan_sha256"]==sha256(canonical(cal)).hexdigest()
        proposed=choose_attacks(cal,src,len(pred))
        assert record["greedy_attacks"]=={
            name:[list(a) for a in sets] for name,sets in proposed.items()}
        assert record["unique_greedy_attacks"]=={
            name:len(set(sets)) for name,sets in proposed.items()}
        blind=source_blind_attacks(src,atoms,u,cu,cv,bu,bv,len(pred))
        assert record["blind_attacks"]=={
            name:[list(a) for a in sets] for name,sets in blind.items()}
        all_attacks={**proposed,**blind}
        plans=policy_schedules(old,atoms,scores,src,"eval")
        assert record["heldout_plan_sha256"]==sha256(canonical(plans)).hexdigest()
        mixed=[mixed_draw(src,t) for t in range(EVAL)]
        assert record["realized_mix_arm_draw_counts"]==dict(Counter(mixed))
        cross={}
        for name,attacks in all_attacks.items():
            cross[name]={}
            for policy in POLICIES:
                score=0
                for mode,arm,w in components(policy):
                    for changed in attacks:
                        aset=set(changed)
                        for checked in plans[mode][arm]:
                            score+=w*int(bool(aset&set(checked)))
                            decisions+=1
                cross[name][policy]=score
        assert record["crossplay_weighted_numerators"]==cross
        assert record["crossplay_denominator"]==16*ATTACKS*EVAL
        top=pure_rate(plans,cu,cv)
        assert record["top_support_exact_fraction"]=={
            name:[x.numerator,x.denominator] for name,x in top.items()}
        exact_uniform=Fraction(1)-Fraction(comb(len(pred)-16,4),comb(len(pred),4))
        assert record["uniform_blind_exact_fraction"]==[
            exact_uniform.numerator,exact_uniform.denominator]
        assert record["honest_false_rejects"]==0
        assert record["schedule_informed_detection"]==0
        assert record["atom_quota_mismatches"]==0
        assert record["nonzero_shift_disagreements"]==0
        # Earlier code checks exact original program parity, and all
        # checked points are exact outside the committed attack sets.
        selfcurv=cross["atom_curv_h50"]["atom_curv_h50"]/(16*ATTACKS*EVAL)
        selforig=cross["original_h50"]["original_h50"]/(16*ATTACKS*EVAL)
        samecurv=cross["atom_curv_h50"]["atom_curv_h50"]/(16*ATTACKS*EVAL)
        sameorig=cross["atom_curv_h50"]["original_h50"]/(16*ATTACKS*EVAL)
        selfmixcurv=cross["atom_curv_mix"]["atom_curv_mix"]/(16*ATTACKS*EVAL)
        selfmixorig=cross["original_mix"]["original_mix"]/(16*ATTACKS*EVAL)
        rows.append({
            "public_sha256":src,"degree":row["degree"],"family":row["family"],
            "self_curv":selfcurv,"self_original":selforig,
            "self_curv_minus_original":selfcurv-selforig,
            "fixed_curv_attack_curv_minus_original":samecurv-sameorig,
            "top_curv_h50_gain":float(top["atom_curv_h50"]-top["original_h50"]),
            "mix_self_curv_minus_original":selfmixcurv-selfmixorig,
            "top_curv_mix_gain":float(top["atom_curv_mix"]-top["original_mix"]),
            "uniform_self":cross["uniform16"]["uniform16"]/(16*ATTACKS*EVAL),
        })
    for gotten,expected in zip(grade["primary_by_source"],rows,strict=True):
        assert gotten.keys()==expected.keys()
        for k,val in expected.items():
            if isinstance(val,float):
                assert isclose(gotten[k],val,rel_tol=0,abs_tol=1e-14),(k,gotten[k],val)
            else:assert gotten[k]==val
    mfields=("self_curv_minus_original",
             "fixed_curv_attack_curv_minus_original","top_curv_h50_gain",
             "mix_self_curv_minus_original","top_curv_mix_gain")
    means={f:statistics.mean(r[f] for r in rows) for f in mfields}
    wins={
        "self_curv_weaker":sum(r["self_curv_minus_original"]<0 for r in rows),
        "fixed_curv_attacks_expose_weakness":sum(
            r["fixed_curv_attack_curv_minus_original"]<0 for r in rows),
        "top_h50_gain":sum(r["top_curv_h50_gain"]>0 for r in rows),
        "mix_self_curv_weaker":sum(r["mix_self_curv_minus_original"]<0 for r in rows),
        "top_mix_gain":sum(r["top_curv_mix_gain"]>0 for r in rows)
    }
    gates={
        "curvature_selfplay_less_robust":wins["self_curv_weaker"]>=18 and
            means["self_curv_minus_original"]<=-.005,
        "same_curvature_attack_exposes_weakness":
            wins["fixed_curv_attacks_expose_weakness"]>=18 and
            means["fixed_curv_attack_curv_minus_original"]<=-.005,
        "top_curvature_source_aligned_gain":
            wins["top_h50_gain"]>=18 and means["top_curv_h50_gain"]>=.10,
        "mixture_selfplay_erosion_and_top_gain":
            wins["mix_self_curv_weaker"]>=18 and
            means["mix_self_curv_minus_original"]<=-.005 and
            wins["top_mix_gain"]>=18 and means["top_curv_mix_gain"]>=.10,
        "source_attacks_audits_truth_and_replay":True,
    }
    assert grade["source_direction_wins"]==wins
    for k,val in means.items():
        assert isclose(grade["source_equal_mean_primary"][k],val,abs_tol=1e-14)
    assert grade["registered_five_gates"]==gates
    assert grade["registered_primary_pass"]==all(gates.values())
    return {
        "new_public_source_units":24,"excluded_historical_source_units":132,
        "replayed_greedy_attacks":24*6*ATTACKS,
        "replayed_blind_attack_sets":24*4*ATTACKS,
        "independent_crossplay_policy_world_decisions":decisions,
        "source_means":means,"source_wins":wins,
        "registered_primary_pass":all(gates.values()),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--full",type=Path,required=True)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    result=replay(args.fixture.read_bytes(),args.full.read_bytes())
    print("NOR36 INDEPENDENT SOURCE GREEDY ATTACK CROSSPLAY REPLAY",result)
    if args.check:
        assert result["new_public_source_units"]==24
        assert result["replayed_greedy_attacks"]==24*6*8
        assert result["replayed_blind_attack_sets"]==24*4*8
        # 4 pure auditor classes + 2 mixes with five components each.
        assert result["independent_crossplay_policy_world_decisions"]==(
            24*10*8*512*(4+5+5))


if __name__=="__main__":
    main()
