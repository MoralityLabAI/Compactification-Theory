"""NOR-35: prospective source-heldout within-eight-atom curvature auditor.

This application retains each NOR25 original policy's EXACT eight-atom
quota t_j at each source/trial, then resamples within each public atom
at equal total k=16 witness queries. Primary scientific outcomes use
exact Fraction conditional attack laws, not 16 sampled attack hits.
"""
from __future__ import annotations

import argparse, json,random,statistics,sys
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
from research.nor33_overlap_matched_transfer import experiment as nor33

FIXTURE_SCHEMA="nor35.public_only_new_reps48to51.v1"
RESULT_SCHEMA="nor35.eight_atom_curvature_targeted_vs_quota_random.v1"
REGISTRATION="3773c1b1522f419e54f34252b0c743018105bc79"
NOR34_SOURCE_SHA="1cbf58a851df898a6e46fe4859d25ac5cf5b68a2e2b3e81d8e7afeede8dacb00"
NOR34_RESULT_SHA="4507769666f5560d33b8a0c99b4b5c85c5c2710b88e0021e77a85679106f6bd3"
REPS=(48,49,50,51)
DEGREES=(512,1024)
FAMILIES=nor21.FAMILIES
OFFSET=130035000
ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
TRANSFORMS=("original","atom_random","atom_curvature50","atom_inverse50")
ATTACKS=("top_curvature","bottom_curvature","eight_atom_matched","full_uniform")
ATTACK_STREAMS={"top_curvature":202611350071,
                "bottom_curvature":202611350072,
                "eight_atom_matched":202611350073,
                "full_uniform":202611350074}
TRANSFORM_STREAMS={"atom_random":202611350076,
                   "atom_curvature50":202611350077,
                   "atom_inverse50":202611350078}
TRIALS=512
ATTACK_REPS=16
K=16
R=4
AUDIT_STREAM=202611350075
MIX_STREAM=202611350079
DELTAS=(1,4294049777)


def canonical(x):
    return (json.dumps(x,indent=2,sort_keys=True,ensure_ascii=True)+"\n").encode()


def hash_future(y):
    return sha256(json.dumps(y,separators=(",",":")).encode()).hexdigest()


def seed(*parts):
    return int.from_bytes(sha256("|".join(map(str,parts)).encode("ascii")).digest(),"big")


def source_seed(rep,degree,family):
    assert rep in REPS and degree in DEGREES and family in FAMILIES
    return (corpus.SEED+OFFSET+1000003*rep+
            100009*DEGREES.index(degree)+10007*FAMILIES.index(family))


def historical_ids():
    original=nor34.historic_source_hashes()
    archive=ROOT/"research/nor34_eight_atom_curvature/results/nor34_original_hosted.zip"
    with ZipFile(archive) as z:
        assert z.testzip() is None
        fixture=z.read("nor34_fixture.json")
        full=z.read("nor34_full.json")
    assert len(fixture)==5011440
    assert len(full)==337379
    assert sha256(fixture).hexdigest()==NOR34_SOURCE_SHA
    assert sha256(full).hexdigest()==NOR34_RESULT_SHA
    past=json.loads(fixture)
    assert past["schema"]=="nor34.new_public_only_reps44to47.v1"
    ids=sorted(original+[r["full_public_input_sha256"] for r in past["cases"]])
    assert len(ids)==len(set(ids))==108
    return ids


def source_public(rep,degree,family,seen):
    q=3*degree//2
    rng=random.Random(source_seed(rep,degree,family))
    for attempt in range(100):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,family,q),
                     corpus.execute(x,planted)) for _ in range(2))
        query=nor21.public_source(rng,family,q)
        case=corpus.Case("nor35-source",family,demos,query,(),None)
        headers=corpus.possible_headers(case)
        if len(headers)!=1:continue
        h=headers[0]
        if (h.base,h.delay,h.spacing,h.amplitude)!=planted[:4]:continue
        source_hash=nor21.public_digest(case,degree)
        if source_hash in seen:continue
        mask=nor21.native_mask(case,h,degree)
        if not mask:continue
        candidates=nor21.candidate_predictions(query,h,q)
        if len({candidates[j] for j in range(16) if mask&(1<<j)})!=1:continue
        true_class=4*corpus.OPS.index(planted[4])+corpus.OPS.index(planted[5])
        assert mask&(1<<true_class)
        seen.add(source_hash)
        return {
            "split":"TEST","replicate":rep,"degree":degree,
            "query_degree":q,"family":family,
            "source_seed":source_seed(rep,degree,family),
            "selected_attempt":attempt,"full_public_input_sha256":source_hash,
            "public_header":{"base":h.base,"delay":h.delay,
                             "spacing":h.spacing,"amplitude":h.amplitude},
            "demos":[[list(x[:degree+1]),list(y[:degree+1])] for x,y in demos],
            "public_query_input":list(query),"original_native_public_mask":mask}
    raise RuntimeError("NOR35 public source exhausted registered 100 attempts")


def make_fixture():
    past=historical_ids()
    seen=set(past)
    rows=[source_public(rep,d,family,seen)
          for rep in REPS for d in DEGREES for family in FAMILIES]
    assert len(rows)==24 and len(seen)==132
    return {"schema":FIXTURE_SCHEMA,"registration_commit":REGISTRATION,
            "replicates":list(REPS),"degrees":list(DEGREES),
            "families":list(FAMILIES),"offset":OFFSET,
            "public_only":True,"private_program_and_future_absent":True,
            "attack_witness_and_policy_absent":True,
            "excluded_history_count":108,
            "historical_sha256":sha256(canonical(past)).hexdigest(),
            "cases":rows}


def read_fixture(path):
    raw=Path(path).read_bytes()
    f=json.loads(raw)
    past=historical_ids()
    assert f["schema"]==FIXTURE_SCHEMA and f["registration_commit"]==REGISTRATION
    assert f["replicates"]==list(REPS) and f["degrees"]==list(DEGREES)
    assert f["offset"]==OFFSET and f["public_only"]
    assert f["private_program_and_future_absent"] and f["attack_witness_and_policy_absent"]
    assert f["excluded_history_count"]==len(past)==108
    assert f["historical_sha256"]==sha256(canonical(past)).hexdigest()
    rows=f["cases"]
    assert len(rows)==24
    keys=[r["full_public_input_sha256"] for r in rows]
    assert len(set(keys))==24 and set(keys).isdisjoint(past)
    assert {(r["replicate"],r["degree"],r["family"]) for r in rows}=={
        (rep,d,f) for rep in REPS for d in DEGREES for f in FAMILIES}
    assert all(r["split"]=="TEST" and not any(
        k in r for k in ("program","future","target","attack","witness","policies"))
        for r in rows)
    return f,sha256(raw).hexdigest()


def proposal(row):
    d=row["degree"]
    hr=row["public_header"]
    h=corpus.Header(hr["base"],hr["delay"],hr["spacing"],hr["amplitude"])
    case=corpus.Case("nor35-public",row["family"],
        tuple((tuple(x),tuple(y)) for x,y in row["demos"]),
        tuple(row["public_query_input"]),(),None)
    assert case.program is None and case.future==()
    assert nor21.public_digest(case,d)==row["full_public_input_sha256"]
    result=nor21.certify(case,h,d,row["query_degree"],"stream16")
    assert result["mask"]==row["original_native_public_mask"]>0
    assert result["unanimous"] and result["chosen_class"] is not None
    pred=corpus.execute(case.query,nor21.program(h,result["chosen_class"]))
    return {"pred":pred,"header":h,"mask":result["mask"],
            "class":result["chosen_class"],"sha":nor24.sha_target(pred)}


def sealed_native_truth(row,predicted):
    rep,d,fam=row["replicate"],row["degree"],row["family"]
    rng=random.Random(source_seed(rep,d,fam))
    for _ in range(row["selected_attempt"]+1):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,fam,3*d//2),
                     corpus.execute(x,planted)) for _ in range(2))
        query=nor21.public_source(rng,fam,3*d//2)
    assert row["source_seed"]==source_seed(rep,d,fam)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    truth=corpus.execute(query,planted)
    assert truth==predicted
    return truth


def curvature_score(pred):
    m=len(pred)
    return tuple(abs(pred[(i-1)%m]-2*pred[i]+pred[(i+1)%m])
                 for i in range(m))


def source_atoms(pred,head,digest):
    old,cats,u,cu,cv=nor34.partition(pred,head,digest)
    # NOR34 top curvature support is already fixed at 48 U + 48 V.
    rank=nor33.full_rankings(pred,head,digest,old)["curvature"]
    bu=frozenset([i for i in reversed(rank) if i in u][:48])
    bv=frozenset([i for i in reversed(rank) if i not in u][:48])
    assert len(bu)==len(bv)==48
    assert len(cu)==len(cv)==48
    m=len(pred)
    assert len(u)>=96 and m-len(u)>=96
    return old,cats,u,cu,cv,bu,bv,curvature_score(pred)


def attacks_for_source(digest,cats,u,cu,cv,bu,bv,m):
    attacks={a:[] for a in ATTACKS}
    quotas=[]
    for t in range(ATTACK_REPS):
        rand=random.Random(seed(ATTACK_STREAMS["top_curvature"],digest,t))
        top=tuple(sorted(rand.sample(sorted(cu),2)+rand.sample(sorted(cv),2)))
        rand=random.Random(seed(ATTACK_STREAMS["bottom_curvature"],digest,t))
        bottom=tuple(sorted(rand.sample(sorted(bu),2)+rand.sample(sorted(bv),2)))
        quota=tuple(sum(i in atom for i in top) for atom in cats)
        assert quota[0]==2 and sum(quota)==4
        rand=random.Random(seed(ATTACK_STREAMS["eight_atom_matched"],digest,t))
        fine=tuple(sorted(i for j,q in enumerate(quota)
                          for i in rand.sample(sorted(cats[j]),q)))
        rand=random.Random(seed(ATTACK_STREAMS["full_uniform"],digest,t))
        uniform=tuple(sorted(rand.sample(range(m),4)))
        assert len(set(top))==len(set(bottom))==len(set(fine))==len(set(uniform))==4
        assert all(sum(i in a for i in fine)==q for a,q in zip(cats,quota))
        assert sum(i in u for i in bottom)==2
        attacks["top_curvature"].append(top)
        attacks["bottom_curvature"].append(bottom)
        attacks["eight_atom_matched"].append(fine)
        attacks["full_uniform"].append(uniform)
        quotas.append(quota)
    return attacks,quotas


def transformed_plan(original,cats,scores,digest,arm,trial,mode):
    if mode=="original":
        return tuple(original)
    assert mode in TRANSFORM_STREAMS
    inspected=set(original)
    out=[]
    for j,atom in enumerate(cats):
        quota=len(inspected&atom)
        if quota==0:continue
        candidates=sorted(atom)
        rand=random.Random(seed(TRANSFORM_STREAMS[mode],digest,arm,trial,j))
        if mode=="atom_random":
            out.extend(rand.sample(candidates,quota))
            continue
        keep=(quota+1)//2
        ranks=sorted(candidates,key=(
            (lambda i:(-scores[i],i)) if mode=="atom_curvature50"
            else (lambda i:(scores[i],i))))
        top=ranks[:keep]
        remain=[i for i in candidates if i not in set(top)]
        out.extend(top)
        out.extend(rand.sample(remain,quota-keep))
    assert len(out)==len(set(out))==K
    assert all(len(set(out)&atom)==len(inspected&atom) for atom in cats)
    return tuple(out)


def policy_plans(old,cats,scores,digest):
    phase=sha256(f"{AUDIT_STREAM}|nor35|{digest}".encode("ascii")).hexdigest()
    out={mode:{arm:[] for arm in ARMS} for mode in TRANSFORMS}
    for trial in range(TRIALS):
        original=nor25.policy_schedules(old,phase,trial)
        for arm in ARMS:
            plan=tuple(original[(arm,K)])
            assert len(set(plan))==K
            for mode in TRANSFORMS:
                out[mode][arm].append(transformed_plan(
                    plan,cats,scores,digest,arm,trial,mode))
    return out


def choose_realized_arm(digest,trial):
    position=random.Random(seed(MIX_STREAM,digest,trial)).randrange(16)
    total=0
    for arm,w in zip(ARMS,WEIGHTS):
        total+=w
        if position<total:return arm
    raise AssertionError("invalid mix")


def ideal_uniform_attack_detect(m):
    return Fraction(1)-Fraction(comb(m-16,4),comb(m,4))


def two_block_detect(supportU,supportV,checked):
    tc=len(set(checked)&supportU)
    tv=len(set(checked)&supportV)
    return Fraction(1)-Fraction(comb(48-tc,2),comb(48,2))*Fraction(
        comb(48-tv,2),comb(48,2))


def exact_analytic(prob_function,cats,cu,cv,bu,bv,checked,m):
    atom_counts=tuple(len(set(checked)&atom) for atom in cats)
    cu_count=len(set(checked)&cu)
    cv_count=len(set(checked)&cv)
    fine=prob_function(atom_counts,cu_count,cv_count)["eight_atom"]
    return {
        "top_curvature":two_block_detect(cu,cv,checked),
        "bottom_curvature":two_block_detect(bu,bv,checked),
        "eight_atom_matched":fine,
        "full_uniform":ideal_uniform_attack_detect(m),
    }


def mix_score(arm_rates):
    return sum((Fraction(w,16)*arm_rates[a] for a,w in zip(ARMS,WEIGHTS)),
               Fraction(0))


def evaluate(fixture_path):
    fixture,fixture_hash=read_fixture(fixture_path)
    # Freeze all 24 proposals from public demonstrations alone.
    proposals={r["full_public_input_sha256"]:proposal(r) for r in fixture["cases"]}
    assert len(proposals)==24
    records=[]
    for row in fixture["cases"]:
        key=row["full_public_input_sha256"]
        pub=proposals[key]
        predicted=pub["pred"]
        old,cats,u,cu,cv,bu,bv,scores=source_atoms(predicted,pub["header"],key)
        attacks,quotas=attacks_for_source(key,cats,u,cu,cv,bu,bv,len(predicted))
        plans=policy_plans(old,cats,scores,key)
        chosen_arms=[choose_realized_arm(key,t) for t in range(TRIALS)]
        truth=sealed_native_truth(row,predicted)
        assert nor24.sha_target(truth)==pub["sha"]
        fineprob=nor34.exact_probabilities(cats,cu,cv)
        counts={mode:{arm:{a:0 for a in ATTACKS} for arm in ARMS} for mode in TRANSFORMS}
        sums={mode:{arm:{a:Fraction(0) for a in ATTACKS} for arm in ARMS}
              for mode in TRANSFORMS}
        realized={mode:{a:0 for a in ATTACKS} for mode in TRANSFORMS}
        policy_digests={mode:{arm:sha256(canonical(plans[mode][arm])).hexdigest()
                              for arm in ARMS} for mode in TRANSFORMS}
        legit=adaptive=direct=shift_mismatch=quota_mismatch=0
        for mode in TRANSFORMS:
            for arm in ARMS:
                for trial,inspected in enumerate(plans[mode][arm]):
                    if not nor24.exact_witness_accepts(predicted,truth,inspected,
                                                      frozenset(),1):
                        legit+=1
                    adversarial=nor24.informed_attack(inspected,old["uniform"],4)
                    adaptive+=bool(set(inspected)&adversarial)
                    # Exact same old eight-atom inspection quotas as the
                    # ORIGINAL policy schedule on this source/trial.
                    original_inspected=set(plans["original"][arm][trial])
                    quota_mismatch+=sum(
                        len(set(inspected)&atom)!=len(original_inspected&atom)
                        for atom in cats)
                    fractions=exact_analytic(fineprob,cats,cu,cv,bu,bv,
                                             inspected,len(predicted))
                    for family,p in fractions.items():
                        sums[mode][arm][family]+=p
                    checkmask=sum(1<<i for i in inspected)
                    for family,variations in attacks.items():
                        for attack_idx,attack in enumerate(variations):
                            found=bool(checkmask & sum(1<<i for i in attack))
                            counts[mode][arm][family]+=found
                            if chosen_arms[trial]==arm:
                                realized[mode][family]+=found
                            if trial==0 and attack_idx==0:
                                for delta in DELTAS:
                                    witnessed=not nor24.exact_witness_accepts(
                                        predicted,truth,inspected,frozenset(attack),delta)
                                    assert witnessed==found
                                direct+=1
        assert legit==adaptive==quota_mismatch==shift_mismatch==0
        assert direct==len(TRANSFORMS)*len(ARMS)*len(ATTACKS)
        analytic={mode:{arm:{f:sumv/TRIALS for f,sumv in fs.items()}
                        for arm,fs in arms.items()} for mode,arms in sums.items()}
        # Equality required by same atom counts for quota-standardized
        # attack distribution under ANY original source/trial plan.
        assert all(analytic[mode][arm]["eight_atom_matched"]==
                   analytic["original"][arm]["eight_atom_matched"]
                   for mode in TRANSFORMS for arm in ARMS)
        assert all(analytic[mode][arm]["full_uniform"]==ideal_uniform_attack_detect(
                   len(predicted)) for mode in TRANSFORMS for arm in ARMS)
        mix={mode:{family:mix_score({
             arm:analytic[mode][arm][family] for arm in ARMS})
             for family in ATTACKS} for mode in TRANSFORMS}
        source_metrics={
            "C":float(analytic["atom_curvature50"]["hybrid_50"]["top_curvature"]-
                      analytic["atom_random"]["hybrid_50"]["top_curvature"]),
            "B":float(analytic["atom_curvature50"]["hybrid_50"]["top_curvature"]-
                      analytic["original"]["hybrid_50"]["top_curvature"]),
            "I":float(analytic["atom_curvature50"]["hybrid_50"]["bottom_curvature"]-
                      analytic["atom_random"]["hybrid_50"]["bottom_curvature"]),
            "M":float(mix["atom_curvature50"]["top_curvature"]-
                      mix["original"]["top_curvature"]),
        }
        records.append({
            "public_sha256":key,"replicate":row["replicate"],"degree":row["degree"],
            "family":row["family"],"future_length":len(predicted),
            "native_mask":pub["mask"],"chosen_class":pub["class"],
            "public_prediction_sha256":pub["sha"],
            "postchoice_true_future_sha256":nor24.sha_target(truth),
            "old_atom_sizes":[len(c) for c in cats],
            "curvature_top_atom_support_counts":[len((cu|cv)&c) for c in cats],
            "curvature_bottom_atom_support_counts":[len((bu|bv)&c) for c in cats],
            "curvature_top_support_sha256":sha256(canonical(sorted(cu|cv))).hexdigest(),
            "curvature_bottom_support_sha256":sha256(canonical(sorted(bu|bv))).hexdigest(),
            "attack_sets":{f:[list(a) for a in variants] for f,variants in attacks.items()},
            "top_curvature_attack_quotas":[list(q) for q in quotas],
            "per_transform_arm_plan_sha256":policy_digests,
            "realized_mixed_arm_counts":dict(Counter(chosen_arms)),
            "exact_analytic_detection":{
                mode:{arm:{f:[v.numerator,v.denominator] for f,v in fam.items()}
                      for arm,fam in arms.items()} for mode,arms in analytic.items()},
            "sample_detected_out_of_8192":counts,
            "realized_mixture_detected_out_of_8192":realized,
            "exact_mixture_detection":{
                mode:{f:[v.numerator,v.denominator] for f,v in fam.items()}
                for mode,fam in mix.items()},
            "source_primary_C_B_I_M":source_metrics,
            "legitimate_witness_false_rejections":legit,
            "schedule_informed_attacker_detected":adaptive,
            "eight_atom_quota_mismatches":quota_mismatch,
            "shift_decision_disagreements":shift_mismatch,
            "direct_authentic_integer_witness_pair_checks":direct,
        })
    paired=[r["source_primary_C_B_I_M"] for r in records]
    means={s:statistics.mean(r[s] for r in paired) for s in ("C","B","I","M")}
    wins={"C":sum(r["C"]>0 for r in paired),
          "B":sum(r["B"]>0 for r in paired),
          "I":sum(r["I"]<0 for r in paired),
          "M":sum(r["M"]>0 for r in paired)}
    allvalid=all(r["native_mask"]>0 and
        r["public_prediction_sha256"]==r["postchoice_true_future_sha256"]
        and r["eight_atom_quota_mismatches"]==r["schedule_informed_attacker_detected"]==
        r["legitimate_witness_false_rejections"]==r["shift_decision_disagreements"]==0
        and r["direct_authentic_integer_witness_pair_checks"]==96
        for r in records)
    gates={
        "curvature_beats_atom_random":wins["C"]>=18 and means["C"]>=.010,
        "curvature_beats_original_hybrid50":wins["B"]>=18 and means["B"]>=.005,
        "inverse_curvature_cost":wins["I"]>=18 and means["I"]<=-.002,
        "transformed_mix_and_atom_invariance":
            wins["M"]>=18 and means["M"]>=.005 and allvalid,
        "source_oracle_and_independent_replay":allvalid,
    }
    return {
        "schema":RESULT_SCHEMA,"preregistered_first_commit":REGISTRATION,
        "source_public_fixture_sha256":fixture_hash,
        "source_units":len(records),"excluded_historical_units":108,
        "auditor_arms":list(ARMS),"frozen_mix_weights":list(WEIGHTS),
        "transforms":list(TRANSFORMS),"attack_laws":list(ATTACKS),
        "attacks_per_source_family":ATTACK_REPS,"audit_trials_per_source_arm":TRIALS,
        "budget":K,"radius":R,"nonzero_shifts":list(DELTAS),
        "public_proposals_frozen_before_private_truth":True,
        "all_attacks_and_plans_fixed_before_native_truth":True,
        "authentic_external_target_witness_assumed":True,
        "source_records":records,
        "source_mean_primary":means,"source_wins":wins,
        "registered_five_gates":gates,
        "registered_primary_pass":all(gates.values()),
        "scope":("Conditional public eight-atom quota-matched risk targeting. "
                 "Original RRC25 finite program family, honest exogenous "
                 "target coefficients and independent genuinely hidden audit "
                 "coins assumed; deterministic public SHA streams are replay receipts."),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--make-fixture",type=Path)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    if args.make_fixture:
        if args.fixture or args.out:p.error("source creation must precede test scoring")
        raw=canonical(make_fixture())
        args.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        args.make_fixture.write_bytes(raw)
        print("NOR35 ORIGINAL FROZEN PUBLIC SOURCE",len(raw),sha256(raw).hexdigest())
        return
    if not args.fixture:p.error("--fixture required")
    outcome=evaluate(args.fixture)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(canonical(outcome))
    print("NOR35 REGISTERED SCIENTIFIC PASS",outcome["registered_primary_pass"])
    print("NOR35 FIVE GATES",outcome["registered_five_gates"])
    print("NOR35 SOURCE WINS",outcome["source_wins"])
    print("NOR35 SOURCE MEANS",outcome["source_mean_primary"])


if __name__=="__main__":
    main()
