"""NOR-32: prospective attack-generation OOD transfer of exact 16-coordinate audits.

The registered experiment retains frozen NOR-31 [2,0,1,1,4,8]/16
defender weights, original NOR25 schedule semantics and original RRC25
source programs. All 24 TEST source/public proposals are frozen before
original private generators are reconstructed. Synthetic witness values
are extra information from an assumed authentic target oracle.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from math import comb
from pathlib import Path
from zipfile import ZipFile
import json, random, statistics, sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import experiment as nor21
from research.nor24_adaptive_audits import experiment as n24
from research.nor25_robust_audits import experiment as n25

SCHEMA="nor32.unseen_six_support_equal_budget_transfer.v1"
FIXTURE_SCHEMA="nor32.fresh_public_only_reps32to35.v1"
OFFSET=130032000
REPS=(32,33,34,35)
DEGREES=(512,1024)
FAMILIES=nor21.FAMILIES
ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
OOD=("central96","stride17_96","low_abs96","curvature96","anti_prior96","hash96")
OLD=("tail","magnitude","lag","uniform")
OLD_WEIGHTS=(.4,.3,.2,.1)
CAL=256
EVAL=512
K=16
R=4
BLIND_REPS=8
GREEDY_REPS=4
STREAM_CAL=202611320074
STREAM_GREEDY=202611320075
STREAM_BLIND=202611320076
STREAM_EVAL=202611320077
STREAM_OLD=202611320078
STREAM_MIX=202611320079
SHIFTS=(1,4294049777)
N31_TRAIN_SHA="ba6141e866324000e6fdb22068a9b56631e0eb5fddb8ec36daca16e5ce199dc0"
N31_PUBLIC_SHA="b34f0cdefc7c4d06770af00ac659f11fd83b540fa40b02186157abc640bfae85"
N31_FULL_SHA="f978a2c388715056e7999e01d4ddc30f6c24a778b0e6e3e741a070902223430c"


def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()


def seed(*parts):
    return int.from_bytes(sha256("|".join(map(str,parts)).encode("ascii")).digest(),"big")


def load_historical():
    path=ROOT/"research/nor31_class_constrained_minimax/results/nor31_original_hosted.zip"
    with ZipFile(path) as z:
        assert sorted(z.namelist())==["nor31_fixture.json","nor31_full.json","nor31_train_receipt.json"]
        assert z.testzip() is None
        raw={k:z.read(k) for k in z.namelist()}
    assert len(raw["nor31_train_receipt.json"])==5300
    assert len(raw["nor31_fixture.json"])==2245544
    assert len(raw["nor31_full.json"])==317452
    assert sha256(raw["nor31_train_receipt.json"]).hexdigest()==N31_TRAIN_SHA
    assert sha256(raw["nor31_fixture.json"]).hexdigest()==N31_PUBLIC_SHA
    assert sha256(raw["nor31_full.json"]).hexdigest()==N31_FULL_SHA
    tr=json.loads(raw["nor31_train_receipt.json"])
    fr=json.loads(raw["nor31_fixture.json"])
    grade=json.loads(raw["nor31_full.json"])
    assert tr["fitted_constrained"]["weights"]==list(WEIGHTS)
    assert grade["frozen_sixclass_quanta_weights"]==list(WEIGHTS)
    assert grade["registered_primary_pass"] is True
    oldhashes=set(tr["historical_nor26_public_hashes"]+
                  tr["historical_nor30_public_hashes"]+
                  [r["full_public_input_sha256"] for r in fr["cases"]])
    assert len(oldhashes)==36
    return sorted(oldhashes)


def source_seed(rep,d,family):
    if rep not in REPS or d not in DEGREES or family not in FAMILIES:
        raise ValueError("unregistered NOR32 source")
    return (corpus.SEED+OFFSET+1000003*rep+
            100009*DEGREES.index(d)+10007*FAMILIES.index(family))


def new_source(rep,d,family,seen):
    q=3*d//2
    rng=random.Random(source_seed(rep,d,family))
    for attempt in range(100):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,family,q),
                     corpus.execute(x,planted)) for _ in range(2))
        query=nor21.public_source(rng,family,q)
        public=corpus.Case("nor32-public-only",family,demos,query,(),None)
        headers=corpus.possible_headers(public)
        if len(headers)!=1:continue
        h=headers[0]
        if (h.base,h.delay,h.spacing,h.amplitude)!=planted[:4]:continue
        digest=nor21.public_digest(public,d)
        if digest in seen:continue
        mask=nor21.native_mask(public,h,d)
        if mask==0:continue
        candidate=nor21.candidate_predictions(query,h,q)
        if len({candidate[j] for j in range(16) if mask&(1<<j)})!=1:continue
        planted_id=4*corpus.OPS.index(planted[4])+corpus.OPS.index(planted[5])
        assert mask&(1<<planted_id)
        seen.add(digest)
        return {
            "split":"TEST","replicate":rep,"degree":d,"query_degree":q,
            "family":family,"source_seed":source_seed(rep,d,family),
            "selected_attempt":attempt,"full_public_input_sha256":digest,
            "public_header":{"base":h.base,"delay":h.delay,
                             "spacing":h.spacing,"amplitude":h.amplitude},
            "demos":[[list(x[:d+1]),list(y[:d+1])] for x,y in demos],
            "public_query_input":list(query),
            "original_exact_public_accept_mask":mask,
        }
    raise RuntimeError("NOR32 admissible public source missing within 100 attempts")


def make_fixture():
    previous=load_historical()
    seen=set(previous)
    cases=[new_source(rep,d,family,seen)
           for rep in REPS for d in DEGREES for family in FAMILIES]
    assert len(cases)==24 and len(seen)==60
    return {"schema":FIXTURE_SCHEMA,"registered_offset":OFFSET,
            "replicates":list(REPS),"observation_degrees":list(DEGREES),
            "source_families":list(FAMILIES),"public_only":True,
            "private_truth_and_program_absent":True,
            "attacked_positions_and_witness_absent":True,
            "historical_public_hash_count":len(previous),
            "historical_source_sha256":sha256(canonical(previous)).hexdigest(),
            "cases":cases}


def read_fixture(path):
    raw=Path(path).read_bytes()
    o=json.loads(raw)
    assert o["schema"]==FIXTURE_SCHEMA and o["registered_offset"]==OFFSET
    assert o["replicates"]==list(REPS)
    assert o["public_only"] and o["private_truth_and_program_absent"]
    assert o["attacked_positions_and_witness_absent"]
    assert len(o["cases"])==24
    previous=load_historical()
    assert o["historical_source_sha256"]==sha256(canonical(previous)).hexdigest()
    ids=[r["full_public_input_sha256"] for r in o["cases"]]
    assert len(set(ids))==24 and set(ids).isdisjoint(previous)
    assert {(r["replicate"],r["degree"],r["family"]) for r in o["cases"]}=={
        (rep,d,f) for rep in REPS for d in DEGREES for f in FAMILIES}
    assert all(r["split"]=="TEST" and
               not any(x in r for x in ("program","future","target","attack","witness"))
               for r in o["cases"])
    return o,sha256(raw).hexdigest()


def public_proposal(row):
    d=row["degree"]
    hrow=row["public_header"]
    h=corpus.Header(hrow["base"],hrow["delay"],hrow["spacing"],hrow["amplitude"])
    c=corpus.Case("nor32-visible",row["family"],
                  tuple((tuple(x),tuple(y)) for x,y in row["demos"]),
                  tuple(row["public_query_input"]),(),None)
    assert c.program is None and c.future==()
    assert nor21.public_digest(c,d)==row["full_public_input_sha256"]
    chosen=nor21.certify(c,h,d,row["query_degree"],"stream16")
    assert chosen["mask"]==row["original_exact_public_accept_mask"]>0
    assert chosen["unanimous"] and chosen["chosen_class"] is not None
    pred=corpus.execute(c.query,nor21.program(h,chosen["chosen_class"]))
    return {"pred":pred,"header":h,"mask":chosen["mask"],
            "chosen_class":chosen["chosen_class"],
            "future_sha256":n24.sha_target(pred)}


def reconstruct_private(row,predicted):
    rep,d,fam=row["replicate"],row["degree"],row["family"]
    rng=random.Random(source_seed(rep,d,fam))
    for _ in range(row["selected_attempt"]+1):
        program=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,fam,3*d//2),
                     corpus.execute(x,program)) for _ in range(2))
        query=nor21.public_source(rng,fam,3*d//2)
    assert row["source_seed"]==source_seed(rep,d,fam)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    true=corpus.execute(query,program)
    assert predicted==true
    return true


def unseen_supports(pred,h,public_digest,old):
    m=len(pred)
    assert m in (769,1537)
    offset=int(public_digest[:16],16)%m
    center=m//2
    scores=n24.risk_scores(old,n25.PRIOR,m)
    curv=lambda i:abs(pred[(i-1)%m]-2*pred[i]+pred[(i+1)%m])
    regions={
        "central96":frozenset(range(center-48,center+48)),
        "stride17_96":frozenset((offset+17*j)%m for j in range(96)),
        "low_abs96":frozenset(sorted(range(m),key=lambda i:(abs(pred[i]),i))[:96]),
        "curvature96":frozenset(sorted(range(m),key=lambda i:(-curv(i),i))[:96]),
        "anti_prior96":frozenset(sorted(range(m),key=lambda i:(scores[i],i))[:96]),
        "hash96":frozenset(sorted(range(m),key=lambda i:(
            sha256(f"NOR32|{public_digest}|{i}".encode("ascii")).digest(),i))[:96]),
    }
    assert list(regions)==list(OOD)
    assert all(len(s)==96 and min(s)>=0 and max(s)<m for s in regions.values())
    return regions


def phase_tag(digest,phase):
    stream={"calibration":STREAM_CAL,"evaluation":STREAM_EVAL}[phase]
    return sha256(f"{stream}|{phase}|{digest}".encode("ascii")).hexdigest()


def policy_plans(old,digest,phase):
    tag=phase_tag(digest,phase)
    n=CAL if phase=="calibration" else EVAL
    result={arm:[] for arm in ARMS}
    for trial in range(n):
        plans=n25.policy_schedules(old,tag,trial)
        for arm in ARMS:
            coord=tuple(plans[(arm,K)])
            assert len(coord)==len(set(coord))==K
            result[arm].append(coord)
    return result


def trial_bitsets(plans,m):
    return {arm:[sum(1<<t for t,p in enumerate(plans[arm]) if i in p)
                  for i in range(m)] for arm in ARMS}


def trial_bitsets_fast(plans,m):
    out={arm:[0]*m for arm in ARMS}
    for arm in ARMS:
        cov=out[arm]
        for t,p in enumerate(plans[arm]):
            bit=1<<t
            for i in p:cov[i]|=bit
    return out


def blind_attack(digest,family,reg,rep):
    rng=random.Random(seed(STREAM_BLIND,digest,family,rep))
    return tuple(sorted(rng.sample(sorted(reg[family]),R)))


def greedy_attack(digest,family,target,reg,coverage,rep):
    assert target in ("mixture","hybrid_50","uniform") and rep in range(GREEDY_REPS)
    order=list(sorted(reg[family]))
    random.Random(seed(STREAM_GREEDY,digest,family,target,rep)).shuffle(order)
    chosen=set()
    unions={a:0 for a in ARMS}
    for _ in range(R):
        best=None;best_score=None
        for i in order:
            if i in chosen:continue
            if target=="mixture":
                score=sum(w*(unions[a]|coverage[a][i]).bit_count()
                          for a,w in zip(ARMS,WEIGHTS))
            else:
                score=(unions[target]|coverage[target][i]).bit_count()
            if best_score is None or score<best_score:
                best=i;best_score=score
        assert best is not None
        chosen.add(best)
        for a in ARMS:
            if target=="mixture" or a==target:
                unions[a]|=coverage[a][best]
    return tuple(sorted(chosen))


def old_attack(digest,family,regions,rep):
    rng=random.Random(seed(STREAM_OLD,digest,family,rep))
    return tuple(sorted(rng.sample(sorted(regions[family]),R)))


def mixed_arm(digest,trial):
    i=random.Random(seed(STREAM_MIX,digest,trial)).randrange(16)
    cum=0
    for a,w in zip(ARMS,WEIGHTS):
        cum+=w
        if i<cum:return a
    raise AssertionError("invalid frozen categorical mixture")


def ideal_floor(m,arm):
    if arm=="uniform":
        return Fraction(1)-Fraction(comb(m-R,K),comb(m,K))
    if arm.startswith("hybrid_"):
        u={"hybrid_25":4,"hybrid_50":8,"hybrid_75":12}[arm]
        t=K-u
        return Fraction(1)-Fraction(comb(m-t-R,u),comb(m-t,u))
    return Fraction(0)


def witness_detect(pred,true,checked,altered):
    a=not n24.exact_witness_accepts(pred,true,checked,altered,SHIFTS[0])
    b=not n24.exact_witness_accepts(pred,true,checked,altered,SHIFTS[1])
    exact=bool(set(checked)&set(altered))
    assert a==b==exact
    return a


def evaluate(fixture_file):
    fixture,fixture_sha=read_fixture(fixture_file)
    proposals={row["full_public_input_sha256"]:public_proposal(row)
               for row in fixture["cases"]}
    assert len(proposals)==24
    records=[]
    for row in fixture["cases"]:
        digest=row["full_public_input_sha256"]
        public=proposals[digest]
        pred=public["pred"]
        old=n24.supports(pred,public["header"])
        ood=unseen_supports(pred,public["header"],digest,old)
        overlap={f:{old_f:len(newset&old[old_f])
                    for old_f in ("tail","magnitude","lag","uniform","off_template")}
                 for f,newset in ood.items()}
        cal=policy_plans(old,digest,"calibration")
        exposure=trial_bitsets_fast(cal,len(pred))
        attacks={}
        for f in OOD:
            attacks["blind."+f]=[blind_attack(digest,f,ood,r) for r in range(BLIND_REPS)]
            for target in ("mixture","hybrid_50","uniform"):
                attacks["aware."+target+"."+f]=[
                    greedy_attack(digest,f,target,ood,exposure,r)
                    for r in range(GREEDY_REPS)]
        for f in OLD:
            attacks["legacy."+f]=[old_attack(digest,f,old,r) for r in range(BLIND_REPS)]
        eval_plans=policy_plans(old,digest,"evaluation")
        selected=[mixed_arm(digest,t) for t in range(EVAL)]
        # Only now reconstruct the sealed generating program and actual target.
        truth=reconstruct_private(row,pred)
        assert n24.sha_target(truth)==public["future_sha256"]
        counts={(f,a):0 for f in attacks for a in ARMS}
        realized=Counter({f:0 for f in attacks})
        checked_exact_witness=0
        honest_false_rejects=fully_informed_detected=0
        # Every plan is a genuine exact coefficient witness; all remaining
        # outcomes use equivalent finite integer bitset intersections.
        planmasks={a:[sum(1<<i for i in plan) for plan in plans]
                   for a,plans in eval_plans.items()}
        for arm in ARMS:
            for t,plan in enumerate(eval_plans[arm]):
                if not n24.exact_witness_accepts(pred,truth,plan,frozenset(),1):
                    honest_false_rejects+=1
                informed=n24.informed_attack(plan,old["uniform"],R)
                fully_informed_detected+=int(bool(set(plan)&informed))
            for name,changes in attacks.items():
                for changed in changes:
                    mask=sum(1<<i for i in changed)
                    for t,pm in enumerate(planmasks[arm]):
                        detected=int(bool(pm&mask))
                        counts[(name,arm)]+=detected
                        if selected[t]==arm:realized[name]+=detected
                        if t<4:
                            assert witness_detect(pred,truth,eval_plans[arm][t],changed)==bool(detected)
                            checked_exact_witness+=1
        assert honest_false_rejects==fully_informed_detected==0
        denom=16*EVAL
        total={group:sum(w*counts[(group,arm)] for arm,w in zip(ARMS,WEIGHTS))
               for group in attacks}
        # All groups consist of 4 or 8 frozen attack sets.
        coverage={group:total[group]/(denom*len(changes))
                  for group,changes in attacks.items()}
        rates={arm:{group:counts[(group,arm)]/(EVAL*len(changes))
                    for group,changes in attacks.items()} for arm in ARMS}
        p_floor=sum(Fraction(w,16)*ideal_floor(len(pred),a)
                    for a,w in zip(ARMS,WEIGHTS))
        assert p_floor>0
        records.append({
            "public_sha256":digest,"replicate":row["replicate"],"degree":row["degree"],
            "family":row["family"],"m":len(pred),"mask":public["mask"],
            "chosen_class":public["chosen_class"],
            "public_prediction_sha256":public["future_sha256"],
            "sealed_true_future_sha256_after_freeze":n24.sha_target(truth),
            "old_support_sizes":{k:len(v) for k,v in old.items()},
            "ood_support_sha256":{f:sha256(canonical(sorted(ood[f]))).hexdigest() for f in OOD},
            "ood_old_overlap":overlap,
            "calibration_plans_sha256":sha256(canonical(cal)).hexdigest(),
            "evaluation_plans_sha256":sha256(canonical(eval_plans)).hexdigest(),
            "attacks":{group:[list(ch) for ch in changes] for group,changes in attacks.items()},
            "arm_counts":[{"group":group,"arm":arm,
                           "detected":counts[(group,arm)],
                           "total":EVAL*len(attacks[group])}
                          for group in attacks for arm in ARMS],
            "expected_mixture_numerators":total,
            "expected_mixture_denominators":{group:denom*len(changes)
                                             for group,changes in attacks.items()},
            "realized_mixture_detected":dict(realized),
            "realized_mixture_selected_arms":dict(Counter(selected)),
            "ideal_mixture_floor_fraction":[p_floor.numerator,p_floor.denominator],
            "direct_exact_witness_pairs_verified":checked_exact_witness,
            "honest_target_false_rejections":honest_false_rejects,
            "fully_informed_attack_detections":fully_informed_detected,
            "nonzero_shift_decision_disagreements":0,
        })
    paired=[]
    for rec in records:
        den=rec["expected_mixture_denominators"]
        exp=rec["expected_mixture_numerators"]
        o={(c["group"],c["arm"]):c for c in rec["arm_counts"]}
        mix_blind={f:exp["blind."+f]/den["blind."+f] for f in OOD}
        h_blind={f:o[("blind."+f,"hybrid_50")]["detected"]/o[("blind."+f,"hybrid_50")]["total"]
                 for f in OOD}
        u_blind={f:o[("blind."+f,"uniform")]["detected"]/o[("blind."+f,"uniform")]["total"]
                 for f in OOD}
        mixed_mean=statistics.mean(mix_blind.values())
        h_mean=statistics.mean(h_blind.values())
        u_mean=statistics.mean(u_blind.values())
        aware_mix=statistics.mean(exp["aware.mixture."+f]/den["aware.mixture."+f]
                                  for f in OOD)
        aware_h=statistics.mean(
            o[("aware.hybrid_50."+f,"hybrid_50")]["detected"]/
            o[("aware.hybrid_50."+f,"hybrid_50")]["total"] for f in OOD)
        legacy_mix=sum(w*exp["legacy."+f]/den["legacy."+f] for f,w in zip(OLD,OLD_WEIGHTS))
        legacy_h=sum(w*o[("legacy."+f,"hybrid_50")]["detected"]/
                     o[("legacy."+f,"hybrid_50")]["total"]
                     for f,w in zip(OLD,OLD_WEIGHTS))
        paired.append({
            "public_sha256":rec["public_sha256"],"replicate":rec["replicate"],
            "degree":rec["degree"],"family":rec["family"],
            "ood_blind_mix_family_rates":mix_blind,
            "ood_blind_hybrid50_family_rates":h_blind,
            "ood_blind_uniform_family_rates":u_blind,
            "ood_blind_mix":mixed_mean,"ood_blind_hybrid50":h_mean,
            "ood_blind_uniform":u_mean,
            "ood_blind_mix_minus_hybrid50":mixed_mean-h_mean,
            "ood_worst_family_mix":min(mix_blind.values()),
            "ood_worst_family_hybrid50":min(h_blind.values()),
            "ood_policyaware_mix_self":aware_mix,
            "ood_policyaware_hybrid50_self":aware_h,
            "ood_policyaware_self_gain":aware_mix-aware_h,
            "legacy_template_mix":legacy_mix,
            "legacy_template_hybrid50":legacy_h,
            "legacy_mix_over_hybrid50":legacy_mix/legacy_h if legacy_h>0 else None,
        })
    source_wins=sum(p["ood_blind_mix_minus_hybrid50"]>0 for p in paired)
    worst_wins=sum(p["ood_worst_family_mix"]>p["ood_worst_family_hybrid50"] for p in paired)
    gain=statistics.mean(p["ood_blind_mix_minus_hybrid50"] for p in paired)
    mix_ood=statistics.mean(p["ood_blind_mix"] for p in paired)
    uniform_ood=statistics.mean(p["ood_blind_uniform"] for p in paired)
    aware_wins=sum(p["ood_policyaware_self_gain"]>0 for p in paired)
    aware_gain=statistics.mean(p["ood_policyaware_self_gain"] for p in paired)
    legacy=statistics.mean(p["legacy_template_mix"] for p in paired)
    retained=sum(p["legacy_template_mix"]>=.75*p["legacy_template_hybrid50"]
                 for p in paired)
    exact_control=all(rec["honest_target_false_rejections"]==0 and
                      rec["fully_informed_attack_detections"]==0 and
                      rec["nonzero_shift_decision_disagreements"]==0 and
                      rec["public_prediction_sha256"]==rec["sealed_true_future_sha256_after_freeze"] and
                      Fraction(*rec["ideal_mixture_floor_fraction"])>0
                      for rec in records)
    gates={
        "blind_ood_mean_and_source_wins":source_wins>=18 and gain>=.002,
        "worst_family_source_wins":worst_wins>=16,
        "retain_60pct_of_pure_uniform_ood":mix_ood>0 and mix_ood>=.60*uniform_ood,
        "calibrated_ood_and_legacy_template_retention":
            aware_wins>=16 and aware_gain>=0 and legacy>=.15 and retained>=16,
        "source_witness_floor_replay_controls":exact_control,
    }
    return {
        "schema":SCHEMA,"public_fixture_sha256":fixture_sha,
        "frozen_nor31_outcome_sha256":N31_FULL_SHA,
        "frozen_nor31_train_sha256":N31_TRAIN_SHA,
        "frozen_nor31_public_sha256":N31_PUBLIC_SHA,
        "frozen_mixture_weights":list(WEIGHTS),"auditor_arms":list(ARMS),
        "ood_support_families":list(OOD),"legacy_template_controls":list(OLD),
        "fresh_public_source_count":len(records),
        "calibration_schedules_per_source":CAL,
        "evaluation_schedules_per_source":EVAL,
        "blind_attacks_per_ood_support":BLIND_REPS,
        "greedy_attacks_per_ood_support_policy":GREEDY_REPS,
        "attacked_coordinates":R,"inspected_coordinates":K,
        "nonzero_deltas":list(SHIFTS),
        "public_proposals_frozen_before_private_truth":True,
        "attacks_fixed_before_independent_evaluation_challenges":True,
        "actual_witness_values_exogenous_and_assumed_honest":True,
        "source_records":records,"source_level_primary":paired,
        "primary_ood_blind_mix_mean":mix_ood,
        "primary_ood_blind_uniform_mean":uniform_ood,
        "primary_ood_blind_hybrid50_mean":statistics.mean(p["ood_blind_hybrid50"] for p in paired),
        "primary_ood_blind_source_wins":source_wins,
        "primary_ood_blind_mean_gain_vs_hybrid50":gain,
        "primary_worst_family_source_wins":worst_wins,
        "primary_policyaware_ood_source_wins":aware_wins,
        "primary_policyaware_ood_mean_gain":aware_gain,
        "primary_legacy_template_detection":legacy,
        "primary_legacy_relative_retention_source_count":retained,
        "registered_five_gates":gates,
        "registered_primary_pass":all(gates.values()),
        "scope":("New OOD support-generator rules, not guaranteed position-disjoint "
                 "from any historical support. Four changed positions selected "
                 "blind before heldout audit draws; test seeds are replayable, "
                 "not cryptographic secrets; authenticated target witness external.")
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--make-fixture",type=Path)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    if args.make_fixture:
        if args.fixture or args.out:p.error("fixture generation separate from grading")
        raw=canonical(make_fixture())
        args.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        args.make_fixture.write_bytes(raw)
        print("NOR32 FROZEN NEW PUBLIC SOURCE FIXTURE",len(raw),sha256(raw).hexdigest())
        return
    if not args.fixture:p.error("--fixture required")
    result=evaluate(args.fixture)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(canonical(result))
    print("NOR32 OOD PRIMARY GATES",result["registered_five_gates"])
    print("NOR32 OOD SOURCE PAIRED",[
        (r["degree"],r["family"],
         round(r["ood_blind_mix_minus_hybrid50"],6),
         round(r["ood_policyaware_self_gain"],6),
         round(r["legacy_template_mix"],6))
        for r in result["source_level_primary"]])
    print("NOR32 PRIMARY",result["registered_primary_pass"],
          "OOD BLIND WINS",result["primary_ood_blind_source_wins"],
          "BLIND MEAN GAIN",round(result["primary_ood_blind_mean_gain_vs_hybrid50"],7),
          "WORST SUPPORT WINS",result["primary_worst_family_source_wins"],
          "AWARE WINS",result["primary_policyaware_ood_source_wins"],
          "LEGACY TEMPLATE",round(result["primary_legacy_template_detection"],6))


if __name__=="__main__":
    main()
