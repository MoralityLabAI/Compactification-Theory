"""NOR-36 source-heldout policy-aware curvature auditing.

Registered BEFORE data in REGISTRATION.md first commit ede05db.
Public RRC25 hypotheses, attack choices and independent CAL/EVAL audit
streams are sealed before original native source truth is reconstructed.
All graders use exact target-output coefficient witnesses as an assumed
external information channel. Original NOR25 + NOR35 auditor semantics
remain unchanged; trial randomness is domain separated between phases.
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
from research.nor24_adaptive_audits import experiment as n24
from research.nor25_robust_audits import experiment as n25
from research.nor34_eight_atom_curvature import experiment as n34
from research.nor35_intraatom_curvature import experiment as n35

SCHEMA="nor36.policy_aware_curvature_frozen_greedy_attack.v1"
FIXTURE_SCHEMA="nor36.new_public_only_reps52to55.v1"
REGISTRATION="ede05db28b814d5000ac855fecb3b762876c6f9e"
NOR35_SOURCE_SHA="7a517c8a9f87b15e12362da2dd99d188933a018a233fe2d043ea4aca9c4b02ef"
NOR35_GRADE_SHA="194a0709ed5672dbb8b6706b4d40d6f8f3a28e28164dc3275005d8421ccd05a6"
REPS=(52,53,54,55)
DEGREES=(512,1024)
FAMILIES=nor21.FAMILIES
OFFSET=130036000
ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
POLICIES=("original_h50","atom_random_h50","atom_curv_h50",
          "uniform16","original_mix","atom_curv_mix")
CONTROLS=("top_curvature","bottom_curvature","eight_atom_matched","full_uniform")
CAL=256
EVAL=512
K=16
R=4
RESTARTS=8
STREAM_CAL=202611360071
STREAM_ATTACK=202611360072
STREAM_EVAL=202611360073
STREAM_BLIND=202611360074
STREAM_MIX=202611360075
DELTAS=(1,4294049777)


def canonical(obj):
    return (json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()


def digest(b):
    return sha256(b).hexdigest()


def seed(*parts):
    return int.from_bytes(sha256("|".join(map(str,parts)).encode("ascii")).digest(),"big")


def history():
    ids=n35.historical_ids()
    p=ROOT/"research/nor35_intraatom_curvature/results/nor35_original_hosted.zip"
    with ZipFile(p) as z:
        assert z.testzip() is None
        a=z.read("nor35_fixture.json")
        b=z.read("nor35_full.json")
    assert len(a)==4539616 and digest(a)==NOR35_SOURCE_SHA
    assert len(b)==706150 and digest(b)==NOR35_GRADE_SHA
    past=json.loads(a)
    assert past["schema"]=="nor35.public_only_new_reps48to51.v1"
    old=sorted(ids+[row["full_public_input_sha256"] for row in past["cases"]])
    assert len(old)==len(set(old))==132
    return old


def source_seed(rep,degree,family):
    if rep not in REPS or degree not in DEGREES or family not in FAMILIES:
        raise ValueError("unregistered new NOR36 source")
    return (corpus.SEED+OFFSET+1000003*rep+
            100009*DEGREES.index(degree)+10007*FAMILIES.index(family))


def generate_source(rep,d,family,seen):
    q=3*d//2
    rng=random.Random(source_seed(rep,d,family))
    for attempt in range(100):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,family,q),
                     corpus.execute(x,planted)) for _ in range(2))
        query=nor21.public_source(rng,family,q)
        public=corpus.Case("nor36-new-public",family,demos,query,(),None)
        headers=corpus.possible_headers(public)
        if len(headers)!=1:continue
        h=headers[0]
        if (h.base,h.delay,h.spacing,h.amplitude)!=planted[:4]:continue
        src=nor21.public_digest(public,d)
        if src in seen:continue
        mask=nor21.native_mask(public,h,d)
        if not mask:continue
        futures=nor21.candidate_predictions(query,h,q)
        if len({futures[j] for j in range(16) if mask&(1<<j)})!=1:continue
        truej=4*corpus.OPS.index(planted[4])+corpus.OPS.index(planted[5])
        assert mask&(1<<truej)
        seen.add(src)
        return {
            "split":"TEST","replicate":rep,"degree":d,"query_degree":q,
            "family":family,"source_seed":source_seed(rep,d,family),
            "selected_attempt":attempt,"full_public_input_sha256":src,
            "public_header":{"base":h.base,"delay":h.delay,
                "spacing":h.spacing,"amplitude":h.amplitude},
            "demos":[[list(x[:d+1]),list(y[:d+1])] for x,y in demos],
            "public_query_input":list(query),
            "original_native_public_mask":mask
        }
    raise RuntimeError("NOR36 source exhausted 100 public attempts")


def make_fixture():
    old=history()
    seen=set(old)
    rows=[generate_source(rep,d,fam,seen)
          for rep in REPS for d in DEGREES for fam in FAMILIES]
    assert len(rows)==24 and len(seen)==156
    return {"schema":FIXTURE_SCHEMA,"registration_first_commit":REGISTRATION,
            "replicates":list(REPS),"degrees":list(DEGREES),
            "families":list(FAMILIES),"source_offset":OFFSET,
            "source_only":True,"planted_program_and_private_target_absent":True,
            "attacks_and_challenge_schedules_absent":True,
            "excluded_historical_source_count":len(old),
            "historical_source_roster_sha256":digest(canonical(old)),
            "cases":rows}


def read_fixture(path):
    raw=Path(path).read_bytes()
    f=json.loads(raw)
    past=history()
    assert f["schema"]==FIXTURE_SCHEMA
    assert f["registration_first_commit"]==REGISTRATION
    assert f["replicates"]==list(REPS) and f["source_offset"]==OFFSET
    assert f["source_only"] and f["planted_program_and_private_target_absent"]
    assert f["attacks_and_challenge_schedules_absent"]
    assert f["excluded_historical_source_count"]==len(past)==132
    assert f["historical_source_roster_sha256"]==digest(canonical(past))
    assert len(f["cases"])==24
    ids=[r["full_public_input_sha256"] for r in f["cases"]]
    assert len(set(ids))==24 and set(ids).isdisjoint(past)
    assert {(r["replicate"],r["degree"],r["family"]) for r in f["cases"]}=={
        (rep,d,fam) for rep in REPS for d in DEGREES for fam in FAMILIES}
    assert all(r["split"]=="TEST" and not any(v in r for v in
        ("program","future","target","attack","witness","audit"))
        for r in f["cases"])
    return f,digest(raw)


def proposal(row):
    return n35.proposal(row)


def sealed_truth(row,pred):
    rep,d,fam=row["replicate"],row["degree"],row["family"]
    rng=random.Random(source_seed(rep,d,fam))
    for _ in range(row["selected_attempt"]+1):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,fam,3*d//2),
                     corpus.execute(x,planted)) for _ in range(2))
        query=nor21.public_source(rng,fam,3*d//2)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    target=corpus.execute(query,planted)
    assert pred==target
    return target


def public_regions(pred,header,digest_id):
    return n35.source_atoms(pred,header,digest_id)


def phase_tag(digest_id,phase):
    stream={"cal":STREAM_CAL,"eval":STREAM_EVAL}[phase]
    return sha256(f"{stream}|{phase}|{digest_id}".encode("ascii")).hexdigest()


def schedules(old,cats,scores,digest_id,phase):
    """All original and transformed plans, with CAL/EVAL-separated tail coins."""
    n=CAL if phase=="cal" else EVAL
    root=phase_tag(digest_id,phase)
    # All original NOR35 transformation algorithms are unchanged. Passing
    # phase-separated source keys decorrelates their reserve randomness.
    transform_key=sha256(f"{root}|nor35-transform".encode("ascii")).hexdigest()
    result={mode:{arm:[] for arm in ARMS}
            for mode in ("original","atom_random","atom_curvature50")}
    for t in range(n):
        original=n25.policy_schedules(old,root,t)
        for arm in ARMS:
            base=tuple(original[(arm,K)])
            assert len(set(base))==K
            for mode in result:
                plan=n35.transformed_plan(base,cats,scores,transform_key,
                                          arm,t,mode)
                assert len(plan)==len(set(plan))==K
                result[mode][arm].append(plan)
    return result


def pure_components(policy):
    assert policy in POLICIES
    if policy=="original_h50":
        return (("original","hybrid_50",16),)
    if policy=="atom_random_h50":
        return (("atom_random","hybrid_50",16),)
    if policy=="atom_curv_h50":
        return (("atom_curvature50","hybrid_50",16),)
    if policy=="uniform16":
        return (("original","uniform",16),)
    mode="original" if policy=="original_mix" else "atom_curvature50"
    return tuple((mode,a,w) for a,w in zip(ARMS,WEIGHTS) if w>0)


def trial_coverage(plans,m):
    cov={}
    for mode,arms in plans.items():
        for arm,rows in arms.items():
            acc=[0]*m
            for t,checked in enumerate(rows):
                b=1<<t
                for i in checked:
                    acc[i]|=b
            cov[(mode,arm)]=acc
    return cov


def greedy_attack(exposure,policy,digest_id,restart):
    parts=pure_components(policy)
    m=len(exposure[parts[0][:2]])
    indices=list(range(m))
    random.Random(seed(STREAM_ATTACK,digest_id,policy,restart)).shuffle(indices)
    selected=set()
    seen={p[:2]:0 for p in parts}
    for _ in range(R):
        best=None
        best_cost=None
        for i in indices:
            if i in selected:continue
            cost=sum(w*(seen[(mode,arm)]|exposure[(mode,arm)][i]).bit_count()
                     for mode,arm,w in parts)
            if best_cost is None or cost<best_cost:
                best=i;best_cost=cost
        assert best is not None
        selected.add(best)
        for mode,arm,w in parts:
            seen[(mode,arm)]|=exposure[(mode,arm)][best]
    assert len(selected)==R
    return tuple(sorted(selected))


def blind_attacks(digest_id,cats,union,cu,cv,bu,bv,m):
    result={name:[] for name in CONTROLS}
    for rep in range(RESTARTS):
        rng=random.Random(seed(STREAM_BLIND,digest_id,"top_curvature",rep))
        top=tuple(sorted(rng.sample(sorted(cu),2)+rng.sample(sorted(cv),2)))
        rng=random.Random(seed(STREAM_BLIND,digest_id,"bottom_curvature",rep))
        bottom=tuple(sorted(rng.sample(sorted(bu),2)+rng.sample(sorted(bv),2)))
        quotas=[sum(i in a for i in top) for a in cats]
        assert quotas[0]==2 and sum(quotas)==R
        rng=random.Random(seed(STREAM_BLIND,digest_id,"eight_atom_matched",rep))
        matched=tuple(sorted(i for j,q in enumerate(quotas)
                             for i in rng.sample(sorted(cats[j]),q)))
        rng=random.Random(seed(STREAM_BLIND,digest_id,"full_uniform",rep))
        uniform=tuple(sorted(rng.sample(range(m),R)))
        assert len(set(top))==len(set(bottom))==len(set(matched))==len(set(uniform))==R
        assert sum(i in union for i in bottom)==2
        assert [sum(i in a for i in matched) for a in cats]==quotas
        for name,a in zip(CONTROLS,(top,bottom,matched,uniform)):
            result[name].append(a)
    return result


def mixed_choice(digest_id,t):
    coin=random.Random(seed(STREAM_MIX,digest_id,t)).randrange(16)
    total=0
    for arm,w in zip(ARMS,WEIGHTS):
        total+=w
        if coin<total:return arm
    raise AssertionError("bad mixture")


def probability_top(cu,cv,plan):
    return n35.two_block_detect(cu,cv,plan)


def exact_policy_top(plans,cu,cv):
    return {
        policy:sum((Fraction(w,16)*sum(
            (probability_top(cu,cv,s) for s in plans[mode][arm]),Fraction(0))/EVAL
            for mode,arm,w in pure_components(policy)),Fraction(0))
        for policy in POLICIES
    }


def probability_uniform(m):
    return Fraction(1)-Fraction(comb(m-16,4),comb(m,4))


def evaluate(fixture_path):
    source,fixture_sha=read_fixture(fixture_path)
    proposed={r["full_public_input_sha256"]:proposal(r) for r in source["cases"]}
    assert len(proposed)==24
    records=[]
    for row in source["cases"]:
        src=row["full_public_input_sha256"]
        p=proposed[src]
        pred=p["pred"]
        old,cats,union,cu,cv,bu,bv,scores=public_regions(pred,p["header"],src)
        cal=schedules(old,cats,scores,src,"cal")
        exposure=trial_coverage(cal,len(pred))
        greedy={a:[greedy_attack(exposure,a,src,i) for i in range(RESTARTS)]
                for a in POLICIES}
        blind=blind_attacks(src,cats,union,cu,cv,bu,bv,len(pred))
        all_attacks={**greedy,**blind}
        evaluation=schedules(old,cats,scores,src,"eval")
        mixed=[mixed_choice(src,t) for t in range(EVAL)]
        # All attacks and actual audit plans were now fixed on public-only
        # decisions; restore original private generator only here.
        target=sealed_truth(row,pred)
        assert p["sha"]==n24.sha_target(target)
        cov=trial_coverage(evaluation,len(pred))
        counts={attacker:{policy:0 for policy in POLICIES}
                for attacker in all_attacks}
        realized={attacker:0 for attacker in all_attacks}
        honest_errors=informed_hits=direct_checks=0
        arm_draw_count=Counter(mixed)
        for mode,arms in evaluation.items():
            for arm,plans in arms.items():
                for t,checked in enumerate(plans):
                    if not n24.exact_witness_accepts(pred,target,checked,frozenset(),1):
                        honest_errors+=1
                    informed=n24.informed_attack(checked,old["uniform"],R)
                    informed_hits+=int(bool(set(checked)&informed))
        # Count each pure-policy arm independently; then exact 16-quantum
        # averages provide both mixed-policy detector payoffs.
        pure_counts={}
        for name,attacks in all_attacks.items():
            attackmasks=[sum(1<<i for i in a) for a in attacks]
            for policy in POLICIES:
                components=pure_components(policy)
                numerator=0
                for mode,arm,w in components:
                    for attackmask in attackmasks:
                        # Coordinate exposure is an exact k16 witness decision
                        # for a fixed blind alteration set.
                        union_bits=0
                        for i in range(len(pred)):
                            if attackmask & (1<<i):
                                union_bits|=cov[(mode,arm)][i]
                        numerator+=w*union_bits.bit_count()
                counts[name][policy]=numerator
        denom=16*RESTARTS*EVAL
        for policy in POLICIES:
            assert sum(w for _,_,w in pure_components(policy))==16
        # Direct authentic source-target integer comparisons on two
        # representative plans and two fixed attacks in every class.
        for policy in POLICIES:
            for name,attacks in all_attacks.items():
                for attack in attacks[:2]:
                    for t in range(2):
                        for mode,arm,w in pure_components(policy):
                            plan=evaluation[mode][arm][t]
                            detected=bool(set(plan)&set(attack))
                            a=not n24.exact_witness_accepts(
                                pred,target,plan,frozenset(attack),DELTAS[0])
                            b=not n24.exact_witness_accepts(
                                pred,target,plan,frozenset(attack),DELTAS[1])
                            assert a==b==detected
                            direct_checks+=1
        # The transformed source/trial component must preserve the complete
        # original eight atom quota vector for every individual component.
        atom_violations=0
        for arm in ARMS:
            for t in range(EVAL):
                base=set(evaluation["original"][arm][t])
                quotas=tuple(len(base&a) for a in cats)
                for mode in ("atom_random","atom_curvature50"):
                    plan=evaluation[mode][arm][t]
                    atom_violations+=int(tuple(len(set(plan)&a) for a in cats)!=quotas)
        assert honest_errors==informed_hits==atom_violations==0
        top=exact_policy_top(evaluation,cu,cv)
        uniform_exact=probability_uniform(len(pred))
        assert 0<uniform_exact<1
        floor=sum((Fraction(w,16)*n35.nor34.ideal_floor(len(pred),arm)
                   for arm,w in zip(ARMS,WEIGHTS)),Fraction(0))
        assert floor>0
        records.append({
            "public_sha256":src,"replicate":row["replicate"],
            "degree":row["degree"],"family":row["family"],
            "future_length":len(pred),"mask":p["mask"],
            "chosen_class":p["class"],
            "public_prediction_sha256":p["sha"],
            "private_true_future_sha256_postchoice":n24.sha_target(target),
            "historical_atom_sizes":[len(a) for a in cats],
            "top_curvature_support_sha256":digest(canonical(sorted(cu|cv))),
            "bottom_curvature_support_sha256":digest(canonical(sorted(bu|bv))),
            "calibration_plan_sha256":digest(canonical(cal)),
            "heldout_plan_sha256":digest(canonical(evaluation)),
            "greedy_attacks":{k:[list(a) for a in v] for k,v in greedy.items()},
            "blind_attacks":{k:[list(a) for a in v] for k,v in blind.items()},
            "unique_greedy_attacks":{k:len(set(v)) for k,v in greedy.items()},
            "crossplay_weighted_numerators":counts,
            "crossplay_denominator":denom,
            "realized_mix_arm_draw_counts":dict(arm_draw_count),
            "top_support_exact_fraction":{k:[v.numerator,v.denominator] for k,v in top.items()},
            "uniform_blind_exact_fraction":[uniform_exact.numerator,uniform_exact.denominator],
            "original_mix_blind_floor_fraction":[floor.numerator,floor.denominator],
            "honest_false_rejects":honest_errors,
            "schedule_informed_detection":informed_hits,
            "atom_quota_mismatches":atom_violations,
            "nonzero_shift_disagreements":0,
            "direct_witness_pairs":direct_checks,
        })
    per_source=[]
    for rec in records:
        denom=rec["crossplay_denominator"]
        a=rec["crossplay_weighted_numerators"]
        t={k:Fraction(*v) for k,v in rec["top_support_exact_fraction"].items()}
        self_curv=a["atom_curv_h50"]["atom_curv_h50"]/denom
        self_orig=a["original_h50"]["original_h50"]/denom
        x_curv=a["atom_curv_h50"]["atom_curv_h50"]/denom
        x_orig=a["atom_curv_h50"]["original_h50"]/denom
        mix_self_curv=a["atom_curv_mix"]["atom_curv_mix"]/denom
        mix_self_orig=a["original_mix"]["original_mix"]/denom
        per_source.append({
            "public_sha256":rec["public_sha256"],
            "degree":rec["degree"],"family":rec["family"],
            "self_curv":self_curv,"self_original":self_orig,
            "self_curv_minus_original":self_curv-self_orig,
            "fixed_curv_attack_curv_minus_original":x_curv-x_orig,
            "top_curv_h50_gain":float(t["atom_curv_h50"]-t["original_h50"]),
            "mix_self_curv_minus_original":mix_self_curv-mix_self_orig,
            "top_curv_mix_gain":float(t["atom_curv_mix"]-t["original_mix"]),
            "uniform_self":a["uniform16"]["uniform16"]/denom,
        })
    means={name:statistics.mean(s[name] for s in per_source)
           for name in ("self_curv_minus_original",
                        "fixed_curv_attack_curv_minus_original",
                        "top_curv_h50_gain",
                        "mix_self_curv_minus_original",
                        "top_curv_mix_gain")}
    wins={
        "self_curv_weaker":sum(r["self_curv_minus_original"]<0 for r in per_source),
        "fixed_curv_attacks_expose_weakness":sum(
            r["fixed_curv_attack_curv_minus_original"]<0 for r in per_source),
        "top_h50_gain":sum(r["top_curv_h50_gain"]>0 for r in per_source),
        "mix_self_curv_weaker":sum(r["mix_self_curv_minus_original"]<0 for r in per_source),
        "top_mix_gain":sum(r["top_curv_mix_gain"]>0 for r in per_source),
    }
    valid=all(rec["mask"]>0 and
              rec["private_true_future_sha256_postchoice"]==rec["public_prediction_sha256"] and
              rec["honest_false_rejects"]==rec["schedule_informed_detection"]==
              rec["atom_quota_mismatches"]==rec["nonzero_shift_disagreements"]==0 and
              Fraction(*rec["original_mix_blind_floor_fraction"])>0 and
              rec["direct_witness_pairs"]>0
              for rec in records)
    gates={
        "curvature_selfplay_less_robust":wins["self_curv_weaker"]>=18 and
            means["self_curv_minus_original"]<=-.005,
        "same_curvature_attack_exposes_weakness":
            wins["fixed_curv_attacks_expose_weakness"]>=18 and
            means["fixed_curv_attack_curv_minus_original"]<=-.005,
        "top_curvature_source_aligned_gain":
            wins["top_h50_gain"]>=18 and means["top_curv_h50_gain"]>=.10,
        "mixture_selfplay_erosion_and_top_gain":
            wins["mix_self_curv_weaker"]>=18 and means["mix_self_curv_minus_original"]<=-.005 and
            wins["top_mix_gain"]>=18 and means["top_curv_mix_gain"]>=.10,
        "source_attacks_audits_truth_and_replay":valid,
    }
    return {
        "schema":SCHEMA,"registration_first_commit":REGISTRATION,
        "public_only_fixture_sha256":fixture_sha,
        "fresh_source_count":len(records),"excluded_historical_sources":132,
        "frozen_mixture_weights":list(WEIGHTS),"attacker_and_auditor_policies":list(POLICIES),
        "blind_control_families":list(CONTROLS),
        "calibration_trials":CAL,"heldout_trials":EVAL,
        "greedy_attack_restarts":RESTARTS,"budget":K,"attack_radius":R,
        "nonzero_shifts":list(DELTAS),
        "source_proposals_frozen_before_private_truth":True,
        "attack_sets_committed_before_heldout_schedules":True,
        "external_authenticated_exact_target_witness_assumed":True,
        "source_records":records,"primary_by_source":per_source,
        "source_equal_mean_primary":means,"source_direction_wins":wins,
        "registered_five_gates":gates,"registered_primary_pass":all(gates.values()),
        "scope":("Approximate full-output greedy policy-aware attacker using "
                 "256 calibration schedules and 8 restarts, evaluated on 512 "
                 "distinct seeded schedules. External true-target coefficient "
                 "oracle and hidden independent audit coins are assumptions.")
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--make-fixture",type=Path)
    parser.add_argument("--fixture",type=Path)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.make_fixture:
        if args.fixture or args.out:parser.error("fixture and private grading are separate jobs")
        raw=canonical(make_fixture())
        args.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        args.make_fixture.write_bytes(raw)
        print("NOR36 FIRST FROZEN SOURCE-ONLY FIXTURE",len(raw),digest(raw))
        return
    if not args.fixture:parser.error("--fixture required")
    r=evaluate(args.fixture)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(canonical(r))
    print("NOR36 REGISTERED PRIMARY PASS",r["registered_primary_pass"])
    print("NOR36 FIVE GATES",r["registered_five_gates"])
    print("NOR36 SOURCE WINS",r["source_direction_wins"])
    print("NOR36 MEAN GAINS",r["source_equal_mean_primary"])


if __name__=="__main__":
    main()
