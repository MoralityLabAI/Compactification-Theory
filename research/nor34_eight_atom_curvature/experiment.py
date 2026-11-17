"""NOR-34: prospectively registered eight-atom curvature transfer experiment.

All public RRC25 sources/proposals, attack variants, and audit schedules
are fixed BEFORE original generating programs/hidden targets are restored.
The scientific PRIMARY estimands integrate exact conditional attack laws
with Fraction, rather than estimating them using 16 sampled attacks.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from math import comb
from pathlib import Path
from zipfile import ZipFile
import json,random,statistics,sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import experiment as nor21
from research.nor24_adaptive_audits import experiment as nor24
from research.nor25_robust_audits import experiment as nor25
from research.nor33_overlap_matched_transfer import experiment as nor33

SCHEMA="nor34.sourceheldout_eight_atom_curvature_residual.v1"
FIXTURE_SCHEMA="nor34.new_public_only_reps44to47.v1"
REGISTRATION_SHA="ef6d7b4d0374d0310a640258d94eaf474fb66c4b"
NOR33_PUBLIC_SHA="16f3b397aea17adbd1462666142a13e933c8da67a1286e1e6418d32b22090db8"
NOR33_GRADE_SHA="88d37a957f7d3be42b76213b351df921a08556654a90a2ed21df1b66788d6a62"
OFFSET=130034000
REPS=(44,45,46,47)
DEGREES=(512,1024)
FAMILIES=nor21.FAMILIES
ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
LAWS=("curvature","coarse","eight_atom")
STREAM_CURV=202611340071
STREAM_FINE=202611340072
STREAM_COARSE=202611340073
STREAM_AUDIT=202611340074
STREAM_MIXED=202611340075
DELTAS=(1,4294049777)
ATTACKS=16
TRIALS=512
K=16
R=4


def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()


def digest(x):
    return sha256(x).hexdigest()


def seed(*fields):
    return int.from_bytes(sha256("|".join(map(str,fields)).encode("ascii")).digest(),"big")


def historic_source_hashes():
    prev=nor33.past_source_digests()
    archive=ROOT/"research/nor33_overlap_matched_transfer/results/nor33_original_hosted.zip"
    with ZipFile(archive) as z:
        assert z.testzip() is None
        fixture=z.read("nor33_fixture.json")
        grade=z.read("nor33_full.json")
    assert len(fixture)==4547200 and digest(fixture)==NOR33_PUBLIC_SHA
    assert len(grade)==1761233 and digest(grade)==NOR33_GRADE_SHA
    old=json.loads(fixture)
    assert old["schema"]=="nor33.source_only_reps40to43.v1"
    ids=sorted(prev+[r["full_public_input_sha256"] for r in old["cases"]])
    assert len(ids)==len(set(ids))==84
    return ids


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
        public=corpus.Case("nor34-source",fam,demos,query,(),None)
        heads=corpus.possible_headers(public)
        if len(heads)!=1:continue
        h=heads[0]
        if (h.base,h.delay,h.spacing,h.amplitude)!=planted[:4]:continue
        sha=nor21.public_digest(public,d)
        if sha in seen:continue
        mask=nor21.native_mask(public,h,d)
        if mask==0:continue
        pred=nor21.candidate_predictions(query,h,q)
        if len({pred[j] for j in range(16) if mask&(1<<j)})!=1:continue
        planted_idx=4*corpus.OPS.index(planted[4])+corpus.OPS.index(planted[5])
        assert mask&(1<<planted_idx)
        seen.add(sha)
        return {
            "split":"TEST","replicate":rep,"degree":d,
            "query_degree":q,"family":fam,
            "source_seed":source_seed(rep,d,fam),"selected_attempt":attempt,
            "full_public_input_sha256":sha,
            "public_header":{"base":h.base,"delay":h.delay,
                             "spacing":h.spacing,"amplitude":h.amplitude},
            "demos":[[list(x[:d+1]),list(y[:d+1])] for x,y in demos],
            "public_query_input":list(query),"native_public_mask":mask}
    raise RuntimeError("NOR34 source exhausted 100 admissible attempts")


def make_fixture():
    historic=historic_source_hashes()
    seen=set(historic)
    cases=[new_source(rep,d,fam,seen)
           for rep in REPS for d in DEGREES for fam in FAMILIES]
    assert len(cases)==24 and len(seen)==108
    return {"schema":FIXTURE_SCHEMA,"registration_commit":REGISTRATION_SHA,
            "replicates":list(REPS),"offset":OFFSET,
            "observation_degrees":list(DEGREES),
            "families":list(FAMILIES),"public_only":True,
            "private_future_and_program_absent":True,
            "attacks_and_witness_absent":True,
            "historical_source_count":84,
            "historical_public_digests_sha256":digest(canonical(historic)),
            "cases":cases}


def read_fixture(path):
    raw=Path(path).read_bytes()
    fixture=json.loads(raw)
    historic=historic_source_hashes()
    assert fixture["schema"]==FIXTURE_SCHEMA
    assert fixture["registration_commit"]==REGISTRATION_SHA
    assert fixture["replicates"]==list(REPS) and fixture["offset"]==OFFSET
    assert fixture["public_only"] and fixture["private_future_and_program_absent"]
    assert fixture["attacks_and_witness_absent"]
    assert fixture["historical_source_count"]==len(historic)==84
    assert fixture["historical_public_digests_sha256"]==digest(canonical(historic))
    assert len(fixture["cases"])==24
    ids=[r["full_public_input_sha256"] for r in fixture["cases"]]
    assert len(set(ids))==24 and set(ids).isdisjoint(historic)
    assert {(r["replicate"],r["degree"],r["family"]) for r in fixture["cases"]}=={
        (rep,d,f) for rep in REPS for d in DEGREES for f in FAMILIES}
    assert all(r["split"]=="TEST" and not any(k in r for k in
               ("program","future","target","attack","witness"))
               for r in fixture["cases"])
    return fixture,digest(raw)


def private_future_after_all_public_choices(row,predicted):
    rep,d,fam=row["replicate"],row["degree"],row["family"]
    rng=random.Random(source_seed(rep,d,fam))
    for _ in range(row["selected_attempt"]+1):
        program=nor21.new_header_program(rng)
        demos=tuple((src:=nor21.public_source(rng,fam,3*d//2),
                     corpus.execute(src,program)) for _ in range(2))
        query=nor21.public_source(rng,fam,3*d//2)
    assert row["source_seed"]==source_seed(rep,d,fam)
    assert row["demos"]==[[list(src[:d+1]),list(y[:d+1])] for src,y in demos]
    assert row["public_query_input"]==list(query)
    target=corpus.execute(query,program)
    assert target==predicted
    return target


def pattern_index(i,old):
    return ((1 if i in old["tail"] else 0)
            +(2 if i in old["magnitude"] else 0)
            +(4 if i in old["lag"] else 0))


def partition(pred,header,public_sha):
    old=nor24.supports(pred,header)
    m=len(pred)
    categories=[frozenset(i for i in range(m) if pattern_index(i,old)==j)
                for j in range(8)]
    assert sum(len(v) for v in categories)==m
    u=frozenset(range(m))-categories[0]
    assert u==nor33.risk_union(old)
    rank=nor33.full_rankings(pred,header,public_sha,old)["curvature"]
    c=nor33.forced_support(rank,u,m,2)
    cu=frozenset(c&u)
    cv=frozenset(c-u)
    assert len(cu)==len(cv)==48 and len(c)==96
    return old,categories,u,cu,cv


def attack_triplets(public_sha,categories,u,cu,cv):
    atoms=[sorted(x) for x in categories]
    result=[]
    for t in range(ATTACKS):
        rng=random.Random(seed(STREAM_CURV,public_sha,t))
        curvature=tuple(sorted(rng.sample(sorted(cu),2)+rng.sample(sorted(cv),2)))
        quotas=[sum(x in cell for x in curvature) for cell in categories]
        assert sum(quotas)==R and quotas[0]==2
        fine_rng=random.Random(seed(STREAM_FINE,public_sha,t))
        fine=tuple(sorted(i for j,q in enumerate(quotas)
                          for i in fine_rng.sample(atoms[j],q)))
        coarse_rng=random.Random(seed(STREAM_COARSE,public_sha,t))
        coarse=tuple(sorted(
            coarse_rng.sample(sorted(u),2)+
            coarse_rng.sample(sorted(categories[0]),2)))
        assert len(set(curvature))==len(set(fine))==len(set(coarse))==4
        assert sum(i in u for i in coarse)==2
        assert all(sum(i in cell for i in fine)==q
                   for cell,q in zip(categories,quotas))
        result.append({
            "curvature":curvature,"eight_atom":fine,
            "coarse":coarse,"quotas":quotas})
    return result


def schedules(old,public_sha):
    phase=sha256(f"{STREAM_AUDIT}|nor34|{public_sha}".encode("ascii")).hexdigest()
    result={a:[] for a in ARMS}
    for t in range(TRIALS):
        original=nor25.policy_schedules(old,phase,t)
        for a in ARMS:
            checked=tuple(original[(a,K)])
            assert len(checked)==len(set(checked))==K
            result[a].append(checked)
    return result


def mixed_arm(public_sha,t):
    rand=random.Random(seed(STREAM_MIXED,public_sha,t)).randrange(16)
    acc=0
    for arm,w in zip(ARMS,WEIGHTS):
        acc+=w
        if rand<acc:return arm
    raise AssertionError("Invalid NOR31 weights")


def conditional_binom_nohit(size,checked,quota):
    assert 0<=checked<=size and 0<=quota<=size
    return Fraction(comb(size-checked,quota),comb(size,quota))


def exact_probabilities(categories,cu,cv):
    """Exact conditional clean-witness fractions for each membership pattern.

    Curvature law is hypergeometric on curated 48+48 support.
    Coarse law is hypergeometric on full U and full V.
    Fine law first selects two+two curated positions, then redraws
    from the full category of every originally chosen position.
    """
    sizes=tuple(len(s) for s in categories)
    ccounts=tuple(len((cu|cv)&s) for s in categories)
    u_size=sum(sizes[1:])
    v_size=sizes[0]
    assert ccounts[0]==48 and sum(ccounts[1:])==48
    assert v_size>=48 and u_size>=48
    u_pairs=[]
    for j in range(1,8):
        sj=ccounts[j]
        if sj>=2:
            u_pairs.append((j,j,comb(sj,2)))
        for k in range(j+1,8):
            if sj and ccounts[k]:
                u_pairs.append((j,k,sj*ccounts[k]))
    assert sum(weight for _,_,weight in u_pairs)==comb(48,2)

    @lru_cache(maxsize=250000)
    def from_checked(
            counts:tuple[int,...], curv_u:int,curv_v:int):
        assert len(counts)==8
        assert curv_u<=48 and curv_v<=48
        unhit_curv=(
            conditional_binom_nohit(48,curv_u,2)*
            conditional_binom_nohit(48,curv_v,2))
        unhit_coarse=(
            conditional_binom_nohit(u_size,sum(counts[1:]),2)*
            conditional_binom_nohit(v_size,counts[0],2))
        unhit_u=Fraction(0)
        for j,k,w in u_pairs:
            factor=(conditional_binom_nohit(sizes[j],counts[j],2)
                    if j==k else
                    Fraction(sizes[j]-counts[j],sizes[j])*
                    Fraction(sizes[k]-counts[k],sizes[k]))
            unhit_u+=Fraction(w,comb(48,2))*factor
        unhit_fine=unhit_u*conditional_binom_nohit(sizes[0],counts[0],2)
        return {"curvature":1-unhit_curv,
                "coarse":1-unhit_coarse,
                "eight_atom":1-unhit_fine}
    return from_checked


def ideal_floor(m,arm):
    if arm=="uniform":
        return Fraction(1)-Fraction(comb(m-4,16),comb(m,16))
    if arm.startswith("hybrid_"):
        u={"hybrid_25":4,"hybrid_50":8,"hybrid_75":12}[arm]
        t=K-u
        return Fraction(1)-Fraction(comb(m-t-R,u),comb(m-t,u))
    return Fraction(0)


def compute_means(theory):
    """Source-conditional signed hybrid-minus-mixture and residuals."""
    expected={}
    H={}
    for law in LAWS:
        v=theory[law]
        mix=sum((Fraction(w,16)*v[arm] for w,arm in zip(WEIGHTS,ARMS)),
                Fraction(0))
        expected[law]=float(mix)
        H[law]=float(v["hybrid_50"]-mix)
    return expected,H,{"coarse":H["curvature"]-H["coarse"],
                       "fine":H["curvature"]-H["eight_atom"]}


def evaluate(fixture_path):
    fixture,fixture_sha=read_fixture(fixture_path)
    # All proposed target continuations selected from public data first.
    proposals={row["full_public_input_sha256"]:nor33.public_proposal(row)
               for row in fixture["cases"]}
    assert len(proposals)==24
    records=[]
    for row in fixture["cases"]:
        key=row["full_public_input_sha256"]
        p=proposals[key]
        predicted=p["pred"]
        old,cats,u,cu,cv=partition(predicted,p["header"],key)
        attack=attack_triplets(key,cats,u,cu,cv)
        policy=schedules(old,key)
        selections=[mixed_arm(key,i) for i in range(TRIALS)]
        # Private generating program is only reconstructed after attack/plan
        # decisions have been frozen by public source identity and seed.
        target=private_future_after_all_public_choices(row,predicted)
        assert p["future_sha256"]==nor24.sha_target(target)
        calc=exact_probabilities(cats,cu,cv)
        theory={law:{a:Fraction(0) for a in ARMS} for law in LAWS}
        empirical={law:{a:0 for a in ARMS} for law in LAWS}
        realized={law:0 for law in LAWS}
        honest_false=adaptive_hits=direct_checks=0
        attacksByLaw={law:[tuple(a[law]) for a in attack] for law in LAWS}
        attackMasks={law:[sum(1<<i for i in a) for a in vals]
                     for law,vals in attacksByLaw.items()}
        for arm in ARMS:
            for trial,checked in enumerate(policy[arm]):
                if not nor24.exact_witness_accepts(
                    predicted,target,checked,frozenset(),DELTAS[0]):
                    honest_false+=1
                informed=nor24.informed_attack(checked,old["uniform"],4)
                adaptive_hits+=bool(set(checked)&informed)
                check_counts=tuple(len(set(checked)&c) for c in cats)
                curvu=len(set(checked)&cu)
                curvv=len(set(checked)&cv)
                probs=calc(check_counts,curvu,curvv)
                for law in LAWS:
                    theory[law][arm]+=probs[law]
                pm=sum(1<<i for i in checked)
                for law in LAWS:
                    for j,am in enumerate(attackMasks[law]):
                        hit=bool(pm&am)
                        empirical[law][arm]+=int(hit)
                        if selections[trial]==arm:
                            realized[law]+=int(hit)
                        if trial<4:
                            attackset=frozenset(attacksByLaw[law][j])
                            hit1=not nor24.exact_witness_accepts(
                                predicted,target,checked,attackset,DELTAS[0])
                            hit2=not nor24.exact_witness_accepts(
                                predicted,target,checked,attackset,DELTAS[1])
                            assert hit1==hit2==hit
                            direct_checks+=1
        assert honest_false==adaptive_hits==0
        assert direct_checks==len(ARMS)*len(LAWS)*ATTACKS*4
        theory={law:{arm:val/TRIALS for arm,val in arms.items()}
                for law,arms in theory.items()}
        expected_mix,H,residual=compute_means(theory)
        empirical_mix={}
        empirical_H={}
        for law in LAWS:
            v=empirical[law]
            mix=sum(w*v[a] for w,a in zip(WEIGHTS,ARMS))/(
                16*ATTACKS*TRIALS)
            empirical_mix[law]=mix
            empirical_H[law]=v["hybrid_50"]/(ATTACKS*TRIALS)-mix
        empirical_residual={
            "coarse":empirical_H["curvature"]-empirical_H["coarse"],
            "fine":empirical_H["curvature"]-empirical_H["eight_atom"]}
        theoretical_fraction={
            law:{arm:[x.numerator,x.denominator] for arm,x in arms.items()}
            for law,arms in theory.items()}
        theoretical_float={
            law:{arm:float(x) for arm,x in arms.items()} for law,arms in theory.items()}
        floor=sum((Fraction(w,16)*ideal_floor(len(predicted),arm)
                   for w,arm in zip(WEIGHTS,ARMS)),Fraction(0))
        assert floor>0
        records.append({
            "public_sha256":key,"replicate":row["replicate"],
            "degree":row["degree"],"family":row["family"],
            "native_public_mask":p["mask"],
            "chosen_class":p["chosen_class"],
            "public_prediction_sha256":p["future_sha256"],
            "private_native_true_future_sha256_postchoice":nor24.sha_target(target),
            "future_length":len(predicted),
            "pattern_population":[len(c) for c in cats],
            "curvature_support_pattern_counts":[len((cu|cv)&c) for c in cats],
            "curvature_support_U_size":len(cu),
            "curvature_support_V_size":len(cv),
            "historic_union_size":len(u),
            "curvature_support_sha256":digest(canonical(sorted(cu|cv))),
            "paired_attack_records":[{
                "curvature":list(a["curvature"]),
                "eight_atom":list(a["eight_atom"]),
                "coarse":list(a["coarse"]),
                "eight_atom_quota":a["quotas"]} for a in attack],
            "frozen_policy_schedule_sha256":digest(canonical(policy)),
            "realized_mixed_arm_counts":dict(Counter(selections)),
            "theoretical_conditional_detection_exact":theoretical_fraction,
            "theoretical_conditional_detection_float":theoretical_float,
            "empirical_detected_per_arm":empirical,
            "empirical_total_per_arm":ATTACKS*TRIALS,
            "mixed_realized_detected":realized,
            "source_expected_mixture_detection":expected_mix,
            "source_expected_hybrid_advantage":H,
            "source_theoretical_residual":residual,
            "source_empirical_mixture_detection":empirical_mix,
            "source_empirical_hybrid_advantage":empirical_H,
            "source_empirical_residual":empirical_residual,
            "ideal_mixture_floor_fraction":[floor.numerator,floor.denominator],
            "honest_false_rejections":honest_false,
            "schedule_informed_attacker_detections":adaptive_hits,
            "shift_detection_disagreements":0,
            "direct_exact_integer_witness_checks":direct_checks,
        })
    original=[r["source_theoretical_residual"] for r in records]
    empirical_res=[r["source_empirical_residual"] for r in records]
    coarse=statistics.mean(r["coarse"] for r in original)
    fine=statistics.mean(r["fine"] for r in original)
    empirical_coarse=statistics.mean(r["coarse"] for r in empirical_res)
    empirical_fine=statistics.mean(r["fine"] for r in empirical_res)
    count_coarse=sum(r["coarse"]>0 for r in original)
    count_att=sum(abs(r["fine"])<abs(r["coarse"]) for r in original)
    checks={
        "coarse_residual_replicates":count_coarse>=18 and coarse>=.015,
        "fine_atom_matching_attenuates":count_att>=18 and
            coarse!=0 and abs(fine)<=.30*abs(coarse),
        "fine_residual_at_most_half_percent":abs(fine)<=.005,
        "paired_samples_calibrate_exact_laws":abs(empirical_coarse-coarse)<=.010 and
            abs(empirical_fine-fine)<=.010,
        "full_original_source_witness_replay_controls":all(
            r["public_prediction_sha256"]==r["private_native_true_future_sha256_postchoice"]
            and r["honest_false_rejections"]==0
            and r["schedule_informed_attacker_detections"]==0
            and r["shift_detection_disagreements"]==0
            and r["curvature_support_U_size"]==r["curvature_support_V_size"]==48
            and r["direct_exact_integer_witness_checks"]==1152
            and Fraction(*r["ideal_mixture_floor_fraction"])>0
            for r in records),
    }
    return {
        "schema":SCHEMA,"preregistered_first_commit":REGISTRATION_SHA,
        "source_public_fixture_sha256":fixture_sha,
        "new_public_source_count":len(records),"excluded_historic_sources":84,
        "auditor_arms":list(ARMS),"frozen_mix_weights":list(WEIGHTS),
        "paired_attack_laws":list(LAWS),"attacks_per_source_per_law":ATTACKS,
        "heldout_audit_schedules_per_source_arm":TRIALS,
        "audit_budget":K,"altered_coordinates":R,"nonzero_shifts":list(DELTAS),
        "public_proposals_frozen_before_private_truth":True,
        "all_attack_sets_fixed_before_audit_schedule":True,
        "honest_external_target_coefficient_witness_assumed":True,
        "source_records":records,
        "source_theoretical_coarse_residual_mean":coarse,
        "source_theoretical_fine_residual_mean":fine,
        "source_empirical_coarse_residual_mean":empirical_coarse,
        "source_empirical_fine_residual_mean":empirical_fine,
        "coarse_residual_positive_sources":count_coarse,
        "fine_residual_attenuation_sources":count_att,
        "source_mean_residual_fraction_explained":
            (1-abs(fine)/abs(coarse)) if coarse else None,
        "preregistered_five_gates":checks,
        "registered_primary_pass":all(checks.values()),
        "scope":("Source-public RRC25 and eight tail/magnitude/lag atoms; "
                 "nonzero target changes fixed before simulated independent "
                 "audits. Honest external target witness assumed. Public SHA "
                 "streams reproduce experiments and do not provide secrecy."),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--make-fixture",type=Path)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    if args.make_fixture:
        if args.fixture or args.out:p.error("source-only creation separated")
        raw=canonical(make_fixture())
        args.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        args.make_fixture.write_bytes(raw)
        print("NOR34 FROZEN NEW SOURCE-ONLY FIXTURE",len(raw),digest(raw))
        return
    if not args.fixture:p.error("--fixture required")
    outcome=evaluate(args.fixture)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(canonical(outcome))
    print("NOR34 PRIMARY",outcome["registered_primary_pass"])
    print("NOR34 FIVE GATES",outcome["preregistered_five_gates"])
    print("NOR34 MEAN COARSE / FINE RESIDUAL",round(outcome["source_theoretical_coarse_residual_mean"],8),
          round(outcome["source_theoretical_fine_residual_mean"],8))
    print("NOR34 SAMPLE-BASED CALIBRATION",round(outcome["source_empirical_coarse_residual_mean"],8),
          round(outcome["source_empirical_fine_residual_mean"],8))
    print("NOR34 SOURCE PAIRS",[(r["degree"],r["family"],r["source_theoretical_residual"])
                                 for r in outcome["source_records"]])


if __name__=="__main__":
    main()
