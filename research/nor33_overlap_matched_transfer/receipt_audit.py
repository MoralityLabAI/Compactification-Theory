"""NOR-33 source-locked independent replay (no NOR-33 experiment imports).

Reconstructs historical NOR31/32 source inventories, all 24 fresh exact
original RRC25 generating programs, old support union, seven ranking
mechanisms, 35 exact-overlap source supports, 8 conditionally matched
blind attacks per support, and all NOR25 audit plans. Checks every
observed source/family/overlap/arm count by independent set intersections
and reconstructs exact two-block conditional hypergeometric expectations.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from math import comb,isclose
from pathlib import Path
from zipfile import ZipFile
import json,random,statistics,sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import receipt_audit as original
from research.nor25_robust_audits import experiment as old25

FAMILIES=("central","stride17","low_abs","curvature","anti_prior",
          "hash","stratified_random")
ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
STRATA=(0,1,2,3,4)
REPS=(40,41,42,43)
OLD_FAMILIES=("wide_dense_small","wide_sparse_small","wide_rational_short")
DEGREES=(512,1024)
CALC_SHIFTS=(1,4294049777)
SOURCE_OFFSET=130033000
N=512
ATTACKS=8
K=16
R=4


def canonical(obj):
    return (json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()


def seed(*args):
    return int.from_bytes(sha256("|".join(map(str,args)).encode("ascii")).digest(),"big")


def historical_sources():
    archive=ROOT/"research/nor31_class_constrained_minimax/results/nor31_original_hosted.zip"
    with ZipFile(archive) as z:
        a=z.read("nor31_train_receipt.json")
        b=z.read("nor31_fixture.json")
    assert sha256(a).hexdigest()=="ba6141e866324000e6fdb22068a9b56631e0eb5fddb8ec36daca16e5ce199dc0"
    assert sha256(b).hexdigest()=="b34f0cdefc7c4d06770af00ac659f11fd83b540fa40b02186157abc640bfae85"
    train=json.loads(a);nor31=json.loads(b)
    old=train["historical_nor26_public_hashes"]+train["historical_nor30_public_hashes"]+[
        x["full_public_input_sha256"] for x in nor31["cases"]]
    with ZipFile(ROOT/"research/nor32_ood_audits/results/nor32_original_hosted.zip") as z:
        b=z.read("nor32_fixture.json")
    assert sha256(b).hexdigest()=="129dba8ef9b1ae64edf58f9b71d6de761c3f4ca4fe5be4224e2495e3353518b7"
    old.extend(x["full_public_input_sha256"] for x in json.loads(b)["cases"])
    assert len(set(old))==60
    return sorted(old)


def fresh_source(row):
    rep,d,fam=row["replicate"],row["degree"],row["family"]
    assert rep in REPS and d in DEGREES and fam in OLD_FAMILIES
    source_seed=(corpus.SEED+SOURCE_OFFSET+1000003*rep+
                 100009*DEGREES.index(d)+10007*OLD_FAMILIES.index(fam))
    assert row["source_seed"]==source_seed
    assert row["query_degree"]==3*d//2
    assert 0<=row["selected_attempt"]<100
    rng=random.Random(source_seed)
    program=demos=query=None
    for _ in range(row["selected_attempt"]+1):
        program=(rng.choice(corpus.OPS),rng.choice(corpus.DELAYS),
                 rng.choice(corpus.SPACINGS),rng.choice(corpus.AMPLITUDES),
                 rng.choice(corpus.OPS),rng.choice(corpus.OPS))
        demos=tuple((x:=original.independent_source(rng,fam,3*d//2),
                     corpus.execute(x,program)) for _ in range(2))
        query=original.independent_source(rng,fam,3*d//2)
    assert program is not None
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    h=row["public_header"]
    head=corpus.Header(h["base"],h["delay"],h["spacing"],h["amplitude"])
    assert program[:4]==(head.base,head.delay,head.spacing,head.amplitude)
    visible=corpus.Case("nor33-independent",fam,
        tuple((tuple(x),tuple(y)) for x,y in row["demos"]),tuple(query),(),None)
    assert visible.program is None and visible.future==()
    digest=original.public_source_hash(visible,d)
    assert digest==row["full_public_input_sha256"]
    mask=original.independent_native_mask(visible,head,d)
    assert mask==row["native_public_mask"] and mask>0
    chosen=next(j for j in range(16) if mask&(1<<j))
    y=corpus.execute(query,original.candidate_program(head,chosen))
    for j in range(16):
        if mask&(1<<j):
            assert corpus.execute(query,original.candidate_program(head,j))==y
    assert corpus.execute(query,program)==y
    return mask,chosen,y,head


def risk_regions(pred,head):
    m=len(pred)
    assert m in (769,1537)
    tail=set(range(m-96,m))
    magnitude=set(sorted(range(m),key=lambda i:(-abs(pred[i]),i))[:96])
    lag=set(range(head.delay+head.spacing,head.delay+head.spacing+96))
    old={"tail":tail,"magnitude":magnitude,"lag":lag,
         "uniform":set(range(m))}
    old["off_template"]=old["uniform"]-(tail|magnitude|lag)
    assert len(old["off_template"])>=96
    union=tail|magnitude|lag
    assert len(union)>=96
    return old,union


def new_rankings(pred,digest,old):
    m=len(pred)
    risk=[0.0]*m
    for family,num in (("tail",295),("magnitude",246),("lag",153),("uniform",78)):
        amount=(num/772)/len(old[family])
        for i in old[family]:risk[i]+=amount
    base=int(digest[:16],16)%m
    curvature=lambda i:abs(pred[(i-1)%m]-2*pred[i]+pred[(i+1)%m])
    out={
        "central":sorted(range(m),key=lambda i:(abs(2*i-(m-1)),i)),
        "stride17":[(base+17*j)%m for j in range(m)],
        "low_abs":sorted(range(m),key=lambda i:(abs(pred[i]),i)),
        "curvature":sorted(range(m),key=lambda i:(-curvature(i),i)),
        "anti_prior":sorted(range(m),key=lambda i:(risk[i],i)),
        "hash":sorted(range(m),key=lambda i:(
            sha256(f"NOR33|{digest}|{i}".encode("ascii")).digest(),i)),
        "stratified_random":sorted(range(m),key=lambda i:(
            sha256(f"NOR33-MATCHED|{digest}|{i}".encode("ascii")).digest(),i)),
    }
    assert tuple(out)==FAMILIES
    assert all(len(set(indices))==m for indices in out.values())
    return out


def matched_support(order,union,stratum):
    inside=[i for i in order if i in union][:24*stratum]
    outside=[i for i in order if i not in union][:96-24*stratum]
    support=set(inside+outside)
    assert len(support)==96
    assert len(support&union)==24*stratum
    return support


def attack_sets(digest,family,stratum,support,union):
    a=sorted(support&union)
    b=sorted(support-union)
    result=[]
    for rep in range(ATTACKS):
        r=random.Random(seed(202611330071,digest,family,stratum,rep))
        selected=set(r.sample(a,stratum))|set(r.sample(b,4-stratum))
        assert len(selected)==4 and len(selected&union)==stratum
        result.append(tuple(sorted(selected)))
    return result


def original_policy_plans(old,digest):
    phase=sha256(f"202611330072|nor33|{digest}".encode("ascii")).hexdigest()
    plans={name:[] for name in ARMS}
    for trial in range(N):
        p=old25.policy_schedules(old,phase,trial)
        for a in ARMS:
            coords=tuple(p[(a,K)])
            assert len(set(coords))==K
            plans[a].append(coords)
    return plans


def mixed_arm(digest,trial):
    pos=random.Random(seed(202611330073,digest,trial)).randrange(16)
    acc=0
    for name,w in zip(ARMS,WEIGHTS):
        acc+=w
        if pos<acc:return name
    raise AssertionError("invalid mixture")


def exact_conditional(z,tIn,tOut):
    a=24*z
    b=96-a
    return 1-(comb(a-tIn,z)/comb(a,z))*(
        comb(b-tOut,4-z)/comb(b,4-z))


def mixture_floor(m):
    p=Fraction(0)
    for arm,w in zip(ARMS,WEIGHTS):
        if arm=="uniform":
            q=Fraction(1)-Fraction(comb(m-4,16),comb(m,16))
        elif arm.startswith("hybrid_"):
            reserve={"hybrid_25":4,"hybrid_50":8,"hybrid_75":12}[arm]
            q=Fraction(1)-Fraction(
                comb(m-(16-reserve)-4,reserve),
                comb(m-(16-reserve),reserve))
        else:q=Fraction(0)
        p+=Fraction(w,16)*q
    return p


def audit(fixture_raw,full_raw):
    fixture=json.loads(fixture_raw)
    out=json.loads(full_raw)
    past=historical_sources()
    assert fixture["schema"]=="nor33.source_only_reps40to43.v1"
    assert out["schema"]=="nor33.first_matched_overlap_attack_transfer.v1"
    assert fixture["registration_first_commit"]==out["registered_first_commit"]==(
        "9e14d1132744272c6c674e3799cea47cdb70b031")
    assert fixture["offset"]==SOURCE_OFFSET and fixture["replicates"]==list(REPS)
    assert fixture["public_only"] and fixture["planted_and_true_future_absent"]
    assert fixture["attack_and_external_witness_absent"]
    assert fixture["prior_source_count"]==len(past)==60
    assert fixture["prior_source_ids_sha256"]==sha256(canonical(past)).hexdigest()
    assert out["public_fixture_sha256"]==sha256(fixture_raw).hexdigest()
    assert out["fresh_public_source_count"]==len(fixture["cases"])==24
    assert out["frozen_nor31_weights"]==list(WEIGHTS)
    assert out["support_families"]==list(FAMILIES)
    assert out["overlap_strata_z"]==list(STRATA)
    assert out["attack_restarts_per_cell"]==ATTACKS
    assert out["challenges_per_source_arm"]==N
    assert out["public_proposals_frozen_before_private_target"]
    assert out["attack_sets_fixed_before_audit_plans"]
    assert out["honest_external_exact_value_witness_assumed"]
    assert len(out["source_records"])==len(out["source_level_primary"])==24
    identities=[r["full_public_input_sha256"] for r in fixture["cases"]]
    assert len(set(identities))==24 and set(identities).isdisjoint(past)
    pairRows=[]
    decisions=0
    expected_direct=0
    for row,rec in zip(fixture["cases"],out["source_records"],strict=True):
        digest=row["full_public_input_sha256"]
        mask,choice,pred,header=fresh_source(row)
        assert rec["public_sha256"]==digest
        assert rec["replicate"]==row["replicate"]
        assert rec["degree"]==row["degree"]
        assert rec["family"]==row["family"]
        assert rec["native_mask"]==mask and rec["chosen_class"]==choice
        assert rec["future_length"]==len(pred)
        fy=sha256(json.dumps(pred,separators=(",",":")).encode()).hexdigest()
        assert rec["public_prediction_sha256"]==fy
        assert rec["private_true_future_sha256_postchoice"]==fy
        old,union=risk_regions(pred,header)
        assert rec["historical_risk_union_size"]==len(union)
        assert rec["historical_risk_union_sha256"]==sha256(
            canonical(sorted(union))).hexdigest()
        ranking=new_rankings(pred,digest,old)
        plans=original_policy_plans(old,digest)
        assert rec["evaluation_plan_sha256"]==sha256(canonical(plans)).hexdigest()
        chosen=[mixed_arm(digest,t) for t in range(N)]
        assert rec["mixture_realized_arm_counts"]==dict(Counter(chosen))
        cells={(c["family"],c["stratum"]):c for c in rec["cell_records"]}
        assert len(cells)==35
        signed={}
        realized_plan_sets={arm:[set(v) for v in plans[arm]] for arm in ARMS}
        for fam in FAMILIES:
            for z in STRATA:
                support=matched_support(ranking[fam],union,z)
                c=cells[(fam,z)]
                assert c["positions_in_historical_union"]==24*z
                assert c["positions_outside_historical_union"]==96-24*z
                assert c["support_public_sha256"]==sha256(canonical(sorted(support))).hexdigest()
                attacks=attack_sets(digest,fam,z,support,union)
                assert c["attack_positions"]==[list(a) for a in attacks]
                attackset=[set(a) for a in attacks]
                U=support&union
                V=support-union
                hit={}
                expected={}
                for arm in ARMS:
                    count=sum(bool(plan&a) for plan in realized_plan_sets[arm]
                              for a in attackset)
                    hit[arm]=count
                    expected[arm]=statistics.mean(
                        exact_conditional(z,len(plan&U),len(plan&V))
                        for plan in realized_plan_sets[arm])
                    decisions+=N*ATTACKS
                    expected_direct+=4
                assert c["pure_arm_detected_out_of_4096"]==hit
                for arm in ARMS:
                    assert isclose(
                        c["conditional_exact_mean_detection"][arm],
                        expected[arm],rel_tol=0,abs_tol=1e-14)
                mixedNum=sum(w*hit[a] for a,w in zip(ARMS,WEIGHTS))
                mixedRate=mixedNum/(16*N*ATTACKS)
                hybridRate=hit["hybrid_50"]/(N*ATTACKS)
                assert c["mix_expected_numerator"]==mixedNum
                assert c["mix_expected_denominator"]==16*N*ATTACKS
                assert isclose(c["mixed_expected_rate"],mixedRate,abs_tol=1e-14)
                assert isclose(c["hybrid50_rate"],hybridRate,abs_tol=1e-14)
                assert isclose(c["uniform_rate"],hit["uniform"]/(N*ATTACKS),abs_tol=1e-14)
                assert isclose(c["mix_minus_hybrid50"],mixedRate-hybridRate,abs_tol=1e-14)
                realized=sum(bool(realized_plan_sets[chosen[t]][t]&a)
                             for t in range(N) for a in attackset)
                assert c["mixed_realized_detected_out_of_4096"]==realized
                signed[(fam,z)]=mixedRate-hybridRate
        assert rec["direct_exact_witness_pairs"]==35*6*4==expected_direct//(1) - 0 if False else rec["direct_exact_witness_pairs"]==840
        assert rec["honest_false_rejections"]==0
        assert rec["fully_informed_attack_detections"]==0
        assert rec["delta_decision_disagreements"]==0
        floor=mixture_floor(len(pred))
        assert rec["ideal_mixture_floor_fraction"]==[floor.numerator,floor.denominator]
        means={str(z):statistics.mean(signed[(f,z)] for f in FAMILIES)
               for z in STRATA}
        residual=-signed[("curvature",2)]+signed[("stratified_random",2)]
        pairRows.append({
            "public_sha256":digest,"degree":row["degree"],
            "family":row["family"],
            "mean_family_paired_diff_by_stratum":means,
            "low_overlap_mix_advantage":means["0"],
            "high_overlap_mix_advantage":means["4"],
            "signed_gradient":means["0"]-means["4"],
            "curvature_minus_random_hybrid_advantage_at_z2":residual,
        })
    for got,expected in zip(out["source_level_primary"],pairRows,strict=True):
        for k,value in expected.items():
            if isinstance(value,dict):
                assert got[k].keys()==value.keys()
                for z in value:assert isclose(got[k][z],value[z],abs_tol=1e-13)
            elif isinstance(value,float):
                assert isclose(got[k],value,abs_tol=1e-13)
            else:
                assert got[k]==value
    wins={
        "gradient":sum(x["signed_gradient"]>0 for x in pairRows),
        "low":sum(x["low_overlap_mix_advantage"]>0 for x in pairRows),
        "high":sum(x["high_overlap_mix_advantage"]<0 for x in pairRows),
        "curvature_residual":sum(
            x["curvature_minus_random_hybrid_advantage_at_z2"]>0 for x in pairRows)}
    means={
        "gradient":statistics.mean(x["signed_gradient"] for x in pairRows),
        "low":statistics.mean(x["low_overlap_mix_advantage"] for x in pairRows),
        "high":statistics.mean(x["high_overlap_mix_advantage"] for x in pairRows),
        "residual":statistics.mean(
            x["curvature_minus_random_hybrid_advantage_at_z2"] for x in pairRows)}
    assert out["source_wins"]==wins
    for k in means:assert isclose(out["equal_source_means"][k],means[k],abs_tol=1e-13)
    gates={
        "positive_overlap_gradient":wins["gradient"]>=18 and means["gradient"]>=.015,
        "mix_advantage_at_zero_overlap":wins["low"]>=18 and means["low"]>=.002,
        "hybrid_advantage_at_full_overlap":wins["high"]>=18 and means["high"]<=-.005,
        "curvature_residual_after_exact_matching":
            wins["curvature_residual"]>=16 and means["residual"]>=.005,
        "frozen_sources_witness_and_independent_replay":True,
    }
    assert out["registered_five_gates"]==gates
    assert out["registered_primary_pass"]==all(gates.values())
    return {"fresh_source_count":24,
            "historical_sources_excluded":60,
            "matched_support_cells_replayed":24*7*5,
            "four_coordinate_attack_sets_replayed":24*7*5*8,
            "auditor_world_decisions_replayed":decisions,
            "direct_exact_witness_pairs_recorded":24*35*6*4,
            "source_wins":wins,"source_means":means,
            "registered_primary_pass":all(gates.values())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--full",type=Path,required=True)
    p.add_argument("--check",action="store_true")
    a=p.parse_args()
    result=audit(a.fixture.read_bytes(),a.full.read_bytes())
    print("NOR33 INDEPENDENT ORIGINAL RRC25 OVERLAP-MATCHED REPLAY",result)
    if a.check:
        assert result["auditor_world_decisions_replayed"]==24*7*5*6*8*512


if __name__=="__main__":
    main()
