"""NOR-34 independent exact source, audit, eight-atom and attack replay.

This replayer imports ONLY original NOR21 and NOR25 established semantics,
and prior historical archived fixtures. No NOR34 evaluator, source,
attack, probabilistic helper or scoring module is imported.

The applied scorer uses integer bitsets and an optimized source-supported
pair formula. This auditor uses sets and a separate exact Fraction
enumeration of all unordered category-pair quota configurations.
"""
from __future__ import annotations
import argparse,json,random,statistics,sys
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
from math import comb,isclose
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import receipt_audit as original
from research.nor25_robust_audits import experiment as old25

ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
LAWS=("curvature","coarse","eight_atom")
REPS=(44,45,46,47)
DEGREES=(512,1024)
FAMILIES=("wide_dense_small","wide_sparse_small","wide_rational_short")
N=512
ATTACKS=16


def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()


def seed(*parts):
    return int.from_bytes(sha256("|".join(map(str,parts)).encode("ascii")).digest(),"big")


def historic_ids():
    p=ROOT/"research/nor31_class_constrained_minimax/results/nor31_original_hosted.zip"
    with ZipFile(p) as z:
        a=z.read("nor31_train_receipt.json")
        b=z.read("nor31_fixture.json")
    assert sha256(a).hexdigest()=="ba6141e866324000e6fdb22068a9b56631e0eb5fddb8ec36daca16e5ce199dc0"
    assert sha256(b).hexdigest()=="b34f0cdefc7c4d06770af00ac659f11fd83b540fa40b02186157abc640bfae85"
    tr=json.loads(a);f31=json.loads(b)
    ids=tr["historical_nor26_public_hashes"]+tr["historical_nor30_public_hashes"]+[
        r["full_public_input_sha256"] for r in f31["cases"]]
    p32=ROOT/"research/nor32_ood_audits/results/nor32_original_hosted.zip"
    with ZipFile(p32) as z:
        f32=z.read("nor32_fixture.json")
    assert sha256(f32).hexdigest()=="129dba8ef9b1ae64edf58f9b71d6de761c3f4ca4fe5be4224e2495e3353518b7"
    ids.extend(x["full_public_input_sha256"] for x in json.loads(f32)["cases"])
    p33=ROOT/"research/nor33_overlap_matched_transfer/results/nor33_original_hosted.zip"
    with ZipFile(p33) as z:
        f33=z.read("nor33_fixture.json")
        o33=z.read("nor33_full.json")
    assert len(f33)==4547200 and sha256(f33).hexdigest()=="16f3b397aea17adbd1462666142a13e933c8da67a1286e1e6418d32b22090db8"
    assert sha256(o33).hexdigest()=="88d37a957f7d3be42b76213b351df921a08556654a90a2ed21df1b66788d6a62"
    ids.extend(x["full_public_input_sha256"] for x in json.loads(f33)["cases"])
    assert len(ids)==len(set(ids))==84
    return sorted(ids)


def independently_regenerate(row):
    rep,d,fam=row["replicate"],row["degree"],row["family"]
    assert rep in REPS and d in DEGREES and fam in FAMILIES
    formula=corpus.SEED+130034000+1000003*rep+100009*DEGREES.index(d)+10007*FAMILIES.index(fam)
    assert formula==row["source_seed"] and row["query_degree"]==3*d//2
    assert 0<=row["selected_attempt"]<100
    rng=random.Random(formula)
    program=demos=query=None
    for _ in range(row["selected_attempt"]+1):
        program=(rng.choice(corpus.OPS),rng.choice(corpus.DELAYS),
                 rng.choice(corpus.SPACINGS),rng.choice(corpus.AMPLITUDES),
                 rng.choice(corpus.OPS),rng.choice(corpus.OPS))
        demos=tuple((src:=original.independent_source(rng,fam,3*d//2),
                     corpus.execute(src,program)) for _ in range(2))
        query=original.independent_source(rng,fam,3*d//2)
    assert program is not None
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    hr=row["public_header"]
    h=corpus.Header(hr["base"],hr["delay"],hr["spacing"],hr["amplitude"])
    assert program[:4]==(h.base,h.delay,h.spacing,h.amplitude)
    pub=corpus.Case("nor34-independent-public",fam,
        tuple((tuple(x),tuple(y)) for x,y in row["demos"]),tuple(query),(),None)
    assert pub.program is None and pub.future==()
    assert original.public_source_hash(pub,d)==row["full_public_input_sha256"]
    mask=original.independent_native_mask(pub,h,d)
    assert mask==row["native_public_mask"] and mask>0
    j=next(i for i in range(16) if mask&(1<<i))
    predicted=corpus.execute(query,original.candidate_program(h,j))
    for n in range(16):
        if mask&(1<<n):
            assert corpus.execute(query,original.candidate_program(h,n))==predicted
    assert corpus.execute(query,program)==predicted
    return mask,j,predicted,h


def source_risk_atoms(pred,h):
    m=len(pred)
    tail=set(range(m-96,m))
    magnitude=set(sorted(range(m),key=lambda i:(-abs(pred[i]),i))[:96])
    lag=set(range(h.delay+h.spacing,h.delay+h.spacing+96))
    old={"tail":tail,"magnitude":magnitude,"lag":lag,"uniform":set(range(m))}
    old["off_template"]=set(range(m))-(tail|magnitude|lag)
    atoms=[set() for _ in range(8)]
    for i in range(m):
        pattern=int(i in tail)+2*int(i in magnitude)+4*int(i in lag)
        atoms[pattern].add(i)
    assert sum(len(a) for a in atoms)==m
    union=set(range(m))-atoms[0]
    assert union==tail|magnitude|lag
    rank=sorted(range(m),key=lambda i:(
        -abs(pred[(i-1)%m]-2*pred[i]+pred[(i+1)%m]),i))
    cu=set(x for x in rank if x in union)
    cv=set(x for x in rank if x not in union)
    curvU=set([x for x in rank if x in union][:48])
    curvV=set([x for x in rank if x not in union][:48])
    assert len(curvU)==len(curvV)==48 and not curvU&curvV
    return old,atoms,union,curvU,curvV


def independent_attacks(digest,atoms,union,curvU,curvV):
    out=[]
    for t in range(16):
        rng=random.Random(seed(202611340071,digest,t))
        attack=tuple(sorted(rng.sample(sorted(curvU),2)+rng.sample(sorted(curvV),2)))
        quota=[sum(i in a for i in attack) for a in atoms]
        assert quota[0]==2 and sum(quota)==4
        fine_rng=random.Random(seed(202611340072,digest,t))
        matched=tuple(sorted(i for j,q in enumerate(quota)
                             for i in fine_rng.sample(sorted(atoms[j]),q)))
        coarse_rng=random.Random(seed(202611340073,digest,t))
        coarse=tuple(sorted(coarse_rng.sample(sorted(union),2)+
                            coarse_rng.sample(sorted(atoms[0]),2)))
        assert len(set(attack))==len(set(matched))==len(set(coarse))==4
        assert [sum(i in atom for i in matched) for atom in atoms]==quota
        assert sum(i in union for i in coarse)==2
        out.append({"curvature":attack,"eight_atom":matched,
                    "coarse":coarse,"quotas":quota})
    return out


def fixed_schedules(old,digest):
    phase=sha256(f"202611340074|nor34|{digest}".encode("ascii")).hexdigest()
    out={a:[] for a in ARMS}
    for t in range(N):
        p=old25.policy_schedules(old,phase,t)
        for a in ARMS:
            c=tuple(p[(a,16)])
            assert len(set(c))==16
            out[a].append(c)
    return out


def selected_mixed_policy(digest,t):
    r=random.Random(seed(202611340075,digest,t)).randrange(16)
    n=0
    for a,w in zip(ARMS,WEIGHTS):
        n+=w
        if r<n:return a
    raise AssertionError("invalid original mixture")


def misses_of_category_pair(j,k,atoms,counts):
    """A source atom pair is redrawn uniformly among all atom positions."""
    if j==k:
        n=len(atoms[j]);t=counts[j]
        return Fraction(comb(n-t,2),comb(n,2))
    return Fraction(len(atoms[j])-counts[j],len(atoms[j]))*Fraction(
        len(atoms[k])-counts[k],len(atoms[k]))


def exact_expected_detection(atoms,curvU,curvV,check):
    """Separate categorical pair enumeration from applied cached helper."""
    checked=set(check)
    m=sum(map(len,atoms))
    union=set().union(*atoms[1:])
    ns=[len(a) for a in atoms]
    tc=[len(checked&a) for a in atoms]
    tu=len(checked&curvU)
    tv=len(checked&curvV)
    cu=Fraction(comb(48-tu,2),comb(48,2))
    cv=Fraction(comb(48-tv,2),comb(48,2))
    miss_curv=cu*cv
    miss_coarse=(Fraction(comb(len(union)-len(checked&union),2),comb(len(union),2))*
                 Fraction(comb(ns[0]-tc[0],2),comb(ns[0],2)))
    supp=[len(curvU&a) for a in atoms]
    category_pair_probability=Fraction(0)
    for j in range(1,8):
        if supp[j]>=2:
            category_pair_probability+=(
                Fraction(comb(supp[j],2),comb(48,2))*
                misses_of_category_pair(j,j,atoms,tc))
        for k in range(j+1,8):
            if supp[j] and supp[k]:
                category_pair_probability+=(
                    Fraction(supp[j]*supp[k],comb(48,2))*
                    misses_of_category_pair(j,k,atoms,tc))
    miss_fine=category_pair_probability*Fraction(
        comb(ns[0]-tc[0],2),comb(ns[0],2))
    return {"curvature":Fraction(1)-miss_curv,
            "coarse":Fraction(1)-miss_coarse,
            "eight_atom":Fraction(1)-miss_fine}


def floor_for_arm(m,a):
    if a=="uniform":
        return Fraction(1)-Fraction(comb(m-4,16),comb(m,16))
    if a.startswith("hybrid_"):
        u={"hybrid_25":4,"hybrid_50":8,"hybrid_75":12}[a]
        t=16-u
        return Fraction(1)-Fraction(comb(m-t-4,u),comb(m-t,u))
    return Fraction(0)


def audit(fixture_raw,result_raw):
    fixture=json.loads(fixture_raw)
    result=json.loads(result_raw)
    history=historic_ids()
    assert fixture["schema"]=="nor34.new_public_only_reps44to47.v1"
    assert result["schema"]=="nor34.sourceheldout_eight_atom_curvature_residual.v1"
    assert fixture["registration_commit"]==result["preregistered_first_commit"]==(
        "ef6d7b4d0374d0310a640258d94eaf474fb66c4b")
    assert fixture["public_only"] and fixture["private_future_and_program_absent"]
    assert fixture["attacks_and_witness_absent"]
    assert fixture["historical_source_count"]==len(history)==84
    assert fixture["historical_public_digests_sha256"]==sha256(canonical(history)).hexdigest()
    assert result["source_public_fixture_sha256"]==sha256(fixture_raw).hexdigest()
    assert result["new_public_source_count"]==24 and result["excluded_historic_sources"]==84
    assert result["auditor_arms"]==list(ARMS)
    assert result["frozen_mix_weights"]==list(WEIGHTS)
    assert result["paired_attack_laws"]==["curvature","coarse","eight_atom"]
    assert result["public_proposals_frozen_before_private_truth"]
    assert result["all_attack_sets_fixed_before_audit_schedule"]
    assert result["honest_external_target_coefficient_witness_assumed"]
    assert len(fixture["cases"])==len(result["source_records"])==24
    ids={r["full_public_input_sha256"] for r in fixture["cases"]}
    assert len(ids)==24 and ids.isdisjoint(history)
    coarseResiduals=[];fineResiduals=[]
    sampledCoarse=[];sampledFine=[]
    fullDecisions=0
    directPairs=0
    for row,record in zip(fixture["cases"],result["source_records"],strict=True):
        key=row["full_public_input_sha256"]
        mask,j,pred,header=independently_regenerate(row)
        assert record["public_sha256"]==key
        assert record["replicate"]==row["replicate"]
        assert record["degree"]==row["degree"]
        assert record["family"]==row["family"]
        assert record["native_public_mask"]==mask and record["chosen_class"]==j
        digest=sha256(json.dumps(pred,separators=(",",":")).encode()).hexdigest()
        assert record["public_prediction_sha256"]==digest
        assert record["private_native_true_future_sha256_postchoice"]==digest
        assert record["future_length"]==len(pred)
        old,atoms,union,cu,cv=source_risk_atoms(pred,header)
        assert record["pattern_population"]==[len(a) for a in atoms]
        assert record["curvature_support_pattern_counts"]==[
            len((cu|cv)&a) for a in atoms]
        assert record["curvature_support_U_size"]==record["curvature_support_V_size"]==48
        assert record["curvature_support_sha256"]==sha256(
            canonical(sorted(cu|cv))).hexdigest()
        attacks=independent_attacks(key,atoms,union,cu,cv)
        assert record["paired_attack_records"]==[{
            "curvature":list(a["curvature"]),"eight_atom":list(a["eight_atom"]),
            "coarse":list(a["coarse"]),"eight_atom_quota":a["quotas"]}
            for a in attacks]
        plans=fixed_schedules(old,key)
        assert record["frozen_policy_schedule_sha256"]==sha256(canonical(plans)).hexdigest()
        selections=[selected_mixed_policy(key,t) for t in range(N)]
        assert record["realized_mixed_arm_counts"]==dict(Counter(selections))
        armsTheory={law:{arm:Fraction(0) for arm in ARMS}
                    for law in ("curvature","coarse","eight_atom")}
        counts={law:{arm:0 for arm in ARMS}
                for law in ("curvature","coarse","eight_atom")}
        realized=Counter()
        for arm in ARMS:
            for t,positions in enumerate(plans[arm]):
                predProb=exact_expected_detection(atoms,cu,cv,positions)
                for law,p in predProb.items():
                    armsTheory[law][arm]+=p
                checkset=set(positions)
                for law in ("curvature","coarse","eight_atom"):
                    for a in attacks:
                        found=bool(checkset&set(a[law]))
                        counts[law][arm]+=int(found)
                        if selections[t]==arm:realized[law]+=int(found)
                        fullDecisions+=1
                        if t<4:
                            directPairs+=1
                            # Two exact integer truth shifts are
                            # pointwise equivalent to intersection.
                            for delta in (1,4294049777):
                                foundViaTruth=any(
                                    pred[i] != pred[i]+(delta if i in a[law] else 0)
                                    for i in positions)
                                assert foundViaTruth==found
        for law in armsTheory:
            for arm in ARMS:
                value=armsTheory[law][arm]/N
                assert record["theoretical_conditional_detection_exact"][law][arm]==[
                    value.numerator,value.denominator]
                assert isclose(record["theoretical_conditional_detection_float"][law][arm],
                               float(value),rel_tol=0,abs_tol=1e-14)
        assert record["empirical_detected_per_arm"]==counts
        assert record["empirical_total_per_arm"]==ATTACKS*N
        assert record["mixed_realized_detected"]==dict(realized)
        H={};empH={}
        for law in armsTheory:
            mix=sum((Fraction(w,16)*armsTheory[law][arm]/N
                     for arm,w in zip(ARMS,WEIGHTS)),Fraction(0))
            theoryH=armsTheory[law]["hybrid_50"]/N-mix
            H[law]=float(theoryH)
            empiricalMix=sum(w*counts[law][arm] for arm,w in zip(ARMS,WEIGHTS))/(
                16*ATTACKS*N)
            empH[law]=counts[law]["hybrid_50"]/(ATTACKS*N)-empiricalMix
            assert isclose(record["source_expected_mixture_detection"][law],float(mix),abs_tol=1e-14)
            assert isclose(record["source_expected_hybrid_advantage"][law],H[law],abs_tol=1e-14)
            assert isclose(record["source_empirical_mixture_detection"][law],empiricalMix,abs_tol=1e-14)
            assert isclose(record["source_empirical_hybrid_advantage"][law],empH[law],abs_tol=1e-14)
        c=H["curvature"]-H["coarse"]
        f=H["curvature"]-H["eight_atom"]
        ec=empH["curvature"]-empH["coarse"]
        ef=empH["curvature"]-empH["eight_atom"]
        assert isclose(record["source_theoretical_residual"]["coarse"],c,abs_tol=1e-14)
        assert isclose(record["source_theoretical_residual"]["fine"],f,abs_tol=1e-14)
        assert isclose(record["source_empirical_residual"]["coarse"],ec,abs_tol=1e-14)
        assert isclose(record["source_empirical_residual"]["fine"],ef,abs_tol=1e-14)
        assert record["honest_false_rejections"]==0
        assert record["schedule_informed_attacker_detections"]==0
        assert record["shift_detection_disagreements"]==0
        assert record["direct_exact_integer_witness_checks"]==1152
        fmin=sum((Fraction(w,16)*floor_for_arm(len(pred),a)
                  for a,w in zip(ARMS,WEIGHTS)),Fraction(0))
        assert record["ideal_mixture_floor_fraction"]==[fmin.numerator,fmin.denominator]
        coarseResiduals.append(c);fineResiduals.append(f)
        sampledCoarse.append(ec);sampledFine.append(ef)
    meanCoarse=statistics.mean(coarseResiduals)
    meanFine=statistics.mean(fineResiduals)
    empiricalCoarse=statistics.mean(sampledCoarse)
    empiricalFine=statistics.mean(sampledFine)
    coarseWins=sum(x>0 for x in coarseResiduals)
    fineWins=sum(abs(f)<abs(c) for f,c in zip(fineResiduals,coarseResiduals))
    gates={
        "coarse_residual_replicates":coarseWins>=18 and meanCoarse>=.015,
        "fine_atom_matching_attenuates":fineWins>=18 and meanCoarse!=0
            and abs(meanFine)<=.30*abs(meanCoarse),
        "fine_residual_at_most_half_percent":abs(meanFine)<=.005,
        "paired_samples_calibrate_exact_laws":
            abs(empiricalCoarse-meanCoarse)<=.010 and
            abs(empiricalFine-meanFine)<=.010,
        "full_original_source_witness_replay_controls":True,
    }
    assert result["preregistered_five_gates"]==gates
    assert result["registered_primary_pass"]==all(gates.values())
    assert result["coarse_residual_positive_sources"]==coarseWins
    assert result["fine_residual_attenuation_sources"]==fineWins
    for field,value in (
        ("source_theoretical_coarse_residual_mean",meanCoarse),
        ("source_theoretical_fine_residual_mean",meanFine),
        ("source_empirical_coarse_residual_mean",empiricalCoarse),
        ("source_empirical_fine_residual_mean",empiricalFine)):
        assert isclose(result[field],value,rel_tol=0,abs_tol=1e-14)
    return {"new_source_count":24,
            "historical_sources_excluded":84,
            "paired_attack_triplets_replayed":24*ATTACKS,
            "eight_atom_quota_vectors_replayed":24*ATTACKS,
            "full_audit_world_decisions_replayed":fullDecisions,
            "authentic_integer_witness_pair_checks":directPairs,
            "mean_coarse_residual":meanCoarse,
            "mean_fine_residual":meanFine,
            "source_coarse_wins":coarseWins,
            "source_fine_attenuation_wins":fineWins,
            "registered_primary_pass":all(gates.values())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--full",type=Path,required=True)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    result=audit(args.fixture.read_bytes(),args.full.read_bytes())
    print("NOR34 INDEPENDENT SOURCE/ATTACK/EXACT EIGHT-ATOM REPLAY",result)
    if args.check:
        assert result["full_audit_world_decisions_replayed"]==24*3*ATTACKS*6*N
        assert result["new_source_count"]==24


if __name__=="__main__":
    main()
