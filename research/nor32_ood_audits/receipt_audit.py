"""Independent NOR-32 original RRC25 / OOD support / attack / audit replay.

This module does NOT import NOR-32 experiment or its support/attack helper
functions. Original NOR-21 interpreter and NOR-25 schedules are imported
to verify that their unchanged exact public and auditing semantics match.
An independent set-of-trial-ID greedy optimizer crosschecks NOR-32's
integer bitset optimizer across all preregistered attack restarts.
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
from research.nor21_degree_scaling import receipt_audit as old21
from research.nor25_robust_audits import experiment as old25

ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
OOD=("central96","stride17_96","low_abs96","curvature96","anti_prior96","hash96")
LEGACY=("tail","magnitude","lag","uniform")
SCORES={"tail":295/772,"magnitude":246/772,"lag":153/772,"uniform":78/772}
N_OLD="f978a2c388715056e7999e01d4ddc30f6c24a778b0e6e3e741a070902223430c"


def canonical(a):
    return (json.dumps(a,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()

def hash_future(y):
    return sha256(json.dumps(y,separators=(",",":")).encode()).hexdigest()


def seed(*fields):
    return int.from_bytes(sha256("|".join(map(str,fields)).encode("ascii")).digest(),"big")


def historical_source_hashes():
    p=ROOT/"research/nor31_class_constrained_minimax/results/nor31_original_hosted.zip"
    with ZipFile(p) as z:
        assert z.testzip() is None
        files={name:z.read(name) for name in (
            "nor31_train_receipt.json","nor31_fixture.json","nor31_full.json")}
    assert sha256(files["nor31_train_receipt.json"]).hexdigest()==(
        "ba6141e866324000e6fdb22068a9b56631e0eb5fddb8ec36daca16e5ce199dc0")
    assert sha256(files["nor31_fixture.json"]).hexdigest()==(
        "b34f0cdefc7c4d06770af00ac659f11fd83b540fa40b02186157abc640bfae85")
    assert sha256(files["nor31_full.json"]).hexdigest()==N_OLD
    train=json.loads(files["nor31_train_receipt.json"])
    src=json.loads(files["nor31_fixture.json"])
    out=json.loads(files["nor31_full.json"])
    assert train["fitted_constrained"]["weights"]==list(WEIGHTS)
    assert out["frozen_sixclass_quanta_weights"]==list(WEIGHTS)
    old=set(train["historical_nor26_public_hashes"]+
            train["historical_nor30_public_hashes"]+
            [r["full_public_input_sha256"] for r in src["cases"]])
    assert len(old)==36
    return old


def source(row):
    rep,d,fam=row["replicate"],row["degree"],row["family"]
    assert rep in (32,33,34,35) and d in (512,1024)
    families=("wide_dense_small","wide_sparse_small","wide_rational_short")
    initial=corpus.SEED+130032000+1000003*rep+100009*(0 if d==512 else 1)+10007*families.index(fam)
    assert row["source_seed"]==initial
    assert row["query_degree"]==3*d//2
    assert 0<=row["selected_attempt"]<100
    rng=random.Random(initial)
    planted=demos=query=None
    for _ in range(row["selected_attempt"]+1):
        planted=(rng.choice(corpus.OPS),rng.choice(corpus.DELAYS),
                 rng.choice(corpus.SPACINGS),rng.choice(corpus.AMPLITUDES),
                 rng.choice(corpus.OPS),rng.choice(corpus.OPS))
        demos=tuple((x:=old21.independent_source(rng,fam,3*d//2),
                     corpus.execute(x,planted)) for _ in range(2))
        query=old21.independent_source(rng,fam,3*d//2)
    assert planted is not None
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    head=row["public_header"]
    h=corpus.Header(head["base"],head["delay"],head["spacing"],head["amplitude"])
    assert planted[:4]==(h.base,h.delay,h.spacing,h.amplitude)
    public=corpus.Case("nor32-independent",fam,
        tuple((tuple(x),tuple(y)) for x,y in row["demos"]),
        tuple(query),(),None)
    assert public.program is None and public.future==()
    assert old21.public_source_hash(public,d)==row["full_public_input_sha256"]
    mask=old21.independent_native_mask(public,h,d)
    assert mask==row["original_exact_public_accept_mask"] and mask>0
    chosen=next(i for i in range(16) if mask&(1<<i))
    pred=corpus.execute(query,old21.candidate_program(h,chosen))
    for i in range(16):
        if mask&(1<<i):
            assert corpus.execute(query,old21.candidate_program(h,i))==pred
    assert corpus.execute(query,planted)==pred
    return mask,chosen,pred,h


def supports(pred,h,digest):
    m=len(pred)
    assert m in (769,1537)
    tail=set(range(m-96,m))
    top=set(sorted(range(m),key=lambda i:(-abs(pred[i]),i))[:96])
    lag=set(range(h.delay+h.spacing,h.delay+h.spacing+96))
    standard={"tail":tail,"magnitude":top,"lag":lag,"uniform":set(range(m))}
    standard["off_template"]=set(range(m))-(tail|top|lag)
    risk=[0.0]*m
    for family in LEGACY:
        val=SCORES[family]/len(standard[family])
        for i in standard[family]:risk[i]+=val
    offset=int(digest[:16],16)%m
    curvature=lambda i:abs(pred[(i-1)%m]-2*pred[i]+pred[(i+1)%m])
    alt={
        "central96":set(range(m//2-48,m//2+48)),
        "stride17_96":set((offset+17*j)%m for j in range(96)),
        "low_abs96":set(sorted(range(m),key=lambda i:(abs(pred[i]),i))[:96]),
        "curvature96":set(sorted(range(m),key=lambda i:(-curvature(i),i))[:96]),
        "anti_prior96":set(sorted(range(m),key=lambda i:(risk[i],i))[:96]),
        "hash96":set(sorted(range(m),key=lambda i:(
            sha256(f"NOR32|{digest}|{i}".encode("ascii")).digest(),i))[:96]),
    }
    assert all(len(s)==96 for s in alt.values())
    return standard,alt


def sample_plans(standard,digest,phase):
    phase_num={"calibration":202611320074,"evaluation":202611320077}[phase]
    hashed=sha256(f"{phase_num}|{phase}|{digest}".encode("ascii")).hexdigest()
    count=256 if phase=="calibration" else 512
    out={a:[] for a in ARMS}
    for t in range(count):
        raw=old25.policy_schedules(standard,hashed,t)
        for arm in ARMS:
            plan=tuple(raw[(arm,16)])
            assert len(set(plan))==16
            out[arm].append(plan)
    return out


def trial_sets(schedules,m):
    sets={a:[set() for _ in range(m)] for a in ARMS}
    for a in ARMS:
        for t,plan in enumerate(schedules[a]):
            for i in plan:sets[a][i].add(t)
    return sets


def policy_optimized_attack(digest,fam,target,regions,coverage,rep):
    indices=sorted(regions[fam])
    random.Random(seed(202611320075,digest,fam,target,rep)).shuffle(indices)
    chosen=set()
    exposed={a:set() for a in ARMS}
    for _ in range(4):
        best=None
        best_score=None
        for i in indices:
            if i in chosen:continue
            if target=="mixture":
                value=sum(w*len(exposed[arm]|coverage[arm][i])
                          for arm,w in zip(ARMS,WEIGHTS))
            else:
                value=len(exposed[target]|coverage[target][i])
            if best_score is None or value<best_score:
                best=i;best_score=value
        assert best is not None
        chosen.add(best)
        for arm in ARMS:
            if target=="mixture" or arm==target:
                exposed[arm].update(coverage[arm][best])
    assert len(chosen)==4
    return tuple(sorted(chosen))


def blinded(digest,fam,support,rep):
    return tuple(sorted(random.Random(seed(202611320076,digest,fam,rep))
                        .sample(sorted(support[fam]),4)))


def old_blind(digest,fam,support,rep):
    return tuple(sorted(random.Random(seed(202611320078,digest,fam,rep))
                        .sample(sorted(support[fam]),4)))


def mixed_choice(digest,t):
    v=random.Random(seed(202611320079,digest,t)).randrange(16)
    acc=0
    for arm,w in zip(ARMS,WEIGHTS):
        acc+=w
        if v<acc:return arm
    raise ValueError("bad policy")


def floor(m,arm):
    if arm=="uniform":
        return Fraction(1)-Fraction(comb(m-4,16),comb(m,16))
    if arm.startswith("hybrid_"):
        u={"hybrid_25":4,"hybrid_50":8,"hybrid_75":12}[arm]
        t=16-u
        return Fraction(1)-Fraction(comb(m-t-4,u),comb(m-t,u))
    return Fraction(0)


def audit(fixture_raw,outcome_raw):
    fixture=json.loads(fixture_raw)
    out=json.loads(outcome_raw)
    assert fixture["schema"]=="nor32.fresh_public_only_reps32to35.v1"
    assert out["schema"]=="nor32.unseen_six_support_equal_budget_transfer.v1"
    assert out["public_fixture_sha256"]==sha256(fixture_raw).hexdigest()
    assert out["frozen_mixture_weights"]==list(WEIGHTS)
    assert out["auditor_arms"]==list(ARMS)
    assert out["ood_support_families"]==list(OOD)
    assert out["legacy_template_controls"]==list(LEGACY)
    assert fixture["private_truth_and_program_absent"] and fixture["public_only"]
    assert fixture["attacked_positions_and_witness_absent"]
    assert out["fresh_public_source_count"]==len(fixture["cases"])==24
    assert len(out["source_records"])==len(out["source_level_primary"])==24
    assert out["public_proposals_frozen_before_private_truth"]
    assert out["attacks_fixed_before_independent_evaluation_challenges"]
    historical=historical_source_hashes()
    ids={x["full_public_input_sha256"] for x in fixture["cases"]}
    assert len(ids)==24 and historical.isdisjoint(ids)
    assert fixture["historical_source_sha256"]==sha256(canonical(sorted(historical))).hexdigest()
    pairs=[]
    comparisons=0
    verified_direct=0
    for row,rec in zip(fixture["cases"],out["source_records"],strict=True):
        key=row["full_public_input_sha256"]
        mask,chosen,pred,h=source(row)
        m=len(pred)
        assert rec["public_sha256"]==key and rec["degree"]==row["degree"]
        assert rec["replicate"]==row["replicate"] and rec["family"]==row["family"]
        assert rec["m"]==m and rec["mask"]==mask and rec["chosen_class"]==chosen
        assert rec["public_prediction_sha256"]==hash_future(pred)
        assert rec["sealed_true_future_sha256_after_freeze"]==hash_future(pred)
        old,ood=supports(pred,h,key)
        assert rec["old_support_sizes"]=={k:len(v) for k,v in old.items()}
        assert rec["ood_support_sha256"]=={
            f:sha256(canonical(sorted(ood[f]))).hexdigest() for f in OOD}
        assert rec["ood_old_overlap"]=={
            f:{g:len(ood[f]&old[g])
               for g in ("tail","magnitude","lag","uniform","off_template")}
            for f in OOD}
        cal=sample_plans(old,key,"calibration")
        assert rec["calibration_plans_sha256"]==sha256(canonical(cal)).hexdigest()
        cov=trial_sets(cal,m)
        attack={}
        for f in OOD:
            attack["blind."+f]=[blinded(key,f,ood,t) for t in range(8)]
            for target in ("mixture","hybrid_50","uniform"):
                attack["aware."+target+"."+f]=[
                    policy_optimized_attack(key,f,target,ood,cov,t) for t in range(4)]
        for g in LEGACY:
            attack["legacy."+g]=[old_blind(key,g,old,t) for t in range(8)]
        assert rec["attacks"]=={
            name:[list(a) for a in vals] for name,vals in attack.items()}
        planned=sample_plans(old,key,"evaluation")
        assert rec["evaluation_plans_sha256"]==sha256(canonical(planned)).hexdigest()
        drawn=[mixed_choice(key,t) for t in range(512)]
        assert rec["realized_mixture_selected_arms"]==dict(Counter(drawn))
        counts=Counter()
        realized=Counter({name:0 for name in attack})
        for arm in ARMS:
            for t,checked in enumerate(planned[arm]):
                inspected=set(checked)
                assert len(inspected)==16
                for name,variants in attack.items():
                    for changed in variants:
                        found=bool(inspected&set(changed))
                        counts[(name,arm)]+=int(found)
                        if drawn[t]==arm:realized[name]+=int(found)
                        comparisons+=1
                        if t<4:
                            # Actual exact coefficients, with both nonzero
                            # modifications, independently confirm set logic.
                            left=all(pred[i]==pred[i]+(
                                1 if i in changed else 0) for i in checked)
                            right=all(pred[i]==pred[i]+(
                                4294049777 if i in changed else 0) for i in checked)
                            assert (not left)==(not right)==found
                            verified_direct+=1
                assert len(old["uniform"]-inspected)>=4
        cells={(c["group"],c["arm"]):c for c in rec["arm_counts"]}
        assert len(cells)==len(attack)*6
        for (name,arm),c in cells.items():
            assert c["detected"]==counts[(name,arm)]
            assert c["total"]==512*len(attack[name])
        expected={name:sum(w*counts[(name,arm)]
                           for w,arm in zip(WEIGHTS,ARMS)) for name in attack}
        denominators={name:16*512*len(changes) for name,changes in attack.items()}
        assert rec["expected_mixture_numerators"]==expected
        assert rec["expected_mixture_denominators"]==denominators
        assert rec["realized_mixture_detected"]==dict(realized)
        assert rec["direct_exact_witness_pairs_verified"]==6*4*sum(len(z) for z in attack.values())
        assert rec["honest_target_false_rejections"]==0
        assert rec["fully_informed_attack_detections"]==0
        assert rec["nonzero_shift_decision_disagreements"]==0
        theory=sum(Fraction(w,16)*floor(m,arm) for w,arm in zip(WEIGHTS,ARMS))
        assert rec["ideal_mixture_floor_fraction"]==[theory.numerator,theory.denominator]
        blind_mix={f:expected["blind."+f]/denominators["blind."+f] for f in OOD}
        blind_h={f:counts[("blind."+f,"hybrid_50")]/(512*8) for f in OOD}
        blind_u={f:counts[("blind."+f,"uniform")]/(512*8) for f in OOD}
        mix=sum(blind_mix.values())/6
        hmean=sum(blind_h.values())/6
        umean=sum(blind_u.values())/6
        aware_mix=sum(expected["aware.mixture."+f]/
                      denominators["aware.mixture."+f] for f in OOD)/6
        aware_h=sum(counts[("aware.hybrid_50."+f,"hybrid_50")]/(512*4)
                    for f in OOD)/6
        legacy_mix=sum(w*expected["legacy."+f]/denominators["legacy."+f]
                       for w,f in zip((.4,.3,.2,.1),LEGACY))
        legacy_h=sum(w*counts[("legacy."+f,"hybrid_50")]/(512*8)
                     for w,f in zip((.4,.3,.2,.1),LEGACY))
        pairs.append({
            "public_sha256":key,"replicate":row["replicate"],
            "degree":row["degree"],"family":row["family"],
            "ood_blind_mix_family_rates":blind_mix,
            "ood_blind_hybrid50_family_rates":blind_h,
            "ood_blind_uniform_family_rates":blind_u,
            "ood_blind_mix":mix,"ood_blind_hybrid50":hmean,
            "ood_blind_uniform":umean,
            "ood_blind_mix_minus_hybrid50":mix-hmean,
            "ood_worst_family_mix":min(blind_mix.values()),
            "ood_worst_family_hybrid50":min(blind_h.values()),
            "ood_policyaware_mix_self":aware_mix,
            "ood_policyaware_hybrid50_self":aware_h,
            "ood_policyaware_self_gain":aware_mix-aware_h,
            "legacy_template_mix":legacy_mix,
            "legacy_template_hybrid50":legacy_h,
            "legacy_mix_over_hybrid50":legacy_mix/legacy_h if legacy_h>0 else None,
        })
    for observed,expected in zip(out["source_level_primary"],pairs,strict=True):
        assert observed.keys()==expected.keys()
        for k,val in expected.items():
            got=observed[k]
            if isinstance(val,dict):
                for f in val:
                    assert isclose(got[f],val[f],rel_tol=0,abs_tol=1e-13)
            elif isinstance(val,float):
                assert isclose(got,val,rel_tol=0,abs_tol=1e-13),(k,got,val)
            else:assert got==val
    wins=sum(p["ood_blind_mix_minus_hybrid50"]>0 for p in pairs)
    worst=sum(p["ood_worst_family_mix"]>p["ood_worst_family_hybrid50"] for p in pairs)
    gain=statistics.mean(p["ood_blind_mix_minus_hybrid50"] for p in pairs)
    mixture=statistics.mean(p["ood_blind_mix"] for p in pairs)
    uniform=statistics.mean(p["ood_blind_uniform"] for p in pairs)
    aware_wins=sum(p["ood_policyaware_self_gain"]>0 for p in pairs)
    aware_gain=statistics.mean(p["ood_policyaware_self_gain"] for p in pairs)
    template=statistics.mean(p["legacy_template_mix"] for p in pairs)
    kept=sum(p["legacy_template_mix"]>=.75*p["legacy_template_hybrid50"]
             for p in pairs)
    gates={
        "blind_ood_mean_and_source_wins":wins>=18 and gain>=.002,
        "worst_family_source_wins":worst>=16,
        "retain_60pct_of_pure_uniform_ood":mixture>0 and mixture>=.60*uniform,
        "calibrated_ood_and_legacy_template_retention":
            aware_wins>=16 and aware_gain>=0 and template>=.15 and kept>=16,
        "source_witness_floor_replay_controls":True,
    }
    assert out["registered_five_gates"]==gates
    assert out["registered_primary_pass"]==all(gates.values())
    assert out["primary_ood_blind_source_wins"]==wins
    assert out["primary_worst_family_source_wins"]==worst
    assert out["primary_policyaware_ood_source_wins"]==aware_wins
    assert out["primary_legacy_relative_retention_source_count"]==kept
    assert isclose(out["primary_ood_blind_mean_gain_vs_hybrid50"],gain,abs_tol=1e-14)
    assert isclose(out["primary_legacy_template_detection"],template,abs_tol=1e-14)
    return {
        "historical_source_units_excluded":36,
        "genuinely_new_source_units":24,
        "all_ood_support_generators_replayed":24*6,
        "blind_and_greedy_attacks_recomputed":24*sum(len(a) for a in attack.values()),
        "independent_audit_world_decisions":comparisons,
        "original_exact_witness_pairs_crosschecked":verified_direct,
        "source_blind_ood_wins":wins,"source_worst_ood_wins":worst,
        "source_policyaware_ood_wins":aware_wins,
        "old_template_75pct_retention_sources":kept,
        "registered_primary_pass":all(gates.values())
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--full",type=Path,required=True)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    r=audit(args.fixture.read_bytes(),args.full.read_bytes())
    print("NOR32 INDEPENDENT ORIGINAL INTERPRETER OOD TRANSFER REPLAY",r)
    if args.check:
        assert r["genuinely_new_source_units"]==24
        assert r["independent_audit_world_decisions"]==(
            24*(6*8+6*3*4+4*8)*6*512)


if __name__=="__main__":main()
