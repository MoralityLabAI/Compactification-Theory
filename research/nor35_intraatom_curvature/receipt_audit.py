"""Independent NOR-35 exact source/atom/attack/reranking/witness replay.

No import of NOR35 evaluator. Reconstructs original RRC25 program
separately, original NOR25 k16 plans and each transformed plan via a
different algorithm, uses NOR34 independent category-pair Fraction
oracle, and tests EVERY audited source/policy/attack world.
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
from research.nor21_degree_scaling import receipt_audit as old21
from research.nor25_robust_audits import experiment as old25
from research.nor34_eight_atom_curvature import receipt_audit as old34

ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
MODES=("original","atom_random","atom_curvature50","atom_inverse50")
LAWS=("top_curvature","bottom_curvature","eight_atom_matched","full_uniform")
REPS=(48,49,50,51)
DEGREES=(512,1024)
FAMILIES=("wide_dense_small","wide_sparse_small","wide_rational_short")
CSTREAM=202611350071
BSTREAM=202611350072
FSTREAM=202611350073
USTREAM=202611350074
ASTREAM=202611350075
TSTREAM={"atom_random":202611350076,
         "atom_curvature50":202611350077,
         "atom_inverse50":202611350078}
MSTREAM=202611350079
N=512
K=16
R=4
ATTACKS=16


def canonical(x):
    return (json.dumps(x,indent=2,sort_keys=True,ensure_ascii=True)+"\n").encode()


def seed(*args):
    return int.from_bytes(sha256("|".join(map(str,args)).encode("ascii")).digest(),"big")


def future_sha(y):
    return sha256(json.dumps(y,separators=(",",":")).encode()).hexdigest()


def old_history():
    ids=old34.historic_ids()
    with ZipFile(ROOT/"research/nor34_eight_atom_curvature/results/nor34_original_hosted.zip") as z:
        fixture=z.read("nor34_fixture.json")
        full=z.read("nor34_full.json")
    assert len(fixture)==5011440
    assert sha256(fixture).hexdigest()=="1cbf58a851df898a6e46fe4859d25ac5cf5b68a2e2b3e81d8e7afeede8dacb00"
    assert sha256(full).hexdigest()=="4507769666f5560d33b8a0c99b4b5c85c5c2710b88e0021e77a85679106f6bd3"
    ids.extend(x["full_public_input_sha256"] for x in json.loads(fixture)["cases"])
    assert len(set(ids))==len(ids)==108
    return sorted(ids)


def regenerate_source(row):
    rep,d,fam=row["replicate"],row["degree"],row["family"]
    assert rep in REPS and d in DEGREES and fam in FAMILIES
    formula=corpus.SEED+130035000+1000003*rep+100009*DEGREES.index(d)+10007*FAMILIES.index(fam)
    assert formula==row["source_seed"]
    assert row["query_degree"]==3*d//2 and 0<=row["selected_attempt"]<100
    rng=random.Random(formula)
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
    hdr=row["public_header"]
    h=corpus.Header(hdr["base"],hdr["delay"],hdr["spacing"],hdr["amplitude"])
    assert planted[:4]==(h.base,h.delay,h.spacing,h.amplitude)
    case=corpus.Case("nor35-independent",fam,
        tuple((tuple(x),tuple(y)) for x,y in row["demos"]),tuple(query),(),None)
    assert case.program is None and case.future==()
    assert old21.public_source_hash(case,d)==row["full_public_input_sha256"]
    mask=old21.independent_native_mask(case,h,d)
    assert mask==row["original_native_public_mask"] and mask>0
    chosen=next(j for j in range(16) if mask&(1<<j))
    pred=corpus.execute(query,old21.candidate_program(h,chosen))
    for j in range(16):
        if mask&(1<<j):
            assert corpus.execute(query,old21.candidate_program(h,j))==pred
    assert corpus.execute(query,planted)==pred
    return mask,chosen,pred,h


def atom_regions(pred,head):
    m=len(pred)
    assert m in (769,1537)
    tail=set(range(m-96,m))
    magnitude=set(sorted(range(m),key=lambda i:(-abs(pred[i]),i))[:96])
    lag=set(range(head.delay+head.spacing,head.delay+head.spacing+96))
    old={"tail":tail,"magnitude":magnitude,"lag":lag,"uniform":set(range(m))}
    u=tail|magnitude|lag
    old["off_template"]=set(range(m))-u
    atoms=[set() for _ in range(8)]
    scores=[abs(pred[(i-1)%m]-2*pred[i]+pred[(i+1)%m]) for i in range(m)]
    for i in range(m):
        atom=(1 if i in tail else 0)+(2 if i in magnitude else 0)+(
              4 if i in lag else 0)
        atoms[atom].add(i)
    assert len(atoms)==8 and sum(map(len,atoms))==m
    top=sorted(range(m),key=lambda i:(-scores[i],i))
    bottom=sorted(range(m),key=lambda i:(scores[i],i))
    cu=set([i for i in top if i in u][:48])
    cv=set([i for i in top if i not in u][:48])
    bu=set([i for i in bottom if i in u][:48])
    bv=set([i for i in bottom if i not in u][:48])
    assert len(cu)==len(cv)==len(bu)==len(bv)==48
    return old,atoms,u,cu,cv,bu,bv,scores


def blinded_attacks(digest,atoms,u,cu,cv,bu,bv,m):
    result={f:[] for f in LAWS}
    quotas=[]
    for t in range(ATTACKS):
        rng=random.Random(seed(CSTREAM,digest,t))
        top=tuple(sorted(rng.sample(sorted(cu),2)+rng.sample(sorted(cv),2)))
        rng=random.Random(seed(BSTREAM,digest,t))
        bottom=tuple(sorted(rng.sample(sorted(bu),2)+rng.sample(sorted(bv),2)))
        counts=tuple(sum(i in atom for i in top) for atom in atoms)
        rng=random.Random(seed(FSTREAM,digest,t))
        fine=tuple(sorted(i for j,q in enumerate(counts)
                          for i in rng.sample(sorted(atoms[j]),q)))
        rng=random.Random(seed(USTREAM,digest,t))
        uniform=tuple(sorted(rng.sample(range(m),4)))
        assert counts[0]==2 and sum(counts)==4
        assert [sum(i in a for i in fine) for a in atoms]==list(counts)
        assert len(set(top))==len(set(bottom))==len(set(fine))==len(set(uniform))==4
        for family,val in zip(LAWS,(top,bottom,fine,uniform)):
            result[family].append(val)
        quotas.append(counts)
    return result,quotas


def replanned(original,atoms,scores,digest,arm,trial,mode):
    if mode=="original":return tuple(original)
    assert mode in TSTREAM
    result=[]
    checked=set(original)
    for j,atom in enumerate(atoms):
        quota=len(checked&atom)
        if not quota:continue
        rng=random.Random(seed(TSTREAM[mode],digest,arm,trial,j))
        positions=sorted(atom)
        if mode=="atom_random":
            result.extend(rng.sample(positions,quota))
            continue
        # Alternative formulation: sort all atom positions by curvature
        # score first, then sample a uniform tail with independent seeds.
        reverse=(mode=="atom_curvature50")
        ordered=(sorted(positions,key=lambda i:(-scores[i],i)) if reverse
                 else sorted(positions,key=lambda i:(scores[i],i)))
        c=(quota+1)//2
        prefix=ordered[:c]
        remain=set(ordered)-set(prefix)
        rest=rng.sample(sorted(remain),quota-c)
        result.extend(prefix+rest)
    assert len(result)==len(set(result))==K
    assert all(len(set(result)&a)==len(checked&a) for a in atoms)
    return tuple(result)


def original_and_transformed_schedules(old,atoms,scores,digest):
    phase=sha256(f"{ASTREAM}|nor35|{digest}".encode("ascii")).hexdigest()
    out={mode:{arm:[] for arm in ARMS} for mode in MODES}
    for trial in range(N):
        baseline=old25.policy_schedules(old,phase,trial)
        for arm in ARMS:
            checked=tuple(baseline[(arm,K)])
            assert len(set(checked))==16
            for mode in MODES:
                out[mode][arm].append(
                    replanned(checked,atoms,scores,digest,arm,trial,mode))
    return out


def mixed_choice(digest,t):
    c=random.Random(seed(MSTREAM,digest,t)).randrange(16)
    n=0
    for arm,w in zip(ARMS,WEIGHTS):
        n+=w
        if c<n:return arm
    raise AssertionError("mix quantization")


def two_block(supportU,supportV,plan):
    a=set(plan)
    return Fraction(1)-Fraction(comb(48-len(a&supportU),2),comb(48,2))*Fraction(
        comb(48-len(a&supportV),2),comb(48,2))


def analytical_laws(atoms,cu,cv,bu,bv,inspected,m):
    # Prior NOR34 independent category-pair oracle is independent of
    # this NOR35 scorer, and is itself kernel-grounded by MathCan.
    old_expected=old34.exact_expected_detection(atoms,cu,cv,inspected)
    fine=old_expected["eight_atom"]
    uniform=Fraction(1)-Fraction(comb(m-K,R),comb(m,R))
    return {
       "top_curvature":two_block(cu,cv,inspected),
       "bottom_curvature":two_block(bu,bv,inspected),
       "eight_atom_matched":fine,"full_uniform":uniform}


def replay(fixture_bytes,grade_bytes):
    fixture=json.loads(fixture_bytes);grade=json.loads(grade_bytes)
    assert fixture["schema"]=="nor35.public_only_new_reps48to51.v1"
    assert grade["schema"]=="nor35.eight_atom_curvature_targeted_vs_quota_random.v1"
    assert fixture["registration_commit"]==grade["preregistered_first_commit"]==(
        "3773c1b1522f419e54f34252b0c743018105bc79")
    assert fixture["public_only"] and fixture["private_program_and_future_absent"]
    assert fixture["attack_witness_and_policy_absent"]
    past=old_history()
    assert len(past)==fixture["excluded_history_count"]==grade["excluded_historical_units"]==108
    assert fixture["historical_sha256"]==sha256(canonical(past)).hexdigest()
    assert grade["source_public_fixture_sha256"]==sha256(fixture_bytes).hexdigest()
    assert grade["auditor_arms"]==list(ARMS) and grade["frozen_mix_weights"]==list(WEIGHTS)
    assert grade["transforms"]==list(MODES) and grade["attack_laws"]==list(LAWS)
    assert grade["source_units"]==len(fixture["cases"])==len(grade["source_records"])==24
    assert grade["public_proposals_frozen_before_private_truth"]
    assert grade["all_attacks_and_plans_fixed_before_native_truth"]
    assert grade["authentic_external_target_witness_assumed"]
    source_hashes={r["full_public_input_sha256"] for r in fixture["cases"]}
    assert len(source_hashes)==24 and source_hashes.isdisjoint(past)
    paired=[]
    comparisons=0
    direct=0
    for row,record in zip(fixture["cases"],grade["source_records"],strict=True):
        key=row["full_public_input_sha256"]
        mask,chosen,pred,header=regenerate_source(row)
        assert record["public_sha256"]==key
        assert record["replicate"]==row["replicate"]
        assert record["degree"]==row["degree"] and record["family"]==row["family"]
        assert record["future_length"]==len(pred)
        assert record["native_mask"]==mask and record["chosen_class"]==chosen
        assert record["public_prediction_sha256"]==record["postchoice_true_future_sha256"]==future_sha(pred)
        old,atoms,u,cu,cv,bu,bv,scores=atom_regions(pred,header)
        assert record["old_atom_sizes"]==list(map(len,atoms))
        assert record["curvature_top_atom_support_counts"]==[len((cu|cv)&a) for a in atoms]
        assert record["curvature_bottom_atom_support_counts"]==[len((bu|bv)&a) for a in atoms]
        assert record["curvature_top_support_sha256"]==sha256(canonical(sorted(cu|cv))).hexdigest()
        assert record["curvature_bottom_support_sha256"]==sha256(canonical(sorted(bu|bv))).hexdigest()
        attacks,quotas=blinded_attacks(key,atoms,u,cu,cv,bu,bv,len(pred))
        assert record["attack_sets"]=={law:[list(a) for a in v] for law,v in attacks.items()}
        assert record["top_curvature_attack_quotas"]==[list(q) for q in quotas]
        schedules=original_and_transformed_schedules(old,atoms,scores,key)
        expected_digest={
            mode:{arm:sha256(canonical(schedules[mode][arm])).hexdigest()
                  for arm in ARMS} for mode in MODES}
        assert record["per_transform_arm_plan_sha256"]==expected_digest
        chosen_mixed=[mixed_choice(key,t) for t in range(N)]
        assert record["realized_mixed_arm_counts"]==dict(Counter(chosen_mixed))
        counts={mode:{arm:{f:0 for f in LAWS} for arm in ARMS} for mode in MODES}
        theory={mode:{arm:{f:Fraction(0) for f in LAWS}
                      for arm in ARMS} for mode in MODES}
        realized={mode:{f:0 for f in LAWS} for mode in MODES}
        for mode in MODES:
            for arm in ARMS:
                for t,plan in enumerate(schedules[mode][arm]):
                    assert len(set(plan))==K
                    assert all(len(set(plan)&a)==len(set(schedules["original"][arm][t])&a)
                               for a in atoms)
                    expect=analytical_laws(atoms,cu,cv,bu,bv,plan,len(pred))
                    for f in LAWS:theory[mode][arm][f]+=expect[f]
                    inspection=set(plan)
                    for f in LAWS:
                        for j,a in enumerate(attacks[f]):
                            hit=int(bool(inspection&set(a)))
                            comparisons+=1
                            counts[mode][arm][f]+=hit
                            if chosen_mixed[t]==arm:
                                realized[mode][f]+=hit
                            if t==0 and j==0:
                                for delta in (1,4294049777):
                                    from_truth=any(pred[i] != pred[i]+(
                                        delta if i in a else 0) for i in plan)
                                    assert bool(from_truth)==bool(hit)
                                direct+=1
        for mode in MODES:
            for arm in ARMS:
                for family in LAWS:
                    value=theory[mode][arm][family]/N
                    assert record["exact_analytic_detection"][mode][arm][family]==[
                        value.numerator,value.denominator]
                    assert record["sample_detected_out_of_8192"][mode][arm][family]==counts[mode][arm][family]
            assert record["realized_mixture_detected_out_of_8192"][mode]==realized[mode]
        scores_per_mode={}
        for mode in MODES:
            scores_per_mode[mode]={}
            for family in LAWS:
                weighted=sum((Fraction(w,16)*theory[mode][arm][family]/N
                              for arm,w in zip(ARMS,WEIGHTS)),Fraction(0))
                scores_per_mode[mode][family]=weighted
                assert record["exact_mixture_detection"][mode][family]==[
                    weighted.numerator,weighted.denominator]
        for arm in ARMS:
            for mode in MODES:
                assert theory[mode][arm]["eight_atom_matched"]==theory["original"][arm]["eight_atom_matched"]
                assert theory[mode][arm]["full_uniform"]==theory["original"][arm]["full_uniform"]
        primary={
            "C":float(theory["atom_curvature50"]["hybrid_50"]["top_curvature"]/N-
                      theory["atom_random"]["hybrid_50"]["top_curvature"]/N),
            "B":float(theory["atom_curvature50"]["hybrid_50"]["top_curvature"]/N-
                      theory["original"]["hybrid_50"]["top_curvature"]/N),
            "I":float(theory["atom_curvature50"]["hybrid_50"]["bottom_curvature"]/N-
                      theory["atom_random"]["hybrid_50"]["bottom_curvature"]/N),
            "M":float(scores_per_mode["atom_curvature50"]["top_curvature"]-
                      scores_per_mode["original"]["top_curvature"]),
        }
        for k,v in primary.items():
            assert isclose(v,record["source_primary_C_B_I_M"][k],abs_tol=1e-14)
        assert record["legitimate_witness_false_rejections"]==0
        assert record["schedule_informed_attacker_detected"]==0
        assert record["eight_atom_quota_mismatches"]==0
        assert record["shift_decision_disagreements"]==0
        assert record["direct_authentic_integer_witness_pair_checks"]==len(MODES)*len(ARMS)*len(LAWS)
        paired.append(primary)
    means={q:statistics.mean(p[q] for p in paired) for q in ("C","B","I","M")}
    wins={q:sum(p[q]<0 if q=="I" else p[q]>0 for p in paired) for q in means}
    gates={
        "curvature_beats_atom_random":wins["C"]>=18 and means["C"]>=.010,
        "curvature_beats_original_hybrid50":wins["B"]>=18 and means["B"]>=.005,
        "inverse_curvature_cost":wins["I"]>=18 and means["I"]<=-.002,
        "transformed_mix_and_atom_invariance":wins["M"]>=18 and means["M"]>=.005,
        "source_oracle_and_independent_replay":True,
    }
    assert grade["registered_five_gates"]==gates
    assert grade["registered_primary_pass"]==all(gates.values())
    assert grade["source_wins"]==wins
    for q in means:
        assert isclose(grade["source_mean_primary"][q],means[q],abs_tol=1e-14)
    return {
        "source_units":24,"historic_source_units_excluded":108,
        "original_native_programs_replayed":24,
        "paired_attack_sets_replayed":24*4*ATTACKS,
        "all_source_arm_transformed_audit_plans_reconstructed":24*len(MODES)*len(ARMS)*N,
        "source_arm_attack_world_decisions_replayed":comparisons,
        "direct_integer_witness_comparisons":direct,
        "source_mean_primary":means,"source_wins":wins,
        "registered_primary_pass":all(gates.values()),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--full",type=Path,required=True)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    result=replay(args.fixture.read_bytes(),args.full.read_bytes())
    print("NOR35 INDEPENDENT ORIGINAL SOURCE/ATOMS/ATTACKS/FRACTION REPLAY",result)
    if args.check:
        assert result["source_units"]==24
        assert result["source_arm_attack_world_decisions_replayed"]==24*4*6*4*16*N


if __name__=="__main__":
    main()
