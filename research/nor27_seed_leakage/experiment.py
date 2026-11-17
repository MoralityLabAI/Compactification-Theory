"""NOR-27 prospective audit-output and seed-prefix leakage experiments.

FROZEN REGISTRATION before source data: bcc49dfb96acd9a9aae51925630b2f04f69e2315.
Python 3.12 standard library. Deterministic salted seed streams are a
scientific replay device, not certified production cryptographic entropy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from math import comb
from pathlib import Path
import json
import random
import statistics
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.nor24_adaptive_audits import experiment as n24
from research.nor25_robust_audits import experiment as n25
from research.nor26_policy_aware_audits import experiment as n26
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus

SCHEMA="nor27.seed_prefix_and_reserve_preview_leakage.v1"
FIXTURE_SCHEMA="nor27.fresh_public_only_12_sources.v1"
REPS=(8,9)
OFFSET=119327000
DEGREES=(512,1024)
FAMILIES=nor21.FAMILIES
PRIOR_NUMERATORS=(295,246,153,78)
PRIOR_DENOMINATOR=772
TARGETED=8
RESERVE=8
BUDGET=16
RADIUS=4
LEAK_REVEAL=(0,2,4,6,8)
PREVIEW_TRIALS=1024
SEED_LEAK_BITS=(0,2,4)
CAL_PER_BUCKET=128
EVAL_PER_BUCKET=256
ATTACK_REPLICATES=4
PREVIEW_EVAL_STREAM=202611270081
PREVIEW_ATTACK_STREAM=202611270082
CAL_STREAM=202611270083
EVAL_STREAM=202611270084
ATTACK_STREAM=202611270085
SEED_AUDITORS=("hybrid_50","uniform16")
DELTAS=(1,4294049777)
assert n25.PRIOR_NUMERATORS==PRIOR_NUMERATORS
assert n25.PRIOR_DENOMINATOR==PRIOR_DENOMINATOR


def hash_seed(*parts):
    return int.from_bytes(sha256("|".join(map(str,parts)).encode("ascii")).digest(),"big")


def source_seed(rep, degree, family):
    if rep not in REPS or degree not in DEGREES or family not in FAMILIES:
        raise ValueError("source outside frozen NOR27 prospective grid")
    return (corpus.SEED+OFFSET+1000003*rep+
            100009*DEGREES.index(degree)+10007*FAMILIES.index(family))


def new_source(rep,degree,family,seen):
    q=3*degree//2
    rng=random.Random(source_seed(rep,degree,family))
    for attempt in range(100):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,family,q),
                     corpus.execute(x,planted)) for _ in range(2))
        query=nor21.public_source(rng,family,q)
        case=corpus.Case("nor27-public-only",family,demos,query,(),None)
        headers=corpus.possible_headers(case)
        if len(headers)!=1:
            continue
        header=headers[0]
        if (header.base,header.delay,header.spacing,header.amplitude)!=planted[:4]:
            continue
        digest=nor21.public_digest(case,degree)
        if digest in seen:
            continue
        mask=nor21.native_mask(case,header,degree)
        if not mask:
            continue
        predictions=nor21.candidate_predictions(query,header,q)
        if len({predictions[i] for i in range(16) if mask&(1<<i)})!=1:
            continue
        planted_j=corpus.OPS.index(planted[4])*4+corpus.OPS.index(planted[5])
        assert mask&(1<<planted_j)
        seen.add(digest)
        return {
            "split":"TEST","replicate":rep,
            "observation_degree":degree,"query_degree":q,"family":family,
            "source_seed":source_seed(rep,degree,family),
            "selected_attempt":attempt,"full_public_input_sha256":digest,
            "public_header":{"base":header.base,"delay":header.delay,
                             "spacing":header.spacing,"amplitude":header.amplitude},
            "demos":[[list(x[:degree+1]),list(y[:degree+1])] for x,y in demos],
            "public_query_input":list(query),
            "original_public_accept_mask":mask,
        }
    raise RuntimeError("NOR27 failed source-admissibility search")


def make_fixture():
    seen=set()
    rows=[new_source(rep,d,f,seen)
          for rep in REPS for d in DEGREES for f in FAMILIES]
    assert len(rows)==len(seen)==12
    return {
        "schema":FIXTURE_SCHEMA,"offset":OFFSET,"replicates":list(REPS),
        "source_only":True,"hidden_truth_absent":True,
        "audit_seeds_absent":True,"leakage_and_attack_absent":True,
        "preview_trials":PREVIEW_TRIALS,
        "seed_buckets":[1<<b for b in SEED_LEAK_BITS],
        "calibration_per_bucket":CAL_PER_BUCKET,
        "evaluation_per_bucket":EVAL_PER_BUCKET,
        "cases":rows,
    }


def read_fixture(path):
    raw=Path(path).read_bytes()
    f=json.loads(raw)
    assert f["schema"]==FIXTURE_SCHEMA and f["offset"]==OFFSET
    assert f["replicates"]==list(REPS)
    assert f["source_only"] and f["hidden_truth_absent"]
    assert f["audit_seeds_absent"] and f["leakage_and_attack_absent"]
    assert f["preview_trials"]==PREVIEW_TRIALS
    assert f["seed_buckets"]==[1<<b for b in SEED_LEAK_BITS]
    assert f["calibration_per_bucket"]==CAL_PER_BUCKET
    assert f["evaluation_per_bucket"]==EVAL_PER_BUCKET
    assert len(f["cases"])==12
    assert {(x["replicate"],x["observation_degree"],x["family"]) for x in f["cases"]}=={
        (rep,d,fam) for rep in REPS for d in DEGREES for fam in FAMILIES}
    assert len({x["full_public_input_sha256"] for x in f["cases"]})==12
    assert all(x["split"]=="TEST" and not any(k in x for k in
               ("program","future","attack","witness","audit_seed")) for x in f["cases"])
    return f,sha256(raw).hexdigest()


def restore_private_after_public_choice(row,predicted):
    rep,d,family=row["replicate"],row["observation_degree"],row["family"]
    rng=random.Random(source_seed(rep,d,family))
    for _ in range(row["selected_attempt"]+1):
        prog=nor21.new_header_program(rng)
        demos=tuple((x:=nor21.public_source(rng,family,3*d//2),
                     corpus.execute(x,prog)) for _ in range(2))
        query=nor21.public_source(rng,family,3*d//2)
    assert row["source_seed"]==source_seed(rep,d,family)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    target=corpus.execute(query,prog)
    assert target==predicted
    return target


def targeted_prefix(regions,digest):
    order=n24.static_schedule(regions,n25.PRIOR,digest,0)
    t=tuple(order[:TARGETED])
    assert len(set(t))==TARGETED
    return t


def draw_schedule(digest,seed64,m,policy,target):
    """Hash-expanded sampler: ideal randomness is an assumption, not a proof."""
    if not (0<=seed64<1<<64):
        raise ValueError("64-bit secret seed required")
    if policy not in SEED_AUDITORS:
        raise ValueError("unknown registered auditor")
    private_rng_seed=hash_seed("NOR27|audit",digest,policy,seed64)
    rng=random.Random(private_rng_seed)
    if policy=="hybrid_50":
        risk=set(target)
        remaining=[i for i in range(m) if i not in risk]
        return target+tuple(rng.sample(remaining,RESERVE))
    return tuple(rng.sample(range(m),BUDGET))


def preview_attacker(digest,leaked,target,m,ell):
    """The attacker receives disclosed probes and public data only."""
    if ell not in LEAK_REVEAL or len(leaked)!=ell:
        raise ValueError("leaked length outside NOR27 registration")
    seen=set(target)|set(leaked)
    assert len(seen)==TARGETED+ell
    options=[i for i in range(m) if i not in seen]
    rng=random.Random(hash_seed(PREVIEW_ATTACK_STREAM,digest,ell,*leaked))
    changed=tuple(sorted(rng.sample(options,RADIUS)))
    assert len(changed)==RADIUS and set(changed).isdisjoint(seen)
    return changed


def preview_floor(m,ell):
    if m not in (769,1537) or ell not in LEAK_REVEAL:
        raise ValueError("unregistered source length or leak level")
    N=m-TARGETED-ell
    v=RESERVE-ell
    return Fraction(1)-Fraction(comb(N-RADIUS,v),comb(N,v))


def conditional_seed(digest,b,bucket,phase,ordinal):
    """Explicit 64-bit seed with known high b bits and hidden lower bits."""
    if b not in SEED_LEAK_BITS or not 0<=bucket<1<<b:
        raise ValueError("unregistered high-bit leakage")
    if phase=="cal":
        stream=CAL_STREAM
    elif phase=="eval":
        stream=EVAL_STREAM
    else:
        raise ValueError("unregistered phase")
    lo=64-b
    unknown=hash_seed(stream,digest,b,bucket,ordinal)&((1<<lo)-1)
    S=(bucket<<lo)|unknown
    assert S>>(64-b)==bucket
    return S


def exposure_masks(schedules,m):
    masks=[0]*m
    for j,plan in enumerate(schedules):
        for i in plan:
            masks[i]|=1<<j
    return masks


def greedy_calibration_attacker(exposure,digest,policy,b,bucket,rep):
    """Approximate exposure-minimizing best response using calibration alone."""
    if (policy not in SEED_AUDITORS or b not in SEED_LEAK_BITS
            or not 0<=bucket<1<<b or not 0<=rep<ATTACK_REPLICATES):
        raise ValueError("unregistered attack selection")
    universe=list(range(len(exposure)))
    random.Random(hash_seed(ATTACK_STREAM,digest,policy,b,bucket,rep)).shuffle(universe)
    selected=set()
    current=0
    for _ in range(RADIUS):
        best=None
        score=None
        for i in universe:
            if i in selected:
                continue
            new=(current|exposure[i]).bit_count()
            if score is None or new<score:
                best,score=i,new
        assert best is not None
        selected.add(best)
        current|=exposure[best]
    return tuple(sorted(selected)),current.bit_count()


def exact_detect(proposed,target,checked,altered,delta):
    return not n24.exact_witness_accepts(
        proposed,target,checked,frozenset(altered),delta)


def evaluate(path):
    fixture,fixture_hash=read_fixture(path)
    proposals={row["full_public_input_sha256"]:n24.public_proposal(row)
               for row in fixture["cases"]}
    assert len(proposals)==12
    records=[]
    for row in fixture["cases"]:
        digest=row["full_public_input_sha256"]
        proposal=proposals[digest]
        predicted=proposal["predicted"]
        m=len(predicted)
        regions=n24.supports(predicted,proposal["header"])
        T=targeted_prefix(regions,digest)
        # Game A: all hidden reserve schedules are drawn before any private reconstruction.
        all_preview=[]
        for trial in range(PREVIEW_TRIALS):
            s=hash_seed(PREVIEW_EVAL_STREAM,digest,trial)&((1<<64)-1)
            checked=draw_schedule(digest,s,m,"hybrid_50",T)
            assert checked[:TARGETED]==T
            reserve=checked[TARGETED:]
            choices={ell:preview_attacker(digest,reserve[:ell],T,m,ell)
                     for ell in LEAK_REVEAL}
            all_preview.append((checked,choices))
        # Game B: fit each b-bit bucket solely to its independent calibration seeds.
        buckets=[]
        for b in SEED_LEAK_BITS:
            for label in range(1<<b):
                cal_seeds=[conditional_seed(digest,b,label,"cal",j)
                           for j in range(CAL_PER_BUCKET)]
                eval_seeds=[conditional_seed(digest,b,label,"eval",j)
                            for j in range(EVAL_PER_BUCKET)]
                assert len(set(cal_seeds))==CAL_PER_BUCKET
                assert len(set(eval_seeds))==EVAL_PER_BUCKET
                assert set(cal_seeds).isdisjoint(eval_seeds)
                policies={}
                for arm in SEED_AUDITORS:
                    cal_plans=[draw_schedule(digest,s,m,arm,T) for s in cal_seeds]
                    exposure=exposure_masks(cal_plans,m)
                    attacks=[greedy_calibration_attacker(
                        exposure,digest,arm,b,label,j)
                        for j in range(ATTACK_REPLICATES)]
                    assert all(len(a)==RADIUS for a,_ in attacks)
                    eval_plans=[draw_schedule(digest,s,m,arm,T) for s in eval_seeds]
                    policies[arm]=(attacks,eval_plans)
                buckets.append((b,label,policies))
        target=restore_private_after_public_choice(row,predicted)
        preview_counts={ell:0 for ell in LEAK_REVEAL}
        preview_informed_escape=0
        legitimate_failures=0
        delta_disagreements=0
        preview_receipt=sha256()
        for checked,choices in all_preview:
            if not n24.exact_witness_accepts(predicted,target,checked,frozenset(),1):
                legitimate_failures+=1
            for ell in LEAK_REVEAL:
                attack=choices[ell]
                d1=exact_detect(predicted,target,checked,attack,DELTAS[0])
                d2=exact_detect(predicted,target,checked,attack,DELTAS[1])
                delta_disagreements+=d1!=d2
                preview_counts[ell]+=int(d1)
                if ell==RESERVE:
                    preview_informed_escape+=int(d1)
                preview_receipt.update(
                    (str((ell,attack,d1))+"\n").encode("ascii"))
        assert preview_informed_escape==0
        seed_records=[]
        for b,label,policies in buckets:
            for arm in SEED_AUDITORS:
                attacks,eval_plans=policies[arm]
                hits=[0]*ATTACK_REPLICATES
                for plan in eval_plans:
                    if not n24.exact_witness_accepts(predicted,target,plan,frozenset(),1):
                        legitimate_failures+=1
                    for j,(attack,_) in enumerate(attacks):
                        d1=exact_detect(predicted,target,plan,attack,DELTAS[0])
                        d2=exact_detect(predicted,target,plan,attack,DELTAS[1])
                        delta_disagreements+=d1!=d2
                        hits[j]+=int(d1)
                # Full seed disclosure gives the complete schedule and zero detection.
                for plan in eval_plans:
                    informed=n24.informed_attack(plan,regions["uniform"],RADIUS)
                    assert not set(plan)&informed
                seed_records.append({
                    "bits":b,"bucket":label,"policy":arm,
                    "attacks":[list(a) for a,_ in attacks],
                    "calibration_hit_counts":[hit for _,hit in attacks],
                    "distinct_attacks":len({a for a,_ in attacks}),
                    "evaluation_hits_by_attacker":hits,
                    "detected":sum(hits),"total":ATTACK_REPLICATES*EVAL_PER_BUCKET,
                })
        assert legitimate_failures==delta_disagreements==0
        assert preview_counts[RESERVE]==0
        preview=[{
            "revealed_reserve_coordinates":ell,
            "detected":preview_counts[ell],"total":PREVIEW_TRIALS,
            "theoretical_floor_fraction":[preview_floor(m,ell).numerator,
                                           preview_floor(m,ell).denominator],
            "rate":preview_counts[ell]/PREVIEW_TRIALS,
        } for ell in LEAK_REVEAL]
        seed_summary=[]
        for b in SEED_LEAK_BITS:
            for arm in SEED_AUDITORS:
                group=[x for x in seed_records if x["bits"]==b and x["policy"]==arm]
                assert len(group)==1<<b
                count=sum(x["detected"] for x in group)
                denom=sum(x["total"] for x in group)
                seed_summary.append({
                    "bits":b,"policy":arm,"detected":count,
                    "total":denom,"rate":count/denom,
                })
        records.append({
            "public_sha256":digest,"replicate":row["replicate"],
            "degree":row["observation_degree"],"family":row["family"],
            "m":m,"public_mask":proposal["mask"],
            "public_prediction_sha256":proposal["sha256"],
            "sealed_target_sha256_after_public_choice":n24.sha_target(target),
            "targeted_public_prefix":list(T),
            "preview_decision_sha256":preview_receipt.hexdigest(),
            "preview":preview,
            "seed_prefix_buckets":seed_records,"seed_summary":seed_summary,
            "honest_target_failures":legitimate_failures,
            "delta_disagreements":delta_disagreements,
            "fully_informed_preview_detections":preview_informed_escape,
        })
    means={ell:statistics.mean(
        next(x["rate"] for x in rec["preview"]
             if x["revealed_reserve_coordinates"]==ell)
        for rec in records) for ell in LEAK_REVEAL}
    preview_diff=means[0]-means[4]
    theoretical_by_degree={}
    empirical_by_degree={}
    for m in (769,1537):
        group=[x for x in records if x["m"]==m]
        assert len(group)==6
        theoretical_by_degree[str(m)]={str(ell):float(preview_floor(m,ell))
                                       for ell in LEAK_REVEAL}
        empirical_by_degree[str(m)]={
            str(ell):statistics.mean(
                next(x["rate"] for x in r["preview"] if
                     x["revealed_reserve_coordinates"]==ell)
                for r in group) for ell in LEAK_REVEAL}
    preview_theory_ok=all(
        abs(empirical_by_degree[str(m)][str(ell)]-
            theoretical_by_degree[str(m)][str(ell)])<=0.015
        for m in (769,1537) for ell in LEAK_REVEAL)
    preview_gate=(preview_diff>=.008 and means[4]>means[8] and
                  all(next(x["detected"] for x in r["preview"]
                           if x["revealed_reserve_coordinates"]==8)==0
                      for r in records))
    seed_mean={}
    for b in SEED_LEAK_BITS:
        for arm in SEED_AUDITORS:
            seed_mean[f"{arm}|{b}"]=statistics.mean(
                next(x["rate"] for x in rec["seed_summary"]
                     if x["bits"]==b and x["policy"]==arm)
                for rec in records)
    seed_gate=all(
        abs(seed_mean[f"{arm}|{b}"]-seed_mean[f"{arm}|0"])<=.015
        for b in (2,4) for arm in SEED_AUDITORS)
    controls_ok=all(
        r["honest_target_failures"]==r["delta_disagreements"]==
        r["fully_informed_preview_detections"]==0 for r in records)
    passed=preview_gate and preview_theory_ok and seed_gate and controls_ok
    return {
        "schema":SCHEMA,"fixture_sha256":fixture_hash,
        "source_count":12,
        "frozen_training_prior_numerators":list(PRIOR_NUMERATORS),
        "frozen_training_prior_denominator":PRIOR_DENOMINATOR,
        "public_targeted_size":TARGETED,"uniform_reserve_size":RESERVE,
        "altered_radius":RADIUS,"preview_trials":PREVIEW_TRIALS,
        "calibration_per_seed_bucket":CAL_PER_BUCKET,
        "evaluation_per_seed_bucket":EVAL_PER_BUCKET,
        "source_records":records,
        "preview_source_means":{str(k):v for k,v in means.items()},
        "preview_zero_minus_four":preview_diff,
        "preview_theoretical_by_degree":theoretical_by_degree,
        "preview_empirical_by_degree":empirical_by_degree,
        "seed_prefix_source_means":seed_mean,
        "primary_preview_degradation_pass":preview_gate,
        "primary_preview_theory_agreement_pass":preview_theory_ok,
        "primary_seed_prefix_sensitivity_pass":seed_gate,
        "primary_exact_witness_controls_pass":controls_ok,
        "registered_primary_pass":passed,
        "scope":("Output-preview leakage and raw seed-prefix-bit leakage are distinct. "
                 "A seed and a full audit schedule are private to the attacker by protocol. "
                 "SHA256-expanded Python Random is a synthetic seed-mixing benchmark, "
                 "not a certified CSPRNG. An independent authenticated exact target witness "
                 "is assumed. 12 public source units, correlated within-source draws."),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--make-fixture",type=Path)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    if args.make_fixture:
        if args.fixture or args.out:
            p.error("source freezing precedes challenge evaluation")
        raw=n24.canonical(make_fixture())
        args.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        args.make_fixture.write_bytes(raw)
        print("NOR27 PUBLIC-ONLY FIRST SOURCE",len(raw),sha256(raw).hexdigest())
        return
    if args.fixture is None:
        p.error("--fixture required")
    result=evaluate(args.fixture)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(n24.canonical(result))
    print("NOR27 PREVIEW MEANS",result["preview_source_means"])
    print("NOR27 SEED BITS MEANS",result["seed_prefix_source_means"])
    print("NOR27 PRIMARY",result["registered_primary_pass"],
          result["primary_preview_degradation_pass"],
          result["primary_preview_theory_agreement_pass"],
          result["primary_seed_prefix_sensitivity_pass"],
          result["primary_exact_witness_controls_pass"])


if __name__=="__main__":
    main()
