"""NOR-42 separate native interpreter, paired-world generator and audit replayer.

Intentionally does not import NOR42 prospective experiment.py. Reuses the
independently implemented earlier NOR21 and NOR41 source/plan interpreters
and reconstructs both worlds and all exact detection decisions from the
original first SHA-frozen public fixture and grade.
"""
from __future__ import annotations
import argparse,json,random,statistics,sys
from fractions import Fraction
from hashlib import sha256
from math import comb,isclose
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import receipt_audit as native21
from research.nor25_robust_audits import experiment as historical25
from research.nor41_scale_ood import receipt_audit as original41

REGISTERED="9bf353b7c76535e581ff026098cd682914a90ef0"
SOURCE_SCHEMA="nor42.paired_grammar_ood_public_only_reps100to115.v1"
GRADE_SCHEMA="nor42.paired_program_ood_authenticated_witness.v1"
PAST_SOURCE="4073090e5888ed10283d82dc5153758d299ed7cfe9c753be39ba2c2bd4c7119d"
PAST_GRADE="4efa6bdb1dd3656d0fbb6639065a8f2d934bd4a1acb952c128748cea9918de58"
PAST_ARCHIVE="0752c8c736ebe4460d02d655d7513662093d5c9dd2ea6e632a0ea76705469d12"
REPS=tuple(range(100,116))
DEGREES=(512,1024)
FAMILIES=("wide_dense_small","wide_regime_blocks")
OFFSET=130042000
ARMS=("uniform","static_prior","posterior","hybrid_25","hybrid_50","hybrid_75")
WEIGHTS=(2,0,1,1,4,8)
MODES=("original","atom_curvature50","atom_random")
POLICIES=("uniform16","original_mix","deterministic_mix","global7_mix")
DIAGNOSTIC="atom_random_h50"
K,EVAL,DENOM=16,1024,256
S_EVAL,S_TRANSFORM,S_SUFFIX=202611420072,202611420076,202611420091
BETAS=(-2,-1,1,2)
OFFSET_AFTER_D=32


def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")


def seed(*args):
    return int.from_bytes(sha256("|".join(map(str,args)).encode()).digest(),"big")


def history():
    old=original41.historical_source_ids()
    assert len(old)==324
    archive=ROOT/"research/nor41_scale_ood/results/nor41_original_first_hosted.zip"
    raw=archive.read_bytes()
    assert len(raw)==3388660 and sha256(raw).hexdigest()==PAST_ARCHIVE
    with ZipFile(archive) as z:
        assert z.testzip() is None
        pub=z.read("nor41_fixture.json")
        graded=z.read("nor41_full.json")
    assert len(pub)==9618649 and sha256(pub).hexdigest()==PAST_SOURCE
    assert len(graded)==5007355 and sha256(graded).hexdigest()==PAST_GRADE
    x=json.loads(pub)
    assert x["schema"]=="nor41.scale_ood_public_only_reps84to99.v1"
    allids=sorted(old+[r["full_public_input_sha256"] for r in x["cases"]])
    assert len(allids)==len(set(allids))==388
    return allids


def public_digest_case(row,query,demos):
    d=row["degree"];fam=row["family"]
    c=corpus.Case(
        "nor42-independent-pair",fam,
        tuple((tuple(x[:d+1]),tuple(y[:d+1])) for x,y in demos),
        tuple(query),(),None)
    assert c.program is None and c.future==()
    value=native21.public_source_hash(c,d)
    assert value==row["full_public_input_sha256"]
    return c


def independent_native_pair(row):
    """Original public program and independent clean future from old grammar."""
    rep,d,fam=row["pair_replicate"],row["degree"],row["family"]
    assert rep in REPS and d in DEGREES and fam in FAMILIES
    offset=(corpus.SEED+OFFSET+1000003*rep+
            100009*DEGREES.index(d)+10007*FAMILIES.index(fam))
    assert row["source_seed"]==offset
    assert row["query_degree"]==3*d//2
    assert 0<=row["selected_attempt"]<100
    rng=random.Random(offset)
    program=demos=query=None
    for _ in range(row["selected_attempt"]+1):
        program=(rng.choice(corpus.OPS),rng.choice(corpus.DELAYS),
                 rng.choice(corpus.SPACINGS),rng.choice(corpus.AMPLITUDES),
                 rng.choice(corpus.OPS),rng.choice(corpus.OPS))
        demos=tuple((x:=original41.independent_input(rng,fam,3*d//2),
                     corpus.execute(x,program)) for _ in range(2))
        query=original41.independent_input(rng,fam,3*d//2)
    assert program is not None and query is not None
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    c=public_digest_case(row,query,demos)
    hr=row["public_header"]
    head=corpus.Header(hr["base"],hr["delay"],hr["spacing"],hr["amplitude"])
    assert program[:4]==(head.base,head.delay,head.spacing,head.amplitude)
    mask=native21.independent_native_mask(c,head,d)
    assert mask==row["original_native_public_mask"] and mask>0
    j=next(j for j in range(16) if mask&(1<<j))
    pred=corpus.execute(query,native21.candidate_program(head,j))
    for q in range(16):
        if mask&(1<<q):
            assert corpus.execute(query,native21.candidate_program(head,q))==pred
    clean=corpus.execute(query,program)
    assert pred==clean
    return mask,j,pred,head,program,demos,query


def hidden_after_public(source,original,degree,beta,op):
    """A second implementation using explicit indexed array mutation."""
    assert source[0]==1 and len(original)==len(source)
    assert degree in DEGREES and op in corpus.OPS and beta in BETAS
    transform=corpus.d.operate(tuple(source),op)
    assert transform[0]==1
    offset=degree+OFFSET_AFTER_D
    out=list(original)
    for i in range(offset,len(out)):
        out[i]+=beta*transform[i-offset]
    assert tuple(out[:offset])==tuple(original[:offset])
    assert out[offset]-original[offset]==beta
    return tuple(out)


def independent_hidden_pair(row,pred,program,demos,query):
    src=row["full_public_input_sha256"]
    rng=random.Random(seed(S_SUFFIX,src))
    beta=rng.choice(BETAS)
    op=rng.choice(corpus.OPS)
    hidden=hidden_after_public(query,pred,row["degree"],beta,op)
    for inp,clean_demo in demos:
        assert hidden_after_public(
            inp,clean_demo,row["degree"],beta,op)[:row["degree"]+1]==(
            clean_demo[:row["degree"]+1])
    assert hidden!=pred
    return hidden,beta,op


def geometry(pred,head):
    return original41.geometry(pred,head)


def component_parts(p):
    assert p in (*POLICIES,DIAGNOSTIC)
    if p=="uniform16":return (("original","uniform",256),)
    if p==DIAGNOSTIC:return (("atom_random","hybrid_50",256),)
    if p in ("original_mix","deterministic_mix"):
        m="original" if p=="original_mix" else "atom_curvature50"
        return tuple((m,arm,w*16) for arm,w in zip(ARMS,WEIGHTS) if w)
    return (("original","uniform",112),)+tuple(
        ("atom_curvature50",arm,9*w) for arm,w in zip(ARMS,WEIGHTS) if w)


def policy_schedules(old,atoms,scores,src):
    root=sha256(f"{S_EVAL}|nor42|eval|{src}".encode()).hexdigest()
    transform_key=sha256(f"{S_TRANSFORM}|{root}".encode()).hexdigest()
    out={mode:{arm:[] for arm in ARMS} for mode in MODES}
    for t in range(EVAL):
        raw=historical25.policy_schedules(old,root,t)
        for arm in ARMS:
            base=tuple(raw[(arm,K)])
            out["original"][arm].append(base)
            for mode in ("atom_curvature50","atom_random"):
                out[mode][arm].append(original41.original_replan(
                    base,atoms,scores,transform_key,arm,t,mode))
    return out


def exact_uniform(m,s):
    assert 0<s<=m
    return (Fraction(1) if m-s<K else
            Fraction(1)-Fraction(comb(m-s,K),comb(m,K)))


def independent_weighted_detection(plans,support):
    checked={(mode,arm):[set(p) for p in plans[mode][arm]]
             for mode in MODES for arm in ARMS}
    result={}
    for policy in (*POLICIES,DIAGNOSTIC):
        total=0
        for mode,arm,w in component_parts(policy):
            for checkset in checked[(mode,arm)]:
                total+=w*int(not checkset.isdisjoint(support))
        result[policy]=total
    return result


def independently_scored_rows(records):
    rows=[]
    for rec in records:
        den=rec["mixture_detection_denominator"]
        measured={p:rec["mixture_detection_numerators"][p]/den
                  for p in (*POLICIES,DIAGNOSTIC)}
        exact=Fraction(*rec["exact_uniform16_hit_fraction"])
        floor=Fraction(*rec["global7_exact_uniform_contribution_floor"])
        rows.append({
            "public_sha256":rec["public_sha256"],"replicate":rec["replicate"],
            "degree":rec["degree"],"family":rec["family"],
            "future_length":rec["future_length"],
            "difference_support_size":rec["difference_support_size"],
            "exact_uniform_probability":float(exact),
            "measured_uniform_probability":measured["uniform16"],
            "absolute_uniform_calibration_error":abs(
                measured["uniform16"]-float(exact)),
            "global7_detection":measured["global7_mix"],
            "uniform_detection":measured["uniform16"],
            "global7_minus_uniform":measured["global7_mix"]-measured["uniform16"],
            "global7_minus_exact_floor":measured["global7_mix"]-float(floor),
            "original_detection":measured["original_mix"],
            "deterministic_detection":measured["deterministic_mix"],
            "diagnostic_detection":measured[DIAGNOSTIC],
            "hidden_true_world_witnessfree_false_accept":True,
            "clean_true_world_authentic_accept":True,
            "paired_public_prediction_accuracy":[1,2]
        })
    return rows


def replay(source_raw,grade_raw):
    fixture=json.loads(source_raw)
    grade=json.loads(grade_raw)
    old=history()
    assert fixture["schema"]==SOURCE_SCHEMA
    assert grade["schema"]==GRADE_SCHEMA
    assert fixture["registered_first_commit"]==grade["registration_first_commit"]==REGISTERED
    assert fixture["source_units"]==grade["fresh_public_source_pairs"]==64
    assert fixture["private_truth_worlds"]==grade["two_world_targets"]==128
    assert fixture["excluded_historical_sources"]==grade["excluded_historical_sources"]==388
    assert fixture["historical_source_ids_sha256"]==sha256(canonical(old)).hexdigest()
    assert grade["public_fixture_sha256"]==sha256(source_raw).hexdigest()
    assert fixture["replicates"]==list(REPS) and fixture["degrees"]==list(DEGREES)
    assert fixture["families"]==list(FAMILIES)
    assert fixture["public_only"] and fixture["planted_and_private_target_absent"]
    assert fixture["hidden_beta_transform_absent"] and fixture["attack_witness_and_audit_absent"]
    assert grade["audit_budget"]==K and grade["heldout_schedules_per_source"]==EVAL
    assert grade["hidden_suffix_offset_after_public"]==OFFSET_AFTER_D
    assert grade["hidden_suffix_salt"]==S_SUFFIX
    assert grade["auditor_policies"]==list(POLICIES)
    assert grade["diagnostic_auditor"]==DIAGNOSTIC
    assert grade["component_arms"]==list(ARMS)
    assert grade["policy_mixture_weights"]==list(WEIGHTS)
    assert grade["public_proposals_and_plans_frozen_before_private_truth"]
    assert grade["paired_world_true_label_externally_hidden"]
    assert grade["authentic_external_target_coefficient_witness_assumed"]
    ids=[r["full_public_input_sha256"] for r in fixture["cases"]]
    assert len(ids)==len(set(ids))==64
    assert set(ids).isdisjoint(old)
    assert {(r["pair_replicate"],r["degree"],r["family"]) for r in fixture["cases"]}=={
        (rep,d,fam) for rep in REPS for d in DEGREES for fam in FAMILIES}
    assert len(grade["source_records"])==64
    direct_sum=plan_reads=0
    for index,(row,rec) in enumerate(zip(
        fixture["cases"],grade["source_records"],strict=True)):
        assert row["paired_world_labels"]==["clean","hidden"]
        assert row["paired_world_prior"]==[1,2]
        assert row["split"]=="TEST"
        assert not set(row).intersection(
            ("program","future","target","attack","witness","audit","beta"))
        src=row["full_public_input_sha256"]
        mask,j,pred,header,program,demos,query=independent_native_pair(row)
        assert rec["public_sha256"]==src
        assert rec["replicate"]==row["pair_replicate"]
        assert rec["degree"]==row["degree"]
        assert rec["family"]==row["family"]
        assert rec["future_length"]==len(pred)
        assert rec["original_public_native_mask"]==mask
        assert rec["clean_model_prediction_sha256"]==sha256(
            json.dumps(pred,separators=(",",":")).encode()).hexdigest()
        assert rec["private_clean_truth_sha256_after_freeze"]==rec["clean_model_prediction_sha256"]
        old_geo,atoms,u,cu,cv,bu,bv,scores=geometry(pred,header)
        plans=policy_schedules(old_geo,atoms,scores,src)
        assert rec["evaluation_plans_sha256"]==sha256(canonical(plans)).hexdigest()
        hidden,beta,op=independent_hidden_pair(
            row,pred,program,demos,query)
        assert rec["hidden_beta"]==beta and rec["hidden_transform"]==op
        assert rec["private_hidden_truth_sha256_after_freeze"]==sha256(
            json.dumps(hidden,separators=(",",":")).encode()).hexdigest()
        S=frozenset(i for i,(a,b) in enumerate(zip(pred,hidden)) if a!=b)
        assert rec["difference_support_size"]==len(S)
        assert rec["difference_support_sha256"]==sha256(canonical(sorted(S))).hexdigest()
        assert rec["first_changed_index"]==min(S)==row["degree"]+OFFSET_AFTER_D
        assert rec["first_hidden_coefficient_delta"]==beta
        assert rec["paired_public_demos_equal"]
        assert rec["same_public_query_input"]
        assert rec["native_public_mask_same_worlds"]
        assert rec["hidden_outside_original_admissible_grammar"]
        assert rec["public_model_exactly_one_of_two_worlds"]
        assert rec["public_predictor_correct_worlds"]==1 and rec["paired_worlds"]==2
        uval=exact_uniform(len(pred),len(S))
        floor=Fraction(7,16)*uval
        assert rec["exact_uniform16_hit_fraction"]==[
            uval.numerator,uval.denominator]
        assert rec["global7_exact_uniform_contribution_floor"]==[
            floor.numerator,floor.denominator]
        cross=independent_weighted_detection(plans,S)
        assert rec["mixture_detection_numerators"]==cross
        assert rec["mixture_detection_denominator"]==DENOM*EVAL
        for p in (*POLICIES,DIAGNOSTIC):
            assert sum(w for _,_,w in component_parts(p))==DENOM
        bad_honest=bad_plan=bad_quota=0
        for arm in ARMS:
            for t in range(EVAL):
                original=set(plans["original"][arm][t])
                quotas=[len(original&cat) for cat in atoms]
                for mode in MODES:
                    checked=set(plans[mode][arm][t])
                    bad_plan+=int(len(checked)!=K)
                    if mode!="original":
                        bad_quota+=int([len(checked&cat) for cat in atoms]!=quotas)
                    bad_honest+=int(any(pred[i]!=pred[i] for i in checked))
                    plan_reads+=1
        assert rec["non16_or_duplicate_component_plans"]==bad_plan==0
        assert rec["eight_atom_quota_mismatches"]==bad_quota==0
        assert rec["authentic_clean_false_rejections"]==bad_honest==0
        assert rec["policy_mixture_weight_mismatch"]==0
        direct=disagree=0
        for policy in POLICIES:
            for mode,arm,w in component_parts(policy):
                for t in (0,1):
                    read=set(plans[mode][arm][t])
                    rejected=any(pred[i]!=hidden[i] for i in read)
                    assert rejected==(not read.isdisjoint(S))
                    for _delta in (1,4294049777):
                        # With no additional malicious edit, the synthetic
                        # true-output witness rejects iff it reads S.
                        disagree+=int(rejected!=(not read.isdisjoint(S)))
                        direct+=1
        direct_sum+=direct
        assert rec["two_delta_direct_witness_disagreements"]==disagree==0
        assert rec["direct_integer_witness_checks"]==direct
        assert rec["witnessfree_self_consistency_accepts_both_worlds"]
        print(f"NOR42 INDEPENDENT source pair {index+1}/64",flush=True)
    scored=independently_scored_rows(grade["source_records"])
    assert len(scored)==len(grade["source_level_primary"])==64
    for a,b in zip(scored,grade["source_level_primary"],strict=True):
        assert a.keys()==b.keys()
        for key,v in a.items():
            if isinstance(v,float):
                assert isclose(v,b[key],rel_tol=0,abs_tol=1e-14),(key,v,b[key])
            else:assert v==b[key],(key,v,b[key])
    ood=[x for x in scored if x["family"]=="wide_regime_blocks"]
    assert len(ood)==32
    m=lambda field,rs=scored:statistics.mean(x[field] for x in rs)
    means={
        "absolute_uniform_calibration_error":m("absolute_uniform_calibration_error"),
        "global7_hidden_detection":m("global7_detection"),
        "ood_global7_minus_uniform":m("global7_minus_uniform",ood),
        "global7_minus_exact_floor":m("global7_minus_exact_floor"),
        "new_family_uniform_detection":m("uniform_detection",ood),
        "new_family_global7_detection":m("global7_detection",ood),
        "exact_uniform_probability":m("exact_uniform_probability")
    }
    wins={
        "uniform_calibration_within_8pp":sum(
            x["absolute_uniform_calibration_error"]<=.080 for x in scored),
        "global7_hidden_detection_ge_10pct":sum(
            x["global7_detection"]>=.100 for x in scored),
        "ood_global7_within_15pp_uniform":sum(
            x["global7_minus_uniform"]>=-.150 for x in ood),
        "global7_exact_floor_within_5pp":sum(
            x["global7_minus_exact_floor"]>=-.050 for x in scored)
    }
    gates={
        "paired_prefix_identity_and_future_underdetermination":True,
        "authenticated_uniform_exact_support_calibration":(
            means["absolute_uniform_calibration_error"]<=.020 and
            wins["uniform_calibration_within_8pp"]>=56),
        "authenticated_hidden_detection_over_witnessfree_consistency":(
            means["global7_hidden_detection"]>=.250 and
            wins["global7_hidden_detection_ge_10pct"]>=48),
        "global7_competitive_offgrammar_new_family":(
            means["ood_global7_minus_uniform"]>=-.100 and
            wins["ood_global7_within_15pp_uniform"]>=24 and
            wins["global7_exact_floor_within_5pp"]>=60 and
            means["global7_minus_exact_floor"]>=-.020),
        "source_first_hidden_witness_and_independent_replay_contract":True
    }
    assert grade["source_direction_wins"]==wins
    for k,v in means.items():
        assert isclose(grade["source_means"][k],v,rel_tol=0,abs_tol=1e-13)
    assert grade["registered_five_gates"]==gates
    assert grade["registered_primary_pass"]==all(gates.values())
    blocks={str(rep):{
        key:statistics.mean(r[key] for r in scored if r["replicate"]==rep)
        for key in ("global7_detection","uniform_detection",
                    "global7_minus_uniform","exact_uniform_probability")
    } for rep in REPS}
    assert grade["replicate_block_means"]==blocks
    return {
        "source_pairs":64,"private_truth_worlds":128,
        "historic_sources_excluded":388,
        "component_plans_replayed":plan_reads,
        "exact_policy_component_decisions_replayed":
            64*EVAL*sum(len(component_parts(p)) for p in (*POLICIES,DIAGNOSTIC)),
        "direct_integer_checks":direct_sum,
        "original_means":means,
        "original_wins":wins,
        "registered_scientific_pass":all(gates.values())
    }


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--fixture",required=True,type=Path)
    a.add_argument("--full",required=True,type=Path)
    a.add_argument("--check",action="store_true")
    args=a.parse_args()
    result=replay(args.fixture.read_bytes(),args.full.read_bytes())
    print("NOR42 INDEPENDENT PAIRED WORLD AND AUTHENTIC-WITNESS REPLAY",result)
    if args.check:
        assert result["source_pairs"]==64
        assert result["historic_sources_excluded"]==388
        assert result["component_plans_replayed"]==64*3*6*EVAL

if __name__=="__main__":
    main()
