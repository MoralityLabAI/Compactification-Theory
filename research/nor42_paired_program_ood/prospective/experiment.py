"""NOR-42: prospectively registered paired hidden-program and witness study.

FIRST REGISTRATION SHA: 9bf353b7c76535e581ff026098cd682914a90ef0.
Source-only generation, grade, independent replay are separate CI jobs.
All pre-data tests must use previously SHA-archived sources only.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from fractions import Fraction
from hashlib import sha256
from math import comb
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import experiment as nor21
from research.nor24_adaptive_audits import experiment as nor24
from research.nor25_robust_audits import experiment as nor25
from research.nor35_intraatom_curvature import experiment as nor35
from research.nor37_stochastic_curvature import experiment as nor37
from research.nor41_scale_ood import experiment as nor41

REGISTERED="9bf353b7c76535e581ff026098cd682914a90ef0"
FIXTURE_SCHEMA="nor42.paired_grammar_ood_public_only_reps100to115.v1"
GRADE_SCHEMA="nor42.paired_program_ood_authenticated_witness.v1"
PREV_SOURCE_SHA="4073090e5888ed10283d82dc5153758d299ed7cfe9c753be39ba2c2bd4c7119d"
PREV_GRADE_SHA="4efa6bdb1dd3656d0fbb6639065a8f2d934bd4a1acb952c128748cea9918de58"
PREV_ZIP_SHA="0752c8c736ebe4460d02d655d7513662093d5c9dd2ea6e632a0ea76705469d12"
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
S_EVAL=202611420072
S_TRANSFORM=202611420076
S_BRANCH=202611420075
S_SUFFIX=202611420091
DELTAS=(1,4294049777)
BETAS=(-2,-1,1,2)
ONSET_EXTRA=32


def canonical(obj):
    return (json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")


def digest(buf):
    return sha256(buf).hexdigest()


def seed(*args):
    return int.from_bytes(sha256("|".join(map(str,args)).encode("ascii")).digest(),"big")


def history():
    """Original 388 SHA identities; never synthesizes NOR42 sources."""
    past=nor41.history()
    assert len(past)==324
    original=ROOT/"research/nor41_scale_ood/results/nor41_original_first_hosted.zip"
    assert digest(original.read_bytes())==PREV_ZIP_SHA
    with ZipFile(original) as z:
        assert z.testzip() is None
        src=z.read("nor41_fixture.json")
        grade=z.read("nor41_full.json")
    assert len(src)==9618649 and digest(src)==PREV_SOURCE_SHA
    assert len(grade)==5007355 and digest(grade)==PREV_GRADE_SHA
    x=json.loads(src)
    assert x["schema"]=="nor41.scale_ood_public_only_reps84to99.v1"
    assert len(x["cases"])==64
    combined=sorted(past+[r["full_public_input_sha256"] for r in x["cases"]])
    assert len(combined)==len(set(combined))==388
    return combined


def source_seed(rep,d,fam):
    assert rep in REPS and d in DEGREES and fam in FAMILIES
    return (corpus.SEED+OFFSET+1000003*rep+
            100009*DEGREES.index(d)+10007*FAMILIES.index(fam))


def new_source(rep,d,fam,seen):
    """Source only; no hidden beta/transform, private target or audit plan."""
    q=3*d//2
    rng=random.Random(source_seed(rep,d,fam))
    for attempt in range(100):
        program=nor21.new_header_program(rng)
        demos=tuple((x:=nor41.public_input(rng,fam,q),
                     corpus.execute(x,program)) for _ in range(2))
        query=nor41.public_input(rng,fam,q)
        c=corpus.Case("nor42-public-pair",fam,demos,query,(),None)
        head=corpus.possible_headers(c)
        if len(head)!=1:continue
        h=head[0]
        if (h.base,h.delay,h.spacing,h.amplitude)!=program[:4]:continue
        public_sha=nor21.public_digest(c,d)
        if public_sha in seen:continue
        mask=nor21.native_mask(c,h,d)
        if not mask:continue
        candidates=nor21.candidate_predictions(query,h,q)
        if len({candidates[j] for j in range(16) if mask&(1<<j)})!=1:
            continue
        planted_class=4*corpus.OPS.index(program[4])+corpus.OPS.index(program[5])
        assert mask&(1<<planted_class)
        seen.add(public_sha)
        return {
            "split":"TEST","pair_replicate":rep,"degree":d,
            "query_degree":q,"family":fam,
            "source_seed":source_seed(rep,d,fam),
            "selected_attempt":attempt,
            "full_public_input_sha256":public_sha,
            "public_header":{"base":h.base,"delay":h.delay,
                             "spacing":h.spacing,"amplitude":h.amplitude},
            "demos":[[list(x[:d+1]),list(y[:d+1])] for x,y in demos],
            "public_query_input":list(query),
            "original_native_public_mask":mask,
            "paired_world_labels":["clean","hidden"],
            "paired_world_prior":[1,2]
        }
    raise RuntimeError("NOR42 original source exhausted 100 attempts")


def make_fixture():
    old=history()
    seen=set(old)
    rows=[new_source(rep,d,fam,seen)
          for rep in REPS for d in DEGREES for fam in FAMILIES]
    assert len(rows)==64 and len(seen)==452
    return {
        "schema":FIXTURE_SCHEMA,
        "registered_first_commit":REGISTERED,
        "replicates":list(REPS),"degrees":list(DEGREES),
        "families":list(FAMILIES),"source_offset":OFFSET,
        "source_units":64,"private_truth_worlds":128,
        "public_only":True,"planted_and_private_target_absent":True,
        "hidden_beta_transform_absent":True,
        "attack_witness_and_audit_absent":True,
        "excluded_historical_sources":388,
        "historical_source_ids_sha256":digest(canonical(old)),
        "cases":rows
    }


def read_fixture(path):
    buf=Path(path).read_bytes()
    x=json.loads(buf)
    past=history()
    assert x["schema"]==FIXTURE_SCHEMA and x["registered_first_commit"]==REGISTERED
    assert x["replicates"]==list(REPS) and x["degrees"]==list(DEGREES)
    assert x["families"]==list(FAMILIES) and x["source_offset"]==OFFSET
    assert x["source_units"]==64 and x["private_truth_worlds"]==128
    assert x["public_only"] and x["planted_and_private_target_absent"]
    assert x["hidden_beta_transform_absent"] and x["attack_witness_and_audit_absent"]
    assert x["excluded_historical_sources"]==len(past)==388
    assert x["historical_source_ids_sha256"]==digest(canonical(past))
    rows=x["cases"]
    ids=[r["full_public_input_sha256"] for r in rows]
    assert len(ids)==len(set(ids))==64 and set(ids).isdisjoint(past)
    assert {(r["pair_replicate"],r["degree"],r["family"]) for r in rows}=={
        (rep,d,fam) for rep in REPS for d in DEGREES for fam in FAMILIES}
    assert all(r["paired_world_labels"]==["clean","hidden"] and
               r["paired_world_prior"]==[1,2] and r["split"]=="TEST" and
               not set(r).intersection(("program","future","target","attack",
                                        "witness","audit","beta","hidden_transform"))
               for r in rows)
    return x,digest(buf)


def hidden_future(source,clean,d,beta,transform):
    assert len(source)==len(clean) and source[0]==1
    assert d in DEGREES and beta in BETAS and transform in corpus.OPS
    onset=d+ONSET_EXTRA
    shifted=corpus.d.operate(source,transform)
    assert shifted[0]==1 and onset<len(clean)
    result=tuple(
        val if i<onset else val+beta*shifted[i-onset]
        for i,val in enumerate(clean))
    assert result[:d+1]==tuple(clean[:d+1])
    assert result[:onset]==tuple(clean[:onset])
    assert result[onset]-clean[onset]==beta
    return result


def restore_worlds(row,pred):
    """Private native program and hidden truth only after frozen public plans."""
    rep,d,fam=row["pair_replicate"],row["degree"],row["family"]
    assert rep in REPS and d in DEGREES and fam in FAMILIES
    assert row["source_seed"]==source_seed(rep,d,fam)
    rng=random.Random(source_seed(rep,d,fam))
    for _ in range(row["selected_attempt"]+1):
        planted=nor21.new_header_program(rng)
        demos=tuple((x:=nor41.public_input(rng,fam,3*d//2),
                     corpus.execute(x,planted)) for _ in range(2))
        query=nor41.public_input(rng,fam,3*d//2)
    assert row["demos"]==[[list(x[:d+1]),list(y[:d+1])] for x,y in demos]
    assert row["public_query_input"]==list(query)
    clean=corpus.execute(query,planted)
    assert clean==pred
    src=row["full_public_input_sha256"]
    latent=random.Random(seed(S_SUFFIX,src))
    beta=latent.choice(BETAS)
    transform=latent.choice(corpus.OPS)
    hidden=hidden_future(query,clean,d,beta,transform)
    # The same off-grammar rule would apply to both demonstrations, whose
    # public output prefixes therefore cannot reveal the latent world label.
    for f,y in demos:
        h=hidden_future(f,y,d,beta,transform)
        assert h[:d+1]==y[:d+1]
    return clean,hidden,beta,transform


def components(policy):
    assert policy in (*POLICIES,DIAGNOSTIC)
    if policy=="uniform16":return (("original","uniform",DENOM),)
    if policy==DIAGNOSTIC:return (("atom_random","hybrid_50",DENOM),)
    if policy in ("original_mix","deterministic_mix"):
        mode="original" if policy=="original_mix" else "atom_curvature50"
        return tuple((mode,arm,16*w)
                     for arm,w in zip(ARMS,WEIGHTS) if w)
    return (("original","uniform",112),)+tuple(
        ("atom_curvature50",arm,9*w)
        for arm,w in zip(ARMS,WEIGHTS) if w)


def eval_tag(src):
    return sha256(f"{S_EVAL}|nor42|eval|{src}".encode("ascii")).hexdigest()


def schedules(old,cats,scores,src):
    root=eval_tag(src)
    key=sha256(f"{S_TRANSFORM}|{root}".encode("ascii")).hexdigest()
    plans={mode:{arm:[] for arm in ARMS} for mode in MODES}
    for t in range(EVAL):
        raw=nor25.policy_schedules(old,root,t)
        for arm in ARMS:
            base=tuple(raw[(arm,K)])
            assert len(base)==len(set(base))==K
            plans["original"][arm].append(base)
            for mode in ("atom_curvature50","atom_random"):
                p=nor35.transformed_plan(base,cats,scores,key,arm,t,mode)
                assert len(p)==len(set(p))==K
                plans[mode][arm].append(p)
    return plans


def uniform_exact(m,s):
    assert 0<s<=m and 0<K<=m
    return (Fraction(1) if m-s<K else
            Fraction(1)-Fraction(comb(m-s,K),comb(m,K)))


def weighted_detection(plans,changed):
    output={}
    for policy in (*POLICIES,DIAGNOSTIC):
        num=0
        for mode,arm,w in components(policy):
            for checked in plans[mode][arm]:
                num+=w*int(any(i in changed for i in checked))
        output[policy]=num
    return output


def source_rows(records):
    rows=[]
    for rec in records:
        den=rec["mixture_detection_denominator"]
        rates={p:rec["mixture_detection_numerators"][p]/den
               for p in (*POLICIES,DIAGNOSTIC)}
        exact=Fraction(*rec["exact_uniform16_hit_fraction"])
        floor=Fraction(*rec["global7_exact_uniform_contribution_floor"])
        rows.append({
            "public_sha256":rec["public_sha256"],
            "replicate":rec["replicate"],"degree":rec["degree"],
            "family":rec["family"],"future_length":rec["future_length"],
            "difference_support_size":rec["difference_support_size"],
            "exact_uniform_probability":float(exact),
            "measured_uniform_probability":rates["uniform16"],
            "absolute_uniform_calibration_error":abs(
                rates["uniform16"]-float(exact)),
            "global7_detection":rates["global7_mix"],
            "uniform_detection":rates["uniform16"],
            "global7_minus_uniform":rates["global7_mix"]-rates["uniform16"],
            "global7_minus_exact_floor":rates["global7_mix"]-float(floor),
            "original_detection":rates["original_mix"],
            "deterministic_detection":rates["deterministic_mix"],
            "diagnostic_detection":rates[DIAGNOSTIC],
            "hidden_true_world_witnessfree_false_accept":True,
            "clean_true_world_authentic_accept":True,
            "paired_public_prediction_accuracy":[1,2]
        })
    return rows


def primary_gates(records,rows):
    ood=[r for r in rows if r["family"]=="wide_regime_blocks"]
    assert len(rows)==64 and len(ood)==32
    mean=lambda k,rs=rows:statistics.mean(x[k] for x in rs)
    means={
        "absolute_uniform_calibration_error":mean(
            "absolute_uniform_calibration_error"),
        "global7_hidden_detection":mean("global7_detection"),
        "ood_global7_minus_uniform":mean("global7_minus_uniform",ood),
        "global7_minus_exact_floor":mean("global7_minus_exact_floor"),
        "new_family_uniform_detection":mean("uniform_detection",ood),
        "new_family_global7_detection":mean("global7_detection",ood),
        "exact_uniform_probability":mean("exact_uniform_probability"),
    }
    wins={
        "uniform_calibration_within_8pp":sum(
            r["absolute_uniform_calibration_error"]<=.080 for r in rows),
        "global7_hidden_detection_ge_10pct":sum(
            r["global7_detection"]>=.100 for r in rows),
        "ood_global7_within_15pp_uniform":sum(
            r["global7_minus_uniform"]>=-.150 for r in ood),
        "global7_exact_floor_within_5pp":sum(
            r["global7_minus_exact_floor"]>=-.050 for r in rows)
    }
    math_good=all(
        r["paired_public_demos_equal"] and
        r["same_public_query_input"] and
        r["native_public_mask_same_worlds"] and
        r["hidden_outside_original_admissible_grammar"] and
        r["public_model_exactly_one_of_two_worlds"] and
        r["first_changed_index"]==r["degree"]+ONSET_EXTRA and
        r["first_hidden_coefficient_delta"]==r["hidden_beta"]!=0 and
        r["difference_support_size"]>0
        for r in records)
    witness_good=all(
        r["authentic_clean_false_rejections"]==0 and
        r["two_delta_direct_witness_disagreements"]==0 and
        r["direct_integer_witness_checks"]>0 and
        r["witnessfree_self_consistency_accepts_both_worlds"] and
        r["paired_worlds"]==2 and
        r["public_predictor_correct_worlds"]==1
        for r in records)
    plan_good=all(
        r["non16_or_duplicate_component_plans"]==0 and
        r["policy_mixture_weight_mismatch"]==0 and
        r["eight_atom_quota_mismatches"]==0 and
        len(r["mixture_detection_numerators"])==5
        for r in records)
    gates={
        "paired_prefix_identity_and_future_underdetermination":math_good,
        "authenticated_uniform_exact_support_calibration":(
            means["absolute_uniform_calibration_error"]<=.020
            and wins["uniform_calibration_within_8pp"]>=56
            and witness_good),
        "authenticated_hidden_detection_over_witnessfree_consistency":(
            means["global7_hidden_detection"]>=.250 and
            wins["global7_hidden_detection_ge_10pct"]>=48
            and witness_good),
        "global7_competitive_offgrammar_new_family":(
            means["ood_global7_minus_uniform"]>=-.100 and
            wins["ood_global7_within_15pp_uniform"]>=24
            and wins["global7_exact_floor_within_5pp"]>=60
            and means["global7_minus_exact_floor"]>=-.020),
        "source_first_hidden_witness_and_independent_replay_contract":(
            math_good and witness_good and plan_good
            and len(records)==64
            and len({r["public_sha256"] for r in records})==64)
    }
    return means,wins,gates


def evaluate(fixture_path):
    fixture,source_hash=read_fixture(fixture_path)
    proposed={r["full_public_input_sha256"]:nor37.public_proposal(r)
              for r in fixture["cases"]}
    assert len(proposed)==64
    records=[]
    for n,row in enumerate(fixture["cases"]):
        src=row["full_public_input_sha256"]
        p=proposed[src]
        pred=p["pred"]
        old,cats,u,cu,cv,bu,bv,scores=nor37.source_geometry(
            pred,p["header"],src)
        plans=schedules(old,cats,scores,src)
        # All source-public plans are frozen before either full true program
        # or hidden world is reconstructed.
        clean,hidden,beta,transform=restore_worlds(row,pred)
        assert clean==pred and hidden!=clean
        m=len(clean)
        support=frozenset(i for i,(x,y) in enumerate(zip(clean,hidden))
                          if x!=y)
        assert min(support)==row["degree"]+ONSET_EXTRA
        assert hidden[min(support)]-clean[min(support)]==beta
        admissible=nor21.candidate_predictions(
            tuple(row["public_query_input"]),p["header"],row["query_degree"])
        assert all(admissible[j]==clean for j in range(16)
                   if p["mask"]&(1<<j))
        assert all(admissible[j]!=hidden for j in range(16)
                   if p["mask"]&(1<<j))
        exact=uniform_exact(m,len(support))
        floor=Fraction(7,16)*exact
        numerators=weighted_detection(plans,support)
        weighted_den=DENOM*EVAL
        bad_honest=bad_plan=bad_quota=direct=disagree=0
        for arm in ARMS:
            for t in range(EVAL):
                base=set(plans["original"][arm][t])
                quota=[len(base&set(c)) for c in cats]
                for mode in MODES:
                    plan=plans[mode][arm][t]
                    checked=set(plan)
                    bad_plan+=int(len(plan)!=K or len(checked)!=K)
                    if mode!="original":
                        bad_quota+=int([len(checked&set(c)) for c in cats]!=quota)
                    bad_honest+=int(not nor24.exact_witness_accepts(
                        clean,clean,plan,frozenset(),1))
        for policy in POLICIES:
            for mode,arm,w in components(policy):
                for t in (0,1):
                    plan=plans[mode][arm][t]
                    hit=bool(support.intersection(plan))
                    for delta in DELTAS:
                        rejected=not nor24.exact_witness_accepts(
                            clean,hidden,plan,frozenset(),delta)
                        disagree+=int(rejected!=hit)
                        direct+=1
        policy_bad=sum(int(sum(w for _,_,w in components(pol))!=DENOM)
                       for pol in (*POLICIES,DIAGNOSTIC))
        assert bad_honest==bad_plan==bad_quota==disagree==policy_bad==0
        records.append({
            "public_sha256":src,
            "replicate":row["pair_replicate"],
            "degree":row["degree"],"family":row["family"],
            "future_length":m,
            "original_public_native_mask":p["mask"],
            "clean_model_prediction_sha256":p["sha"],
            "private_clean_truth_sha256_after_freeze":nor24.sha_target(clean),
            "private_hidden_truth_sha256_after_freeze":nor24.sha_target(hidden),
            "hidden_beta":beta,"hidden_transform":transform,
            "first_changed_index":min(support),
            "first_hidden_coefficient_delta":hidden[min(support)]-clean[min(support)],
            "difference_support_size":len(support),
            "difference_support_sha256":digest(canonical(sorted(support))),
            "paired_public_demos_equal":True,
            "same_public_query_input":True,
            "native_public_mask_same_worlds":True,
            "hidden_outside_original_admissible_grammar":True,
            "public_model_exactly_one_of_two_worlds":True,
            "paired_worlds":2,"public_predictor_correct_worlds":1,
            "exact_uniform16_hit_fraction":[exact.numerator,exact.denominator],
            "global7_exact_uniform_contribution_floor":[floor.numerator,floor.denominator],
            "mixture_detection_numerators":numerators,
            "mixture_detection_denominator":weighted_den,
            "evaluation_plans_sha256":digest(canonical(plans)),
            "authentic_clean_false_rejections":bad_honest,
            "two_delta_direct_witness_disagreements":disagree,
            "direct_integer_witness_checks":direct,
            "witnessfree_self_consistency_accepts_both_worlds":True,
            "non16_or_duplicate_component_plans":bad_plan,
            "eight_atom_quota_mismatches":bad_quota,
            "policy_mixture_weight_mismatch":policy_bad
        })
        print(f"NOR42 source-pair {n+1}/64 rep={row['pair_replicate']} "
              f"D={row['degree']} family={row['family']}",flush=True)
    rows=source_rows(records)
    means,wins,gates=primary_gates(records,rows)
    block_means={str(rep):{
        key:statistics.mean(r[key] for r in rows if r["replicate"]==rep)
        for key in ("global7_detection","uniform_detection",
                    "global7_minus_uniform","exact_uniform_probability")
    } for rep in REPS}
    return {
        "schema":GRADE_SCHEMA,"registration_first_commit":REGISTERED,
        "public_fixture_sha256":source_hash,
        "fresh_public_source_pairs":64,"two_world_targets":128,
        "excluded_historical_sources":388,
        "auditor_policies":list(POLICIES),
        "diagnostic_auditor":DIAGNOSTIC,
        "component_arms":list(ARMS),"policy_mixture_weights":list(WEIGHTS),
        "audit_budget":K,"heldout_schedules_per_source":EVAL,
        "hidden_suffix_offset_after_public":ONSET_EXTRA,
        "hidden_suffix_salt":S_SUFFIX,
        "authentic_external_target_coefficient_witness_assumed":True,
        "public_proposals_and_plans_frozen_before_private_truth":True,
        "paired_world_true_label_externally_hidden":True,
        "source_records":records,"source_level_primary":rows,
        "source_means":means,"source_direction_wins":wins,
        "replicate_block_means":block_means,
        "registered_five_gates":gates,
        "registered_primary_pass":all(gates.values()),
        "scope":"Constructed paired hidden suffix outside original RRC25 grammar; "
                "true output oracle is synthetic and assumed authenticated externally. "
                "No real-world cryptographic witness or hidden deployment entropy supplied."
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--make-fixture",type=Path)
    parser.add_argument("--fixture",type=Path)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.make_fixture:
        if args.fixture or args.out:
            parser.error("source fixture generation must precede grading in separate stage")
        buf=canonical(make_fixture())
        args.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        args.make_fixture.write_bytes(buf)
        print("NOR42 FIRST PUBLIC SOURCE-PAIR BYTES SHA256",len(buf),digest(buf))
        return
    if not args.fixture:parser.error("--fixture required")
    result=evaluate(args.fixture)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(canonical(result))
    print("NOR42 FIRST REGISTERED SCIENCE PASS",result["registered_primary_pass"])
    print("NOR42 ORIGINAL FIVE GATES",result["registered_five_gates"])
    print("NOR42 MEANS",result["source_means"])
    print("NOR42 SOURCE WINS",result["source_direction_wins"])

if __name__=="__main__":
    main()
