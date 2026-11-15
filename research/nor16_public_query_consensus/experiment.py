"""NOR-16: public query-predictive consensus beyond aliased GRU features.

The public target task supplies 2 degree-192 demonstration input/output
series and a fully public degree-256 QUERY INPUT. The degree-256 QUERY
OUTPUT is withheld; it is accessed only through the original grade()
function after a candidate choice is finalized.

A candidate can be simulated on the known query INPUT, yielding a
hypothetical output. Equal predicted outputs form exact equivalence classes,
and class mass supplies a predictive-consensus prior. Public template
acceptance determines which hypothetical programs are consistent with the
demonstrations. Neither source program nor withheld query output is used
to choose any candidate.

Controls:
 original_4: frozen trained GRU top-four, exactly 4 candidate checks maximum
 consensus_4: 16 public-query simulations, same 4-check inference budget
 hash_4: matched 16 public simulations, hash-shuffled ranking negative control
 full_mask_original_16: 16 exact public checks, original rank among accepted
 full_mask_map_16: 16 checks + public query-output consensus among accepted

The full-mask arms have a larger exact public checking budget and MUST NOT
be described as native four-check improvements.

Counterfactual observational-collision certificates arise by selecting
two publicly accepted candidate programs with different publicly
computable query predictions. They have identical public demonstrations
and query input, but distinct candidate futures. No private native future
is accessed to construct those witnesses.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.rrc_compactification import compound25 as corpus, model25
from research.nor11_attractor_controller import controller as nor11
from research.nor12_task_readout import readout as nor12
from research.nor12_task_readout import independent_confirmation as nor12b
from research.nor13_gated_readout import interaction as nor13
from research.nor15_trajectory_sufficiency import task_test as nor15b

SCHEMA="nor16.public_query_predictive_consensus_source_locked.v1"
CHECKPOINT_SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"
FRESH_OFFSET=93116000
REPS=4
LATE_PER_REP=48
SIGNED_PER_REP=12
RATIONAL_PER_REP=12
FAMILIES=("late_source","signed","rational")
ARMS=("original_4","consensus_4","hash_4","full_mask_original_16",
      "full_mask_map_16")
N_CANDIDATES=16
PAID_BUDGET=4
PREREG_P=.05/2


def visible_source_digest(case):
    return sha256(json.dumps(case.visible,separators=(",",":")).encode()).hexdigest()


def full_public_digest(case):
    # Include only the public two degree-192 demos and degree-256 query input.
    return sha256(json.dumps([case.visible,case.query],
                             separators=(",",":")).encode()).hexdigest()


def fresh_cohorts(roster,reps=REPS,late_per=LATE_PER_REP,
                  signed_per=SIGNED_PER_REP,rational_per=RATIONAL_PER_REP):
    if not 1<=reps<=REPS or not 1<=late_per<=LATE_PER_REP or not 1<=signed_per<=SIGNED_PER_REP or not 1<=rational_per<=RATIONAL_PER_REP:
        raise ValueError("independent NOR16 source budget exceeded")
    # NOR-15B already source-excludes all original NOR11/12/13/14/15A.
    previous=nor15b.source_disjoint_new(roster,
        reps=nor15b.REPS,nonlate=nor15b.NONLATE_PER,late=nor15b.LATE_PER)
    known_vis={visible_source_digest(c)
               for cohort in roster.values() for c,head,digest in cohort}
    known_nonlate={d for cohort in roster.values() for c,head,d in cohort}
    for batch in previous:
        for case,head,digest in batch:
            known_vis.add(visible_source_digest(case))
            if case.family!="late_source":
                known_nonlate.add(digest)
    result=[]
    sizes={"late_source":late_per,"signed":signed_per,"rational":rational_per}
    for rep in range(reps):
        batch=[]
        for fi,fam in enumerate(FAMILIES):
            rng=random.Random(corpus.SEED+FRESH_OFFSET+1000003*rep+10007*fi)
            found=0
            for trial in range(sizes[fam]*2500):
                case=corpus.make_case(
                    rng,fam,f"nor16-r{rep}-{fam}-{trial:05d}")
                headers=corpus.possible_headers(case)
                if len(headers)!=1:
                    continue
                head=headers[0]
                digest=nor11.public_digest(case,head)
                visible=visible_source_digest(case)
                if visible in known_vis or (
                        fam!="late_source" and digest in known_nonlate):
                    continue
                known_vis.add(visible)
                if fam!="late_source":
                    known_nonlate.add(digest)
                batch.append((case,head,digest))
                found+=1
                if found==sizes[fam]:
                    break
            if found!=sizes[fam]:
                raise RuntimeError("NOR16 fresh public source exhausted "+fam)
        result.append(batch)
    return result


def program_for(header,j):
    if not 0<=j<N_CANDIDATES:
        raise ValueError("class out of 16 bounds")
    return (header.base,header.delay,header.spacing,header.amplitude,
            corpus.OPS[j//4],corpus.OPS[j%4])


def public_candidate_predictions(case,header):
    """All 16 possible outputs using public QUERY INPUT, never true future.

    Four unique source transforms are cached and reused. The 16 candidate
    output constructions are exact; 4 expensive unary source operations
    plus 16 linear combination passes per query. Audit against corpus.execute.
    """
    f=case.query
    if len(f)!=corpus.FUTURE+1 or f[0]!=1:
        raise ValueError("requires full public query INPUT through degree 256")
    transforms={op:corpus.d.operate(f,op) for op in corpus.OPS}
    base=transforms[header.base]
    k=header.delay;delta=header.spacing;c=header.amplitude
    outputs=[]
    for j in range(N_CANDIDATES):
        u=corpus.OPS[j//4];v=corpus.OPS[j%4]
        y=list(base)
        y[k]+=c
        start1=k+delta
        for n in range(start1,len(f)):
            y[n]+=c*transforms[u][n-start1]
        start2=k+2*delta
        for n in range(start2,len(f)):
            y[n]+=c*transforms[v][n-start2]
        outputs.append(tuple(y))
    assert len(outputs)==N_CANDIDATES and all(len(z)==corpus.FUTURE+1
                                               for z in outputs)
    return outputs


def group_candidates(predictions):
    if len(predictions)!=N_CANDIDATES:
        raise ValueError("expected sixteen full candidate predictions")
    out=defaultdict(list)
    for j,value in enumerate(predictions):
        out[value].append(j)
    groups=sorted((tuple(v) for v in out.values()),
                  key=lambda g:(-len(g),g[0]))
    sizes=[0]*N_CANDIDATES
    for group in groups:
        for j in group:
            sizes[j]=len(group)
    return groups,sizes


def exact_public_mask(case,header):
    # An additional SIXTEEN paid public demonstration checks, separate
    # from the four-check arms and any native heldout grader calls.
    bits=0
    for j in range(N_CANDIDATES):
        if corpus.verify_template(case,header,corpus.OPS[j//4],corpus.OPS[j%4]):
            bits|=1<<j
    if not bits:
        raise AssertionError("positive generator yielded no public template")
    return bits


def original_order(case,header,ck):
    scores=model25.logits(case,header,ck)
    return sorted(range(N_CANDIDATES),key=lambda j:(-scores[j],j))


def hash_order(case):
    # All available public QUERY INPUT coefficients are admissible.
    # Hash random is deliberately label-independent and frozen.
    digest=sha256(json.dumps(case.query,separators=(",",":")).encode()).hexdigest()
    ids=list(range(N_CANDIDATES))
    random.Random(int(digest[:16],16)).shuffle(ids)
    return ids


def consensus_order(sizes,original):
    return sorted(range(N_CANDIDATES),
                  key=lambda j:(-sizes[j],original.index(j)))


def mask_map_choice(mask,predictions,original):
    """Maximum conditional output-cluster multiplicity for the accepted set.

    The selected candidate represents a most frequent *future prediction*
    among publicly accepted equally-prior programs. Tie-break by the
    original frozen rank, selected before inspecting future target output.
    """
    if not 0<mask<1<<N_CANDIDATES:
        raise ValueError("nonempty exact 16-bit mask required")
    buckets=defaultdict(list)
    for j,y in enumerate(predictions):
        if mask&(1<<j):
            buckets[y].append(j)
    assert buckets and sum(len(x) for x in buckets.values())==mask.bit_count()
    winner=max(buckets.values(),key=lambda group:(
        len(group),-min(original.index(j) for j in group)))
    selected=min(winner,key=original.index)
    assert mask&(1<<selected)
    return selected,max(map(len,buckets.values())),len(buckets)


def proposed_four(case,header,order):
    """Stage A inference: first public-positive within four paid checks.

    Access withheld query OUTPUT only after the candidate is chosen. Policy
    uses public demonstrations via verify_template, never case.future.
    """
    trace=[]
    chosen=None
    for j in order[:PAID_BUDGET]:
        trace.append(j)
        if corpus.verify_template(case,header,corpus.OPS[j//4],
                                  corpus.OPS[j%4]):
            chosen=j
            break
    assert 1<=len(trace)<=PAID_BUDGET
    return chosen,trace


def grade_once(case,header,chosen):
    """Sealed final grader; only function reading withheld future truth."""
    program=program_for(header,chosen) if chosen is not None else None
    return corpus.grade(case,program)


def single_case(case,header,ck):
    base=original_order(case,header,ck)
    predictions=public_candidate_predictions(case,header)
    groups,sizes=group_candidates(predictions)
    cons=consensus_order(sizes,base)
    shuffled=hash_order(case)
    mask=exact_public_mask(case,header)
    choice,posterior_mass,posterior_groups=mask_map_choice(mask,predictions,base)
    first_orig=next(j for j in base if mask & (1<<j))
    proposals={}
    for name,order in (("original_4",base),("consensus_4",cons),
                       ("hash_4",shuffled)):
        j,trace=proposed_four(case,header,order)
        proposals[name]=(j,trace,grade_once(case,header,j))
    # 16 exact public template checks have already been paid; selected
    # class is known accepted. A final candidate check is reused.
    for name,j in (("full_mask_original_16",first_orig),
                   ("full_mask_map_16",choice)):
        proposals[name]=(j,[],grade_once(case,header,j))
    assert set(proposals)==set(ARMS)
    return {
        "model_input_sha256":nor11.public_digest(case,header),
        "public_demos_sha256":visible_source_digest(case),
        "full_public_input_sha256":full_public_digest(case),
        "public_query_input_sha256":sha256(json.dumps(
            case.query,separators=(",",":")).encode()).hexdigest(),
        "family":case.family,
        "public_accept_mask":mask,
        "accepted_candidates":mask.bit_count(),
        "candidate_public_prediction_sha256":[sha256(json.dumps(
            pred,separators=(",",":")).encode()).hexdigest()
            for pred in predictions],
        "query_output_equivalence_group_sizes":sorted(
            (len(g) for g in groups),reverse=True),
        "accepted_candidate_posterior_group_count":posterior_groups,
        "uniform_accepted_candidate_MAP_future_probability_num":posterior_mass,
        "uniform_accepted_candidate_MAP_future_probability_den":mask.bit_count(),
        "max_prediction_group_mass_all_classes":max(map(len,groups)),
        "model_original_top_four":base[:4],
        "consensus_top_four":cons[:4],
        "hash_top_four":shuffled[:4],
        "counterfactual_indistinguishable_pair":counterfactual_pair(
            mask,predictions),
        "arms":{
            name:{
                "chosen_class":j,
                "verification_trace":trace,
                "public_checks":(len(trace) if name.endswith("_4") else 16),
                "query_prediction_candidates":(
                    0 if name=="original_4" else N_CANDIDATES),
                "public_unary_query_transforms":(
                    0 if name=="original_4" else len(corpus.OPS)),
                "grade":grade,
            }
            for name,(j,trace,grade) in proposals.items()
        },
    }


def counterfactual_pair(mask,predictions):
    accepted=[j for j in range(N_CANDIDATES) if mask&(1<<j)]
    for i,a in enumerate(accepted):
        for b in accepted[i+1:]:
            if predictions[a]!=predictions[b]:
                first=next(t for t,(v,w) in enumerate(
                    zip(predictions[a],predictions[b],strict=True))
                    if v!=w)
                return {
                    "candidate_classes":[a,b],
                    "first_divergent_future_output_degree":first,
                    "publicly_accepted_both":True,
                    "same_complete_public_demos_and_query_input":True,
                    "candidate_outputs_different":True,
                }
    return None


def paired(records,first,second,primary_family="late_source"):
    pairs=[r for r in records if r["family"]==primary_family]
    gains=losses=0
    for r in pairs:
        left=r["arms"][first]["grade"]["future_correct"]
        right=r["arms"][second]["grade"]["future_correct"]
        gains+=right>left
        losses+=right<left
    return {"primary_distinct_public_inputs":len(pairs),
            "paired_gains":gains,"paired_losses":losses,
            "net":gains-losses,
            "two_sided_exact_case_sign_p":float(
                nor12b.exact_sign_p(gains,losses)),
            "family":primary_family}


def summarize_arm(records,kind):
    group=defaultdict(list)
    for r in records:
        group[r["family"]].append(r["arms"][kind])
    return {
        "cases":len(records),
        "future_correct":sum(r["arms"][kind]["grade"]["future_correct"]
                             for r in records),
        "false_accept":sum(r["arms"][kind]["grade"]["false_accept"]
                           for r in records),
        "public_checks":sum(r["arms"][kind]["public_checks"] for r in records),
        "public_query_prediction_candidates":sum(
            r["arms"][kind]["query_prediction_candidates"] for r in records),
        "public_unary_query_transforms":sum(
            r["arms"][kind]["public_unary_query_transforms"] for r in records),
        "by_family":{family:{
            "cases":len(vals),
            "future_correct":sum(v["grade"]["future_correct"] for v in vals),
            "false_accept":sum(v["grade"]["false_accept"] for v in vals),
            "public_checks":sum(v["public_checks"] for v in vals),
        } for family,vals in sorted(group.items())}
    }


def run(reps=REPS,late_per=LATE_PER_REP,
        signed_per=SIGNED_PER_REP,rational_per=RATIONAL_PER_REP):
    roster=nor11.sample_cohorts(nor12.SPLIT_COUNTS)
    assert sha256("\n".join(d for c,h,d in roster["train"]).encode()
                  ).hexdigest()==nor12.BASELINE_MANIFESTS["train"]
    assert sha256("\n".join(d for c,h,d in roster["dev"]).encode()
                  ).hexdigest()==nor12.BASELINE_MANIFESTS["dev"]
    ck,p1,p2=nor11.train_source()
    assert ck["parameters"]==1000
    assert len(p1)==len(p2)==model25.HIDDEN
    # train_source returns checkpoint and two frozen certified reference centers
    assert nor12.BASIN_SHA==CHECKPOINT_SHA
    cohorts=fresh_cohorts(roster,reps,late_per,signed_per,rational_per)
    out=[]
    cohort_summaries=[]
    for rep,batch in enumerate(cohorts):
        records=[single_case(case,head,ck) for case,head,digest in batch]
        assert all(r["model_input_sha256"]==digest
                   for r,(_,_,digest) in zip(records,batch,strict=True))
        out.extend(records)
        cohort_summaries.append({
            "replicate":rep,
            "cases":len(records),
            "public_demo_manifest_sha256":sha256("\n".join(
                r["public_demos_sha256"] for r in records).encode()).hexdigest(),
            "full_public_manifest_sha256":sha256("\n".join(
                r["full_public_input_sha256"] for r in records).encode()).hexdigest(),
            "original_future_correct":sum(
                r["arms"]["original_4"]["grade"]["future_correct"]
                for r in records),
            "consensus_four_future_correct":sum(
                r["arms"]["consensus_4"]["grade"]["future_correct"]
                for r in records),
            "full_mask_map_future_correct":sum(
                r["arms"]["full_mask_map_16"]["grade"]["future_correct"]
                for r in records),
        })
    late=[r for r in out if r["family"]=="late_source"]
    assert len(late)==late_per*reps
    assert len({r["model_input_sha256"] for r in late})==1
    assert len({r["public_demos_sha256"] for r in late})==len(late)
    assert len({r["full_public_input_sha256"] for r in late})==len(late)
    nonlate=[r for r in out if r["family"]!="late_source"]
    assert len({r["model_input_sha256"] for r in nonlate})==len(nonlate)
    assert not {r["model_input_sha256"] for r in late}&{
        r["model_input_sha256"] for r in nonlate}
    all_arms={k:summarize_arm(out,k) for k in ARMS}
    primary_comp={
        "consensus_vs_original":paired(out,"original_4","consensus_4"),
        "consensus_vs_hash":paired(out,"hash_4","consensus_4"),
        "full_mask_map_vs_full_mask_original":paired(
            out,"full_mask_original_16","full_mask_map_16"),
        "full_mask_map_vs_original":paired(out,"original_4","full_mask_map_16"),
    }
    criterion=all(primary_comp[key]["net"]>0 and
                  primary_comp[key]["two_sided_exact_case_sign_p"]<=PREREG_P
                  for key in ("consensus_vs_original","consensus_vs_hash"))
    collisions=sum(r["counterfactual_indistinguishable_pair"] is not None
                   for r in late)
    return {
        "schema":SCHEMA,
        "source_checkpoint_sha256":CHECKPOINT_SHA,
        "ninety_six_original_source_train_manifest":nor12.BASELINE_MANIFESTS["train"],
        "thirty_six_original_source_dev_manifest":nor12.BASELINE_MANIFESTS["dev"],
        "fresh_seed_offset":FRESH_OFFSET,
        "fresh_repetitions":reps,
        "primary_late_per_repetition":late_per,
        "secondary_signed_per_repetition":signed_per,
        "secondary_rational_per_repetition":rational_per,
        "fresh_cohorts":cohort_summaries,
        "public_query_input_degree":corpus.FUTURE,
        "public_demo_observation_degree":corpus.OBSERVED,
        "original_native_four_check_budget":PAID_BUDGET,
        "four_public_query_unary_ops_cached_per_case":len(corpus.OPS),
        "extra_full_mask_public_checks_per_case":N_CANDIDATES,
        "all_cases":len(out),
        "late_source_primary_public_distinct_inputs":len(late),
        "late_source_unique_gru_visible_input_tensors":1,
        "nonlate_secondary_distinct_gru_inputs":len(nonlate),
        "late_source_exact_public_masks":len({
            x["public_accept_mask"] for x in late}),
        "late_source_counterfactual_observation_collision_witnesses":collisions,
        "late_source_all_sixteen_public_accept_rate":sum(
            r["public_accept_mask"]==(1<<N_CANDIDATES)-1 for r in late)/len(late),
        "late_source_uniform_posterior_predictive_MAP_expected_success":sum(
            r["uniform_accepted_candidate_MAP_future_probability_num"]/
            r["uniform_accepted_candidate_MAP_future_probability_den"]
            for r in late),
        "arms":all_arms,
        "predeclared_primary_comparisons":primary_comp,
        "consensus_four_passed_two_control_requirement":criterion,
        "primary_Bonferroni_two_control_threshold":PREREG_P,
        "records":out,
        "scope":{
            "public_demo_coefficients_only_for_verifier":True,
            "query_input_degree_256_is_public":True,
            "withheld_target_output_only_seen_by_sealed_grader":True,
            "candidate_program_identity_never_read_in_policy":True,
            "query_simulation_costs_reported_separately":True,
            "full_mask_sixteen_checks_not_equal_cost_to_native_four":True,
            "old_gru_weights_and_original_ranker_frozen":True,
            "late_primary_grouped_by_distinct_complete_public_sources":True,
            "late_model_feature_is_one_aliased_input":True,
            "uniform_program_posterior_requires_uniform_prior_assumption":True,
            "counterfactual_pairs_are_synthetic_possible_futures":True,
            "no_retrospective_oracle_tuning_for_test":True,
            "nonlate_is_secondary_generalization_control":True
        }
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repetitions",type=int,default=REPS)
    p.add_argument("--late-per-repetition",type=int,default=LATE_PER_REP)
    p.add_argument("--signed-per-repetition",type=int,default=SIGNED_PER_REP)
    p.add_argument("--rational-per-repetition",type=int,default=RATIONAL_PER_REP)
    p.add_argument("--out",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    res=run(args.repetitions,args.late_per_repetition,
            args.signed_per_repetition,args.rational_per_repetition)
    if args.check:
        assert res["source_checkpoint_sha256"]==CHECKPOINT_SHA
        assert res["original_native_four_check_budget"]==4
        assert res["public_query_input_degree"]==256
        assert res["late_source_primary_public_distinct_inputs"]==(
            args.repetitions*args.late_per_repetition)
        assert res["nonlate_secondary_distinct_gru_inputs"]==(
            args.repetitions*(args.signed_per_repetition+
                              args.rational_per_repetition))
        assert res["all_cases"]==sum(
            (args.late_per_repetition+args.signed_per_repetition+
             args.rational_per_repetition) for _ in range(args.repetitions))
        assert all(x["public_checks"]<=4*res["all_cases"]
                   for k,x in res["arms"].items() if k.endswith("_4"))
        assert all(x["public_checks"]==16*res["all_cases"]
                   for k,x in res["arms"].items() if k.endswith("_16"))
    serialized=json.dumps(res,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(serialized)
    else:
        print(serialized,end="")


if __name__=="__main__":
    main()
