"""NOR-17: source-locked budgeted, sound public-output certification.

The RRC25 synthetic task publicly supplies two demonstration pairs through
degree 192 and query INPUT through 256. The query OUTPUT is withheld.
Under exact public interpreters and planted-program membership in the
complete 16-candidate template family, every candidate consistent with
the full public demonstrations is a possible generating world.

A short-prefix rejection is sound: a program disagreeing on the first
161 coefficients of either demonstration cannot pass a full 193-
coefficient exact verification. After screening, a budget-k certifier
tests at most k survivors against the full demonstrations. Negative full
checks eliminate those candidates. A full-accepted candidate is returned
ONLY when every still-possible candidate predicts the same hypothetical
output on the known query INPUT. Otherwise abstain, including after
observing two accepted candidates with different future predictions.

Training and source choice never consult the planted program or withheld
future query target. A separate sealed grader tests precommitted choices.
No candidate posterior probability is used as a proof of correctness.

Computational ledger for a screened arm:
  16 prefix candidate comparisons on 2 demos through degree 160;
  8 cached unary public demo transforms and 32 prefix compositions;
  4 cached unary public query transforms + 16 hypothetical outputs;
  <=4, <=8 or <=16 separately paid full public template checks.
Unscreened certificate controls pay no prefix checks, and original GRU
greedy arms pay no query-output or prefix simulation costs.

The source-only offline audit of all 16 full public checks is executed
AFTER frozen decisions, separately charged and never used by policies.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
import random
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for p in (ROOT,HERE):
    if str(p) not in sys.path:
        sys.path.insert(0,str(p))

from research.rrc_compactification import compound25 as corpus,model25
from research.nor11_attractor_controller import controller as nor11
from research.nor12_task_readout import readout as nor12
from research.nor12_task_readout import independent_confirmation as nor12b
from research.nor16_public_query_consensus import experiment as nor16

SCHEMA="nor17.risk_bounded_prefix_exclusion_public_output_certificate.v1"
CHECKPOINT_SHA=nor16.CHECKPOINT_SHA
FRESH_OFFSET=93317000
PREFIX_DEGREE=160
CANDIDATES=16
BUDGETS=(4,8,16)
REPETITIONS=4
LATE_PER=48
SIGNED_PER=12
RATIONAL_PER=12
FAMILIES=("late_source","signed","rational")
KINDS=(
    "greedy_4","greedy_8","greedy_16",
    "unscreened_4","unscreened_8","unscreened_16",
    "prefix_4","prefix_8","prefix_16",
    "prefix_original_4","prefix_hash_4",
)
PRIMARY=("late_source",)
PRIMARY_ALPHA=0.05/2
FULL_MASK_RETROSPECTIVE_CHECKS=16


def public_case_digest(case):
    return sha256(json.dumps(case.visible,separators=(",",":")).encode()).hexdigest()


def frozen_cohorts(roster,replicates=REPETITIONS,late_per=LATE_PER,
                   signed_per=SIGNED_PER,rational_per=RATIONAL_PER):
    if not (1<=replicates<=REPETITIONS and
            1<=late_per<=LATE_PER and
            1<=signed_per<=SIGNED_PER and
            1<=rational_per<=RATIONAL_PER):
        raise ValueError("NOR17 cohort budget exceeded")
    # NOR16 sample generator excludes NOR11-15B. Rebuilding all NOR16
    # source cases here additionally excludes the NOR16 test set itself.
    old=nor16.fresh_cohorts(roster,reps=nor16.REPS,
        late_per=nor16.LATE_PER_REP,
        signed_per=nor16.SIGNED_PER_REP,
        rational_per=nor16.RATIONAL_PER_REP)
    used_public={public_case_digest(c)
                 for split in roster.values() for c,h,d in split}
    used_nonlate={d for split in roster.values() for c,h,d in split}
    for group in old:
        for c,h,d in group:
            used_public.add(public_case_digest(c))
            if c.family!="late_source":
                used_nonlate.add(d)
    counts={"late_source":late_per,"signed":signed_per,"rational":rational_per}
    cohorts=[]
    for rep in range(replicates):
        batch=[]
        for fi,family in enumerate(FAMILIES):
            rng=random.Random(corpus.SEED+FRESH_OFFSET+
                              1000003*rep+10007*fi)
            found=0
            for trial in range(counts[family]*2500):
                case=corpus.make_case(rng,family,
                                     f"nor17-r{rep}-{family}-{trial:05d}")
                heads=corpus.possible_headers(case)
                if len(heads)!=1:
                    continue
                header=heads[0]
                public_hash=public_case_digest(case)
                digest=nor11.public_digest(case,header)
                if public_hash in used_public or (
                    family!="late_source" and digest in used_nonlate):
                    continue
                used_public.add(public_hash)
                if family!="late_source":
                    used_nonlate.add(digest)
                batch.append((case,header,digest))
                found+=1
                if found==counts[family]:
                    break
            if found!=counts[family]:
                raise RuntimeError(f"NOR17 source depleted {rep} {family}")
        cohorts.append(batch)
    return cohorts


def program(header,j):
    if not isinstance(j,int) or not 0<=j<CANDIDATES:
        raise ValueError("candidate index out of range")
    return (header.base,header.delay,header.spacing,header.amplitude,
            corpus.OPS[j//4],corpus.OPS[j%4])


def candidate_outputs(source,header,degree):
    """Exact public candidate construction with four cached unary maps."""
    if degree<max(corpus.DELAYS) or degree>=len(source):
        raise ValueError("candidate source prefix too short")
    f=source[:degree+1]
    transformations={op:corpus.d.operate(f,op) for op in corpus.OPS}
    base=transformations[header.base]
    k=header.delay;delta=header.spacing;c=header.amplitude
    result=[]
    for j in range(CANDIDATES):
        u,v=corpus.OPS[j//4],corpus.OPS[j%4]
        y=list(base)
        y[k]+=c
        for pos in range(k+delta,degree+1):
            y[pos]+=c*transformations[u][pos-k-delta]
        for pos in range(k+2*delta,degree+1):
            y[pos]+=c*transformations[v][pos-k-2*delta]
        result.append(tuple(y))
    return result


def prefix_mask(case,header,degree=PREFIX_DEGREE):
    """Reject via exact public prefix mismatches; no future query access."""
    if not max(corpus.DELAYS)<=degree<=corpus.OBSERVED:
        raise ValueError("prefix degree must include all possible K")
    survivors=(1<<CANDIDATES)-1
    for source,target in case.demos:
        predictions=candidate_outputs(source,header,degree)
        observed=target[:degree+1]
        matches=sum(1<<j for j,prediction in enumerate(predictions)
                    if prediction==observed)
        survivors &= matches
    if survivors==0:
        raise AssertionError("public generator produced zero prefix matches")
    return survivors


def accepted_full(case,header,j):
    """One *paid* exact full public demonstration check on two demos."""
    return bool(corpus.verify_template(
        case,header,corpus.OPS[j//4],corpus.OPS[j%4]))


def original_order(case,header,ck):
    scores=model25.logits(case,header,ck)
    return sorted(range(CANDIDATES),key=lambda j:(-scores[j],j))


def consensus_order(viable_mask,predictions,original):
    """Rank by public-predicted future class multiplicity among survivors."""
    if viable_mask==0:
        raise ValueError("empty sound candidate universe")
    counts=Counter(predictions[j] for j in range(CANDIDATES)
                   if viable_mask&(1<<j))
    return sorted((j for j in range(CANDIDATES) if viable_mask&(1<<j)),
                  key=lambda j:(-counts[predictions[j]],original.index(j)))


def hash_order(case,viable_mask):
    seed=int(sha256(json.dumps(case.query,separators=(",",":")).encode()
                 ).hexdigest()[:16],16)
    rng=random.Random(seed)
    out=list(range(CANDIDATES))
    rng.shuffle(out)
    return [j for j in out if viable_mask&(1<<j)]


def policy_greedy(case,header,order,budget):
    """Original GRU first-positive verifier, no risk-controlled abstention."""
    if budget not in BUDGETS:
        raise ValueError("invalid frozen verification budget")
    tested=[]
    accepted=None
    for j in order[:budget]:
        passed=accepted_full(case,header,j)
        tested.append({"class":j,"accepted":passed})
        if passed:
            accepted=j
            break
    return {
        "certified":False,
        "chosen_class":accepted,
        "public_checks":len(tested),
        "trace":tested,
        "initial_viable_mask":(1<<CANDIDATES)-1,
        "remaining_viable_mask":None,
        "confirmed_accepted_classes":[accepted] if accepted is not None else [],
        "exit":"first_public_accept" if accepted is not None else "budget_exhausted",
        "candidate_query_compositions":0,
        "query_unary_transformations":0,
        "prefix_checks":0,
        "prefix_demo_compositions":0,
        "prefix_unary_transformations":0,
        "full_output_unanimity":None,
    }


def certifier(case,header,predictions,order,budget,initial_mask,
              use_prefix=False):
    """Risk-controlled candidate selection, no program/future data.

    Full rejection removes a candidate from the possible-generators set.
    Full acceptance confirms a candidate program, but it is returned only
    after the remaining set's public query predictions are unanimous.
    """
    if budget not in BUDGETS or not 0<initial_mask<1<<CANDIDATES:
        raise ValueError("unsupported certifier budget or empty universe")
    if len(predictions)!=CANDIDATES:
        raise ValueError("all sixteen public hypothetical futures required")
    if len(order)!=initial_mask.bit_count() or set(order)!={
        j for j in range(CANDIDATES) if initial_mask&(1<<j)}:
        raise ValueError("ranking must contain each soundly viable class once")
    possible=initial_mask
    checked=[]
    positives=[]
    chosen=None
    status="budget_exhausted"
    for j in order[:budget]:
        passed=accepted_full(case,header,j)
        checked.append({"class":j,"accepted":passed})
        if passed:
            positives.append(j)
        else:
            possible &= ~(1<<j)
        if len({predictions[k] for k in positives})>1:
            status="incompatible_full_accepted_outputs"
            break
        if positives:
            predicted_set={predictions[k] for k in range(CANDIDATES)
                           if possible&(1<<k)}
            if len(predicted_set)==1:
                status="complete_public_output_certificate"
                chosen=positives[0]
                break
    if not positives and len(checked)==len(order):
        status="no_full_accepted_candidate"
    if not positives and len(checked)==budget:
        status="budget_exhausted_unverified"
    if chosen is not None:
        assert possible&(1<<chosen)
        assert len({predictions[k] for k in range(CANDIDATES)
                    if possible&(1<<k)})==1
    return {
        "certified":chosen is not None,
        "chosen_class":chosen,
        "public_checks":len(checked),
        "trace":checked,
        "initial_viable_mask":initial_mask,
        "remaining_viable_mask":possible,
        "confirmed_accepted_classes":positives,
        "exit":status,
        "candidate_query_compositions":CANDIDATES,
        "query_unary_transformations":len(corpus.OPS),
        "prefix_checks":CANDIDATES if use_prefix else 0,
        "prefix_demo_compositions":2*CANDIDATES if use_prefix else 0,
        "prefix_unary_transformations":2*len(corpus.OPS) if use_prefix else 0,
        "full_output_unanimity":(
            len({predictions[k] for k in range(CANDIDATES)
                 if possible&(1<<k)})==1
        ),
    }


def policies_before_grading(case,header,ck):
    """All policy proposals frozen before the first withheld future read."""
    original=original_order(case,header,ck)
    predictions=candidate_outputs(case.query,header,corpus.FUTURE)
    prefix=prefix_mask(case,header,PREFIX_DEGREE)
    unscreened=(1<<CANDIDATES)-1
    orders={
        "unscreened":consensus_order(unscreened,predictions,original),
        "prefix":consensus_order(prefix,predictions,original),
        "prefix_original":[j for j in original if prefix&(1<<j)],
        "prefix_hash":hash_order(case,prefix),
    }
    arms={}
    for budget in BUDGETS:
        arms[f"greedy_{budget}"]=policy_greedy(
            case,header,original,budget)
        arms[f"unscreened_{budget}"]=certifier(
            case,header,predictions,orders["unscreened"],
            budget,unscreened,use_prefix=False)
        arms[f"prefix_{budget}"]=certifier(
            case,header,predictions,orders["prefix"],
            budget,prefix,use_prefix=True)
    arms["prefix_original_4"]=certifier(
        case,header,predictions,orders["prefix_original"],
        4,prefix,use_prefix=True)
    arms["prefix_hash_4"]=certifier(
        case,header,predictions,orders["prefix_hash"],
        4,prefix,use_prefix=True)
    assert set(arms)==set(KINDS)
    return predictions,prefix,original,arms


def grade_frozen_choice(case,header,choice):
    """Sole function allowed to call true withheld output evaluation."""
    return corpus.grade(case,program(header,choice) if choice is not None
                        else None)


def single_case(case,header,ck):
    predictions,prefix,original,arms=policies_before_grading(case,header,ck)
    # No grader has executed before every arm's chosen class is fixed.
    outcomes={}
    for name,policy in arms.items():
        grade=grade_frozen_choice(case,header,policy["chosen_class"])
        if policy["certified"] and grade["future_correct"]!=1:
            raise AssertionError("claimed complete certificate failed sealed truth")
        outcomes[name]={**policy,"grade":grade}
    # Additional SOURCE-ONLY retrospective 16-check audit, excluded from
    # each policy's paid inference budget and candidate decisions.
    full=sum(1<<j for j in range(CANDIDATES)
             if accepted_full(case,header,j))
    assert full and (full&prefix)==full
    for name,arm in outcomes.items():
        if arm["remaining_viable_mask"] is not None:
            assert (full&arm["remaining_viable_mask"])==full
        assert arm["public_checks"]<=int(name.rsplit("_",1)[-1])
        assert arm["grade"]["accepted"]==(
            arm["grade"]["future_correct"]+arm["grade"]["false_accept"])
    return {
        "family":case.family,
        "public_demos_sha256":nor16.visible_source_digest(case),
        "full_public_input_sha256":nor16.full_public_digest(case),
        "model_input_sha256":nor11.public_digest(case,header),
        "public_query_input_sha256":sha256(json.dumps(
            case.query,separators=(",",":")).encode()).hexdigest(),
        "original_top_four":original[:4],
        "full_public_accept_mask_retrospective":full,
        "short_prefix_survivor_mask":prefix,
        "initial_possible_generator_count":prefix.bit_count(),
        "hypothetical_query_output_hashes":[sha256(json.dumps(
            p,separators=(",",":")).encode()).hexdigest() for p in predictions],
        "full_public_future_output_identifiable":len({
            predictions[j] for j in range(CANDIDATES) if full&(1<<j)})==1,
        "short_prefix_future_output_identifiable":len({
            predictions[j] for j in range(CANDIDATES) if prefix&(1<<j)})==1,
        "original_public_accepted_count":full.bit_count(),
        "retrospective_public_full_checks":FULL_MASK_RETROSPECTIVE_CHECKS,
        "arms":outcomes,
    }


def paired(records,left,right,family="late_source",key="certified"):
    selected=[x for x in records if x["family"]==family]
    a=lambda r: r["arms"][left][key] if key=="certified" else r["arms"][left]["grade"]["future_correct"]
    b=lambda r: r["arms"][right][key] if key=="certified" else r["arms"][right]["grade"]["future_correct"]
    gain=sum(b(r)>a(r) for r in selected)
    loss=sum(b(r)<a(r) for r in selected)
    return {
        "family":family,
        "cases":len(selected),
        "gains":gain,"losses":loss,"net":gain-loss,
        "exact_unadjusted_p":float(nor12b.exact_sign_p(gain,loss)),
        "comparison_measure":key,
    }


def summarize(records,name):
    selected=[r["arms"][name] for r in records]
    primary=[r["arms"][name] for r in records if r["family"]=="late_source"]
    family_names=FAMILIES
    def stats(rows):
        return {
            "cases":len(rows),
            "returned_proposals":sum(x["chosen_class"] is not None for x in rows),
            "certificates":sum(x["certified"] for x in rows),
            "future_correct":sum(x["grade"]["future_correct"] for x in rows),
            "false_accept":sum(x["grade"]["false_accept"] for x in rows),
            "full_public_checks":sum(x["public_checks"] for x in rows),
            "prefix_candidate_checks":sum(x["prefix_checks"] for x in rows),
            "prefix_demo_candidate_compositions":sum(
                x["prefix_demo_compositions"] for x in rows),
            "prefix_unary_maps":sum(
                x["prefix_unary_transformations"] for x in rows),
            "query_candidate_compositions":sum(
                x["candidate_query_compositions"] for x in rows),
            "query_unary_maps":sum(
                x["query_unary_transformations"] for x in rows),
            "by_exit":dict(Counter(x["exit"] for x in rows)),
        }
    all_stats=stats(selected)
    return {
        **all_stats,
        "late_source":stats(primary),
        "by_family":{family:stats([r["arms"][name]
                  for r in records if r["family"]==family])
                     for family in family_names}
    }


def run(repetitions=REPETITIONS,late_per=LATE_PER,
        signed_per=SIGNED_PER,rational_per=RATIONAL_PER):
    ck,point1,point2=nor11.train_source()
    assert ck["parameters"]==1000
    assert len(point1)==len(point2)==model25.HIDDEN
    assert nor12.BASIN_SHA==CHECKPOINT_SHA
    original=nor11.sample_cohorts(nor12.SPLIT_COUNTS)
    assert sha256("\n".join(x[2] for x in original["train"]).encode()
                  ).hexdigest()==nor12.BASELINE_MANIFESTS["train"]
    assert sha256("\n".join(x[2] for x in original["dev"]).encode()
                  ).hexdigest()==nor12.BASELINE_MANIFESTS["dev"]
    cohorts=frozen_cohorts(original,repetitions,late_per,signed_per,rational_per)
    records=[]
    replicate_summaries=[]
    for rep,batch in enumerate(cohorts):
        subset=[single_case(case,header,ck) for case,header,digest in batch]
        assert all(r["model_input_sha256"]==d
                   for r,(_,_,d) in zip(subset,batch,strict=True))
        records.extend(subset)
        replicate_summaries.append({
            "replicate":rep,
            "cases":len(subset),
            "public_manifest_sha256":sha256("\n".join(
                r["full_public_input_sha256"] for r in subset).encode()).hexdigest(),
            "primary_certificate_by_budget":{
                str(b):sum(r["arms"][f"prefix_{b}"]["certified"]
                           for r in subset if r["family"]=="late_source")
                for b in BUDGETS
            },
            "primary_greedy_correct_by_budget":{
                str(b):sum(r["arms"][f"greedy_{b}"]["grade"]["future_correct"]
                           for r in subset if r["family"]=="late_source")
                for b in BUDGETS
            }
        })
    late=[r for r in records if r["family"]=="late_source"]
    nonlate=[r for r in records if r["family"]!="late_source"]
    assert len(late)==late_per*repetitions
    assert len({r["full_public_input_sha256"] for r in records})==len(records)
    assert len({r["model_input_sha256"] for r in late})==1
    assert len({r["model_input_sha256"] for r in nonlate})==len(nonlate)
    assert not ({r["model_input_sha256"] for r in late}&
                {r["model_input_sha256"] for r in nonlate})
    outcomes={kind:summarize(records,kind) for kind in KINDS}
    # Certificate coverage must be monotonically nondecreasing with paid
    # verifier budget under identical ranking and initial public filter.
    for family in FAMILIES:
        for label in ("prefix","unscreened"):
            counts=[outcomes[f"{label}_{b}"]["by_family"][family][
                "certificates"] for b in BUDGETS]
            assert counts==sorted(counts),(family,label,counts)
    for kind,entry in outcomes.items():
        if not kind.startswith("greedy"):
            assert entry["false_accept"]==0
    primary_tests={
        "prefix4_vs_unscreened4":paired(
            records,"unscreened_4","prefix_4"),
        "prefix4_vs_prefix_original4":paired(
            records,"prefix_original_4","prefix_4"),
        "prefix4_vs_prefix_hash4":paired(
            records,"prefix_hash_4","prefix_4"),
        "prefix8_vs_prefix4":paired(records,"prefix_4","prefix_8"),
        "prefix16_vs_prefix8":paired(records,"prefix_8","prefix_16"),
        "prefix16_vs_unscreened16":paired(records,"unscreened_16","prefix_16"),
        "greedy4_vs_prefix4_future_correct":paired(
            records,"prefix_4","greedy_4",key="future_correct"),
    }
    criterion=all(primary_tests[key]["net"]>0 and
                  primary_tests[key]["exact_unadjusted_p"]<=PRIMARY_ALPHA
                  for key in ("prefix4_vs_unscreened4",
                              "prefix4_vs_prefix_original4"))
    return {
        "schema":SCHEMA,
        "source_checkpoint_sha256":CHECKPOINT_SHA,
        "frozen_train_manifest":nor12.BASELINE_MANIFESTS["train"],
        "frozen_dev_manifest":nor12.BASELINE_MANIFESTS["dev"],
        "fresh_seed_offset":FRESH_OFFSET,
        "repetitions":repetitions,
        "late_source_per_repetition":late_per,
        "signed_per_repetition":signed_per,
        "rational_per_repetition":rational_per,
        "source_public_prefix_degree":PREFIX_DEGREE,
        "demo_public_degree":corpus.OBSERVED,
        "public_query_input_degree":corpus.FUTURE,
        "policy_budgets":list(BUDGETS),
        "arm_names":list(KINDS),
        "all_cases":len(records),
        "primary_distinct_complete_public_late_cases":len(late),
        "primary_unique_gru_input_groups":1,
        "nonlate_distinct_gru_input_groups":len(nonlate),
        "retrospective_full_source_audit_checks":len(records)*CANDIDATES,
        "cohorts":replicate_summaries,
        "arms":outcomes,
        "primary_paired_tests":primary_tests,
        "preregistered_two_control_alpha":PRIMARY_ALPHA,
        "prefix_rank4_passed_two_control_criterion":criterion,
        "source_wide_certified_future_count":sum(
            r["short_prefix_future_output_identifiable"] for r in records),
        "primary_full_public_future_output_identifiable":sum(
            r["full_public_future_output_identifiable"] for r in late),
        "primary_short_prefix_future_output_identifiable":sum(
            r["short_prefix_future_output_identifiable"] for r in late),
        "records":records,
        "scope":{
            "all_generator_members_in_enumerated_candidate_family":True,
            "certificates_require_one_exact_full_public_accepted_candidate":True,
            "every_non_rejected_public_candidate_must_predict_same_output":True,
            "prefix_only_elimination_is_exact_and_sound":True,
            "offline_full16_public_mask_audit_after_all_policy_choices":True,
            "offline_audit_cost_separate_from_inference":True,
            "private_future_used_only_by_sealed_grade_after_choice":True,
            "original_1000_param_gru_and_ranker_immutable":True,
            "model_visible_late_inputs_aliased_but_complete_public_distinct":True,
            "prefix_filter_unary_and_composition_costs_reported":True,
            "query_unary_and_composition_costs_reported":True,
            "no_general_theorem_prover_soundness_claim":True,
            "no_adjustment_to_heldout_policy_or_source_seed":True,
        }
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repetitions",type=int,default=REPETITIONS)
    p.add_argument("--late-per-repetition",type=int,default=LATE_PER)
    p.add_argument("--signed-per-repetition",type=int,default=SIGNED_PER)
    p.add_argument("--rational-per-repetition",type=int,default=RATIONAL_PER)
    p.add_argument("--out",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    result=run(args.repetitions,args.late_per_repetition,
               args.signed_per_repetition,args.rational_per_repetition)
    if args.check:
        assert result["source_checkpoint_sha256"]==CHECKPOINT_SHA
        assert result["all_cases"]==args.repetitions*(
            args.late_per_repetition+args.signed_per_repetition+
            args.rational_per_repetition)
        assert result["primary_distinct_complete_public_late_cases"]==(
            args.repetitions*args.late_per_repetition)
        assert result["nonlate_distinct_gru_input_groups"]==(
            args.repetitions*(args.signed_per_repetition+
                              args.rational_per_repetition))
        assert all(result["arms"][k]["false_accept"]==0
                   for k in KINDS if not k.startswith("greedy"))
        for k in KINDS:
            budget=int(k.rsplit("_",1)[-1])
            assert result["arms"][k]["full_public_checks"]<=(
                budget*result["all_cases"])
    rendered=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(rendered)
    else:
        print(rendered,end="")


if __name__=="__main__":
    main()
