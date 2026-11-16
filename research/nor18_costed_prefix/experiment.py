"""NOR-18: common-budget arithmetic-work frontier for public proof certificates.

The entire experiment is preregistered in README.md before opening the new
held-out NOR-18 source generator. Models, ranks, candidate family and exact
public verifier semantics are frozen from RRC25 / NOR17.

Costs are a transparent count-proxy for coefficient operations, NOT a CPU
time, big-integer bit-complexity, or hardware-independent speedup claim.
No private program identity / future target may be used by any policy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from research.rrc_compactification import compound25 as corpus
from research.nor11_attractor_controller import controller as nor11
from research.nor12_task_readout import readout as nor12
from research.nor12_task_readout import independent_confirmation as nor12b
from research.nor13_gated_readout import interaction as nor13
from research.nor14_conditional_basin import experiment as nor14
from research.nor15_trajectory_sufficiency import task_test as nor15b
from research.nor16_public_query_consensus import experiment as nor16
from research.nor17_prefix_certificates import experiment as nor17

SCHEMA="nor18.common_work_cached_prefix_certificate_frontier.v1"
CK_SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"
FRESH_OFFSET=93518000
REPETITIONS=4
LATE_PER=48
SIGNED_PER=12
RATIONAL_PER=12
FAMILIES=("late_source","signed","rational")
PREFIX_DEGREES=(112,128,144,160,176)
BATCH_DEGREE=192
WORK_MULTIPLIERS=(4,8,16)
N_CLASSES=16
PRIMARY_ALPHA=.025

ARM_NAMES=(
    tuple(f"greedy_{budget}" for budget in WORK_MULTIPLIERS)+
    tuple(f"none_{budget}" for budget in WORK_MULTIPLIERS)+
    tuple(f"d{degree}_{budget}" for degree in PREFIX_DEGREES
          for budget in WORK_MULTIPLIERS)+
    tuple(f"batch192_{budget}" for budget in WORK_MULTIPLIERS)+
    ("d160_original_4","d160_hash_4","d176_original_4","d176_hash_4")
)


def public_visible_sha(case):
    return sha256(json.dumps(case.visible,separators=(",",":")).encode()).hexdigest()


def fresh_cohorts(roster,replicates=REPETITIONS,late_per=LATE_PER,
                  signed_per=SIGNED_PER,rational_per=RATIONAL_PER):
    if not (1<=replicates<=REPETITIONS and 1<=late_per<=LATE_PER
            and 1<=signed_per<=SIGNED_PER and 1<=rational_per<=RATIONAL_PER):
        raise ValueError("NOR18 source budget exceeded")
    # Independently regenerate prior cohorts for explicit source exclusion.
    prior=(
        nor12b.fresh_cohorts(roster,replicates=nor12b.REPETITIONS,
                            per_family=nor12b.CASES_PER_FAMILY)
        + nor13.new_heldout(roster,replicates=nor13.CONFIRMATIONS,
                            per_family=nor13.FAMILY_PER)
        + nor14.fresh_cohorts(roster,repetitions=nor14.REPETITIONS,
                              per_family=nor14.PER_FAMILY)
        + nor15b.source_disjoint_new(roster)
        + nor16.fresh_cohorts(roster)
        + nor17.frozen_cohorts(roster)
    )
    known_visible={public_visible_sha(case)
                   for batch in roster.values() for case,h,d in batch}
    known_nonlate={d for batch in roster.values() for c,h,d in batch}
    for batch in prior:
        for case,h,d in batch:
            known_visible.add(public_visible_sha(case))
            if case.family!="late_source":
                known_nonlate.add(d)
    sizes={"late_source":late_per,"signed":signed_per,"rational":rational_per}
    out=[]
    for rep in range(replicates):
        group=[]
        for fi,family in enumerate(FAMILIES):
            rng=random.Random(corpus.SEED+FRESH_OFFSET+
                              rep*1000003+fi*10007)
            found=0
            for trial in range(sizes[family]*2500):
                case=corpus.make_case(
                    rng,family,f"nor18-r{rep}-{family}-{trial:05d}")
                headers=corpus.possible_headers(case)
                if len(headers)!=1:
                    continue
                header=headers[0]
                digest=nor11.public_digest(case,header)
                vis=public_visible_sha(case)
                if vis in known_visible or (
                        family!="late_source" and digest in known_nonlate):
                    continue
                known_visible.add(vis)
                if family!="late_source":
                    known_nonlate.add(digest)
                group.append((case,header,digest))
                found+=1
                if found==sizes[family]:
                    break
            if found!=sizes[family]:
                raise RuntimeError(f"source family exhausted {rep}/{family}")
        out.append(group)
    return out


def unary_work(op,degree):
    if degree<0:
        raise ValueError("negative degree")
    if op=="R":
        return degree*(degree+1)//2
    if op in ("I","N","S"):
        return degree+1
    raise ValueError("unknown public operation")


def composition_work(header,degree,compare=True):
    """One candidate's public combination and optional output equality."""
    if degree<header.delay:
        raise ValueError("prefix earlier than the inferred first shift")
    count_u=max(0,degree-header.delay-header.spacing+1)
    count_v=max(0,degree-header.delay-2*header.spacing+1)
    return ((degree+1)+1+2*(count_u+count_v)
            + ((degree+1) if compare else 0))


def full_check_work(header,j,degree=corpus.OBSERVED):
    if not 0<=j<16:
        raise ValueError("class outside enumerated candidate family")
    u=corpus.OPS[j//4]
    v=corpus.OPS[j%4]
    return 2*(unary_work(header.base,degree)+unary_work(u,degree)
              +unary_work(v,degree)+composition_work(header,degree,True))


def cached_screen_work(header,degree):
    if not max(corpus.DELAYS)<=degree<=corpus.OBSERVED:
        raise ValueError("unregistered screen degree")
    return 2*(sum(unary_work(op,degree) for op in corpus.OPS)
              +N_CLASSES*composition_work(header,degree,True))


def query_simulation_work(header):
    degree=corpus.FUTURE
    return (sum(unary_work(op,degree) for op in corpus.OPS)
            + N_CLASSES*composition_work(header,degree,False))


def baseline_unit(header):
    """Frozen public-header-only unit: ceil(mean independent full check)."""
    return (sum(full_check_work(header,j) for j in range(N_CLASSES))
            +N_CLASSES-1)//N_CLASSES


def cost_register(header):
    checks=[full_check_work(header,j) for j in range(N_CLASSES)]
    screens={str(d):cached_screen_work(header,d) for d in
             PREFIX_DEGREES+(BATCH_DEGREE,)}
    unit=baseline_unit(header)
    assert unit==(sum(checks)+15)//16
    return {"full_check_costs":checks,
            "screen_setup_costs":screens,
            "query_simulation_cost":query_simulation_work(header),
            "public_full_check_unit":unit,
            "reference_unit_uses_public_header_only":True,
            "work_formula_version":"nor18.coefficient_operation_proxy.v1"}


def ordered_candidates(mask,predictions,original,kind,case=None):
    allowed=[j for j in range(N_CLASSES) if mask&(1<<j)]
    if kind=="consensus":
        return nor17.consensus_order(mask,predictions,original)
    if kind=="original":
        return [j for j in original if mask&(1<<j)]
    if kind=="hash":
        assert case is not None
        return nor17.hash_order(case,mask)
    raise ValueError("unknown ranking policy")


def decide_budgeted(case,header,public_checker,predictions,original,
                    costs,kind,degree,multiplier,prefix_survivors):
    """One budgeted independent policy. Only queries charged public checks.

    The checker is an exact public-demo membership oracle. It never consults
    the held-out future output; response lookup is allowed ONLY after the
    proposed full check's work is within the remaining budget.
    """
    if multiplier not in WORK_MULTIPLIERS:
        raise ValueError("unregistered work cap")
    if kind not in ("greedy","consensus","original","hash","batch"):
        raise ValueError("unregistered policy")
    if degree is not None and degree not in PREFIX_DEGREES+(BATCH_DEGREE,):
        raise ValueError("unregistered prefix degree")
    if kind=="batch" and degree!=BATCH_DEGREE:
        raise ValueError("batch only available at complete public degree")
    if len(predictions)!=16 or len(original)!=16:
        raise ValueError("full candidate and ranking universe required")

    work_cap=multiplier*costs["public_full_check_unit"]
    query_charge=0 if kind=="greedy" else costs["query_simulation_cost"]
    screen_charge=(costs["screen_setup_costs"][str(degree)]
                   if degree is not None else 0)
    setup=query_charge+screen_charge

    def receipt(status,chosen=None,remaining_mask=None,
                charges=0,trace=None,full_charge=0):
        trace=[] if trace is None else trace
        total=charges+full_charge
        paid=len(trace)
        stage_executed=charges>0
        return {
            "work_multiplier":multiplier,
            "reference_full_check_unit":costs["public_full_check_unit"],
            "work_cap":work_cap,
            "total_work":total,
            "query_work":query_charge if stage_executed else 0,
            "screen_work":screen_charge if stage_executed else 0,
            "full_candidate_check_work":full_charge,
            "full_candidate_checks":paid,
            "batched_degree192_public_comparisons":(
                16 if stage_executed and kind=="batch" else 0),
            "query_public_candidate_outputs":(
                16 if stage_executed and kind!="greedy" else 0),
            "query_public_unary_maps":(
                4 if stage_executed and kind!="greedy" else 0),
            "screen_public_comparisons":(
                16 if stage_executed and degree is not None else 0),
            "screen_public_demo_compositions":(
                32 if stage_executed and degree is not None else 0),
            "screen_public_unary_maps":(
                8 if stage_executed and degree is not None else 0),
            "screen_degree":degree,
            "setup_feasible":setup<=work_cap,
            "initial_viable_mask":prefix_survivors if stage_executed
                and degree is not None else 65535,
            "remaining_viable_mask":remaining_mask,
            "certified":chosen is not None and kind!="greedy",
            "chosen_class":chosen,
            "trace":trace,
            "exit":status,
            "ranking":"original" if kind=="greedy" else kind,
            "early_abstention_work_limited":status in
                ("setup_over_work_cap","next_check_over_work_cap"),
        }

    if kind=="greedy":
        assert degree is None and prefix_survivors==65535
    if setup>work_cap:
        return receipt("setup_over_work_cap")

    assert setup>=0
    if kind=="batch":
        mask=prefix_survivors
        assert mask>0
        accepted=[j for j in original if mask&(1<<j)]
        outputs={predictions[j] for j in accepted}
        if len(outputs)==1:
            return receipt("certified_batched_full_public",
                chosen=accepted[0],remaining_mask=mask,charges=setup)
        return receipt("ambiguous_batched_full_public",
                       remaining_mask=mask,charges=setup)

    viable=65535 if degree is None else prefix_survivors
    if not viable:
        raise AssertionError("sound screen produced empty possible set")
    order=(list(original) if kind=="greedy"
           else ordered_candidates(viable,predictions,original,kind,case))
    charge=0
    checked=[]
    confirmed=[]
    chosen=None
    status="all_viable_checked"
    remaining=viable
    for j in order:
        price=costs["full_check_costs"][j]
        if setup+charge+price>work_cap:
            status="next_check_over_work_cap"
            break
        accepted=bool(public_checker(j))
        charge+=price
        checked.append({"class":j,"accepted":accepted,"charged_work":price})
        if kind=="greedy":
            if accepted:
                chosen=j
                status="first_public_accept"
                break
            continue
        if accepted:
            confirmed.append(j)
        else:
            remaining &=~(1<<j)
        if len({predictions[k] for k in confirmed})>1:
            status="incompatible_full_accepted_outputs"
            break
        if confirmed and len({predictions[k] for k in range(16)
                              if remaining&(1<<k)})==1:
            chosen=confirmed[0]
            status="certified_unanimous_remaining"
            break

    if not checked and not order:
        status="no_viable_candidates"  # never reachable on planted positive case
    if chosen is not None and kind!="greedy":
        assert remaining&(1<<chosen)
        assert len({predictions[k] for k in range(16)
                    if remaining&(1<<k)})==1
    return receipt(status,chosen=chosen,remaining_mask=(
        None if kind=="greedy" else remaining),
        charges=setup,trace=checked,full_charge=charge)


def policies_before_grading(case,header,ck):
    """All policies select or abstain using only public source and checker."""
    original=nor17.original_order(case,header,ck)
    predictions=nor17.candidate_outputs(case.query,header,corpus.FUTURE)
    screens={d:nor17.prefix_mask(case,header,d)
             for d in PREFIX_DEGREES+(BATCH_DEGREE,)}
    register=cost_register(header)
    assert all(0<mask<65536 for mask in screens.values())
    # This cache is private to the *public exact-checker closure*. Each arm
    # is charged for every logical lookup. It never exposes the full mask.
    public_cache={}
    def checked_public(j):
        if j not in public_cache:
            public_cache[j]=nor17.accepted_full(case,header,j)
        return public_cache[j]
    decisions={}
    for budget in WORK_MULTIPLIERS:
        decisions[f"greedy_{budget}"]=decide_budgeted(
            case,header,checked_public,predictions,original,register,
            "greedy",None,budget,65535)
        decisions[f"none_{budget}"]=decide_budgeted(
            case,header,checked_public,predictions,original,register,
            "consensus",None,budget,65535)
        for depth in PREFIX_DEGREES:
            decisions[f"d{depth}_{budget}"]=decide_budgeted(
                case,header,checked_public,predictions,original,register,
                "consensus",depth,budget,screens[depth])
        decisions[f"batch192_{budget}"]=decide_budgeted(
            case,header,checked_public,predictions,original,register,
            "batch",BATCH_DEGREE,budget,screens[BATCH_DEGREE])
    for depth in (160,176):
        for kind in ("original","hash"):
            decisions[f"d{depth}_{kind}_4"]=decide_budgeted(
                case,header,checked_public,predictions,original,register,
                kind,depth,4,screens[depth])
    assert set(decisions)==set(ARM_NAMES)
    # Retro source audit after policies frozen. Charge as separate research.
    fullmask=sum(1<<j for j in range(16) if checked_public(j))
    assert fullmask==screens[192]
    assert all((fullmask&screen)==fullmask for screen in screens.values())
    for name,choice in decisions.items():
        assert choice["total_work"]<=choice["work_cap"]
        if choice["chosen_class"] is not None:
            assert fullmask&(1<<choice["chosen_class"])
        if choice["certified"]:
            truth={predictions[j] for j in range(16) if fullmask&(1<<j)}
            assert len(truth)==1
            assert predictions[choice["chosen_class"]] in truth
    return predictions,screens,register,original,decisions,fullmask


def sealed_grade(case,header,j):
    """Sole function permitted to consult withheld future query output."""
    return corpus.grade(case,nor17.program(header,j) if j is not None else None)


def single_case(case,header,ck):
    predictions,screens,costs,original,decisions,fullmask=policies_before_grading(
        case,header,ck)
    # All choices frozen before first true-future comparison.
    grade_cache={}
    for policy in decisions.values():
        chosen=policy["chosen_class"]
        if chosen not in grade_cache:
            grade_cache[chosen]=sealed_grade(case,header,chosen)
        policy["grade"]=grade_cache[chosen]
        if policy["certified"] and policy["grade"]["future_correct"]!=1:
            raise AssertionError("invalid complete-public-future certificate")
    return {
        "family":case.family,
        "public_demos_sha256":nor16.visible_source_digest(case),
        "full_public_input_sha256":nor16.full_public_digest(case),
        "model_visible_sha256":nor11.public_digest(case,header),
        "public_query_input_sha256":sha256(json.dumps(
            case.query,separators=(",",":")).encode()).hexdigest(),
        "header":{"base":header.base,"delay":header.delay,
                  "spacing":header.spacing,"amplitude":header.amplitude},
        "cost_register":costs,
        "public_query_output_prediction_sha256":[sha256(json.dumps(
            x,separators=(",",":")).encode()).hexdigest()
            for x in predictions],
        "public_prefix_survivor_masks":{
            str(k):v for k,v in screens.items()},
        "retro_full_accepted_mask":fullmask,
        "retro_public_candidate_checks":16,
        "certificate_possible_from_complete_public_mask":(
            len({predictions[j] for j in range(16) if fullmask&(1<<j)})==1),
        "arms":decisions,
    }


def summarize(records,arm):
    chosen=[r["arms"][arm] for r in records]
    def tally(cases):
        return {
            "cases":len(cases),
            "returned_candidates":sum(x["chosen_class"] is not None for x in cases),
            "certificates":sum(x["certified"] for x in cases),
            "future_correct":sum(x["grade"]["future_correct"] for x in cases),
            "false_accept":sum(x["grade"]["false_accept"] for x in cases),
            "total_work":sum(x["total_work"] for x in cases),
            "query_work":sum(x["query_work"] for x in cases),
            "screen_work":sum(x["screen_work"] for x in cases),
            "full_check_work":sum(x["full_candidate_check_work"] for x in cases),
            "full_check_calls":sum(x["full_candidate_checks"] for x in cases),
            "batch_full_public_comparisons":sum(
                x["batched_degree192_public_comparisons"] for x in cases),
            "public_query_candidates":sum(
                x["query_public_candidate_outputs"] for x in cases),
            "screen_public_comparisons":sum(
                x["screen_public_comparisons"] for x in cases),
            "work_cap":sum(x["work_cap"] for x in cases),
            "setup_feasible":sum(x["setup_feasible"] for x in cases),
            "exit_reasons":dict(Counter(x["exit"] for x in cases)),
        }
    return {
        **tally(chosen),
        "by_family":{fam:tally([r["arms"][arm] for r in records
                               if r["family"]==fam]) for fam in FAMILIES},
    }


def paired(records,first,second,family="late_source",measure="certified"):
    rows=[r for r in records if r["family"]==family]
    if measure not in ("certified","future_correct"):
        raise ValueError("unregistered outcome")
    def value(r,name):
        a=r["arms"][name]
        return a["certified"] if measure=="certified" else a["grade"]["future_correct"]
    gain=sum(value(r,second)>value(r,first) for r in rows)
    loss=sum(value(r,second)<value(r,first) for r in rows)
    return {
        "family":family,"cases":len(rows),
        "gains":gain,"losses":loss,"net":gain-loss,
        "comparison_measure":measure,
        "exact_unadjusted_p":float(nor12b.exact_sign_p(gain,loss)),
    }


def run(repetitions=REPETITIONS,late_per=LATE_PER,
        signed_per=SIGNED_PER,rational_per=RATIONAL_PER):
    ck,p1,p2=nor11.train_source()
    assert ck["parameters"]==1000 and len(p1)==len(p2)==12
    assert nor12.BASIN_SHA==CK_SHA
    roster=nor11.sample_cohorts(nor12.SPLIT_COUNTS)
    assert sha256("\n".join(x[2] for x in roster["train"]).encode()
                  ).hexdigest()==nor12.BASELINE_MANIFESTS["train"]
    assert sha256("\n".join(x[2] for x in roster["dev"]).encode()
                  ).hexdigest()==nor12.BASELINE_MANIFESTS["dev"]
    batches=fresh_cohorts(roster,repetitions,late_per,signed_per,rational_per)
    records=[]
    cohorts=[]
    for rep,batch in enumerate(batches):
        new=[single_case(case,header,ck) for case,header,digest in batch]
        assert all(row["model_visible_sha256"]==digest
                   for row,(_,_,digest) in zip(new,batch,strict=True))
        records.extend(new)
        cohorts.append({
            "replicate":rep,"cases":len(new),
            "public_manifest_sha256":sha256("\n".join(
                row["full_public_input_sha256"] for row in new).encode()).hexdigest(),
            "certificates_by_arm":{name:sum(
                r["arms"][name]["certified"]
                for r in new if r["family"]=="late_source")
                for name in ARM_NAMES},
        })
    late=[r for r in records if r["family"]=="late_source"]
    nonlate=[r for r in records if r["family"]!="late_source"]
    assert len(late)==late_per*repetitions
    assert len({r["full_public_input_sha256"] for r in records})==len(records)
    assert len({r["model_visible_sha256"] for r in late})==1
    assert len({r["model_visible_sha256"] for r in nonlate})==len(nonlate)
    assert {r["model_visible_sha256"] for r in late}.isdisjoint({
        r["model_visible_sha256"] for r in nonlate})
    arms={name:summarize(records,name) for name in ARM_NAMES}
    # All prefixes must be nested, and certificates across increasing
    # budget tiers must be monotone for the same preprocessing/ranking.
    for r in records:
        masks=[r["public_prefix_survivor_masks"][str(d)]
               for d in PREFIX_DEGREES+(BATCH_DEGREE,)]
        assert all(b&a==b for a,b in zip(masks,masks[1:],strict=True))
        for label in ("none",)+(tuple(f"d{x}" for x in PREFIX_DEGREES)) + ("batch192",):
            hits=[r["arms"][f"{label}_{b}"]["certified"]
                  for b in WORK_MULTIPLIERS]
            assert hits==sorted(hits),(label,hits)
    for name,out in arms.items():
        if not name.startswith("greedy"):
            assert out["false_accept"]==0
    primary_comparisons={
        "d176_4_vs_d160_4":paired(records,"d160_4","d176_4"),
        "d176_4_vs_none_4":paired(records,"none_4","d176_4"),
        "d160_4_vs_none_4":paired(records,"none_4","d160_4"),
        "batch192_4_vs_d176_4":paired(records,"d176_4","batch192_4"),
        "d160_4_vs_d160_original_4":paired(records,"d160_original_4","d160_4"),
        "d176_4_vs_d176_hash_4":paired(records,"d176_hash_4","d176_4"),
        "batch192_8_vs_none_8":paired(records,"none_8","batch192_8"),
    }
    success=all(primary_comparisons[k]["net"]>0 and
                primary_comparisons[k]["exact_unadjusted_p"]<=PRIMARY_ALPHA
                for k in ("d176_4_vs_d160_4","d176_4_vs_none_4"))
    return {
        "schema":SCHEMA,
        "source_checkpoint_sha256":CK_SHA,
        "frozen_train_manifest":nor12.BASELINE_MANIFESTS["train"],
        "frozen_dev_manifest":nor12.BASELINE_MANIFESTS["dev"],
        "seed_offset":FRESH_OFFSET,
        "repetitions":repetitions,
        "per_rep_late":late_per,
        "per_rep_signed":signed_per,
        "per_rep_rational":rational_per,
        "frozen_prefix_degrees":list(PREFIX_DEGREES),
        "batched_full_degree":BATCH_DEGREE,
        "frozen_work_multipliers":list(WORK_MULTIPLIERS),
        "work_model":"nor18.coefficient_operation_proxy.v1",
        "arm_names":list(ARM_NAMES),
        "all_cases":len(records),
        "primary_late_distinct_complete_public_inputs":len(late),
        "primary_late_unique_model_feature_tensors":1,
        "nonlate_unique_model_feature_tensors":len(nonlate),
        "retrospective_full_public_check_calls":16*len(records),
        "cohorts":cohorts,
        "arms":arms,
        "paired_primary_and_secondary":primary_comparisons,
        "preregistered_primary_two_control_passed":success,
        "primary_alpha":PRIMARY_ALPHA,
        "primary_full_mask_output_certifiable":sum(
            r["certificate_possible_from_complete_public_mask"] for r in late),
        "records":records,
        "scope":{
            "public_demo_prefix_only_for_screen":True,
            "query_simulated_from_public_input_only":True,
            "private_future_sealed_until_all_decisions_frozen":True,
            "original_1000_param_gru_immutable":True,
            "all_work_caps_apply_to_total_query_screen_and_full_check":True,
            "full_batch192_charges_16_full_public_candidate_comparisons":True,
            "prefix_candidates_never_exclude_full_accepted_member":True,
            "no_hidden_program_identity_in_any_policy":True,
            "proxy_excludes_bigint_bit_complexity_python_runtime_and_memory":True,
            "control_work_is_charged_independently_even_if_runner_caches":True,
            "retro_full_mask_checked_only_after_all_choices":True,
            "synthetic_generator_family_conditional_soundness_only":True,
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repetitions",type=int,default=REPETITIONS)
    p.add_argument("--late-per-repetition",type=int,default=LATE_PER)
    p.add_argument("--signed-per-repetition",type=int,default=SIGNED_PER)
    p.add_argument("--rational-per-repetition",type=int,default=RATIONAL_PER)
    p.add_argument("--check",action="store_true")
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    res=run(args.repetitions,args.late_per_repetition,
            args.signed_per_repetition,args.rational_per_repetition)
    if args.check:
        assert res["source_checkpoint_sha256"]==CK_SHA
        assert res["all_cases"]==args.repetitions*(
            args.late_per_repetition+args.signed_per_repetition+
            args.rational_per_repetition)
        assert res["primary_late_distinct_complete_public_inputs"]==(
            args.repetitions*args.late_per_repetition)
        assert res["nonlate_unique_model_feature_tensors"]==(
            args.repetitions*(args.signed_per_repetition+
                              args.rational_per_repetition))
        assert all(v["false_accept"]==0 for name,v in res["arms"].items()
                   if not name.startswith("greedy"))
        for r in res["records"]:
            for name,arm in r["arms"].items():
                assert arm["total_work"]<=arm["work_cap"]
                assert arm["total_work"]==(arm["query_work"]+
                    arm["screen_work"]+arm["full_candidate_check_work"])
    raw=json.dumps(res,indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(raw)
    else:
        print(raw,end="")


if __name__=="__main__":
    main()
