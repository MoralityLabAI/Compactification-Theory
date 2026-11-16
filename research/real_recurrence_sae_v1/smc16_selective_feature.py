"""SMC-16 source-locked selective active feature erasure experiment.

SMC-13C alignments are fully rebuilt and pinned on calibration first.
SMC-16 cohorts are created only then; all goal inclusion decisions use
the SAE geometry without ground-truth digits or continuation outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from smc16_selective_geometry import (
    FAMILIES, N_PUZZLES, N_GOALS, fresh_rosters, selective_candidate,
    score_rows, bootstrap_seed,
)
import smc15_active_necessity as smc15


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_lock_load(original_artifact,independent_artifact,conditioned_receipt,
                     allow_fetch=False,source_checkpoint=None):
    from smc14_causal_transfer import (
        load_frozen_sources,source_model_from_lock,load_independent_models,
        load_conditioned,source_group_context,exact_frozen_assignments,PAIRS,
        ORIGINAL_SOURCE_RECEIPT_SHA256,NEW_TEACHER_RECEIPT_SHA256,
        CONDITIONED_JSON_SHA256,SOURCE_TEACHER_SHA256,
    )
    orig=Path(original_artifact)
    independent=Path(independent_artifact)
    if _sha(orig/"joint-full.json")!=ORIGINAL_SOURCE_RECEIPT_SHA256:
        raise ValueError("original source receipt hash changed")
    if _sha(independent/"full.json")!=NEW_TEACHER_RECEIPT_SHA256:
        raise ValueError("independent teacher training receipt hash changed")
    if _sha(conditioned_receipt)!=CONDITIONED_JSON_SHA256:
        raise ValueError("SMC13 conditioned mapping receipt hash changed")
    _, original_saes,_=load_frozen_sources(extracted_dir=original_artifact)
    teacher_a,_=source_model_from_lock(allow_fetch,source_checkpoint)
    _,original_rosters,new=load_independent_models(independent_artifact)
    conditioned=load_conditioned(conditioned_receipt)
    teachers={"A":teacher_a}
    components={"pretrainedA_71":("A",*original_saes[71])}
    for seed,group in new.items():
        label=f"B{seed}"
        teachers[label]=group["teacher"]
        for sae_seed,params in group["saes"].items():
            components[f"newB_{seed}_SAE_{sae_seed}"]=(label,*params)
    calibration_ctx,cal_support=source_group_context(
        teachers,components,original_rosters,("feature_calibration",))
    all_assign=[]
    for i,(a,b) in enumerate(PAIRS):
        original=conditioned["results"][i]
        if (original["source"],original["target"])!=(a,b):
            raise ValueError("original SMC13 pair order changed")
        inputs=calibration_ctx[components[a][0]]["cohorts"]["feature_calibration"]["inputs"]
        permutation,sha=exact_frozen_assignments(
            cal_support[a]["feature_calibration"]["support"],
            cal_support[b]["feature_calibration"]["support"],
            inputs,original,i)
        all_assign.append({"index":i,"source":a,"target":b,
                           "permutations":permutation,
                           "assignment_hashes":sha})
    if len(all_assign)!=7 or not all(x["assignment_hashes"]["matched_to_original_smc13c_receipt"]
                                   for x in all_assign):
        raise AssertionError("seven frozen assignments not verified")
    print("SMC16_ALL_SEVEN_ORIGINAL_SMC13C_ASSIGNMENTS_FROZEN",flush=True)
    return (teachers,components,original_rosters,all_assign,PAIRS,
            {
              "source_receipt_sha256":ORIGINAL_SOURCE_RECEIPT_SHA256,
              "independent_training_receipt_sha256":NEW_TEACHER_RECEIPT_SHA256,
              "conditioned_mapping_receipt_sha256":CONDITIONED_JSON_SHA256,
              "original_teacher_sha256":SOURCE_TEACHER_SHA256,
            })


def evaluate_pair_split(ctx,components,a,b,assignment,split,rows,pair_index):
    from smc15_active_necessity import (
        encoded_goals,physical_columns,ablate_margin,
    )
    ta=ctx[components[a][0]]["cohorts"][split]
    tb=ctx[components[b][0]]["cohorts"][split]
    if not bool((ta["inputs"]==tb["inputs"]).all()):
        raise ValueError("teacher input roster changed")
    pos=ta["positions"]
    if tuple(pos.shape)!=(len(rows),N_GOALS):
        raise ValueError("incorrect goal selection")
    za,source_top,source_strict=encoded_goals(ta["stage1"],pos,*components[a][1:])
    zb,_,_=encoded_goals(tb["stage1"],pos,*components[b][1:])
    mapping=torch.tensor(assignment,dtype=torch.int64)
    matched=mapping[source_top]
    target_sae,mean,scale=components[b][1:]
    native=physical_columns(target_sae,scale)
    kill=torch.zeros((len(rows),N_GOALS,48),dtype=torch.float64)
    sham=torch.zeros_like(kill)
    case_rows=[]
    # Construct all edit candidates before any margin outcome is inspected.
    for p,row in enumerate(rows):
        goals=[]
        for g in range(N_GOALS):
            goal_pos=int(pos[p,g])
            source_j=int(source_top[p,g])
            target_j=int(matched[p,g])
            details,dk,ds=selective_candidate(
                tb["stage1"][p,goal_pos],target_sae,mean,scale,
                zb[p,g],target_j,native,pair_index,split,row["seed"],g,
                bool(source_strict[p,g]))
            kill[p,g]=dk
            sham[p,g]=ds
            details.update({"goal_slot":g,"goal_position":goal_pos,
                            "source_top_id":source_j})
            goals.append(details)
        case_rows.append({"puzzle_index":p,"seed":row["seed"],
                          "goal_positions":pos[p].tolist(),"goals":goals})
    print("SMC16_PREOUTCOME_ELIGIBILITY_FROZEN",pair_index,split,
          sum(g["eligible"] for row in case_rows for g in row["goals"]),flush=True)
    # Continue from exactly the same clean frozen FP32 stage-one anchors.
    # Ineligible displacements remain zero; only eligible outcomes are scored.
    truth=tb["truth_at_goals"]
    m_k,clean_correct,flip_k=ablate_margin(
        ctx[components[b][0]]["model64"],tb["stage1"],tb["clean_logits"],truth,pos,kill)
    m_s,clean_correct2,flip_s=ablate_margin(
        ctx[components[b][0]]["model64"],tb["stage1"],tb["clean_logits"],truth,pos,sham)
    if not bool((clean_correct==clean_correct2).all()):
        raise AssertionError("clean teacher prediction changed across arms")
    for p,row in enumerate(case_rows):
        for g,detail in enumerate(row["goals"]):
            if detail["eligible"]:
                detail.update({
                  "clean_correct":bool(clean_correct[p,g]),
                  "kill_drop":float(m_k[p,g]),
                  "sham_drop":float(m_s[p,g]),
                  "kill_flip":bool(flip_k[p,g]),
                  "sham_flip":bool(flip_s[p,g]),
                })
    return case_rows


def run(original_artifact,independent_artifact,conditioned_receipt,
        allow_fetch=False,source_checkpoint=None,pilot=0,pair_limit=4):
    if pilot not in (0,4) or pair_limit not in (1,2,3,4):
        raise ValueError("only preregistered full or four-puzzle pilot settings")
    torch.set_num_threads(2)
    (teachers,components,old,all_assign,pairs,locks)=source_lock_load(
        original_artifact,independent_artifact,conditioned_receipt,
        allow_fetch,source_checkpoint)
    if len(pairs)!=7 or any(pairs[i][0]!="pretrainedA_71" for i in range(4)):
        raise ValueError("primary roster drifted")
    # Regenerate the full prior SMC15 heldout solution family so collisions
    # against BOTH prior SMC13 and SMC15 can be rejected before intervention.
    smc15_rosters,smc15_manifest=smc15.fresh_rosters(old,limit=N_PUZZLES)
    source_and_prior=dict(old)
    source_and_prior.update(smc15_rosters)
    new_rosters,manifest=fresh_rosters(source_and_prior,limit=pilot or N_PUZZLES)
    from smc14_causal_transfer import source_group_context
    test_ctx,_=source_group_context(teachers,components,new_rosters,tuple(FAMILIES))
    results=[]
    for i,(a,b) in enumerate(pairs[:pair_limit]):
        heldout={}
        for fam in FAMILIES:
            rows=evaluate_pair_split(test_ctx,components,a,b,
                all_assign[i]["permutations"]["matched"],
                fam,new_rosters[fam],i)
            stats=score_rows(rows,bootstrap_seed(i,fam))
            heldout[fam]={"per_puzzle":rows,"statistics":stats}
        passed=(not pilot and all(heldout[f]["statistics"]["registered_success"]
                                  for f in FAMILIES))
        results.append({"source":a,"target":b,"pair_index":i,
                        "mapping_index":i,"heldout":heldout,
                        "registered_success_on_both_fresh_families":bool(passed)})
        print("SMC16_SELECTIVE_KILL",a,b,{f:heldout[f]["statistics"] for f in FAMILIES},flush=True)
    return {
        "schema":"SMC-16/selective-active-encoder-kill-v1",
        **locks,
        "all_seven_original_smc13c_assignments":all_assign,
        "all_seven_assignments_frozen_before_fresh_states":True,
        "fresh_families":{k:{"seed_base":v[0],"holes":v[1]} for k,v in FAMILIES.items()},
        "smc15_prior_cohort_manifest_sha256":smc15_manifest,
        "fresh_cohort_rosters":new_rosters,
        "fresh_cohort_manifest_sha256":manifest,
        "fresh_puzzles_per_family":pilot or N_PUZZLES,
        "primary_pair_limit":pair_limit,
        "full_protocol":bool(not pilot and pair_limit==4),
        "selection_outcome_blind":True,
        "compare":results,
        "interpretation_limit":"Numerical FP64 task effects conditional on top-four active encoder edits; not interval-certified or identification of a unique mechanism.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-artifact",type=Path,required=True)
    p.add_argument("--independent-artifact",type=Path,required=True)
    p.add_argument("--conditioned-receipt",type=Path,required=True)
    p.add_argument("--fetch-source",action="store_true")
    p.add_argument("--source-checkpoint",type=Path)
    p.add_argument("--pilot-puzzles",type=int,choices=(4,))
    p.add_argument("--pair-limit",type=int,choices=(1,2,3,4),default=4)
    p.add_argument("--out",type=Path,default=Path("runs/smc16/full.json"))
    args=p.parse_args()
    result=run(args.original_artifact,args.independent_artifact,
        args.conditioned_receipt,args.fetch_source,args.source_checkpoint,
        args.pilot_puzzles or 0,args.pair_limit)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"output":str(args.out),
        "full_protocol":result["full_protocol"],
        "pair_results":[{"target":r["target"],
             "passed":r["registered_success_on_both_fresh_families"]}
                       for r in result["compare"]]},indent=2))


if __name__=="__main__":
    main()
