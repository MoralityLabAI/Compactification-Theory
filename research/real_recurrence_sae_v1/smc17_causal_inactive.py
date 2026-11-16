"""SMC-17: source-locked heldout inactive-preactivation matched causal test.

Follows the original SMC-13C calibration assignment and SMC-16 selective
kill protocol. All 131 sham geometry candidates are screened before
ground-truth digits or frozen continuation margins are examined.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from smc15_active_necessity import (
    encoded_goals, physical_columns, ablate_margin,
)
from smc16_selective_kill import (
    physical_covectors,selective_vectors,score as prior_score,PAIRS,
)
from smc17_inactive_control import (
    fresh_rosters,best_inactive_matched_sham,FAMILIES,
)

PILOT_SIZE=4
FULL_SIZE=64


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_models(original_artifact,independent_artifact,conditioned_receipt,
                  allow_fetch=False,source_checkpoint=None):
    from smc14_causal_transfer import (
        load_frozen_sources,source_model_from_lock,
        load_independent_models,load_conditioned,
        source_group_context,exact_frozen_assignments,PAIRS as ALL_PAIRS,
        ORIGINAL_SOURCE_RECEIPT_SHA256,NEW_TEACHER_RECEIPT_SHA256,
        CONDITIONED_JSON_SHA256,
    )
    if _sha(Path(original_artifact)/"joint-full.json")!=ORIGINAL_SOURCE_RECEIPT_SHA256:
        raise ValueError("original SMC11 source receipt changed")
    if _sha(Path(independent_artifact)/"full.json")!=NEW_TEACHER_RECEIPT_SHA256:
        raise ValueError("SMC13 independent teacher receipt changed")
    if _sha(conditioned_receipt)!=CONDITIONED_JSON_SHA256:
        raise ValueError("SMC13C conditioned assignment source receipt changed")
    _,source_saes,_=load_frozen_sources(extracted_dir=original_artifact)
    teacher_a,_=source_model_from_lock(allow_fetch,source_checkpoint)
    _,original_rosters,trained=load_independent_models(independent_artifact)
    condition=load_conditioned(conditioned_receipt)
    teachers={"A":teacher_a}
    components={"pretrainedA_71":("A",*source_saes[71])}
    for seed,group in trained.items():
        teacher_label=f"B{seed}"
        teachers[teacher_label]=group["teacher"]
        for sae_seed,params in group["saes"].items():
            components[f"newB_{seed}_SAE_{sae_seed}"]=(teacher_label,*params)
    if tuple(ALL_PAIRS[:4])!=PAIRS:
        raise ValueError("frozen primary source-to-new-teacher roster changed")
    cal_ctx,cal_features=source_group_context(
        teachers,components,original_rosters,("feature_calibration",))
    assignments=[]
    for i,(a,b) in enumerate(ALL_PAIRS):
        source_record=condition["results"][i]
        if (source_record["source"],source_record["target"])!=(a,b):
            raise ValueError("source SMC13C model pair ordering changed")
        inputs=cal_ctx[components[a][0]]["cohorts"]["feature_calibration"]["inputs"]
        maps,hashes=exact_frozen_assignments(
            cal_features[a]["feature_calibration"]["support"],
            cal_features[b]["feature_calibration"]["support"],
            inputs,source_record,i)
        assignments.append({"source":a,"target":b,"matched":maps["matched"],
                            "hashes":hashes})
    if len(assignments)!=7:
        raise AssertionError("seven SMC13 conditioned pair receipts required")
    print("SMC17_SEVEN_ORIGINAL_CALIBRATED_MAPPINGS_FROZEN",flush=True)
    return (teachers,components,original_rosters,assignments,
            {
              "source_receipt_sha256":ORIGINAL_SOURCE_RECEIPT_SHA256,
              "independent_training_receipt_sha256":NEW_TEACHER_RECEIPT_SHA256,
              "conditioned_mapping_receipt_sha256":CONDITIONED_JSON_SHA256,
            })


def evaluate_family(ctx,components,a,b,assign,roster,family,pair_idx):
    src=ctx[components[a][0]]["cohorts"][family]
    dst=ctx[components[b][0]]["cohorts"][family]
    if not bool((src["inputs"]==dst["inputs"]).all()):
        raise ValueError("source/receiver puzzle input differs")
    positions=src["positions"]
    _,top,strict=encoded_goals(src["stage1"],positions,*components[a][1:])
    sae,mean,scale=components[b][1:]
    codes,_,_=encoded_goals(dst["stage1"],positions,sae,mean,scale)
    mapping=torch.tensor(assign,dtype=torch.long)[top]
    cov=physical_covectors(sae,scale)
    dec=physical_columns(sae,scale)
    n=len(roster)
    kill=torch.zeros((n,8,48),dtype=torch.float64)
    base=torch.zeros_like(kill)
    best=torch.zeros_like(kill)
    infos=[]
    states=Counter()
    for i,row in enumerate(roster):
        one=[]
        for j in range(8):
            feature=int(mapping[i,j])
            h=dst["stage1"][i,int(positions[i,j])]
            if not bool(strict[i,j]):
                dk=db=torch.zeros(48,dtype=torch.float64)
                info={"status":"source_top_not_strict"}
                geometry=None
            else:
                dk,db,info=selective_vectors(
                    h,codes[i,j],feature,sae,mean,scale,
                    cov,dec,row["seed"],pair_idx,family,j)
                if info["status"]=="eligible":
                    selected,geometry=best_inactive_matched_sham(
                        h,codes[i,j],feature,sae,mean,scale,cov,
                        dk,db,pair_idx,family,row["seed"],j)
                else:
                    selected=torch.zeros(48,dtype=torch.float64)
                    geometry=None
            kill[i,j]=dk
            base[i,j]=db
            best[i,j]=selected if info["status"]=="eligible" else torch.zeros(48,dtype=torch.float64)
            info=dict(info)
            info["candidate_geometry"]=geometry
            one.append(info)
            states[info["status"]]+=1
        infos.append(one)
    good=torch.tensor([[g["status"]=="eligible" for g in r] for r in infos],dtype=torch.bool)
    print("SMC17_PREOUTCOME_GEOMETRIC_ELIGIBILITY_FROZEN",
          family,pair_idx,int(good.sum()),flush=True)
    truth=dst["truth_at_goals"]
    teacher=ctx[components[b][0]]["model64"]
    stage=dst["stage1"]
    clean=dst["clean_logits"]
    k,correct,k_flip=ablate_margin(teacher,stage,clean,truth,positions,kill)
    s,correct_b,s_flip=ablate_margin(teacher,stage,clean,truth,positions,base)
    t,correct_t,t_flip=ablate_margin(teacher,stage,clean,truth,positions,best)
    if not (bool((correct==correct_b).all()) and bool((correct==correct_t).all())):
        raise AssertionError("clean target prediction drifted between branches")
    raw=[]
    for i,row in enumerate(roster):
        raw.append({
         "puzzle_index":i,"seed":row["seed"],
         "goal_positions":positions[i].tolist(),
         "goal_truth_digits":truth[i].tolist(),
         "source_top_feature_ids":top[i].tolist(),
         "target_matched_feature_ids":mapping[i].tolist(),
         "eligible_mask":good[i].tolist(),
         "edit_diagnostics":infos[i],
         "kill_margin_drop":[float(k[i,j]) if good[i,j] else None for j in range(8)],
         "sham_margin_drop":[float(t[i,j]) if good[i,j] else None for j in range(8)],
         "baseline_sham_margin_drop":[float(s[i,j]) if good[i,j] else None for j in range(8)],
         "clean_correct":correct[i].tolist(),
         "kill_correct_to_wrong":[bool(k_flip[i,j]) if good[i,j] else None for j in range(8)],
         "sham_correct_to_wrong":[bool(t_flip[i,j]) if good[i,j] else None for j in range(8)],
         "baseline_sham_correct_to_wrong":[bool(s_flip[i,j]) if good[i,j] else None for j in range(8)],
        })
    return raw,{
      "ineligibility_reason_counts":dict(states),
      "selected_blanks":n*8,
      "eligible_goals":int(good.sum()),
      "target_clean_goal_accuracy":float(correct.float().mean()),
    }


def score_family(raw,pair_idx,family):
    seed=202617130+100*pair_idx+len(family)
    best=prior_score(raw,seed)
    baseline=[]
    eps_base=[];eps_opt=[];rho=[]
    selected={}
    for row in raw:
        replica=dict(row)
        replica["sham_margin_drop"]=row["baseline_sham_margin_drop"]
        replica["sham_correct_to_wrong"]=row["baseline_sham_correct_to_wrong"]
        baseline.append(replica)
        for info in row["edit_diagnostics"]:
            if info["status"]!="eligible":
                continue
            d=info["candidate_geometry"]
            eps_base.append(d["epsilon_baseline"])
            eps_opt.append(d["epsilon_best"])
            rho.append(d["inactive_residual_floor"])
            key=d["chosen_candidate"]
            selected[key]=selected.get(key,0)+1
    comparison=prior_score(baseline,seed)
    if len(eps_base)!=best["eligible_goals"]:
        raise AssertionError("eligible geometry/outcome counts differ")
    total_base=sum(eps_base)
    matching_ratio=sum(eps_opt)/total_base if total_base>0 else None
    improved=(matching_ratio is not None and matching_ratio<=0.90)
    full_success=bool(best["registered_success"] and improved)
    return {
      "best_sham_causal_score":best,
      "original_four_seed_sham_causal_score":comparison,
      "inactive_epsilon_baseline_sum":total_base,
      "inactive_epsilon_best_sum":sum(eps_opt),
      "inactive_matching_ratio":matching_ratio,
      "mean_residual_floor":sum(rho)/len(rho) if rho else None,
      "best_candidate_choice_counts":selected,
      "improved_inactive_disturbance_by_10pct":improved,
      "registered_success":full_success,
    }


def run(original_artifact,independent_artifact,conditioned_receipt,
        allow_fetch=False,source_checkpoint=None,pilot=0,pair_limit=4):
    if pilot not in (0,PILOT_SIZE) or pair_limit not in (1,2,3,4):
        raise ValueError("only registered SMC17 pilot/full experiment sizes")
    torch.set_num_threads(2)
    teachers,components,old,assignments,lock=source_models(
        original_artifact,independent_artifact,conditioned_receipt,
        allow_fetch,source_checkpoint)
    rosters,manifests=fresh_rosters(old,limit=pilot or FULL_SIZE)
    from smc14_causal_transfer import source_group_context
    ctx,_=source_group_context(teachers,components,rosters,tuple(FAMILIES))
    compare=[]
    for i,(a,b) in enumerate(PAIRS[:pair_limit]):
        heldout={}
        for fam in FAMILIES:
            raw,coverage=evaluate_family(
                ctx,components,a,b,assignments[i]["matched"],
                rosters[fam],fam,i)
            stats=score_family(raw,i,fam)
            heldout[fam]={"per_puzzle":raw,"coverage":coverage,"statistics":stats}
        complete=not bool(pilot or pair_limit<4)
        success=bool(complete and all(heldout[f]["statistics"]["registered_success"]
                                      for f in FAMILIES))
        compare.append({"source":a,"target":b,"mapping_index":i,
                        "frozen_matched_feature_mapping":assignments[i]["matched"],
                        "source_mapping_sha256":assignments[i]["hashes"],
                        "heldout":heldout,
                        "registered_success_both_families":success})
        print("SMC17_INACTIVE_CONTROL",b,{
            f:{"effect":heldout[f]["statistics"]["best_sham_causal_score"].get(
                    "mean_selective_kill_minus_sham_margin_drop"),
               "epsilon_ratio":heldout[f]["statistics"]["inactive_matching_ratio"],
               "gate":heldout[f]["statistics"]["registered_success"]}
            for f in FAMILIES},flush=True)
    return {
      "schema":"SMC-17/source-locked-inactive-matching-sham-v1",
      "preregistration_commit":"59bad2ad422faaf3b7bfe7410f7b48656cb279f0",
      **lock,
      "all_seven_original_assignment_receipts":assignments,
      "seven_calibration_assignments_frozen_before_heldout":True,
      "candidate_random_budget":128,
      "test_seed_bases":{k:v[0] for k,v in FAMILIES.items()},
      "fresh_rosters":rosters,"fresh_manifest_sha256":manifests,
      "puzzles_per_family":pilot or FULL_SIZE,
      "full_protocol":not bool(pilot or pair_limit<4),
      "comparison_roster":compare,
      "limit":"Finite-budget inactive-preactivation matching and numerical FP64 task margins; target-feature causal identity still requires gradient and curvature assumptions.",
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
    p.add_argument("--out",type=Path,default=Path("runs/smc17/full.json"))
    a=p.parse_args()
    result=run(a.original_artifact,a.independent_artifact,a.conditioned_receipt,
               a.fetch_source,a.source_checkpoint,a.pilot_puzzles or 0,a.pair_limit)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"out":str(a.out),
         "full_protocol":result["full_protocol"],
         "per_pair":[{"target":x["target"],"passed":x["registered_success_both_families"]}
                     for x in result["comparison_roster"]]},indent=2))


if __name__=="__main__":
    main()
