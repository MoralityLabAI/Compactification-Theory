"""NOR-12B: frozen-weights, independent-seed confirmation and q ablation.

The original NOR-12 task head weights and dev-selected scales are pinned by
SHA-256 BEFORE these four fresh synthetic test seeds are examined. All
models are reconstructed exclusively from the original 96 public training
cases and candidate-only degree-192 verification labels. A separate
96/36/72 source roster is reconstructed only to prove the split hashes.

The new test cases never enter model fitting or development scale selection.
Test family "late_source" has inevitable aliases: independent source cases
can share an identical model-visible 9x8 tensor. This script reports
case-level and feature-group-weighted effects, and isolates unique
nonlate inputs for a descriptive paired exact sign test.

A retrospective q-feature ablation on the first NOR-12 test is labeled
as post-hoc and is NEVER used to tune the fresh evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from math import comb
import json
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for p in (ROOT,HERE):
    if str(p) not in sys.path:
        sys.path.insert(0,str(p))

from research.rrc_compactification import compound25 as source,model25
from research.nor11_attractor_controller import controller as prior
from research.nor12_task_readout import readout as task

FROZEN=HERE/"results"/"nor12_initial_receipt.json"
SEED_FRESH=92137000
REPETITIONS=4
CASES_PER_FAMILY=12
FIT_HASH_KEYS=("basin","shuffled_basin","random_feature","hidden_only")


def frozen_design():
    d=json.loads(FROZEN.read_text())
    assert d["schema"]=="nor12.public_demo_supervised_residual_readout.frozen_first_run.v1"
    assert d["source_checkpoint_sha256"]==task.BASIN_SHA
    assert d["full_result_json_sha256"]=="a1995f628038ab6bd447c7d4438ee861ca3c10ef90ca5a87c4063bf99811db95"
    assert d["independent_confirmation_completed"] is False
    assert set(d["fitted_weight_sha256_by_kind"])==set(FIT_HASH_KEYS)
    return d


def weight_digest(W):
    return sha256(json.dumps(W,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def rebuild_frozen_models():
    design=frozen_design()
    ck,probe,headsha,old=task.locked_rows(task.SPLIT_COUNTS)
    assert headsha==design["frozen_probe_json_sha256"]
    cases=prior.sample_cohorts(task.SPLIT_COUNTS)
    p1,p2=prior.train_source()[1:]
    diag=prior.build_rows({"train":cases["train"]},ck,p1,p2)["train"]
    shuffled=[z["label"] for z in diag]
    random.Random(prior.SEED+77).shuffle(shuffled)
    negative_probe=prior.fit_head(diag,seed=prior.SEED+77,
                                  labels_override=shuffled)
    task.attach_features(old,probe,negative_probe)
    for row in old["train"]:
        row["visible_mask"]=task.visible_mask(row)
    fitted={k:task.fit_residual(old["train"],k) for k in FIT_HASH_KEYS}
    for key,v in fitted.items():
        assert weight_digest(v["weights"])==design["fitted_weight_sha256_by_kind"][key]
    for key in ("train","dev","test"):
        assert old[key] and sha256("\n".join(r["digest"] for r in old[key]).encode()).hexdigest()==design["manifests"][key]
    return ck,probe,negative_probe,fitted,old,design,cases


def visible_digest(case):
    return sha256(json.dumps(case.visible,separators=(",",":")).encode()).hexdigest()


def fresh_cohorts(original,replicates=REPETITIONS,per_family=CASES_PER_FAMILY):
    if not 1<=replicates<=8 or not 1<=per_family<=24:
        raise ValueError("bounded independent roster required")
    seen_visible={visible_digest(c) for group in original.values() for c,_,_ in group}
    seen_train_dev={digest for split in ("train","dev") for _,_,digest in original[split]}
    seen_nonlate={digest for _,_,digest in original["test"]}
    output=[]
    for rep in range(replicates):
        fresh=[]
        for ix,family in enumerate(source.TEST_FAMILIES):
            rng=random.Random(source.SEED+SEED_FRESH+rep*1000003+ix*10007)
            count=0
            for trial in range(per_family*1500):
                candidate=source.make_case(
                    rng,family,f"nor12-confirm-r{rep}-{family}-{trial:05d}")
                headers=source.possible_headers(candidate)
                if len(headers)!=1:
                    continue
                digest=prior.public_digest(candidate,headers[0])
                vis=visible_digest(candidate)
                if vis in seen_visible or digest in seen_train_dev:
                    continue
                # The model-visible late-source tensor is often exactly
                # identical across many different independently generated
                # cases. Preserve those distinct cases as a single feature
                # cluster; other families require novel feature tensors.
                if family!="late_source" and digest in seen_nonlate:
                    continue
                seen_visible.add(vis)
                if family!="late_source":
                    seen_nonlate.add(digest)
                fresh.append((candidate,headers[0],digest))
                count+=1
                if count==per_family:
                    break
            if count!=per_family:
                raise RuntimeError(f"insufficient fresh cases rep={rep} family={family} found={count}")
        output.append(fresh)
    return output


def new_rows(items,checkpoint,probe,negative):
    rows=[]
    for case,header,digest in items:
        h,parity=prior.model_prefix_hidden(case,header,checkpoint)
        base=model25.affine(checkpoint["state_dict"]["head.weight"],
                            checkpoint["state_dict"]["head.bias"],h)
        rows.append({"case":case,"header":header,"digest":digest,
                     "h":h,"base":base,"p":prior.head_score(h,probe),
                     "parity":parity})
    task.attach_features({"confirmation":rows},probe,negative)
    return rows


def exact_sign_p(gains,losses):
    discordant=gains+losses
    if discordant==0:
        return Fraction(1)
    numerator=2*sum(comb(discordant,k)
                    for k in range(min(gains,losses)+1))
    return min(Fraction(1),Fraction(numerator,2**discordant))


def independent_feature_groups(original,alternative):
    assert len(original["records"])==len(alternative["records"])
    group=defaultdict(lambda:[0,0,0])
    for a,b in zip(original["records"],alternative["records"],strict=True):
        assert a["public_features_sha256"]==b["public_features_sha256"]
        dig=a["public_features_sha256"]
        group[dig][0]+=b["future_correct"]-a["future_correct"]
        group[dig][1]+=1
        group[dig][2]+=int(a["family"]=="late_source")
    positive=sum(a>0 for a,_,_ in group.values())
    negative=sum(a<0 for a,_,_ in group.values())
    tied=sum(a==0 for a,_,_ in group.values())
    weighted=sum(a/n for a,n,_ in group.values())
    return {
        "unique_model_inputs":len(group),
        "positive_feature_groups":positive,
        "negative_feature_groups":negative,
        "tied_feature_groups":tied,
        "feature_group_mean_accuracy_delta_sum":weighted,
        "feature_group_average_accuracy_delta":weighted/len(group),
        "case_level_gain_concentration":{
            "largest_single_feature_gain":max(a for a,_,_ in group.values()),
            "nonlate_feature_count":sum(v==0 for _,_,v in group.values()),
            "late_source_feature_count":sum(v>0 for _,_,v in group.values()),
        },
        "posthoc_feature_group_sign_test_p":float(exact_sign_p(positive,negative)),
    }


def ablate_frozen_basin_feature(test,fitted,design):
    alpha=design["fixed_development_scales"]["basin"]
    original_feature=[r["x"]["basin"][12] for r in test]
    zero=[0.0]*len(test)
    scrambled=[r["x"]["shuffled_basin"][12] for r in test]
    group_feature={}
    for r in test:
        group_feature.setdefault(r["digest"],r["x"]["basin"][12])
        assert group_feature[r["digest"]]==r["x"]["basin"][12]
    keys=sorted(group_feature)
    vals=[group_feature[k] for k in keys]
    random.Random(task.SEED+1201).shuffle(vals)
    permuted=dict(zip(keys,vals,strict=True))
    variants={
        "original_basin_feature":original_feature,
        "set_basin_feature_zero":zero,
        "replace_with_shuffled_probe":scrambled,
        "permuted_unique_feature_groups":[permuted[r["digest"]] for r in test],
    }
    outputs={}
    baseline=task.evaluate(test)
    for name,scalar in variants.items():
        rows=[]
        for r,x in zip(test,scalar,strict=True):
            t={**r,"x":dict(r["x"])}
            vec=list(r["x"]["basin"])
            vec[12]=x
            t["x"]["basin"]=vec
            rows.append(t)
        ev=task.evaluate(rows,fitted["basin"],"basin",alpha)
        outputs[name]={
            "future_correct":ev["future_correct"],
            "false_accept":ev["false_accept"],
            "paid_checks":ev["paid_checks_total"],
            "paired_baseline":task.paired_comparison(baseline,ev),
            "feature_groups":independent_feature_groups(baseline,ev),
        }
    return {
        "analysis_type":"posthoc original-test feature ablation; no model refit",
        "frozen_basin_readout_scale":alpha,
        "variants":outputs,
        "feature_group_permutation_seed":task.SEED+1201,
    }


def evaluate_confirm(replicates=REPETITIONS,per_family=CASES_PER_FAMILY):
    ck,probe,negative,fitted,old,design,original= rebuild_frozen_models()
    cohorts=fresh_cohorts(original,replicates,per_family)
    retained_scales=design["fixed_development_scales"]
    audits=[]
    global_stats={k:Counter() for k in ("original",)+FIT_HASH_KEYS}
    all_group_diffs=defaultdict(list)
    manifests=[]
    for rep,items in enumerate(cohorts):
        rows=new_rows(items,ck,probe,negative)
        baseline=task.evaluate(rows)
        arms={"original":baseline}
        for kind in FIT_HASH_KEYS:
            arms[kind]=task.evaluate(rows,fitted[kind],kind,retained_scales[kind])
        rep_stats={}
        for name,ev in arms.items():
            for stat in ("cases","future_correct","false_accept","paid_checks_total"):
                global_stats[name][stat]+=ev[stat]
            rep_stats[name]={
                "future_correct":ev["future_correct"],
                "false_accept":ev["false_accept"],
                "paid_checks":ev["paid_checks_total"],
                "paired":task.paired_comparison(baseline,ev),
                "grouped":independent_feature_groups(baseline,ev),
                "family_counts":ev["by_family"]
            }
        for i,row in enumerate(rows):
            dig=row["digest"]
            base_out=baseline["records"][i]["future_correct"]
            all_group_diffs[("original",dig)].append((0,row["case"].family))
            for key in FIT_HASH_KEYS:
                delta=arms[key]["records"][i]["future_correct"]-base_out
                all_group_diffs[(key,dig)].append((delta,row["case"].family))
        manifest=sha256("\n".join(r["digest"] for r in rows).encode()).hexdigest()
        manifests.append(manifest)
        audits.append({
            "replicate":rep,"seed":source.SEED+SEED_FRESH+rep*1000003,
            "cases":len(rows),
            "unique_feature_tensors":len({r["digest"] for r in rows}),
            "manifest_sha256":manifest,
            "arms":rep_stats,
        })
    aggregated={}
    for key,v in global_stats.items():
        ref=global_stats["original"]
        gains=sum(x["arms"][key]["paired"]["newly_correct"] for x in audits)
        losses=sum(x["arms"][key]["paired"]["lost_correct"] for x in audits)
        bykey={dig:vals for (kind,dig),vals in all_group_diffs.items() if kind==key}
        positive=sum(sum(delta for delta,_ in vals)>0 for vals in bykey.values())
        negativecount=sum(sum(delta for delta,_ in vals)<0 for vals in bykey.values())
        nonlate=[(delta,fam) for vals in bykey.values() for delta,fam in vals if fam!="late_source"]
        nongain=sum(d>0 for d,f in nonlate)
        nonloss=sum(d<0 for d,f in nonlate)
        aggregated[key]={
            "cases":v["cases"],"future_correct":v["future_correct"],
            "false_accept":v["false_accept"],
            "paid_checks":v["paid_checks_total"],
            "paired_newly_correct":gains,"paired_lost_correct":losses,
            "net_change_from_original":v["future_correct"]-ref["future_correct"],
            "unique_test_input_groups":len(bykey),
            "positive_unique_feature_groups":positive,
            "negative_unique_feature_groups":negativecount,
            "nonlate_pairwise_gains":nongain,
            "nonlate_pairwise_losses":nonloss,
            "posthoc_case_mcnemar_exact_p":float(exact_sign_p(gains,losses)),
            "posthoc_nonlate_case_mcnemar_exact_p":float(exact_sign_p(nongain,nonloss)),
            "posthoc_unique_feature_group_sign_p":float(exact_sign_p(positive,negative)),
        }
    first_ablation=ablate_frozen_basin_feature(old["test"],fitted,design)
    return {
        "schema":"nor12b.independent_frozen_head_confirmations.v1",
        "source_checkpoint_sha256":task.BASIN_SHA,
        "first_source_receipt_sha256":design["full_result_json_sha256"],
        "source_training_manifest_sha256":design["manifests"]["train"],
        "source_dev_manifest_sha256":design["manifests"]["dev"],
        "fixed_readout_weight_hashes":design["fitted_weight_sha256_by_kind"],
        "frozen_development_scales":retained_scales,
        "new_holdout_seed_offset":SEED_FRESH,
        "independent_repetitions":replicates,
        "new_cases_per_family_per_repetition":per_family,
        "new_heldout_manifests":manifests,
        "retrospective_first_split_ablation":first_ablation,
        "independent_holdout_replicates":audits,
        "independent_holdout_aggregate":aggregated,
        "scope":{
            "readout_weights_retrained_with_original_training_only":True,
            "all_weight_hashes_verified_before_new_cases_scored":True,
            "all_scales_pinned_before_independent_test":True,
            "heldout_future_labels_used_for_training":False,
            "late_source_alias_clusters_reported":True,
            "multiple_model_selection_or_testing_controls":True,
            "hypothesis_priority":"exploratory confirmation after initial split inspection",
            "native_recurrent_checkpoint_modified":False,
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repetitions",type=int,default=REPETITIONS)
    p.add_argument("--per-family",type=int,default=CASES_PER_FAMILY)
    p.add_argument("--check",action="store_true")
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    result=evaluate_confirm(args.repetitions,args.per_family)
    if args.check:
        assert result["source_checkpoint_sha256"]==task.BASIN_SHA
        assert result["scope"]["all_weight_hashes_verified_before_new_cases_scored"]
        assert not result["scope"]["native_recurrent_checkpoint_modified"]
        assert result["independent_holdout_aggregate"]["original"]["cases"]==6*args.per_family*args.repetitions
        assert all(v["paid_checks"]<=4*v["cases"]
                   for v in result["independent_holdout_aggregate"].values())
    s=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(s)
    else:
        print(s,end="")


if __name__=="__main__":
    main()
