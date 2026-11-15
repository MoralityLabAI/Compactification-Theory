"""NOR-15B: trained trajectory innovation vs endpoint-only task readouts.

Preregistered protocol: 4x(5*50+10)=1040 fresh public examples with 1000
distinct nonlate feature tensors as PRIMARY. Remaining forty late_source
cases are a single aliased model input and SECONDARY.

Frozen RRC25 recurrent checkpoint, its native logits and 4 exact paid
candidate-template verification checks. Each of six 400-slot public-only
task correction models receives the same 96 training cases and is
selected only on original 36 DEV cases. The H4 reconstruction predictor
is independently selected from public hidden-state data; no private
future labels or source program enter any fit.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
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
from research.nor12_task_readout import readout as old
from research.nor12_task_readout import independent_confirmation as oldfresh
from research.nor13_gated_readout import interaction as nor13
from research.nor14_conditional_basin import experiment as nor14
from research.nor15_trajectory_sufficiency import audit as nor15a

SCHEMA="nor15b.public_verified_trajectory_innovation.v1"
SEED_OFFSET=92815000
REPS=4
NONLATE_PER=50
LATE_PER=10
FAMILIES=tuple(corpus.TEST_FAMILIES)
KINDS=("residual_early","raw_early","reconstructed_early",
       "endpoint_nonlinear","shuffled_early","hidden_only")
SLOT_WIDTH=25
CLASSES=16
EPOCHS=150
LEARNING_RATE=.023
WEIGHT_DECAY=.075
SCALES=(.25,.5,1.,2.)
BUDGET=4


def source_disjoint_new(roster,reps=REPS,nonlate=NONLATE_PER,late=LATE_PER):
    if not 1<=reps<=4 or not 1<=nonlate<=50 or not 1<=late<=10:
        raise ValueError("frozen NOR15B fresh case budget exceeded")
    old12=oldfresh.fresh_cohorts(roster,replicates=oldfresh.REPETITIONS,
                                per_family=oldfresh.CASES_PER_FAMILY)
    old13=nor13.new_heldout(roster,replicates=nor13.CONFIRMATIONS,
                           per_family=nor13.FAMILY_PER)
    old14=nor14.fresh_cohorts(roster,repetitions=nor14.REPETITIONS,
                             per_family=nor14.PER_FAMILY)
    old15=nor15a.new_heldout(roster,repetitions=nor15a.REPETITIONS,
                             per_family=nor15a.PER_FAMILY)
    known_vis={nor13.source_fingerprint(case)
               for split in roster.values() for case,header,digest in split}
    known_feat={digest for split in roster.values() for case,header,digest in split}
    for cohort in old12+old13+old14+old15:
        for case,header,digest in cohort:
            known_vis.add(nor13.source_fingerprint(case))
            known_feat.add(digest)
    cohorts=[]
    for rep in range(reps):
        batch=[]
        for fi,family in enumerate(FAMILIES):
            limit=late if family=="late_source" else nonlate
            rng=random.Random(corpus.SEED+SEED_OFFSET+rep*1000003+fi*10007)
            got=0
            for trial in range(limit*2200):
                case=corpus.make_case(rng,family,
                    f"nor15b-r{rep}-{family}-{trial:05d}")
                heads=corpus.possible_headers(case)
                if len(heads)!=1:continue
                header=heads[0]
                digest=nor14.prior_probe.public_digest(case,header)
                vis=nor13.source_fingerprint(case)
                if vis in known_vis or (family!="late_source" and digest in known_feat):
                    continue
                known_vis.add(vis)
                if family!="late_source":known_feat.add(digest)
                batch.append((case,header,digest));got+=1
                if got==limit:break
            if got!=limit:raise RuntimeError(
                f"fresh NOR15B source budget depleted {rep} {family}")
        cohorts.append(batch)
    return cohorts


def train_reconstructor(train,dev,ck):
    path_train=[nor15a.path_row(r,ck) for r in train]
    path_dev=[nor15a.path_row(r,ck) for r in dev]
    xstat,ystat,rff,predictors,selected,grid=nor15a.freeze_dev(
        path_train,path_dev)
    # This is a training/public-dev-selected H9 -> H4 decoder.
    model=predictors["nonlinear"]
    assert selected["nonlinear"]["lambda"] in nor15a.RIDGES
    return {"xstat":xstat,"ystat":ystat,"rff":rff,"model":model,
            "dev_selection":selected,
            "label_provenance":"public nine-row source-input hidden-state reconstruction"}


def features_from_public(rows,ck,reconstruction,split_seed):
    """Attach 12-dimensional companion columns; shuffle by UNIQUE digest."""
    xstat=reconstruction["xstat"]
    ystat=reconstruction["ystat"]
    rff=reconstruction["rff"]
    model=reconstruction["model"]
    intermediates={}
    prepared=[]
    for row in rows:
        record=nor15a.path_row(row,ck)
        z9=nor15a.standardized(record["h9"],xstat)
        z4=nor15a.standardized(record["h4"],ystat)
        predicted=nor15a.predict(model,record,xstat,rff)
        zrandom=[
            math.tanh(.8*math.fsum(a*b for a,b in zip(w,z9,strict=True))+bias)
            for w,bias in zip(rff["weights"][:12],rff["biases"][:12],strict=True)]
        assert len(z4)==len(z9)==len(predicted)==len(zrandom)==12
        if row["digest"] in intermediates:
            assert intermediates[row["digest"]]==z4
        intermediates[row["digest"]]=z4
        prepared.append((row,z9,z4,predicted,zrandom))
    ids=sorted(intermediates)
    shuf=ids[:]
    random.Random(202611151500 + split_seed).shuffle(shuf)
    assignments={a:intermediates[b] for a,b in zip(ids,shuf,strict=True)}
    for row,z9,z4,predicted,zrandom in prepared:
        permuted=assignments[row["digest"]]
        row["nor15b_x"]={
            "residual_early":z9+[a-b for a,b in zip(z4,predicted,strict=True)]+[1.],
            "raw_early":z9+z4+[1.],
            "reconstructed_early":z9+predicted+[1.],
            "endpoint_nonlinear":z9+zrandom+[1.],
            "shuffled_early":z9+[a-b for a,b in zip(permuted,predicted,strict=True)]+[1.],
            "hidden_only":z9+[0.]*12+[1.],
        }
        assert all(len(x)==SLOT_WIDTH for x in row["nor15b_x"].values())
    return rows


def raw_scores(row,weights,kind,scale):
    if kind not in KINDS or not 0<=scale<=2:
        raise ValueError("invalid readout kind or mixing amplitude")
    if len(weights)!=CLASSES or any(len(w)!=SLOT_WIDTH for w in weights):
        raise ValueError("invalid 400-slot task residual")
    x=row["nor15b_x"][kind]
    return [b+scale*math.fsum(a*v for a,v in zip(w,x,strict=True))
            for b,w in zip(row["base"],weights,strict=True)]


def gradient(row,weights,kind):
    scores=raw_scores(row,weights,kind,1.)
    p=old.softmax(scores)
    accepted=row["visible_mask"]
    assert len(accepted)==CLASSES and any(accepted)
    mass=math.fsum(p[j] for j in range(CLASSES) if accepted[j])
    if mass<=0:raise ArithmeticError("underflowed accepted-set mass")
    x=row["nor15b_x"][kind]
    return -math.log(mass), [
        [(p[j]-(p[j]/mass if accepted[j] else 0.))*v for v in x]
        for j in range(CLASSES)]


def fit(train,kind,epochs=EPOCHS):
    if len(train)!=96 or not 1<=epochs<=EPOCHS or kind not in KINDS:
        raise ValueError("requires frozen public train set and parameter budget")
    if any("visible_mask" not in x or "nor15b_x" not in x for x in train):
        raise ValueError("public training acceptance labels not attached")
    W=[[0.]*SLOT_WIDTH for _ in range(CLASSES)]
    m=[[0.]*SLOT_WIDTH for _ in range(CLASSES)]
    v=[[0.]*SLOT_WIDTH for _ in range(CLASSES)]
    trace=[]
    for epoch in range(1,epochs+1):
        g=[[0.]*SLOT_WIDTH for _ in range(CLASSES)]
        loss=0.
        for row in train:
            ce,grad=gradient(row,W,kind)
            loss+=ce
            for j in range(CLASSES):
                for k in range(SLOT_WIDTH):
                    g[j][k]+=grad[j][k]
        for j in range(CLASSES):
            for k in range(SLOT_WIDTH):
                delta=g[j][k]/len(train)+WEIGHT_DECAY*W[j][k]
                m[j][k]=.9*m[j][k]+.1*delta
                v[j][k]=.999*v[j][k]+.001*delta*delta
                W[j][k]-=LEARNING_RATE*(m[j][k]/(1-.9**epoch))/(
                    math.sqrt(v[j][k]/(1-.999**epoch))+1e-8)
        if epoch in (1,40,80,120,150):
            trace.append({"epoch":epoch,"train_public_accepted_set_NLL":loss/len(train)})
    return {"kind":kind,"weights":W,"allocated_slots":400,
            "active_slots":208 if kind=="hidden_only" else 400,
            "trained_cases":len(train),"public_template_label_exposures":1536,
            "private_future_labels_used":False,"trained_epochs":epochs,"trace":trace}


def evaluate(rows,model=None,scale=0):
    if model is None:
        assert scale==0
        return old.evaluate(rows)
    modified=[{**row,"base":raw_scores(row,model["weights"],model["kind"],scale)}
              for row in rows]
    result=old.evaluate(modified)
    assert result["paid_checks_total"]<=BUDGET*len(rows)
    return result


def dev_selection(dev,models):
    utility=lambda x:x["future_correct"]-.5*x["false_accept"]-.015*x["paid_checks_total"]
    base=evaluate(dev)
    grid=[{"kind":"original","scale":0.,"future_correct":base["future_correct"],
           "false_accept":base["false_accept"],"paid_checks":base["paid_checks_total"],
           "utility":utility(base)}]
    for kind in KINDS:
        for scale in SCALES:
            r=evaluate(dev,models[kind],scale)
            grid.append({"kind":kind,"scale":scale,"future_correct":r["future_correct"],
                         "false_accept":r["false_accept"],"paid_checks":r["paid_checks_total"],
                         "utility":utility(r)})
    selected={k:max((v for v in grid if v["kind"]==k),
                     key=lambda z:(z["utility"],-z["scale"])) for k in KINDS}
    global_best=max(grid,key=lambda z:(z["utility"],z["kind"]=="original",-z["scale"]))
    return selected,global_best,grid


def paired(a,b,nonlate_only=True):
    assert len(a)==len(b)
    groups=defaultdict(int)
    gains=losses=cases=0
    for x,y in zip(a,b,strict=True):
        assert x["public_features_sha256"]==y["public_features_sha256"]
        assert x["family"]==y["family"]
        if nonlate_only and x["family"]=="late_source":continue
        cases+=1
        delta=y["future_correct"]-x["future_correct"]
        gains+=delta>0; losses+=delta<0
        groups[x["public_features_sha256"]]+=delta
    pos=sum(x>0 for x in groups.values())
    neg=sum(x<0 for x in groups.values())
    return {"cases":cases,"unique_public_inputs":len(groups),
            "paired_gains":gains,"paired_losses":losses,"net":gains-losses,
            "positive_input_groups":pos,"negative_input_groups":neg,
            "unadjusted_exact_sign_p":float(oldfresh.exact_sign_p(pos,neg))}


def run(reps=REPS,nonlate=NONLATE_PER,late=LATE_PER,epochs=EPOCHS):
    ck,probe,negative,earlier_models,original,provenance,roster=nor13.frozen_data()
    assert ck["parameters"]==1000 and old.BASIN_SHA==nor13.READOUT_SHA
    train,dev=original["train"],original["dev"]
    for row in train:
        assert row["visible_mask"]==old.visible_mask(row)
    reconstructor=train_reconstructor(train,dev,ck)
    features_from_public(train,ck,reconstructor,split_seed=101)
    features_from_public(dev,ck,reconstructor,split_seed=211)
    models={kind:fit(train,kind,epochs) for kind in KINDS}
    chosen,global_choice,grid=dev_selection(dev,models)
    cohorts=source_disjoint_new(roster,reps,nonlate,late)
    all_records=defaultdict(list)
    result_by_rep=[]
    for rep,items in enumerate(cohorts):
        rows=oldfresh.new_rows(items,ck,probe,negative)
        features_from_public(rows,ck,reconstructor,split_seed=701+rep)
        arms={"original":evaluate(rows)}
        for kind in KINDS:
            arms[kind]=evaluate(rows,models[kind],chosen[kind]["scale"])
        if global_choice["kind"]=="original":arms["dev_global"]=arms["original"]
        else:
            k=global_choice["kind"]
            arms["dev_global"]=evaluate(rows,models[k],global_choice["scale"])
        for kind,ev in arms.items():
            all_records[kind].extend(ev["records"])
            assert ev["paid_checks_total"]<=4*len(rows)
        result_by_rep.append({
            "replicate":rep,
            "cases":len(rows),
            "unique_nonlate_input_groups":len({
                r["digest"] for r in rows if r["case"].family!="late_source"}),
            "source_visible_manifest_sha256":sha256("\n".join(
                nor13.source_fingerprint(r["case"]) for r in rows).encode()).hexdigest(),
            "public_features_manifest_sha256":sha256("\n".join(
                r["digest"] for r in rows).encode()).hexdigest(),
            "arms":{k:{"future_correct":v["future_correct"],
                       "false_accept":v["false_accept"],
                       "paid_checks":v["paid_checks_total"]} for k,v in arms.items()}
        })
    summaries={}
    for kind,records in all_records.items():
        allfamilies={fam:[r for r in records if r["family"]==fam] for fam in FAMILIES}
        primary=[r for r in records if r["family"]!="late_source"]
        summaries[kind]={
            "cases":len(records),
            "future_correct":sum(r["future_correct"] for r in records),
            "false_accept":sum(r["false_accept"] for r in records),
            "paid_checks":sum(r["paid_checks"] for r in records),
            "nonlate_cases":len(primary),
            "nonlate_correct":sum(r["future_correct"] for r in primary),
            "nonlate_false_accept":sum(r["false_accept"] for r in primary),
            "nonlate_unique_inputs":len({r["public_features_sha256"] for r in primary}),
            "by_family":{f:{"cases":len(z),
                            "future_correct":sum(v["future_correct"] for v in z),
                            "false_accept":sum(v["false_accept"] for v in z)}
                         for f,z in allfamilies.items()}
        }
    records=all_records
    primary={
        "residual_vs_original":paired(records["original"],records["residual_early"]),
        "residual_vs_reconstructed":paired(records["reconstructed_early"],records["residual_early"]),
        "residual_vs_endpoint_nonlinear":paired(records["endpoint_nonlinear"],records["residual_early"]),
        "residual_vs_shuffled_early":paired(records["shuffled_early"],records["residual_early"]),
        "raw_vs_residual":paired(records["residual_early"],records["raw_early"]),
        "residual_vs_hidden_only":paired(records["hidden_only"],records["residual_early"])
    }
    hypothesis=("residual_vs_original","residual_vs_reconstructed",
                "residual_vs_endpoint_nonlinear")
    strong=all(primary[k]["net"]>0 and primary[k]["unadjusted_exact_sign_p"]<=.05/3
               for k in hypothesis)
    expected_nonlate=5*nonlate*reps
    assert all(v["nonlate_unique_inputs"]==expected_nonlate for v in summaries.values())
    assert all(v["paid_checks"]<=4*v["cases"] for v in summaries.values())
    return {
        "schema":SCHEMA,"source_checkpoint_sha256":old.BASIN_SHA,
        "frozen_probe_sha256":provenance["frozen_probe_json_sha256"],
        "original_train_manifest":provenance["manifests"]["train"],
        "original_dev_manifest":provenance["manifests"]["dev"],
        "source_cases_train":len(train),"source_cases_dev":len(dev),
        "shared_public_train_template_checks":16*len(train),
        "per_model_public_label_exposures":16*len(train),
        "trained_readout_slots":400,"training_epochs":epochs,
        "reconstructor":reconstructor,
        "model_weights_sha256":{kind:sha256(json.dumps(v["weights"],sort_keys=True,
            separators=(",",":")).encode()).hexdigest() for kind,v in models.items()},
        "model_training_trace":{k:v["trace"] for k,v in models.items()},
        "dev_local_scales":chosen,"dev_global_selected":global_choice,
        "dev_policy_grid":grid,"fresh_seed_offset":SEED_OFFSET,
        "repetitions":reps,
        "nonlate_per_family_per_repetition":nonlate,
        "late_source_per_repetition":late,
        "cohorts":result_by_rep,
        "aggregated":summaries,
        "predeclared_primary_paired_signs":primary,
        "bonferroni_three_control_threshold":.05/3,
        "residual_early_passed_preregistered_primary":strong,
        "records":records,
        "scope":{
            "original_checkpoint_immutable":True,
            "no_private_future_used_for_train":True,
            "source_input_and_feature_disjoint_previous_NOR11_12_13_14_15A":True,
            "single_late_source_hash_secondary":True,
            "global_selection_frozen_on_36_dev":True,
            "test_inference_four_paid_checks":True,
            "readout_marginal_gain_is_not_conditional_mutual_information_proof":True,
            "primary_unique_input_group_unit":True,
        }
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repetitions",type=int,default=REPS)
    p.add_argument("--nonlate-per-family",type=int,default=NONLATE_PER)
    p.add_argument("--late-per-repetition",type=int,default=LATE_PER)
    p.add_argument("--epochs",type=int,default=EPOCHS)
    p.add_argument("--check",action="store_true")
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    out=run(args.repetitions,args.nonlate_per_family,args.late_per_repetition,args.epochs)
    if args.check:
        assert out["source_checkpoint_sha256"]==old.BASIN_SHA
        assert out["source_cases_train"]==96 and out["source_cases_dev"]==36
        assert out["shared_public_train_template_checks"]==1536
        assert out["trained_readout_slots"]==400
        assert out["aggregated"]["original"]["nonlate_cases"]==(
            args.repetitions*5*args.nonlate_per_family)
        assert out["aggregated"]["original"]["cases"]==(
            args.repetitions*(5*args.nonlate_per_family+args.late_per_repetition))
    rendered=json.dumps(out,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(rendered)
    else:print(rendered,end="")


if __name__=="__main__":
    main()
