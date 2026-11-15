"""NOR-13: genuinely basin-gated task readout for a frozen trained RRC25 GRU.

The original 1000-parameter recurrent checkpoint, 16-class candidate head,
and existing 13-parameter basin probe are frozen. This experiment fits
only PUBLIC exact-demo supervised residual heads, 16 x 26 parameters each.

Rather than adding the basin probability as one class-offset feature
(NOR-12), the new head uses the interaction (2q(h)-1)*h. This permits
separate readout slopes in the two recurrent modes while retaining the
original hidden state for all native tasks.

Matched 416-parameter-slot controls: shuffled-label NOR11 probe gate,
fixed seeded random sigmoid projection of hidden state, constant gate,
and additive-only basin scalar. All train on identical public accepted
template sets and have separately dev-selected amplitudes. The first
heldout source split and FOUR NEW seed-disjoint 72-case rosters are
evaluated without updating either frozen source or trained task weights.

A gate-knockout and feature-group permutation on the fresh cohorts is
predeclared in this source file, with no further model selection.
The late_source family can alias many visible cases to one 9x8 tensor;
every result reports 72 case outcomes and effective unique input count.
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

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
for directory in (ROOT,HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))

from research.rrc_compactification import compound25 as corpus,model25
from research.nor11_attractor_controller import controller as prior
from research.nor12_task_readout import readout as previous
from research.nor12_task_readout import independent_confirmation as confirmed

SOURCE_SHA=previous.BASIN_SHA
SEED=2026111513
CONFIRM_SEED=96110000
TRAIN_COUNTS=dict(previous.SPLIT_COUNTS)
REP_COUNT=4
PER_FAMILY=12
KINDS=("basin_interaction","shuffled_basin_interaction",
       "random_projection_interaction","flat_readout","additive_only_basin")
WIDTH=26
OUTPUTS=16
EPOCHS=220
LR=.023
RIDGE=.1
SCALES=(.125,.25,.5,1.0)
BUDGET=4
RANDOM_VECTOR_SEED=2026111321


def sigmoid(x):
    return prior.logistic(x)


def frozen_gate_inputs():
    ck,probe,probe_sha,old=previous.locked_rows(TRAIN_COUNTS)
    cohort=prior.sample_cohorts(TRAIN_COUNTS)
    p1,p2=prior.train_source()[1:]
    diag=prior.build_rows({"train":cohort["train"]},ck,p1,p2)["train"]
    perm=[r["label"] for r in diag]
    random.Random(prior.SEED+77).shuffle(perm)
    neg=prior.fit_head(diag,seed=prior.SEED+77,labels_override=perm)
    assert probe_sha==json.loads(previous.BASELINE_SUMMARY.read_text() if
                                 hasattr(previous,"BASELINE_SUMMARY") else
                                 (ROOT/"research"/"nor11_attractor_controller"/"results"/"nor11_executed_summary.json").read_text()
                                 )["source_checkpoint_sha256"] if False else True
    assert ck["parameters"]==1000
    attach_gated_features(old,probe,neg)
    return ck,probe,neg,old,cohort,probe_sha


def random_projection():
    rng=random.Random(RANDOM_VECTOR_SEED)
    vec=[rng.gauss(0,1) for _ in range(12)]
    norm=math.sqrt(sum(v*v for v in vec))
    return tuple(v/norm for v in vec)


PROJECTION=random_projection()


def gated_row(h,digest,probe,negative):
    z=prior.features(h,probe["mean"],probe["scale"])
    if len(z)!=13 or z[-1]!=1.0:
        raise AssertionError("NOR11 feature standardizer changed")
    basin=2*prior.head_score(h,probe)-1
    permuted=2*prior.head_score(h,negative)-1
    random_gate=2*sigmoid(sum(w*x for w,x in zip(PROJECTION,z[:12])))-1
    gate={
        "basin_interaction":basin,
        "shuffled_basin_interaction":permuted,
        "random_projection_interaction":random_gate,
        "flat_readout":0.0,
        "additive_only_basin":basin,
    }
    feat={}
    for kind,g in gate.items():
        if kind=="additive_only_basin":
            mod=[0.0]*12+[g]
        elif kind=="flat_readout":
            mod=[0.0]*13
        else:
            mod=[g*v for v in z]
        feat[kind]=z+mod
        if len(feat[kind])!=WIDTH:
            raise AssertionError("wrong conditional head width")
    return feat,gate


def attach_gated_features(rows,probe,negative):
    for split,group in rows.items():
        for r in group:
            feature,gate=gated_row(r["h"],r["digest"],probe,negative)
            r["gate_features"]=feature
            r["gate_values"]=gate


def corrected_scores(row,weights,kind,scale):
    if kind not in KINDS or not 0<=scale<=1:
        raise ValueError("unknown or out-of-grid residual")
    inputs=row["gate_features"][kind]
    return [base+scale*math.fsum(a*b for a,b in zip(w,inputs,strict=True))
            for base,w in zip(row["base"],weights,strict=True)]


def gradient(row,W,kind):
    probabilities=previous.softmax(corrected_scores(row,W,kind,1.0))
    mask=row["visible_mask"]
    mass=math.fsum(p for p,ok in zip(probabilities,mask,strict=True) if ok)
    if mass<=0:
        raise ArithmeticError("no valid observable candidate mass")
    feature=row["gate_features"][kind]
    gr=[]
    for j,p in enumerate(probabilities):
        delta=p-(p/mass if mask[j] else 0.0)
        gr.append([delta*x for x in feature])
    return -math.log(mass),gr


def fit(group,kind):
    if len(group)<8 or kind not in KINDS:
        raise ValueError("invalid source or gate")
    if any("visible_mask" not in r for r in group):
        raise ValueError("training requires public-visible candidate checks")
    W=[[0.0]*WIDTH for _ in range(OUTPUTS)]
    m=[[0.0]*WIDTH for _ in range(OUTPUTS)]
    v=[[0.0]*WIDTH for _ in range(OUTPUTS)]
    trace=[]
    for t in range(1,EPOCHS+1):
        grads=[[0.0]*WIDTH for _ in range(OUTPUTS)]
        loss=0
        for row in group:
            ce,g=gradient(row,W,kind)
            loss+=ce
            for j in range(OUTPUTS):
                for k in range(WIDTH):
                    grads[j][k]+=g[j][k]
        for j in range(OUTPUTS):
            for k in range(WIDTH):
                der=grads[j][k]/len(group)+RIDGE*W[j][k]
                m[j][k]=.9*m[j][k]+.1*der
                v[j][k]=.999*v[j][k]+.001*der*der
                W[j][k]-=LR*(m[j][k]/(1-.9**t))/(
                    math.sqrt(v[j][k]/(1-.999**t))+1e-8)
        if t in (1,55,110,165,220):
            trace.append({"update":t,"public_accepted_mass_loss":loss/len(group)})
    return {
        "kind":kind,"weights":W,"parameter_slots":OUTPUTS*WIDTH,
        "effective_parameter_slots":
          OUTPUTS*(14 if kind=="additive_only_basin" else
                   13 if kind=="flat_readout" else WIDTH),
        "train_only_exact_public_candidate_checks":OUTPUTS*len(group),
        "epochs":EPOCHS,"learning_rate":LR,"ridge":RIDGE,
        "train_loss":trace,
        "private_256_degree_future_used_to_fit":False,
    }


def evaluate(rows,model=None,kind=None,scale=0):
    total=Counter()
    by_family=defaultdict(Counter)
    records=[]
    for r in rows:
        if model is None:
            assert kind is None and scale==0
            logits=list(r["base"])
        else:
            logits=corrected_scores(r,model["weights"],kind,scale)
        x=previous.run_four_checks(r,logits)
        total["cases"]+=1
        for key in ("accepted","future_correct","false_accept","paid_checks"):
            total[key]+=x[key]
            by_family[r["case"].family][key]+=x[key]
        by_family[r["case"].family]["cases"]+=1
        records.append({
            "public_features_sha256":r["digest"],
            "family":r["case"].family,
            "future_correct":x["future_correct"],
            "accepted":x["accepted"],
            "false_accept":x["false_accept"],
            "paid_checks":x["paid_checks"],
            "top1":x["top_1"],
            "verified_class":x["verified_class"],
        })
    if total["accepted"]!=total["future_correct"]+total["false_accept"]:
        raise AssertionError("original grade conservation violated")
    return {
        "cases":total["cases"],
        "unique_input_tensors":len({r["digest"] for r in rows}),
        "future_correct":total["future_correct"],
        "accepted":total["accepted"],
        "false_accept":total["false_accept"],
        "paid_checks_total":total["paid_checks"],
        "by_family":{f:dict(v) for f,v in by_family.items()},
        "records":records,
    }


def choice_on_dev(dev,heads):
    grid=[]
    base=evaluate(dev)
    grid.append({
        "kind":"original","scale":0.0,
        "future_correct":base["future_correct"],
        "false_accept":base["false_accept"],
        "paid_checks":base["paid_checks_total"]
    })
    for kind in KINDS:
        for alpha in SCALES:
            x=evaluate(dev,heads[kind],kind,alpha)
            grid.append({
                "kind":kind,"scale":alpha,
                "future_correct":x["future_correct"],
                "false_accept":x["false_accept"],
                "paid_checks":x["paid_checks_total"],
            })
    for x in grid:
        x["utility"]=x["future_correct"]-.5*x["false_accept"]-.015*x["paid_checks"]
    best=max(grid,key=lambda x:(x["utility"],
                                int(x["kind"]=="original"),-x["scale"]))
    local={k:max((x for x in grid if x["kind"]==k),
                 key=lambda x:(x["utility"],-x["scale"]))
           for k in KINDS}
    return best,local,grid


def fresh_source_cohorts(original,replicates=REP_COUNT,per_family=PER_FAMILY):
    if not 1<=replicates<=6 or not 1<=per_family<=24:
        raise ValueError("bounded new test cohort required")
    used_visible={confirmed.visible_digest(c) for group in original.values()
                  for c,header,d in group}
    train_dev={d for split in ("train","dev") for c,header,d in original[split]}
    nonlate={d for c,header,d in original["test"]}
    groups=[]
    for rep in range(replicates):
        records=[]
        for idx,family in enumerate(corpus.TEST_FAMILIES):
            rng=random.Random(corpus.SEED+CONFIRM_SEED+rep*1000003+idx*10007)
            count=0
            for trial in range(per_family*1500):
                candidate=corpus.make_case(
                    rng,family,f"nor13-fresh-{rep}-{family}-{trial:05d}")
                headers=corpus.possible_headers(candidate)
                if len(headers)!=1:
                    continue
                d=prior.public_digest(candidate,headers[0])
                visible=confirmed.visible_digest(candidate)
                if visible in used_visible or d in train_dev:
                    continue
                if family!="late_source" and d in nonlate:
                    continue
                used_visible.add(visible)
                if family!="late_source":
                    nonlate.add(d)
                records.append((candidate,headers[0],d))
                count+=1
                if count==per_family:
                    break
            if count!=per_family:
                raise RuntimeError(f"fresh source exhausted family {family} rep {rep}")
        groups.append(records)
    return groups


def make_test_rows(items,ck,probe,negative):
    rows=[]
    for c,header,d in items:
        h,parity=prior.model_prefix_hidden(c,header,ck)
        base=model25.affine(ck["state_dict"]["head.weight"],
                            ck["state_dict"]["head.bias"],h)
        rows.append({"case":c,"header":header,"digest":d,
                     "h":h,"p":prior.head_score(h,probe),"base":base})
    attach_gated_features({"heldout":rows},probe,negative)
    return rows


def paired(base,other):
    assert len(base["records"])==len(other["records"])
    pos=neg=changed=0
    clusters=defaultdict(lambda:[0,0])
    for x,y in zip(base["records"],other["records"],strict=True):
        if x["public_features_sha256"]!=y["public_features_sha256"]:
            raise AssertionError("unpaired source")
        delta=y["future_correct"]-x["future_correct"]
        pos+=int(delta>0)
        neg+=int(delta<0)
        changed+=int(x["top1"]!=y["top1"])
        group=clusters[x["public_features_sha256"]]
        group[0]+=delta
        group[1]+=1
    return {
        "newly_correct":pos,"lost_correct":neg,"changed_top1":changed,
        "positive_unique_feature_groups":sum(n>0 for n,k in clusters.values()),
        "negative_unique_feature_groups":sum(n<0 for n,k in clusters.values()),
        "net_group_weighted_accuracy_delta":sum(n/k for n,k in clusters.values())/len(clusters),
        "unique_feature_groups":len(clusters),
    }


def ablate_gate(rows,head,scale):
    baseline=evaluate(rows)
    first_features=[(r["digest"],r["gate_values"]["basin_interaction"])
                    for r in rows]
    grouped={}
    for d,v in first_features:
        if d in grouped and grouped[d]!=v:
            raise AssertionError("identical input has inconsistent frozen basin gate")
        grouped[d]=v
    keys=sorted(grouped)
    choices=[grouped[k] for k in keys]
    random.Random(SEED+31).shuffle(choices)
    perm=dict(zip(keys,choices,strict=True))
    substitutions={
      "true_basin_gate":lambda r:r["gate_values"]["basin_interaction"],
      "gate_zero":lambda r:0.,
      "shuffled_label_gate":lambda r:r["gate_values"]["shuffled_basin_interaction"],
      "permuted_feature_clusters":lambda r:perm[r["digest"]],
    }
    allout={}
    for name,func in substitutions.items():
        new=[]
        for row in rows:
            r=dict(row)
            r["gate_features"]=dict(row["gate_features"])
            z=row["gate_features"]["basin_interaction"][:13]
            g=func(row)
            r["gate_features"]["basin_interaction"]=z+[g*x for x in z]
            new.append(r)
        ev=evaluate(new,head,"basin_interaction",scale)
        allout[name]={
          "future_correct":ev["future_correct"],
          "false_accept":ev["false_accept"],
          "checks":ev["paid_checks_total"],
          "paired":paired(baseline,ev),
        }
    return {
      "type":"predeclared gate knockout and cluster permutation, weights fixed",
      "gate_permutation_seed":SEED+31,
      "variants":allout
    }


def run(reps=REP_COUNT,per_family=PER_FAMILY):
    ck,probe,negative,original,source_cohort,probe_sha=frozen_gate_inputs()
    train=original["train"]
    for r in train:
        r["visible_mask"]=previous.visible_mask(r)
    fitted={kind:fit(train,kind) for kind in KINDS}
    selected,local,grid=choice_on_dev(original["dev"],fitted)
    first=original["test"]
    old=evaluate(first)
    first_results={"original":old}
    for k in KINDS:
        first_results[k]=evaluate(first,fitted[k],k,local[k]["scale"])
    first_results["development_selected"]=(
        evaluate(first) if selected["kind"]=="original"
        else evaluate(first,fitted[selected["kind"]],
                      selected["kind"],selected["scale"]))
    first_summary={
        key:{
            "future_correct":v["future_correct"],
            "false_accept":v["false_accept"],
            "paid_checks":v["paid_checks_total"],
            "paired":paired(old,v),
        } for key,v in first_results.items()
    }
    all_new=fresh_source_cohorts(source_cohort,reps,per_family)
    count=Counter()
    compared={k:Counter() for k in first_results}
    by_rep=[]
    ablation=[]
    clusters={}
    for i,cohort in enumerate(all_new):
        rows=make_test_rows(cohort,ck,probe,negative)
        base=evaluate(rows)
        arms={"original":base}
        for k in KINDS:
            arms[k]=evaluate(rows,fitted[k],k,local[k]["scale"])
        arms["development_selected"]=(
          evaluate(rows) if selected["kind"]=="original"
          else evaluate(rows,fitted[selected["kind"]],
                        selected["kind"],selected["scale"]))
        rep_metrics={}
        for k,x in arms.items():
            stats=compared[k]
            for field in ("cases","future_correct","false_accept","paid_checks_total"):
                stats[field]+=x[field]
            pair=paired(base,x)
            stats["gains"]+=pair["newly_correct"]
            stats["losses"]+=pair["lost_correct"]
            rep_metrics[k]={
                "future_correct":x["future_correct"],
                "false_accept":x["false_accept"],
                "paid_checks":x["paid_checks_total"],
                "paired":pair,
                "by_family":x["by_family"],
            }
        # Planned manipulation of q on the SAME fully trained gated
        # readout. Every input cluster gets one common permuted q.
        ablation.append({
            "replicate":i,
            "ablation":ablate_gate(rows,fitted["basin_interaction"],
                                   local["basin_interaction"]["scale"])
        })
        clusters[i]=len({x["digest"] for x in rows})
        by_rep.append({
            "replicate":i,
            "seed":corpus.SEED+CONFIRM_SEED+i*1000003,
            "cases":len(rows),"unique_feature_tensors":clusters[i],
            "feature_manifest_sha256":sha256(
                "\n".join(r["digest"] for r in rows).encode()).hexdigest(),
            "arms":rep_metrics,
        })
    aggregate={}
    for k,v in compared.items():
        aggregate[k]={
            "cases":v["cases"],"future_correct":v["future_correct"],
            "false_accept":v["false_accept"],
            "paid_checks":v["paid_checks_total"],
            "paired_newly_correct":v["gains"],
            "paired_lost_correct":v["losses"],
            "net_case_gain":v["gains"]-v["losses"],
        }
    for k,v in aggregate.items():
        assert v["net_case_gain"]==v["future_correct"]-aggregate["original"]["future_correct"]
    return {
        "schema":"nor13.multiplicative_basin_x_hidden_task_readout.v1",
        "trained_gru_checkpoint_sha256":SOURCE_SHA,
        "original_gru_parameters":ck["parameters"],
        "original_gru_weights_modified":False,
        "original_task_readout_weights_modified":False,
        "frozen_probe_json_sha256":probe_sha,
        "new_readout_parameter_slots_per_arm":OUTPUTS*WIDTH,
        "training_cases":len(train),
        "development_cases":len(original["dev"]),
        "training_full_public_exact_template_checks":16*len(train),
        "training_target":"public accepted template mass only",
        "private_future_used_to_train_gated_readouts":False,
        "train_config":{"epochs":EPOCHS,"lr":LR,"ridge":RIDGE,
                        "gate_random_projection_seed":RANDOM_VECTOR_SEED},
        "fitted":fitted,
        "development":grid,
        "development_selected":selected,
        "per_kind_dev_choices":local,
        "original_test_results_retrospective":first_summary,
        "new_test_seed_offset":CONFIRM_SEED,
        "fresh_repetitions":reps,"new_cases_per_family":per_family,
        "fresh_replicates":by_rep,
        "fresh_aggregate":aggregate,
        "predeclared_gate_ablations_on_fresh_test":ablation,
        "scope":{
            "new_test_roster_seeds_fixed_in_source_before_results":True,
            "dev_only_hyperparameter_selection":True,
            "head_input_only_public_source_h_and_pretrained_probe":True,
            "test_future_truth_only_for_independent_scoring":True,
            "new_source_interval_capture_certified":False,
            "late_source_model_input_aliases_counted":True,
            "native_inference_template_check_budget":4,
            "train_time_exhaustive_public_templates_charged_separately":True,
            "underlying_GRU_retrained":False,
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--replicates",type=int,default=REP_COUNT)
    p.add_argument("--per-family",type=int,default=PER_FAMILY)
    p.add_argument("--out",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    record=run(args.replicates,args.per_family)
    if args.check:
        assert record["trained_gru_checkpoint_sha256"]==SOURCE_SHA
        assert record["new_readout_parameter_slots_per_arm"]==416
        assert record["training_full_public_exact_template_checks"]==1536
        assert not record["private_future_used_to_train_gated_readouts"]
        assert record["fresh_aggregate"]["original"]["cases"]==6*args.replicates*args.per_family
        for m,x in record["fresh_aggregate"].items():
            assert x["cases"]==record["fresh_aggregate"]["original"]["cases"]
            assert x["future_correct"]==x["paired_newly_correct"]-x["paired_lost_correct"]+record["fresh_aggregate"]["original"]["future_correct"]
    body=json.dumps(record,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(body)
    else:
        print(body,end="")


if __name__=="__main__":
    main()
