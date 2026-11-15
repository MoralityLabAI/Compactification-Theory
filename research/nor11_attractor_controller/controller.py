"""NOR-11: learned basin head and intervention on a frozen trained RRC25 GRU.

One hundred percent standard-library CPU, with an independently source-locked
1000-parameter trained GRU. An auxiliary linear sigmoid head learns to
predict which of the two NOR-07 *certified* attractors is approached during
a 720-step zero-input continuation. Its labels are FP64 trajectory
diagnostics, NOT newly interval-certified basin entries.

Train: only the original RRC25 training families. Development selects
the intervention strength/threshold using original native verification
outcomes. Test: separate public-feature tensors, six source families.
The model weights and 16-class native ranker never change.

Evaluate exact 4-check template verification and future correctness on
baseline, frozen trained-head control, fixed intervention, unconditional
center shift and a shuffled-label negative control. Source programs and
future truth are consulted ONLY in the final evaluation function.
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
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
for path in (ROOT,HERE):
    if str(path) not in sys.path:
        sys.path.insert(0,str(path))

from research.rrc_compactification import compound25 as corpus
from research.rrc_compactification import bench25,model25
from research.nor01_numerical_orbit_sketch import nor06_basin_survey as survey
from research.nor01_numerical_orbit_sketch import trained_gru_cert as recurrent
from research.nor01_numerical_orbit_sketch.invariant_ball import CENTER as FIRST

CHECKPOINT_SHA=survey.KNOWN_SHA
TRAIN_FAMILIES=tuple(corpus.TRAIN_FAMILIES)
TEST_FAMILIES=tuple(corpus.TEST_FAMILIES)
COUNTS={"train":32,"dev":12,"test":12}
SPLIT_SEED_OFFSETS={"train":12110000,"dev":12220000,"test":12330000}
SEED=2026111511
FUTURE_ZERO_STEPS=720
ATTRACTOR_MAX_DISTANCE=.05
PAID_TEMPLATE_CHECK_BUDGET=4
HEAD_EPOCHS=240
HEAD_LEARNING_RATE=.035
HEAD_L2=.015
TARGET_GUARD_THRESHOLD=.65
TARGET_GUARD_ALPHA=.5
SELECT_THRESHOLDS=(.6,.8)
SELECT_ALPHAS=(.125,.25,.5)
SECOND_FILE=HERE.parent/"nor01_numerical_orbit_sketch"/"results"/"nor07_two_attractors_summary.json"


def train_source():
    ck,digests=bench25.checkpoints()
    assert digests["recurrent"]==CHECKPOINT_SHA
    assert ck["recurrent"]["parameters"]==1000
    doc=json.loads(SECOND_FILE.read_text())
    assert doc["original_checkpoint_sha256"]==CHECKPOINT_SHA
    assert doc["two_locally_attracting_fixed_points_certified"]
    p1=tuple(float(x) for x in FIRST)
    p2=tuple(float(x) for x in doc["second_attractor"]["center_decimals"])
    assert len(p1)==len(p2)==model25.HIDDEN==12
    assert math.dist(p1,p2)>5
    return ck["recurrent"],p1,p2


def public_digest(case,header):
    x=corpus.features(case,header)
    return sha256(json.dumps(x,separators=(",",":")).encode()).hexdigest()


def sample_split(split,n,seen):
    if split not in SPLIT_SEED_OFFSETS or not 1<=n<=128:
        raise ValueError("unknown split or bounded sample count")
    families=TRAIN_FAMILIES if split!="test" else TEST_FAMILIES
    out=[]
    for ix,family in enumerate(families):
        rng=random.Random(corpus.SEED+SPLIT_SEED_OFFSETS[split]+10007*ix)
        found=0
        for trial in range(n*300):
            case=corpus.make_case(rng,family,f"nor11-{split}-{family}-{trial:05d}")
            headers=corpus.possible_headers(case)
            if len(headers)!=1:
                continue
            digest=public_digest(case,headers[0])
            if digest in seen:
                continue
            seen.add(digest)
            out.append((case,headers[0],digest))
            found+=1
            if found>=n:
                break
        if found!=n:
            raise RuntimeError("exhausted distinct public tensors for "+split+"/"+family)
    return out


def sample_cohorts(counts=COUNTS):
    seen=set()
    all_splits={}
    for name in ("train","dev","test"):
        all_splits[name]=sample_split(name,counts[name],seen)
    expected=len(TRAIN_FAMILIES)*(counts["train"]+counts["dev"])+len(TEST_FAMILIES)*counts["test"]
    assert len(seen)==expected
    return all_splits


def model_prefix_hidden(case,header,ck):
    h,parity,heads=survey.prefix_hidden(case,ck)
    assert heads==1 and parity<1e-11
    baseline=model25.logits(case,header,ck)
    direct=model25.affine(ck["state_dict"]["head.weight"],
                          ck["state_dict"]["head.bias"],h)
    assert max(abs(x-y) for x,y in zip(baseline,direct))<1e-11
    return h,parity


def basin_mode(h,ck,p1,p2):
    # Diagnostic outcome of the frozen GRU, rather than a trained model
    # supervision target drawn from private synthetic program labels.
    cur=list(h)
    for _ in range(FUTURE_ZERO_STEPS):
        cur=recurrent.native_zero_step(cur,ck)
    d1=math.dist(cur,p1)
    d2=math.dist(cur,p2)
    mode="first" if d1<=ATTRACTOR_MAX_DISTANCE and d1<d2 else (
        "second" if d2<=ATTRACTOR_MAX_DISTANCE and d2<d1 else "unresolved")
    return mode,d1,d2


def build_rows(cohorts,ck,p1,p2):
    outputs={}
    for split,items in cohorts.items():
        rows=[]
        for case,header,digest in items:
            h,parity=model_prefix_hidden(case,header,ck)
            mode,d1,d2=basin_mode(h,ck,p1,p2)
            rows.append({
                "case":case,"header":header,"digest":digest,
                "hidden":h,"native_logit_parity":parity,
                "mode":mode,"label":int(mode=="second"),
                "reference_distance_first":d1,
                "reference_distance_second":d2,
            })
        outputs[split]=rows
    return outputs


def logistic(x):
    if x>=0:
        return 1/(1+math.exp(-x))
    z=math.exp(x)
    return z/(1+z)


def standardizer(train):
    n=len(train)
    mu=[math.fsum(z["hidden"][j] for z in train)/n for j in range(12)]
    scale=[max(.02,math.sqrt(math.fsum((z["hidden"][j]-mu[j])**2 for z in train)/n))
           for j in range(12)]
    return mu,scale


def features(h,mu,scale):
    return [(v-m)/s for v,m,s in zip(h,mu,scale,strict=True)]+[1.0]


def fit_head(train,seed=SEED,labels_override=None):
    if not train or any(row["mode"]=="unresolved" for row in train):
        raise ValueError("all train rows need diagnostic basin labels")
    mu,scale=standardizer(train)
    examples=[features(row["hidden"],mu,scale) for row in train]
    actual=[z["label"] for z in train]
    labels=actual if labels_override is None else list(labels_override)
    if len(labels)!=len(train):
        raise ValueError("head supervision length mismatch")
    if not set(labels).issubset({0,1}) or len(set(labels))!=2:
        raise ValueError("binary train labels with both classes required")
    # Balanced training only. The optimizer sees no source identity, header,
    # hidden program or future task labels; those remain evaluation-only.
    counts=Counter(labels)
    sample_weight=[len(labels)/(2*counts[y]) for y in labels]
    w=[0.0]*13
    m=[0.0]*13
    v=[0.0]*13
    trajectory=[]
    for step in range(1,HEAD_EPOCHS+1):
        g=[0.0]*13
        total=0.0
        for row,y,weight in zip(examples,labels,sample_weight,strict=True):
            p=logistic(math.fsum(a*b for a,b in zip(row,w)))
            total+=weight*(-y*math.log(max(p,1e-12))-
                           (1-y)*math.log(max(1-p,1e-12)))
            for j in range(13):
                g[j]+=weight*(p-y)*row[j]
        for j in range(13):
            g[j]/=len(labels)
            if j!=12:
                g[j]+=HEAD_L2*w[j]
            m[j]=.9*m[j]+.1*g[j]
            v[j]=.999*v[j]+.001*g[j]*g[j]
            w[j]-=HEAD_LEARNING_RATE*(m[j]/(1-.9**step))/(
                 math.sqrt(v[j]/(1-.999**step))+1e-8)
        if step in (1,80,160,240):
            trajectory.append({"step":step,"weighted_cross_entropy":total/len(labels)})
    return {"weights":w,"mean":mu,"scale":scale,"seed":seed,
            "epochs":HEAD_EPOCHS,"train_counts":dict(counts),
            "learning_rate":HEAD_LEARNING_RATE,
            "l2":HEAD_L2,"training_trace":trajectory,
            "frozen_base_checkpoint_sha256":CHECKPOINT_SHA}


def head_score(h,head):
    vector=features(h,head["mean"],head["scale"])
    return logistic(math.fsum(a*b for a,b in zip(vector,head["weights"],strict=True)))


def predicted_metrics(rows,head):
    verified=[r for r in rows if r["mode"]!="unresolved"]
    t=Counter()
    for row in verified:
        p=head_score(row["hidden"],head)
        truth=row["label"]
        guessed=int(p>=.5)
        t[(truth,guessed)]+=1
    tp=t[(1,1)];fp=t[(0,1)];fn=t[(1,0)];tn=t[(0,0)]
    brier=math.fsum((head_score(r["hidden"],head)-r["label"])**2 for r in verified)/max(1,len(verified))
    pos=[head_score(r["hidden"],head) for r in verified if r["label"]==1]
    neg=[head_score(r["hidden"],head) for r in verified if r["label"]==0]
    auc= (sum((p>n)+.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))
          if pos and neg else None)
    return {"n":len(verified),"unresolved":len(rows)-len(verified),
            "first":tn+fp,"second":tp+fn,
            "accuracy":(tp+tn)/max(1,len(verified)),
            "balanced_accuracy":.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1)),
            "brier":brier,"auc":auc,"tp":tp,"fp":fp,"tn":tn,"fn":fn}


def intervene(h,alpha,p1):
    if not 0<=alpha<=1:
        raise ValueError("intervention alpha outside [0,1]")
    return [max(-1,min(1,(1-alpha)*v+alpha*c)) for v,c in zip(h,p1,strict=True)]


def score_with_hidden(case,header,h,ck):
    # Exactly one decoded public header per case is pinned at sampling.
    # The same max 4 paid checks and original exact future grader are used
    # in every arm. This preserves the native RRC25 ranking protocol.
    scores=model25.affine(ck["state_dict"]["head.weight"],
                          ck["state_dict"]["head.bias"],h)
    order=sorted(range(model25.VOCAB),key=lambda j:(-scores[j],j))
    found=None
    checks=0
    for j in order[:PAID_TEMPLATE_CHECK_BUDGET]:
        checks+=1
        u=corpus.OPS[j//4]
        v=corpus.OPS[j%4]
        if corpus.verify_template(case,header,u,v):
            found=(header.base,header.delay,header.spacing,header.amplitude,u,v)
            break
    # Future/private program correctness is read *only here*, at scoring.
    score=corpus.grade(case,found)
    return {**score,"paid_checks":checks,
            "top1_class":order[0],
            "has_verified_proposal":found is not None}


def evaluate(rows,ck,head,p1,threshold,alpha,mode,per_case_modes=False,predict_head=None):
    stats=Counter()
    by_family=defaultdict(Counter)
    baseline_futures=[]
    modes={"first_to_second":0,"second_to_first":0,
           "same":0,"other":0}
    records=[]
    for row in rows:
        source_h=row["hidden"]
        q=head_score(source_h,predict_head or head)
        if mode=="baseline":
            action=False
        elif mode=="conditioned":
            action=q>=threshold and alpha>0
        elif mode=="unconditional":
            action=alpha>0
        elif mode=="random_control":
            action=q>=threshold and alpha>0
        else:
            raise ValueError("unknown controller mode")

        changed=list(source_h)
        if action:
            if mode=="random_control":
                # Matching the source->first-target perturbation norm
                # but scrambling direction, while clipping cube bounds.
                rng=random.Random(int(row["digest"][:16],16)+5105)
                z=[rng.gauss(0,1) for _ in source_h]
                norm=math.hypot(*z)
                distance=math.dist(source_h,p1)*alpha
                changed=[max(-1,min(1,h+distance*v/norm))
                         for h,v in zip(source_h,z,strict=True)]
            else:
                changed=intervene(source_h,alpha,p1)
        assert all(-1<=v<=1 for v in changed)
        scored=score_with_hidden(row["case"],row["header"],changed,ck)
        stats["cases"]+=1
        stats["actions"]+=int(action)
        for key in ("accepted","future_correct","false_accept","paid_checks"):
            stats[key]+=scored[key]
            by_family[row["case"].family][key]+=scored[key]
        by_family[row["case"].family]["cases"]+=1
        by_family[row["case"].family]["actions"]+=int(action)
        if per_case_modes and action:
            after,_,_=basin_mode(changed,ck,p1,EVAL_P2[0])
            old=row["mode"]
            if old==after:
                modes["same"]+=1
            elif old=="first" and after=="second":
                modes["first_to_second"]+=1
            elif old=="second" and after=="first":
                modes["second_to_first"]+=1
            else:
                modes["other"]+=1
        if per_case_modes:
            records.append({
                "public_features_sha256":row["digest"],
                "family":row["case"].family,
                "original_mode":row["mode"],
                "probe_p_second":q,
                "action_taken":action,
                "native_future_correct":scored["future_correct"],
                "native_accepted":scored["accepted"],
                "native_false_accept":scored["false_accept"],
                "paid_checks":scored["paid_checks"],
            })
    n=max(1,stats["cases"])
    return {
        "mode":mode,"threshold":threshold,"alpha":alpha,
        "cases":stats["cases"],"actions":stats["actions"],
        "action_rate":stats["actions"]/n,
        "accepted":stats["accepted"],
        "future_correct":stats["future_correct"],
        "future_correct_rate":stats["future_correct"]/n,
        "false_accept":stats["false_accept"],
        "false_accept_rate":stats["false_accept"]/n,
        "paid_checks_total":stats["paid_checks"],
        "mean_paid_checks":stats["paid_checks"]/n,
        "family_counts":{name:dict(cnt) for name,cnt in sorted(by_family.items())},
        "mode_switches_if_actuated":modes if per_case_modes else None,
        "per_case":records if per_case_modes else [],
    }


# The fixed second center is initialized once by run() to avoid changing the
# task interface of the original train/model/checkpoint repository.
EVAL_P2=((),)


def choose_dev_policy(dev,ck,head,p1):
    candidates=[{"alpha":0.0,"threshold":1.0,"label":"no_intervention"}]
    candidates.extend({"alpha":a,"threshold":t,"label":"conditional"}
                      for t in SELECT_THRESHOLDS for a in SELECT_ALPHAS)
    result=[]
    for c in candidates:
        score=evaluate(dev,ck,head,p1,c["threshold"],c["alpha"],
                       "conditioned" if c["alpha"] else "baseline")
        # Native task selection only on development, with explicit penalty
        # for false accepts and intervention frequency, not hidden labels.
        utility=score["future_correct"]-.5*score["false_accept"]-.05*score["actions"]
        result.append({**c,"dev_future_correct":score["future_correct"],
                       "dev_false_accept":score["false_accept"],
                       "dev_actions":score["actions"],"dev_utility":utility})
    # On ties use fewer actions and lower alpha, including abstention.
    selected=max(result,key=lambda z:(z["dev_utility"],-z["dev_actions"],-z["alpha"]))
    return selected,result


def run(counts=COUNTS):
    ck,p1,p2=train_source()
    global EVAL_P2
    EVAL_P2=(p2,)
    cases=sample_cohorts(counts)
    rows=build_rows(cases,ck,p1,p2)
    train=[x for x in rows["train"] if x["mode"]!="unresolved"]
    if len({z["label"] for z in train})<2:
        raise AssertionError("training cohort lacks both dynamical basins")
    head=fit_head(train)
    scrambled=list(z["label"] for z in train)
    random.Random(SEED+77).shuffle(scrambled)
    if len({z for z in scrambled})<2:
        raise AssertionError("shuffled training labels degenerated")
    shuffled_head=fit_head(train,seed=SEED+77,labels_override=scrambled)
    chosen,candidates=choose_dev_policy(rows["dev"],ck,head,p1)
    test=rows["test"]
    protocol={
        "original":evaluate(test,ck,head,p1,1.0,0,"baseline",True),
        "selected":evaluate(test,ck,head,p1,chosen["threshold"],chosen["alpha"],
                            "conditioned",True),
        "fixed_escape":evaluate(test,ck,head,p1,TARGET_GUARD_THRESHOLD,
                                 TARGET_GUARD_ALPHA,"conditioned",True),
        "unconditional":evaluate(test,ck,head,p1,0.,TARGET_GUARD_ALPHA,
                                 "unconditional",True),
        "shuffled_head_escape":evaluate(test,ck,head,p1,TARGET_GUARD_THRESHOLD,
                                        TARGET_GUARD_ALPHA,"conditioned",True,
                                        predict_head=shuffled_head),
        "random_direction":evaluate(test,ck,head,p1,TARGET_GUARD_THRESHOLD,
                                     TARGET_GUARD_ALPHA,"random_control",True),
    }
    provenance={
        split:{
            "cases":len(group),
            "distinct_public_features":len({z["digest"] for z in group}),
            "source_families":dict(Counter(z["case"].family for z in group)),
            "reference_modes":dict(Counter(z["mode"] for z in group)),
            "public_feature_manifest_sha256":sha256(
                "\n".join(z["digest"] for z in group).encode()).hexdigest(),
        } for split,group in rows.items()
    }
    assert len(set(z["digest"] for group in rows.values() for z in group))==sum(
        len(z) for z in rows.values())
    base=protocol["original"]
    assert base["cases"]==sum(len(x) for x in rows["test"] for _ in [0])
    assert base["actions"]==0
    assert all(s["paid_checks_total"]<=PAID_TEMPLATE_CHECK_BUDGET*s["cases"]
               for s in protocol.values())
    return {
        "schema":"nor11.frozen_gru.trained_basin_head_and_escape_controller.v1",
        "source_checkpoint_sha256":CHECKPOINT_SHA,
        "experiment_seed":SEED,
        "frozen_gru_parameters":ck["parameters"],
        "trained_auxiliary_head_parameters":13,
        "trained_gru_weights_modified":False,
        "optimizer":"deterministic full-batch balanced logistic Adam, CPU stdlib",
        "head":head,"shuffled_label_control_head":shuffled_head,
        "cohorts":provenance,
        "head_diagnostics":{
            split:{"trained":predicted_metrics(group,head),
                   "shuffled_label":predicted_metrics(group,shuffled_head)}
            for split,group in rows.items()
        },
        "development_policy_grid":candidates,
        "selected_policy":chosen,
        "native_template_verifier_budget":PAID_TEMPLATE_CHECK_BUDGET,
        "heldout_protocol":protocol,
        "declared_fixed_intervention":{
            "probability_threshold":TARGET_GUARD_THRESHOLD,
            "first_attractor_interpolation_alpha":TARGET_GUARD_ALPHA,
            "target":"certified first period-one fixed-point center, as an FP64 vector",
        },
        "train_restrictions":{
            "head_input":"12D recurrent hidden state after 9 public feature rows",
            "head_target":"FP64 720-zero-step attractor proximity label from frozen GRU",
            "native_future_truth_used_to_fit_head":False,
            "future_truth_used_in_development_policy_selection":True,
            "future_truth_used_in_heldout_scoring_only":True,
            "original_checkpoint_frozen":True,
            "no_model_gru_gradient_update":True,
            "source_future_uses_for_controller":False,
            "per_new_sample_source_basin_interval_certified":False,
            "native_proof_search_performance_gain_claimed":False,
        },
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-family",type=int,default=COUNTS["train"])
    parser.add_argument("--dev-per-family",type=int,default=COUNTS["dev"])
    parser.add_argument("--test-per-family",type=int,default=COUNTS["test"])
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    c={"train":args.train_per_family,"dev":args.dev_per_family,
       "test":args.test_per_family}
    result=run(c)
    if args.check:
        assert result["frozen_gru_parameters"]==1000
        assert not result["trained_gru_weights_modified"]
        assert result["cohorts"]["train"]["cases"]==len(TRAIN_FAMILIES)*c["train"]
        assert result["cohorts"]["dev"]["cases"]==len(TRAIN_FAMILIES)*c["dev"]
        assert result["cohorts"]["test"]["cases"]==len(TEST_FAMILIES)*c["test"]
        assert result["native_template_verifier_budget"]==4
        assert not result["train_restrictions"]["native_future_truth_used_to_fit_head"]
    text=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(text)
    else:
        print(text,end="")


if __name__=="__main__":
    main()
