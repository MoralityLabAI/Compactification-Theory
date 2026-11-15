"""NOR-12: task-aware residual readout with an auxiliary basin coordinate.

The original 1,000-parameter RRC-25 recurrent checkpoint and its 16-way
template readout are immutable. Four deterministic full-batch residual
heads learn from PUBLIC, exactly checked degree-192 demonstration suffixes.
An original already-trained NOR-11 13-parameter basin probe is pinned.

Every inference arm ranks the same sixteen templates and performs at
most FOUR exact public candidate verifications, followed by the existing
independent withheld-future scorer. No private generating program or
future query is used for training any readout.

Primary: 16x14 basin-conditioned residual head.
Controls: 16x14 shuffled-basin, 16x14 random public-feature coordinate,
          16x14 hidden-only (the extra column is identically zero).
Dev-only scale selection includes the zero-correction original baseline.
Test: 72 source cases, of which only 61 feature tensors are distinct;
the twelve late_source cases alias to the same 9x8 model input.

The training objective is -log softmax probability MASS on all public-
accepted candidate templates. This is well-defined for ambiguous cases.
Its full 16-template public training oracle must be reported as a
distinct training cost and never counted as one of the four inference
verification checks.
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
for x in (HERE,ROOT):
    if str(x) not in sys.path:
        sys.path.insert(0,str(x))

from research.rrc_compactification import compound25 as corpus, model25
from research.nor11_attractor_controller import controller as previous

SEED=2026111512
BASIN_SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"
BASIN_HEAD_FILE=ROOT/"research"/"nor11_attractor_controller"/"results"/"nor11_auxiliary_head.json"
BASELINE_SUMMARY=ROOT/"research"/"nor11_attractor_controller"/"results"/"nor11_executed_summary.json"
BASELINE_MANIFESTS={
    "train":"8383cd3df9fe582497f1e1df84ba5a2d1d48a190a3f4a321f8fa9f35b040071c",
    "dev":"c236311a67369f453ee68c7d3d40acecb62e1c1a9c86867001410a71645d5af2",
    "test":"0058fb02ceb109b7ac36d6f3799c5fd097e247ff6a7ed1c5998420bd4681f545",
}
SPLIT_COUNTS=dict(previous.COUNTS)
KINDS=("basin","shuffled_basin","random_feature","hidden_only")
OUTPUTS=16
WIDTH=14
EPOCHS=200
LR=.025
RIDGE=.065
MIXING_SCALES=(0.0,.25,.5,1.0,2.0)
BUDGET=4
LOWER_BOUND=1e-12


def load_probe():
    raw=BASIN_HEAD_FILE.read_bytes()
    receipt=json.loads(raw)
    summary=json.loads(BASELINE_SUMMARY.read_text())
    assert receipt["schema"]=="nor11.auxiliary.basin_head.v1"
    assert receipt["source_checkpoint_sha256"]==BASIN_SHA
    assert receipt["native_future_truth_used_for_fit"] is False
    assert summary["source_checkpoint_sha256"]==BASIN_SHA
    assert receipt["public_training_feature_manifest_sha256"]==BASELINE_MANIFESTS["train"]
    head=receipt["head"]
    assert head["frozen_base_checkpoint_sha256"]==BASIN_SHA
    assert len(head["weights"])==13 and len(head["mean"])==12 and len(head["scale"])==12
    return head, sha256(raw).hexdigest()


def locked_rows(counts=SPLIT_COUNTS):
    checkpoint,p1,p2=previous.train_source()
    assert checkpoint["parameters"]==1000
    cohort=previous.sample_cohorts(counts)
    head,headsha=load_probe()
    training=[]
    for split,groups in cohort.items():
        rows=[]
        for case,header,digest in groups:
            hidden,parity=previous.model_prefix_hidden(case,header,checkpoint)
            baseline=model25.affine(checkpoint["state_dict"]["head.weight"],
                                    checkpoint["state_dict"]["head.bias"],hidden)
            assert len(baseline)==16
            q=previous.head_score(hidden,head)
            assert 0<=q<=1
            rows.append({"case":case,"header":header,"digest":digest,
                         "h":hidden,"base":baseline,"p":q,"parity":parity})
        training.append((split,rows))
    result=dict(training)
    if counts==SPLIT_COUNTS:
        for split,rows in result.items():
            manifest=sha256("\n".join(r["digest"] for r in rows).encode()).hexdigest()
            assert manifest==BASELINE_MANIFESTS[split],(split,manifest)
    assert set(result)=={"train","dev","test"}
    assert len({r["digest"] for split in ("train","dev") for r in result[split]})==len(result["train"])+len(result["dev"])
    assert not ({r["digest"] for r in result["test"]}&
                {r["digest"] for split in ("train","dev") for r in result[split]})
    return checkpoint,head,headsha,result


def random_scalar(digest):
    # Public-only deterministic negative feature; fixed without looking
    # at exact task outcomes, basin labels or private source programs.
    return 2*(int(digest[:13],16)/(16**13)) - 1


def attach_features(rows,probe,shuffled):
    for split,cohort in rows.items():
        for row in cohort:
            h=row["h"]
            z=previous.features(h,probe["mean"],probe["scale"])
            assert len(z)==13 and z[-1]==1.0
            s=previous.head_score(h,shuffled)
            row["x"]={
                "basin":z[:12]+[2*row["p"]-1,1.0],
                "shuffled_basin":z[:12]+[2*s-1,1.0],
                "random_feature":z[:12]+[random_scalar(row["digest"]),1.0],
                "hidden_only":z[:12]+[0.0,1.0],
            }


def visible_mask(row):
    # TRAIN-ONLY: exhaustive exact public candidate checks over two
    # observed demonstrations. No query-future evaluation is performed.
    case,header=row["case"],row["header"]
    bits=tuple(bool(corpus.verify_template(case,header,corpus.OPS[j//4],corpus.OPS[j%4]))
               for j in range(OUTPUTS))
    if not any(bits):
        raise AssertionError("positive RRC25 training prefix without public candidate")
    return bits


def softmax(scores):
    m=max(scores)
    ex=[math.exp(v-m) for v in scores]
    total=math.fsum(ex)
    return [x/total for x in ex]


def adjusted_scores(row,weights,kind,scale):
    if kind not in KINDS or not 0<=scale<=2:
        raise ValueError("unsupported readout configuration")
    feat=row["x"][kind]
    return [base+scale*math.fsum(v*x for v,x in zip(w,feat,strict=True))
            for base,w in zip(row["base"],weights,strict=True)]


def loss_gradient(row,weights,kind):
    scores=adjusted_scores(row,weights,kind,1.0)
    prob=softmax(scores)
    mask=row["visible_mask"]
    mass=math.fsum(p for p,b in zip(prob,mask,strict=True) if b)
    if mass<=0:
        raise ArithmeticError("underflow of valid mass")
    feat=row["x"][kind]
    grad=[]
    for j in range(OUTPUTS):
        coeff=prob[j]-(prob[j]/mass if mask[j] else 0.0)
        grad.append([coeff*x for x in feat])
    return -math.log(mass),grad


def fit_residual(train,kind):
    if kind not in KINDS or len(train)<8:
        raise ValueError("invalid training group")
    if any("visible_mask" not in r for r in train):
        raise ValueError("public training labels missing")
    W=[[0.0]*WIDTH for _ in range(OUTPUTS)]
    mt=[[0.0]*WIDTH for _ in range(OUTPUTS)]
    vt=[[0.0]*WIDTH for _ in range(OUTPUTS)]
    track=[]
    for t in range(1,EPOCHS+1):
        g=[[0.0]*WIDTH for _ in range(OUTPUTS)]
        ce=0.0
        for row in train:
            loss,gr=loss_gradient(row,W,kind)
            ce+=loss
            for j in range(OUTPUTS):
                for i in range(WIDTH):
                    g[j][i]+=gr[j][i]
        for j in range(OUTPUTS):
            for i in range(WIDTH):
                derivative=g[j][i]/len(train)+RIDGE*W[j][i]
                mt[j][i]=.9*mt[j][i]+.1*derivative
                vt[j][i]=.999*vt[j][i]+.001*derivative*derivative
                W[j][i]-=LR*(mt[j][i]/(1-.9**t))/(
                    math.sqrt(vt[j][i]/(1-.999**t))+1e-8)
        if t in (1,50,100,150,200):
            track.append({"epoch":t,"public_valid_mass_nll":ce/len(train)})
    return {
        "kind":kind,"weights":W,
        "parameter_slots":OUTPUTS*WIDTH,
        "effectively_active_features":WIDTH-(kind=="hidden_only"),
        "epochs":EPOCHS,"lr":LR,"ridge":RIDGE,
        "training_trace":track,
        "public_training_oracle_calls":OUTPUTS*len(train),
        "public_label_target":"logsumexp of the exactly accepted demo-template logits",
        "private_future_truth_access_during_fitting":False
    }


def run_four_checks(row,score):
    case,header=row["case"],row["header"]
    ordered=sorted(range(OUTPUTS),key=lambda j:(-score[j],j))
    accepted=None;checks=0
    for j in ordered[:BUDGET]:
        checks+=1
        if corpus.verify_template(case,header,corpus.OPS[j//4],corpus.OPS[j%4]):
            accepted=(header.base,header.delay,header.spacing,header.amplitude,
                      corpus.OPS[j//4],corpus.OPS[j%4])
            break
    # This is the *only* call site for the withheld 256-degree future.
    grade=corpus.grade(case,accepted)
    return {**grade,"paid_checks":checks,
            "top_1":ordered[0],
            "verified_class":ordered[checks-1] if accepted is not None else None,
            "trace":ordered[:checks]}


def evaluate(rows,correction=None,kind=None,scale=0.0):
    if correction is None:
        assert kind is None and scale==0
    elif kind not in KINDS:
        raise ValueError("invalid correction")
    totals=Counter()
    families=defaultdict(Counter)
    records=[]
    for row in rows:
        score=list(row["base"]) if correction is None else adjusted_scores(
            row,correction["weights"],kind,scale)
        result=run_four_checks(row,score)
        for key in ("accepted","future_correct","false_accept","paid_checks"):
            totals[key]+=result[key]
            families[row["case"].family][key]+=result[key]
        families[row["case"].family]["cases"]+=1
        records.append({
            "public_features_sha256":row["digest"],
            "family":row["case"].family,
            "top1":result["top_1"],
            "accepted":result["accepted"],
            "future_correct":result["future_correct"],
            "false_accept":result["false_accept"],
            "paid_checks":result["paid_checks"],
            "verified_class":result["verified_class"],
            "verification_trace":result["trace"],
        })
    n=len(rows)
    assert totals["accepted"]==totals["future_correct"]+totals["false_accept"]
    return {
        "cases":n,"distinct_feature_tensors":len({r["digest"] for r in rows}),
        "accepted":totals["accepted"],
        "future_correct":totals["future_correct"],
        "false_accept":totals["false_accept"],
        "paid_checks_total":totals["paid_checks"],
        "future_correct_rate":totals["future_correct"]/n,
        "mean_paid_checks":totals["paid_checks"]/n,
        "by_family":{fam:dict(cnt) for fam,cnt in families.items()},
        "records":records,
        "inference_public_template_checks_max_per_case":BUDGET,
        "inference_exhaustive_preview_used":False,
    }


def select_development(dev,fitted):
    options=[]
    for kind in (None,)+KINDS:
        for scale in ((0.0,) if kind is None else MIXING_SCALES[1:]):
            score=evaluate(dev,None,None,0) if kind is None else evaluate(
                dev,fitted[kind],kind,scale)
            utility=score["future_correct"]-.5*score["false_accept"]-.015*score["paid_checks_total"]
            options.append({
                "kind":kind or "original","scale":scale,
                "future_correct":score["future_correct"],
                "false_accept":score["false_accept"],
                "checks":score["paid_checks_total"],
                "utility":utility
            })
    # Break exact ties toward no correction, then smaller amplitude.
    best=max(options,key=lambda o:(
        o["utility"],int(o["kind"]=="original"),-o["scale"]))
    return best,options


def paired_comparison(original,candidate):
    a=original["records"];b=candidate["records"]
    assert len(a)==len(b)
    if any(x["public_features_sha256"]!=y["public_features_sha256"]
           for x,y in zip(a,b,strict=True)):
        raise AssertionError("unpaired experiment cases")
    gain=sum(x["future_correct"]==0 and y["future_correct"]==1 for x,y in zip(a,b,strict=True))
    loss=sum(x["future_correct"]==1 and y["future_correct"]==0 for x,y in zip(a,b,strict=True))
    changed_top=sum(x["top1"]!=y["top1"] for x,y in zip(a,b,strict=True))
    return {"newly_correct":gain,"lost_correct":loss,"changed_top1":changed_top}


def run(counts=SPLIT_COUNTS):
    checkpoint,probe,probe_sha,rows=locked_rows(counts)
    train=rows["train"]
    # Fit the negative probe on permuted basin labels using the NOR-11
    # predeclared training source, under the same base checkpoint.
    p1,p2=previous.train_source()[1:]
    diagnostic_train=previous.build_rows({"train":previous.sample_cohorts(counts)["train"]},
                                         checkpoint,p1,p2)["train"]
    perm=[r["label"] for r in diagnostic_train]
    random.Random(previous.SEED+77).shuffle(perm)
    shuffled_probe=previous.fit_head(diagnostic_train,
        seed=previous.SEED+77,labels_override=perm)
    attach_features(rows,probe,shuffled_probe)
    for row in train:
        row["visible_mask"]=visible_mask(row)
    fitted={name:fit_residual(train,name) for name in KINDS}
    selected,dev_grid=select_development(rows["dev"],fitted)
    test=rows["test"]
    original=evaluate(test)
    arms={"original":original}
    for kind in KINDS:
        # Each type's own scale is selected on development only.
        local=max((z for z in dev_grid if z["kind"]==kind),
                  key=lambda x:(x["utility"],-x["scale"]))
        arms[kind]=evaluate(test,fitted[kind],kind,local["scale"])
        arms[kind]["frozen_scale"]=local["scale"]
    if selected["kind"]=="original":
        arms["development_selected"]=evaluate(test)
    else:
        arms["development_selected"]=evaluate(
            test,fitted[selected["kind"]],selected["kind"],selected["scale"])
    for key in list(arms):
        arms[key]["relative_to_original"]=paired_comparison(original,arms[key])
    # Original public header and checkpoint ranking parity:
    # the first score arm uses the exact old readout and 4-check verifier.
    assert original["cases"]==len(test)
    assert all(arm["paid_checks_total"]<=BUDGET*len(test) for arm in arms.values())
    manifest={
        split: {
          "cases":len(group),
          "unique_public_feature_tensors":len({r["digest"] for r in group}),
          "feature_manifest_sha256":sha256(
              "\n".join(r["digest"] for r in group).encode()).hexdigest(),
          "families":dict(Counter(r["case"].family for r in group))
        } for split,group in rows.items()
    }
    return {
        "schema":"nor12.frozen_rrc25.public_verified_basin_residual_readout.v1",
        "source_checkpoint_sha256":BASIN_SHA,
        "original_recurrent_parameters":checkpoint["parameters"],
        "original_readout_modified":False,
        "frozen_nor11_basin_head_json_sha256":probe_sha,
        "frozen_nor11_basin_head_parameters":13,
        "residual_parameter_slots_per_arm":224,
        "hidden_only_active_parameters":208,
        "fixed_train_counts":counts,
        "seed":SEED,
        "cohorts":manifest,
        "training_full_public_template_checks":len(train)*OUTPUTS,
        "training_label_source":"only exact degree-192 public demonstrations",
        "optimizer":"full batch multiclass public-valid-mass Adam with ridge",
        "fitted":fitted,
        "development_option_grid":dev_grid,
        "development_selected":selected,
        "heldout":arms,
        "scope_flags":{
            "checkpoint_weights_updated":False,
            "private_256_degree_future_or_program_used_to_train_readouts":False,
            "test_private_future_used_for_development_selection":False,
            "full_public_template_checks_during_training_charged_separately":True,
            "test_aliases_not_independent_observations":True,
            "source_model_proof_ranking_outcome_measured":True,
            "new_test_basin_certificates_claimed":False,
            "task_correction_based_on_public_verified_candidate_labels":True,
        }
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train",type=int,default=SPLIT_COUNTS["train"])
    parser.add_argument("--dev",type=int,default=SPLIT_COUNTS["dev"])
    parser.add_argument("--test",type=int,default=SPLIT_COUNTS["test"])
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    counts={"train":args.train,"dev":args.dev,"test":args.test}
    out=run(counts)
    if args.check:
        assert out["source_checkpoint_sha256"]==BASIN_SHA
        assert out["cohorts"]["train"]["cases"]==3*args.train
        assert out["cohorts"]["dev"]["cases"]==3*args.dev
        assert out["cohorts"]["test"]["cases"]==6*args.test
        assert not out["scope_flags"]["checkpoint_weights_updated"]
        assert out["training_full_public_template_checks"]==16*3*args.train
        assert all(z["paid_checks_total"]<=4*z["cases"] for z in out["heldout"].values())
    s=json.dumps(out,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(s)
    else:
        print(s,end="")


if __name__=="__main__":
    main()
