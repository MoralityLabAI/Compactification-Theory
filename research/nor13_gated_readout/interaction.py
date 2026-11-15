"""NOR-13: interaction-gated recurrent task readouts with fresh locked controls.

The frozen NOR-11 basin head supplies g(h)=2*p(second|h)-1. The new
residual uses BOTH the standardised hidden state z and products g*z:
    scores_j = original_j + scale*(U_j.z + V_j.(g*z) + bias_j).

Exact observed-prefix verification (16 public candidate checks / training
case) supplies valid-template sets. No future correctness or program
identity is used in fitting. The 1000-param GRU, original task head and
13-param basin head never change.

Controls have identical 16x25 parameter-slot allocations: pretrained
shuffled-label gate, fixed random public hash gate, and a constant-zero
gate. Two previously trained NOR-12 additive readouts are retained as
reference arms. DEV alone selects scale, TEST consists of four new
seed-separated cohorts, excluding all nonlate input tensors from NOR-12
train, dev, initial test and four previous confirmation cohorts.

Each test arm uses 4 exact public candidate checks, independently
graded against withheld degree-256 future only after candidate choice.
This is a synthetic trained-model proof-template experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction as Q
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for x in (ROOT,HERE):
    if str(x) not in sys.path:
        sys.path.insert(0,str(x))

from research.rrc_compactification import compound25 as corpus
from research.nor11_attractor_controller import controller as baseline
from research.nor12_task_readout import readout as old
from research.nor12_task_readout import independent_confirmation as prior

KINDS=("basin_gate","shuffled_gate","random_gate","constant_gate")
WIDTH=25
CLASSES=16
EPOCHS=180
LR=.023
RIDGE=.075
SCALES=(0.25,0.5,1.0,2.0)
TEST_SEED_OFFSET=92315000
CONFIRMATIONS=4
FAMILY_PER=12
READOUT_SHA=old.BASIN_SHA
SCHEME="nor13.frozen_gru.interaction_gated_public_task_readout.v1"


def frozen_data():
    # The NOR-12 reconstruction checks all four original trained-head
    # SHA-256 hashes and the original NOR-11 train/dev/test manifests.
    ck,probe,negative,previous_weights,original,design,roster=prior.rebuild_frozen_models()
    assert design["source_checkpoint_sha256"]==READOUT_SHA
    assert old.BASIN_SHA==READOUT_SHA
    return ck,probe,negative,previous_weights,original,design,roster


def source_fingerprint(case):
    return sha256(json.dumps(case.visible,separators=(",",":")).encode()).hexdigest()


def new_heldout(roster,replicates=CONFIRMATIONS,per_family=FAMILY_PER):
    if not 1<=replicates<=6 or not 1<=per_family<=16:
        raise ValueError("bounded independent test budget exceeded")
    # Regenerate the already evaluated NOR-12B cohorts and exclude
    # every nonlate tensor and all original visible synthetic cases.
    previously_scored=prior.fresh_cohorts(
        roster,replicates=prior.REPETITIONS,per_family=prior.CASES_PER_FAMILY)
    seen_vis={source_fingerprint(case)
              for split in roster.values() for case,header,digest in split}
    known_features={digest for split in roster.values()
                    for case,header,digest in split}
    for batch in previously_scored:
        for case,header,digest in batch:
            seen_vis.add(source_fingerprint(case))
            known_features.add(digest)
    # Only late_source is allowed to alias model-visible feature tensors:
    # its construction intentionally places information after the
    # observed 9-row feature window. Evaluation remains case-clustered.
    output=[]
    for rep in range(replicates):
        cohort=[]
        for fi,family in enumerate(corpus.TEST_FAMILIES):
            rng=random.Random(corpus.SEED+TEST_SEED_OFFSET+rep*1000003+fi*10007)
            found=0
            for trial in range(per_family*1600):
                case=corpus.make_case(
                    rng,family,f"nor13-r{rep}-{family}-{trial:05d}")
                headers=corpus.possible_headers(case)
                if len(headers)!=1:
                    continue
                digest=baseline.public_digest(case,headers[0])
                visible=source_fingerprint(case)
                if visible in seen_vis:
                    continue
                if family!="late_source" and digest in known_features:
                    continue
                seen_vis.add(visible)
                if family!="late_source":
                    known_features.add(digest)
                cohort.append((case,headers[0],digest))
                found+=1
                if found==per_family:
                    break
            if found!=per_family:
                raise RuntimeError("public heldout source exhausted: "+family)
        output.append(cohort)
    return output


def scalar(row,kind):
    if kind=="basin_gate":
        return row["x"]["basin"][12]
    if kind=="shuffled_gate":
        return row["x"]["shuffled_basin"][12]
    if kind=="random_gate":
        return row["x"]["random_feature"][12]
    if kind=="constant_gate":
        return 0.0
    raise ValueError(kind)


def gated_vector(row,kind,override=None):
    z=row["x"]["hidden_only"][:12]
    assert len(z)==12
    gate=scalar(row,kind) if override is None else override
    assert -1.000000001<=gate<=1.000000001
    return z+[gate*v for v in z]+[1.0]


def raw_scores(row,W,kind,scale,override=None):
    if not 0<=scale<=2:
        raise ValueError("invalid frozen residual amplitude")
    if len(W)!=CLASSES or any(len(line)!=WIDTH for line in W):
        raise ValueError("mismatched task correction shape")
    x=gated_vector(row,kind,override)
    return [b+scale*math.fsum(a*v for a,v in zip(w,x,strict=True))
            for b,w in zip(row["base"],W,strict=True)]


def gradient(row,W,kind):
    scores=raw_scores(row,W,kind,1.)
    p=old.softmax(scores)
    mask=row["visible_mask"]
    mass=math.fsum(v for v,accepted in zip(p,mask,strict=True) if accepted)
    if mass<=0:
        raise ArithmeticError("zero accepted mass in public supervised fit")
    x=gated_vector(row,kind)
    g=[]
    for j in range(CLASSES):
        delta=p[j]-(p[j]/mass if mask[j] else 0.0)
        g.append([delta*v for v in x])
    return -math.log(mass),g


def fit(train,kind):
    if kind not in KINDS or len(train)<8:
        raise ValueError("missing positive training set / invalid gate")
    if any("visible_mask" not in r or not any(r["visible_mask"]) for r in train):
        raise ValueError("missing exact public verified candidate labels")
    W=[[0.0]*WIDTH for _ in range(CLASSES)]
    m=[[0.0]*WIDTH for _ in range(CLASSES)]
    v=[[0.0]*WIDTH for _ in range(CLASSES)]
    trace=[]
    for epoch in range(1,EPOCHS+1):
        g=[[0.0]*WIDTH for _ in range(CLASSES)]
        loss=0.
        for row in train:
            ce,grad=gradient(row,W,kind)
            loss+=ce
            for j in range(CLASSES):
                for i in range(WIDTH):
                    g[j][i]+=grad[j][i]
        for j in range(CLASSES):
            for i in range(WIDTH):
                dj=g[j][i]/len(train)+RIDGE*W[j][i]
                m[j][i]=.9*m[j][i]+.1*dj
                v[j][i]=.999*v[j][i]+.001*dj*dj
                W[j][i]-=LR*(m[j][i]/(1-.9**epoch))/(
                    math.sqrt(v[j][i]/(1-.999**epoch))+1e-8)
        if epoch in (1,45,90,135,180):
            trace.append({"epoch":epoch,"public_accepted_mass_nll":loss/len(train)})
    return {
        "kind":kind,"weights":W,"slots":CLASSES*WIDTH,
        "active_slots":CLASSES*(13 if kind=="constant_gate" else WIDTH),
        "train_public_template_oracle_calls":CLASSES*len(train),
        "hidden_program_used_for_train":False,
        "future_256_degree_used_for_train":False,
        "optimizer":"deterministic full-batch Adam",
        "epochs":EPOCHS,"learning_rate":LR,"ridge":RIDGE,
        "training_trace":trace,
    }


def task_eval(rows,W=None,kind=None,scale=0.,override=None):
    if W is None:
        if kind is not None or scale:
            raise ValueError("invalid baseline mode")
        return old.evaluate(rows)
    altered=[]
    for row in rows:
        q=override(row) if callable(override) else override
        altered.append({**row,"base":raw_scores(row,W,kind,scale,q)})
    return old.evaluate(altered)


def dev_selection(dev,fitted):
    grid=[]
    base=task_eval(dev)
    utility=lambda x:x["future_correct"]-.5*x["false_accept"]-.015*x["paid_checks_total"]
    grid.append({"kind":"original","scale":0.0,"utility":utility(base),
                 "future_correct":base["future_correct"],
                 "false_accept":base["false_accept"],
                 "paid_checks":base["paid_checks_total"]})
    for kind in KINDS:
        for scale in SCALES:
            x=task_eval(dev,fitted[kind]["weights"],kind,scale)
            grid.append({"kind":kind,"scale":scale,"utility":utility(x),
                         "future_correct":x["future_correct"],
                         "false_accept":x["false_accept"],
                         "paid_checks":x["paid_checks_total"]})
    select=lambda ar:max((r for r in grid if r["kind"]==ar),
                     key=lambda x:(x["utility"],-x["scale"]))
    chosen={k:select(k) for k in KINDS}
    global_best=max(grid,key=lambda r:(r["utility"],r["kind"]=="original",-r["scale"]))
    return chosen,global_best,grid


def feature_paired_stats(a,b):
    paired=old.paired_comparison(a,b)
    groups=defaultdict(lambda:[0,0])
    for left,right in zip(a["records"],b["records"],strict=True):
        assert left["public_features_sha256"]==right["public_features_sha256"]
        digest=left["public_features_sha256"]
        groups[digest][0]+=right["future_correct"]-left["future_correct"]
        groups[digest][1]+=1
    positive=sum(v[0]>0 for v in groups.values())
    negative=sum(v[0]<0 for v in groups.values())
    return {**paired,
        "unique_model_inputs":len(groups),
        "positive_input_groups":positive,
        "negative_input_groups":negative,
        "tied_input_groups":sum(v[0]==0 for v in groups.values()),
        "feature_weighted_accuracy_change":sum(v[0]/v[1] for v in groups.values())/len(groups),
        "unadjusted_group_sign_p":float(prior.exact_sign_p(positive,negative)),
        "unadjusted_case_mcnemar_p":float(prior.exact_sign_p(
            paired["newly_correct"],paired["lost_correct"])),
    }


def gate_ablation(rows,fitted,scale):
    W=fitted["basin_gate"]["weights"]
    unique={}
    for row in rows:
        digest=row["digest"]
        g=scalar(row,"basin_gate")
        if digest in unique:
            assert unique[digest]==g
        unique[digest]=g
    keys=sorted(unique)
    vals=[unique[k] for k in keys]
    random.Random(202611151313).shuffle(vals)
    permuted=dict(zip(keys,vals,strict=True))
    ablations={
        "trained_gate":None,
        "gate_zero":lambda row:0.0,
        "gate_replaced_by_shuffled_probe":lambda row:scalar(row,"shuffled_gate"),
        "gate_replaced_by_random_hash":lambda row:scalar(row,"random_gate"),
        "gate_permuted_by_public_input":lambda row:permuted[row["digest"]],
        "gate_flipped_sign":lambda row:-scalar(row,"basin_gate"),
    }
    base=task_eval(rows)
    outputs={}
    for name,fn in ablations.items():
        ev=task_eval(rows,W,"basin_gate",scale,override=fn)
        outputs[name]={
            "future_correct":ev["future_correct"],
            "false_accept":ev["false_accept"],
            "paid_checks":ev["paid_checks_total"],
            "paired_to_original":feature_paired_stats(base,ev),
            "records":ev["records"],
        }
    return outputs


def checkpoint_intervals(rows,W,scale):
    """Count candidate pairs whose rank can vary with gate in [-1,1].

    The score for each template is affine in gate. Exact Fraction
    endpoint comparisons of the binary64 constants show whether
    any pair of the 16 candidates could change relative order.
    This is a rational certificate for the affine score-lift;
    machine float rounding is checked separately by a min-margin
    diagnostic and is not covered as a theorem here.
    """
    records=[]
    for row in rows:
        z=row["x"]["hidden_only"][:12]
        # Treat the stored Python float inputs as *exact binary rationals*.
        a=[];b=[]
        for score,w in zip(row["base"],W,strict=True):
            aa=Q(score)+Q(scale)*(sum(
                (Q(w[k])*Q(z[k]) for k in range(12)),Q(0))+Q(w[24]))
            bb=Q(scale)*sum((Q(w[12+k])*Q(z[k]) for k in range(12)),Q(0))
            a.append(aa);b.append(bb)
        guaranteed=0
        crossing_pairs=0
        minimum_endpoint_abs=None
        for i in range(CLASSES):
            for j in range(i+1,CLASSES):
                lo=(a[i]-b[i])-(a[j]-b[j])
                hi=(a[i]+b[i])-(a[j]+b[j])
                if lo==0 or hi==0 or (lo>0)!=(hi>0):
                    crossing_pairs+=1
                else:
                    guaranteed+=1
                    mm=min(abs(lo),abs(hi))
                    minimum_endpoint_abs=mm if minimum_endpoint_abs is None else min(mm,minimum_endpoint_abs)
        assert crossing_pairs+guaranteed==120
        lo_scores=[aa-bb for aa,bb in zip(a,b,strict=True)]
        hi_scores=[aa+bb for aa,bb in zip(a,b,strict=True)]
        lo_order=sorted(range(CLASSES),key=lambda j:(-lo_scores[j],j))
        hi_order=sorted(range(CLASSES),key=lambda j:(-hi_scores[j],j))
        agreed_top4=lo_order[:4]==hi_order[:4]
        # Every one of these 4*12+6 relevant comparisons must retain
        # an exact strictly positive endpoint gap; lower-ranked pairs
        # are free to cross with no effect on the four-paid-check trace.
        margins=[]
        if agreed_top4:
            for position in range(4):
                chosen=lo_order[position]
                for competitor in lo_order[position+1:]:
                    margins.append(lo_scores[chosen]-lo_scores[competitor])
                    margins.append(hi_scores[chosen]-hi_scores[competitor])
        stable_top4=agreed_top4 and len(margins)==108 and all(v>0 for v in margins)
        top4_min_margin=min(margins) if margins else Q(0)
        records.append({
            "digest":row["digest"],
            "family":row["case"].family,
            "pair_crossings_or_endpoint_ties":crossing_pairs,
            "ranking_is_exactly_gate_invariant":crossing_pairs==0,
            "ordered_top4_agree_at_both_endpoints":agreed_top4,
            "ordered_top4_rationally_gate_invariant":stable_top4,
            "ordered_top4_minimum_strict_endpoint_margin_float":float(top4_min_margin),
            "ordered_top4_at_gate_minus_one":lo_order[:4],
            "ordered_top4_at_gate_plus_one":hi_order[:4],
            "minimum_invariant_pair_endpoint_margin_float":float(minimum_endpoint_abs or Q(0)),
        })
    count_unique={}
    for x in records:
        count_unique[x["digest"]]=x
    return {
        "cases":len(rows),
        "unique_public_inputs":len(count_unique),
        "gate_invariant_cases":sum(r["ranking_is_exactly_gate_invariant"] for r in records),
        "gate_variant_cases":sum(not r["ranking_is_exactly_gate_invariant"] for r in records),
        "gate_invariant_unique_inputs":sum(r["ranking_is_exactly_gate_invariant"] for r in count_unique.values()),
        "gate_variant_unique_inputs":sum(not r["ranking_is_exactly_gate_invariant"] for r in count_unique.values()),
        "top4_rationally_gate_invariant_cases":sum(r["ordered_top4_rationally_gate_invariant"] for r in records),
        "top4_rationally_gate_variant_cases":sum(not r["ordered_top4_rationally_gate_invariant"] for r in records),
        "top4_rationally_gate_invariant_unique_inputs":sum(r["ordered_top4_rationally_gate_invariant"] for r in count_unique.values()),
        "top4_rationally_gate_variant_unique_inputs":sum(not r["ordered_top4_rationally_gate_invariant"] for r in count_unique.values()),
        "certification_scope":"exact affine extension of stored float scores, readout weights and standardised hidden vectors, gate restricted to [-1,1]",
        "float_rounding_proven":False,
        "records":records,
    }


def run(replicates=CONFIRMATIONS,per_family=FAMILY_PER):
    ck,probe,negative,old_weights,original,design,roster=frozen_data()
    train=original["train"]
    for row in train:
        assert row["visible_mask"]==old.visible_mask(row)
    fitted={kind:fit(train,kind) for kind in KINDS}
    selected,global_policy,grid=dev_selection(original["dev"],fitted)
    cohorts=new_heldout(roster,replicates,per_family)
    summaries=[]
    aggreg={kind:Counter() for kind in ("original",)+KINDS+
            ("nor12_additive_basin","nor12_additive_random","dev_global")}
    all_groups=defaultdict(list)
    first_abl=None
    margin_survey=None
    for index,items in enumerate(cohorts):
        rows=prior.new_rows(items,ck,probe,negative)
        original_eval=task_eval(rows)
        arms={"original":original_eval}
        for kind in KINDS:
            arms[kind]=task_eval(rows,fitted[kind]["weights"],
                kind,selected[kind]["scale"])
        arms["nor12_additive_basin"]=old.evaluate(
            rows,old_weights["basin"],"basin",design["fixed_development_scales"]["basin"])
        arms["nor12_additive_random"]=old.evaluate(
            rows,old_weights["random_feature"],"random_feature",design["fixed_development_scales"]["random_feature"])
        if global_policy["kind"]=="original":
            arms["dev_global"]=task_eval(rows)
        else:
            kind=global_policy["kind"]
            arms["dev_global"]=task_eval(rows,fitted[kind]["weights"],kind,global_policy["scale"])
        entry={}
        for kind,ev in arms.items():
            assert ev["cases"]==len(rows)
            assert ev["paid_checks_total"]<=4*len(rows)
            paired=feature_paired_stats(original_eval,ev)
            entry[kind]={
                "future_correct":ev["future_correct"],
                "false_accept":ev["false_accept"],
                "checks":ev["paid_checks_total"],
                "paired":paired,
                "by_family":ev["by_family"],
            }
            for key in ("cases","future_correct","false_accept","paid_checks_total"):
                aggreg[kind][key]+=ev[key]
            for source_record,outcome in zip(original_eval["records"],ev["records"],strict=True):
                assert source_record["public_features_sha256"]==outcome["public_features_sha256"]
                all_groups[(kind,source_record["public_features_sha256"])].append(
                    (outcome["future_correct"]-source_record["future_correct"],
                     source_record["family"]))
        summaries.append({"replicate":index,
            "cases":len(rows),
            "unique_model_inputs":len({r["digest"] for r in rows}),
            "public_manifest":sha256("\n".join(r["digest"] for r in rows).encode()).hexdigest(),
            "arm_results":entry})
        if index==0:
            first_abl=gate_ablation(rows,fitted,selected["basin_gate"]["scale"])
            margin_survey=checkpoint_intervals(rows,fitted["basin_gate"]["weights"],
                                               selected["basin_gate"]["scale"])
            # Keeping the detailed records for this diagnostic cohort
            # permits external, independent rank and oracle replay.
    out={}
    for kind,count in aggreg.items():
        groups={digest:vals for (k,digest),vals in all_groups.items() if k==kind}
        positive=sum(sum(d for d,f in v)>0 for v in groups.values())
        negativecount=sum(sum(d for d,f in v)<0 for v in groups.values())
        gain=sum(d>0 for vals in groups.values() for d,f in vals)
        loss=sum(d<0 for vals in groups.values() for d,f in vals)
        nonlate_gains=sum(d>0 for vals in groups.values() for d,f in vals if f!="late_source")
        nonlate_losses=sum(d<0 for vals in groups.values() for d,f in vals if f!="late_source")
        out[kind]={
            "cases":count["cases"],
            "future_correct":count["future_correct"],
            "false_accept":count["false_accept"],
            "paid_checks":count["paid_checks_total"],
            "unique_feature_groups":len(groups),
            "paired_gains":gain,"paired_losses":loss,
            "net_future_correct_vs_original":count["future_correct"]-aggreg["original"]["future_correct"],
            "positive_input_groups":positive,"negative_input_groups":negativecount,
            "nonlate_case_gains":nonlate_gains,
            "nonlate_case_losses":nonlate_losses,
            "unadjusted_group_sign_p":float(prior.exact_sign_p(positive,negativecount)),
            "unadjusted_case_mcnemar_p":float(prior.exact_sign_p(gain,loss)),
        }
        assert out[kind]["paired_gains"]-out[kind]["paired_losses"]==out[kind]["net_future_correct_vs_original"]
    return {
        "schema":SCHEME,
        "source_checkpoint_sha256":READOUT_SHA,
        "original_checkpoint_parameters":ck["parameters"],
        "frozen_auxiliary_basin_head_sha256":design["frozen_probe_json_sha256"],
        "ninety_six_training_input_manifest":design["manifests"]["train"],
        "training_full_public_template_verifications_per_fit":16*len(train),
        "matched_parameter_slots_per_head":CLASSES*WIDTH,
        "original_GRU_or_task_readout_modified":False,
        "epochs":EPOCHS,"ridge":RIDGE,"learning_rate":LR,
        "fitted":fitted,
        "dev_model_selection":global_policy,
        "dev_frozen_head_scales":selected,
        "dev_option_grid":grid,
        "fresh_seed_offset":TEST_SEED_OFFSET,
        "fresh_repetitions":replicates,
        "cases_per_family_per_repetition":per_family,
        "fresh_replicate_results":summaries,
        "aggregated":out,
        "first_fresh_test_basin_gate_ablations":first_abl,
        "first_fresh_test_exact_affine_rank_stability":margin_survey,
        "scope":{
            "new_nonlate_public_features_excluded_from_all_prior_NOR12_cases":True,
            "late_source_aliases_reported_as_feature_clusters":True,
            "future_labels_used_for_training":False,
            "development_model_selection_before_new_test":True,
            "control_architectures_matched_by_parameter_slots":True,
            "original_fixed_4_paid_check_budget":True,
            "new_source_basin_membership_interval_certified":False,
            "statistical_p_values_unadjusted_exploratory":True,
        }
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repetitions",type=int,default=CONFIRMATIONS)
    parser.add_argument("--per-family",type=int,default=FAMILY_PER)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    result=run(args.repetitions,args.per_family)
    if args.check:
        assert result["source_checkpoint_sha256"]==READOUT_SHA
        assert result["matched_parameter_slots_per_head"]==400
        assert result["training_full_public_template_verifications_per_fit"]==1536
        assert result["scope"]["original_fixed_4_paid_check_budget"]
        assert result["aggregated"]["original"]["cases"]==6*args.per_family*args.repetitions
        assert result["first_fresh_test_exact_affine_rank_stability"]["cases"]==6*args.per_family
        assert all(r["paid_checks"]<=4*r["cases"] for r in result["aggregated"].values())
    s=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(s)
    else:
        print(s,end="")


if __name__=="__main__":
    main()
