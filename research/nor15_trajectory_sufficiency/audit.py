"""NOR-15A: frozen-GRU trajectory reconstructibility and local information loss.

No withheld future truth, source program, or private proof label is read by
the reconstruction fit. H4 and H9 derive solely from nine model-visible
public feature rows. New test rosters exclude prior NOR-11/12/13/14 inputs.

The local Jacobian fixes x5..x9 while varying H4. Its singular spectrum
is not a global conditional-entropy or invertibility certificate when the
last five input rows themselves vary.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.rrc_compactification import compound25 as corpus, model25
from research.nor12_task_readout import readout as old
from research.nor12_task_readout import independent_confirmation as nor12b
from research.nor13_gated_readout import interaction as nor13
from research.nor14_conditional_basin import experiment as nor14

SCHEMA="nor15a.frozen_gru.trajectory_endpoint_reconstructibility.v1"
SEED=2026111515
TEST_OFFSET=92715000
REPETITIONS=4
PER_FAMILY=12
DIM=12
MID_STEP=4
END_STEP=9
RIDGES=(.001,.01,.1,1.,10.)
WIDTH_RFF=24
JACOBIAN_CASES_PER_FAMILY=4
JAC_EPS=1.e-5


def native_step(hidden, x, checkpoint):
    s=checkpoint["state_dict"]
    a=model25.affine(s["rnn.weight_ih_l0"],s["rnn.bias_ih_l0"],x)
    v=model25.affine(s["rnn.weight_hh_l0"],s["rnn.bias_hh_l0"],hidden)
    out=[]
    for i in range(DIM):
        reset=model25.sigmoid(a[i]+v[i])
        update=model25.sigmoid(a[i+DIM]+v[i+DIM])
        candidate=math.tanh(a[i+2*DIM]+reset*v[i+2*DIM])
        out.append((1.-update)*candidate+update*hidden[i])
    return out


def full_trace(case,header,ck):
    features=corpus.features(case,header)
    assert len(features)==9 and all(len(r)==8 for r in features)
    hidden=[0.]*DIM
    traces=[]
    for x in features:
        hidden=native_step(hidden,x,ck)
        traces.append(hidden[:])
    expected=model25.logits(case,header,ck)
    got=model25.affine(ck["state_dict"]["head.weight"],ck["state_dict"]["head.bias"],hidden)
    assert max(abs(a-b) for a,b in zip(expected,got,strict=True))<1e-11
    return traces,features


def path_row(row,ck):
    traces,inputs=full_trace(row["case"],row["header"],ck)
    assert max(abs(a-b) for a,b in zip(traces[-1],row["h"],strict=True))<1e-11
    return {"digest":row["digest"],"family":row["case"].family,
            "h4":traces[3], "h9":traces[8], "tail":inputs[4:],
            "source_visible_sha256":nor13.source_fingerprint(row["case"])}


def tail_map(state,tail,ck):
    h=list(state)
    for x in tail:
        h=native_step(h,x,ck)
    return h


def transition_jacobian(rec,ck,step=JAC_EPS):
    """Finite-difference local differential of five remaining recurrent steps."""
    if not 0<step<.01 or len(rec["tail"])!=5:raise ValueError("invalid Jacobian window")
    columns=[]
    for k in range(DIM):
        plus=rec["h4"][:]; minus=rec["h4"][:]
        plus[k]+=step; minus[k]-=step
        a=tail_map(plus,rec["tail"],ck)
        b=tail_map(minus,rec["tail"],ck)
        columns.append([(x-y)/(2*step) for x,y in zip(a,b,strict=True)])
    return [[columns[k][i] for k in range(DIM)] for i in range(DIM)]


def gram_spectrum(jac):
    """Jacobi diagonalisation of the symmetric 12x12 Gram matrix J^T J."""
    assert len(jac)==DIM and all(len(r)==DIM for r in jac)
    A=[[math.fsum(jac[k][i]*jac[k][j] for k in range(DIM))
        for j in range(DIM)] for i in range(DIM)]
    for _ in range(70*DIM*DIM):
        p,q=max(((i,j) for i in range(DIM) for j in range(i+1,DIM)),
                key=lambda ij:abs(A[ij[0]][ij[1]]))
        if abs(A[p][q])<1e-13:break
        ttheta=(A[q][q]-A[p][p])/(2*A[p][q])
        t=(1 if ttheta>=0 else -1)/(abs(ttheta)+math.sqrt(1+ttheta*ttheta))
        c=1/math.sqrt(1+t*t)
        s=t*c
        pp=A[p][p]; qq=A[q][q]; pq=A[p][q]
        A[p][p]=pp-t*pq; A[q][q]=qq+t*pq
        A[p][q]=A[q][p]=0.
        for k in range(DIM):
            if k==p or k==q:continue
            pk=A[p][k]; qk=A[q][k]
            A[p][k]=A[k][p]=c*pk-s*qk
            A[q][k]=A[k][q]=s*pk+c*qk
    singular=sorted(math.sqrt(max(0.,A[i][i])) for i in range(DIM))
    return singular


def standardizer(train,key):
    mu=[math.fsum(r[key][k] for r in train)/len(train) for k in range(DIM)]
    sd=[max(.001,math.sqrt(math.fsum((r[key][k]-mu[k])**2 for r in train)/len(train)))
        for k in range(DIM)]
    return {"mean":mu,"scale":sd}


def standardized(v,stats):
    return [(x-m)/s for x,m,s in zip(v,stats["mean"],stats["scale"],strict=True)]


def fixed_rff():
    rng=random.Random(SEED)
    return {"weights":[[rng.gauss(0,1/math.sqrt(DIM)) for _ in range(DIM)]
                       for _ in range(WIDTH_RFF)],
            "biases":[rng.uniform(-1,1) for _ in range(WIDTH_RFF)]}


def design(row,xstat,kind,rff):
    z=standardized(row["h9"],xstat)
    if kind not in ("linear","nonlinear","shuffled_linear"):
        raise ValueError("bad reconstruction arm")
    if kind=="nonlinear":
        extra=[math.tanh(.8*math.fsum(a*b for a,b in zip(w,z,strict=True))+bias)
               for w,bias in zip(rff["weights"],rff["biases"],strict=True)]
        return z+extra+[1.]
    return z+[1.]


def solve_linear(A,b):
    """Positive-ridge normal equations with pivoted elimination."""
    n=len(b)
    M=[list(row)+[v] for row,v in zip(A,b,strict=True)]
    for j in range(n):
        k=max(range(j,n),key=lambda i:abs(M[i][j]))
        if abs(M[k][j])<1e-14:raise ArithmeticError("ridge design is singular")
        M[j],M[k]=M[k],M[j]
        q=M[j][j]
        M[j]=[v/q for v in M[j]]
        for i in range(n):
            if i==j:continue
            a=M[i][j]
            if a:
                M[i]=[u-a*v for u,v in zip(M[i],M[j],strict=True)]
    return [row[-1] for row in M]


def ridge(train,xstat,ystat,kind,rff,regularizer):
    X=[design(r,xstat,kind,rff) for r in train]
    Y=[standardized(r["h4"],ystat) for r in train]
    if kind=="shuffled_linear":
        shuffled=list(Y)
        random.Random(SEED+9).shuffle(shuffled)
        Y=shuffled
    d=len(X[0]); n=len(train)
    gram=[[math.fsum(x[i]*x[j] for x in X)/n+
           (regularizer if i==j and i!=d-1 else 0.)
           for j in range(d)] for i in range(d)]
    weights=[]
    for k in range(DIM):
        rhs=[math.fsum(x[i]*y[k] for x,y in zip(X,Y,strict=True))/n
             for i in range(d)]
        weights.append(solve_linear(gram,rhs))
    return {"weights":weights,"kind":kind,"lambda":regularizer,
            "dimensions":d,"train_cases":n}


def predict(model,row,xstat,rff):
    x=design(row,xstat,model["kind"],rff)
    return [math.fsum(a*b for a,b in zip(w,x,strict=True))
            for w in model["weights"]]


def mse(model,row,xstat,ystat,rff):
    target=standardized(row["h4"],ystat)
    if model=="mean":
        pred=[0.]*DIM
    elif model=="copy_endpoint":
        # Endpoint-only physical identity baseline, transformed to H4 units.
        pred=standardized(row["h9"],ystat)
    else:
        pred=predict(model,row,xstat,rff)
    return math.fsum((u-v)**2 for u,v in zip(target,pred,strict=True))/DIM


def freeze_dev(train,dev):
    assert len(train)==96 and len(dev)==36
    xstat=standardizer(train,"h9"); ystat=standardizer(train,"h4")
    rff=fixed_rff()
    models={}; selected={}; grid=[]
    for kind in ("linear","nonlinear","shuffled_linear"):
        options=[]
        for lam in RIDGES:
            candidate=ridge(train,xstat,ystat,kind,rff,lam)
            err=math.fsum(mse(candidate,r,xstat,ystat,rff) for r in dev)/len(dev)
            options.append((err,-lam,candidate))
            grid.append({"kind":kind,"lambda":lam,"dev_standardized_MSE":err})
        best=min(options,key=lambda t:(t[0],t[1]))
        models[kind]=best[2]
        selected[kind]={"lambda":best[2]["lambda"],"dev_standardized_MSE":best[0]}
    for kind in ("mean","copy_endpoint"):
        err=math.fsum(mse(kind,r,xstat,ystat,rff) for r in dev)/len(dev)
        selected[kind]={"lambda":None,"dev_standardized_MSE":err}
    return xstat,ystat,rff,models,selected,grid


def new_heldout(roster,repetitions=REPETITIONS,per_family=PER_FAMILY):
    if not 1<=repetitions<=4 or not 1<=per_family<=12:
        raise ValueError("test budget must be frozen")
    prior12=nor12b.fresh_cohorts(roster,replicates=nor12b.REPETITIONS,
                                per_family=nor12b.CASES_PER_FAMILY)
    prior13=nor13.new_heldout(roster,replicates=nor13.CONFIRMATIONS,
                             per_family=nor13.FAMILY_PER)
    prior14=nor14.fresh_cohorts(roster,repetitions=nor14.REPETITIONS,
                               per_family=nor14.PER_FAMILY)
    old_vis={nor13.source_fingerprint(case) for split in roster.values()
             for case,header,digest in split}
    old_features={digest for split in roster.values()
                  for case,header,digest in split}
    for batch in prior12+prior13+prior14:
        for case,header,digest in batch:
            old_vis.add(nor13.source_fingerprint(case))
            old_features.add(digest)
    out=[]
    for rep in range(repetitions):
        batch=[]
        for fi,family in enumerate(corpus.TEST_FAMILIES):
            rng=random.Random(corpus.SEED+TEST_OFFSET+rep*1000003+fi*10007)
            found=0
            for trial in range(per_family*2000):
                case=corpus.make_case(rng,family,f"nor15a-r{rep}-{family}-{trial:05d}")
                headers=corpus.possible_headers(case)
                if len(headers)!=1:continue
                digest=nor14.prior_probe.public_digest(case,headers[0])
                vis=nor13.source_fingerprint(case)
                if vis in old_vis or (family!="late_source" and digest in old_features):
                    continue
                old_vis.add(vis)
                if family!="late_source":old_features.add(digest)
                batch.append((case,headers[0],digest));found+=1
                if found==per_family:break
            if found!=per_family:
                raise RuntimeError(f"new source exhaustion {rep} {family}")
        out.append(batch)
    return out


def nearest_endpoint_pairs(rows):
    """Nearest-neighbor witnesses, descriptive only; equal H9 is never assumed."""
    nonlate=[r for r in rows if r["family"]!="late_source"]
    assert len(nonlate)==len({r["digest"] for r in nonlate})
    output=[]
    for a in nonlate:
        others=(b for b in nonlate if b["digest"]!=a["digest"])
        b=min(others,key=lambda c:math.dist(a["h9"],c["h9"]))
        d9=math.dist(a["h9"],b["h9"])
        d4=math.dist(a["h4"],b["h4"])
        output.append({"a":a["digest"],"b":b["digest"],
                       "endpoint_distance":d9,"early_distance":d4,
                       "early_to_endpoint_distance_ratio":d4/max(1e-12,d9),
                       "same_source_family":a["family"]==b["family"]})
    return output


def run(repetitions=REPETITIONS,per_family=PER_FAMILY):
    ck,probe,negative,old_weights,original,source,roster=nor13.frozen_data()
    assert ck["parameters"]==1000
    assert nor13.READOUT_SHA==old.BASIN_SHA
    train=[path_row(row,ck) for row in original["train"]]
    dev=[path_row(row,ck) for row in original["dev"]]
    xstat,ystat,rff,models,selected,grid=freeze_dev(train,dev)
    cohorts=new_heldout(roster,repetitions,per_family)
    test=[]
    for batch in cohorts:
        fresh=nor12b.new_rows(batch,ck,probe,negative)
        test.extend(path_row(row,ck) for row in fresh)
    arms=("mean","copy_endpoint","linear","nonlinear","shuffled_linear")
    records=[]
    for row in test:
        errors={k:mse(k if k in ("mean","copy_endpoint") else models[k],
                      row,xstat,ystat,rff) for k in arms}
        records.append({"digest":row["digest"],"family":row["family"],
                        "visible_sha256":row["source_visible_sha256"],
                        "h4":row["h4"],"h9":row["h9"],
                        "standardized_reconstruction_MSE":errors})
    outcome={}
    for kind in arms:
        subset=[z for z in records if z["family"]!="late_source"]
        outcome[kind]={
            "primary_nonlate_cases":len(subset),
            "primary_mean_MSE":statistics.fmean(z["standardized_reconstruction_MSE"][kind]
                                                for z in subset),
            "primary_median_MSE":statistics.median(z["standardized_reconstruction_MSE"][kind]
                                                  for z in subset),
            "all_case_mean_MSE":statistics.fmean(z["standardized_reconstruction_MSE"][kind]
                                                for z in records),
            "by_family_mean_MSE":{fam:statistics.fmean(
                z["standardized_reconstruction_MSE"][kind] for z in records if z["family"]==fam)
                                  for fam in corpus.TEST_FAMILIES}}
    def paired_error(first,second):
        samples=[z for z in records if z["family"]!="late_source"]
        gain=sum(z["standardized_reconstruction_MSE"][first] <
                 z["standardized_reconstruction_MSE"][second] for z in samples)
        loss=sum(z["standardized_reconstruction_MSE"][first] >
                 z["standardized_reconstruction_MSE"][second] for z in samples)
        return {"first_lower_error":gain,"second_lower_error":loss,
                "ties":len(samples)-gain-loss,
                "unadjusted_two_sided_sign_p":float(nor12b.exact_sign_p(gain,loss))}
    witnesses=nearest_endpoint_pairs(test)
    # Compute local differential spectra on a preregistered bounded subset:
    # first four case indices of each family in first fresh cohort.
    spectrum=[]
    for i,rec in enumerate(test[:6*per_family]):
        if i%per_family>=JACOBIAN_CASES_PER_FAMILY:continue
        jac=transition_jacobian(rec,ck)
        sig=gram_spectrum(jac)
        true_forward=tail_map(rec["h4"],rec["tail"],ck)
        assert max(abs(a-b) for a,b in zip(true_forward,rec["h9"],strict=True))<1e-11
        spectrum.append({"digest":rec["digest"],"family":rec["family"],
                         "singular_values":sig,
                         "sigma_min":sig[0],"sigma_max":sig[-1],
                         "condition_estimate":sig[-1]/max(1e-15,sig[0])})
    return {
        "schema":SCHEMA,"checkpoint_sha256":old.BASIN_SHA,
        "frozen_probe_sha256":source["frozen_probe_json_sha256"],
        "train_manifest":source["manifests"]["train"],
        "dev_manifest":source["manifests"]["dev"],
        "train_cases":len(train),"dev_cases":len(dev),
        "source_families":list(corpus.TEST_FAMILIES),
        "new_seed_offset":TEST_OFFSET,
        "fresh_repetitions":repetitions,
        "cases_per_family_per_repetition":per_family,
        "test_manifest_sha256":sha256("\n".join(z["digest"] for z in test).encode()).hexdigest(),
        "test_cases":len(test),
        "nonlate_unique_inputs":len({z["digest"] for z in test if z["family"]!="late_source"}),
        "late_source_unique_inputs":len({z["digest"] for z in test if z["family"]=="late_source"}),
        "train_only_stats":{"endpoint":xstat,"step4":ystat,
                            "rff":rff,"model_weights":{k:v["weights"] for k,v in models.items()}},
        "development_frozen_selection":selected,
        "development_grid":grid,
        "outcomes":outcome,
        "paired_primary_signs":{
            "nonlinear_vs_linear":paired_error("nonlinear","linear"),
            "linear_vs_shuffled":paired_error("linear","shuffled_linear"),
            "nonlinear_vs_mean":paired_error("nonlinear","mean")},
        "nearest_endpoint_pairs":witnesses,
        "local_jacobian_spectra":spectrum,
        "local_jacobian_bounded_cases":len(spectrum),
        "records":records,
        "scope":{
            "public_feature_only_recurrence":True,
            "training_uses_hidden_labels_only_not_proof_futures":True,
            "frozen_gru":True,
            "original_readout_not_modified":True,
            "test_input_disjoint_all_NOR11_12_13_14_nonlate":True,
            "source_aliased_late_cases_secondary_only":True,
            "finite_sample_reconstructibility_not_conditional_mutual_information":True,
            "local_jacobian_holds_last_five_inputs_fixed":True,
            "jacobian_is_finite_difference_F64_not_interval_certified":True,
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repetitions",type=int,default=REPETITIONS)
    p.add_argument("--per-family",type=int,default=PER_FAMILY)
    p.add_argument("--check",action="store_true")
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    r=run(args.repetitions,args.per_family)
    if args.check:
        assert r["checkpoint_sha256"]==old.BASIN_SHA
        assert r["train_cases"]==96 and r["dev_cases"]==36
        assert r["test_cases"]==6*args.per_family*args.repetitions
        assert r["nonlate_unique_inputs"]==5*args.per_family*args.repetitions
        assert r["scope"]["public_feature_only_recurrence"]
        assert r["scope"]["test_input_disjoint_all_NOR11_12_13_14_nonlate"]
        assert all(math.isfinite(v["primary_mean_MSE"]) for v in r["outcomes"].values())
    out=json.dumps(r,indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(out)
    else:
        print(out,end="")


if __name__=="__main__":
    main()
