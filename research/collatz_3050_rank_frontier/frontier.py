"""Development-only cross-root rank frontier for BPD prefix displacement geometry."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean
import numpy as np
from activation_io import SOURCE, WIDTH, read_run, require


def deltas(data):
    r=data["recipient.pass1_boundary"][0,:-1,:]
    return {k:data[k+"_donor.pass1_boundary"][0,:-1,:].astype(np.float64)-r for k in ("effect","sham")}


def fit_basis(samples):
    matrix=np.stack(samples)
    _,sv,v=np.linalg.svd(matrix,full_matrices=False)
    rank=int(np.sum(sv>max(matrix.shape)*np.finfo(float).eps*sv[0])) if sv[0]>0 else 0
    return v[:rank].T,sv[:rank].tolist()


def retained(delta,basis):
    total=float(np.sum(delta*delta))
    return float(np.clip(np.sum((delta@basis)**2)/total,0,1)) if total>0 else None


def run(manifest,loaded,fit_roots,eval_roots,ranks=(1,2,4,8)):
    fit_roots=set(fit_roots);eval_roots=set(eval_roots)
    require(fit_roots and eval_roots and not fit_roots&eval_roots,"fit and evaluation roots must be disjoint and nonempty")
    planned=defaultdict(int)
    groups=defaultdict(list)
    for c in manifest["cases"]:
        planned[c["root_id"]]+=1
    for c,rec,a in loaded:
        groups[c["root_id"]].append(deltas(a))
    require(all(d in planned and len(groups[d])==planned[d] for d in fit_roots|eval_roots),"all selected roots must have complete saved cases")
    # One mean vector per fit root, without using model outcomes or oracle labels.
    means={k:[np.mean([x[k].mean(axis=0) for x in groups[d]],axis=0)
              for d in sorted(fit_roots)] for k in ("effect","sham")}
    bases={};spectra={}
    for k in means:
        bases[k],spectra[k]=fit_basis(means[k])
    feasible=min(bases["effect"].shape[1],bases["sham"].shape[1],8)
    rng=np.random.default_rng(20261126)
    random,_=np.linalg.qr(rng.normal(size=(WIDTH,max(1,feasible))))
    bases["random"]=random[:,:feasible]
    rows=[]
    for rank in sorted(set(ranks)):
        require(type(rank) is int and rank>0,"invalid rank")
        if rank>feasible:
            rows.append({"rank":rank,"status":"infeasible_matched_rank"})
            continue
        per={}
        for d in sorted(eval_roots):
            per[str(d)]={}
            for basis,B in bases.items():
                per[str(d)][basis]={}
                for label in ("effect","sham"):
                    vals=[retained(x[label],B[:,:rank]) for x in groups[d]]
                    per[str(d)][basis][label]=fmean(vals) if all(v is not None for v in vals) else None
        macro={}
        for basis in bases:
            macro[basis]={}
            for label in ("effect","sham"):
                vals=[x[basis][label] for x in per.values()]
                macro[basis][label]=fmean(vals) if all(v is not None for v in vals) else None
        rows.append({"rank":rank,"status":"measured_geometry","per_root":per,"root_macro_retained_energy":macro})
    return {"source":SOURCE,"verdict":"PILOT_ONLY","fit_roots":sorted(fit_roots),"eval_roots":sorted(eval_roots),
            "basis_fit_singular_values":spectra,"matched_rank_ceiling":feasible,"rows":rows,
            "scope":"geometric energy of unscaled clean donor prefix differences; held-out development roots only; projected edits have not been run through native recurrence; no claim of causal retention"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--fit",nargs="+",type=int,required=True)
    parser.add_argument("--eval",nargs="+",type=int,required=True)
    parser.add_argument("--ranks",nargs="+",type=int,default=[1,2,4,8])
    args=parser.parse_args()
    m,loaded=read_run(args.run)
    print(json.dumps(run(m,loaded,args.fit,args.eval,args.ranks),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
