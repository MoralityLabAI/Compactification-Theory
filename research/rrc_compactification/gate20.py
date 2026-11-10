"""RRC-20: tiny portable mixture of *frozen* certified class predictors.

No candidate interpreter is called to make a proposal.  The gate sees public
source coefficients and topology; the verifier is a separate downstream step.
"""
from __future__ import annotations
import math
from statistics import median

try:
    from . import graph11 as g, learn18 as l
except ImportError:
    import graph11 as g
    import learn18 as l

EXPERTS=('balanced_recurrent','skewed_recurrent','balanced_feedforward','certified_class_mass')
FEATURES=14


def source_features(case):
    """Fourteen public, bounded coordinates; never reads future/labels/family."""
    seq=[tuple(f[:13]) for f,_ in case.demos]
    a=[x for f in seq for x in f[1:]]
    nonzero=[x for x in a if x]
    nz=len(nonzero)
    first=sum(next((i for i,v in enumerate(f[1:],1) if v),13) for f in seq)/2/12
    head=sum(abs(v) for f in seq for v in f[1:7]);tail=sum(abs(v) for f in seq for v in f[7:13])
    data=[nz/24,
          sum(v<0 for v in nonzero)/max(1,nz),
          sum(math.log1p(abs(v)) for v in a)/24/5,
          math.log1p(max((abs(v) for v in a),default=0))/10,
          sum(abs(v)>3 for v in a)/24,
          sum(f[i]==0 for f in seq for i in (1,3,5,7,9,11))/12,
          sum(f[i]==0 for f in seq for i in (2,4,6,8,10,12))/12,
          min(5.,math.log1p(tail)/(1+math.log1p(head)))/5,
          sum(abs(v)==1 for v in a)/24,
          first]
    data.extend(l.topo_onehot(case.topology))
    assert len(data)==FEATURES and all(math.isfinite(x) for x in data)
    return tuple(data)


def expert_distributions(topology,logits,classes):
    """Normalized probabilities on certified class representatives.

    All four distributions use only cached pretrained outputs and public class
    multiplicities. Nonrepresentative syntax carries exactly zero mass.
    """
    info=classes[topology]
    reps=info['representatives'];count=len(reps)
    p=[]
    for role in EXPERTS[:3]:
        scores=[float(logits[role][g.GRAPHS.index(gr)]) for gr in reps]
        high=max(scores);exps=[math.exp(x-high) for x in scores]
        z=sum(exps)
        p.append(tuple(v/z for v in exps))
    class_mass={gr:0 for gr in reps}
    for gr,canonical in info['mapping'].items():class_mass[canonical]+=1
    p.append(tuple(class_mass[gr]/64 for gr in reps))
    if any(abs(sum(values)-1)>1e-10 for values in p):raise AssertionError('distribution invalid')
    return tuple(p)


def weights(case,checkpoint):
    if checkpoint['kind']=='constant':
        scores=checkpoint['bias']
    elif checkpoint['kind'] in ('adaptive','shuffled'):
        x=source_features(case)
        scores=[b+sum(a*v for a,v in zip(row,x)) for b,row in zip(checkpoint['bias'],checkpoint['matrix'])]
    else:raise ValueError('wrong gate kind')
    high=max(scores)
    exps=[math.exp(v-high) for v in scores];z=sum(exps)
    return tuple(v/z for v in exps)


def order(case,expert_logits,classes,checkpoint=None,*,fixed=None):
    """Rank each representative once; verification cost is charged elsewhere."""
    experts=expert_distributions(case.topology,expert_logits,classes)
    if fixed is not None:
        if len(fixed)!=4 or any(z<0 for z in fixed) or sum(fixed)<=0:raise ValueError('invalid fixed weights')
        w=tuple(v/sum(fixed) for v in fixed)
    elif checkpoint is not None:w=weights(case,checkpoint)
    else:raise ValueError('a trained gate or fixed mixture is required')
    reps=classes[case.topology]['representatives']
    scores=[sum(w[k]*experts[k][j] for k in range(4)) for j in range(len(reps))]
    return tuple(rep for _,rep in sorted(zip(range(len(reps)),reps),key=lambda v:(-scores[v[0]],g.GRAPHS.index(v[1]))))


def dense_mac_proxy(checkpoint):
    return 0 if checkpoint is None else (4*FEATURES if checkpoint['kind']!='constant' else 0)
