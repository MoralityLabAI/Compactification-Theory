"""OIC-21 one-sensor/step active monitoring with exact posterior and learned policy.

The observer receives only (n, action histories, binary labeled-sensor reports).
Its *belief state* enumerates all n^2 candidate (shift, initial phase) states
and is therefore an explicitly structure-assisted component, not a learned
transition model. The controller's sensor query policy is learned separately.
"""
from __future__ import annotations
from functools import lru_cache
from math import gcd, log2, ceil
from dataclasses import dataclass
from oic21_exact import ExactActiveOracle


def ent(counts):
    S = sum(counts)
    return -sum((c/S)*log2(c/S) for c in counts if c) if S else 0.0


def scores(mask, classes):
    counts=[(mask & c).bit_count() for c in classes]
    return max(counts) if counts else 0, ent(counts)


def feature_matrix(o:ExactActiveOracle, mask:int, t:int):
    """Shift-invariant normalized per-marker feature vectors, seven features."""
    if not mask or t>=o.horizon:raise ValueError('invalid belief/time')
    total=mask.bit_count()
    shift_max,shift_h=scores(mask,o.d_masks)
    gcd_max,gcd_h=scores(mask,o.g_masks)
    denom=log2(o.n)
    out=[]
    for m in range(o.n):
        hit,no=o.split(mask,t,m)
        h=hit.bit_count(); p=h/total
        y_m,y_h=scores(hit,o.d_masks) if hit else (0,0.0)
        n_m,n_h=scores(no,o.d_masks) if no else (0,0.0)
        y_g,y_gh=scores(hit,o.g_masks) if hit else (0,0.0)
        n_g,n_gh=scores(no,o.g_masks) if no else (0,0.0)
        acc_gain=(y_m+n_m-shift_max)/total
        gcd_gain=(y_g+n_g-gcd_max)/total
        entropy_gain=max(0.,(shift_h-p*y_h-(1-p)*n_h)/denom)
        gcd_entropy_gain=max(0.,(gcd_h-p*y_gh-(1-p)*n_gh)/denom)
        out.append((acc_gain,entropy_gain, p*(1-p), gcd_gain,
                    gcd_entropy_gain, p, 1.0/(1.0+abs(p-0.5)*2)))
    return tuple(out)


def greedy_policy(o,mask,t):
    features=feature_matrix(o,mask,t)
    return max(range(o.n),key=lambda m:(features[m][0],features[m][1],features[m][2],-m))


def entropy_policy(o,mask,t):
    features=feature_matrix(o,mask,t)
    return max(range(o.n),key=lambda m:(features[m][1],features[m][0],-m))


def random_policy(seed:int):
    import random
    rng=random.Random(seed)
    return lambda o,mask,t:rng.randrange(o.n)


def fixed_cycle_policy(o,mask,t):
    return t % o.n


def fixed_marker_policy(o,mask,t):
    return 0


def evaluate_policy(n:int,horizon:int,policy,target='shift'):
    """Exact expected accuracy, Brier, conditional entropy, and probability of 0-hit path.

    Entire n^2 hidden-state population evaluated once via deterministic
    posterior observation tree, avoiding Monte Carlo sampling.
    """
    o=ExactActiveOracle(n,horizon)
    node={(o.all,0)}
    queries=0
    actions={}
    for t in range(horizon):
        new=set()
        for mask,_ in node:
            m=policy(o,mask,t)
            assert isinstance(m,int) and 0<=m<n
            actions[(mask,t)]=m
            yes,no=o.split(mask,t,m)
            if yes:new.add((yes,t+1))
            if no:new.add((no,t+1))
            queries+=1
        node=new
    leaves=[m for m,_ in node]
    masks=o.d_masks if target=='shift' else o.g_masks
    N=n*n
    correct=sum(max((mask&dmask).bit_count() for dmask in masks) for mask in leaves)
    brier=0.0; H=0.0
    for mask in leaves:
        size=mask.bit_count()
        cnt=[(mask&dm).bit_count() for dm in masks]
        probabilities=[c/size for c in cnt]
        H+=size/N*ent(cnt)
        # multiclass Brier sum_k (p_k - 1[k==truth])^2
        brier+=size/N*(1.0-sum(p*p for p in probabilities))
    assert sum(x.bit_count() for x in leaves)==N
    return {'n':n,'horizon':horizon,'shift_accuracy':correct/N,'brier':brier,
            'conditional_entropy_bits':H,'leaves':len(leaves),'queried_nodes':queries,
            'output_bits':horizon,'adaptive_address_bits':horizon*(n-1).bit_length(),
            'total_link_bits':horizon*(1+(n-1).bit_length())}


class FrozenFixedPolicy:
    def __init__(self, word):self.word=tuple(word)
    def __call__(self,o,mask,t):return self.word[t]


def fixed_multisensor_accuracy(n:int,marks:tuple[int,...],steps:int):
    """Simultaneous bit per labeled marker, s*steps readout budget."""
    o=ExactActiveOracle(n,steps)
    leaf={o.all}
    for t in range(steps):
        nexts=set()
        for mask in leaf:
            remaining=mask
            for m in marks:
                hit=remaining&o.masks[t][m]
                if hit:
                    nexts.add(hit)
                    remaining^=hit
            if remaining:nexts.add(remaining)
        leaf=nexts
    assert sum(x.bit_count() for x in leaf)==n*n
    N=n*n
    return {'accuracy':sum(o.terminal(mask) for mask in leaf)/N,
            'output_bits':steps*len(marks), 'active_time_steps':steps,
            'sensors_installed':len(marks),'installed_sensor_address_bits':len(marks)*(n-1).bit_length()}


if __name__=='__main__':
 for n in (4,5,6,7,8,12):
  h=n+2
  for name,p in [('greedy',greedy_policy),('entropy',entropy_policy),('fixed0',fixed_marker_policy),('cycle',fixed_cycle_policy)]:
   r=evaluate_policy(n,h,p)
   print(n,h,name,r['shift_accuracy'],r['conditional_entropy_bits'],r['leaves'],flush=True)


def numpy_model_policy(weights):
    """Independent NumPy-only forward pass for the learned sensor scorer."""
    import numpy as np
    w1=np.array(weights['network.0.weight'],dtype=np.float32)
    b1=np.array(weights['network.0.bias'],dtype=np.float32)
    w2=np.array(weights['network.2.weight'],dtype=np.float32)
    b2=np.array(weights['network.2.bias'],dtype=np.float32)
    def choose(o,mask,t):
        feature=np.array(feature_matrix(o,mask,t),dtype=np.float32)
        scores=(np.tanh(feature@w1.T+b1)@w2.T+b2).ravel()
        return int(np.argmax(scores))
    return choose


def evaluate_torch_batched(model,n:int,horizon:int):
    """Exact population replay, one neural forward per clock step (rather than per leaf)."""
    import torch
    torch.set_num_threads(1)
    o=ExactActiveOracle(n,horizon)
    masks={o.all}
    queried_nodes=0
    with torch.no_grad():
        for t in range(horizon):
            current=sorted(masks)
            feats=torch.tensor([features for mask in current for features in feature_matrix(o,mask,t)],dtype=torch.float32)
            logits=model(feats).view(len(current),n)
            indices=logits.argmax(-1).tolist()
            next_masks=set()
            for mask,m in zip(current,indices):
                yes,no=o.split(mask,t,m)
                if yes:next_masks.add(yes)
                if no:next_masks.add(no)
            masks=next_masks
            queried_nodes+=len(current)
    N=n*n
    correct=sum(o.terminal(mask) for mask in masks)
    brier=H=0.0
    for mask in masks:
        size=mask.bit_count()
        cnt=[(mask&dm).bit_count() for dm in o.d_masks]
        probabilities=[c/size for c in cnt]
        H+=size/N*ent(cnt)
        brier+=size/N*(1-sum(p*p for p in probabilities))
    assert sum(x.bit_count() for x in masks)==N
    return {'n':n,'horizon':horizon,'shift_accuracy':correct/N,'brier':brier,
            'conditional_entropy_bits':H,'leaves':len(masks),'queried_nodes':queried_nodes,
            'output_bits':horizon,'adaptive_address_bits':horizon*(n-1).bit_length(),
            'total_link_bits':horizon*(1+(n-1).bit_length())}
