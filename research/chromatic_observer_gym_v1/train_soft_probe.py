"""COE-03 synthetic CPU training pilot: one-hot states, continuous 3-head probe.

No external packages. Linear one-hot probe weights are trained by projected
subgradient descent using *only* pairwise output-distance hinge loss.
By default all K5 pairs are used. An optional exploratory edge holdout can
measure transfer to missing evaluator distinctions; synthetic only.
"""
from __future__ import annotations
from itertools import combinations
from random import Random

try:
    from .soft_precision_codes import quantize_threshold, binary_diameter_two
except ImportError:
    from soft_precision_codes import quantize_threshold, binary_diameter_two


def fit_soft_probe(seed: int, n: int = 5, k: int = 3,
                   steps: int = 4000, learning_rate: float = 0.45,
                   heldout_edges: tuple[tuple[int, int], ...] = ()) -> dict:
    if n < 2 or k < 1 or steps < 1 or learning_rate <= 0:
        raise ValueError('invalid pilot hyperparameters')
    rng=Random(seed)
    state=[[rng.random() for _ in range(k)] for _ in range(n)]
    pairs=tuple(combinations(range(n),2))
    heldout = set(heldout_edges)
    if any(e not in pairs for e in heldout):
        raise ValueError('holdout must be an unordered canonical pair')
    train_pairs = tuple(e for e in pairs if e not in heldout)
    if not train_pairs:
        raise ValueError('training edge set cannot be empty')
    m=len(train_pairs)
    history=[]
    for step in range(steps):
        grad=[[0.0]*k for _ in range(n)]
        loss=0.0
        for u,v in train_pairs:
            d=sum(abs(a-b) for a,b in zip(state[u],state[v]))
            deviation=(d-1.0) if d<1.0 else ((d-2.0) if d>2.0 else 0.0)
            loss+=deviation*deviation/m
            if deviation:
                for j in range(k):
                    sign=(1 if state[u][j]>state[v][j] else -1) if state[u][j]!=state[v][j] else 0
                    grad[u][j]+=2*deviation*sign/m
                    grad[v][j]-=2*deviation*sign/m
        for i in range(n):
            for j in range(k):
                state[i][j] = min(1.,max(0.,state[i][j]-learning_rate*grad[i][j]))
        if step in (0,99,499,999,1999,steps-1):
            history.append((step+1,round(loss,10)))
        if loss < 1e-12:
            break
    ds=[sum(abs(a-b) for a,b in zip(state[u],state[v])) for u,v in pairs]
    train_distances=[d for d,pair in zip(ds,pairs) if pair not in heldout]
    heldout_distances=[d for d,pair in zip(ds,pairs) if pair in heldout]
    lower=min(ds);upper=max(ds)
    hard=[sum(int(x>=0.5)<<j for j,x in enumerate(v)) for v in state]
    hard_d=[(hard[u]^hard[v]).bit_count() for u,v in pairs]
    return {'seed':seed,'steps_completed':step+1,'loss':round(loss,10),
            'soft_min':round(lower,7),'soft_max':round(upper,7),
            'soft_feasible_1e3':lower>=1-1e-3 and upper<=2+1e-3,
            'train_feasible_1e3':min(train_distances)>=1-1e-3 and max(train_distances)<=2+1e-3,
            'heldout_feasible_1e3':(min(heldout_distances)>=1-1e-3 and max(heldout_distances)<=2+1e-3) if heldout_distances else None,
            'train_edges':len(train_pairs),'heldout_edges':len(heldout),
            'hard_min':min(hard_d),'hard_max':max(hard_d),
            'hard_feasible':binary_diameter_two(tuple(hard)),
            'hard_codes':hard,
            'state':[[round(x,8) for x in v] for v in state],
            'history':history}


def run_pilot(seeds=range(10), heldout_edges: tuple[tuple[int, int], ...] = ()) -> dict:
    outputs=[fit_soft_probe(seed, heldout_edges=heldout_edges) for seed in seeds]
    return {'format':'coe03-training-pilot-v1', 'task':'synthetic K5 pair-distance hinge fit',
            'model':'3-channel linear probe on five frozen one-hot states',
            'hyperparameters':{'steps':4000,'learning_rate':0.45,'all_ten_pairs_in_train':True},
            'seeds':[o['seed'] for o in outputs],
            'heldout_edges':[list(e) for e in heldout_edges],
            'soft_feasible_count':sum(o['soft_feasible_1e3'] for o in outputs),
            'threshold_half_hard_feasible_count':sum(o['hard_feasible'] for o in outputs),
            'train_feasible_count':sum(o['train_feasible_1e3'] for o in outputs),
            'heldout_feasible_count':sum(bool(o['heldout_feasible_1e3']) for o in outputs) if heldout_edges else None,
            'runs':outputs}


if __name__=='__main__':
    import json
    print(json.dumps(run_pilot(),indent=2))
