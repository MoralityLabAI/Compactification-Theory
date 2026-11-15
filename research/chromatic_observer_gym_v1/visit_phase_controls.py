"""Fixed-parameter, diagnostic-only visit-order counterfactuals."""
from __future__ import annotations
import random
from temporal_observers import EDGES, N, T, dot,covered_pairs,TEST_CONTEXTS


def visit_scramble(item,head,seed_offset):
    w,threshold=head
    observations=[]
    for c in TEST_CONTEXTS:
        row=[]
        for a in range(N):
            bits=[int(dot(w,item['activations'][f'{a}:{c}'][t])>threshold) for t in range(T)]
            rng=random.Random(item['model_seed']*100000+c*100+a*10+seed_offset)
            rng.shuffle(bits)
            row.append(tuple(bits))
        observations.append(row)
    return covered_pairs(observations)


def layer_separation_diagnostic(item):
    from math import dist
    # Same distractor context for each compared pair: frozen task label varies.
    values=[]
    for t in range(T):
        pairs=[dist(item['activations'][f'{a}:{c}'][t],
                    item['activations'][f'{b}:{c}'][t])
            for c in TEST_CONTEXTS for a,b in EDGES]
        values.append({'visit':t+1,'minimum_l2':min(pairs),'mean_l2':sum(pairs)/len(pairs),'maximum_l2':max(pairs)})
    return values


def count_only_score(item,head):
    """Order-free count of threshold-positive visits, diagnostic control."""
    w,threshold=head
    obs=[]
    for c in TEST_CONTEXTS:
        row=[]
        for a in range(N):
            row.append(sum(int(dot(w,item['activations'][f'{a}:{c}'][t])>threshold) for t in range(T)))
        obs.append(row)
    return {'edge_hits':covered_pairs(obs),'unique_counts':[len(set(row)) for row in obs]}


def unordered_qary_upper_bound(visits,heads):
    """Number of T-element multisets over alphabet of size 2**heads."""
    from math import comb
    if visits<0 or heads<0:raise ValueError
    q=1<<heads
    return comb(visits+q-1,q-1)


def clique_max_separated_with_categories(n,classes):
    """Exact largest number of K_n edges separable by at most classes bins."""
    from math import comb
    if n<0 or classes<1:raise ValueError
    q,r=divmod(n,classes)
    collisions=r*comb(q+1,2)+(classes-r)*comb(q,2)
    return comb(n,2)-collisions
